"""
01_face_quality.py — đo chất lượng mặt + embedding trong stage 02_scoring

Đọc tier*_v3_clips_*.csv, với mỗi clip lấy K frame rải đều, chạy InsightFace để:
  - đếm số frame phát hiện được mặt        -> det_ratio
  - đo kích thước mặt (lớn nhất mỗi frame) -> mean_face_area
  - tính face embedding đại diện (định danh)-> embeddings.npy
  - đo độ nhất quán danh tính giữa các frame-> embed_consistency

KHÔNG loại bỏ clip nào ở bước này. Chỉ đo và ghi metadata.
Quyết định (cluster / gate / cân bằng) để 04_curate.py xử lý.

Chạy trên Kaggle T4 (ctx_id=0). Clip ngắn nên decode rẻ -> không cần NVDEC ở đây.

LOCAL (mặc định — chạy KHÔNG cần tham số, từ thư mục gốc dự án):
  D:/Anaconda/envs/vn_av_df/python.exe src/pipeline/02_curate/02_scoring/01_face_quality.py
  -> đọc data/01_collect/cut_clips/all_manifest.csv, xuất
     data/02_curate/measurements/tier1_scored_all.csv + embeddings_all.npy
  GPU Windows: file đã gọi onnxruntime.preload_dlls() để onnxruntime-gpu thấy CUDA;
     cần env có onnxruntime-gpu + nvidia-cudnn-cu12==9.8.0.87 (xem memory env). Không
     có GPU -> tự lùi về CPU (chậm ~6x). Đo: GPU ~32ms/frame, CPU ~195ms/frame.
KAGGLE: python src/pipeline/02_curate/02_scoring/01_face_quality.py --input_csv /kaggle/working/all_manifest.csv --out_dir /kaggle/working

Trình tự bộ script:
  02_scoring/01_face_quality.py                    <-- file này
  02_scoring/02_active_speaker/                    (temporal active speaker)
  03_diagnostics_optional/                         (motion/sync, tùy chọn)
  04_curate.py       (cluster speaker -> gate -> cân bằng -> xuất tập sạch)

Output:
  - <OUT_DIR>/tier1_scored_<TAG>.csv   (CSV gốc + cột face stats, cùng thứ tự dòng với npy)
  - <OUT_DIR>/embeddings_<TAG>.npy     (N, 512) float32, đã L2-normalize, theo thứ tự dòng CSV
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import cv2

try:                                  # in được tiếng Việt khi pipe/redirect (console cp1252)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Nạp DLL CUDA/cuDNN từ wheel pip nvidia-* (cần để onnxruntime-gpu thấy GPU trên Windows).
# Vô hại nếu chạy CPU hoặc onnxruntime đã tự nạp.
try:
    # pyrefly: ignore [missing-import]
    import onnxruntime as _ort
    if hasattr(_ort, "preload_dlls"):
        _ort.preload_dlls()
except Exception:
    pass

# pyrefly: ignore [missing-import]
from insightface.app import FaceAnalysis

# ----------------------------- Config -----------------------------
K_FRAMES = 9            # số frame lấy mẫu mỗi clip (rải đều). Nâng từ 5 -> 9 để
                        # det_ratio mịn hơn và embedding ổn định hơn (pass chạy 1 lần).
DET_SIZE = (640, 640)   # input size cho detector
MIN_DET_SCORE = 0.5     # ngưỡng tin cậy detection để tính là "có mặt"
EMBED_DIM = 512         # buffalo_l -> 512
CONSISTENCY_AVG_MIN = 0.5  # cosine tương đồng trung bình giữa các frame >= ngưỡng này
                           # -> coi như cùng 1 người, mới trung bình embedding.


def sample_frames(path, k):
    """Lấy k frame rải đều từ một clip. Trả list ndarray (BGR). Bỏ qua frame đọc lỗi."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    # vị trí rải đều, tránh frame đầu/cuối sát biên
    idxs = np.linspace(total * 0.1, total * 0.9, k).astype(int)
    idxs = np.clip(idxs, 0, total - 1)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok and fr is not None:
            frames.append(fr)
    cap.release()
    return frames


def largest_face(faces):
    """Chọn mặt có bbox lớn nhất (heuristic: trong khung tin tức, mặt to nhất thường là speaker chính)."""
    best, best_area = None, -1.0
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        if area > best_area:
            best, best_area = f, area
    return best, best_area


def aggregate_embeddings(embs):
    """
    Gộp embedding các frame thành 1 vector đại diện + đo độ nhất quán danh tính.

    Vì largest_face chọn mặt to nhất ĐỘC LẬP từng frame, clip 2 người / cắt cảnh
    có thể lấy nhầm 2 danh tính khác nhau. Trung bình thẳng sẽ ra vector "lai" vô
    nghĩa -> cluster sai speaker. Khắc phục:
      - các frame cùng người (consistency cao): trung bình (giảm nhiễu).
      - lẫn nhiều người (consistency thấp): lấy medoid (1 danh tính thật).
    Trả về (emb_512_l2norm, consistency).
      consistency = cosine tương đồng trung bình giữa các cặp frame (1.0 = đồng nhất).
    """
    X = np.stack(embs, 0)  # (k, 512), mỗi hàng đã L2-norm sẵn trong insightface
    n = len(X)
    if n == 1:
        return X[0].astype(np.float32), 1.0

    sim = X @ X.T  # cosine vì đã norm
    consistency = float((sim.sum() - n) / (n * (n - 1)))  # trung bình ngoài đường chéo

    if consistency >= CONSISTENCY_AVG_MIN:
        emb = X.mean(0)
    else:
        # medoid: frame tương đồng nhất với phần còn lại -> đại diện cho danh tính trội
        emb = X[int(sim.sum(1).argmax())]

    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb.astype(np.float32), consistency


def score_clip(app, path, k_frames):
    """
    Trả về dict:
      det_ratio        : tỉ lệ frame có mặt (0..1)
      mean_face_area   : diện tích mặt lớn nhất / diện tích frame, trung bình các frame có mặt
      embedding        : (512,) đã L2-normalize, hoặc None nếu không có mặt nào
      embed_consistency: độ đồng nhất danh tính giữa các frame (0..1)
    """
    frames = sample_frames(path, k_frames)
    if not frames:
        return dict(det_ratio=0.0, mean_face_area=0.0, embedding=None,
                    n_sampled=0, embed_consistency=0.0)

    embs, areas, hit = [], [], 0
    for fr in frames:
        h, w = fr.shape[:2]
        faces = app.get(fr)
        faces = [f for f in faces if getattr(f, "det_score", 1.0) >= MIN_DET_SCORE]
        if not faces:
            continue
        face, area = largest_face(faces)
        if face is None:
            continue
        hit += 1
        areas.append(area / float(w * h))
        emb = face.normed_embedding  # đã L2-norm sẵn trong insightface
        if emb is not None:
            embs.append(emb)

    n = len(frames)
    if hit == 0 or not embs:
        return dict(det_ratio=hit / n, mean_face_area=0.0, embedding=None,
                    n_sampled=n, embed_consistency=0.0)

    emb, consistency = aggregate_embeddings(embs)
    return dict(
        det_ratio=hit / n,
        mean_face_area=float(np.mean(areas)),
        embedding=emb,
        n_sampled=n,
        embed_consistency=consistency,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="data/01_collect/cut_clips/all_manifest.csv",
                    help="manifest từ 01_prep_manifest (mặc định local)")
    ap.add_argument("--out_dir", default="data/02_curate/measurements")
    ap.add_argument("--tag", default="all", help="hậu tố tên file output")
    ap.add_argument("--path_col", default="file_path", help="cột chứa đường dẫn .mp4")
    ap.add_argument("--k_frames", type=int, default=K_FRAMES, help="số frame lấy mẫu mỗi clip")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {"clip_id", args.path_col}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if df.empty:
        raise ValueError("Input manifest is empty")
    if df["clip_id"].isna().any() or df["clip_id"].duplicated().any():
        raise ValueError("clip_id must be non-empty and unique")

    os.makedirs(args.out_dir, exist_ok=True)
    csv_out = os.path.join(args.out_dir, f"tier1_scored_{args.tag}.csv")
    npy_out = os.path.join(args.out_dir, f"embeddings_{args.tag}.npy")
    existing = [p for p in (csv_out, npy_out) if os.path.exists(p)]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output already exists; pass --overwrite only for an intentional rerun: "
            + ", ".join(existing)
        )

    missing_paths = [
        str(path) for path in df[args.path_col]
        if not isinstance(path, str) or not os.path.isfile(path)
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)}/{len(df)} media paths do not exist. "
            f"Examples: {', '.join(missing_paths[:5])}"
        )
    print(f"Đọc {len(df)} clip từ {args.input_csv}  (k_frames={args.k_frames})")

    # InsightFace: detection + recognition, chạy GPU (ctx_id=0)
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    det_ratios, face_areas, has_emb, n_sampled, consist = [], [], [], [], []
    embeddings = np.zeros((len(df), EMBED_DIM), dtype=np.float32)

    failures = []
    t0 = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        path = getattr(row, args.path_col)
        try:
            r = score_clip(app, path, args.k_frames)
        except Exception as e:
            failures.append((str(path), str(e)))
            r = dict(det_ratio=0.0, mean_face_area=0.0, embedding=None,
                     n_sampled=0, embed_consistency=0.0)

        det_ratios.append(r["det_ratio"])
        face_areas.append(r["mean_face_area"])
        n_sampled.append(r["n_sampled"])
        consist.append(r["embed_consistency"])
        if r["embedding"] is not None:
            embeddings[i] = r["embedding"]
            has_emb.append(True)
        else:
            has_emb.append(False)

        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(df)}  ({el:.0f}s, {(i+1)/el:.1f} clip/s)")

    df["det_ratio"] = det_ratios
    df["mean_face_area"] = face_areas
    df["has_embedding"] = has_emb
    df["n_sampled"] = n_sampled
    df["embed_consistency"] = consist

    if failures:
        preview = "; ".join(f"{path}: {error}" for path, error in failures[:5])
        raise RuntimeError(
            f"Refusing to publish: scoring failed for {len(failures)}/{len(df)} clips. "
            f"Examples: {preview}"
        )

    csv_partial = csv_out + ".partial"
    npy_partial = npy_out + ".partial"
    try:
        df.to_csv(csv_partial, index=False)
        with open(npy_partial, "wb") as f:
            np.save(f, embeddings)
        os.replace(csv_partial, csv_out)
        os.replace(npy_partial, npy_out)
    finally:
        for partial in (csv_partial, npy_partial):
            if os.path.exists(partial):
                os.remove(partial)

    no_face = len(has_emb) - sum(has_emb)
    print(f"\nXong {len(df)} clip trong {(time.time()-t0)/60:.1f} phút")
    print(f"  có mặt: {sum(has_emb)} | không mặt: {no_face}")
    print(f"  CSV : {csv_out}")
    print(f"  NPY : {npy_out}  shape={embeddings.shape}")

    if no_face / len(df) > 0.5:
        print(f"\n[CẢNH BÁO] {no_face}/{len(df)} (>50%) clip 'không mặt'. "
              "Kiểm tra lại đường dẫn / video lỗi trước khi tin kết quả.")

    print("\nTiếp theo: active-speaker scoring, diagnostics tùy chọn, rồi 04_curate.py.")


if __name__ == "__main__":
    main()
