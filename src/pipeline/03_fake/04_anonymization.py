"""
04_anonymization.py — Pseudo-fake: NHIỄU/BLUR KHUÔN MẶT (Anonymization)

Tuần 2 (Pseudo-fake Engineering) — kỹ thuật thứ 4 theo docs/Pipeline.

Ý tưởng: che/méo vùng mặt — Gaussian blur mạnh hoặc pixelate. AUDIO GIỮ NGUYÊN
(-c:a copy). Mô phỏng fake đời thực hay che/méo mặt: "CapCut noise" và nén H.264
mạnh làm mặt mờ-nhiễu để né detection (xem CLAUDE.md, mục Tạo fake). Khẩu hình bị
xóa/mờ -> nhánh khớp môi–tiếng mất tín hiệu -> kênh tấn công VISUAL-IDENTITY, khác
02 (visual-motion) và 03 (audio-prosody).

CÁCH LÀM (tối ưu tốc độ): detect mặt trên VÀI FRAME MẪU (YOLO, GPU) -> lấy VÙNG
UNION bao trọn mặt -> làm méo vùng đó cho CẢ clip bằng MỘT lệnh ffmpeg (native C,
1 pass). Nhanh ~1-2s/clip. (Bản cũ dùng cv2 blur từng frame -> ~15s/clip do
VideoWriter encode từng frame — đã bỏ.) Vùng union tĩnh hợp với clip talking-head
đã curate (mặt gần như cố định, face_ratio ~1.0).

Detector: yolov8n-face.pt sẵn có ở root (ultralytics).

⚠️ CẢNH BÁO LEAKAGE: blur CHỈ ở fake -> model học tắt "mờ = fake". BẮT BUỘC bước
train áp blur ĐỐI XỨNG lên một phần REAL (augmentation), hoặc trộn cùng SNVSM.
Generator V1 còn dùng `-shortest`; smoke đã thấy video ngắn hơn real ghép cặp
40 ms. V2 bỏ `-shortest`, ghi structured timeline và chỉ publish nếu
frame/FPS/duration video cùng audio target vẫn khớp source.

Phụ thuộc: ultralytics, opencv-python (đọc frame mẫu + detect). Cần ffmpeg trong PATH.

Ví dụ:
  python src/pipeline/03_fake/04_anonymization.py --device cuda
  python src/pipeline/03_fake/04_anonymization.py --mode pixelate --limit 5
"""

import os
import sys
import csv
import argparse
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pipeline.fake_media_contract import (
    is_valid_repaired_output,
    probe_media,
    publish_validated,
    remove_if_exists,
    same_file_path,
)
from src.pipeline.timeline_contract import (
    TIMELINE_FIELDS,
    build_timeline_contract,
    validate_timeline_contract,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    # pyrefly: ignore [missing-import]
    import cv2
    # pyrefly: ignore [missing-import]
    from ultralytics import YOLO
except Exception as e:
    print(f"Thiếu thư viện: {e}. Cài: pip install ultralytics opencv-python")
    sys.exit(1)

GENERATOR_VERSION = "anonymization_v2_exact_timeline_v1"
DEFAULT_OUT_DIR = "data/03_fake/anonymization_v2"
DEFAULT_LABELS = "data/03_fake/manifests/v2/anonymization.csv"
LABEL_FIELDS = ["clip_id", "file_path", "label", "method", "param",
                "source_clip", "source_video", "speaker_id", "tier",
                *TIMELINE_FIELDS]


def sample_face_box(model, in_path, n_samples, margin, conf, device):
    """
    Đọc n_samples frame rải đều, YOLO detect mặt to nhất mỗi frame, trả UNION box
    (x, y, w, h) đã nới margin & clamp trong khung; None nếu không bắt được mặt nào.
    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total > n_samples:
        idxs = [int(total * k / (n_samples + 1)) for k in range(1, n_samples + 1)]
    else:
        idxs = list(range(max(1, total)))

    boxes = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        res = model.predict(frame, verbose=False, conf=conf, device=device)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            best, best_area = None, -1
            for b in res.boxes.xyxy.cpu().numpy():
                area = (b[2] - b[0]) * (b[3] - b[1])
                if area > best_area:
                    best, best_area = b[:4], area
            boxes.append(best)
    cap.release()
    if not boxes:
        return None

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - margin * bw)
    y1 = max(0, y1 - margin * bh)
    x2 = min(fw, x2 + margin * bw)
    y2 = min(fh, y2 + margin * bh)
    w, h = int(x2 - x1), int(y2 - y1)
    if w < 8 or h < 8:
        return None
    return int(x1), int(y1), w, h


def assert_manifest_compatible(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {}
    for row in rows:
        clip_id = row.get("clip_id", "")
        if (not clip_id or clip_id in by_id
                or row.get("method") != "anonymization"
                or f"generator={GENERATOR_VERSION}" not in row.get("param", "")):
            raise ValueError(f"Manifest anonymization V2 không tương thích: {path}")
        validate_timeline_contract(row, "anonymization")
        by_id[clip_id] = row
    return by_id


def ffmpeg_anon(in_path, out_path, box, mode, sigma, source_media=None):
    """Làm méo vùng box (x,y,w,h) cho cả clip trong 1 pass ffmpeg; audio giữ nguyên."""
    source_media = source_media or probe_media(in_path)
    if source_media is None:
        return False
    x, y, w, h = box
    if mode == "pixelate":
        sw, sh = max(2, w // 16), max(2, h // 16)
        region = f"crop={w}:{h}:{x}:{y},scale={sw}:{sh},scale={w}:{h}:flags=neighbor"
    else:  # blur
        s = sigma if sigma > 0 else max(12, min(min(w, h) // 6, 40))
        region = f"crop={w}:{h}:{x}:{y},gblur=sigma={s}"
    vf = f"[0:v]{region}[b];[0:v][b]overlay={x}:{y}[v]"

    partial_path = out_path + ".part.mp4"

    def run(filter_graph, audio_map, acodec):
        remove_if_exists(partial_path)
        cmd = ["ffmpeg", "-y", "-i", in_path, "-filter_complex", filter_graph,
               "-map", "[v]", "-map", audio_map,
               "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p",
               "-c:a", acodec, "-fps_mode", "passthrough",
               "-loglevel", "error", partial_path]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            remove_if_exists(partial_path)
            return False
        return publish_validated(partial_path, out_path, source_media)
    if run(vf, "0:a:0", "copy"):
        return True
    lossless_graph = (
        vf + f";[0:a:0]atrim=end_sample={source_media['audio_native_samples']},"
        "asetpts=PTS-STARTPTS[aout]"
    )
    return run(lossless_graph, "[aout]", "alac")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="data/02_curate/all_clean.csv",
                    help="CSV clip real (cần cột file_path) — mặc định tập sạch từ 04_curate")
    ap.add_argument("--path_col", default="file_path")
    ap.add_argument("--id_col", default="clip_id")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--face_model", default="yolov8n-face.pt",
                    help="đường dẫn YOLO face (mặc định ở root khi chạy từ repo root)")
    ap.add_argument("--mode", choices=["blur", "pixelate"], default="blur")
    ap.add_argument("--sigma", type=int, default=0, help="cường độ Gaussian; 0 = tự theo cỡ mặt")
    ap.add_argument("--margin", type=float, default=0.20, help="nới vùng union theo tỉ lệ")
    ap.add_argument("--n_samples", type=int, default=6, help="số frame mẫu để detect union box")
    ap.add_argument("--conf", type=float, default=0.25, help="ngưỡng confidence YOLO")
    ap.add_argument("--device", default=None, help="'cuda'/'cpu' (mặc định: tự chọn theo ultralytics)")
    ap.add_argument("--skip_existing", action="store_true", default=True,
                    help="bỏ qua clip đã có .mp4 -> chạy lại là RESUME, không ghi trùng nhãn")
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="chỉ xử lý N clip đầu (để test)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    existing_rows = assert_manifest_compatible(args.labels)
    if not os.path.isfile(args.face_model):
        print(f"Không tìm thấy face model: {args.face_model} (chạy từ repo root hoặc dùng --face_model)")
        sys.exit(1)
    model = YOLO(args.face_model)

    with open(args.input_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]
    print(f"Đọc {len(rows)} clip real từ {args.input_csv} | mode={args.mode} | device={args.device or 'auto'}")

    new_file = not os.path.exists(args.labels) or os.path.getsize(args.labels) == 0
    os.makedirs(os.path.dirname(os.path.abspath(args.labels)) or ".", exist_ok=True)
    lf = open(args.labels, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(lf, fieldnames=LABEL_FIELDS)
    if new_file:
        writer.writeheader()

    made = resumed = repaired = skipped = failed = 0
    for i, r in enumerate(rows):
        src_path = r.get(args.path_col, "")
        src_id = r.get(args.id_col, f"clip{i:06d}")
        if not src_path or not os.path.isfile(src_path):
            skipped += 1
            continue

        media = probe_media(src_path)
        if media is None:
            skipped += 1
            continue

        box = sample_face_box(model, src_path, args.n_samples, args.margin, args.conf, args.device)
        if box is None:
            skipped += 1               # không bắt được mặt -> bỏ
            continue

        fake_id = f"{src_id}_anonv2{args.mode[:3]}"
        out_path = os.path.abspath(os.path.join(args.out_dir, fake_id + ".mp4"))
        expected_param = (
            f"{args.mode}_box={box[0]},{box[1]},{box[2]},{box[3]};"
            f"sigma={args.sigma};generator={GENERATOR_VERSION}"
        )
        existing = existing_rows.get(fake_id)
        if existing:
            recorded_path = os.path.abspath(existing.get("file_path", ""))
            if (not same_file_path(recorded_path, out_path)
                    or existing.get("param", "") != expected_param):
                raise ValueError(f"Resume contract sai cho {fake_id}")
            if (args.skip_existing
                    and is_valid_repaired_output(out_path, media)):
                resumed += 1
                continue

        if ffmpeg_anon(src_path, out_path, box, args.mode, args.sigma, media):
            if existing:
                repaired += 1
                continue
            output_row = {
                "clip_id": fake_id,
                "file_path": out_path,
                "label": 1,
                "method": "anonymization",
                "param": expected_param,
                "source_clip": src_id,
                "source_video": r.get("source_video", ""),
                "speaker_id": r.get("speaker_id", ""),
                "tier": r.get("tier", ""),
            }
            output_row.update(build_timeline_contract(
                media["audio_duration"], media["video_duration"],
                manipulation_scope="global",
            ))
            writer.writerow(output_row)
            made += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}  made={made} resume={resumed} "
                  f"repair={repaired} skip={skipped} fail={failed}")
            lf.flush()

    lf.close()
    print(f"\nXong. Fake tạo được: {made} | resume: {resumed} | "
          f"sửa file: {repaired} | bỏ qua: {skipped} | lỗi: {failed}")
    print(f"  Video -> {args.out_dir}/  | nhãn (append) -> {args.labels}")
    print("⚠️  Nhớ áp blur ĐỐI XỨNG lên một phần REAL ở bước train để tránh leakage 'mờ=fake'.")
    if skipped or failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
