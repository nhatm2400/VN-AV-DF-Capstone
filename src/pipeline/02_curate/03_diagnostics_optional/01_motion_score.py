"""
01_motion_score.py — chẩn đoán ĐỘ ĐỘNG của khung hình — tùy chọn trước 04

Vì sao cần: gate hiện tại (det_ratio / speech_ratio / embed_consistency) KHÔNG thấy
được clip là ảnh tĩnh. Đo trên mẫu 100 clip của all_clean.csv:
  - speech_ratio = 1.000 cho MỌI clip (bão hòa, 0 thông tin)
  - det_ratio ~ 0.997 cho cả clip tĩnh lẫn động (ảnh chân dung thì khung nào cũng có mặt)
  - embed_consistency CAO HƠN ở clip tĩnh (0.881 vs 0.792) -> đang THƯỞNG cho khung đứng yên
  - 25/100 clip gần như tĩnh, 10/100 là ảnh tĩnh tuyệt đối
Clip tĩnh làm hỏng nhãn: frame_reverse(ảnh tĩnh) == ảnh tĩnh -> cặp real/fake là nhiễu.

Đo gì (khung xám, thu nhỏ, lấy mẫu ~10 fps):
  motion_median     median |frame-diff| giữa 2 khung liên tiếp (thang 0-255). Thấp = tĩnh.
  motion_p90        đuôi cao — tách "tĩnh rồi cắt cảnh" khỏi "tĩnh đều"
  frac_near_static  tỉ lệ cặp khung gần trùng (|diff| < 1.0)

Trình tự bộ script:
  02_scoring/01_face_quality.py                    (đo mặt + embedding)
  03_diagnostics_optional/01_motion_score.py       (đo độ động) <-- file này
  03_diagnostics_optional/02_sync_score.py         (đo sync, tùy chọn)
  04_curate.py        (cluster -> gate -> cân bằng -> xuất tập sạch)

Giữ NGUYÊN số dòng và thứ tự dòng của input (04_curate assert len(csv) == len(npy)).

⚠️ CHỈ áp gate motion cho tập REAL NGUỒN, trước khi sinh fake — khi đó cả 4 fake
đều thừa hưởng từ real đã lọc nên vẫn đối xứng. TUYỆT ĐỐI không lọc motion trên tập
fake đã sinh: anonymization làm mờ -> frame-diff giảm -> sẽ loại lệch riêng kênh đó.

CÁCH DÙNG (từ thư mục gốc dự án):
  D:/Anaconda/envs/vn_av_df/python.exe src/pipeline/02_curate/03_diagnostics_optional/01_motion_score.py \
      --input_csv data/02_curate/measurements/tier1_scored_all.csv \
      --out data/02_curate/measurements/tier1_scored_motion.csv
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:                                  # in được tiếng Việt khi pipe/redirect (console cp1252)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import numpy as np
import pandas as pd


NEAR_STATIC_DIFF = 1.0   # |frame-diff| dưới mức này coi như hai khung trùng nhau


def clip_motion(path, sample_fps=10.0, width=160):
    """Trả về dict 3 chỉ số, hoặc None nếu không đọc được / quá ngắn."""
    if not isinstance(path, str) or not os.path.isfile(path):
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / sample_fps)))

    prev, diffs, fi = None, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % step == 0:
            h, w = frame.shape[:2]
            small = cv2.cvtColor(
                cv2.resize(frame, (width, max(1, int(h * width / w)))),
                cv2.COLOR_BGR2GRAY,
            ).astype(np.float32)
            if prev is not None:
                diffs.append(float(np.abs(small - prev).mean()))
            prev = small
        fi += 1
    cap.release()

    if len(diffs) < 3:
        return None
    d = np.array(diffs)
    return {
        "motion_median": float(np.median(d)),
        "motion_p90": float(np.percentile(d, 90)),
        "frac_near_static": float((d < NEAR_STATIC_DIFF).mean()),
    }


def _worker(task):
    clip_id, path, sample_fps, width = task
    return clip_id, clip_motion(path, sample_fps, width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="data/02_curate/measurements/tier1_scored_all.csv",
                    help="output của 02_score_clips")
    ap.add_argument("--out", default="data/02_curate/measurements/tier1_scored_motion.csv")
    ap.add_argument("--col", default="file_path")
    ap.add_argument("--sample_fps", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=160, help="bề ngang khi thu nhỏ")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {"clip_id", args.col}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if df.empty:
        raise ValueError("Input CSV is empty")
    if df["clip_id"].isna().any() or df["clip_id"].duplicated().any():
        raise ValueError("clip_id must be non-empty and unique")
    missing_paths = [
        str(path) for path in df[args.col]
        if not isinstance(path, str) or not os.path.isfile(path)
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)}/{len(df)} media paths do not exist. "
            f"Examples: {', '.join(missing_paths[:5])}"
        )
    if os.path.exists(args.out) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.out}. "
            "Pass --overwrite only for an intentional rerun."
        )
    print(f"Đọc {len(df)} clip từ {args.input_csv}")

    tasks = [(r.clip_id, getattr(r, args.col), args.sample_fps, args.width)
             for r in df.itertuples(index=False)]

    results, done, fail = {}, 0, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for fut in as_completed(futs):
            clip_id, m = fut.result()
            results[clip_id] = m
            done += 1
            if m is None:
                fail += 1
            if done % 500 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(tasks)}  fail={fail}  "
                      f"({el/60:.1f}m, {done/el:.1f} clip/s)", flush=True)

    for col in ("motion_median", "motion_p90", "frac_near_static"):
        df[col] = df["clip_id"].map(lambda c: (results.get(c) or {}).get(col))

    el = time.time() - t0
    print(f"\nXong {len(df)} clip trong {el/60:.1f} phút | không đọc được: {fail}")
    if fail:
        failed_ids = [clip_id for clip_id, result in results.items() if result is None]
        raise RuntimeError(
            f"Refusing to publish: motion scoring failed for {fail}/{len(df)} clips. "
            f"Example clip_id values: {', '.join(failed_ids[:5])}"
        )

    m = df["motion_median"].dropna()
    if len(m):
        print("\n-- motion_median (thang 0-255, thấp = tĩnh) --")
        print(m.describe(percentiles=[.05, .1, .25, .5, .75, .9]).round(3).to_string())
        print("\n-- số clip dưới các ngưỡng --")
        for t in (0.5, 1.0, 2.0, 3.0):
            print(f"  motion_median < {t:<4}: {int((m < t).sum()):5d}/{len(m)} "
                  f"({100*(m < t).mean():.1f}%)")
        st = df["frac_near_static"].dropna()
        print(f"\n  clip có >50% cặp khung gần trùng: {int((st > 0.5).sum())}/{len(st)}")

    out_parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_parent, exist_ok=True)
    partial = args.out + ".partial"
    try:
        df.to_csv(partial, index=False)
        os.replace(partial, args.out)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    print(f"\nCSV mới (có motion_median / motion_p90 / frac_near_static) -> {args.out}")
    print("Tiếp theo: xem phân bố rồi mới chốt ngưỡng; dùng file này làm --scored_csv "
          "cho 04_curate.py (tùy chọn --motion_floor).")


if __name__ == "__main__":
    main()
