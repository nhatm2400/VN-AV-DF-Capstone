"""
build_roi_preview.py — Dựng sẵn video preview ROI cho công cụ lọc tay.

Vì sao cần: `clip_review.py` cho xem VIDEO GỐC, nhưng thứ đi vào model là ô 96×96 cắt
quanh miệng. Hai thứ đó có thể là hai người khác nhau — đã gặp thật: clip gameshow
2A2WBpeuVNw_clip0015_t00284 có 4 mặt, người đang nói xếp hạng 3 về diện tích, nên
`areas.argmax()` của stage 04 cắt trúng một khán giả đang im lặng.

File này chạy ĐÚNG `detect_and_crop` của stage 04 (import trực tiếp, không chép lại
để khỏi lệch), rồi ghép chuỗi ROI với AUDIO GỐC. Người review xem một cái miệng và
nghe tiếng cùng lúc -> lộ ngay cả ba lỗi:
  - cắt nhầm mặt   : miệng đứng im trong khi tiếng đang nói
  - lồng tiếng     : miệng máy nhưng không ăn nhịp với âm
  - ảnh tĩnh       : miệng đóng băng

Phải dựng theo LÔ, không tính lúc đang review — chạy YOLO trực tiếp trong lúc xem sẽ
làm mỗi clip chờ vài giây, xem hàng nghìn clip thì không dùng được.

CÁCH DÙNG (từ thư mục gốc dự án):
  # đo chi phí trước trên mẫu nhỏ
  python src/tools/build_roi_preview.py --limit 30
  # rồi dựng cả lô
  python src/tools/build_roi_preview.py
"""

import argparse
import csv
import importlib.util
import os
import subprocess
import sys
import time

import cv2

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STAGE04 = os.path.join(_ROOT, "src", "pipeline", "04_extract_features",
                        "01_extract_features.py")


def load_stage04():
    """Import module stage 04 (tên bắt đầu bằng số -> không import thường được)."""
    sys.path.insert(0, os.path.join(_ROOT, "src", "pipeline"))
    spec = importlib.util.spec_from_file_location("stage04", _STAGE04)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_one(stage04, yolo, src_mp4, out_mp4, fps, size, scale, detect_every, conf):
    boxes, mouth = stage04.detect_and_crop(yolo, src_mp4, fps, size, detect_every, conf)
    if mouth is None:
        return "khong bat duoc mat"

    tmp = out_mp4 + ".silent.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (size * scale, size * scale))
    if not vw.isOpened():
        return "khong mo duoc VideoWriter"
    for g in mouth:                      # mouth: [T, size, size] grayscale
        big = cv2.resize(g, (size * scale, size * scale),
                         interpolation=cv2.INTER_NEAREST)
        vw.write(cv2.cvtColor(big, cv2.COLOR_GRAY2BGR))
    vw.release()

    # ghép audio GỐC vào — đây mới là thứ làm lộ lồng tiếng / cắt nhầm mặt
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", tmp, "-i", src_mp4,
         "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_mp4],
        capture_output=True, text=True)
    os.remove(tmp)
    if proc.returncode != 0:
        return f"ffmpeg fail: {proc.stderr[-160:]}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/02_curate/all_clean.csv")
    ap.add_argument("--out_dir", default="data/02_curate/roi_preview")
    ap.add_argument("--face_model", default="yolov8n-face.pt")
    ap.add_argument("--fps", type=float, default=25.0, help="khớp stage 04")
    ap.add_argument("--size", type=int, default=96, help="khớp stage 04 (--mouth_size)")
    ap.add_argument("--scale", type=int, default=3, help="phóng to cho dễ nhìn")
    ap.add_argument("--detect_every", type=int, default=2, help="khớp stage 04")
    ap.add_argument("--conf", type=float, default=0.25, help="khớp stage 04")
    ap.add_argument("--limit", type=int, default=0, help="chỉ dựng N clip đầu (đo chi phí)")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    stage04 = load_stage04()
    from ultralytics import YOLO
    yolo = YOLO(args.face_model)

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    done = skipped = 0
    errors = []
    for i, r in enumerate(rows, 1):
        out_mp4 = os.path.join(args.out_dir, r["clip_id"] + ".mp4")
        if args.skip_existing and os.path.isfile(out_mp4):
            skipped += 1
            continue
        err = build_one(stage04, yolo, r["file_path"], out_mp4, args.fps,
                        args.size, args.scale, args.detect_every, args.conf)
        if err:
            errors.append((r["clip_id"], err))
        else:
            done += 1
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(rows)}  loi={len(errors)}  "
                  f"({el/60:.1f}m, {el/max(1,i):.2f}s/clip)", flush=True)

    el = time.time() - t0
    print(f"\nXong {done} clip trong {el/60:.2f} phut "
          f"({el/max(1,done):.2f}s/clip) | bo qua {skipped} | loi {len(errors)}")
    if errors:
        print("\n-- clip loi (10 dau) --")
        for cid, e in errors[:10]:
            print(f"  {cid}: {e}")
    if not args.limit:
        return
    total = len(list(csv.DictReader(open(args.csv, encoding="utf-8"))))
    print(f"\nUOC TINH ca lo {total} clip: {el/max(1,done)*total/60:.0f} phut")


if __name__ == "__main__":
    main()
