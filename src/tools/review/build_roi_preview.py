"""Build mouth ROI previews with original audio for manual review.
The cropper is independent of the retired AVSP feature extractor.
These review crops are not yet validated as AV-HuBERT inputs."""

import argparse
import csv
import importlib.util
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def load_cropper():
    from pathlib import Path
    path = Path(__file__).with_name("mouth_roi.py")
    spec = importlib.util.spec_from_file_location("review_mouth_roi", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mux_original_audio(silent_mp4, src_mp4, out_mp4):
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", silent_mp4, "-i", src_mp4,
         "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_mp4],
        capture_output=True, text=True)
    if proc.returncode != 0:
        return f"ffmpeg fail: {proc.stderr[-160:]}"
    return None


def write_no_face_slate(src_mp4, silent_mp4, fps, side):
    """Tạo hình báo không có ROI; audio gốc được mux ở bước kế tiếp."""
    cap = cv2.VideoCapture(src_mp4)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if source_fps <= 0 or source_frames <= 0:
        return "khong doc duoc duration de tao no-face slate"
    frame_count = max(1, math.ceil(source_frames / source_fps * fps))
    vw = cv2.VideoWriter(silent_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (side, side))
    if not vw.isOpened():
        return "khong mo duoc VideoWriter cho no-face slate"
    frame = np.full((side, side, 3), 18, dtype=np.uint8)
    lines = ("NO FACE ROI", "AUDIO ONLY")
    for index, text in enumerate(lines):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.62
        thickness = 2
        width, height = cv2.getTextSize(text, font, scale, thickness)[0]
        x = max(8, (side - width) // 2)
        y = side // 2 - 10 + index * (height + 18)
        cv2.putText(frame, text, (x, y), font, scale, (230, 230, 230),
                    thickness, cv2.LINE_AA)
    for _ in range(frame_count):
        vw.write(frame)
    vw.release()
    return None


def build_one(cropper, yolo, src_mp4, out_mp4, fps, size, scale, detect_every, conf):
    boxes, mouth = cropper.detect_and_crop(yolo, src_mp4, fps, size, detect_every, conf)
    tmp = out_mp4 + ".silent.mp4"
    no_face = mouth is None
    if no_face:
        err = write_no_face_slate(src_mp4, tmp, fps, size * scale)
        if err:
            return err, False
    else:
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (size * scale, size * scale))
        if not vw.isOpened():
            return "khong mo duoc VideoWriter", False
        for g in mouth:                  # mouth: [T, size, size] grayscale
            big = cv2.resize(g, (size * scale, size * scale),
                             interpolation=cv2.INTER_NEAREST)
            vw.write(cv2.cvtColor(big, cv2.COLOR_GRAY2BGR))
        vw.release()

    # ghép audio GỐC vào — đây mới là thứ làm lộ lồng tiếng / cắt nhầm mặt
    err = mux_original_audio(tmp, src_mp4, out_mp4)
    os.remove(tmp)
    return err, no_face and err is None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/manifests/dataset_v1/curated.csv")
    ap.add_argument("--out_dir", default="cache/previews/dataset_v1")
    ap.add_argument("--face_model", default="weights/face_detector/yolov8n-face.pt")
    ap.add_argument("--fps", type=float, default=25.0, help="review sampling")
    ap.add_argument("--size", type=int, default=96, help="review crop size")
    ap.add_argument("--scale", type=int, default=3, help="phóng to cho dễ nhìn")
    ap.add_argument("--detect_every", type=int, default=2, help="review sampling")
    ap.add_argument("--conf", type=float, default=0.25, help="review sampling")
    ap.add_argument("--limit", type=int, default=0, help="chỉ dựng N clip đầu (đo chi phí)")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    cropper = load_cropper()
    from ultralytics import YOLO
    yolo = YOLO(args.face_model)

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    done = skipped = no_face = 0
    errors = []
    for i, r in enumerate(rows, 1):
        out_mp4 = os.path.join(args.out_dir, r["clip_id"] + ".mp4")
        if args.skip_existing and os.path.isfile(out_mp4):
            skipped += 1
            continue
        err, used_no_face_slate = build_one(
            cropper, yolo, r["file_path"], out_mp4, args.fps,
            args.size, args.scale, args.detect_every, args.conf)
        if err:
            errors.append((r["clip_id"], err))
        else:
            done += 1
            no_face += int(used_no_face_slate)
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(rows)}  loi={len(errors)}  "
                  f"({el/60:.1f}m, {el/max(1,i):.2f}s/clip)", flush=True)

    el = time.time() - t0
    print(f"\nXong {done} clip trong {el/60:.2f} phut "
          f"({el/max(1,done):.2f}s/clip) | bo qua {skipped} | "
          f"no-face audio-only {no_face} | loi {len(errors)}")
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
