"""
01_temporal_desync.py — Pseudo-fake: LỆCH PHA audio-visual (Temporal Desync)

Tuần 2 (Pseudo-fake Engineering) — kỹ thuật ƯU TIÊN LÀM ĐẦU theo docs/Pipeline.

Ý tưởng: dịch audio so với video một khoảng = 3 / 7 / 15 frames (frame-based, đổi
ra giây theo fps thật của clip). Đây là loại fake đúng mục tiêu PAMF: tiếng và
khẩu hình lệch pha.

Chống học-tủ (rất quan trọng):
  - Video stream được COPY nguyên (không re-encode) -> khác biệt DUY NHẤT giữa
    real và fake là TIMING audio, không phải artifact nén. Nếu re-encode video,
    model có thể bắt codec thay vì học lệch pha.
  - Độ lệch + hướng (audio sớm/muộn) chọn NGẪU NHIÊN -> tránh model học "đúng
    0.4s = fake". (Bước nén SNVSM 4 mức CRF ở sau sẽ đồng bộ codec cho cả real
    lẫn fake, xóa nốt chênh lệch do re-encode audio.)

Chỉ dùng thư viện chuẩn + ffmpeg/ffprobe trong PATH (không cần ffmpeg-python).

Ví dụ:
  # Mỗi clip 1 fake, lệch ngẫu nhiên trong {3,7,15} frames, hướng ngẫu nhiên
  python 01_temporal_desync.py --input_csv data/clips/tier1_clean.csv \\
      --out_dir data/fake --labels data/labels.csv

  # Sinh cả 3 mức lệch cho mỗi clip
  python 01_temporal_desync.py --input_csv ... --mode all

  # Thử 5 clip để kiểm
  python 01_temporal_desync.py --input_csv ... --limit 5
"""

import os
import sys
import csv
import json
import random
import argparse
import subprocess

try:                                  # in được tiếng Việt trên console Windows (cp1252)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SHIFTS = [3, 7, 15]   # frames — theo pipeline
LABEL_FIELDS = ["clip_id", "file_path", "label", "method", "shift_frames",
                "source_clip", "source_video", "speaker_id", "tier"]


def ffprobe_fps_dur(path):
    """Trả (fps_float, duration_sec) hoặc (None, None)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate:format=duration",
             "-of", "json", path],
            capture_output=True, text=True
        ).stdout
        d = json.loads(out)
        rate = d["streams"][0]["avg_frame_rate"]      # vd "30000/1001"
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        dur = float(d["format"]["duration"])
        if fps <= 0 or dur <= 0:
            return None, None
        return fps, dur
    except Exception:
        return None, None


def make_desync(in_path, out_path, offset_sec):
    """
    Ghép video(copy) + audio dịch offset_sec. offset>0: audio MUỘN hơn;
    offset<0: audio SỚM hơn. Video luôn copy nguyên (không đụng hình).
    """
    cmd = ["ffmpeg", "-y",
           "-i", in_path,                       # input0: lấy video
           "-itsoffset", f"{offset_sec:.4f}", "-i", in_path,  # input1: audio đã dịch
           "-map", "0:v", "-map", "1:a",
           "-c:v", "copy", "-c:a", "aac",
           "-shortest", "-loglevel", "error", out_path]
    subprocess.run(cmd, capture_output=True)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="data/curate/all_clean.csv",
                    help="CSV clip real (cần cột file_path) — mặc định tập sạch từ 04_curate")
    ap.add_argument("--path_col", default="file_path")
    ap.add_argument("--id_col", default="clip_id")
    ap.add_argument("--out_dir", default="data/fake")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--mode", choices=["random", "all"], default="random",
                    help="random: 1 fake/clip lệch ngẫu nhiên; all: 1 fake cho mỗi mức 3/7/15")
    ap.add_argument("--shifts", default="3,7,15", help="danh sách frame lệch, vd '3,7,15'")
    ap.add_argument("--both_dirs", action="store_true",
                    help="cho phép audio sớm HOẶC muộn (mặc định: chỉ ngẫu nhiên ở mode random)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="chỉ xử lý N clip đầu (để test)")
    args = ap.parse_args()

    random.seed(args.seed)
    shifts = [int(s) for s in args.shifts.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.input_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]
    print(f"Đọc {len(rows)} clip real từ {args.input_csv}")

    # mở labels (ghi tiếp nếu đã có, thêm header nếu mới)
    new_file = not os.path.exists(args.labels) or os.path.getsize(args.labels) == 0
    os.makedirs(os.path.dirname(os.path.abspath(args.labels)) or ".", exist_ok=True)
    lf = open(args.labels, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(lf, fieldnames=LABEL_FIELDS)
    if new_file:
        writer.writeheader()

    made = skipped = failed = 0
    for i, r in enumerate(rows):
        src_path = r.get(args.path_col, "")
        src_id = r.get(args.id_col, f"clip{i:06d}")
        if not src_path or not os.path.isfile(src_path):
            skipped += 1
            continue

        fps, dur = ffprobe_fps_dur(src_path)
        if fps is None:
            skipped += 1
            continue

        chosen = shifts if args.mode == "all" else [random.choice(shifts)]
        for n in chosen:
            t = n / fps
            if t >= dur:                      # lệch dài hơn clip -> bỏ
                skipped += 1
                continue
            sign = random.choice([1, -1]) if (args.both_dirs or args.mode == "random") else 1
            offset = sign * t
            tag = f"desync{'p' if sign > 0 else 'm'}{n}f"
            fake_id = f"{src_id}_{tag}"
            out_path = os.path.abspath(os.path.join(args.out_dir, fake_id + ".mp4"))

            if make_desync(src_path, out_path, offset):
                writer.writerow({
                    "clip_id": fake_id,
                    "file_path": out_path,
                    "label": 1,
                    "method": "temporal_desync",
                    "shift_frames": sign * n,
                    "source_clip": src_id,
                    "source_video": r.get("source_video", ""),
                    "speaker_id": r.get("speaker_id", ""),
                    "tier": r.get("tier", ""),
                })
                made += 1
            else:
                failed += 1

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}  made={made} skip={skipped} fail={failed}")

    lf.close()
    print(f"\nXong. Fake tạo được: {made} | bỏ qua: {skipped} | lỗi: {failed}")
    print(f"  Video -> {args.out_dir}/  | nhãn (append) -> {args.labels}")
    print("Lưu ý: nén SNVSM 4 mức CRF ở bước sau sẽ đồng bộ codec cho cả real+fake.")


if __name__ == "__main__":
    main()
