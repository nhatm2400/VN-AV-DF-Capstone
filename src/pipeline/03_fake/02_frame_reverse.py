"""
02_frame_reverse.py — Pseudo-fake: ĐẢO NGƯỢC một đoạn VIDEO (Frame Reverse)

Tuần 2 (Pseudo-fake Engineering) — kỹ thuật thứ 2 theo docs/Pipeline.

Ý tưởng: chọn NGẪU NHIÊN một cửa sổ dài 0.3–1.0s trong clip, đảo ngược thứ tự
frame của RIÊNG đoạn đó (video chạy lùi trong cửa sổ), phần còn lại giữ xuôi;
AUDIO GIỮ NGUYÊN hoàn toàn. Kết quả: trong cửa sổ đó khẩu hình miệng chạy ngược
so với tiếng -> lệch pha cục bộ -> deepfake. Đây là loại fake THỊ GIÁC THUẦN
(đối ngẫu với 03_pitch_flatten là AUDIO thuần).

Khác với 01_temporal_desync (lệch toàn cục, đều): 02 lệch CỤC BỘ và mạnh trong
một khoảng ngắn -> bổ sung đa dạng kiểu lệch pha cho dataset.

Chống học-tủ (rất quan trọng):
  - AUDIO copy nguyên (-c:a copy) -> khác biệt nằm ở HÌNH, không phải audio.
  - Vị trí + độ dài cửa sổ NGẪU NHIÊN -> tránh model học "đảo đúng giữa clip".
  - Video buộc phải re-encode (concat + reverse không copy được). Điều này tạo
    artifact nén CHỈ trên fake -> nếu không xử lý sẽ leak codec. Bước nén SNVSM
    V2 áp ĐỐI XỨNG real+fake để giảm shortcut codec. Tuy nhiên generator V1 còn
    dùng -shortest và smoke đã thấy lệch duration; phải repair trước pilot mới.

Chỉ dùng thư viện chuẩn + ffmpeg/ffprobe trong PATH.

Ví dụ:
  python 02_frame_reverse.py --input_csv data/02_curate/all_clean.csv \\
      --out_dir data/fake --labels data/labels.csv
  # đổi dải độ dài cửa sổ đảo:
  python 02_frame_reverse.py --input_csv ... --min_sec 0.3 --max_sec 1.0
  # thử 5 clip:
  python 02_frame_reverse.py --input_csv ... --limit 5
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

# Schema nhãn DÙNG CHUNG với 01/03/04 -> cùng append vào 1 labels.csv.
LABEL_FIELDS = ["clip_id", "file_path", "label", "method", "param",
                "source_clip", "source_video", "speaker_id", "tier"]


def ffprobe_dur(path):
    """Trả duration (giây, float) hoặc None."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True
        ).stdout
        dur = float(json.loads(out)["format"]["duration"])
        return dur if dur > 0 else None
    except Exception:
        return None


def make_reverse(in_path, out_path, t0, t1):
    """
    Đảo ngược video trong [t0, t1], giữ phần trước/sau xuôi, audio copy nguyên.
    Cắt video thành 3 đoạn: [0,t0] xuôi + [t0,t1] reverse + [t1,end] xuôi -> concat.
    """
    vf = (
        f"[0:v]trim=start=0:end={t0:.4f},setpts=PTS-STARTPTS[a];"
        f"[0:v]trim=start={t0:.4f}:end={t1:.4f},setpts=PTS-STARTPTS,reverse[b];"
        f"[0:v]trim=start={t1:.4f},setpts=PTS-STARTPTS[c];"
        f"[a][b][c]concat=n=3:v=1:a=0[v]"
    )
    cmd = ["ffmpeg", "-y",
           "-i", in_path,
           "-filter_complex", vf,
           "-map", "[v]", "-map", "0:a",
           "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
           "-pix_fmt", "yuv420p",
           "-c:a", "copy",
           "-shortest", "-loglevel", "error", out_path]
    subprocess.run(cmd, capture_output=True)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="data/02_curate/all_clean.csv",
                    help="CSV clip real (cần cột file_path) — mặc định tập sạch từ 04_curate")
    ap.add_argument("--path_col", default="file_path")
    ap.add_argument("--id_col", default="clip_id")
    ap.add_argument("--out_dir", default="data/03_fake")
    ap.add_argument("--labels", default="data/03_fake/labels.csv")
    ap.add_argument("--min_sec", type=float, default=0.3, help="độ dài tối thiểu cửa sổ đảo (s)")
    ap.add_argument("--max_sec", type=float, default=1.0, help="độ dài tối đa cửa sổ đảo (s)")
    ap.add_argument("--n_per_clip", type=int, default=1, help="số fake sinh mỗi clip")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="chỉ xử lý N clip đầu (để test)")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.input_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]
    print(f"Đọc {len(rows)} clip real từ {args.input_csv}")

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

        dur = ffprobe_dur(src_path)
        if dur is None:
            skipped += 1
            continue

        for k in range(args.n_per_clip):
            d = random.uniform(args.min_sec, args.max_sec)
            if dur <= d + 0.4:               # không đủ chỗ chừa 2 đầu -> bỏ
                skipped += 1
                continue
            t0 = random.uniform(0.1, dur - d - 0.1)
            t1 = t0 + d
            fake_id = f"{src_id}_rev{int(round(d*1000))}ms" + (f"_{k}" if args.n_per_clip > 1 else "")
            out_path = os.path.abspath(os.path.join(args.out_dir, fake_id + ".mp4"))

            if make_reverse(src_path, out_path, t0, t1):
                writer.writerow({
                    "clip_id": fake_id,
                    "file_path": out_path,
                    "label": 1,
                    "method": "frame_reverse",
                    "param": f"reverse=[{t0:.2f},{t1:.2f}]s({d:.2f}s)",
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
    print("Lưu ý: video re-encode -> nén SNVSM 4 mức CRF ở bước sau sẽ đồng bộ codec real+fake.")


if __name__ == "__main__":
    main()
