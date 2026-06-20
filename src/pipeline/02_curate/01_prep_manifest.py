"""
01_prep_manifest.py — CHUẨN BỊ đầu vào cho 03a (chạy TRƯỚC bộ 02/03/04)

Ba vấn đề thực tế cần xử lý trước khi đo:
  1) file_path trong CSV là đường dẫn Kaggle (/kaggle/working/clips/...), KHÔNG tồn
     tại ở máy/môi trường khác -> cv2 fail ÂM THẦM -> mọi clip thành "không mặt".
  2) Trên đĩa có nhiều .mp4 HƠN số dòng CSV (vd tier1: 5502 mp4 / 4776 CSV). KIỂM
     THỰC TẾ: các file dư (không có trong CSV) gần như TẤT CẢ đều HỎNG — ghi dở
     lúc cắt ("moov atom not found", từ ~10 video bị cắt ngắt giữa chừng). CSV log
     đã loại đúng, KHÔNG hề ghi thiếu.
  3) => Phải VERIFY từng .mp4 bằng ffprobe và loại file hỏng (mặc định bật).

Cách chạy: liệt kê mọi .mp4 dưới --clips_root, clip_id = tên file, parse
source_video + start_time từ tên ({source_video}_clip{idx}_t{start}.mp4), ghép
metadata CSV nếu có, verify ffprobe, rồi gộp tier + remap file_path về đĩa thật.
Cột has_cut_meta = 1 nếu clip có trong CSV log (đã qua đủ filter bước cắt).

Lưu ý: 4 file ở data/ (youtube_*_urls.csv = bước 00; *_quality_gate_passed.csv =
bước 02) là CẤP VIDEO, KHÔNG phải clip CSV — đừng dùng làm input cho script này.

Chỉ dùng thư viện chuẩn + ffprobe trong PATH -> chạy mọi nơi, không cần pandas.

LOCAL (mặc định — chạy KHÔNG cần tham số, chạy từ thư mục gốc dự án):
  python 01_prep_manifest.py
  -> tự gộp 3 tier trong data/clips/ (glob ** quét mọi batch), verify ffprobe,
     xuất data/clips/all_manifest.csv

Dùng tùy chỉnh / KAGGLE (truyền path /kaggle/input/...):
  python 01_prep_manifest.py \\
    --add tier1 "data/clips/tier1/**/*_v3_clips_*.csv" "data/clips/tier1" \\
    --add tier2 "data/clips/tier2/**/*_v3_clips_*.csv" "data/clips/tier2" \\
    --add tier3 "data/clips/tier3/**/*_v3_clips_*.csv" "data/clips/tier3" \\
    --out data/clips/all_manifest.csv
  # thêm --no_verify nếu muốn bỏ qua bước ffprobe (nhanh hơn, kém an toàn)

Tiếp theo: python 02_score_clips.py --input_csv <manifest> --tag <tag>
"""

import os
import csv
import glob
import argparse
import subprocess

FIELDS = ["clip_id", "source_video", "start_time", "end_time", "duration",
          "face_ratio", "speech_ratio", "snr", "file_path", "tier", "has_cut_meta"]

# Mặc định local: chạy thẳng `python 01_prep_manifest.py` không cần tham số.
# (Override bằng --add khi chạy nơi khác, vd Kaggle /kaggle/input/...)
DEFAULT_ADDS = [
    ["tier1", "data/clips/tier1/**/*_v3_clips_*.csv", "data/clips/tier1"],
    ["tier2", "data/clips/tier2/**/*_v3_clips_*.csv", "data/clips/tier2"],
    ["tier3", "data/clips/tier3/**/*_v3_clips_*.csv", "data/clips/tier3"],
]
DEFAULT_OUT = "data/clips/all_manifest.csv"


def is_valid_video(path):
    """ffprobe: clip mở được + có luồng video. Loại file ghi dở (moov atom not found)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True
        )
        return r.returncode == 0 and "video" in r.stdout
    except Exception:
        return False


def parse_name(stem):
    """{source_video}_clip{idx}_t{start} -> (source_video, start_sec) hoặc (None, None)."""
    try:
        head, start = stem.rsplit("_t", 1)
        vid, _idx = head.rsplit("_clip", 1)
        return vid, int(start)
    except ValueError:
        return None, None


def load_meta(csv_glob):
    """Đọc & gộp CSV khớp glob -> dict {clip_id: row}."""
    meta = {}
    files = sorted(glob.glob(csv_glob, recursive=True))
    if not files:
        print(f"    [canh bao] khong CSV nao khop: {csv_glob} (van chay theo dia)")
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            n = 0
            for r in csv.DictReader(fh):
                meta[r["clip_id"]] = r
                n += 1
        print(f"    + {f}  ({n} dong)")
    return meta


def prep_tier(tier, csv_glob, clips_root, verify=True):
    print(f"\n[{tier}]")
    meta = load_meta(csv_glob)

    rows, seen, parse_fail, corrupt = [], set(), 0, 0
    mp4s = sorted(glob.glob(os.path.join(clips_root, "**", "*.mp4"), recursive=True))
    for p in mp4s:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem in seen:
            continue
        seen.add(stem)

        if verify and not is_valid_video(p):
            corrupt += 1
            continue

        m = meta.get(stem)
        vid, start = parse_name(stem)
        if not m and vid is None:
            parse_fail += 1

        rows.append({
            "clip_id":      stem,
            "source_video": m["source_video"] if m else (vid or ""),
            "start_time":   m["start_time"] if m else ("" if start is None else start),
            "end_time":     m["end_time"] if m else "",
            "duration":     m["duration"] if m else "",
            "face_ratio":   m["face_ratio"] if m else "",
            "speech_ratio": m["speech_ratio"] if m else "",
            "snr":          m["snr"] if m else "",
            "file_path":    os.path.abspath(p),
            "tier":         tier,
            "has_cut_meta": 1 if m else 0,
        })

    n_meta = sum(r["has_cut_meta"] for r in rows)
    orphan = [c for c in meta if c not in seen]   # CSV có nhưng đĩa không có
    print(f"    mp4_quet={len(mp4s)} | hong(bo)={corrupt} | giu={len(rows)} "
          f"| co_cut_meta={n_meta} | thieu_meta={len(rows)-n_meta} | csv_mo_coi={len(orphan)}")
    if parse_fail:
        print(f"    [canh bao] {parse_fail} ten file khong parse duoc source_video/start_time")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs=3, action="append", metavar=("TIER", "CSV_GLOB", "CLIPS_ROOT"),
                    default=None, help="khai bao 1 tier: ten, glob CSV, thu muc goc chua .mp4 "
                    "(khong truyen -> dung mac dinh 3 tier local)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no_verify", action="store_true",
                    help="bo qua ffprobe (nhanh hon nhung giu ca file hong)")
    args = ap.parse_args()

    verify = not args.no_verify
    adds = args.add if args.add else DEFAULT_ADDS
    if not args.add:
        print("(khong co --add -> dung mac dinh 3 tier local: tier1/tier2/tier3)")
    all_rows = []
    for tier, csv_glob, clips_root in adds:
        all_rows.extend(prep_tier(tier, csv_glob, clips_root, verify=verify))

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    print("\n===== MANIFEST =====")
    by_tier = {}
    for r in all_rows:
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
    for t, n in by_tier.items():
        print(f"  {t}: {n}")
    print(f"Tong: {len(all_rows)} clip -> {args.out}")
    if not verify:
        print("  (CHUA verify ffprobe — co the con file hong)")
    print(f"Tiep theo: python 02_score_clips.py --input_csv {args.out} --tag all")


if __name__ == "__main__":
    main()
