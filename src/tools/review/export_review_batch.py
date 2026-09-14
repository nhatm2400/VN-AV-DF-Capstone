"""
export_review_batch.py — Gom clip gốc của một manifest vào MỘT thư mục để phát cho reviewer.

Vì sao cần: chỉ gửi đúng các clip đã giao cho một reviewer thay vì chuyển toàn bộ
dataset cho mọi thành viên.

Copy theo `clip_id`, đặt tên `<clip_id>.mp4` phẳng và tạo một assignment portable
ngay trong thư mục output. Người review có thể giải nén ở bất kỳ đâu.

CÁCH DÙNG (từ thư mục gốc dự án):
  python src/tools/review/export_review_batch.py --csv <assignment.csv> --out_dir <reviewer_folder>
"""

import argparse
import csv
import os
import shutil
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/manifests/dataset_v1/review.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--col", default="file_path")
    ap.add_argument("--dry_run", action="store_true",
                    help="chỉ đếm file và dung lượng, không copy")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    missing = [r["clip_id"] for r in rows if not os.path.isfile(r[args.col])]
    if missing:
        raise SystemExit(f"[LỖI] {len(missing)} clip không có trên đĩa, "
                         f"vd {missing[:3]}")

    total = sum(os.path.getsize(r[args.col]) for r in rows)
    print(f"Manifest : {args.csv}  ({len(rows)} clip, {total / 2**30:.2f} GiB)")
    print(f"Đích     : {args.out_dir}")
    if args.dry_run:
        print("(dry run — không copy)")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    copied = skipped = 0
    for i, r in enumerate(rows, 1):
        dst = os.path.join(args.out_dir, r["clip_id"] + ".mp4")
        src = r[args.col]
        # đã copy đủ và đúng kích thước -> bỏ qua, để chạy lại được sau khi ngắt
        if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
        if i % 500 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(rows)}  ({el / 60:.1f}m)", flush=True)

    n_out = len([f for f in os.listdir(args.out_dir) if f.endswith(".mp4")])
    size_out = sum(os.path.getsize(os.path.join(args.out_dir, f))
                   for f in os.listdir(args.out_dir) if f.endswith(".mp4"))
    print(f"\nXong trong {(time.time() - t0) / 60:.1f} phút | "
          f"copy {copied}, bỏ qua {skipped}")
    print(f"Thư mục đích: {n_out} file, {size_out / 2**30:.2f} GiB")
    if n_out != len(rows):
        raise SystemExit(f"[LỖI] Đích có {n_out} file, manifest có {len(rows)}")

    portable_path = os.path.join(args.out_dir, os.path.basename(args.csv))
    portable_rows = []
    for row in rows:
        portable = dict(row)
        portable[args.col] = row["clip_id"] + ".mp4"
        portable_rows.append(portable)
    fields = list(rows[0])
    partial = portable_path + ".partial"
    with open(partial, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(portable_rows)
    os.replace(partial, portable_path)
    print(f"Assignment portable: {portable_path}")


if __name__ == "__main__":
    main()
