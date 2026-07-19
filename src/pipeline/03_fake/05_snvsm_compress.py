"""
05_snvsm_compress.py — SNVSM: đồng bộ codec real+fake bằng nén H.264 đa mức CRF

Stage cầu nối giữa 03_fake và 04_extract_features (chạy SAU khi sinh fake, TRƯỚC
khi trích feature). KHÔNG sinh fake — đây là bước NORMALIZE áp ĐỐI XỨNG cho cả
clip real lẫn fake.

Vấn đề nó giải quyết (chống leakage codec):
  - 02_frame_reverse re-encode video bằng libx264; 04_anonymization xuất mpeg4;
    01/03 và real thì giữ codec gốc (H.264 tải từ YouTube). => real và fake có
    "vân codec" khác nhau. Nếu để nguyên, model học "codec lạ = fake" thay vì học
    bản chất lệch pha/prosody.
  - Cách xử lý: đưa MỌI clip (real + mọi loại fake) qua CÙNG một pipeline libx264
    ở các mức CRF {23,30,35,40}. Sau bước này real và fake chia sẻ y hệt codec +
    dải nén => xóa vân codec, đồng thời tăng đa dạng chất lượng nén (augmentation).

Cách dùng (chạy 2 lần — real và fake, cùng tham số => codec khớp tuyệt đối):
  # REAL
  python src/pipeline/03_fake/05_snvsm_compress.py \
      --input_csv data/02_curate/all_clean.csv \
      --out_dir data/03_fake/snvsm/real --out_manifest data/03_fake/snvsm/real_snvsm.csv
  # FAKE
  python src/pipeline/03_fake/05_snvsm_compress.py \
      --input_csv data/03_fake/labels.csv \
      --out_dir data/03_fake/snvsm/fake --out_manifest data/03_fake/snvsm/fake_snvsm.csv

Rồi trỏ 04/05 vào manifest SNVSM:
  python src/pipeline/05_build_labels/01_build_labels.py \
      --real_csv data/03_fake/snvsm/real_snvsm.csv \
      --fake_labels data/03_fake/snvsm/fake_snvsm.csv
  python src/pipeline/04_extract_features/01_extract_features.py \
      --real_csv data/03_fake/snvsm/real_snvsm.csv \
      --fake_labels data/03_fake/snvsm/fake_snvsm.csv

Manifest ra: mọi cột của input được giữ nguyên; clip_id -> "<id>_crf{N}",
file_path -> clip nén, thêm cột crf + orig_clip_id. speaker_id/source_video/…
copy nguyên nên split speaker-disjoint ở 05 vẫn đúng.

--mode random (mặc định): 1 mức CRF ngẫu nhiên/clip -> ×1 dung lượng (khuyến nghị).
--mode all: đủ 4 mức/clip -> ×4 (augmentation tối đa, tốn đĩa/thời gian).

Chỉ dùng thư viện chuẩn + ffmpeg trong PATH.
"""

import os
import sys
import csv
import random
import argparse
import subprocess

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CRFS_DEFAULT = [23, 30, 35, 40]


def compress(in_path, out_path, crf, preset, encoder):
    """
    Re-encode video @CRF (audio copy, fallback aac). CÙNG encoder cho real+fake.
      libx264   : -crf N (CPU, chuẩn)
      h264_nvenc: -cq N  (GPU NVENC — nhanh, không chiếm CPU; CQ≈CRF)
    """
    if encoder == "h264_nvenc":
        venc = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    else:
        venc = ["-c:v", "libx264", "-crf", str(crf), "-preset", preset]

    def run(acodec):
        cmd = ["ffmpeg", "-y", "-i", in_path, *venc, "-pix_fmt", "yuv420p",
               "-c:a", acodec, "-movflags", "+faststart", "-loglevel", "error", out_path]
        if acodec == "aac":
            cmd[cmd.index("aac") + 1:cmd.index("aac") + 1] = ["-b:a", "128k"]
        subprocess.run(cmd, capture_output=True)
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    return run("copy") or run("aac")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True,
                    help="CSV clip (real all_clean.csv HOẶC fake 03_fake/labels.csv)")
    ap.add_argument("--path_col", default="file_path")
    ap.add_argument("--id_col", default="clip_id")
    ap.add_argument("--out_dir", default="data/03_fake/snvsm")
    ap.add_argument("--out_manifest", required=True, help="CSV manifest cho 04/05 đọc tiếp")
    ap.add_argument("--crfs", default="23,30,35,40")
    ap.add_argument("--mode", choices=["random", "all"], default="random",
                    help="random: 1 CRF ngẫu nhiên/clip (×1); all: đủ 4 mức/clip (×4)")
    ap.add_argument("--preset", default="veryfast", help="chỉ áp cho libx264")
    ap.add_argument("--encoder", choices=["libx264", "h264_nvenc"], default="libx264",
                    help="h264_nvenc: encode GPU (nhanh, không chiếm CPU)")
    ap.add_argument("--skip_existing", action="store_true", default=True,
                    help="bỏ encode clip đã có -> RESUME (manifest vẫn ghi đủ)")
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    crfs = [int(c) for c in args.crfs.split(",") if c.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.input_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        in_fields = reader.fieldnames or []
        rows = list(reader)
    if args.limit:
        rows = rows[:args.limit]
    print(f"Đọc {len(rows)} clip từ {args.input_csv} | mode={args.mode} | crfs={crfs}")

    out_fields = list(in_fields)
    for extra in ("crf", "orig_clip_id"):
        if extra not in out_fields:
            out_fields.append(extra)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_manifest)) or ".", exist_ok=True)
    mf = open(args.out_manifest, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(mf, fieldnames=out_fields)
    writer.writeheader()

    made = resumed = skipped = failed = 0
    for i, r in enumerate(rows):
        src_path = r.get(args.path_col, "")
        src_id = r.get(args.id_col, f"clip{i:06d}")
        if not src_path or not os.path.isfile(src_path):
            skipped += 1
            continue

        # CRF theo RNG per-clip (deterministic + resume-safe: cùng clip -> cùng CRF)
        if args.mode == "all":
            chosen = crfs
        else:
            chosen = [random.Random(f"{args.seed}:{src_id}").choice(crfs)]
        for crf in chosen:
            new_id = f"{src_id}_crf{crf}"
            out_path = os.path.abspath(os.path.join(args.out_dir, new_id + ".mp4"))
            exists = args.skip_existing and os.path.exists(out_path) and os.path.getsize(out_path) > 0
            ok = exists or compress(src_path, out_path, crf, args.preset, args.encoder)
            if ok:
                row = dict(r)
                row[args.id_col] = new_id
                row[args.path_col] = out_path
                row["crf"] = crf
                row["orig_clip_id"] = src_id
                writer.writerow(row)               # manifest luôn ghi đủ (kể cả clip resume)
                if exists:
                    resumed += 1
                else:
                    made += 1
            else:
                failed += 1

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}  made={made} resume={resumed} skip={skipped} fail={failed}")
            mf.flush()

    mf.close()
    print(f"\nXong. Clip nén: {made} | resume(đã có): {resumed} | bỏ qua: {skipped} | lỗi: {failed}")
    print(f"  Video -> {args.out_dir}/  | manifest -> {args.out_manifest}")
    print("Nhớ chạy CẢ real LẪN fake với CÙNG --crfs/--preset để codec khớp tuyệt đối.")


if __name__ == "__main__":
    main()
