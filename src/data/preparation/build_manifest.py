"""Merge explicitly selected accepted-clip manifests, verify 1:1 media coverage and remap paths.
Use --add GROUP CSV_GLOB CLIPS_ROOT for each group; no automatic legacy-data discovery."""

import os
import csv
import glob
import argparse
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FIELDS = [
    "clip_id", "source_video", "start_time", "end_time", "duration",
    "face_ratio", "speech_ratio", "snr", "file_path", "tier", "has_cut_meta",
    "decode_backend", "cut_backend", "run_id",
]

# Mặc định local: chạy thẳng `python 01_prep_manifest.py` không cần tham số.
# (Override bằng --add khi chạy nơi khác, vd Kaggle /kaggle/input/...)



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
                clip_id = r.get("clip_id", "").strip()
                if not clip_id:
                    raise ValueError(f"clip_id rỗng trong {f}")
                if clip_id in meta:
                    raise ValueError(
                        f"clip_id trùng giữa batch/CSV: {clip_id}"
                    )
                meta[clip_id] = r
                n += 1
        print(f"    + {f}  ({n} dong)")
    return meta


def prep_tier(tier, csv_glob, clips_root, verify=True):
    print(f"\n[{tier}]")
    meta = load_meta(csv_glob)

    rows = []
    mp4s = sorted(glob.glob(os.path.join(clips_root, "**", "*.mp4"), recursive=True))
    media = {}
    for path in mp4s:
        clip_id = os.path.splitext(os.path.basename(path))[0]
        if clip_id in media:
            raise ValueError(f"MP4 trùng clip_id dưới {clips_root}: {clip_id}")
        media[clip_id] = path

    missing_media = sorted(set(meta) - set(media))
    orphan_media = sorted(set(media) - set(meta))
    if missing_media or orphan_media:
        raise ValueError(
            f"{tier} accepted/media không 1-1: thiếu={len(missing_media)} "
            f"vd={missing_media[:3]}, mồ_côi={len(orphan_media)} "
            f"vd={orphan_media[:3]}"
        )

    corrupt = []
    for clip_id, m in sorted(meta.items()):
        path = media[clip_id]
        if verify and not is_valid_video(path):
            corrupt.append(clip_id)
            continue
        row = {field: m.get(field, "") for field in FIELDS}
        row.update({
            "clip_id": clip_id,
            "file_path": os.path.abspath(path),
            "tier": tier,
            "has_cut_meta": 1,
        })
        if not row["source_video"]:
            raise ValueError(f"{tier}/{clip_id} thiếu source_video trong cut-log")
        rows.append(row)
    if corrupt:
        raise ValueError(
            f"{tier} có {len(corrupt)} accepted MP4 hỏng, ví dụ {corrupt[:3]}"
        )

    print(f"    accepted={len(meta)} | mp4={len(mp4s)} | verified={len(rows)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs=3, action="append", metavar=("TIER", "CSV_GLOB", "CLIPS_ROOT"),
                    required=True, help="source group, accepted CSV glob, media root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_verify", action="store_true",
                    help="bo qua ffprobe (nhanh hon nhung giu ca file hong)")
    ap.add_argument("--overwrite", action="store_true",
                    help="cho phép ghi đè manifest đã tồn tại (mặc định từ chối)")
    args = ap.parse_args()

    verify = not args.no_verify
    adds = args.add
    all_rows = []
    for tier, csv_glob, clips_root in adds:
        all_rows.extend(prep_tier(tier, csv_glob, clips_root, verify=verify))

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(
            f"Output đã tồn tại, từ chối ghi đè: {args.out}. "
            "Dùng path staging/run mới."
        )
    partial = args.out + ".partial"
    with open(partial, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    os.replace(partial, args.out)

    print("\n===== MANIFEST =====")
    by_tier = {}
    for r in all_rows:
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
    for t, n in by_tier.items():
        print(f"  {t}: {n}")
    print(f"Tong: {len(all_rows)} clip -> {args.out}")
    if not verify:
        print("  (CHUA verify ffprobe — co the con file hong)")
    print("Next: review clips and confirm speaker/source IDs before building splits.")


if __name__ == "__main__":
    main()
