"""
build_review_manifest.py — Dựng manifest cho công cụ lọc tay (tái lập được).

Ghép `manifests/all_clean.csv` với các phép đo phụ trợ đã chạy sẵn, ra
`manifests/all_clean_review.csv` — scope mặc định của `clip_review.py`.

Trước đây file này được tạo bằng lệnh chạy tay và cột `channel` ghép one-off, nên mất
là không dựng lại được. Script này thay thế hoàn toàn cách làm đó.

Nguồn ghép vào (đều KHÔNG bắt buộc — thiếu cái nào thì bỏ cột đó, có cảnh báo):
  measurements/tier1_scored_motion.csv   motion_median, motion_p90, frac_near_static
      (03_diagnostics_optional/01_motion_score.py — chẩn đoán ảnh tĩnh / B-roll)
  measurements/face_ambiguity.json       n_faces_med, ratio_med, ...
      (scan_face_ambiguity.py — luật "mặt to nhất" của stage 04 có đáng tin không)
  01_collect/youtube_tier*_urls.csv      channel
      (metadata kênh; clip TikTok không tra được -> '[TIKTOK]')

KHÔNG lọc, KHÔNG sắp xếp, KHÔNG đổi số dòng — giữ nguyên `all_clean.csv`. Mọi quyết
định giữ/loại thuộc về người review; script này chỉ gom số liệu để người review nhìn.

FAIL nếu ghép làm đổi số dòng, hoặc nếu một nguồn phủ dưới `--min_coverage` (mặc định
0.95) — thà dừng còn hơn xuất manifest thiếu cột mà không ai biết.

CÁCH DÙNG (từ thư mục gốc dự án):
  D:/Anaconda/envs/vn_av_df/python.exe src/tools/review/build_review_manifest.py
"""

import argparse
import json
import os
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MOTION_COLS = ["motion_median", "motion_p90", "frac_near_static"]
FACE_COLS = ["n_faces_med", "n_faces_max", "ratio_med", "ratio_max", "cx_spread"]


def merge_checked(base, extra, cols, name, min_coverage):
    """Left-join theo clip_id, kiểm số dòng và độ phủ. Trả (df, dòng mô tả)."""
    keep = [c for c in cols if c in extra.columns]
    if not keep:
        print(f"[BỎ QUA] {name}: không có cột nào trong {cols}")
        return base, None
    before = len(base)
    out = base.merge(extra[["clip_id", *keep]], on="clip_id", how="left")
    if len(out) != before:
        raise SystemExit(f"[LỖI] {name} làm đổi số dòng {before} -> {len(out)} "
                         f"(clip_id trùng lặp trong nguồn?)")
    cov = out[keep[0]].notna().mean()
    if cov < min_coverage:
        raise SystemExit(f"[LỖI] {name} chỉ phủ {cov:.1%} < {min_coverage:.0%}. "
                         f"Chạy lại bước đo tương ứng trước.")
    return out, f"{name}: {len(keep)} cột, phủ {cov:.1%}"


def load_channels(paths):
    frames = []
    for p in paths:
        if os.path.isfile(p):
            frames.append(pd.read_csv(p)[["video_id", "channel"]])
        else:
            print(f"[BỎ QUA] không thấy {p}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).drop_duplicates("video_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_csv", default="data/02_curate/manifests/all_clean.csv")
    ap.add_argument("--motion_csv",
                    default="data/02_curate/measurements/tier1_scored_motion.csv")
    ap.add_argument("--face_json",
                    default="data/02_curate/measurements/face_ambiguity.json")
    ap.add_argument("--url_csv", nargs="*", default=[
        "data/01_collect/youtube_tier1_urls.csv",
        "data/01_collect/youtube_tier2_urls.csv",
    ])
    ap.add_argument("--out", default="data/02_curate/manifests/all_clean_review.csv")
    ap.add_argument("--min_coverage", type=float, default=0.95)
    args = ap.parse_args()

    df = pd.read_csv(args.clean_csv)
    n0 = len(df)
    print(f"Nguồn    : {args.clean_csv}  ({n0} clip)")
    if df.clip_id.duplicated().any():
        raise SystemExit("[LỖI] clip_id trùng lặp trong all_clean.csv")

    notes = []
    if os.path.isfile(args.motion_csv):
        df, note = merge_checked(df, pd.read_csv(args.motion_csv), MOTION_COLS,
                                 "motion", args.min_coverage)
        notes.append(note)
    else:
        print(f"[BỎ QUA] không thấy {args.motion_csv}")

    if os.path.isfile(args.face_json):
        face = pd.DataFrame(json.load(open(args.face_json, encoding="utf-8")))
        df, note = merge_checked(df, face, FACE_COLS, "face_ambiguity",
                                 args.min_coverage)
        notes.append(note)
    else:
        print(f"[BỎ QUA] không thấy {args.face_json}")

    urls = load_channels(args.url_csv)
    if urls is not None:
        before = len(df)
        df = df.merge(urls, left_on="source_video", right_on="video_id", how="left")
        df = df.drop(columns=["video_id"])
        if len(df) != before:
            raise SystemExit(f"[LỖI] merge channel đổi số dòng {before} -> {len(df)}")
        n_tiktok = int(df.channel.isna().sum())
        df["channel"] = df.channel.fillna("[TIKTOK]")
        notes.append(f"channel: {df.channel.nunique()} kênh, "
                     f"{n_tiktok} clip không có metadata -> '[TIKTOK]'")

    if len(df) != n0:
        raise SystemExit(f"[LỖI] số dòng đổi {n0} -> {len(df)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print()
    for n in notes:
        if n:
            print(f"  {n}")
    print(f"\n-> {args.out}  ({len(df)} dòng, {len(df.columns)} cột)")
    print(
        "Tiếp theo: D:\\Anaconda\\envs\\vn_av_df\\python.exe "
        "src/tools/review/clip_review.py"
    )


if __name__ == "__main__":
    main()
