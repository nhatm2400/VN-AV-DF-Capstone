"""
05_eda.py — PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) — bước CUỐI của stage 02_curate

Đọc output các bước trước rồi xuất hình + bảng markdown sẵn cho báo cáo Data Tasks:
  - tier1_scored_all.csv  (từ 02_scoring/01_face_quality.py: face stats cho TOÀN BỘ ~6888 clip)
  - all_clean.csv         (từ 04_curate: tập sạch sau gate + cân bằng, có speaker_id)  [tùy chọn]

Sinh ra:
  - <out_dir>/*.png        : từng biểu đồ rời (dễ chèn chọn lọc vào Obsidian/report)
  - <out_dir>/eda_summary.md : tổng hợp bảng số liệu (copy thẳng vào report)
  - in luôn tóm tắt ra stdout

Triết lý EDA ở đây = mô tả CHÍNH bộ real clip đã thu thập + làm sạch:
  phân bố chất lượng (duration/snr/face), cân bằng tier, định danh speaker,
  và so sánh trước/sau curate. KHÔNG cần fake hay feature extraction ở bước này
  (đó là phase modeling sau). Với pseudo-fake temporal-desync, mọi trục visual của
  fake TRÙNG real theo thiết kế (video copy nguyên byte) -> chỉ trục sync khác;
  phần đó để mục "label design + leakage audit" trong report, không cần sinh ở đây.

Chỉ dùng pandas + numpy + matplotlib -> chạy được trên Kaggle lẫn máy.

LOCAL (mặc định — chạy KHÔNG cần tham số, từ thư mục gốc dự án):
  D:/Anaconda/envs/vn_av_df/python.exe src/pipeline/02_curate/05_eda.py
  -> đọc data/02_curate/measurements/tier1_scored_all.csv
     + data/02_curate/manifests/all_clean.csv, xuất data/02_curate/eda_figs/

Ví dụ tùy chỉnh / KAGGLE:
  D:/Anaconda/envs/vn_av_df/python.exe src/pipeline/02_curate/05_eda.py --scored_csv tier1_scored_all.csv --clean_csv all_clean.csv \\
      --out_dir /kaggle/working/eda_figs
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

try:                                  # in được tiếng Việt khi pipe/redirect (console cp1252)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")  # không cần màn hình -> lưu file
import matplotlib.pyplot as plt


# Cột có thể là chuỗi rỗng (clip không có cut-meta) -> ép về số, lỗi thành NaN
NUMERIC_COLS = ["duration", "snr", "face_ratio", "speech_ratio",
                "det_ratio", "mean_face_area", "embed_consistency",
                "n_sampled", "sync_conf", "sync_min_dist", "start_time"]


# ----------------------------- IO + chuẩn hóa -----------------------------
def load_csv(path):
    df = pd.read_csv(path)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def savefig(fig, out_dir, name, figs):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    figs.append(name)
    print(f"  [fig] {name}")


def md_describe(s, label, ndigits=3):
    """Trả về 1 dòng bảng markdown từ describe của 1 Series số."""
    s = s.dropna()
    if len(s) == 0:
        return f"| {label} | 0 | - | - | - | - | - | - |"
    d = s.describe(percentiles=[.1, .5, .9])
    return (f"| {label} | {int(d['count'])} | {d['mean']:.{ndigits}f} | "
            f"{d['std']:.{ndigits}f} | {d['min']:.{ndigits}f} | "
            f"{d['10%']:.{ndigits}f} | {d['50%']:.{ndigits}f} | "
            f"{d['90%']:.{ndigits}f} | {d['max']:.{ndigits}f} |")


# ----------------------------- Các biểu đồ -----------------------------
def plot_hist(df, col, out_dir, figs, title, bins=40, xlabel=None, mask=None):
    if col not in df.columns:
        return
    s = df[col] if mask is None else df.loc[mask, col]
    s = s.dropna()
    if len(s) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(s, bins=bins, color="#4C72B0", edgecolor="white")
    ax.axvline(s.median(), color="#C44E52", ls="--", lw=1.5,
               label=f"median={s.median():.2f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel or col)
    ax.set_ylabel("số clip")
    ax.legend()
    savefig(fig, out_dir, f"hist_{col}.png", figs)


def plot_tier_counts(scored, clean, out_dir, figs):
    """Bar cân bằng tier: trước (toàn bộ scored) vs sau curate (clean) nếu có."""
    raw = scored["tier"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(raw))
    if clean is not None and "tier" in clean.columns:
        cln = clean["tier"].value_counts().reindex(raw.index).fillna(0)
        w = 0.38
        ax.bar(x - w / 2, raw.values, w, label="toàn bộ (scored)", color="#4C72B0")
        ax.bar(x + w / 2, cln.values, w, label="sau curate (clean)", color="#55A868")
        for i, v in enumerate(raw.values):
            ax.text(i - w / 2, v, str(int(v)), ha="center", va="bottom", fontsize=8)
        for i, v in enumerate(cln.values):
            ax.text(i + w / 2, int(v), str(int(v)), ha="center", va="bottom", fontsize=8)
        ax.legend()
    else:
        ax.bar(x, raw.values, color="#4C72B0")
        for i, v in enumerate(raw.values):
            ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(raw.index)
    ax.set_title("Số clip theo tier")
    ax.set_ylabel("số clip")
    savefig(fig, out_dir, "bar_tier_counts.png", figs)


def plot_face_presence(scored, out_dir, figs):
    if "has_embedding" not in scored.columns:
        return
    vc = scored["has_embedding"].astype(bool).value_counts()
    n_face = int(vc.get(True, 0))
    n_noface = int(vc.get(False, 0))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["có mặt", "không mặt"], [n_face, n_noface],
           color=["#55A868", "#C44E52"])
    for i, v in enumerate([n_face, n_noface]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title("Phát hiện khuôn mặt (InsightFace)")
    ax.set_ylabel("số clip")
    savefig(fig, out_dir, "bar_face_presence.png", figs)


def plot_clips_per_speaker(clean, out_dir, figs):
    if clean is None or "speaker_id" not in clean.columns:
        return
    valid = clean[clean["speaker_id"] != -1]
    if len(valid) == 0:
        return
    sizes = valid["speaker_id"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sizes.values, bins=range(1, int(sizes.max()) + 2),
            color="#8172B3", edgecolor="white", align="left")
    ax.set_title(f"Số clip / speaker  ({len(sizes)} speaker, tập sạch)")
    ax.set_xlabel("số clip trong 1 speaker")
    ax.set_ylabel("số speaker")
    savefig(fig, out_dir, "hist_clips_per_speaker.png", figs)


def plot_top_videos(scored, out_dir, figs, top=20):
    if "source_video" not in scored.columns:
        return
    vc = scored["source_video"].value_counts().head(top)
    fig, ax = plt.subplots(figsize=(7, max(4, top * 0.28)))
    ax.barh(range(len(vc))[::-1], vc.values, color="#4C72B0")
    ax.set_yticks(range(len(vc))[::-1])
    ax.set_yticklabels([str(v)[:22] for v in vc.index], fontsize=7)
    ax.set_title(f"Top {top} video gốc theo số clip")
    ax.set_xlabel("số clip")
    savefig(fig, out_dir, "barh_top_videos.png", figs)


# ----------------------------- Tóm tắt markdown -----------------------------
def build_summary_md(scored, clean, figs):
    L = []
    L.append("# EDA — VN-AV-DF Dataset (real clips)\n")

    # --- Tổng quan ---
    n = len(scored)
    n_face = int(scored["has_embedding"].sum()) if "has_embedding" in scored else None
    n_meta = int(scored["has_cut_meta"].sum()) if "has_cut_meta" in scored else None
    n_vid = scored["source_video"].nunique() if "source_video" in scored else None
    L.append("## 1. Tổng quan\n")
    L.append("| Chỉ số | Giá trị |")
    L.append("|---|---|")
    L.append(f"| Tổng clip (scored) | {n} |")
    if n_vid is not None:
        L.append(f"| Số video gốc | {n_vid} |")
    if n_face is not None:
        L.append(f"| Clip có mặt | {n_face} ({n_face/n*100:.1f}%) |")
        L.append(f"| Clip không mặt | {n-n_face} ({(n-n_face)/n*100:.1f}%) |")
    if n_meta is not None:
        L.append(f"| Có cut-metadata | {n_meta} |")
    if clean is not None:
        L.append(f"| Clip sau curate (tập sạch) | {len(clean)} |")
        if "speaker_id" in clean.columns:
            nspk = clean.loc[clean.speaker_id != -1, "speaker_id"].nunique()
            L.append(f"| Số speaker (tập sạch) | {nspk} |")
    L.append("")

    # --- Theo tier ---
    L.append("## 2. Phân bố theo tier\n")
    L.append("| Tier | Scored | Tập sạch |")
    L.append("|---|---|---|")
    raw = scored["tier"].value_counts().sort_index()
    for t in raw.index:
        c = int(clean["tier"].value_counts().get(t, 0)) if (clean is not None and "tier" in clean.columns) else "-"
        L.append(f"| {t} | {int(raw[t])} | {c} |")
    L.append("")

    # --- Bảng describe các đặc trưng ---
    L.append("## 3. Phân bố đặc trưng (toàn bộ scored)\n")
    L.append("| Đặc trưng | n | mean | std | min | p10 | median | p90 | max |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    specs = [
        ("duration", "Thời lượng (s)", 2),
        ("snr", "SNR (dB)", 2),
        ("face_ratio", "face_ratio (cut)", 3),
        ("speech_ratio", "speech_ratio (cut)", 3),
        ("det_ratio", "det_ratio (đo)", 3),
        ("mean_face_area", "mean_face_area", 4),
        ("embed_consistency", "embed_consistency", 3),
    ]
    if "sync_conf" in scored.columns:
        specs.append(("sync_conf", "sync_conf (LSE-C)", 3))
    for col, label, nd in specs:
        if col in scored.columns:
            L.append(md_describe(scored[col], label, nd))
    L.append("")

    # --- Speaker (tập sạch) ---
    if clean is not None and "speaker_id" in clean.columns:
        valid = clean[clean["speaker_id"] != -1]
        if len(valid) > 0:
            sizes = valid["speaker_id"].value_counts()
            L.append("## 4. Định danh speaker (tập sạch)\n")
            L.append("| Chỉ số | Giá trị |")
            L.append("|---|---|")
            L.append(f"| Số speaker | {len(sizes)} |")
            L.append(f"| Clip/speaker (min/median/max) | {sizes.min()} / {int(sizes.median())} / {sizes.max()} |")
            L.append(f"| Speaker chỉ 1 clip | {(sizes == 1).sum()} |")
            L.append("")
            L.append("> `speaker_id` dùng để chia train/val/test theo **speaker-disjoint**, "
                     "chống identity leakage ở bước modeling.\n")

    # --- Danh sách hình ---
    L.append("## 5. Biểu đồ\n")
    for f in figs:
        L.append(f"![{f}]({f})")
    L.append("")
    return "\n".join(L)


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_csv", default="data/02_curate/measurements/tier1_scored_all.csv",
                    help="output 02_scoring/01_face_quality.py: face stats toàn bộ clip (mặc định local)")
    ap.add_argument("--clean_csv", default="data/02_curate/manifests/all_clean.csv",
                    help="output 04_curate: tập sạch có speaker_id (tùy chọn)")
    ap.add_argument("--out_dir", default="data/02_curate/eda_figs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scored = load_csv(args.scored_csv)
    clean = load_csv(args.clean_csv) if args.clean_csv and os.path.exists(args.clean_csv) else None
    print(f"Đọc {len(scored)} clip (scored)" +
          (f" | {len(clean)} clip (clean)" if clean is not None else " | không có clean_csv"))

    figs = []
    # đặc trưng chất lượng (cut-meta)
    plot_hist(scored, "duration", args.out_dir, figs, "Phân bố thời lượng clip", xlabel="giây")
    plot_hist(scored, "snr", args.out_dir, figs, "Phân bố SNR", xlabel="dB")
    plot_hist(scored, "face_ratio", args.out_dir, figs, "Phân bố face_ratio (bước cắt)")
    plot_hist(scored, "speech_ratio", args.out_dir, figs, "Phân bố speech_ratio (bước cắt)")
    # đặc trưng đo (InsightFace) — chỉ clip có mặt cho 2 cái dưới
    plot_hist(scored, "det_ratio", args.out_dir, figs, "Phân bố det_ratio (tỉ lệ frame có mặt)")
    face_mask = scored["has_embedding"].astype(bool) if "has_embedding" in scored else None
    plot_hist(scored, "mean_face_area", args.out_dir, figs,
              "mean_face_area (clip có mặt)", mask=face_mask)
    plot_hist(scored, "embed_consistency", args.out_dir, figs,
              "embed_consistency (clip có mặt)", mask=face_mask)
    if "sync_conf" in scored.columns:
        plot_hist(scored, "sync_conf", args.out_dir, figs, "Phân bố sync_conf (LSE-C)")
    # cấu trúc dataset
    plot_tier_counts(scored, clean, args.out_dir, figs)
    plot_face_presence(scored, args.out_dir, figs)
    plot_top_videos(scored, args.out_dir, figs)
    plot_clips_per_speaker(clean, args.out_dir, figs)

    md = build_summary_md(scored, clean, figs)
    md_path = os.path.join(args.out_dir, "eda_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("\n" + "=" * 60)
    print(md)
    print("=" * 60)
    print(f"\n{len(figs)} biểu đồ + bảng tóm tắt -> {args.out_dir}")
    print(f"Markdown report -> {md_path}  (copy vào Obsidian/Report 2)")


if __name__ == "__main__":
    main()
