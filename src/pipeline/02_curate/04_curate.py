"""
04_curate.py — BƯỚC QUYẾT ĐỊNH (decision pass) — chạy CUỐI trong bộ 02/03/04

Đọc output của 02_score_clips.py (scored CSV + embeddings.npy), tùy chọn có thêm
cột sync_conf từ 03_sync_score.py, rồi:
  [B2] cluster embedding -> speaker_id   (agglomerative, cosine)
  [B3] gate: loại clip không mặt / mặt quá nhỏ (CHỈ rác rõ ràng)
  [B4] cân bằng: cap N clip/speaker, ưu tiên chất lượng + trải đều video/thời điểm
  [B5] xuất tập sạch + metadata đầy đủ

Triết lý (giống bài học SNR): ĐO mọi thứ, chỉ loại rác rõ ràng, giữ phần còn lại
làm metadata. Luôn chạy --calibrate trước để XEM phân bố rồi mới đặt ngưỡng.

⚠️ Đối xứng real/fake: nếu áp gate sync/chất lượng cho tập REAL thì PHẢI áp y hệt
cho tập FAKE (hoặc không gate cho tập nào). Gate một phía sẽ bơm tín hiệu nhãn
("sync thấp = fake") -> model học tắt, rò rỉ chính đặc trưng PAMF phải tự học.

sync_conf / embed_consistency: nếu CSV có thì tự đưa vào quality score; không có thì bỏ qua.
motion_median (từ 02b_motion_score.py): CHỈ dùng làm gate tùy chọn --motion_floor, KHÔNG
đưa vào quality score (tránh âm thầm đổi thành phần tập đã xuất).

LOCAL (mặc định — chạy KHÔNG cần tham số, từ thư mục gốc dự án):
  python 04_curate.py --calibrate   # 1) xem phân bố, chọn ngưỡng
  python 04_curate.py               # 2) export -> data/02_curate/manifests/all_clean.csv
  -> tự đọc data/02_curate/measurements/tier1_scored_all.csv + embeddings_all.npy

Ví dụ tùy chỉnh / KAGGLE:
  # 1) Xem phân bố + số cụm ở vài ngưỡng (không xuất gì)
  python 04_curate.py --scored_csv tier1_scored_tier1.csv --emb embeddings_tier1.npy --calibrate

  # 2) Chốt ngưỡng rồi xuất
  python 04_curate.py --scored_csv ... --emb ... \\
      --cluster_dist 0.5 --min_det_ratio 0.6 --min_face_area 0.01 --cap_per_speaker 12 \\
      --out tier1_clean.csv
"""

import argparse
import os
import sys
import numpy as np

try:                                  # in được tiếng Việt khi pipe/redirect (console cp1252)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


# ----------------------------- IO -----------------------------
def load(scored_csv, emb_path):
    df = pd.read_csv(scored_csv)
    emb = np.load(emb_path)
    assert len(df) == len(emb), f"CSV ({len(df)}) và NPY ({len(emb)}) lệch số dòng"
    return df, emb


# ----------------------------- Helpers -----------------------------
def norm_pctile(s, lo=0.05, hi=0.95):
    """Chuẩn hóa về [0..1] theo phân vị p5..p95 (bền với outlier hơn min-max thuần)."""
    a, b = s.quantile(lo), s.quantile(hi)
    return ((s - a) / (b - a + 1e-8)).clip(0, 1)


# ----------------------------- [B2] Cluster -----------------------------
def cluster_speakers(df, emb, dist):
    """
    Gom các clip CÓ embedding thành speaker_id bằng agglomerative (cosine, average linkage).
    Clip không mặt -> speaker_id = -1 (xử lý ở gate).
    """
    mask = df["has_embedding"].values.astype(bool)
    spk = np.full(len(df), -1, dtype=int)
    if mask.sum() == 0:
        df["speaker_id"] = spk
        return df

    X = emb[mask]
    cl = AgglomerativeClustering(
        n_clusters=None, distance_threshold=dist,
        metric="cosine", linkage="average",
    )
    labels = cl.fit_predict(X)
    spk[np.where(mask)[0]] = labels
    df["speaker_id"] = spk
    return df


# ----------------------------- Quality score -----------------------------
def quality_score(df):
    """
    Điểm chất lượng [0..1] để chọn clip tốt khi cân bằng.
    Thành phần chuẩn hóa NHẤT QUÁN theo phân vị p5..p95 (tránh outlier kéo lệch).
    Tự thêm sync_conf và embed_consistency nếu có; tự chia lại trọng số cho đủ 1.0.
    """
    det = df["det_ratio"].clip(0, 1)
    fa_norm = norm_pctile(df["mean_face_area"])

    parts = [(0.4, det), (0.3, fa_norm)]
    if "sync_conf" in df.columns:
        sc = df["sync_conf"].fillna(df["sync_conf"].median())
        parts.append((0.3, norm_pctile(sc)))
    if "embed_consistency" in df.columns:
        parts.append((0.15, df["embed_consistency"].clip(0, 1)))

    total_w = sum(w for w, _ in parts)
    return sum((w / total_w) * v for w, v in parts)


# ----------------------------- [B4] Balance -----------------------------
def balance(df, cap, time_col="start_time", video_col="source_video"):
    """
    Mỗi speaker giữ tối đa `cap` clip. Ưu tiên:
      1) trải rộng qua nhiều source_video khác nhau (đa dạng buổi quay)
      2) trong mỗi video, trải đều theo thời điểm (tránh clip overlap sát nhau)
      3) điểm chất lượng cao
    """
    keep = []
    for spk, g in df.groupby("speaker_id"):
        if spk == -1:
            continue
        if len(g) <= cap:
            keep.append(g)
            continue

        g = g.sort_values("quality", ascending=False).copy()
        picked = []
        # vòng round-robin theo video: mỗi vòng lấy 1 clip tốt nhất của mỗi video,
        # và trong cùng video ưu tiên clip cách xa (thời điểm) các clip đã chọn.
        groups = {v: gv.to_dict("records") for v, gv in g.groupby(video_col)}
        while len(picked) < cap and any(groups.values()):
            for v in list(groups.keys()):
                if not groups[v] or len(picked) >= cap:
                    continue
                cand = groups[v]
                chosen_times = [p[time_col] for p in picked if p[video_col] == v]
                if chosen_times:
                    # chọn clip xa nhất so với các clip đã lấy trong cùng video
                    cand.sort(key=lambda r: -min(abs(r[time_col] - t) for t in chosen_times))
                best = cand.pop(0)
                picked.append(best)
        keep.append(pd.DataFrame(picked))

    out = pd.concat(keep, ignore_index=True) if keep else df.iloc[0:0]
    return out


# ----------------------------- Calibrate -----------------------------
def calibrate(df, emb):
    print("\n===== CALIBRATE: xem phân bố trước khi đặt ngưỡng =====")
    n_face = int(df["has_embedding"].sum())
    print(f"Tổng clip: {len(df)} | có mặt: {n_face} | không mặt: {len(df)-n_face}")

    print("\n-- det_ratio (tỉ lệ frame có mặt) --")
    print(df["det_ratio"].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
    print("\n-- mean_face_area (mặt / khung) --")
    print(df["mean_face_area"].describe(percentiles=[.1, .25, .5, .75, .9]).round(4).to_string())
    if "embed_consistency" in df.columns:
        print("\n-- embed_consistency (đồng nhất danh tính giữa frame; thấp = clip lẫn người) --")
        print(df["embed_consistency"].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
    if "sync_conf" in df.columns:
        print("\n-- sync_conf --")
        print(df["sync_conf"].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
    if "motion_median" in df.columns:
        print("\n-- motion_median (0-255; thấp = khung đứng yên, ảnh tĩnh/B-roll) --")
        mm = df["motion_median"].dropna()
        print(mm.describe(percentiles=[.05, .1, .25, .5, .75, .9]).round(3).to_string())
        for t in (0.5, 1.0, 2.0):
            print(f"  < {t}: {int((mm < t).sum())} clip ({100*(mm < t).mean():.1f}%)")

    print("\n-- Số speaker theo ngưỡng cluster (cosine distance) --")
    mask = df["has_embedding"].values.astype(bool)
    X = emb[mask]
    for d in [0.3, 0.4, 0.5, 0.6, 0.7]:
        cl = AgglomerativeClustering(n_clusters=None, distance_threshold=d,
                                     metric="cosine", linkage="average")
        lab = cl.fit_predict(X)
        sizes = pd.Series(lab).value_counts()
        print(f"  dist={d}: {len(sizes)} speaker | "
              f"cụm to nhất {sizes.max()} clip | cụm 1-clip: {(sizes==1).sum()}")
    print("\nGợi ý: chọn dist sao cho số speaker hợp lý (vài trăm cho ~472 video) "
          "và cụm to nhất không nuốt quá nhiều clip. Kiểm mắt vài cụm trước khi chốt.")


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_csv", default="data/02_curate/measurements/tier1_scored_all.csv",
                    help="output của 02_score_clips (mặc định local)")
    ap.add_argument("--emb", default="data/02_curate/measurements/embeddings_all.npy")
    ap.add_argument("--calibrate", action="store_true", help="chỉ in phân bố, không xuất")
    ap.add_argument("--cluster_dist", type=float, default=0.5, help="cosine distance threshold")
    ap.add_argument("--min_det_ratio", type=float, default=0.6, help="gate: tỉ lệ frame có mặt tối thiểu")
    ap.add_argument("--min_face_area", type=float, default=0.01, help="gate: mặt/khung tối thiểu")
    ap.add_argument("--min_consistency", type=float, default=0.0,
                    help="gate: embed_consistency tối thiểu (0 = tắt). Loại clip lẫn nhiều người.")
    ap.add_argument("--sync_floor", type=float, default=None,
                    help="gate sync: loại clip RÁC RÕ RÀNG (LSE-C < floor HOẶC sync fail = "
                         "không ai nói). MẶC ĐỊNH None = không động tới sync. CHỈ đặt giá trị "
                         "CỰC THẤP lấy từ calibrate, và PHẢI áp y hệt cho tập fake (đối xứng).")
    ap.add_argument("--motion_floor", type=float, default=None,
                    help="gate motion: loại clip ảnh TĨNH (motion_median < floor HOẶC không "
                         "đo được). MẶC ĐỊNH None = không động tới motion. Chỉ áp cho tập "
                         "REAL NGUỒN trước khi sinh fake — KHÔNG lọc motion trên fake đã sinh "
                         "(anonymization làm mờ -> motion giảm -> loại lệch riêng kênh đó).")
    ap.add_argument("--cap_per_speaker", type=int, default=12)
    ap.add_argument("--out", default="data/02_curate/manifests/all_clean.csv")
    args = ap.parse_args()

    df, emb = load(args.scored_csv, args.emb)

    if args.calibrate:
        calibrate(df, emb)
        return

    # [B2] speaker_id
    df = cluster_speakers(df, emb, args.cluster_dist)
    n_spk = df.loc[df.speaker_id != -1, "speaker_id"].nunique()
    print(f"[B2] {n_spk} speaker từ {df['source_video'].nunique()} video")

    # [B3] gate — chỉ loại rác rõ ràng
    before = len(df)
    df["quality"] = quality_score(df)
    gate = (df["has_embedding"]) & \
           (df["det_ratio"] >= args.min_det_ratio) & \
           (df["mean_face_area"] >= args.min_face_area)
    if "embed_consistency" in df.columns and args.min_consistency > 0:
        gate = gate & (df["embed_consistency"] >= args.min_consistency)
    if "sync_conf" in df.columns and args.sync_floor is not None:
        # CHỈ cắt đuôi rác cực thấp (không ai nói): sync fail (NaN) hoặc LSE-C < floor.
        # KHÔNG dùng để siết clip biên. Nhớ áp y hệt floor này cho tập fake.
        sync_garbage = df["sync_conf"].isna() | (df["sync_conf"] < args.sync_floor)
        gate = gate & ~sync_garbage
        print(f"[B3] sync gate (floor={args.sync_floor}): loại thêm {int(sync_garbage.sum())} "
              f"clip rác sync (NHỚ áp cùng floor cho tập fake để đối xứng)")
    if "motion_median" in df.columns and args.motion_floor is not None:
        # Ảnh tĩnh / B-roll: frame_reverse và temporal_desync trên clip này ra fake
        # KHÔNG phân biệt được với real -> cặp nhãn là nhiễu thuần túy.
        static_garbage = df["motion_median"].isna() | (df["motion_median"] < args.motion_floor)
        gate = gate & ~static_garbage
        print(f"[B3] motion gate (floor={args.motion_floor}): loại thêm "
              f"{int(static_garbage.sum())} clip tĩnh/không đo được")
    rejected = df[~gate].copy()
    df = df[gate].copy()
    print(f"[B3] gate: giữ {len(df)}/{before}, loại {len(rejected)} "
          f"(không mặt / mặt nhỏ / mặt thưa / lẫn người)")

    # [B4] cân bằng
    df = balance(df, args.cap_per_speaker)
    print(f"[B4] sau cân bằng (cap {args.cap_per_speaker}/speaker): {len(df)} clip")
    dist = df["speaker_id"].value_counts()
    print(f"     clip/speaker: min={dist.min()} med={int(dist.median())} max={dist.max()}")

    # [B5] xuất
    df = df.sort_values(["speaker_id", "source_video", "start_time"]).reset_index(drop=True)
    out_parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_parent, exist_ok=True)
    df.to_csv(args.out, index=False)
    rej_path = args.out.replace(".csv", "_rejects.csv")
    rejected.to_csv(rej_path, index=False)
    print(f"[B5] xuất {len(df)} clip sạch -> {args.out}")
    print(f"     reject log -> {rej_path} (nên xem mẫu để xác nhận đúng là rác)")
    print(f"     speaker_id sẵn sàng cho split chống identity leakage ở bước sau.")


if __name__ == "__main__":
    main()
