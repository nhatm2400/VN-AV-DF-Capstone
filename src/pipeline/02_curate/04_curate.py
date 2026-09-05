"""
04_curate.py — BƯỚC QUYẾT ĐỊNH (decision pass) — chạy CUỐI trong bộ 02/03/04

Đọc output của 02_scoring/01_face_quality.py (scored CSV + embeddings.npy), tùy chọn có thêm
cột sync_conf từ 03_diagnostics_optional/02_sync_score.py, rồi:
  [B2] temporal gate đã khóa -> cluster embedding (agglomerative, cosine)
  [B3] face gate: loại clip không mặt / mặt quá nhỏ (CHỈ rác rõ ràng)
  [B4] cân bằng: cap N clip/speaker, ưu tiên chất lượng + trải đều video/thời điểm
  [B5] xuất tập sạch + metadata đầy đủ

Triết lý (giống bài học SNR): ĐO mọi thứ, chỉ loại rác rõ ràng, giữ phần còn lại
làm metadata. Luôn chạy --calibrate trước để XEM phân bố rồi mới đặt ngưỡng.

⚠️ Chỉ chạy curation trên real nguồn trước khi sinh fake. Mọi fake sau này kế thừa
quyết định của source real; không chạy ASD/gate riêng trên fake.

sync_conf, ASD score và full-frame motion chỉ là metadata/chẩn đoán, không làm gate liên tục
hay quality score. Temporal policy chỉ đi vào như quyết định nhị phân đã qua locked validation.

LOCAL (mặc định — chạy KHÔNG cần tham số, từ thư mục gốc dự án):
  D:/Anaconda/envs/vn_av_df/python.exe src/pipeline/02_curate/04_curate.py --calibrate   # 1) xem phân bố, chọn ngưỡng
  D:/Anaconda/envs/vn_av_df/python.exe src/pipeline/02_curate/04_curate.py               # 2) export -> data/02_curate/manifests/all_clean.csv
  -> tự đọc data/02_curate/measurements/tier1_scored_all.csv + embeddings_all.npy

Ví dụ tùy chỉnh / KAGGLE:
  # 1) Xem phân bố + số cụm ở vài ngưỡng (không xuất gì)
  D:/Anaconda/envs/vn_av_df/python.exe src/pipeline/02_curate/04_curate.py --scored_csv tier1_scored_tier1.csv --emb embeddings_tier1.npy --calibrate

  # 2) Chốt ngưỡng rồi xuất
  D:/Anaconda/envs/vn_av_df/python.exe src/pipeline/02_curate/04_curate.py --scored_csv ... --emb ... \\
      --cluster_dist 0.5 --min_det_ratio 0.6 --min_face_area 0.01 --cap_per_speaker 12 \\
      --out tier1_clean.csv
"""

import argparse
import hashlib
import json
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
    if df.empty:
        raise ValueError("Scored CSV is empty")
    if "clip_id" not in df.columns:
        raise ValueError("Scored CSV is missing clip_id")
    if df["clip_id"].isna().any() or df["clip_id"].duplicated().any():
        raise ValueError("clip_id must be non-empty and unique")
    return df, emb


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_temporal_scores(df, scores_path, policy_path):
    """Validate and attach exactly one temporal decision per source-real clip."""
    if bool(scores_path) != bool(policy_path):
        raise ValueError("--temporal_scores and --temporal_policy must be provided together")
    if not scores_path:
        return df, None
    scores = pd.read_csv(scores_path)
    required = {
        "clip_id", "temporal_decision", "temporal_reason", "config_hash",
        "voiced_ms", "visible_active_speech_ratio", "unexplained_speech_ratio",
        "longest_unexplained_speech_ms", "static_speech_ratio",
        "asd_disagreement_ratio",
    }
    missing = sorted(required - set(scores.columns))
    if missing or scores.empty:
        raise ValueError(f"Invalid temporal score CSV; missing={missing}")
    if scores["clip_id"].isna().any() or scores["clip_id"].duplicated().any():
        raise ValueError("Temporal scores require non-empty unique clip_id")
    expected = set(df["clip_id"].astype(str))
    actual = set(scores["clip_id"].astype(str))
    if expected != actual or len(scores) != len(df):
        raise ValueError(
            f"Temporal coverage mismatch: expected={len(expected)}, actual={len(actual)}, "
            f"missing={len(expected-actual)}, extra={len(actual-expected)}"
        )
    config_hashes = set(scores["config_hash"].astype(str))
    if len(config_hashes) != 1:
        raise ValueError("Temporal scores contain multiple config hashes")
    run_config_path = os.path.join(os.path.dirname(os.path.abspath(scores_path)), "run_config.json")
    if not os.path.isfile(run_config_path):
        raise FileNotFoundError(f"Missing run_config.json beside temporal scores: {run_config_path}")
    with open(run_config_path, encoding="utf-8") as handle:
        run_config = json.load(handle)
    if not run_config.get("coverage_passed", False):
        raise ValueError("Temporal run did not pass coverage")
    if str(run_config.get("config_hash")) != next(iter(config_hashes)):
        raise ValueError("Temporal run_config/CSV config hash mismatch")
    with open(policy_path, encoding="utf-8") as handle:
        policy = json.load(handle)
    if policy.get("schema") != "active_speaker_policy_v1":
        raise ValueError("Unknown temporal policy schema")
    if policy.get("policy") != run_config.get("policy"):
        raise ValueError("Scoring policy differs from supplied calibrated policy")
    valid = {"pass", "reject", "manual"}
    unknown = sorted(set(scores["temporal_decision"].astype(str)) - valid)
    if unknown:
        raise ValueError(f"Unknown temporal decisions: {unknown}")
    keep_columns = [column for column in scores.columns if column != "model_versions"]
    merged = df.merge(scores[keep_columns], on="clip_id", how="left", validate="one_to_one")
    provenance = {
        "scores": os.path.abspath(scores_path),
        "scores_sha256": sha256_file(scores_path),
        "run_config": run_config_path,
        "run_config_sha256": sha256_file(run_config_path),
        "policy": os.path.abspath(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "config_hash": next(iter(config_hashes)),
        "gate_passed": bool(policy.get("gate_passed", False)),
        "publish_mode": policy.get("publish_mode", "manual_priority_only"),
    }
    return merged, provenance


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
    Chỉ dùng face quality và embed consistency. Sync/ASD score là metadata chẩn đoán,
    không được dùng liên tục để xếp hạng real.
    """
    det = df["det_ratio"].clip(0, 1)
    fa_norm = norm_pctile(df["mean_face_area"])

    parts = [(0.4, det), (0.3, fa_norm)]
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
                    help="output của 02_scoring/01_face_quality.py (mặc định local)")
    ap.add_argument("--emb", default="data/02_curate/measurements/embeddings_all.npy")
    ap.add_argument("--calibrate", action="store_true", help="chỉ in phân bố, không xuất")
    ap.add_argument("--cluster_dist", type=float, default=0.6, help="cosine distance threshold")
    ap.add_argument("--min_det_ratio", type=float, default=0.6, help="gate: tỉ lệ frame có mặt tối thiểu")
    ap.add_argument("--min_face_area", type=float, default=0.01, help="gate: mặt/khung tối thiểu")
    ap.add_argument("--min_consistency", type=float, default=0.3,
                    help="gate: embed_consistency tối thiểu (0 = tắt). Loại clip lẫn nhiều người.")
    ap.add_argument("--sync_floor", type=float, default=None,
                    help="DEPRECATED: SyncNet chỉ còn là metadata/chẩn đoán")
    ap.add_argument("--motion_floor", type=float, default=None,
                    help="DEPRECATED: full-frame motion không được dùng làm auto gate")
    ap.add_argument("--temporal_scores", default="",
                    help="asd_clip_scores.csv có coverage đúng 100%% scored input")
    ap.add_argument("--temporal_policy", default="",
                    help="active_speaker_policy_v1 JSON; chỉ auto gate khi gate_passed=true")
    ap.add_argument("--cap_per_speaker", type=int, default=30)
    ap.add_argument("--out", default="data/02_curate/manifests/all_clean.csv")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.sync_floor is not None or args.motion_floor is not None:
        raise ValueError(
            "--sync_floor/--motion_floor are disabled: keep SyncNet and full-frame "
            "motion as diagnostics, not automatic gates"
        )

    df, emb = load(args.scored_csv, args.emb)
    df, temporal_provenance = load_temporal_scores(
        df, args.temporal_scores, args.temporal_policy
    )

    if args.calibrate:
        calibrate(df, emb)
        return

    out_stem, out_ext = os.path.splitext(args.out)
    if out_ext.lower() != ".csv":
        raise ValueError("--out must end with .csv")
    rej_path = out_stem + "_rejects.csv"
    balance_path = out_stem + "_balance_dropped.csv"
    config_path = out_stem + "_config.json"
    outputs = (args.out, rej_path, balance_path, config_path)
    existing = [path for path in outputs if os.path.exists(path)]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output already exists; pass --overwrite only for an intentional rerun: "
            + ", ".join(existing)
        )

    # [B2] temporal gate runs before identity clustering/face gate. Rejected
    # source clips cannot influence agglomerative linkage among survivors.
    temporal_reject = pd.Series(False, index=df.index)
    if temporal_provenance:
        if temporal_provenance["gate_passed"]:
            temporal_reject = df["temporal_decision"].eq("reject")
            print(f"[B2] temporal gate: loại {int(temporal_reject.sum())} clip; "
                  f"manual={int(df['temporal_decision'].eq('manual').sum())}")
        else:
            print("[B2] temporal policy chưa đạt validation: chỉ dùng ưu tiên manual, "
                  "không auto-reject")
    survivor_mask = ~temporal_reject
    clustered = cluster_speakers(
        df.loc[survivor_mask].copy(), emb[survivor_mask.to_numpy()], args.cluster_dist
    )
    df["speaker_id"] = -1
    df.loc[survivor_mask, "speaker_id"] = clustered["speaker_id"].to_numpy()
    n_spk = df.loc[df.speaker_id != -1, "speaker_id"].nunique()
    print(f"[B2] {n_spk} speaker từ {df.loc[survivor_mask, 'source_video'].nunique()} video còn lại")

    # [B3] gate — chỉ loại rác rõ ràng
    before = len(df)
    df["quality"] = quality_score(df)
    face_gate = (df["has_embedding"]) & \
           (df["det_ratio"] >= args.min_det_ratio) & \
           (df["mean_face_area"] >= args.min_face_area)
    if "embed_consistency" in df.columns and args.min_consistency > 0:
        face_gate = face_gate & (df["embed_consistency"] >= args.min_consistency)
    gate = ~temporal_reject & face_gate
    df["gate_stage"] = "pass"
    df["gate_reason"] = ""
    df.loc[temporal_reject, "gate_stage"] = "temporal"
    if temporal_provenance:
        df.loc[temporal_reject, "gate_reason"] = df.loc[temporal_reject, "temporal_reason"]
    face_reject = ~temporal_reject & ~face_gate
    df.loc[face_reject, "gate_stage"] = "face_quality"
    df.loc[face_reject, "gate_reason"] = "face_quality"
    rejected = df[~gate].copy()
    gated = df[gate].copy()
    print(f"[B3] gate: giữ {len(gated)}/{before}, loại {len(rejected)} "
          f"(không mặt / mặt nhỏ / mặt thưa / lẫn người)")

    # [B4] cân bằng
    df = balance(gated, args.cap_per_speaker)
    gated_ids = set(gated["clip_id"])
    clean_ids = set(df["clip_id"])
    balance_dropped_ids = gated_ids - clean_ids
    balance_dropped = gated[gated["clip_id"].isin(balance_dropped_ids)].copy()

    rejected_ids = set(rejected["clip_id"])
    if rejected_ids & balance_dropped_ids or rejected_ids & clean_ids or balance_dropped_ids & clean_ids:
        raise RuntimeError("Curation output partitions overlap")
    input_ids = rejected_ids | gated_ids
    if rejected_ids | balance_dropped_ids | clean_ids != input_ids:
        raise RuntimeError("Curation output partitions do not cover the scored input")
    if len(rejected) + len(balance_dropped) + len(df) != before:
        raise RuntimeError("Curation output row counts do not match the scored input")
    print(f"[B4] sau cân bằng (cap {args.cap_per_speaker}/speaker): {len(df)} clip")
    if df.empty:
        raise RuntimeError("Curation produced zero clean clips; refusing to publish")
    dist = df["speaker_id"].value_counts()
    print(f"     clip/speaker: min={dist.min()} med={int(dist.median())} max={dist.max()}")

    # [B5] xuất
    df = df.sort_values(["speaker_id", "source_video", "start_time"]).reset_index(drop=True)
    rejected = rejected.sort_values("clip_id").reset_index(drop=True)
    balance_dropped = balance_dropped.sort_values("clip_id").reset_index(drop=True)
    run_config = {
        "inputs": {
            "scored_csv": os.path.abspath(args.scored_csv),
            "scored_csv_sha256": sha256_file(args.scored_csv),
            "embeddings": os.path.abspath(args.emb),
            "embeddings_sha256": sha256_file(args.emb),
            "temporal": temporal_provenance,
        },
        "parameters": {
            "cluster_dist": args.cluster_dist,
            "min_det_ratio": args.min_det_ratio,
            "min_face_area": args.min_face_area,
            "min_consistency": args.min_consistency,
            "sync_floor": args.sync_floor,
            "motion_floor": args.motion_floor,
            "temporal_auto_gate": bool(
                temporal_provenance and temporal_provenance["gate_passed"]
            ),
            "cap_per_speaker": args.cap_per_speaker,
        },
        "counts": {
            "scored": before,
            "gate_rejected": len(rejected),
            "temporal_rejected": int(temporal_reject.sum()),
            "face_quality_rejected": int(face_reject.sum()),
            "temporal_manual": int(df["temporal_decision"].eq("manual").sum())
                               if temporal_provenance else 0,
            "balance_dropped": len(balance_dropped),
            "clean": len(df),
        },
    }
    out_parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_parent, exist_ok=True)
    partials = {path: path + ".partial" for path in outputs}
    try:
        df.to_csv(partials[args.out], index=False)
        rejected.to_csv(partials[rej_path], index=False)
        balance_dropped.to_csv(partials[balance_path], index=False)
        with open(partials[config_path], "w", encoding="utf-8") as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2)
            f.write("\n")
        for final_path, partial_path in partials.items():
            os.replace(partial_path, final_path)
    finally:
        for partial_path in partials.values():
            if os.path.exists(partial_path):
                os.remove(partial_path)
    print(f"[B5] xuất {len(df)} clip sạch -> {args.out}")
    print(f"     reject log -> {rej_path} (nên xem mẫu để xác nhận đúng là rác)")
    print(f"     balance drop -> {balance_path}")
    print(f"     config + input hashes -> {config_path}")
    print(f"     speaker_id sẵn sàng cho split chống identity leakage ở bước sau.")


if __name__ == "__main__":
    main()
