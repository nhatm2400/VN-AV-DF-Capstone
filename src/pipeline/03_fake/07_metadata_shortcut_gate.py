"""Group-disjoint baseline using only observable media/container metadata."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = (
    "log_file_size",
    "format_duration_s",
    "format_bit_rate",
    "video_duration_s",
    "audio_duration_s",
    "abs_av_duration_gap_s",
    "container_tail_s",
    "video_bit_rate",
    "audio_bit_rate",
    "video_frames",
    "video_fps",
    "width",
    "height",
    "bytes_per_frame",
    "video_start_s",
    "audio_start_s",
    "audio_sample_rate",
    "audio_channels",
)


def _float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _duration(stream):
    if stream.get("duration_ts") is not None and stream.get("time_base"):
        numerator, denominator = stream["time_base"].split("/")
        return int(stream["duration_ts"]) * int(numerator) / int(denominator)
    return _float(stream.get("duration"))


def _fps(stream):
    value = stream.get("r_frame_rate") or stream.get("avg_frame_rate") or "0/1"
    numerator, denominator = value.split("/")
    return int(numerator) / int(denominator)


def probe_features(path):
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,codec_name,pix_fmt,width,height,r_frame_rate,"
         "avg_frame_rate,start_time,duration,duration_ts,time_base,bit_rate,"
         "nb_frames,nb_read_frames,sample_rate,channels:"
         "format=format_name,start_time,duration,size,bit_rate",
         "-of", "json", path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise ValueError(f"ffprobe failed: {path}")
    data = json.loads(proc.stdout)
    video = next(s for s in data["streams"] if s.get("codec_type") == "video")
    audio = next(s for s in data["streams"] if s.get("codec_type") == "audio")
    fmt = data.get("format", {})
    file_size = int(fmt.get("size", os.path.getsize(path)))
    video_duration = _duration(video)
    audio_duration = _duration(audio)
    format_duration = _float(fmt.get("duration"))
    frames = int(video.get("nb_read_frames") or video.get("nb_frames"))
    features = {
        "log_file_size": math.log1p(file_size),
        "format_duration_s": format_duration,
        "format_bit_rate": _float(fmt.get("bit_rate")),
        "video_duration_s": video_duration,
        "audio_duration_s": audio_duration,
        "abs_av_duration_gap_s": abs(video_duration - audio_duration),
        "container_tail_s": format_duration - max(video_duration, audio_duration),
        "video_bit_rate": _float(video.get("bit_rate")),
        "audio_bit_rate": _float(audio.get("bit_rate")),
        "video_frames": frames,
        "video_fps": _fps(video),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "bytes_per_frame": file_size / frames,
        "video_start_s": _float(video.get("start_time", 0) or 0),
        "audio_start_s": _float(audio.get("start_time", 0) or 0),
        "audio_sample_rate": int(audio.get("sample_rate", 0) or 0),
        "audio_channels": int(audio.get("channels", 0) or 0),
    }
    signature = {
        "video_codec": video.get("codec_name", ""),
        "pixel_format": video.get("pix_fmt", ""),
        "audio_codec": audio.get("codec_name", ""),
        "audio_sample_rate": int(audio.get("sample_rate", 0) or 0),
        "audio_channels": int(audio.get("channels", 0) or 0),
    }
    return features, signature


def _method_auc(labels, scores, methods, method):
    keep = np.array([(m == "real" or m == method) for m in methods])
    return float(roc_auc_score(labels[keep], scores[keep]))


def _bootstrap_group_auc(labels, scores, groups, seed=42, repeats=1000):
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        if len(np.unique(labels[indices])) == 2:
            values.append(roc_auc_score(labels[indices], scores[indices]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def evaluate_feature_rows(rows, threshold=0.65, seed=42):
    if not rows:
        raise ValueError("Metadata rows rỗng")
    labels = np.array([int(row["label"]) for row in rows])
    groups = np.array([str(row["source_clip"]) for row in rows])
    methods = np.array([str(row["method"]) for row in rows])
    unique_groups = np.unique(groups)
    if len(unique_groups) < 5 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Cần ít nhất 5 source group và đủ hai nhãn")
    for group in unique_groups:
        group_rows = [row for row in rows if str(row["source_clip"]) == group]
        if len(group_rows) != 5 or sum(int(row["label"]) for row in group_rows) != 4:
            raise ValueError(f"Source {group} không có đúng 1 real + 4 fake")

    matrix = np.array([
        [_float(row.get(feature)) for feature in FEATURE_NAMES] for row in rows
    ])
    models = {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000,
                               random_state=seed),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=500, max_depth=3, min_samples_leaf=2,
                class_weight="balanced", random_state=seed, n_jobs=-1,
            ),
        ),
    }
    splitter = GroupKFold(n_splits=5)
    results = {}
    prediction_columns = {}
    for name, model in models.items():
        scores = np.zeros(len(rows), dtype=np.float64)
        for train_idx, test_idx in splitter.split(matrix, labels, groups):
            model.fit(matrix[train_idx], labels[train_idx])
            scores[test_idx] = model.predict_proba(matrix[test_idx])[:, 1]
        auc = float(roc_auc_score(labels, scores))
        method_auc = {
            method: _method_auc(labels, scores, methods, method)
            for method in sorted(set(methods) - {"real"})
        }
        results[name] = {
            "auc": auc,
            "auc_group_bootstrap_95ci": _bootstrap_group_auc(
                labels, scores, groups, seed=seed
            ),
            "method_auc": method_auc,
        }
        prediction_columns[name] = scores
    max_auc = max(result["auc"] for result in results.values())
    return {
        "n_rows": len(rows),
        "n_sources": len(unique_groups),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "cv": "GroupKFold(n_splits=5), group=source_clip",
        "feature_names": list(FEATURE_NAMES),
        "threshold_max_auc": float(threshold),
        "models": results,
        "max_auc": max_auc,
        "passed": bool(max_auc <= threshold),
    }, prediction_columns


def load_rows(real_csv, fake_csv):
    with open(real_csv, newline="", encoding="utf-8") as handle:
        real_rows = list(csv.DictReader(handle))
    with open(fake_csv, newline="", encoding="utf-8") as handle:
        fake_rows = list(csv.DictReader(handle))
    rows = []
    signatures = set()
    for label, side in ((0, real_rows), (1, fake_rows)):
        for row in side:
            source = (row.get("orig_clip_id") if label == 0
                      else row.get("source_clip"))
            method = "real" if label == 0 else row.get("method", "")
            features, signature = probe_features(row["file_path"])
            signatures.add(tuple(sorted(signature.items())))
            rows.append({
                "clip_id": row["clip_id"],
                "source_clip": source,
                "method": method,
                "label": label,
                **features,
            })
    return rows, [dict(signature) for signature in sorted(signatures)]


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    partial = path + ".part"
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(partial, path)


def write_predictions_atomic(path, rows, predictions):
    fields = ["clip_id", "source_clip", "method", "label", *FEATURE_NAMES,
              *[f"score_{name}" for name in predictions]]
    partial = path + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(partial, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            output = dict(row)
            for name, scores in predictions.items():
                output[f"score_{name}"] = f"{scores[index]:.9f}"
            writer.writerow(output)
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_csv", required=True)
    parser.add_argument("--fake_csv", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_predictions", required=True)
    parser.add_argument("--max_auc", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, signatures = load_rows(args.real_csv, args.fake_csv)
    report, predictions = evaluate_feature_rows(rows, args.max_auc, args.seed)
    report["codec_signatures"] = signatures
    write_json_atomic(args.out_json, report)
    write_predictions_atomic(args.out_predictions, rows, predictions)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
