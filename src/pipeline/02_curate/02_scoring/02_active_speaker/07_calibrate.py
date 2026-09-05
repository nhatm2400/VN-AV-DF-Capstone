"""Select and validate a temporal policy without touching the locked split."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path

import pandas as pd

from policy import TemporalPolicy, summarize_timeline


BAD_REASONS = {"static", "voiceover"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_timeline(path):
    rows = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.setdefault(str(row["clip_id"]), []).append(row)
    for values in rows.values():
        values.sort(key=lambda row: int(row["bin_index"]))
    return rows


def truth_is_bad(row):
    if row.get("decision") != "reject":
        return False
    if row.get("reason") in BAD_REASONS:
        return True
    raw = row.get("bad_intervals_json", "") or "[]"
    try:
        intervals = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid bad_intervals_json for {row.get('clip_id')}")
    return any(item.get("reason") in BAD_REASONS for item in intervals)


def metrics(labels, predictions):
    joined = labels.copy()
    joined["prediction"] = [predictions[str(cid)]["temporal_decision"]
                            for cid in joined.clip_id]
    joined["truth_bad"] = [truth_is_bad(row) for row in joined.to_dict("records")]
    joined["truth_clean"] = joined["decision"].eq("keep")
    ignored = joined[~joined.truth_bad & ~joined.truth_clean]
    bad = joined[joined.truth_bad]
    clean = joined[joined.truth_clean]
    recall = float((bad.prediction == "reject").mean()) if len(bad) else 0.0
    false_reject = float((clean.prediction == "reject").mean()) if len(clean) else 0.0
    tier_fpr = {}
    for tier, group in clean.groupby("tier"):
        tier_fpr[str(tier)] = float((group.prediction == "reject").mean())
    return {
        "clips": len(joined), "bad_clips": len(bad), "clean_clips": len(clean),
        "ignored_non_target_or_uncertain": len(ignored),
        "recall_bad": recall, "false_reject_clean": false_reject,
        "false_reject_by_tier": tier_fpr,
        "confusion_matrix_reject_vs_not": {
            "tp": int((bad.prediction == "reject").sum()),
            "fn": int((bad.prediction != "reject").sum()),
            "fp": int((clean.prediction == "reject").sum()),
            "tn": int((clean.prediction != "reject").sum()),
        },
        "auto_reject": int((joined.prediction == "reject").sum()),
        "manual": int((joined.prediction == "manual").sum()),
        "pass": int((joined.prediction == "pass").sum()),
    }


def predict(timeline, ids, policy):
    missing = sorted(set(map(str, ids)) - set(timeline))
    if missing:
        raise ValueError(f"Timeline missing {len(missing)} calibration clips")
    return {str(cid): summarize_timeline(timeline[str(cid)], policy) for cid in ids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_manifest", required=True)
    parser.add_argument("--consensus_labels", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--light_margins", default="0.3,0.5,0.7")
    parser.add_argument("--mouth_freeze_thresholds", default="0.7,1.0,1.3")
    args = parser.parse_args()
    if os.path.exists(args.out):
        raise FileExistsError(f"Immutable calibration report exists: {args.out}")

    manifest = pd.read_csv(args.calibration_manifest)
    labels = pd.read_csv(args.consensus_labels)
    required = {"clip_id", "tier", "source_video", "calibration_split"}
    if not required <= set(manifest.columns):
        raise ValueError(f"Calibration manifest missing {sorted(required - set(manifest.columns))}")
    if (len(manifest) != 450 or manifest["clip_id"].isna().any()
            or manifest["clip_id"].duplicated().any()):
        raise ValueError("Calibration manifest must contain exactly 450 unique clips")
    if labels.empty or labels["clip_id"].duplicated().any():
        raise ValueError("Consensus labels must be non-empty and unique by clip_id")
    if "rubric_version" not in labels or set(labels["rubric_version"].astype(str)) != {"v3"}:
        raise ValueError("Consensus labels must use rubric v3")
    labels = manifest[list(required)].merge(labels, on="clip_id", how="inner", validate="one_to_one")
    if len(labels) != len(manifest):
        raise ValueError("Consensus labels do not cover all 450 calibration clips")
    if set(labels.calibration_split) != {"tune", "locked_validation"}:
        raise ValueError("Expected tune and locked_validation splits")
    timeline = load_timeline(args.timeline)
    run_config_path = Path(args.timeline).resolve().parent / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Missing run_config.json beside timeline: {run_config_path}")
    run_config = json.loads(run_config_path.read_text("utf-8"))
    if not run_config.get("coverage_passed", False):
        raise ValueError("Timeline source run did not pass coverage")
    tune = labels[labels.calibration_split == "tune"].copy()
    locked = labels[labels.calibration_split == "locked_validation"].copy()
    if len(tune) != 300 or len(locked) != 150:
        raise ValueError("Calibration split must be exactly 300 tune / 150 locked validation")
    if set(tune.source_video.astype(str)) & set(locked.source_video.astype(str)):
        raise ValueError("source_video leakage between tune and locked validation")

    margins = [float(value) for value in args.light_margins.split(",")]
    freezes = [float(value) for value in args.mouth_freeze_thresholds.split(",")]
    candidates = []
    for margin, freeze in itertools.product(margins, freezes):
        policy = TemporalPolicy(light_margin=margin, mouth_freeze_threshold=freeze)
        result = metrics(tune, predict(timeline, tune.clip_id, policy))
        candidates.append((result["false_reject_clean"] <= 0.02,
                           result["auto_reject"], result["recall_bad"], policy, result))
    eligible = [item for item in candidates if item[0]]
    if not eligible:
        best = min(candidates, key=lambda item: (item[4]["false_reject_clean"], -item[1]))
    else:
        best = max(eligible, key=lambda item: (item[1], item[2], -item[4]["manual"]))
    _, _, _, policy, tune_metrics = best
    locked_metrics = metrics(locked, predict(timeline, locked.clip_id, policy))
    locked_tiers = set(locked["tier"].astype(str))
    gate_passed = (
        locked_metrics["bad_clips"] > 0
        and locked_metrics["clean_clips"] > 0
        and locked_metrics["recall_bad"] >= 0.95
        and locked_metrics["false_reject_clean"] <= 0.02
        and set(locked_metrics["false_reject_by_tier"]) == locked_tiers
        and all(value <= 0.03 for value in locked_metrics["false_reject_by_tier"].values())
    )
    report = {
        "schema": "active_speaker_policy_v1",
        "gate_passed": gate_passed,
        "publish_mode": "auto_gate" if gate_passed else "manual_priority_only",
        "policy": policy.to_dict(),
        "tune_metrics": tune_metrics,
        "locked_validation_metrics": locked_metrics,
        "input_sha256": {
            "calibration_manifest": sha256_file(args.calibration_manifest),
            "consensus_labels": sha256_file(args.consensus_labels),
            "timeline": sha256_file(args.timeline),
            "run_config": sha256_file(run_config_path),
        },
        "scoring_config_hash": run_config.get("config_hash"),
        "model_versions": run_config.get("model_versions", {}),
        "grid": {"light_margins": margins, "mouth_freeze_thresholds": freezes,
                 "candidate_count": len(candidates)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out + ".partial"
    Path(tmp).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
    os.replace(tmp, args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
