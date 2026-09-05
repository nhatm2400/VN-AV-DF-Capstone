"""Enrich Light-ASD timeline with selective LASER scores, fail-closed.

LASER score rows are keyed by (clip_id, bin_index) and represent the maximum
active-speaker probability over all faces in that bin. This avoids assuming
that independently generated face-track IDs are interchangeable.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from policy import TemporalPolicy, summarize_timeline


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_laser(path):
    values = {}
    for row in read_jsonl(path):
        key = (str(row["clip_id"]), int(row["bin_index"]))
        if key in values:
            raise ValueError(f"Duplicate LASER key: {key}")
        score = float(row["laser_score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"LASER probability outside [0,1]: {key}={score}")
        values[key] = score
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_run", required=True)
    parser.add_argument("--laser_scores", required=True)
    parser.add_argument("--laser_metadata", required=True)
    parser.add_argument("--out_run", required=True)
    args = parser.parse_args()

    base = Path(args.base_run)
    out = Path(args.out_run)
    if out.exists():
        raise FileExistsError(f"Immutable output exists: {out}")
    required = [base / name for name in (
        "asd_clip_scores.csv", "asd_timeline.jsonl.gz", "failures.csv", "run_config.json"
    )]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Base run is incomplete: {missing}")
    config = json.loads(required[3].read_text("utf-8"))
    if not config.get("coverage_passed", False):
        raise ValueError("Base run did not pass coverage")
    policy = TemporalPolicy(**config["policy"])
    metadata = json.loads(Path(args.laser_metadata).read_text("utf-8"))
    required_meta = {"schema", "model_git_sha", "weights_sha256", "source_timeline_sha256"}
    if metadata.get("schema") != "laser_sidecar_v1" or not required_meta <= set(metadata):
        raise ValueError("Invalid laser_sidecar_v1 metadata")
    timeline_hash = sha256_file(required[1])
    if metadata["source_timeline_sha256"] != timeline_hash:
        raise ValueError("LASER sidecar was not produced from this exact base timeline")

    timeline = read_jsonl(required[1])
    requested = {
        (str(row["clip_id"]), int(row["bin_index"]))
        for row in timeline if row.get("laser_requested", False)
    }
    laser = load_laser(args.laser_scores)
    extra = set(laser) - requested
    if extra:
        raise ValueError(f"LASER sidecar contains {len(extra)} unrequested bins")
    missing_keys = requested - set(laser)

    grouped = {}
    for row in timeline:
        key = (str(row["clip_id"]), int(row["bin_index"]))
        if key in laser:
            row["laser_score"] = laser[key]
            light = row.get("light_asd_score")
            row["asd_disagreement"] = bool(light is not None and (
                (float(light) >= policy.light_active_threshold + policy.light_margin
                 and laser[key] <= policy.laser_active_threshold - policy.laser_margin)
                or (float(light) <= policy.light_active_threshold - policy.light_margin
                    and laser[key] >= policy.laser_active_threshold + policy.laser_margin)
            ))
        elif key in missing_keys:
            row["inference_failure"] = True
            row["laser_error"] = "missing_requested_score"
        grouped.setdefault(str(row["clip_id"]), []).append(row)

    base_scores = pd.read_csv(required[0])
    expected_ids = set(base_scores["clip_id"].astype(str))
    if set(grouped) != expected_ids or base_scores["clip_id"].duplicated().any():
        raise ValueError("Base timeline/summary coverage mismatch")
    model_versions = dict(config.get("model_versions", {}))
    laser_model_identity = {
        key: value for key, value in metadata.items() if key != "source_timeline_sha256"
    }
    model_versions["laser_model"] = laser_model_identity
    enriched_identity = {
        "schema": "active_speaker_laser_enriched_v1",
        "base_config_hash": config.get("config_hash"),
        "policy": policy.to_dict(),
        "model_versions": model_versions,
    }
    config_hash = canonical_hash(enriched_identity)
    versions_json = json.dumps(model_versions, sort_keys=True, separators=(",", ":"))
    summaries = []
    for clip_id in base_scores["clip_id"].astype(str):
        rows = sorted(grouped[clip_id], key=lambda row: int(row["bin_index"]))
        summary = summarize_timeline(rows, policy)
        summary.update(clip_id=clip_id, model_versions=versions_json, config_hash=config_hash)
        summaries.append(summary)

    failures = []
    with open(required[2], encoding="utf-8", newline="") as handle:
        failures.extend(csv.DictReader(handle))
    for clip_id in sorted({key[0] for key in missing_keys}):
        failures.append({
            "clip_id": clip_id, "file_path": "", "error_type": "LaserScoreMissing",
            "error_message": "At least one requested LASER bin has no score; routed to manual",
        })
    out.mkdir(parents=True, exist_ok=False)
    score_tmp = out / "asd_clip_scores.csv.partial"
    timeline_tmp = out / "asd_timeline.jsonl.gz.partial"
    failures_tmp = out / "failures.csv.partial"
    config_tmp = out / "run_config.json.partial"
    pd.DataFrame(summaries).to_csv(score_tmp, index=False)
    with gzip.open(timeline_tmp, "wt", encoding="utf-8") as handle:
        for row in timeline:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = ["clip_id", "file_path", "error_type", "error_message"]
    with open(failures_tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)
    out_config = {
        **enriched_identity,
        "run_id": out.parents[1].name if out.parent.name == "shards" else out.name,
        "batch_start": config.get("batch_start", 0),
        "batch_end": config.get("batch_end", config.get("manifest_rows")),
        "shard_id": out.name if out.parent.name == "shards" else "",
        "config_hash": config_hash,
        "manifest_sha256": config.get("manifest_sha256"),
        "manifest_rows": config.get("manifest_rows"),
        "score_rows": len(summaries),
        "output_rows": len(summaries),
        "timeline_rows": len(timeline),
        "laser_requested_bins": len(requested),
        "laser_scored_bins": len(laser),
        "laser_missing_bins": len(missing_keys),
        "failure_rows": len(failures),
        "coverage_passed": True,
        "preprocessing": config.get("preprocessing", {}),
        "policy": policy.to_dict(),
        "model_versions": model_versions,
        "laser_provenance": {
            "source_timeline_sha256": timeline_hash,
            "sidecar_sha256": sha256_file(args.laser_scores),
            "metadata_sha256": sha256_file(args.laser_metadata),
        },
    }
    config_tmp.write_text(json.dumps(out_config, ensure_ascii=False, indent=2) + "\n", "utf-8")
    for temporary, final in (
        (score_tmp, out / "asd_clip_scores.csv"),
        (timeline_tmp, out / "asd_timeline.jsonl.gz"),
        (failures_tmp, out / "failures.csv"),
        (config_tmp, out / "run_config.json"),
    ):
        os.replace(temporary, final)
    print(json.dumps(out_config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
