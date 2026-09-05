"""Fail-closed merge for immutable active-speaker scoring shards."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/01_collect/cut_clips/all_manifest.csv")
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    outputs = [run_dir / name for name in (
        "asd_clip_scores.csv", "asd_timeline.jsonl.gz", "failures.csv", "run_config.json"
    )]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("Immutable merged output already exists: " + ", ".join(existing))

    shard_dirs = sorted(path.parent for path in (run_dir / "shards").glob("*/run_config.json"))
    if not shard_dirs:
        raise ValueError("No shards found under run_dir/shards")
    configs = [json.loads((path / "run_config.json").read_text("utf-8")) for path in shard_dirs]
    if not all(item.get("coverage_passed", False) for item in configs):
        raise ValueError("At least one shard did not pass its local coverage gate")
    run_ids = {item.get("run_id") for item in configs}
    if len(run_ids) != 1 or next(iter(run_ids)) != run_dir.name:
        raise ValueError(f"Shard run_id mismatch for directory {run_dir.name}: {run_ids}")
    config_hashes = {item.get("config_hash") for item in configs}
    manifest_hashes = {item.get("manifest_sha256") for item in configs}
    if None in config_hashes or len(config_hashes) != 1:
        raise ValueError(f"Shard config_hash mismatch: {sorted(map(str, config_hashes))}")
    if None in manifest_hashes or len(manifest_hashes) != 1:
        raise ValueError("Shard manifest hashes differ")
    if sha256_file(args.manifest) != next(iter(manifest_hashes)):
        raise ValueError("Supplied manifest hash differs from shard provenance")

    score_frames, failures, timeline_lines, timeline_ids = [], [], [], set()
    ranges = []
    for shard_dir, config in zip(shard_dirs, configs):
        for name in ("asd_clip_scores.csv", "asd_timeline.jsonl.gz", "failures.csv"):
            if not (shard_dir / name).is_file():
                raise FileNotFoundError(shard_dir / name)
        frame = pd.read_csv(shard_dir / "asd_clip_scores.csv")
        if len(frame) != int(config.get("output_rows", -1)):
            raise ValueError(f"Score/config row mismatch in {shard_dir}")
        if len(frame) != int(config["batch_end"]) - int(config["batch_start"]):
            raise ValueError(f"Shard range/score count mismatch in {shard_dir}")
        if set(frame["config_hash"].astype(str)) != config_hashes:
            raise ValueError(f"CSV config_hash mismatch in {shard_dir}")
        score_frames.append(frame)
        with gzip.open(shard_dir / "asd_timeline.jsonl.gz", "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    timeline_ids.add(str(json.loads(line)["clip_id"]))
                    timeline_lines.append(line)
        with open(shard_dir / "failures.csv", encoding="utf-8", newline="") as handle:
            failures.extend(csv.DictReader(handle))
        ranges.append((int(config["batch_start"]), int(config["batch_end"]), shard_dir.name))

    ranges.sort()
    cursor = 0
    for start, end, name in ranges:
        if start != cursor or end <= start:
            raise ValueError(f"Shard ranges have a gap/overlap before {name}: expected {cursor}, got {start}")
        cursor = end

    manifest = pd.read_csv(args.manifest, usecols=["clip_id"])
    if cursor != len(manifest):
        raise ValueError(f"Shard range coverage {cursor}/{len(manifest)} is incomplete")
    scores = pd.concat(score_frames, ignore_index=True)
    expected_ids = manifest["clip_id"].astype(str).tolist()
    actual_ids = scores["clip_id"].astype(str).tolist()
    if scores["clip_id"].isna().any() or scores["clip_id"].duplicated().any():
        raise ValueError("Merged scores contain empty/duplicate clip_id")
    if set(actual_ids) != set(expected_ids) or len(actual_ids) != len(expected_ids):
        raise ValueError("Merged scores do not provide exact manifest coverage")
    if timeline_ids != set(expected_ids):
        raise ValueError(
            f"Timeline coverage mismatch: missing={len(set(expected_ids)-timeline_ids)}, "
            f"extra={len(timeline_ids-set(expected_ids))}"
        )
    failure_ids = {str(row["clip_id"]) for row in failures}
    if not failure_ids <= set(expected_ids):
        raise ValueError("Failures contain clip IDs outside the manifest")
    order = {clip_id: index for index, clip_id in enumerate(expected_ids)}
    scores["_manifest_order"] = scores["clip_id"].astype(str).map(order)
    scores = scores.sort_values("_manifest_order").drop(columns="_manifest_order")

    run_dir.mkdir(parents=True, exist_ok=True)
    score_tmp = Path(str(outputs[0]) + ".partial")
    scores.to_csv(score_tmp, index=False)
    timeline_tmp = Path(str(outputs[1]) + ".partial")
    with gzip.open(timeline_tmp, "wt", encoding="utf-8") as handle:
        handle.writelines(timeline_lines)
    failures_tmp = Path(str(outputs[2]) + ".partial")
    fields = ["clip_id", "file_path", "error_type", "error_message"]
    with open(failures_tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)
    merged_config = {
        "schema": "active_speaker_merged_run_v1",
        "config_hash": next(iter(config_hashes)),
        "manifest_sha256": next(iter(manifest_hashes)),
        "manifest_rows": len(manifest),
        "score_rows": len(scores),
        "timeline_rows": len(timeline_lines),
        "failure_rows": len(failures),
        "shards": [name for _, _, name in ranges],
        "coverage_passed": True,
        "model_versions": configs[0].get("model_versions", {}),
        "preprocessing": configs[0].get("preprocessing", {}),
        "policy": configs[0].get("policy", {}),
    }
    config_tmp = Path(str(outputs[3]) + ".partial")
    config_tmp.write_text(json.dumps(merged_config, ensure_ascii=False, indent=2) + "\n", "utf-8")
    for temporary, final in zip(
            (score_tmp, timeline_tmp, failures_tmp, config_tmp), outputs):
        os.replace(temporary, final)
    print(json.dumps(merged_config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
