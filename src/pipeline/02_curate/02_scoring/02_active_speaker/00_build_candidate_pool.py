"""Build a deterministic source-disjoint pool for preliminary ASD scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"clip_id", "source_video", "tier", "file_path"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, *parts: str) -> str:
    value = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_candidate_pool(frame: pd.DataFrame, per_tier: int, seed: int) -> pd.DataFrame:
    if per_tier <= 0:
        raise ValueError("per_tier must be positive")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if frame.empty or missing:
        raise ValueError(f"Invalid manifest; missing columns: {sorted(missing)}")

    data = frame.copy()
    for column in REQUIRED_COLUMNS:
        data[column] = data[column].astype(str).str.strip()
        if (data[column] == "").any():
            raise ValueError(f"Manifest has empty {column}")
    if data["clip_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate clip_id")

    tiers = sorted(data["tier"].unique())
    if tiers != ["tier1", "tier2", "tier3"]:
        raise ValueError(f"Expected tier1/tier2/tier3, found {tiers}")

    selected = []
    used_sources = set()
    # Select the scarcest tier first so a cross-tier source collision cannot
    # accidentally consume a source needed by that tier.
    tier_order = sorted(
        tiers,
        key=lambda tier: (data.loc[data["tier"] == tier, "source_video"].nunique(), tier),
    )
    for tier in tier_order:
        tier_rows = data[data["tier"] == tier]
        candidates = []
        for source_video, source_rows in tier_rows.groupby("source_video", sort=False):
            if source_video in used_sources:
                continue
            rows = source_rows.copy()
            rows["_clip_rank"] = [
                stable_rank(seed, tier, source_video, clip_id)
                for clip_id in rows["clip_id"]
            ]
            chosen = rows.sort_values(["_clip_rank", "clip_id"]).iloc[0].copy()
            chosen["_source_rank"] = stable_rank(seed, tier, source_video)
            candidates.append(chosen)
        if len(candidates) < per_tier:
            raise ValueError(
                f"{tier} has only {len(candidates)} unused sources; need {per_tier}"
            )
        part = pd.DataFrame(candidates).sort_values(
            ["_source_rank", "source_video", "clip_id"]
        ).head(per_tier).copy()
        part["candidate_rank_in_tier"] = range(1, per_tier + 1)
        selected.append(part)
        used_sources.update(part["source_video"])

    output = pd.concat(selected, ignore_index=True)
    output = output.sort_values(["tier", "candidate_rank_in_tier"]).reset_index(drop=True)
    if len(output) != per_tier * 3:
        raise RuntimeError("Candidate pool count contract failed")
    if output["clip_id"].duplicated().any() or output["source_video"].duplicated().any():
        raise RuntimeError("Candidate pool is not source-disjoint")
    return output.drop(columns=["_clip_rank", "_source_rank"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="data/01_collect/cut_clips/all_manifest.csv"
    )
    parser.add_argument(
        "--out", default="data/02_curate/calibration/active_speaker_candidate_pool_750_v1.csv"
    )
    parser.add_argument("--per_tier", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.out)
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    if output_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"Immutable candidate output exists: {output_path} or {summary_path}"
        )

    manifest = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    output = select_candidate_pool(manifest, args.per_tier, args.seed)
    output["candidate_pool_version"] = "active_speaker_candidate_pool_v1"
    output["selection_seed"] = args.seed

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = Path(str(output_path) + ".partial")
    summary_tmp = Path(str(summary_path) + ".partial")
    output.to_csv(output_tmp, index=False)
    summary = {
        "schema": "active_speaker_candidate_pool_v1",
        "clips": len(output),
        "clips_per_tier": args.per_tier,
        "unique_sources": int(output["source_video"].nunique()),
        "source_disjoint": not output["source_video"].duplicated().any(),
        "by_tier": output["tier"].value_counts().sort_index().to_dict(),
        "seed": args.seed,
        "selection": "sha256_rank_one_clip_per_source_v1",
        "input_manifest_sha256": sha256_file(args.manifest),
        "candidate_manifest_sha256": sha256_file(output_tmp),
    }
    summary_tmp.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(output_tmp, output_path)
    os.replace(summary_tmp, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
