"""Build a 450-clip, source-disjoint active-speaker calibration manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
from pathlib import Path

import pandas as pd


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_timeline(path):
    facts = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = str(row["clip_id"])
            item = facts.setdefault(cid, {"multi": False, "requested_laser": False})
            item["multi"] |= int(row.get("face_track_count", 0)) > 1
            item["requested_laser"] |= bool(row.get("laser_requested", False))
    return facts


def risk_stratum(row, fact):
    static = float(row.get("static_speech_ratio", 0) or 0)
    unexplained = float(row.get("unexplained_speech_ratio", 0) or 0)
    if fact.get("multi"):
        return "multiple_faces"
    if 0.10 <= unexplained < 0.60:
        return "mixed"
    if static >= 0.20:
        return "static"
    if unexplained >= 0.20:
        return "voiceover"
    return "clean_candidate"


def choose_unique_sources(frame, n, seed):
    """Round-robin strata while selecting at most one clip per source video."""
    groups = {}
    for stratum, part in frame.groupby("risk_stratum"):
        rows = part.to_dict("records")
        random.Random(f"{seed}:{stratum}").shuffle(rows)
        groups[stratum] = rows
    picked, used_sources = [], set()
    strata = sorted(groups)
    while len(picked) < n and any(groups.values()):
        progress = False
        for stratum in strata:
            while groups[stratum]:
                row = groups[stratum].pop()
                source = str(row["source_video"])
                if source in used_sources:
                    continue
                picked.append(row)
                used_sources.add(source)
                progress = True
                break
            if len(picked) == n:
                break
        if not progress:
            break
    if len(picked) != n:
        raise ValueError(f"Need {n} unique sources but selected only {len(picked)}")
    return pd.DataFrame(picked)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/01_collect/cut_clips/all_manifest.csv")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--out", default="data/02_curate/calibration/active_speaker_450_v3.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if os.path.exists(args.out):
        raise FileExistsError(f"Immutable calibration manifest exists: {args.out}")
    manifest = pd.read_csv(args.manifest)
    scores = pd.read_csv(args.scores)
    for name, frame in (("manifest", manifest), ("scores", scores)):
        if frame.empty or "clip_id" not in frame or frame["clip_id"].duplicated().any():
            raise ValueError(f"Invalid {name}")
    required = {"clip_id", "tier", "source_video"}
    if not required <= set(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(manifest.columns))}")
    merged = manifest.merge(scores, on="clip_id", how="inner", validate="one_to_one",
                            suffixes=("", "_asd"))
    if len(merged) != len(manifest):
        raise ValueError("Scores do not cover the complete manifest")
    timeline = load_timeline(args.timeline)
    merged["risk_stratum"] = [
        risk_stratum(row, timeline.get(str(row["clip_id"]), {}))
        for row in merged.to_dict("records")
    ]
    tiers = sorted(merged["tier"].astype(str).unique())
    if len(tiers) != 3:
        raise ValueError(f"Expected exactly three tiers, found {tiers}")

    selected = []
    globally_used_sources = set()
    for tier in tiers:
        tier_rows = merged[merged["tier"].astype(str) == tier]
        tier_rows = tier_rows[~tier_rows["source_video"].astype(str).isin(globally_used_sources)]
        chosen = choose_unique_sources(tier_rows, 150, args.seed)
        # choose_unique_sources returns a round-robin risk order. One clip/source
        # means this exact split is source-disjoint by construction.
        chosen["calibration_split"] = ["tune"] * 100 + ["locked_validation"] * 50
        selected.append(chosen)
        globally_used_sources.update(chosen["source_video"].astype(str))
    output = pd.concat(selected, ignore_index=True)
    if len(output) != 450 or output["source_video"].duplicated().any():
        raise RuntimeError("Calibration selection violated count/source-disjoint contract")
    tune_sources = set(output.loc[output.calibration_split == "tune", "source_video"])
    val_sources = set(output.loc[output.calibration_split == "locked_validation", "source_video"])
    if tune_sources & val_sources:
        raise RuntimeError("Tune/validation source leakage")

    output["assignment_role"] = "calibration"
    output["rubric_version"] = "v3"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out + ".partial"
    output.to_csv(tmp, index=False)
    os.replace(tmp, args.out)
    summary = {
        "schema": "active_speaker_calibration_manifest_v1",
        "clips": len(output),
        "tune": int((output.calibration_split == "tune").sum()),
        "locked_validation": int((output.calibration_split == "locked_validation").sum()),
        "unique_sources": int(output.source_video.nunique()),
        "by_tier": output.tier.astype(str).value_counts().sort_index().to_dict(),
        "by_risk_stratum": output.risk_stratum.value_counts().sort_index().to_dict(),
        "seed": args.seed,
        "input_sha256": {
            "manifest": sha256_file(args.manifest),
            "scores": sha256_file(args.scores),
            "timeline": sha256_file(args.timeline),
        },
    }
    summary_path = str(Path(args.out).with_suffix("")) + "_summary.json"
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
