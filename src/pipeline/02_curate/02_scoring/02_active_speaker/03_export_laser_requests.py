"""Export exactly the ambiguous bins that require selective LASER inference."""

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Immutable LASER request output exists: {out_dir}")

    with open(args.manifest, encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    by_id = {str(row["clip_id"]): row for row in manifest_rows}
    if not by_id or len(by_id) != len(manifest_rows):
        raise ValueError("Manifest is empty or contains duplicate clip_id")
    opener = gzip.open if args.timeline.endswith(".gz") else open
    requests = []
    seen = set()
    with opener(args.timeline, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("laser_requested", False):
                continue
            clip_id, bin_index = str(row["clip_id"]), int(row["bin_index"])
            key = (clip_id, bin_index)
            if key in seen:
                raise ValueError(f"Duplicate requested timeline bin: {key}")
            if clip_id not in by_id:
                raise ValueError(f"Timeline clip is absent from manifest: {clip_id}")
            seen.add(key)
            requests.append({
                "clip_id": clip_id,
                "bin_index": bin_index,
                "start_ms": int(row["start_ms"]),
                "end_ms": int(row["end_ms"]),
                "file_path": by_id[clip_id].get("file_path", ""),
                "light_asd_score": row.get("light_asd_score"),
                "mouth_motion": row.get("mouth_motion"),
                "face_track_count": row.get("face_track_count", 0),
            })
    requests.sort(key=lambda row: (row["clip_id"], row["bin_index"]))
    out_dir.mkdir(parents=True, exist_ok=False)
    request_path = out_dir / "laser_requests.csv"
    fields = ["clip_id", "bin_index", "start_ms", "end_ms", "file_path",
              "light_asd_score", "mouth_motion", "face_track_count"]
    with open(request_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(requests)
    config = {
        "schema": "laser_request_v1",
        "source_timeline": os.path.abspath(args.timeline),
        "source_timeline_sha256": sha256_file(args.timeline),
        "manifest_sha256": sha256_file(args.manifest),
        "requested_bins": len(requests),
        "requested_clips": len({row["clip_id"] for row in requests}),
        "expected_score_schema": ["clip_id", "bin_index", "laser_score"],
        "score_semantics": "maximum active-speaker probability over every visible face in the bin",
    }
    (out_dir / "request_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
