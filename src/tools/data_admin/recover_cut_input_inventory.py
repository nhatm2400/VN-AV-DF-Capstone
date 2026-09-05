"""Phục hồi inventory input Stage 04 từ cut logs lịch sử.

Đây KHÔNG phải quality-gate manifest thay thế. Nó chỉ phục hồi chính xác tập
``source_video`` đã đi vào một cut run cũ để có thể đối chiếu với raw media trên
Kaggle khi quality CSV gốc bị thất lạc.
"""

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def recover_rows(accepted_rows, rejected_rows, extension=".mp4"):
    if not re.fullmatch(r"\.[A-Za-z0-9]+", extension):
        raise ValueError(f"Invalid extension: {extension}")
    accepted = Counter(
        str(row.get("source_video", "")).strip() for row in accepted_rows
    )
    rejected = Counter(
        str(row.get("video", "")).strip() for row in rejected_rows
    )
    accepted.pop("", None)
    rejected.pop("", None)
    source_ids = sorted(set(accepted) | set(rejected))
    if not source_ids:
        raise ValueError("Cut logs do not contain any source video")
    unsafe = [
        source_id for source_id in source_ids
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", source_id)
    ]
    if unsafe:
        raise ValueError(f"Unsafe source_video values: {unsafe[:3]}")

    rows = []
    for source_id in source_ids:
        seen = []
        if accepted[source_id]:
            seen.append("accepted")
        if rejected[source_id]:
            seen.append("rejected")
        rows.append({
            "filename": source_id + extension,
            "source_video": source_id,
            "accepted_clip_count": accepted[source_id],
            "rejected_row_count": rejected[source_id],
            "recovered_from": "+".join(seen),
        })
    return rows


def write_csv_atomic(path, rows):
    partial = str(path) + ".partial"
    with open(partial, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def write_json_atomic(path, payload):
    partial = str(path) + ".partial"
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--rejected", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected_count", type=int, required=True)
    parser.add_argument("--extension", default=".mp4")
    args = parser.parse_args()

    accepted_path = Path(args.accepted)
    rejected_path = Path(args.rejected)
    out_path = Path(args.out)
    summary_path = out_path.with_suffix(".summary.json")
    existing = [path for path in (out_path, summary_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite recovered inventory: "
            + ", ".join(map(str, existing))
        )

    accepted_rows = read_csv(accepted_path)
    rejected_rows = read_csv(rejected_path)
    rows = recover_rows(accepted_rows, rejected_rows, args.extension)
    if len(rows) != args.expected_count:
        raise ValueError(
            f"Recovered source count mismatch: {len(rows)} != "
            f"{args.expected_count}"
        )
    if len({row["filename"] for row in rows}) != len(rows):
        raise ValueError("Recovered filename values are not unique")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "recovered_cut_input_inventory_v1",
        "warning": (
            "Not a replacement for the Stage 03 quality-gate manifest. "
            "Must be matched 1:1 to the Kaggle raw media before use."
        ),
        "accepted_log": accepted_path.as_posix(),
        "accepted_log_sha256": sha256_file(accepted_path),
        "rejected_log": rejected_path.as_posix(),
        "rejected_log_sha256": sha256_file(rejected_path),
        "source_count": len(rows),
        "seen_in_accepted": sum(
            row["accepted_clip_count"] > 0 for row in rows
        ),
        "seen_in_rejected": sum(
            row["rejected_row_count"] > 0 for row in rows
        ),
    }
    write_csv_atomic(out_path, rows)
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Recovered inventory: {out_path}")


if __name__ == "__main__":
    main()
