"""Tạo master composition không mang temporal V1; không chứng nhận timing media."""

import argparse
import csv
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pipeline.timeline_contract import validate_timeline_contract

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

GENERATOR_VERSION = "temporal_v2_structured_timeline_v1"
TEMPORAL_PARAM_KEYS = (
    f"generator={GENERATOR_VERSION}",
)
NON_TEMPORAL_METHODS = ("frame_reverse", "pitch_flatten", "anonymization")
METHOD_ORDER = ("temporal_desync", *NON_TEMPORAL_METHODS)
DEFAULT_LEGACY = "data/03_fake/labels.csv"
DEFAULT_TEMPORAL = "data/03_fake/manifests/v2/temporal_desync.csv"
DEFAULT_OUTPUT = "data/03_fake/manifests/v2/fake_all.csv"


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def compose_rows(legacy_rows, temporal_rows, check_files=True):
    """Giữ ba method V1 không-temporal và thay toàn bộ temporal bằng V2."""
    if not temporal_rows:
        raise ValueError("Manifest temporal V2 rỗng")

    temporal_by_source = {}
    all_ids = set()
    for row in temporal_rows:
        source = row.get("source_clip", "")
        clip_id = row.get("clip_id", "")
        if (row.get("method") != "temporal_desync"
                or any(key not in row.get("param", "")
                       for key in TEMPORAL_PARAM_KEYS)):
            raise ValueError(f"Temporal row không thuộc generator V2: {clip_id}")
        validate_timeline_contract(row, "temporal_desync")
        if not source or source in temporal_by_source:
            raise ValueError(f"Mỗi source phải có đúng một temporal V2: {source!r}")
        if not clip_id or clip_id in all_ids:
            raise ValueError(f"clip_id temporal rỗng hoặc trùng: {clip_id!r}")
        temporal_by_source[source] = row
        all_ids.add(clip_id)

    legacy_by_key = {}
    for row in legacy_rows:
        method = row.get("method", "")
        if method == "temporal_desync":
            continue
        if method not in NON_TEMPORAL_METHODS:
            raise ValueError(f"Method không thuộc contract V2: {method!r}")
        source = row.get("source_clip", "")
        if source not in temporal_by_source:
            continue
        key = (source, method)
        if key in legacy_by_key:
            raise ValueError(f"Trùng method/source trong legacy manifest: {key}")
        legacy_by_key[key] = row

    output = []
    for source, temporal in temporal_by_source.items():
        group = {"temporal_desync": temporal}
        for method in NON_TEMPORAL_METHODS:
            key = (source, method)
            if key not in legacy_by_key:
                raise ValueError(f"Thiếu {method} cho source {source}")
            group[method] = legacy_by_key[key]

        reference_meta = {
            key: temporal.get(key, "")
            for key in ("source_video", "speaker_id", "tier")
        }
        for method in METHOD_ORDER:
            row = group[method]
            clip_id = row.get("clip_id", "")
            if not clip_id or (method != "temporal_desync" and clip_id in all_ids):
                raise ValueError(f"clip_id rỗng hoặc trùng: {clip_id!r}")
            all_ids.add(clip_id)
            if str(row.get("label", "")) not in ("1", "1.0"):
                raise ValueError(f"Fake row phải có label=1: {clip_id}")
            validate_timeline_contract(row, method)
            for key, expected in reference_meta.items():
                actual = row.get(key, "")
                if not expected or actual != expected:
                    raise ValueError(
                        f"Metadata {key} không khớp cho source {source}: "
                        f"{expected!r} != {actual!r}"
                    )
            if check_files and not os.path.isfile(row.get("file_path", "")):
                raise ValueError(f"Không tìm thấy media: {row.get('file_path', '')}")
            output.append(row)

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy_labels", default=DEFAULT_LEGACY)
    parser.add_argument("--temporal_v2", default=DEFAULT_TEMPORAL)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--check_files", action="store_true", default=True)
    parser.add_argument("--no_check_files", dest="check_files", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_abs = os.path.normcase(os.path.abspath(args.out))
    protected_inputs = {
        os.path.normcase(os.path.abspath(DEFAULT_LEGACY)),
        os.path.normcase(os.path.abspath(args.legacy_labels)),
        os.path.normcase(os.path.abspath(args.temporal_v2)),
    }
    if output_abs in protected_inputs:
        raise ValueError(f"Từ chối ghi đè manifest đầu vào/V1: {args.out}")
    if os.path.exists(args.out) and not args.overwrite:
        raise FileExistsError(
            f"Output đã tồn tại: {args.out}. Dùng path run mới hoặc --overwrite có chủ đích."
        )

    legacy_fields, legacy_rows = read_manifest(args.legacy_labels)
    temporal_fields, temporal_rows = read_manifest(args.temporal_v2)
    rows = compose_rows(legacy_rows, temporal_rows, args.check_files)

    fields = list(legacy_fields)
    for field in temporal_fields:
        if field not in fields:
            fields.append(field)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    partial = args.out + ".part"
    try:
        with open(partial, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(partial, args.out)
    finally:
        if os.path.exists(partial):
            os.remove(partial)

    counts = {method: 0 for method in METHOD_ORDER}
    for row in rows:
        counts[row["method"]] += 1
    print(f"Xong. {len(rows)} fake từ {len(rows) // 4} source -> {args.out}")
    print("  " + " | ".join(f"{method}={counts[method]}" for method in METHOD_ORDER))


if __name__ == "__main__":
    main()
