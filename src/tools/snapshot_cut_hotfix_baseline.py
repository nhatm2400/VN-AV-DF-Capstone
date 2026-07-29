"""Khóa provenance của dataset cũ trước khi hotfix Stage 04 Cut Clips.

Script chỉ đọc artifact hiện tại. Nó không di chuyển/xóa media và không copy
22+ GiB MP4. Output gồm:

- bản sao các CSV cut-log đang bị .gitignore;
- SHA-256 cho manifest/measurement/manual/config nhỏ;
- inventory path + size của media cũ (không hash media để tránh job dài).
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "archive" / "cut_clips_v1_decode_bug"

HASH_INPUTS = (
    "data/01_collect/tier1_quality_gate_passed.csv",
    "data/01_collect/tier2_quality_gate_passed.csv",
    "data/01_collect/youtube_tier1_urls.csv",
    "data/01_collect/youtube_tier2_urls.csv",
    "data/02_curate",
    "src/pipeline/01_collect/tier1/04_cut_clips.ipynb",
    "src/pipeline/01_collect/tier3/04_cut_clips.ipynb",
    "src/pipeline/02_curate",
    "src/tools/build_review_manifest.py",
    "src/tools/build_review_assignments.py",
    "src/tools/build_roi_preview.py",
    "src/tools/export_review_batch.py",
    "src/tools/merge_review_results.py",
)

MEDIA_ROOTS = (
    "data/01_collect/cut_clips",
    "data/01_collect/final_clips_batch1",
    "data/02_curate/roi_preview",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_hash_files():
    seen = set()
    for value in HASH_INPUTS:
        path = ROOT / value
        candidates = path.rglob("*") if path.is_dir() else (path,)
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() in {
                    ".mp4", ".pt", ".pyc"}:
                continue
            rel = candidate.relative_to(ROOT).as_posix()
            if rel not in seen:
                seen.add(rel)
                yield candidate
    cut_root = ROOT / "data/01_collect/cut_clips"
    if cut_root.exists():
        for candidate in sorted(cut_root.rglob("*.csv")):
            rel = candidate.relative_to(ROOT).as_posix()
            if rel not in seen:
                seen.add(rel)
                yield candidate


def git_head():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def write_json_atomic(path, payload):
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out = Path(args.out).resolve()
    snapshot_path = out / "lineage_snapshot.json"
    if snapshot_path.exists():
        raise SystemExit(f"Snapshot đã tồn tại, từ chối ghi đè: {snapshot_path}")
    out.mkdir(parents=True, exist_ok=True)

    hashes = []
    for path in sorted(iter_hash_files()):
        stat = path.stat()
        hashes.append({
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": stat.st_size,
            "sha256": sha256(path),
        })

    cut_logs_dir = out / "cut_logs"
    copied_logs = []
    cut_root = ROOT / "data/01_collect/cut_clips"
    for source in sorted(cut_root.rglob("*.csv")):
        relative = source.relative_to(cut_root)
        destination = cut_logs_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_logs.append(destination.relative_to(out).as_posix())

    media = []
    media_summary = {}
    for value in MEDIA_ROOTS:
        root = ROOT / value
        rows = []
        if root.exists():
            for path in sorted(root.rglob("*.mp4")):
                stat = path.stat()
                row = {
                    "path": path.relative_to(ROOT).as_posix(),
                    "size_bytes": stat.st_size,
                }
                rows.append(row)
                media.append(row)
        media_summary[value] = {
            "files": len(rows),
            "size_bytes": sum(row["size_bytes"] for row in rows),
        }

    payload = {
        "schema": "cut_hotfix_baseline_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "policy": {
            "media_hashed": False,
            "media_mutated": False,
            "reason": "Inventory only; preserve large ignored media in place.",
        },
        "hashed_files": hashes,
        "copied_cut_logs": copied_logs,
        "media_summary": media_summary,
        "media_inventory": media,
    }
    write_json_atomic(snapshot_path, payload)
    print(f"Snapshot: {snapshot_path}")
    print(f"Hashed files: {len(hashes)}")
    print(f"Copied cut CSV: {len(copied_logs)}")
    for name, summary in media_summary.items():
        print(f"{name}: {summary['files']} MP4, "
              f"{summary['size_bytes'] / 2**30:.3f} GiB")


if __name__ == "__main__":
    main()
