"""
Chia manifest review cho một hoặc nhiều reviewer.

Mỗi clip calibration xuất hiện trong assignment của mọi reviewer. Các clip còn lại
chỉ xuất hiện trong đúng một assignment và được cân bằng theo tier.

Ví dụ:
  D:/Anaconda/envs/vn_av_df/python.exe src/tools/review/build_review_assignments.py \
      --reviewers nguyenminhnhat
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not value:
        raise ValueError("reviewer ID rỗng hoặc không hợp lệ")
    return value


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "clip_id" not in rows[0]:
        raise SystemExit(f"[LỖI] CSV rỗng hoặc thiếu clip_id: {path}")
    ids = [r["clip_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"[LỖI] clip_id trùng trong {path}")
    return rows


def calibration_ids(path):
    rows = read_rows(path)
    is_selection_manifest = (
        "calibration_split" in rows[0]
        or all(row.get("assignment_role") == "calibration" for row in rows)
    )
    ids = [row["clip_id"] for row in rows if is_selection_manifest or
           row.get("decision") in {"keep", "reject", "uncertain"}]
    if not ids:
        raise SystemExit(f"[LỖI] Không có quyết định calibration hợp lệ trong {path}")
    return ids


def distribute(rows, reviewers, seed):
    """Chia clip theo tier, luôn đưa clip kế tiếp cho reviewer đang có ít việc nhất."""
    loads = Counter()
    assigned = {reviewer: [] for reviewer in reviewers}
    tiers = {}
    for row in rows:
        tiers.setdefault(row.get("tier", "[UNKNOWN]"), []).append(row)
    for tier in sorted(tiers):
        group = tiers[tier]
        random.Random(f"{seed}:{tier}").shuffle(group)
        for row in group:
            reviewer = min(reviewers, key=lambda r: (loads[r], reviewers.index(r)))
            assigned[reviewer].append(row)
            loads[reviewer] += 1
    return assigned


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="data/02_curate/calibration/active_speaker_450_v3.csv")
    ap.add_argument("--calibration",
                    default="data/02_curate/calibration/active_speaker_450_v3.csv",
                    help="CSV chọn tập calibration chung; không cần có decision")
    ap.add_argument("--no_shared_calibration", action="store_true",
                    help="chia scope manual cuối sau khi calibration đã hoàn tất")
    ap.add_argument("--reviewers", nargs="+", required=True,
                    help="Một hoặc nhiều reviewer ID ổn định")
    ap.add_argument("--out_dir", default="data/02_curate/assignments/v3/calibration_450")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    reviewers = [safe_name(r) for r in args.reviewers]
    if not reviewers or len(reviewers) != len(set(reviewers)):
        raise SystemExit("[LỖI] Cần ít nhất một reviewer ID và không được trùng")

    rows = read_rows(args.manifest)
    by_id = {r["clip_id"]: r for r in rows}
    cal_ids = [] if args.no_shared_calibration else calibration_ids(args.calibration)
    missing = sorted(set(cal_ids) - set(by_id))
    if missing:
        raise SystemExit(f"[LỖI] {len(missing)} clip calibration không thuộc manifest")
    cal_set = set(cal_ids)
    remaining = [r for r in rows if r["clip_id"] not in cal_set]
    primary = distribute(remaining, reviewers, args.seed)

    fields = list(rows[0])
    for field in ("assignment_role", "assigned_reviewer"):
        if field not in fields:
            fields.append(field)

    outputs = []
    for reviewer in reviewers:
        out = os.path.join(args.out_dir, f"assignment_{reviewer}.csv")
        if os.path.exists(out) and not args.overwrite:
            raise SystemExit(f"[LỖI] Assignment đã tồn tại: {out}; dùng --overwrite nếu chủ ý")
        assignment = []
        for cid in cal_ids:
            row = dict(by_id[cid])
            row.update(assignment_role="calibration", assigned_reviewer=reviewer)
            assignment.append(row)
        for source in primary[reviewer]:
            row = dict(source)
            row.update(assignment_role="primary", assigned_reviewer=reviewer)
            assignment.append(row)
        write_csv(out, assignment, fields)
        outputs.append({
            "reviewer": reviewer,
            "path": out.replace("\\", "/"),
            "calibration": len(cal_ids),
            "primary": len(primary[reviewer]),
            "total": len(assignment),
            "tiers_primary": dict(Counter(r.get("tier", "[UNKNOWN]")
                                          for r in primary[reviewer])),
            "sources_primary": len({r.get("source_video", "") for r in primary[reviewer]}),
        })

    all_primary = [r["clip_id"] for reviewer in reviewers for r in primary[reviewer]]
    if len(all_primary) != len(remaining) or set(all_primary) != {
            r["clip_id"] for r in remaining}:
        raise SystemExit("[LỖI] Coverage primary không đúng 1 lần/clip")

    summary = {
        "schema": "manual_review_assignment_v1",
        "review_mode": "single_reviewer" if len(reviewers) == 1 else "multi_reviewer",
        "reviewer_count": len(reviewers),
        "manifest": args.manifest.replace("\\", "/"),
        "calibration_source": None if args.no_shared_calibration else
                              args.calibration.replace("\\", "/"),
        "seed": args.seed,
        "reviewers": reviewers,
        "manifest_clips": len(rows),
        "calibration_clips_per_reviewer": len(cal_ids),
        "primary_clips_disjoint": len(remaining),
        "outputs": outputs,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "assignment_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Manifest: {len(rows)} clip | calibration chung: {len(cal_ids)}")
    for item in outputs:
        print(f"  {item['reviewer']}: {item['primary']} primary + "
              f"{item['calibration']} calibration = {item['total']}")
    print(f"-> {summary_path}")


if __name__ == "__main__":
    main()
