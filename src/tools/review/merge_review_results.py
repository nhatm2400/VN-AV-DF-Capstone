"""
Audit và gộp kết quả của một hoặc nhiều reviewer.

Clip primary có đúng một reviewer. Clip calibration phải có kết quả của mọi reviewer;
nếu không đồng thuận hoặc có nhãn uncertain thì được đưa vào needs_resolution.csv.
Trong workflow một reviewer, người đó phải sửa nhãn uncertain trong file kết quả rồi
chạy lại. Script chỉ xuất manual_clean_v3.csv khi coverage hoàn tất và không còn clip
cần xử lý.
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


VALID = {"keep", "reject", "uncertain"}
REASONS = {"static", "voiceover", "dubbed", "wrong_face", "mouth", "cut", "broken"}


def parse_intervals(row, field="bad_intervals_json"):
    try:
        values = json.loads(row.get(field, "") or "[]")
    except json.JSONDecodeError as error:
        raise SystemExit(f"[LỖI] bad_intervals_json không hợp lệ: {row.get('clip_id')}") from error
    if not isinstance(values, list):
        raise SystemExit(f"[LỖI] bad_intervals_json không phải list: {row.get('clip_id')}")
    output = []
    for item in values:
        try:
            start, end, reason = int(item["start_ms"]), int(item["end_ms"]), item["reason"]
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"[LỖI] Interval sai schema: {row.get('clip_id')}") from error
        if start < 0 or end <= start or reason not in REASONS:
            raise SystemExit(f"[LỖI] Interval sai giá trị: {row.get('clip_id')}")
        output.append({"start_ms": start, "end_ms": end, "reason": reason})
    return sorted(output, key=lambda item: (item["start_ms"], item["end_ms"], item["reason"]))


def intervals_agree(rows, tolerance_ms=200):
    parsed = [parse_intervals(row) for row in rows]
    if not parsed:
        return True, []
    reference = parsed[0]
    for values in parsed[1:]:
        if len(values) != len(reference):
            return False, []
        for left, right in zip(reference, values):
            if (left["reason"] != right["reason"]
                    or abs(left["start_ms"] - right["start_ms"]) > tolerance_ms
                    or abs(left["end_ms"] - right["end_ms"]) > tolerance_ms):
                return False, []
    consensus = []
    for index in range(len(reference)):
        consensus.append({
            "start_ms": round(sum(values[index]["start_ms"] for values in parsed) / len(parsed)),
            "end_ms": round(sum(values[index]["end_ms"] for values in parsed) / len(parsed)),
            "reason": reference[index]["reason"],
        })
    return True, consensus


def intervals_are_material(intervals, voiced_ms):
    spans = sorted((item["start_ms"], item["end_ms"]) for item in intervals)
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    lengths = [end - start for start, end in merged]
    total = sum(lengths)
    return bool(lengths) and (max(lengths) >= 800 or (
        total >= 500 and voiced_ms > 0 and total / voiced_ms >= 0.20
    ))


def longest_reason(intervals):
    return max(intervals, key=lambda item: (
        item["end_ms"] - item["start_ms"], -item["start_ms"]
    ))["reason"] if intervals else ""


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    return paths


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignments", nargs="+", required=True)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--manifest",
                    default="data/02_curate/manifests/all_clean_review.csv")
    ap.add_argument("--rubric", default="v3")
    ap.add_argument("--out_dir", default="data/02_curate/manual/merged_v3")
    ap.add_argument("--resolution", "--adjudication", dest="resolution", default="",
                    help="needs_resolution.csv đã điền final_decision/final_reason/resolved_by")
    ap.add_argument("--final_clean",
                    default="data/02_curate/manifests/manual_clean_v3.csv")
    ap.add_argument("--allow_partial", action="store_true",
                    help="cho phép xuất manifest khi CHỦ Ý dừng sớm (đã đủ keep). "
                         "Chỉ gộp clip đã có phán quyết; summary ghi partial=true. "
                         "Clip bất đồng/uncertain vẫn chặn — phải phân xử trước.")
    args = ap.parse_args()

    manifest = read_csv(args.manifest)
    manifest_by_id = {r["clip_id"]: r for r in manifest}
    expected = {}
    calibration_reviewers = defaultdict(set)
    assignment_paths = expand_paths(args.assignments)
    result_paths = expand_paths(args.results)
    for path in assignment_paths:
        for row in read_csv(path):
            cid = row.get("clip_id", "")
            reviewer = row.get("assigned_reviewer", "")
            role = row.get("assignment_role", "")
            key = (reviewer, cid)
            if not cid or not reviewer or role not in {"primary", "calibration"}:
                raise SystemExit(f"[LỖI] Assignment sai schema: {path}")
            if cid not in manifest_by_id:
                raise SystemExit(f"[LỖI] Assignment có clip ngoài manifest: {cid}")
            if key in expected:
                raise SystemExit(f"[LỖI] Assignment trùng {reviewer}/{cid}")
            expected[key] = role
            if role == "calibration":
                calibration_reviewers[cid].add(reviewer)

    actual = {}
    for path in result_paths:
        for row in read_csv(path):
            cid = row.get("clip_id", "")
            reviewer = row.get("reviewer_id", "")
            decision = row.get("decision", "")
            key = (reviewer, cid)
            if key not in expected:
                raise SystemExit(f"[LỖI] Kết quả ngoài assignment: {reviewer}/{cid}")
            if key in actual:
                raise SystemExit(f"[LỖI] Kết quả trùng: {reviewer}/{cid}")
            if decision not in VALID:
                raise SystemExit(f"[LỖI] Decision không hợp lệ: {reviewer}/{cid}")
            if row.get("rubric_version") != args.rubric:
                raise SystemExit(f"[LỖI] Sai rubric: {reviewer}/{cid}")
            intervals = parse_intervals(row)
            if decision == "reject" and not intervals:
                raise SystemExit(f"[LỖI] Reject rubric v3 thiếu interval: {reviewer}/{cid}")
            if decision == "reject" and row.get("reason", "") != longest_reason(intervals):
                raise SystemExit(f"[LỖI] reason không khớp interval dài nhất: {reviewer}/{cid}")
            source = manifest_by_id[cid]
            try:
                voiced_ms = int(round(float(source.get("voiced_ms", 0))))
            except (TypeError, ValueError):
                voiced_ms = 0
            if not voiced_ms:
                try:
                    voiced_ms = int(round(float(source.get("duration", 0)) * 1000))
                except (TypeError, ValueError):
                    voiced_ms = 0
            if decision == "reject" and not intervals_are_material(intervals, voiced_ms):
                raise SystemExit(f"[LỖI] Reject chưa đạt duration rule v3: {reviewer}/{cid}")
            actual[key] = row

    missing = sorted(set(expected) - set(actual))
    by_clip = defaultdict(list)
    for key, row in actual.items():
        by_clip[key[1]].append(row)

    resolved = {}
    pending = []
    for cid in manifest_by_id:
        rows = by_clip.get(cid, [])
        roles = {expected[(r["reviewer_id"], cid)] for r in rows}
        decisions = {r["decision"] for r in rows}
        is_calibration = cid in calibration_reviewers
        complete_calibration = (not is_calibration or
                                {r["reviewer_id"] for r in rows} ==
                                calibration_reviewers[cid])
        if not rows:
            continue
        interval_match, consensus_intervals = intervals_agree(rows)
        if ("uncertain" in decisions or len(decisions) != 1 or not complete_calibration
                or (decisions == {"reject"} and not interval_match)):
            pending.append({
                "clip_id": cid,
                "file_path": manifest_by_id[cid].get("file_path", ""),
                "assignment_role": "calibration" if is_calibration else
                                   next(iter(roles), "primary"),
                "reviewers": ";".join(sorted(r["reviewer_id"] for r in rows)),
                "decisions": ";".join(f"{r['reviewer_id']}={r['decision']}" for r in rows),
                "reasons": ";".join(f"{r['reviewer_id']}={r.get('reason', '')}" for r in rows),
                "bad_intervals_by_reviewer_json": json.dumps({
                    r["reviewer_id"]: parse_intervals(r) for r in rows
                }, ensure_ascii=False, separators=(",", ":")),
                "final_decision": "",
                "final_reason": "",
                "final_bad_intervals_json": "",
                "resolved_by": "",
            })
        else:
            decision = next(iter(decisions))
            resolved[cid] = (decision, rows[0].get("reason", ""), consensus_intervals)

    resolved_from_resolution = 0
    if args.resolution:
        resolution = {r.get("clip_id", ""): r for r in read_csv(args.resolution)}
        pending_ids = {r["clip_id"] for r in pending}
        extra = sorted(set(resolution) - pending_ids - {""})
        if extra:
            raise SystemExit(f"[LỖI] Resolution có {len(extra)} clip không cần xử lý")
        unresolved = []
        for row in pending:
            final = resolution.get(row["clip_id"], {})
            decision = final.get("final_decision", "")
            resolved_by = final.get("resolved_by", "") or final.get("adjudicator", "")
            if decision in {"keep", "reject"} and resolved_by:
                intervals = parse_intervals(final, "final_bad_intervals_json")
                if decision == "reject" and not intervals:
                    unresolved.append(row)
                    continue
                source = manifest_by_id[row["clip_id"]]
                try:
                    voiced_ms = int(round(float(source.get("voiced_ms", 0))))
                except (TypeError, ValueError):
                    voiced_ms = 0
                if not voiced_ms:
                    try:
                        voiced_ms = int(round(float(source.get("duration", 0)) * 1000))
                    except (TypeError, ValueError):
                        voiced_ms = 0
                if decision == "reject" and not intervals_are_material(intervals, voiced_ms):
                    unresolved.append(row)
                    continue
                reason = longest_reason(intervals) if decision == "reject" else ""
                resolved[row["clip_id"]] = (decision, reason, intervals)
                resolved_from_resolution += 1
            else:
                unresolved.append(row)
        pending = unresolved

    os.makedirs(args.out_dir, exist_ok=True)
    pending_path = os.path.join(args.out_dir, "needs_resolution.csv")
    write_csv(pending_path, pending, [
        "clip_id", "file_path", "assignment_role", "reviewers", "decisions",
        "reasons", "bad_intervals_by_reviewer_json", "final_decision", "final_reason",
        "final_bad_intervals_json", "resolved_by",
    ])
    # Dừng sớm vì đã đủ keep là ý định hợp lệ, nhưng phải GHI RÕ là partial —
    # manifest partial không đại diện cho toàn manifest, nên mọi tỉ lệ tính từ nó
    # (keep-rate, phân bố tier/channel) chỉ áp cho phần đã review.
    partial = bool(missing) and args.allow_partial
    summary = {
        "schema": "manual_review_merge_v1",
        "partial": partial,
        "expected_judgements": len(expected),
        "received_judgements": len(actual),
        "missing_judgements": len(missing),
        "resolved_clips": len(resolved),
        "needs_resolution": len(pending),
        "resolved_from_resolution": resolved_from_resolution,
        "decisions_resolved": dict(Counter(value[0] for value in resolved.values())),
    }
    if partial:
        summary["reviewed_by_reviewer"] = dict(Counter(
            reviewer for reviewer, _ in actual))
    with open(os.path.join(args.out_dir, "merge_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"-> {pending_path}")
    blocked = pending or (missing and not args.allow_partial)
    if blocked:
        if os.path.exists(args.final_clean):
            print(f"[CẢNH BÁO] Không ghi đè final manifest đang có: {args.final_clean}")
        if missing and not args.allow_partial:
            print(f"[GỢI Ý] Thiếu {len(missing)} phán quyết. Nếu CHỦ Ý dừng sớm "
                  f"(đã đủ keep), chạy lại với --allow_partial.")
        raise SystemExit("[CHƯA XONG] Thiếu coverage hoặc còn clip cần phân xử")

    reviewed_ids = set(resolved)
    clean = [row for row in manifest
             if row["clip_id"] in reviewed_ids and resolved[row["clip_id"]][0] == "keep"]
    if os.path.exists(args.final_clean):
        raise SystemExit(f"[LỖI] Final manifest đã tồn tại: {args.final_clean}")
    write_csv(args.final_clean, clean, list(manifest[0]))
    labels_path = os.path.join(args.out_dir, "review_labels_v3.csv")
    label_rows = [{
        "clip_id": cid,
        "decision": decision,
        "reason": reason,
        "bad_intervals_json": json.dumps(intervals, ensure_ascii=False, separators=(",", ":")),
        "rubric_version": args.rubric,
    } for cid, (decision, reason, intervals) in sorted(resolved.items())]
    write_csv(labels_path, label_rows,
              ["clip_id", "decision", "reason", "bad_intervals_json", "rubric_version"])
    if partial:
        print(f"[PARTIAL] {len(reviewed_ids)}/{len(manifest)} clip đã có phán quyết; "
              f"{len(missing)} phán quyết còn thiếu.")
    print(f"-> {args.final_clean} ({len(clean)} keep/{len(reviewed_ids)} đã review)")


if __name__ == "__main__":
    main()
