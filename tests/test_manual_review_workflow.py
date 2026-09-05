"""End-to-end test cho quy trình lọc tay một hoặc nhiều người.

Chia assignment (primary disjoint + calibration dùng chung) rồi gộp kết quả: kiểm
coverage đúng 1 lần/clip, clip bất đồng bị đẩy sang needs_resolution thay vì âm
thầm chọn một phía, và manifest cuối chỉ ra khi đã phân xử hết.
"""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "src/tools/review/build_review_assignments.py"
MERGE = ROOT / "src/tools/review/merge_review_results.py"


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class ManualReviewWorkflowTest(unittest.TestCase):
    def test_single_reviewer_calibration_is_complete_without_adjudication(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest = tmp / "manifest.csv"
            assignments_dir = tmp / "assignments"
            merge_dir = tmp / "merged"
            final_clean = tmp / "manual_clean.csv"
            reviewer = "r1"
            rows = [{
                "clip_id": f"c{i}",
                "file_path": f"c{i}.mp4",
                "tier": f"tier{1 + i % 3}",
                "source_video": f"s{i}",
                "voiced_ms": "5000",
                "calibration_split": "tune" if i < 3 else "locked_validation",
            } for i in range(6)]
            write_csv(manifest, rows, list(rows[0]))

            subprocess.run([
                sys.executable, str(BUILD),
                "--manifest", str(manifest),
                "--calibration", str(manifest),
                "--reviewers", reviewer,
                "--out_dir", str(assignments_dir),
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            assignment = assignments_dir / f"assignment_{reviewer}.csv"
            assigned = read_csv(assignment)
            self.assertEqual(len(assigned), 6)
            self.assertEqual({r["assignment_role"] for r in assigned}, {"calibration"})
            self.assertEqual({r["assigned_reviewer"] for r in assigned}, {reviewer})

            result = tmp / "result_r1.csv"
            reviewed = []
            for index, row in enumerate(assigned):
                decision = "reject" if index == 0 else "keep"
                reviewed.append({
                    "clip_id": row["clip_id"],
                    "file_path": row["file_path"],
                    "decision": decision,
                    "reason": "static" if decision == "reject" else "",
                    "bad_intervals_json": (
                        '[{"start_ms":1000,"end_ms":2000,"reason":"static"}]'
                        if decision == "reject" else "[]"
                    ),
                    "reviewer_id": reviewer,
                    "rubric_version": "v3",
                    "ts": "",
                })
            write_csv(result, reviewed, list(reviewed[0]))
            subprocess.run([
                sys.executable, str(MERGE),
                "--assignments", str(assignment),
                "--results", str(result),
                "--manifest", str(manifest),
                "--out_dir", str(merge_dir),
                "--final_clean", str(final_clean),
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            summary = json.loads((merge_dir / "merge_summary.json").read_text("utf-8"))
            self.assertEqual(summary["missing_judgements"], 0)
            self.assertEqual(summary["needs_resolution"], 0)
            assignment_summary = json.loads(
                (assignments_dir / "assignment_summary.json").read_text("utf-8")
            )
            self.assertEqual(assignment_summary["review_mode"], "single_reviewer")
            self.assertEqual(assignment_summary["reviewer_count"], 1)
            self.assertEqual(len(read_csv(merge_dir / "review_labels_v3.csv")), 6)
            self.assertEqual(len(read_csv(final_clean)), 5)

    def test_disjoint_primary_shared_calibration_and_adjudication(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest = tmp / "manifest.csv"
            calibration = tmp / "calibration.csv"
            assignments_dir = tmp / "assignments"
            merge_dir = tmp / "merged"
            final_clean = tmp / "manual_clean.csv"
            reviewers = ["r1", "r2", "r3"]

            rows = [{
                "clip_id": f"c{i}",
                "file_path": f"c{i}.mp4",
                "tier": f"tier{1 + i % 3}",
                "source_video": f"s{i % 4}",
            } for i in range(9)]
            write_csv(manifest, rows, list(rows[0]))
            cal_rows = [{
                "clip_id": f"c{i}", "decision": "keep", "reason": "",
                "bad_intervals_json": "[]", "reviewer_id": "r1",
                "rubric_version": "v3", "ts": "",
            } for i in range(3)]
            write_csv(calibration, cal_rows, list(cal_rows[0]))

            subprocess.run([
                sys.executable, str(BUILD),
                "--manifest", str(manifest),
                "--calibration", str(calibration),
                "--reviewers", *reviewers,
                "--out_dir", str(assignments_dir),
            ], cwd=ROOT, check=True, capture_output=True, text=True)

            assignment_paths = [
                assignments_dir / f"assignment_{reviewer}.csv"
                for reviewer in reviewers
            ]
            assignment_rows = [read_csv(path) for path in assignment_paths]
            self.assertEqual([len(x) for x in assignment_rows], [5, 5, 5])
            primary_ids = [
                row["clip_id"] for rows_for_reviewer in assignment_rows
                for row in rows_for_reviewer if row["assignment_role"] == "primary"
            ]
            self.assertEqual(len(primary_ids), 6)
            self.assertEqual(len(set(primary_ids)), 6)
            for rows_for_reviewer in assignment_rows:
                self.assertEqual(
                    {r["clip_id"] for r in rows_for_reviewer
                     if r["assignment_role"] == "calibration"},
                    {"c0", "c1", "c2"},
                )

            result_paths = []
            for reviewer, assigned in zip(reviewers, assignment_rows):
                result = tmp / f"result_{reviewer}.csv"
                output = []
                for row in assigned:
                    cid = row["clip_id"]
                    decision = "reject" if cid == "c2" else "keep"
                    if cid == "c1" and reviewer == "r3":
                        decision = "reject"
                    output.append({
                        "clip_id": cid,
                        "file_path": row["file_path"],
                        "decision": decision,
                        "reason": "static" if decision == "reject" else "",
                        "bad_intervals_json": (
                            '[{"start_ms":1000,"end_ms":2000,"reason":"static"}]'
                            if decision == "reject" else "[]"
                        ),
                        "reviewer_id": reviewer,
                        "rubric_version": "v3",
                        "ts": "",
                    })
                write_csv(result, output, list(output[0]))
                result_paths.append(result)

            command = [
                sys.executable, str(MERGE),
                "--assignments", *map(str, assignment_paths),
                "--results", *map(str, result_paths),
                "--manifest", str(manifest),
                "--out_dir", str(merge_dir),
                "--final_clean", str(final_clean),
            ]
            first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(first.returncode, 0)
            pending_path = merge_dir / "needs_resolution.csv"
            pending = read_csv(pending_path)
            self.assertEqual([r["clip_id"] for r in pending], ["c1"])

            pending[0]["final_decision"] = "keep"
            pending[0]["final_bad_intervals_json"] = "[]"
            pending[0]["resolved_by"] = "lead"
            write_csv(pending_path, pending, list(pending[0]))
            subprocess.run(
                [*command, "--resolution", str(pending_path)],
                cwd=ROOT, check=True, capture_output=True, text=True,
            )
            self.assertTrue(final_clean.is_file())
            summary = json.loads((merge_dir / "merge_summary.json").read_text("utf-8"))
            self.assertEqual(summary["missing_judgements"], 0)
            self.assertEqual(summary["needs_resolution"], 0)
            self.assertEqual(summary["resolved_from_resolution"], 1)
            labels = read_csv(merge_dir / "review_labels_v3.csv")
            self.assertEqual(len(labels), 9)


if __name__ == "__main__":
    unittest.main()
