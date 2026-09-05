import csv
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/pipeline/02_curate/02_scoring/02_active_speaker/07_calibrate.py"
PYTHON = sys.executable


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


class ActiveSpeakerCalibrationTest(unittest.TestCase):
    def test_300_tune_150_locked_can_publish_only_after_locked_metrics_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            manifest_rows, labels, timeline_rows = [], [], []
            index = 0
            for tier in ("tier1", "tier2", "tier3"):
                for within in range(150):
                    clip_id = f"c{index}"
                    split = "tune" if within < 100 else "locked_validation"
                    bad = within % 5 == 0
                    manifest_rows.append({
                        "clip_id": clip_id, "tier": tier,
                        "source_video": f"source_{index}", "calibration_split": split,
                    })
                    labels.append({
                        "clip_id": clip_id, "decision": "reject" if bad else "keep",
                        "reason": "static" if bad else "",
                        "bad_intervals_json": (
                            '[{"start_ms":0,"end_ms":800,"reason":"static"}]' if bad else "[]"
                        ),
                        "rubric_version": "v3",
                    })
                    for bin_index in range(4):
                        timeline_rows.append({
                            "clip_id": clip_id, "bin_index": bin_index,
                            "speech": True, "face_visible": True,
                            "mouth_motion": 0.0 if bad else 4.0,
                            "light_asd_score": -1.0 if bad else 1.0,
                            "multiple_competing_faces": False,
                            "inference_failure": False,
                        })
                    index += 1
            manifest = temp / "calibration.csv"
            label_path = temp / "labels.csv"
            timeline = temp / "asd_timeline.jsonl.gz"
            write_csv(manifest, manifest_rows, list(manifest_rows[0]))
            write_csv(label_path, labels, list(labels[0]))
            with gzip.open(timeline, "wt", encoding="utf-8") as handle:
                for row in timeline_rows:
                    handle.write(json.dumps(row) + "\n")
            (temp / "run_config.json").write_text(json.dumps({
                "coverage_passed": True, "config_hash": "score-config",
                "model_versions": {"light_weights_sha256": "weight-hash"},
            }), encoding="utf-8")
            output = temp / "policy.json"
            subprocess.run([
                PYTHON, str(SCRIPT), "--calibration_manifest", str(manifest),
                "--consensus_labels", str(label_path), "--timeline", str(timeline),
                "--out", str(output), "--light_margins", "0.5",
                "--mouth_freeze_thresholds", "1.0",
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            policy = json.loads(output.read_text("utf-8"))
            self.assertTrue(policy["gate_passed"])
            self.assertEqual(policy["locked_validation_metrics"]["recall_bad"], 1.0)
            self.assertEqual(policy["locked_validation_metrics"]["false_reject_clean"], 0.0)


if __name__ == "__main__":
    unittest.main()
