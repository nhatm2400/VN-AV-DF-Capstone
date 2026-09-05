import csv
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/pipeline/02_curate/02_scoring/02_active_speaker/05_apply_laser_scores.py"
PYTHON = sys.executable


class LaserEnrichmentTest(unittest.TestCase):
    def test_laser_confirms_ambiguous_voiceover_bins(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            base = temp / "base"
            out = temp / "enriched"
            base.mkdir()
            policy = {
                "bin_ms": 200, "min_contiguous_bad_ms": 800,
                "min_cumulative_bad_ms": 500, "min_bad_voiced_ratio": 0.2,
                "light_active_threshold": 0.0, "light_margin": 0.5,
                "laser_active_threshold": 0.5, "laser_margin": 0.15,
                "mouth_freeze_threshold": 1.0,
            }
            with (base / "asd_clip_scores.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["clip_id", "config_hash"])
                writer.writeheader(); writer.writerow({"clip_id": "c1", "config_hash": "base"})
            rows = []
            for index in range(4):
                rows.append({
                    "clip_id": "c1", "bin_index": index, "start_ms": index*200,
                    "end_ms": (index+1)*200, "speech": True, "face_visible": True,
                    "mouth_motion": 4.0, "light_asd_score": 0.1,
                    "laser_requested": True, "multiple_competing_faces": False,
                    "inference_failure": False,
                })
            timeline = base / "asd_timeline.jsonl.gz"
            with gzip.open(timeline, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            (base / "failures.csv").write_text(
                "clip_id,file_path,error_type,error_message\n", encoding="utf-8"
            )
            (base / "run_config.json").write_text(json.dumps({
                "coverage_passed": True, "policy": policy, "config_hash": "base",
                "model_versions": {"light": "x"}, "manifest_sha256": "manifest",
                "manifest_rows": 1, "batch_start": 0, "batch_end": 1,
            }), encoding="utf-8")
            sidecar = temp / "laser.jsonl"
            sidecar.write_text("".join(json.dumps({
                "clip_id": "c1", "bin_index": index, "laser_score": 0.1,
            }) + "\n" for index in range(4)), encoding="utf-8")
            metadata = temp / "laser_metadata.json"
            metadata.write_text(json.dumps({
                "schema": "laser_sidecar_v1", "model_git_sha": "abc",
                "weights_sha256": "def",
                "source_timeline_sha256": hashlib.sha256(timeline.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            subprocess.run([
                PYTHON, str(SCRIPT), "--base_run", str(base),
                "--laser_scores", str(sidecar), "--laser_metadata", str(metadata),
                "--out_run", str(out),
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            with (out / "asd_clip_scores.csv").open(encoding="utf-8") as handle:
                result = next(csv.DictReader(handle))
            self.assertEqual(result["temporal_decision"], "reject")
            self.assertEqual(result["temporal_reason"], "voiceover")


if __name__ == "__main__":
    unittest.main()
