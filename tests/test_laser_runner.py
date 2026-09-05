import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "src/pipeline/02_curate/02_scoring/02_active_speaker/04_run_laser.py"
)
SPEC = importlib.util.spec_from_file_location("laser_runner", MODULE_PATH)
LASER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LASER
SPEC.loader.exec_module(LASER)


class LaserRunnerTest(unittest.TestCase):
    def test_checkpoint_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "weights.model"
            checkpoint.write_bytes(b"pinned checkpoint")
            actual = LASER.sha256_file(checkpoint)
            self.assertEqual(
                LASER.require_sha256(checkpoint, actual, "checkpoint"), actual
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                LASER.require_sha256(checkpoint, "0" * 64, "checkpoint")

    def test_minimum_track_matches_one_200ms_bin(self):
        self.assertEqual(LASER.MIN_TRACK_FRAMES, 5)

    def test_request_bundle_requires_exact_counts_and_unique_bins(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [
                {
                    "clip_id": "c1", "bin_index": 2, "start_ms": 400,
                    "end_ms": 600, "file_path": "c1.mp4",
                },
                {
                    "clip_id": "c1", "bin_index": 3, "start_ms": 600,
                    "end_ms": 800, "file_path": "c1.mp4",
                },
            ]
            with (root / "laser_requests.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            (root / "request_config.json").write_text(json.dumps({
                "schema": "laser_request_v1",
                "requested_bins": 2,
                "requested_clips": 1,
                "source_timeline_sha256": "a" * 64,
            }), encoding="utf-8")
            loaded, config, _, _ = LASER.load_request_bundle(root)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(config["requested_clips"], 1)

            rows[1]["bin_index"] = 2
            with (root / "laser_requests.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Duplicate LASER request"):
                LASER.load_request_bundle(root)

    def test_bin_score_is_maximum_of_per_track_medians(self):
        requests = [{
            "clip_id": "c1", "bin_index": 1, "start_ms": 200, "end_ms": 400,
        }]
        tracks = [
            {
                "frames": np.arange(0, 10),
                "scores": np.asarray([0.1] * 5 + [0.2, 0.4, 0.6, 0.8, 1.0]),
            },
            {
                "frames": np.arange(5, 10),
                "scores": np.asarray([0.7, 0.7, 0.7, 0.7, 0.7]),
            },
        ]
        scores, missing = LASER.aggregate_requested_bins(requests, tracks)
        self.assertEqual(missing, [])
        self.assertEqual(scores[0]["clip_id"], "c1")
        self.assertEqual(scores[0]["bin_index"], 1)
        self.assertAlmostEqual(scores[0]["laser_score"], 0.7, places=6)

    def test_context_order_is_deterministic_and_has_three_speakers(self):
        primary = {
            "track_id": 5, "frames": np.arange(4),
            "faces": np.full((4, 112, 112), 5, dtype=np.uint8),
        }
        first = {
            "track_id": 1, "frames": np.arange(4),
            "faces": np.full((4, 112, 112), 1, dtype=np.uint8),
        }
        second = {
            "track_id": 2, "frames": np.arange(4),
            "faces": np.full((4, 112, 112), 2, dtype=np.uint8),
        }
        visual = LASER.visual_context(primary, [second, primary, first])
        self.assertEqual(visual.shape, (3, 4, 112, 112))
        self.assertEqual(int(visual[0, 0, 0, 0]), 5)
        self.assertEqual(int(visual[1, 0, 0, 0]), 1)
        self.assertEqual(int(visual[2, 0, 0, 0]), 2)


if __name__ == "__main__":
    unittest.main()
