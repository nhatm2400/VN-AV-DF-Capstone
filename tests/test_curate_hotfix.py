import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CURATE_PATH = ROOT / "src" / "pipeline" / "02_curate" / "04_curate.py"


def load_curate_module():
    spec = importlib.util.spec_from_file_location("curate_hotfix", CURATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CurateHotfixTest(unittest.TestCase):
    def test_sync_score_is_diagnostic_and_does_not_change_quality_ranking(self):
        module = load_curate_module()
        frame = pd.DataFrame({
            "det_ratio": [0.8, 0.8], "mean_face_area": [0.1, 0.1],
            "embed_consistency": [0.9, 0.9], "sync_conf": [-10.0, 10.0],
        })
        scores = module.quality_score(frame)
        self.assertAlmostEqual(float(scores.iloc[0]), float(scores.iloc[1]))

    def test_temporal_gate_rejects_only_when_locked_policy_passed(self):
        module = load_curate_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scored = root / "scored.csv"
            embeddings = root / "embeddings.npy"
            temporal_dir = root / "temporal"
            temporal_dir.mkdir()
            temporal = temporal_dir / "asd_clip_scores.csv"
            policy = root / "policy.json"
            output = root / "all_clean.csv"
            rows = [{
                "clip_id": f"clip_{index}", "source_video": f"video_{index}",
                "start_time": float(index), "has_embedding": True,
                "det_ratio": 1.0, "mean_face_area": 0.2,
                "embed_consistency": 1.0,
            } for index in range(3)]
            pd.DataFrame(rows).to_csv(scored, index=False)
            np.save(embeddings, np.eye(3, 4, dtype=np.float32))
            decisions = ["pass", "reject", "manual"]
            temporal_rows = []
            for index, decision in enumerate(decisions):
                temporal_rows.append({
                    "clip_id": f"clip_{index}", "voiced_ms": 1000,
                    "visible_active_speech_ratio": 1.0 if decision == "pass" else 0.0,
                    "unexplained_speech_ratio": 0.0 if decision == "pass" else 1.0,
                    "longest_unexplained_speech_ms": 0 if decision == "pass" else 1000,
                    "static_speech_ratio": 1.0 if decision == "reject" else 0.0,
                    "asd_disagreement_ratio": 1.0 if decision == "manual" else 0.0,
                    "temporal_decision": decision,
                    "temporal_reason": "static" if decision == "reject" else
                                       ("ambiguous" if decision == "manual" else ""),
                    "config_hash": "abc123",
                })
            pd.DataFrame(temporal_rows).to_csv(temporal, index=False)
            policy_values = {
                "bin_ms": 200, "min_contiguous_bad_ms": 800,
                "min_cumulative_bad_ms": 500, "min_bad_voiced_ratio": 0.2,
                "light_active_threshold": 0.0, "light_margin": 0.5,
                "laser_active_threshold": 0.5, "laser_margin": 0.15,
                "mouth_freeze_threshold": 1.0,
            }
            (temporal_dir / "run_config.json").write_text(json.dumps({
                "coverage_passed": True, "config_hash": "abc123", "policy": policy_values,
            }), encoding="utf-8")
            policy.write_text(json.dumps({
                "schema": "active_speaker_policy_v1", "gate_passed": True,
                "publish_mode": "auto_gate", "policy": policy_values,
            }), encoding="utf-8")
            argv = [
                "04_curate.py", "--scored_csv", str(scored), "--emb", str(embeddings),
                "--min_consistency", "0", "--cap_per_speaker", "30",
                "--temporal_scores", str(temporal), "--temporal_policy", str(policy),
                "--out", str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                module.main()
            clean = pd.read_csv(output)
            rejected = pd.read_csv(root / "all_clean_rejects.csv")
            self.assertEqual(set(clean.clip_id), {"clip_0", "clip_2"})
            self.assertEqual(set(rejected.clip_id), {"clip_1"})
            self.assertEqual(rejected.iloc[0].gate_stage, "temporal")
            self.assertEqual(rejected.iloc[0].gate_reason, "static")

    def test_outputs_form_complete_disjoint_partition_and_refuse_overwrite(self):
        module = load_curate_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scored = root / "scored.csv"
            embeddings = root / "embeddings.npy"
            output = root / "all_clean.csv"

            rows = []
            for index in range(4):
                rows.append({
                    "clip_id": f"clip_{index}",
                    "source_video": f"video_{index % 2}",
                    "start_time": float(index),
                    "has_embedding": True,
                    "det_ratio": 1.0,
                    "mean_face_area": 0.2,
                    "embed_consistency": 1.0,
                })
            pd.DataFrame(rows).to_csv(scored, index=False)
            values = np.tile(
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (4, 1),
            )
            np.save(embeddings, values)

            argv = [
                "04_curate.py",
                "--scored_csv", str(scored),
                "--emb", str(embeddings),
                "--cluster_dist", "0.6",
                "--min_consistency", "0",
                "--cap_per_speaker", "2",
                "--out", str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                module.main()

            rejected_path = root / "all_clean_rejects.csv"
            balance_path = root / "all_clean_balance_dropped.csv"
            config_path = root / "all_clean_config.json"
            clean = pd.read_csv(output)
            rejected = pd.read_csv(rejected_path)
            balance = pd.read_csv(balance_path)
            sets = [
                set(frame["clip_id"])
                for frame in (clean, rejected, balance)
            ]
            self.assertEqual(sum(map(len, sets)), 4)
            self.assertEqual(set.union(*sets), {f"clip_{i}" for i in range(4)})
            self.assertFalse(sets[0] & sets[1])
            self.assertFalse(sets[0] & sets[2])
            self.assertFalse(sets[1] & sets[2])

            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                config["counts"],
                {
                    "scored": 4,
                    "gate_rejected": 0,
                    "temporal_rejected": 0,
                    "face_quality_rejected": 0,
                    "temporal_manual": 0,
                    "balance_dropped": 2,
                    "clean": 2,
                },
            )
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(FileExistsError):
                    module.main()


if __name__ == "__main__":
    unittest.main()
