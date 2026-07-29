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
                    "balance_dropped": 2,
                    "clean": 2,
                },
            )
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(FileExistsError):
                    module.main()


if __name__ == "__main__":
    unittest.main()
