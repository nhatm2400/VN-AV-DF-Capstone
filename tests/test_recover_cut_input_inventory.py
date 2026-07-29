import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "tools" / "recover_cut_input_inventory.py"
SPEC = importlib.util.spec_from_file_location("recover_cut_inventory", SCRIPT)
RECOVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RECOVER
SPEC.loader.exec_module(RECOVER)


class RecoverCutInputInventoryTest(unittest.TestCase):
    def test_union_is_exact_and_tracks_provenance(self):
        accepted = [
            {"source_video": "a"},
            {"source_video": "a"},
            {"source_video": "b"},
        ]
        rejected = [
            {"video": "b"},
            {"video": "c"},
            {"video": "c"},
        ]
        rows = RECOVER.recover_rows(accepted, rejected)
        self.assertEqual([row["filename"] for row in rows], [
            "a.mp4", "b.mp4", "c.mp4",
        ])
        by_id = {row["source_video"]: row for row in rows}
        self.assertEqual(by_id["a"]["recovered_from"], "accepted")
        self.assertEqual(by_id["b"]["recovered_from"], "accepted+rejected")
        self.assertEqual(by_id["c"]["rejected_row_count"], 2)

    def test_unsafe_source_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            RECOVER.recover_rows(
                [{"source_video": "../escape"}],
                [],
            )


if __name__ == "__main__":
    unittest.main()
