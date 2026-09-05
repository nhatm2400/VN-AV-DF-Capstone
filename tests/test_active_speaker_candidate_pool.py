import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "src/pipeline/02_curate/02_scoring/02_active_speaker/00_build_candidate_pool.py"
)
SPEC = importlib.util.spec_from_file_location("active_speaker_candidate_pool", SCRIPT)
POOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POOL
SPEC.loader.exec_module(POOL)


def manifest_rows():
    rows = []
    for tier in ("tier1", "tier2", "tier3"):
        for source in range(4):
            for clip in range(2):
                rows.append({
                    "clip_id": f"{tier}_s{source}_c{clip}",
                    "source_video": f"{tier}_source_{source}",
                    "tier": tier,
                    "file_path": f"{tier}_s{source}_c{clip}.mp4",
                })
    return rows


class ActiveSpeakerCandidatePoolTest(unittest.TestCase):
    def test_balanced_source_disjoint_selection(self):
        output = POOL.select_candidate_pool(pd.DataFrame(manifest_rows()), 3, 42)
        self.assertEqual(len(output), 9)
        self.assertEqual(output["tier"].value_counts().to_dict(), {
            "tier1": 3, "tier2": 3, "tier3": 3,
        })
        self.assertEqual(output["source_video"].nunique(), 9)
        self.assertFalse(output["clip_id"].duplicated().any())

    def test_selection_is_independent_of_manifest_row_order(self):
        frame = pd.DataFrame(manifest_rows())
        first = POOL.select_candidate_pool(frame, 3, 7)
        second = POOL.select_candidate_pool(
            frame.sample(frac=1, random_state=99).reset_index(drop=True), 3, 7
        )
        self.assertEqual(first["clip_id"].tolist(), second["clip_id"].tolist())

    def test_rejects_insufficient_unique_sources(self):
        with self.assertRaisesRegex(ValueError, "unused sources"):
            POOL.select_candidate_pool(pd.DataFrame(manifest_rows()), 5, 42)


if __name__ == "__main__":
    unittest.main()
