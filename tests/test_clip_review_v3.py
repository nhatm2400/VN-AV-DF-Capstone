import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/tools/review/clip_review.py"


def load_module():
    spec = importlib.util.spec_from_file_location("clip_review_v3", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClipReviewV3Test(unittest.TestCase):
    def test_normalizes_intervals_and_derives_longest_reason(self):
        module = load_module()
        intervals = module.normalize_bad_intervals([
            {"start_ms": 2000.4, "end_ms": 3000.2, "reason": "voiceover"},
            {"start_ms": 100, "end_ms": 500, "reason": "static"},
        ])
        self.assertEqual(intervals[0]["reason"], "static")
        self.assertEqual(module.longest_interval_reason(intervals), "voiceover")

    def test_rejects_invalid_interval(self):
        module = load_module()
        with self.assertRaises(ValueError):
            module.normalize_bad_intervals([
                {"start_ms": 900, "end_ms": 200, "reason": "static"}
            ])

    def test_material_duration_rule(self):
        module = load_module()
        self.assertFalse(module.intervals_are_material(
            [{"start_ms": 0, "end_ms": 200, "reason": "static"}], 2000
        ))
        self.assertTrue(module.intervals_are_material(
            [{"start_ms": 0, "end_ms": 800, "reason": "static"}], 5000
        ))
        self.assertTrue(module.intervals_are_material([
            {"start_ms": 0, "end_ms": 300, "reason": "static"},
            {"start_ms": 1000, "end_ms": 1300, "reason": "voiceover"},
        ], 2000))


if __name__ == "__main__":
    unittest.main()
