import importlib.util
import unittest
import io
import json
from unittest.mock import Mock, patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "src/tools/review/clip_review.py"


def load_module():
    spec = importlib.util.spec_from_file_location("clip_review_v3", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClipReviewV3Test(unittest.TestCase):
    def test_reason_only_reject_v4_and_invalid_reason(self):
        module = load_module()
        module.CLIPS = [{'clip_id': 'clip', 'duration': 5}]
        module.REVIEWER_ID = 'tester'
        handler = object.__new__(module.Handler)
        handler.path = '/api/mark'
        for reason, accepted in [('mouth', True), ('not_a_reason', False), ('', False)]:
            body = json.dumps(dict(i=0, decision='reject', reason=reason, bad_intervals=[])).encode()
            handler.headers = {'Content-Length': str(len(body))}
            handler.rfile = io.BytesIO(body)
            handler._json = Mock()
            with patch.object(module, 'save_decisions') as save:
                handler.do_POST()
            response = handler._json.call_args.args[0]
            self.assertEqual(response['ok'], accepted)
            if accepted:
                self.assertEqual(response['reason'], 'mouth')
                self.assertEqual(response['bad_intervals'], [])
                self.assertFalse(response['warning'])
                self.assertEqual(module.RUBRIC['clip'], 'v4')
                save.assert_called_once()
            else:
                save.assert_not_called()

    def test_short_reject_is_saved_with_warning_but_empty_reject_is_blocked(self):
        module = load_module()
        module.CLIPS = [{'clip_id': 'short', 'duration': 5}]
        module.REVIEWER_ID = 'tester'
        handler = object.__new__(module.Handler)
        handler.path = '/api/mark'
        for intervals, accepted in [([{'start_ms': 100, 'end_ms': 300, 'reason': 'mouth'}], True), ([], False)]:
            body = json.dumps(dict(i=0, decision='reject', bad_intervals=intervals)).encode()
            handler.headers = {'Content-Length': str(len(body))}
            handler.rfile = io.BytesIO(body)
            handler._json = Mock()
            with patch.object(module, 'save_decisions') as save:
                handler.do_POST()
            response = handler._json.call_args.args[0]
            self.assertEqual(response['ok'], accepted)
            if accepted:
                self.assertTrue(response['warning'])
                save.assert_called_once()
                self.assertEqual(module.DECISIONS['short'], 'reject')
            else:
                save.assert_not_called()

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
