import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/pipeline/02_curate/02_scoring/02_active_speaker/policy.py"


def load_module():
    spec = importlib.util.spec_from_file_location("active_speaker_policy", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bin_row(*, speech=True, visible=True, motion=4.0, light=1.0,
            laser=None, laser_requested=False, competing=False, failure=False,
            disagreement=False):
    return {
        "speech": speech,
        "face_visible": visible,
        "mouth_motion": motion,
        "light_asd_score": light,
        "laser_score": laser,
        "laser_requested": laser_requested,
        "multiple_competing_faces": competing,
        "asd_disagreement": disagreement,
        "inference_failure": failure,
    }


class TemporalPolicyTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.policy = self.module.TemporalPolicy()

    def summarize(self, rows):
        return self.module.summarize_timeline(rows, self.policy)

    def test_continuous_active_speaker_passes(self):
        result = self.summarize([bin_row() for _ in range(25)])
        self.assertEqual(result["temporal_decision"], "pass")
        self.assertEqual(result["visible_active_speech_ratio"], 1.0)

    def test_three_seconds_speaking_then_two_seconds_static_rejects(self):
        rows = [bin_row() for _ in range(15)]
        rows += [bin_row(motion=0.0, light=-1.0) for _ in range(10)]
        result = self.summarize(rows)
        self.assertEqual((result["temporal_decision"], result["temporal_reason"]),
                         ("reject", "static"))
        self.assertEqual(result["longest_static_ms"], 2000)

    def test_three_seconds_speaking_then_moving_broll_voiceover_rejects(self):
        rows = [bin_row() for _ in range(15)]
        rows += [bin_row(visible=False, motion=5.0, light=None) for _ in range(10)]
        result = self.summarize(rows)
        self.assertEqual((result["temporal_decision"], result["temporal_reason"]),
                         ("reject", "voiceover"))

    def test_static_tail_without_speech_does_not_reject(self):
        rows = [bin_row() for _ in range(15)]
        rows += [bin_row(speech=False, motion=0.0, light=-1.0) for _ in range(10)]
        self.assertEqual(self.summarize(rows)["temporal_decision"], "pass")

    def test_camera_motion_does_not_rescue_a_frozen_mouth(self):
        # Full-frame/camera motion is intentionally not part of the policy.
        rows = [bin_row(motion=0.0, light=-1.0) for _ in range(5)]
        result = self.summarize(rows)
        self.assertEqual(result["temporal_reason"], "static")

    def test_sub_400ms_pause_does_not_reject(self):
        rows = [bin_row() for _ in range(10)]
        rows[4] = bin_row(motion=0.0, light=-1.0)
        self.assertEqual(self.summarize(rows)["temporal_decision"], "pass")

    def test_multi_face_clip_passes_when_one_track_is_active(self):
        rows = [bin_row(light=1.2, competing=False) for _ in range(20)]
        self.assertEqual(self.summarize(rows)["temporal_decision"], "pass")

    def test_material_ambiguity_and_any_failure_go_to_manual(self):
        ambiguous = [bin_row(light=0.1, competing=True) for _ in range(5)]
        self.assertEqual(self.summarize(ambiguous)["temporal_decision"], "manual")
        failed = [bin_row(), bin_row(failure=True)]
        self.assertEqual(self.summarize(failed)["temporal_reason"], "inference_failure")

    def test_requested_laser_without_score_is_fail_closed(self):
        rows = [bin_row(light=-1.0, laser_requested=True) for _ in range(5)]
        result = self.summarize(rows)
        self.assertEqual((result["temporal_decision"], result["temporal_reason"]),
                         ("manual", "ambiguous"))

    def test_material_ambiguity_takes_priority_over_candidate_reject(self):
        rows = [bin_row(visible=False) for _ in range(5)]
        rows += [bin_row(light=0.1, laser_requested=True) for _ in range(8)]
        result = self.summarize(rows)
        self.assertEqual((result["temporal_decision"], result["temporal_reason"]),
                         ("manual", "ambiguous"))

    def test_laser_disagreement_is_manual(self):
        rows = [bin_row(light=-1.0, laser=0.9, disagreement=True) for _ in range(5)]
        result = self.summarize(rows)
        self.assertEqual((result["temporal_decision"], result["temporal_reason"]),
                         ("manual", "ambiguous"))


if __name__ == "__main__":
    unittest.main()
