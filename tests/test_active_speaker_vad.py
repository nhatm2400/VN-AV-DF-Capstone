import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CURATE_DIR = ROOT / "src/pipeline/02_curate/02_scoring/02_active_speaker"
MODULE_PATH = CURATE_DIR / "01_score.py"


def load_module():
    sys.path.insert(0, str(CURATE_DIR))
    try:
        spec = importlib.util.spec_from_file_location("active_speaker_score", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(CURATE_DIR))


class ActiveSpeakerVADTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_int16_pcm_is_normalized_for_silero(self):
        pcm = np.array([-32768, 0, 32767], dtype=np.int16)
        normalized = self.module.normalize_audio(pcm)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertAlmostEqual(float(normalized[0]), -1.0)
        self.assertLessEqual(float(normalized[-1]), 1.0)

    def test_short_probability_burst_is_not_speech(self):
        probabilities = [0.0] * 4 + [0.9] * 4 + [0.0] * 5
        samples = len(probabilities) * self.module.SileroVAD.WINDOW
        spans = self.module.SileroVAD.speech_timestamps(probabilities, samples)
        self.assertEqual(spans, [])

    def test_speech_span_uses_hysteresis_and_padding(self):
        probabilities = [0.0] * 2 + [0.9] * 10 + [0.1] * 5
        samples = len(probabilities) * self.module.SileroVAD.WINDOW
        spans = self.module.SileroVAD.speech_timestamps(probabilities, samples)
        self.assertEqual(len(spans), 1)
        self.assertLess(spans[0]["start"], 2 * self.module.SileroVAD.WINDOW)
        self.assertGreater(spans[0]["end"], 12 * self.module.SileroVAD.WINDOW)

    def test_face_then_broll_is_voiceover_not_inference_failure(self):
        frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(125)]
        tracked_frames = np.arange(75)
        track = {
            "track_id": 0,
            "frames": tracked_frames,
            "bbox": np.tile(np.array([[2, 2, 28, 28]], dtype=np.float32), (75, 1)),
            "kps": np.zeros((75, 5, 2), dtype=np.float32),
        }

        class Tracker:
            def track(self, _frames):
                return [track]

        class Light:
            def score(self, _audio, faces, _start):
                return np.ones(len(faces), dtype=np.float32)

        class VAD:
            def bins(self, _audio, count, _bin_ms):
                return [True] * count

        face = np.zeros((112, 112), dtype=np.uint8)
        mouth = np.zeros((38, 62), dtype=np.uint8)
        with mock.patch.object(self.module, "decode_25fps", return_value=frames), \
             mock.patch.object(self.module, "extract_audio",
                               return_value=np.zeros(5 * 16000, dtype=np.float32)), \
             mock.patch.object(self.module, "aligned_face_and_mouth",
                               return_value=(face, mouth)):
            summary, timeline = self.module.score_clip(
                "clip", "unused.mp4", Tracker(), Light(), VAD(), {},
                self.module.TemporalPolicy(mouth_freeze_threshold=-1.0),
            )

        tail = timeline[15:]
        self.assertTrue(all(not row["face_visible"] for row in tail))
        self.assertTrue(all(not row["inference_failure"] for row in tail))
        self.assertEqual((summary["temporal_decision"], summary["temporal_reason"]),
                         ("reject", "voiceover"))

    def test_five_frame_nuisance_track_does_not_fail_the_whole_clip(self):
        frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(10)]

        def track(track_id, count):
            return {
                "track_id": track_id,
                "frames": np.arange(count),
                "bbox": np.tile(
                    np.array([[2, 2, 28, 28]], dtype=np.float32), (count, 1)
                ),
                "kps": np.zeros((count, 5, 2), dtype=np.float32),
            }

        class Tracker:
            def track(self, _frames):
                return [track(0, 10), track(1, 5)]

        class Light:
            calls = []

            def score(self, _audio, faces, _start):
                self.calls.append(len(faces))
                if len(faces) < 6:
                    raise RuntimeError("light_asd_track_too_short")
                return np.ones(len(faces), dtype=np.float32)

        class VAD:
            def bins(self, _audio, count, _bin_ms):
                return [True] * count

        light = Light()
        face = np.zeros((112, 112), dtype=np.uint8)
        mouth = np.zeros((38, 62), dtype=np.uint8)
        with mock.patch.object(self.module, "decode_25fps", return_value=frames), \
             mock.patch.object(self.module, "extract_audio",
                               return_value=np.zeros(6400, dtype=np.float32)), \
             mock.patch.object(self.module, "aligned_face_and_mouth",
                               return_value=(face, mouth)):
            _, timeline = self.module.score_clip(
                "clip", "unused.mp4", Tracker(), light, VAD(), {},
                self.module.TemporalPolicy(),
            )

        self.assertEqual(light.calls, [10])
        self.assertTrue(all(not row["inference_failure"] for row in timeline))


if __name__ == "__main__":
    unittest.main()
