"""Media-contract test cho ba generator KHÔNG đụng trục thời gian.

frame_reverse (đảo hình, audio copy), pitch_flatten (làm phẳng F0, video giữ nguyên),
anonymization (làm mờ mặt). Kiểm mỗi generator chỉ đổi đúng kênh của nó, frame/FPS/
duration khớp source; cộng builder manifest V2 và cổng metadata-shortcut.
"""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MEDIA = _load_module("fake_media_contract", "src/pipeline/fake_media_contract.py")
TIMELINE = _load_module("timeline_contract_non_temporal", "src/pipeline/timeline_contract.py")
REVERSE = _load_module("frame_reverse_v2", "src/pipeline/03_fake/02_frame_reverse.py")
PITCH = _load_module("pitch_flatten_v2", "src/pipeline/03_fake/03_pitch_flatten.py")
ANON = _load_module("anonymization_v2", "src/pipeline/03_fake/04_anonymization.py")
BUILDER = _load_module(
    "fake_manifest_v2_non_temporal",
    "src/pipeline/03_fake/06_build_fake_manifest_v2.py",
)
METADATA_GATE = _load_module(
    "metadata_shortcut_gate",
    "src/pipeline/03_fake/07_metadata_shortcut_gate.py",
)


def _run(args):
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr.decode("utf-8", errors="replace"))


class NonTemporalGeneratorContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise unittest.SkipTest("ffmpeg/ffprobe are required")
        cls.tmp = tempfile.TemporaryDirectory(prefix="non_temporal_v2_test_")
        cls.tmp_path = Path(cls.tmp.name)
        cls.wav_path = cls.tmp_path / "audio.wav"
        cls._write_audio(cls.wav_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def _write_audio(path):
        sample_rate = 48000
        duration = 2.0
        t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
        signal = 0.45 * np.sin(2 * np.pi * (180 * t + 35 * t * t))
        pcm = np.stack([signal, signal * 0.9], axis=1)
        pcm = (np.clip(pcm, -0.98, 0.98) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())

    def _source(self, fps):
        path = self.tmp_path / f"source_{fps.replace('/', '_')}.mp4"
        if not path.exists():
            _run([
                "ffmpeg", "-y", "-f", "lavfi", "-i",
                f"testsrc2=size=96x96:rate={fps}:duration=2",
                "-i", str(self.wav_path), "-t", "2",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                "-loglevel", "error", str(path),
            ])
        return path

    def test_frame_reverse_keeps_paired_media_contract(self):
        for fps in ("25/1", "30000/1001"):
            with self.subTest(fps=fps):
                source = self._source(fps)
                media = MEDIA.probe_media(str(source))
                output = self.tmp_path / f"reverse_{fps.replace('/', '_')}.mp4"
                self.assertTrue(REVERSE.make_reverse(
                    str(source), str(output), 8, 24, media
                ))
                self.assertTrue(MEDIA.is_valid_repaired_output(str(output), media))
                self.assertFalse(Path(str(output) + ".part.mp4").exists())

    def test_pitch_mux_keeps_video_and_audio_target(self):
        source = self._source("25/1")
        media = MEDIA.probe_media(str(source))
        output = self.tmp_path / "pitch.mp4"
        self.assertTrue(PITCH.mux(
            str(source), str(self.wav_path), str(output), media
        ))
        actual = MEDIA.probe_media(str(output))
        self.assertTrue(MEDIA.media_contract_matches(actual, media))
        self.assertEqual(actual["video_codec"], media["video_codec"])
        self.assertEqual(actual["audio_codec"], "alac")

    def test_anonymization_keeps_full_media_contract(self):
        source = self._source("25/1")
        media = MEDIA.probe_media(str(source))
        output = self.tmp_path / "anon.mp4"
        self.assertTrue(ANON.ffmpeg_anon(
            str(source), str(output), (16, 16, 64, 64), "blur", 12, media
        ))
        self.assertTrue(MEDIA.is_valid_repaired_output(str(output), media))

    def test_builder_accepts_only_four_v2_generator_versions(self):
        self.assertEqual(
            BUILDER.GENERATOR_VERSIONS["frame_reverse"],
            REVERSE.GENERATOR_VERSION,
        )
        self.assertEqual(
            BUILDER.GENERATOR_VERSIONS["pitch_flatten"],
            PITCH.GENERATOR_VERSION,
        )
        self.assertEqual(
            BUILDER.GENERATOR_VERSIONS["anonymization"],
            ANON.GENERATOR_VERSION,
        )
        media_path = str(self._source("25/1"))
        common = {
            "file_path": media_path,
            "label": "1",
            "source_clip": "source",
            "source_video": "video",
            "speaker_id": "speaker",
            "tier": "tier1",
        }
        temporal = [{
            **common,
            "clip_id": "temporal",
            "method": "temporal_desync",
            "param": f"generator={BUILDER.GENERATOR_VERSIONS['temporal_desync']}",
            **TIMELINE.build_timeline_contract(
                2.0, 2.0, boundary=TIMELINE.BOUNDARY_CIRCULAR_WRAP,
                audio_valid=(0.2, 2.0), visual_valid=(0.0, 1.8),
                manipulation_scope="global", manipulation=(0.2, 1.8),
            ),
        }]
        repaired = []
        for method in BUILDER.NON_TEMPORAL_METHODS:
            contract_args = {"manipulation_scope": "global"}
            if method == "frame_reverse":
                contract_args = {
                    "manipulation_scope": "local",
                    "manipulation": (0.4, 0.9),
                }
            repaired.append({
                **common,
                "clip_id": method,
                "method": method,
                "param": f"generator={BUILDER.GENERATOR_VERSIONS[method]}",
                **TIMELINE.build_timeline_contract(2.0, 2.0, **contract_args),
            })
        rows = BUILDER.compose_rows(repaired, temporal, check_files=True)
        self.assertEqual({row["method"] for row in rows}, set(BUILDER.METHOD_ORDER))
        legacy = [dict(row) for row in repaired]
        legacy[0]["param"] = "legacy"
        with self.assertRaisesRegex(ValueError, "không thuộc generator V2"):
            BUILDER.compose_rows(legacy, temporal, check_files=False)

    def test_metadata_gate_is_group_disjoint_and_detects_shortcut(self):
        methods = list(BUILDER.METHOD_ORDER)

        def rows(shortcut):
            output = []
            for group_index in range(10):
                source = f"source_{group_index}"
                base = {
                    feature: float(group_index + feature_index / 100)
                    for feature_index, feature in enumerate(METADATA_GATE.FEATURE_NAMES)
                }
                output.append({
                    **base, "clip_id": f"{source}_real",
                    "source_clip": source, "method": "real", "label": 0,
                })
                for method in methods:
                    fake = dict(base)
                    if shortcut:
                        fake["log_file_size"] += 100.0
                    output.append({
                        **fake, "clip_id": f"{source}_{method}",
                        "source_clip": source, "method": method, "label": 1,
                    })
            return output

        clean, _ = METADATA_GATE.evaluate_feature_rows(rows(False), threshold=0.65)
        leaked, _ = METADATA_GATE.evaluate_feature_rows(rows(True), threshold=0.65)
        self.assertAlmostEqual(clean["max_auc"], 0.5)
        self.assertTrue(clean["passed"])
        self.assertGreater(leaked["max_auc"], 0.99)
        self.assertFalse(leaked["passed"])


if __name__ == "__main__":
    unittest.main()
