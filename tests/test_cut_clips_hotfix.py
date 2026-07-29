"""Contract test cho hotfix Stage 04 Cut Clips."""

import csv
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "src/pipeline/01_collect/cut_clips_core.py"
SPEC = importlib.util.spec_from_file_location("cut_clips_core_test", CORE_PATH)
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)
PREP_PATH = ROOT / "src/pipeline/02_curate/01_prep_manifest.py"
PREP_SPEC = importlib.util.spec_from_file_location(
    "prep_manifest_test", PREP_PATH
)
PREP = importlib.util.module_from_spec(PREP_SPEC)
sys.modules[PREP_SPEC.name] = PREP
PREP_SPEC.loader.exec_module(PREP)


def result(returncode=0, stdout=b"", stderr=b""):
    return SimpleNamespace(
        returncode=returncode, stdout=stdout, stderr=stderr
    )


def config(**overrides):
    values = {
        "tier": "tier1",
        "dataset_dir": "dataset",
        "input_csv": "input.csv",
        "output_root": "output",
        "run_id": "test_run",
        "expected_input_count": 1,
        "frame_size": 2,
        "sample_fps": 1.0,
    }
    values.update(overrides)
    return CORE.CutConfig(**values)


class CutClipsHotfixTest(unittest.TestCase):
    def test_tier_configs_lock_exact_repo_manifest_counts(self):
        expected = {"tier1": 472, "tier2": 292, "tier3": 2274}
        for tier, expected_count in expected.items():
            config_path = (
                ROOT / "src/pipeline/01_collect/configs" / f"{tier}.json"
            )
            cfg = CORE.CutConfig.from_json(config_path)
            manifest_path = (
                ROOT / "data/01_collect" / f"{tier}_quality_gate_passed.csv"
            )
            with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            filenames = [row.get("filename", "").strip() for row in rows]
            self.assertEqual(cfg.expected_input_count, expected_count)
            self.assertEqual(len(rows), expected_count)
            self.assertTrue(all(filenames))
            self.assertEqual(len(set(filenames)), expected_count)

    def test_stable_clip_id_does_not_depend_on_order(self):
        windows = [(9.125, 14.125), (30.5, 35.5), (1.0, 4.0)]
        forward = {
            window: CORE.stable_clip_id("source-A", *window)
            for window in windows
        }
        reverse = {
            window: CORE.stable_clip_id("source-A", *window)
            for window in reversed(windows)
        }
        self.assertEqual(forward, reverse)
        self.assertEqual(
            forward[(9.125, 14.125)],
            "source-A_s0000009125_e0000014125",
        )

    def test_decode_cuda_failure_falls_back_to_cpu(self):
        raw = bytes(range(24))  # 2 frame × 2 × 2 × 3 byte
        commands = []

        def runner(command):
            commands.append(command)
            if "-hwaccel" in command:
                return result(1, b"", b"cuda decoder unsupported")
            return result(0, raw, b"")

        decoded = CORE.decode_frames_with_fallback(
            "video.mp4", 0, config(), runner=runner
        )
        self.assertEqual(decoded.backend, "cpu")
        self.assertEqual(len(decoded.frames), 2)
        self.assertEqual(
            [attempt.backend for attempt in decoded.attempts],
            ["cuda", "cpu"],
        )
        self.assertIn("unsupported", decoded.attempts[0].stderr)
        self.assertEqual(len(commands), 2)

    def test_decode_rejects_success_code_with_malformed_bytes(self):
        raw = bytes(range(12))
        calls = 0

        def runner(_command):
            nonlocal calls
            calls += 1
            return result(0, b"broken" if calls == 1 else raw, b"")

        decoded = CORE.decode_frames_with_fallback(
            "video.mp4", 0, config(), runner=runner
        )
        self.assertEqual(decoded.backend, "cpu")
        self.assertFalse(decoded.attempts[0].valid)
        self.assertTrue(decoded.attempts[1].valid)

    def test_face_detection_uses_cpu_when_cuda_is_unavailable(self):
        calls = []

        class FaceModel:
            def predict(self, frames, **kwargs):
                calls.append(kwargs)
                return [SimpleNamespace(boxes=[]) for _ in frames]

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        )
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            flags = CORE.detect_faces_batched(
                FaceModel(), [object()], gpu_id=0, batch=1
            )
        self.assertEqual(flags, [False])
        self.assertEqual(calls[0]["device"], "cpu")
        self.assertFalse(calls[0]["half"])

    def test_cut_nvenc_failure_falls_back_and_publishes_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "clip.mp4"
            backends = []

            def runner(command):
                backend = "nvenc" if "h264_nvenc" in command else "libx264"
                backends.append(backend)
                if backend == "nvenc":
                    return result(1, b"", b"nvenc unavailable")
                Path(command[-1]).write_bytes(b"valid")
                return result()

            cut = CORE.cut_clip_with_fallback(
                "source.mp4",
                target,
                1.0,
                3.0,
                0,
                config(),
                runner=runner,
                validator=lambda _path: (True, ""),
            )
            self.assertTrue(cut.ok)
            self.assertEqual(cut.backend, "libx264")
            self.assertEqual(backends, ["nvenc", "libx264"])
            self.assertEqual(target.read_bytes(), b"valid")
            self.assertEqual(list(Path(tmp).glob("*.partial.mp4")), [])

    def test_cut_both_fail_leaves_no_output_or_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "clip.mp4"
            cut = CORE.cut_clip_with_fallback(
                "source.mp4",
                target,
                1.0,
                3.0,
                0,
                config(),
                runner=lambda _command: result(1, b"", b"failed"),
                validator=lambda _path: (False, "invalid"),
            )
            self.assertFalse(cut.ok)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(tmp).glob("*.partial.mp4")), [])

    def test_batch_coverage_requires_exact_input_status_and_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "clip.mp4"
            media.write_bytes(b"x")
            accepted = [{
                "clip_id": "clip",
                "source_video": "video",
                "file_path": str(media),
            }]
            statuses = [{
                "filename": "video.mp4",
                "video_id": "video",
                "accepted_count": 1,
            }]
            CORE.validate_batch_coverage(
                ["video.mp4"], statuses, accepted, [media]
            )
            with self.assertRaisesRegex(ValueError, "coverage"):
                CORE.validate_batch_coverage(
                    ["video.mp4", "missing.mp4"],
                    statuses,
                    accepted,
                    [media],
                )

    def test_input_count_is_locked_to_stage03_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dataset = tmp / "dataset"
            dataset.mkdir()
            (dataset / "one.mp4").write_bytes(b"x")
            manifest = tmp / "input.csv"
            manifest.write_text("filename\none.mp4\n", encoding="utf-8")
            cfg = config(
                dataset_dir=str(dataset),
                input_csv=str(manifest),
                expected_input_count=2,
            )
            with self.assertRaisesRegex(ValueError, "input count"):
                CORE._load_input(cfg)

    def test_input_source_stems_must_be_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "media"
            media.mkdir()
            (media / "same.mp4").write_bytes(b"x")
            (media / "same.mkv").write_bytes(b"x")
            manifest = root / "input.csv"
            manifest.write_text(
                "filename\nsame.mp4\nsame.mkv\n",
                encoding="utf-8",
            )
            cfg = CORE.CutConfig(
                tier="tier1",
                dataset_dir=str(media),
                input_csv=str(manifest),
                output_root=str(root / "out"),
                run_id="test",
                expected_input_count=2,
            )
            with self.assertRaisesRegex(ValueError, "source stem"):
                CORE._load_input(cfg)

    def test_prep_manifest_uses_accepted_csv_as_source_of_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            media = tmp / "media"
            media.mkdir()
            (media / "clip-a.mp4").write_bytes(b"x")
            accepted = tmp / "accepted_clips.csv"
            accepted.write_text(
                "clip_id,source_video,start_time,end_time,duration,"
                "face_ratio,speech_ratio,snr,file_path,run_id\n"
                "clip-a,source,1.0,4.0,3.0,1.0,1.0,10.0,"
                "/kaggle/old/clip-a.mp4,run\n",
                encoding="utf-8",
            )
            rows = PREP.prep_tier(
                "tier1", str(accepted), str(media), verify=False
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["clip_id"], "clip-a")
            self.assertEqual(rows[0]["has_cut_meta"], 1)
            self.assertEqual(
                Path(rows[0]["file_path"]).resolve(),
                (media / "clip-a.mp4").resolve(),
            )

    def test_prep_manifest_rejects_orphan_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            media = tmp / "media"
            media.mkdir()
            (media / "clip-a.mp4").write_bytes(b"x")
            (media / "orphan.mp4").write_bytes(b"x")
            accepted = tmp / "accepted_clips.csv"
            accepted.write_text(
                "clip_id,source_video\nclip-a,source\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "không 1-1"):
                PREP.prep_tier(
                    "tier1", str(accepted), str(media), verify=False
                )

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg/ffprobe are required",
    )
    def test_real_ffmpeg_cpu_decode_and_cut_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "source.mp4"
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "lavfi", "-i",
                "testsrc2=size=64x64:rate=25:duration=2",
                "-f", "lavfi", "-i",
                "sine=frequency=440:sample_rate=16000:duration=2",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-loglevel", "error", str(source),
            ], check=True, capture_output=True)
            cfg = config(
                use_hwaccel_decode=False,
                use_nvenc=False,
                frame_size=32,
            )
            decoded = CORE.decode_frames_with_fallback(
                source, 0, cfg
            )
            self.assertEqual(decoded.backend, "cpu")
            self.assertGreaterEqual(len(decoded.frames), 1)
            target = tmp / "cut.mp4"
            cut = CORE.cut_clip_with_fallback(
                source, target, 0.25, 1.75, 0, cfg
            )
            self.assertTrue(cut.ok)
            valid, detail = CORE.probe_media(target)
            self.assertTrue(valid, detail)


if __name__ == "__main__":
    unittest.main()
