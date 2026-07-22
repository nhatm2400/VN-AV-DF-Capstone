"""Contract tests for the sample-exact temporal-desync generator."""

import csv
from fractions import Fraction
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
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


TEMPORAL = _load_module(
    "temporal_desync",
    "src/pipeline/03_fake/01_temporal_desync.py",
)
SNVSM = _load_module(
    "snvsm_compress",
    "src/pipeline/03_fake/05_snvsm_compress.py",
)
FAKE_MANIFEST = _load_module(
    "fake_manifest_v2",
    "src/pipeline/03_fake/06_build_fake_manifest_v2.py",
)
BUILD_LABELS = _load_module(
    "build_labels",
    "src/pipeline/05_build_labels/01_build_labels.py",
)
EXTRACT_FEATURES = _load_module(
    "extract_features",
    "src/pipeline/04_extract_features/01_extract_features.py",
)
AVSP_DATASET = _load_module(
    "avsp_dataset",
    "src/train/dataset.py",
)


def _run(args):
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise AssertionError(f"Command failed ({proc.returncode}): {args}\n{stderr}")
    return proc.stdout


def _probe(path, count_frames=False):
    args = ["ffprobe", "-v", "error"]
    if count_frames:
        args.append("-count_frames")
    args += [
        "-show_entries",
        "stream=index,codec_type,codec_name,avg_frame_rate,time_base,start_time,"
        "duration,duration_ts,sample_rate,channels,nb_frames,nb_read_frames:"
        "format=start_time,duration",
        "-of", "json", str(path),
    ]
    return json.loads(_run(args))


def _stream(probe, kind):
    return next(s for s in probe["streams"] if s.get("codec_type") == kind)


def _video_packets(path):
    data = json.loads(_run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_packets", "-show_entries",
        "packet=pts,dts,duration,size,flags,data_hash",
        "-show_data_hash", "sha256", "-of", "json", str(path),
    ]))
    packets = data.get("packets", [])
    if not packets:
        raise AssertionError(f"No video packets: {path}")
    first_pts = int(packets[0].get("pts", 0))
    first_dts = int(packets[0].get("dts", 0))
    normalized = []
    for packet in packets:
        normalized.append({
            "pts": int(packet.get("pts", 0)) - first_pts,
            "dts": int(packet.get("dts", 0)) - first_dts,
            "duration": packet.get("duration"),
            "size": packet.get("size"),
            "flags": packet.get("flags"),
            "data_hash": packet.get("data_hash"),
        })
    return normalized


def _frame_md5(path):
    output = _run([
        "ffmpeg", "-v", "error", "-i", str(path),
        "-map", "0:v:0", "-an", "-f", "framemd5", "-",
    ]).decode("utf-8", errors="replace")
    return [line.strip() for line in output.splitlines()
            if line.strip() and not line.startswith("#")]


def _decode_stage04(path):
    raw = _run([
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-",
    ])
    return np.frombuffer(raw, dtype="<i2").astype(np.float64)


def _best_circular_shift(reference, candidate):
    n = min(len(reference), len(candidate))
    x = reference[:n] - np.mean(reference[:n])
    y = candidate[:n] - np.mean(candidate[:n])
    corr = np.fft.irfft(np.conj(np.fft.rfft(x)) * np.fft.rfft(y), n=n)
    shift = int(np.argmax(corr))
    if shift > n // 2:
        shift -= n
    return shift


def _pearson(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


class TemporalDesyncContractTest(unittest.TestCase):
    FPS_VALUES = ("24/1", "25/1", "2997/100", "30000/1001", "30/1")
    SHIFTS = (-15, -7, -3, 3, 7, 15)

    @classmethod
    def setUpClass(cls):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise unittest.SkipTest("ffmpeg/ffprobe are required")
        cls.tmp = tempfile.TemporaryDirectory(prefix="temporal_desync_test_")
        cls.tmp_path = Path(cls.tmp.name)
        cls.audio_path = cls.tmp_path / "audio.wav"
        cls._write_audio(cls.audio_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def _write_audio(path):
        sample_rate = 48000
        duration = 4.0
        t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
        rng = np.random.default_rng(20260721)
        phase1 = 2 * np.pi * (170 * t + 37 * t * t)
        phase2 = 2 * np.pi * (431 * t + 19 * t * t)
        envelope = 0.70 + 0.20 * np.sin(2 * np.pi * 0.37 * t)
        mono = envelope * (0.52 * np.sin(phase1) + 0.31 * np.sin(phase2))
        mono += 0.025 * rng.standard_normal(len(t))
        stereo = np.stack([mono, 0.93 * mono + 0.02 * np.sin(2 * np.pi * 83 * t)], axis=1)
        pcm = np.clip(stereo, -0.98, 0.98)
        pcm = (pcm * 32767).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())

    def _make_source(self, fps):
        name = fps.replace("/", "_")
        path = self.tmp_path / f"source_{name}.mp4"
        if path.exists():
            return path
        _run([
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"testsrc2=size=96x96:rate={fps}:duration=4",
            "-i", str(self.audio_path), "-t", "4",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", "-loglevel", "error", str(path),
        ])
        return path

    def _assert_case(self, source, fake, shift_frames):
        source_probe = _probe(source, count_frames=True)
        fake_probe = _probe(fake, count_frames=True)
        source_video = _stream(source_probe, "video")
        fake_video = _stream(fake_probe, "video")
        source_audio = _stream(source_probe, "audio")
        fake_audio = _stream(fake_probe, "audio")

        self.assertEqual(_video_packets(source), _video_packets(fake))
        self.assertEqual(_frame_md5(source), _frame_md5(fake))
        self.assertEqual(source_video.get("nb_read_frames"),
                         fake_video.get("nb_read_frames"))
        source_fps = float(Fraction(source_video["avg_frame_rate"]))
        fake_fps = float(Fraction(fake_video["avg_frame_rate"]))
        self.assertLessEqual(abs(source_fps - fake_fps), 1e-3)
        self.assertEqual(source_video["time_base"], fake_video["time_base"])
        self.assertLessEqual(abs(int(source_video["duration_ts"])
                                 - int(fake_video["duration_ts"])), 1)
        self.assertAlmostEqual(float(source_video.get("start_time", 0)),
                               float(fake_video.get("start_time", 0)), places=6)

        self.assertEqual(fake_audio["codec_name"], "alac")
        self.assertEqual(source_audio["sample_rate"], fake_audio["sample_rate"])
        self.assertEqual(source_audio["channels"], fake_audio["channels"])
        self.assertEqual(source_audio["duration_ts"], fake_audio["duration_ts"])
        self.assertAlmostEqual(float(fake_audio.get("start_time", 0)), 0.0, places=6)

        source_pcm = _decode_stage04(source)
        fake_pcm = _decode_stage04(fake)
        source_rate = int(source_audio["sample_rate"])
        aac_tolerance = math.ceil(1024 * 16000 / source_rate)
        self.assertLessEqual(abs(len(source_pcm) - len(fake_pcm)), aac_tolerance)

        fps_num, fps_den = (int(x) for x in source_video["avg_frame_rate"].split("/"))
        expected = round(shift_frames * fps_den / fps_num * 16000)
        measured = _best_circular_shift(source_pcm, fake_pcm)
        self.assertLessEqual(abs(measured - expected), 80,
                             msg=f"expected={expected}, measured={measured}")

        n = min(len(source_pcm), len(fake_pcm))
        rolled = np.roll(source_pcm[:n], measured)
        self.assertGreater(_pearson(rolled, fake_pcm[:n]), 0.95)

        frame = 160
        usable = (n // frame) * frame
        rms = np.sqrt(np.mean(fake_pcm[:usable].reshape(-1, frame) ** 2, axis=1))
        median_rms = float(np.median(rms))
        self.assertGreater(median_rms, 0)
        self.assertGreater(float(rms[0]), 0.02 * median_rms)
        self.assertGreater(float(rms[-1]), 0.02 * median_rms)

        seam = expected if expected > 0 else n + expected
        seam = max(1, min(n - 1, seam))
        derivative = np.abs(np.diff(fake_pcm[:n]))
        baseline = float(np.percentile(derivative, 99.9))
        seam_jump = float(derivative[seam - 1])
        self.assertLessEqual(seam_jump, max(1.0, 10.0 * baseline))

        return {
            "fps": source_video["avg_frame_rate"],
            "shift_frames": shift_frames,
            "expected_lag_ms": expected / 16.0,
            "measured_lag_ms": measured / 16.0,
            "lag_error_ms": abs(measured - expected) / 16.0,
            "video_frames": int(fake_video["nb_read_frames"]),
            "decoded_samples_source": len(source_pcm),
            "decoded_samples_fake": len(fake_pcm),
            "aligned_corr": _pearson(rolled, fake_pcm[:n]),
            "seam_jump_ratio": seam_jump / baseline if baseline else 0.0,
        }

    def test_sample_exact_shift_matrix(self):
        for fps in self.FPS_VALUES:
            source = self._make_source(fps)
            media = TEMPORAL.ffprobe_media(str(source))
            self.assertIsNotNone(media)
            for shift in self.SHIFTS:
                with self.subTest(fps=fps, shift=shift):
                    fake = self.tmp_path / f"fake_{fps.replace('/', '_')}_{shift:+d}.mp4"
                    ok, details = TEMPORAL.make_desync(
                        str(source), str(fake), shift, media
                    )
                    self.assertTrue(ok, details)
                    self.assertFalse(Path(str(fake) + ".part.mp4").exists())
                    self.assertEqual(details["boundary"], "circular_wrap")
                    shift_sec = abs(shift) / float(media["fps"])
                    duration = media["audio_duration"]
                    if shift > 0:
                        expected_ranges = (shift_sec, duration, 0.0,
                                           duration - shift_sec)
                    else:
                        expected_ranges = (0.0, duration - shift_sec,
                                           shift_sec, duration)
                    actual_ranges = (
                        details["audio_valid_start"],
                        details["audio_valid_end"],
                        details["visual_valid_start"],
                        details["visual_valid_end"],
                    )
                    for actual, expected in zip(actual_ranges, expected_ranges):
                        self.assertAlmostEqual(actual, expected, places=6)
                    self._assert_case(source, fake, shift)

    def test_v2_manifest_cannot_mix_with_temporal_v1(self):
        self.assertNotEqual(
            Path(TEMPORAL.DEFAULT_LABELS).as_posix(),
            "data/03_fake/labels.csv",
        )
        manifest = self.tmp_path / "legacy_labels.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TEMPORAL.LABEL_FIELDS)
            writer.writeheader()
            writer.writerow({
                "clip_id": "legacy_desync_7f",
                "file_path": "legacy.mp4",
                "label": 1,
                "method": "temporal_desync",
                "param": "shift=7f",
                "source_clip": "source",
                "source_video": "video",
                "speaker_id": "speaker",
                "tier": "tier1",
            })
        with self.assertRaisesRegex(ValueError, "temporal V1"):
            TEMPORAL.assert_manifest_compatible(str(manifest))

    def test_v2_cli_resume_keeps_manifest_unique(self):
        source = self._make_source("25/1")
        input_csv = self.tmp_path / "one_real.csv"
        labels = self.tmp_path / "temporal_v2.csv"
        out_dir = self.tmp_path / "temporal_v2"
        with input_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "clip_id", "file_path", "source_video", "speaker_id", "tier"
            ])
            writer.writeheader()
            writer.writerow({
                "clip_id": "source_one",
                "file_path": str(source),
                "source_video": "video_one",
                "speaker_id": "speaker_one",
                "tier": "tier1",
            })
        command = [
            sys.executable,
            str(ROOT / "src/pipeline/03_fake/01_temporal_desync.py"),
            "--input_csv", str(input_csv),
            "--out_dir", str(out_dir),
            "--labels", str(labels),
            "--seed", "42",
        ]
        _run(command)
        second = _run(command).decode("utf-8", errors="replace")
        with labels.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(len({row["clip_id"] for row in rows}), 1)
        self.assertIn(f"generator={TEMPORAL.GENERATOR_VERSION}", rows[0]["param"])
        self.assertIn("resume: 1", second)
        output = Path(rows[0]["file_path"])
        output.write_bytes(b"corrupt")
        _run(command)
        with labels.open(newline="", encoding="utf-8") as handle:
            repaired_rows = list(csv.DictReader(handle))
        self.assertEqual(len(repaired_rows), 1)
        self.assertTrue(TEMPORAL.is_valid_output(
            str(output), TEMPORAL.ffprobe_media(str(source))
        ))
        self.assertFalse(Path(str(output) + ".part.mp4").exists())

    def test_fake_manifest_replaces_v1_temporal(self):
        self.assertEqual(FAKE_MANIFEST.GENERATOR_VERSION,
                         TEMPORAL.GENERATOR_VERSION)
        media = str(self._make_source("25/1"))
        common = {
            "file_path": media,
            "label": "1",
            "source_clip": "source_one",
            "source_video": "video_one",
            "speaker_id": "speaker_one",
            "tier": "tier1",
        }
        legacy = []
        for method in ("temporal_desync", *FAKE_MANIFEST.NON_TEMPORAL_METHODS):
            legacy.append({
                **common,
                "clip_id": f"old_{method}",
                "method": method,
                "param": "shift=7f" if method == "temporal_desync" else "legacy",
            })
        temporal = [{
            **common,
            "clip_id": "new_temporal",
            "method": "temporal_desync",
            "param": (
                "boundary=circular_wrap;audio_valid_start=0.28;"
                "audio_valid_end=4.0;visual_valid_start=0.0;"
                "visual_valid_end=3.72;"
                f"generator={TEMPORAL.GENERATOR_VERSION}"
            ),
        }]
        rows = FAKE_MANIFEST.compose_rows(legacy, temporal, check_files=True)
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["method"] for row in rows},
                         set(FAKE_MANIFEST.METHOD_ORDER))
        self.assertIn("new_temporal", {row["clip_id"] for row in rows})
        self.assertNotIn("old_temporal_desync", {row["clip_id"] for row in rows})

    def test_snvsm_reencodes_audio_symmetrically(self):
        config = SNVSM.normalization_config("libx264", "ultrafast")
        self.assertNotEqual(
            config["config_id"],
            SNVSM.normalization_config("libx264", "veryfast")["config_id"],
        )
        self.assertNotEqual(
            config["config_id"],
            SNVSM.normalization_config(
                "libx264", "ultrafast", [23, 40], "random", 42
            )["config_id"],
        )
        self.assertNotEqual(
            config["config_id"],
            SNVSM.normalization_config(
                "libx264", "ultrafast", SNVSM.CRFS_DEFAULT, "random", 43
            )["config_id"],
        )
        with self.assertRaisesRegex(ValueError, "V1 bất biến"):
            SNVSM.assert_v2_destination(
                "data/03_fake/snvsm/real",
                "data/03_fake/snvsm/real_snvsm.csv",
            )
        with self.assertRaisesRegex(ValueError, "input manifest"):
            SNVSM.assert_v2_destination(
                str(self.tmp_path / "snvsm_v2"),
                str(self.tmp_path / "same.csv"),
                str(self.tmp_path / "same.csv"),
            )
        source = self._make_source("25/1")
        fake = self.tmp_path / "temporal_for_snvsm.mp4"
        ok, details = TEMPORAL.make_desync(str(source), str(fake), 7)
        self.assertTrue(ok, details)

        outputs = []
        for label, input_path in (("real", source), ("fake", fake)):
            out = self.tmp_path / f"snvsm_{label}.mp4"
            if label == "real":
                out.write_bytes(b"incomplete")
                self.assertFalse(SNVSM.is_valid_output(str(out)))
            self.assertTrue(SNVSM.compress(
                str(input_path), str(out), 40, "ultrafast", "libx264"
            ))
            self.assertTrue(SNVSM.is_valid_output(str(out)))
            self.assertFalse(Path(str(out) + ".part.mp4").exists())
            outputs.append(out)

        probes = [_probe(path) for path in outputs]
        videos = [_stream(probe, "video") for probe in probes]
        audios = [_stream(probe, "audio") for probe in probes]
        self.assertEqual([video["codec_name"] for video in videos], ["h264", "h264"])
        self.assertEqual([audio["codec_name"] for audio in audios], ["aac", "aac"])
        self.assertEqual(audios[0]["sample_rate"], audios[1]["sample_rate"])
        self.assertEqual(audios[0]["channels"], audios[1]["channels"])
        self.assertEqual([audio["sample_rate"] for audio in audios], ["16000", "16000"])
        self.assertEqual([audio["channels"] for audio in audios], [1, 1])
        target_samples = SNVSM.audio_target_samples(str(source))
        expected_video = SNVSM.video_contract(str(source))
        self.assertEqual([int(audio["duration_ts"]) for audio in audios],
                         [target_samples, target_samples])
        self.assertAlmostEqual(float(audios[0].get("start_time", 0)), 0.0, places=6)
        self.assertAlmostEqual(float(audios[1].get("start_time", 0)), 0.0, places=6)

        real_pcm, fake_pcm = (_decode_stage04(path) for path in outputs)
        self.assertLessEqual(abs(len(real_pcm) - len(fake_pcm)), 256)
        expected = round(7 / 25 * 16000)
        measured = _best_circular_shift(real_pcm, fake_pcm)
        self.assertLessEqual(abs(measured - expected), 80)
        n = min(len(real_pcm), len(fake_pcm))
        self.assertGreater(_pearson(real_pcm[:n - expected],
                                    fake_pcm[expected:n]), 0.98)
        frame = 160
        usable = (n // frame) * frame
        rms = np.sqrt(np.mean(fake_pcm[:usable].reshape(-1, frame) ** 2, axis=1))
        self.assertGreater(float(rms[0]), 1.0)
        self.assertGreater(float(rms[-1]), 1.0)

        truncated = self.tmp_path / "snvsm_truncated_video.mp4"
        _run([
            "ffmpeg", "-y", "-i", str(outputs[0]),
            "-filter_complex", "[0:v:0]trim=end=1,setpts=PTS-STARTPTS[v]",
            "-map", "[v]", "-map", "0:a:0", "-c:v", "libx264",
            "-preset", "ultrafast", "-c:a", "copy", "-loglevel", "error",
            str(truncated),
        ])
        self.assertFalse(SNVSM.is_valid_output(
            str(truncated), target_samples, expected_video
        ))

    def test_stage05_rejects_mismatched_snvsm_config(self):
        with self.assertRaisesRegex(ValueError, "Fake manifest trống"):
            BUILD_LABELS.validate_required_inputs([{"clip_id": "real"}], [])
        BUILD_LABELS.validate_required_inputs(
            [{"clip_id": "real"}], [], allow_real_only=True
        )
        with self.assertRaisesRegex(ValueError, "Thiếu file media"):
            BUILD_LABELS.validate_file_paths([
                {"clip_id": "missing", "file_path": str(self.tmp_path / "none.mp4")}
            ])
        labels_out = self.tmp_path / "immutable_labels.csv"
        BUILD_LABELS.write_rows_atomic(str(labels_out), [])
        with self.assertRaisesRegex(ValueError, "từ chối ghi đè"):
            BUILD_LABELS.write_rows_atomic(str(labels_out), [])
        self.assertIsNone(BUILD_LABELS.validate_snvsm_contract(
            [{"clip_id": "legacy_real"}], [{"clip_id": "legacy_fake"}]
        ))
        with self.assertRaisesRegex(ValueError, "thiếu provenance SNVSM"):
            BUILD_LABELS.validate_snvsm_contract(
                [{"clip_id": "partial_real", "snvsm_target_samples": "64000"}],
                [{"clip_id": "partial_fake"}],
            )
        config = SNVSM.normalization_config(
            "libx264", "ultrafast", [23, 30, 35, 40], "random", 42
        )
        signature = {
            "snvsm_version": SNVSM.SNVSM_VERSION,
            "snvsm_config_id": config["config_id"],
            "snvsm_encoder": "libx264",
            "snvsm_preset": "ultrafast",
            "snvsm_audio": "aac_128k_16khz_mono",
            "snvsm_sample_rate": "16000",
            "snvsm_channels": "1",
            "snvsm_target_samples": "64000",
            "snvsm_mode": "random",
            "snvsm_crf_set": "23,30,35,40",
            "snvsm_seed": "42",
            "snvsm_pair_key": "source",
            "snvsm_video_frames": "100",
            "snvsm_video_fps": "25",
            "snvsm_video_duration_s": "4.000000000",
            "crf": "23",
        }
        real = [{"clip_id": "real", "orig_clip_id": "source", **signature}]
        fake = [
            {"clip_id": f"fake_{method}", "source_clip": "source",
             "method": method, **signature}
            for method in sorted(BUILD_LABELS.EXPECTED_FAKE_METHODS)
        ]
        contract = BUILD_LABELS.validate_snvsm_contract(real, fake)
        self.assertEqual(contract["snvsm_config_id"], config["config_id"])
        fake[0]["snvsm_config_id"] = "config_b"
        with self.assertRaisesRegex(ValueError, "contract SNVSM invalid"):
            BUILD_LABELS.validate_snvsm_contract(real, fake)
        fake[0]["snvsm_config_id"] = config["config_id"]
        fake[0]["snvsm_target_samples"] = ""
        with self.assertRaisesRegex(ValueError, "contract SNVSM invalid"):
            BUILD_LABELS.validate_snvsm_contract(real, fake)
        fake[0]["snvsm_target_samples"] = "64000"
        self.assertEqual(
            BUILD_LABELS.validate_snvsm_pair_targets(real, fake), 4
        )
        fake[0]["snvsm_target_samples"] = "63999"
        with self.assertRaisesRegex(ValueError, "target lệch real-fake"):
            BUILD_LABELS.validate_snvsm_pair_targets(real, fake)
        fake[0]["snvsm_target_samples"] = "64000"
        fake[0]["snvsm_video_frames"] = "99"
        with self.assertRaisesRegex(ValueError, "video contract lệch"):
            BUILD_LABELS.validate_snvsm_pair_targets(real, fake)
        fake[0]["snvsm_video_frames"] = "100"
        fake[0]["crf"] = "40"
        with self.assertRaisesRegex(ValueError, "CRF policy lệch"):
            BUILD_LABELS.validate_snvsm_pair_targets(real, fake)
        fake[0]["crf"] = "23"
        fake[0]["snvsm_version"] = "bogus"
        with self.assertRaisesRegex(ValueError, "contract SNVSM invalid"):
            BUILD_LABELS.validate_snvsm_contract(real, fake)
        fake[0]["snvsm_version"] = SNVSM.SNVSM_VERSION
        with self.assertRaisesRegex(ValueError, "thiếu/thừa method"):
            BUILD_LABELS.validate_snvsm_pair_targets(real, fake[:-1])
        random_real = real + [{**real[0], "clip_id": "real_crf30", "crf": "30"}]
        random_fake = fake + [
            {**row, "clip_id": row["clip_id"] + "_crf30", "crf": "30"}
            for row in fake
        ]
        with self.assertRaisesRegex(ValueError, "mode=random"):
            BUILD_LABELS.validate_snvsm_pair_targets(random_real, random_fake)

    def test_snvsm_preserves_both_extreme_lag_directions(self):
        source = self._make_source("25/1")
        normalized_real = self.tmp_path / "snvsm_extreme_real.mp4"
        self.assertTrue(SNVSM.compress(
            str(source), str(normalized_real), 40, "ultrafast", "libx264"
        ))
        real_pcm = _decode_stage04(normalized_real)
        for shift in (-15, 15):
            with self.subTest(shift=shift):
                temporal = self.tmp_path / f"temporal_extreme_{shift:+d}.mp4"
                normalized_fake = self.tmp_path / f"snvsm_extreme_{shift:+d}.mp4"
                ok, details = TEMPORAL.make_desync(
                    str(source), str(temporal), shift
                )
                self.assertTrue(ok, details)
                self.assertTrue(SNVSM.compress(
                    str(temporal), str(normalized_fake), 40, "ultrafast", "libx264"
                ))
                fake_pcm = _decode_stage04(normalized_fake)
                n = min(len(real_pcm), len(fake_pcm))
                expected = round(shift / 25 * 16000)
                measured = _best_circular_shift(real_pcm, fake_pcm)
                self.assertLessEqual(abs(measured - expected), 80)
                if expected > 0:
                    real_valid = real_pcm[:n - expected]
                    fake_valid = fake_pcm[expected:n]
                else:
                    delta = -expected
                    real_valid = real_pcm[delta:n]
                    fake_valid = fake_pcm[:n - delta]
                self.assertGreater(_pearson(real_valid, fake_valid), 0.98)

    def test_snvsm_normalizes_mixed_audio_formats(self):
        source = self._make_source("25/1")
        pitch_like = self.tmp_path / "pitch_like_16k_mono.mp4"
        _run([
            "ffmpeg", "-y", "-i", str(source), "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", "copy", "-c:a", "aac", "-ar", "16000", "-ac", "1",
            "-loglevel", "error", str(pitch_like),
        ])
        outputs = []
        for name, input_path in (("stereo_44k", source), ("mono_16k", pitch_like)):
            output = self.tmp_path / f"normalized_{name}.mp4"
            target_samples = SNVSM.audio_target_samples(str(input_path))
            self.assertTrue(SNVSM.compress(
                str(input_path), str(output), 40, "ultrafast", "libx264"
            ))
            self.assertTrue(SNVSM.is_valid_output(
                str(output), target_samples, SNVSM.video_contract(str(input_path))
            ))
            wav = self.tmp_path / f"stage04_{name}.wav"
            self.assertTrue(EXTRACT_FEATURES.extract_wav(
                str(output), str(wav), target_samples
            ))
            self.assertEqual(
                len(EXTRACT_FEATURES.read_wav_int16(str(wav))), target_samples
            )
            outputs.append(output)
        audio_formats = []
        for output in outputs:
            audio = _stream(_probe(output), "audio")
            audio_formats.append((audio["codec_name"], audio["sample_rate"],
                                  audio["channels"]))
        self.assertEqual(audio_formats, [("aac", "16000", 1), ("aac", "16000", 1)])

    def test_stage04_requires_target_samples_for_snvsm_rows(self):
        feature_cfg = EXTRACT_FEATURES.feature_config_id(
            25.0, 96, 2, 0.25, "face_hash", False
        )
        self.assertNotEqual(feature_cfg, EXTRACT_FEATURES.feature_config_id(
            25.0, 96, 2, 0.25, "face_hash", True
        ))
        self.assertNotEqual(feature_cfg, EXTRACT_FEATURES.feature_config_id(
            25.0, 112, 2, 0.25, "face_hash", False
        ))
        self.assertNotEqual(feature_cfg, EXTRACT_FEATURES.feature_config_id(
            25.0, 96, 2, 0.25, "face_hash", False, True
        ))
        self.assertIsNone(EXTRACT_FEATURES.snvsm_target_samples({}))
        with self.assertRaisesRegex(ValueError, "snvsm_target_samples_missing"):
            EXTRACT_FEATURES.snvsm_target_samples({"snvsm_version": "v2"})
        with self.assertRaisesRegex(ValueError, "snvsm_target_samples_missing"):
            EXTRACT_FEATURES.snvsm_target_samples({"snvsm_config_id": "config"})
        with self.assertRaisesRegex(ValueError, "snvsm_target_samples_invalid"):
            EXTRACT_FEATURES.snvsm_target_samples({
                "snvsm_version": "v2", "snvsm_target_samples": "0"
            })
        with self.assertRaisesRegex(ValueError, "snvsm_contract_invalid"):
            EXTRACT_FEATURES.snvsm_target_samples({
                "snvsm_version": EXTRACT_FEATURES.EXPECTED_SNVSM_VERSION,
                "snvsm_target_samples": "16000",
            })
        valid_row = {
            "snvsm_version": EXTRACT_FEATURES.EXPECTED_SNVSM_VERSION,
            "snvsm_config_id": SNVSM.normalization_config(
                "libx264", "ultrafast", [23, 30, 35, 40], "random", 42
            )["config_id"],
            "snvsm_audio": EXTRACT_FEATURES.EXPECTED_SNVSM_AUDIO,
            "snvsm_encoder": "libx264",
            "snvsm_preset": "ultrafast",
            "snvsm_sample_rate": "16000",
            "snvsm_channels": "1",
            "snvsm_target_samples": "16000",
            "snvsm_mode": "random",
            "snvsm_crf_set": "23,30,35,40",
            "snvsm_seed": "42",
            "snvsm_pair_key": "source",
            "snvsm_video_frames": "100",
            "snvsm_video_fps": "30000/1001",
            "snvsm_video_duration_s": "3.336667",
            "crf": "30",
        }
        self.assertEqual(
            EXTRACT_FEATURES.snvsm_target_samples(valid_row), 16000
        )
        valid_row["snvsm_video_fps"] = "30000/0"
        with self.assertRaisesRegex(ValueError, "snvsm_contract_invalid"):
            EXTRACT_FEATURES.snvsm_target_samples(valid_row)
        stale_wav = self.tmp_path / "stale_stage04.wav"
        stale_wav.write_bytes(b"stale")
        self.assertFalse(EXTRACT_FEATURES.extract_wav(
            str(self.tmp_path / "missing.mp4"), str(stale_wav), 16000
        ))
        self.assertFalse(stale_wav.exists())
        feature = self.tmp_path / "feature_contract.pt"
        expected_feature = {
            "clip_id": "feature_contract",
            "label": 1,
            "method": "temporal_desync",
            "src": str(self.tmp_path / "source.mp4"),
            "mouth_size": 96,
            "feature_config_id": "feature_cfg",
            "snvsm_config_id": "snvsm_cfg",
            "require_wave": False,
        }
        EXTRACT_FEATURES.torch.save({
            "meta": {"audio_target_samples": 16000, "audio_samples": 16000}
        }, feature)
        self.assertFalse(EXTRACT_FEATURES.is_valid_existing_feature(
            str(feature), 16000, expected_feature
        ))
        EXTRACT_FEATURES.torch.save({
            "clip_id": "feature_contract",
            "label": 1,
            "method": "temporal_desync",
            "mouth": EXTRACT_FEATURES.torch.zeros((2, 96, 96), dtype=EXTRACT_FEATURES.torch.uint8),
            "w2v": EXTRACT_FEATURES.torch.zeros((2, 768), dtype=EXTRACT_FEATURES.torch.float16),
            "prosody": EXTRACT_FEATURES.torch.zeros((2, 4)),
            "meta": {
                "src": str(self.tmp_path / "source.mp4"),
                "feature_schema_version": EXTRACT_FEATURES.FEATURE_SCHEMA_VERSION,
                "feature_config_id": "feature_cfg",
                "snvsm_config_id": "snvsm_cfg",
                "audio_target_samples": 16000,
                "audio_samples": 16000,
            },
        }, feature)
        self.assertTrue(EXTRACT_FEATURES.is_valid_existing_feature(
            str(feature), 16000, expected_feature
        ))
        self.assertFalse(EXTRACT_FEATURES.is_valid_existing_feature(
            str(feature), 15999, expected_feature
        ))
        wrong_config = {**expected_feature, "feature_config_id": "other"}
        self.assertFalse(EXTRACT_FEATURES.is_valid_existing_feature(
            str(feature), 16000, wrong_config
        ))
        feature_no_w2v = self.tmp_path / "feature_no_w2v.pt"
        EXTRACT_FEATURES.torch.save({
            "clip_id": "feature_contract",
            "label": 1,
            "method": "temporal_desync",
            "mouth": EXTRACT_FEATURES.torch.zeros((2, 96, 96), dtype=EXTRACT_FEATURES.torch.uint8),
            "w2v": None,
            "prosody": EXTRACT_FEATURES.torch.zeros((2, 4)),
            "meta": {
                "src": str(self.tmp_path / "source.mp4"),
                "feature_schema_version": EXTRACT_FEATURES.FEATURE_SCHEMA_VERSION,
                "feature_config_id": "feature_cfg",
                "snvsm_config_id": "snvsm_cfg",
                "audio_target_samples": 16000,
                "audio_samples": 16000,
            },
        }, feature_no_w2v)
        self.assertFalse(EXTRACT_FEATURES.is_valid_existing_feature(
            str(feature_no_w2v), 16000, expected_feature
        ))
        self.assertTrue(EXTRACT_FEATURES.is_valid_existing_feature(
            str(feature_no_w2v), 16000, expected_feature, require_w2v=False
        ))

    def test_dataset_fails_closed_on_missing_or_incomplete_features(self):
        labels = self.tmp_path / "labels_features.csv"
        fields = ["clip_id", "split", "label", "method", "param"]
        with labels.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({"clip_id": "missing", "split": "train",
                             "label": "0", "method": "real", "param": ""})
        with self.assertRaisesRegex(RuntimeError, "Từ chối drop âm thầm"):
            AVSP_DATASET.AVSPDataset(
                str(labels), str(self.tmp_path), "train", branches=()
            )

        incomplete = self.tmp_path / "incomplete.pt"
        EXTRACT_FEATURES.torch.save({"clip_id": "incomplete", "label": 0,
                                     "method": "real", "mouth": None,
                                     "w2v": None, "prosody": None}, incomplete)
        with labels.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({"clip_id": "incomplete", "split": "train",
                             "label": "0", "method": "real", "param": ""})
        dataset = AVSP_DATASET.AVSPDataset(
            str(labels), str(self.tmp_path), "train", branches=("audio",)
        )
        with self.assertRaisesRegex(RuntimeError, "thiếu nhánh audio"):
            dataset[0]

    @unittest.skipUnless(os.environ.get("RUN_REAL_AV_AUDIT") == "1",
                         "set RUN_REAL_AV_AUDIT=1 for local real-data smoke")
    def test_real_data_smoke(self):
        manifest = Path(os.environ.get(
            "REAL_AV_AUDIT_CSV", ROOT / "data/02_curate/all_clean.csv"
        ))
        by_tier = {}
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                path = Path(row.get("file_path", ""))
                tier = row.get("tier", "")
                if tier not in by_tier and path.is_file():
                    by_tier[tier] = row
        rows = list(by_tier.values())[:3]
        self.assertGreaterEqual(len(rows), 3, "Need one real clip from each tier")

        report = []
        for row in rows:
            source = Path(row["file_path"])
            media = TEMPORAL.ffprobe_media(str(source))
            self.assertIsNotNone(media)
            for shift in self.SHIFTS:
                with self.subTest(clip_id=row["clip_id"], shift=shift):
                    fake = self.tmp_path / f"real_{row['clip_id']}_{shift:+d}.mp4"
                    ok, details = TEMPORAL.make_desync(
                        str(source), str(fake), shift, media
                    )
                    self.assertTrue(ok, details)
                    record = self._assert_case(source, fake, shift)
                    record.update({"clip_id": row["clip_id"], "tier": row["tier"]})
                    report.append(record)

        report_path = os.environ.get("REAL_AV_AUDIT_REPORT")
        if report_path:
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(report_path).write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    unittest.main()
