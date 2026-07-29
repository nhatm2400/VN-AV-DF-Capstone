"""RNG của generator phải phụ thuộc clip, không phụ thuộc thứ tự xử lý.

Trước khi sửa, cả `01_temporal_desync.py` lẫn `02_frame_reverse.py` gọi
`random.seed(args.seed)` một lần rồi rút số trong vòng lặp. Hệ quả: tham số fake
của một clip phụ thuộc VỊ TRÍ của nó trong manifest, nên `--limit`, resume, hay
chỉ đơn giản là sắp lại manifest đều cho ra tập fake khác — không tái lập được.

Test chạy mỗi generator hai lần trên cùng tập clip nhưng THỨ TỰ ĐẢO NGƯỢC, rồi
đối chiếu `fake_id` sinh ra cho từng clip. `fake_id` mã hóa chính tham số ngẫu
nhiên (mức lệch frame; cửa sổ đảo start-end) nên so `fake_id` là đủ.
"""

import csv
import importlib.util
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DESYNC = ROOT / "src/pipeline/03_fake/01_temporal_desync.py"
REVERSE = ROOT / "src/pipeline/03_fake/02_frame_reverse.py"

N_CLIPS = 6


def _run(args):
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr.decode("utf-8", errors="replace"))


class GeneratorRngDeterminismTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise unittest.SkipTest("ffmpeg/ffprobe are required")
        cls.tmp = tempfile.TemporaryDirectory(prefix="rng_determinism_test_")
        cls.tmp_path = Path(cls.tmp.name)
        wav = cls.tmp_path / "audio.wav"
        cls._write_audio(wav)
        cls.clips = []
        for i in range(N_CLIPS):
            path = cls.tmp_path / f"clip{i:03d}.mp4"
            _run([
                "ffmpeg", "-y", "-f", "lavfi", "-i",
                f"testsrc2=size=96x96:rate=25:duration=4",
                "-i", str(wav), "-t", "4",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart", "-loglevel", "error", str(path),
            ])
            cls.clips.append({"clip_id": f"clip{i:03d}", "file_path": str(path)})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def _write_audio(path):
        sample_rate = 48000
        t = np.arange(int(sample_rate * 4.0), dtype=np.float64) / sample_rate
        mono = 0.5 * np.sin(2 * np.pi * (180 * t + 23 * t * t))
        pcm = (np.clip(np.stack([mono, mono], axis=1), -0.98, 0.98) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())

    def _write_manifest(self, path, rows):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["clip_id", "file_path"])
            writer.writeheader()
            writer.writerows(rows)

    def _fake_ids_by_clip(self, script, rows, tag):
        """Chạy generator trên `rows`, trả {clip_id: [fake_id,...]}."""
        run_dir = self.tmp_path / tag
        run_dir.mkdir()
        manifest = run_dir / "input.csv"
        self._write_manifest(manifest, rows)
        labels = run_dir / "labels.csv"
        _run([
            "python", str(script),
            "--input_csv", str(manifest),
            "--out_dir", str(run_dir / "out"),
            "--labels", str(labels),
            "--seed", "42",
        ])
        mapping = {}
        with open(labels, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                mapping.setdefault(row["source_clip"], []).append(row["clip_id"])
        return mapping

    def _assert_order_invariant(self, script, name):
        forward = self._fake_ids_by_clip(script, self.clips, f"{name}_forward")
        reverse = self._fake_ids_by_clip(script, list(reversed(self.clips)),
                                         f"{name}_reverse")
        self.assertEqual(sorted(forward), sorted(reverse),
                         f"{name}: hai lần chạy phủ khác tập clip")
        self.assertTrue(forward, f"{name}: không sinh được fake nào")
        for clip_id in forward:
            self.assertEqual(
                sorted(forward[clip_id]), sorted(reverse[clip_id]),
                f"{name}: clip {clip_id} ra fake khác khi đảo thứ tự manifest — "
                f"RNG đang phụ thuộc thứ tự xử lý",
            )

    def test_temporal_desync_is_order_invariant(self):
        self._assert_order_invariant(DESYNC, "desync")

    def test_frame_reverse_is_order_invariant(self):
        self._assert_order_invariant(REVERSE, "reverse")


if __name__ == "__main__":
    unittest.main()
