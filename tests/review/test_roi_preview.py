import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/tools/review/build_roi_preview.py"
SPEC = importlib.util.spec_from_file_location("build_roi_preview", SCRIPT)
PREVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREVIEW)


class _NoFaceStage04:
    @staticmethod
    def detect_and_crop(*_args, **_kwargs):
        return None, None


class RoiPreviewTest(unittest.TestCase):
    def test_no_face_clip_gets_audio_only_slate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "source.mp4"
            output = tmp / "preview.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=25:d=1",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-shortest", str(source),
            ], check=True)
            error, no_face = PREVIEW.build_one(
                _NoFaceStage04(), object(), str(source), str(output),
                25.0, 96, 3, 2, 0.25,
            )
            self.assertIsNone(error)
            self.assertTrue(no_face)
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries",
                "stream=codec_type,width,height", "-of", "json", str(output),
            ], text=True))
            self.assertEqual({row["codec_type"] for row in probe["streams"]},
                             {"video", "audio"})
            video = next(row for row in probe["streams"] if row["codec_type"] == "video")
            self.assertEqual((video["width"], video["height"]), (288, 288))


if __name__ == "__main__":
    unittest.main()
