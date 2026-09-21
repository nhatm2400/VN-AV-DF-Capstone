import unittest
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np

from src.generators.preparation.preview_quality import blend_lower_face, detect_scaled
from src.generators.preparation.quality_preview import allowed_size
from src.generators.preparation import quality_preview


class QualityPreviewTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_comparison_preserves_50_frames_with_different_time_bases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [root/f'{i}.mp4' for i in range(3)]
            for path, timebase in zip(paths, (12800, 90000, 25000)):
                subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=64x64:r=25:d=2',
                    '-f','lavfi','-i','anullsrc=r=16000:cl=mono','-t','2','-c:v','libx264',
                    '-video_track_timescale',str(timebase),'-c:a','aac',str(path)],check=True)
            target = root/'comparison.mp4'
            quality_preview.make_comparison(*paths,target)
            quality_preview.wav2lip.check_media(target,50)
            self.assertFalse(target.with_suffix('.building.mp4').exists())

    def test_detect_scales_back_without_touching_original_frames(self):
        frames = np.full((1, 1080, 1920, 3), 73, dtype=np.uint8)
        def detector(small):
            self.assertEqual(small.shape, (1, 540, 960, 3))
            return [(100, 120, 300, 400)]
        self.assertEqual(detect_scaled(frames, detector), [(200, 240, 600, 800)])
        self.assertTrue((frames == 73).all())

    def test_missing_face_is_not_converted_to_a_box(self):
        self.assertEqual(detect_scaled(np.zeros((1, 60, 80, 3), np.uint8),
                                       lambda frames: [None]), [None])

    def test_blend_keeps_eyes_and_edges_but_replaces_mouth(self):
        source = np.full((100, 100, 3), 20, np.uint8)
        generated = np.full_like(source, 220)
        result = blend_lower_face(source, generated)
        np.testing.assert_array_equal(result[:50], source[:50])
        np.testing.assert_array_equal(result[:, 0], source[:, 0])
        np.testing.assert_array_equal(result[-1], source[-1])
        self.assertTrue((result[76, 50] == 220).all())
        self.assertTrue(((result > 20) & (result < 220)).any())

    def test_4k_is_excluded_instead_of_downscaling_whole_video(self):
        self.assertTrue(allowed_size(dict(width=1920, height=1080)))
        self.assertTrue(allowed_size(dict(width=1080, height=1920)))
        self.assertFalse(allowed_size(dict(width=3840, height=2160)))
        self.assertFalse(allowed_size(dict(width=2048, height=1080)))

    def test_resume_rejects_changed_audio_before_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root/'data/generated/dataset_v1/test_run'
            run.mkdir(parents=True)
            manifest, video, audio = root/'manifest.csv', run/'face.avi', run/'driver.wav'
            for path in (manifest, video, audio):
                path.write_bytes(b'original')
            digest = quality_preview.wav2lip.digest
            task = dict(video_path=str(video), audio_path=str(audio),
                        video_sha256=digest(video), audio_sha256=digest(audio))
            (run/'plan.json').write_text(json.dumps(dict(manifest_sha256=digest(manifest), tasks=[task]*10)))
            audio.write_bytes(b'changed')
            with patch.object(quality_preview, 'ROOT', root), \
                 patch.object(quality_preview.settings, 'MANIFEST', manifest), \
                 patch.object(quality_preview, 'generate') as generate:
                with self.assertRaisesRegex(ValueError, 'Prepared input changed'):
                    quality_preview.main('test_run')
                generate.assert_not_called()
