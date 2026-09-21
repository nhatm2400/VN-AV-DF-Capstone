"""Offline preview/download contracts; no real MuseTalk quality claims."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

from src.generators.preparation import download_parts, musetalk_smoke


class DownloadTest(unittest.TestCase):
    def test_reuses_completed_parts_without_network_and_checks_final_hash(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(download_parts, 'CHUNK', 4):
            dest = Path(tmp) / 'weight.bin'
            parts = Path(tmp) / 'weight.bin.parts'
            parts.mkdir()
            payload = b'abcdefghij'
            for index, data in enumerate((b'abcd', b'efgh', b'ij')):
                (parts / f'{index:05}.part').write_bytes(data)
            sha = hashlib.sha256(payload).hexdigest()
            with patch.object(download_parts.subprocess, 'run', side_effect=AssertionError('Unexpected network')):
                download_parts.download('https://example.invalid/file', dest, len(payload), sha)
            self.assertEqual(dest.read_bytes(), payload)
            self.assertFalse(any(parts.glob('*.part')))

    def test_equal_size_corrupt_parts_are_not_published(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(download_parts, 'CHUNK', 4):
            parts = Path(tmp) / 'parts'
            parts.mkdir()
            (parts / '00000.part').write_bytes(b'BAD!')
            dest = Path(tmp) / 'weight.bin'
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                download_parts.assemble(parts, dest, 4, hashlib.sha256(b'good').hexdigest())
            self.assertFalse(dest.exists())
            self.assertTrue((parts / '00000.part').exists())

    def test_part_identity_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'weight.bin'
            parts = Path(tmp) / 'weight.bin.parts'
            parts.mkdir()
            (parts / 'download.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'refusing to mix'):
                download_parts.download('https://example.invalid/file', dest, 4, 'a'*64)

    def test_huggingface_small_file_uses_git_blob_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            payload = b'{"test":1}'
            path.write_bytes(payload)
            expected = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
            self.assertTrue(download_parts.verified(path, len(payload), expected, 'git-sha1'))


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class PreviewTest(unittest.TestCase):
    def test_exact_five_2s_inputs_and_wrong_duration_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            inputs = run / 'input'
            inputs.mkdir()
            video, audio = inputs / 'source.mp4', inputs / 'driver.wav'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=red:s=64x64:r=25',
                            '-t', '2', '-c:v', 'libx264', str(video)], check=True)
            with wave.open(str(audio), 'wb') as handle:
                handle.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                handle.writeframes(b'\0' * 64000)
            tasks = {f'task_{i:02}': dict(video_path='input/source.mp4', audio_path='input/driver.wav',
                     result_name=f'task_{i:02}_musetalk_v15_fake.mp4') for i in range(1, 6)}
            (run / 'smoke.yaml').write_text(json.dumps(tasks), encoding='utf-8')
            self.assertEqual(len(musetalk_smoke.input_tasks(run)), 5)
            with wave.open(str(audio), 'wb') as handle:
                handle.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                handle.writeframes(b'\0' * 32000)
            with self.assertRaisesRegex(ValueError, 'two seconds'):
                musetalk_smoke.input_tasks(run)

    def test_comparison_panel_order_and_length_with_fixture_media(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp, patch.object(musetalk_smoke, 'RUN', Path(tmp)):
            run = Path(tmp)
            (run / 'output/v15').mkdir(parents=True)
            paths = [run / 'real.mp4', run / 'wav2lip.mp4', run / 'output/v15/task_01_musetalk_v15_fake.mp4']
            for color, path in zip(('red', 'green', 'blue'), paths):
                subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'color=c={color}:s=64x64:r=25',
                    '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono', '-t', '2', '-c:v', 'libx264',
                    '-c:a', 'aac', str(path)], check=True)
            with (run / 'mapping.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['task', 'wav2lip_fake'])
                writer.writeheader()
                writer.writerow(dict(task='task_01', wav2lip_fake=str(paths[1])))
            musetalk_smoke.make_comparisons({'task_01': dict(video_path=str(paths[0]), result_name=paths[2].name)})
            target = run / 'comparison/task_01__real_wav2lip_musetalk.mp4'
            reader = cv2.VideoCapture(str(target))
            try:
                ok, frame = reader.read()
                self.assertTrue(ok)
                self.assertEqual(int(reader.get(cv2.CAP_PROP_FRAME_COUNT)), 50)
                # OpenCV channel order is BGR: original red, Wav2Lip green, MuseTalk blue.
                self.assertEqual([int(frame[320, x].argmax()) for x in (320, 960, 1600)], [2, 1, 0])
            finally:
                reader.release()
