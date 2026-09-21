import csv
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from src.features.avhubert import _ProjectionBranch, load_checkpoint, load_source
from src.features.extraction import extract, read_pairs
from src.features.media import (FaceTracker, MediaConfig, decode_window, make_inputs,
                                run_media_command, save_mouth_preview)
from src.training.dataset import PairedFeatureDataset


class MediaExtractionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_face_tracking_ignores_hand_false_positive_and_detection_order(self):
        import dlib
        tracker = FaceTracker()
        original = dlib.rectangle(740, 277, 1061, 598)
        hand = dlib.rectangle(468, 528, 735, 795)
        moved = dlib.rectangle(718, 247, 1103, 632)
        self.assertIs(tracker.select([original]), original)
        self.assertIs(tracker.select([hand, moved]), moved)
        self.assertIs(tracker.select([moved, hand]), moved)
        self.assertIsNone(tracker.select([hand]))
        self.assertIsNone(tracker.select([]))
        self.assertIs(tracker.select([moved]), moved)

    def test_face_tracking_refuses_ambiguous_start_and_overlapping_candidates(self):
        import dlib
        face = dlib.rectangle(10, 10, 110, 110)
        other = dlib.rectangle(12, 10, 112, 110)
        tracker = FaceTracker()
        with self.assertRaisesRegex(ValueError, 'initial'):
            tracker.select([face, other])
        tracker.select([face])
        with self.assertRaisesRegex(ValueError, 'tracking'):
            tracker.select([face, other])

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
    def test_mouth_preview_uses_model_pixels_and_keeps_audio(self):
        from scipy.io import wavfile
        config = MediaConfig()
        signal = np.zeros(16000, dtype=np.int16)
        crops = np.random.default_rng(42).integers(0, 256, (25, 96, 96, 3), dtype=np.uint8)
        _, visual = make_inputs(signal, crops, config)
        original = visual.clone()
        wavfile.write(self.root / 'audio.wav', 16000, signal)
        result = save_mouth_preview(visual, config, self.root / 'audio.wav', self.root / 'mouth.mp4')
        sheet = cv2.imread(str(self.root / result['preview_frames_file']), cv2.IMREAD_GRAYSCALE)
        expected = np.concatenate([cv2.cvtColor(crops[i], cv2.COLOR_BGR2GRAY)[4:92, 4:92]
                                   for i in result['preview_frame_indices']], axis=1)
        np.testing.assert_array_equal(sheet, expected)
        torch.testing.assert_close(visual, original)
        info = json.loads(run_media_command(['ffprobe', '-v', 'error', '-show_streams',
                                             '-of', 'json', self.root / 'mouth.mp4']))
        video = next(s for s in info['streams'] if s['codec_type'] == 'video')
        self.assertEqual((video['width'], video['height'], int(video['nb_frames'])), (88, 88, 25))
        self.assertTrue(any(s['codec_type'] == 'audio' for s in info['streams']))
        with self.assertRaises(FileExistsError):
            save_mouth_preview(visual, config, self.root / 'audio.wav', self.root / 'mouth.mp4')

    def test_inputs_match_audio_stack_and_visual_eval_transform(self):
        from python_speech_features import logfbank
        signal = (1000 * np.sin(np.arange(16000) * 0.1)).astype(np.int16)
        crops = np.full((25, 96, 96, 3), 128, dtype=np.uint8)
        audio, video = make_inputs(signal, crops, MediaConfig())
        reference = logfbank(signal, samplerate=16000).astype(np.float32)
        reference = np.pad(reference, ((0, (-len(reference)) % 4), (0, 0))).reshape(-1, 104)
        reference = torch.nn.functional.layer_norm(torch.from_numpy(reference), (104,))
        self.assertEqual(audio.shape, (1, 104, 25))
        self.assertEqual(video.shape, (1, 1, 25, 88, 88))
        torch.testing.assert_close(audio[0].T, reference)
        self.assertAlmostEqual(video[0, 0, 0, 0, 0].item(), (128 / 255 - .421) / .165, places=5)
        with self.assertRaisesRegex(ValueError, 'more than one frame'):
            make_inputs(signal[:8000], crops, MediaConfig())

    def test_checkpoint_strict_branch_loading_and_freezing(self):
        # Stand-in upstream network: tests loader behavior, not pretrained quality.
        source = self.root / 'avhubert/resnet.py'
        source.parent.mkdir()
        source.write_text('''import torch.nn as nn
class ResEncoder(nn.Module):
    backend_out = 2
    def __init__(self, relu_type, weights):
        super().__init__()
        self.conv = nn.Conv3d(1, 2, 1)
    def forward(self, x):
        return self.conv(x).mean(dim=(-1,-2))
''', encoding='utf-8')
        resnet = load_source(source, 'test_resnet').ResEncoder('prelu', None)
        a, v = _ProjectionBranch(104, 768), _ProjectionBranch(2, 768, resnet)
        state = {f'feature_extractor_{kind}.{key}': value
                 for kind, branch in [('audio', a), ('video', v)]
                 for key, value in branch.state_dict().items()}
        # Official Base omits fields equal to upstream dataclass defaults.
        cfg = dict(model=dict(_name='av_hubert', audio_feat_dim=104),
                   task=dict(sample_rate=25, stack_order_audio=4, normalize=True))
        path = self.root / 'base.pt'
        torch.save(dict(cfg=cfg, model=state), path)
        adapter, config, provenance = load_checkpoint(path, self.root)
        audio, video = torch.randn(1, 104, 3), torch.randn(1, 1, 3, 8, 8)
        output_a, output_v = adapter(audio, video)
        torch.testing.assert_close(output_a, a(audio).transpose(1, 2))
        torch.testing.assert_close(output_v, v(video).transpose(1, 2))
        self.assertFalse(any(p.requires_grad for p in adapter.parameters()))
        self.assertEqual(config.image_crop_size, 88)
        self.assertEqual(len(provenance['checkpoint_sha256']), 64)
        del state['feature_extractor_audio.proj.bias']
        torch.save(dict(cfg=cfg, model=state), path)
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            load_checkpoint(path, self.root)
        cfg['model']['sub_encoder_layers'] = 1
        torch.save(dict(cfg=cfg, model=state), path)
        with self.assertRaisesRegex(ValueError, 'supports AV-HuBERT Base'):
            load_checkpoint(path, self.root)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_decoding_preserves_stream_offset_and_window(self):
        source = self.root / 'offset.mkv'
        run_media_command(['ffmpeg', '-nostdin', '-v', 'error', '-y',
                           '-f', 'lavfi', '-i', 'color=c=red:s=128x128:r=25:d=2',
                           '-f', 'lavfi', '-i', 'aevalsrc=if(between(t\,0.8\,0.81)\,0.8\,0):s=16000:d=2',
                           '-af', 'asetpts=PTS+0.4/TB', '-c:v', 'ffv1', '-c:a', 'pcm_s16le', source])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            decode_window(source, 0, 25, self.root)
        video, waveform = decode_window(source, 20, 25, self.root)
        peak_time = np.argmax(np.abs(waveform.astype(float))) / 16000
        self.assertAlmostEqual(peak_time, 0.4, delta=0.01)
        cap = cv2.VideoCapture(str(video))
        count = 0
        try:
            while cap.read()[0]:
                count += 1
        finally:
            cap.release()
        self.assertEqual(count, 25)
        with self.assertRaisesRegex(ValueError, 'timeline'):
            decode_window(source, 45, 25, self.root)

    def test_pair_cache_is_readable_and_keeps_labels_splits_and_hashes(self):
        manifest = self.root / 'input.csv'
        (self.root / 'original.mp4').write_bytes(b'original')
        (self.root / 'compressed.mp4').write_bytes(b'compressed')
        row = dict(sample_id='fake_clip:0:3', label=1, split='train', media_path='original.mp4',
                   paired_media_path='compressed.mp4', view_id='original', paired_view_id='crf23',
                   start_frame=0, num_frames=3)
        with manifest.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        args = SimpleNamespace(video=None, manifest=manifest, out=self.root / 'features',
                               checkpoint=None, upstream=None, device='cpu', predictor=None, mean_face=None)
        class Processor:
            provenance = {'test_fixture': True}
            def __call__(self, *unused, **kwargs):
                return torch.zeros(1, 104, 3), torch.zeros(1, 1, 3, 8, 8), {}
        def adapter(audio, video):
            return torch.ones(1, 3, 4), torch.ones(1, 3, 6)
        with patch('src.features.extraction.load_checkpoint', return_value=(adapter, MediaConfig(), {})), \
                patch('src.features.extraction.MediaProcessor', return_value=Processor()), \
                patch('src.features.extraction.importlib.metadata.version', return_value='test'), \
                patch('src.features.extraction.run_media_command', return_value=b'ffmpeg test\n'):
            extract(args)
            with self.assertRaises(FileExistsError):
                extract(args)
        sample = PairedFeatureDataset(args.out / 'pairs.csv', 'train')[0]
        self.assertEqual(sample['label'], 1)
        self.assertEqual(sample['audio'].shape, (3, 4))
        report = json.loads((args.out / 'run.json').read_text())
        self.assertNotEqual(report['samples'][0]['source_sha256'], report['samples'][1]['source_sha256'])
        text = manifest.read_text().replace('compressed.mp4', 'original.mp4')
        manifest.write_text(text)
        with self.assertRaisesRegex(ValueError, 'same media file'):
            read_pairs(manifest)


if __name__ == '__main__':
    unittest.main()
