import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from src.features.avhubert import FrozenAVHubertBranches
from src.training.dataset import PairedFeatureDataset, collate_pairs
from src.models.detector import AudioVisualDetector


class FeatureTrainingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'pairs.csv'
        rows = []
        for i, split in enumerate(['train', 'train', 'val', 'val']):
            for view in ['original', 'compressed']:
                torch.save(dict(sample_id=f'clip{i}:0:5', label=i % 2, split=split,
                                view_id=view, extractor_id='synthetic_fixture', timeline_id='25fps:0:5',
                                audio=torch.randn(3 + i, 4), visual=torch.randn(3 + i, 6)),
                           self.root / f'{i}_{view}.pt')
            rows.append(dict(sample_id=f'clip{i}:0:5', label=i % 2, split=split,
                             feature_path=f'{i}_original.pt', paired_feature_path=f'{i}_compressed.pt'))
        with self.manifest.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_batches_pad_without_changing_lengths(self):
        data = PairedFeatureDataset(self.manifest, 'train')
        batch = collate_pairs([data[0], data[1]])
        self.assertEqual(batch['audio'].shape, (2, 4, 4))
        self.assertEqual(batch['lengths'].tolist(), [3, 4])

    def test_reject_wrong_label_timeline_and_checkpoint(self):
        path = self.root / '0_compressed.pt'
        original = torch.load(path, weights_only=True)
        for key, value in [('label', 1), ('timeline_id', 'other'), ('extractor_id', 'other'),
                           ('split', 'test'), ('sample_id', 'other')]:
            torch.save(dict(original, **{key: value}), path)
            with self.assertRaises(ValueError):
                PairedFeatureDataset(self.manifest, 'train')[0]

    def test_train_cli_checkpoint_reload(self):
        script = Path(__file__).resolve().parents[2] / 'src/training/01_train.py'
        for weight in ['0', '1']:
            output = self.root / ('run_' + weight)
            result = subprocess.run([sys.executable, str(script), '--manifest', str(self.manifest),
                                     '--out', str(output), '--epochs', '1', '--hidden-dim', '8',
                                     '--consistency-weight', weight], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = torch.load(output / 'best.pt', weights_only=True)
            model = AudioVisualDetector(**saved['model_config'])
            model.load_state_dict(saved['model'])
            batch = collate_pairs([PairedFeatureDataset(self.manifest, 'val')[0]])
            self.assertTrue(torch.isfinite(model(batch['audio'], batch['visual'], batch['lengths'])).all())

    def test_adapter_uses_two_frozen_branches(self):
        class Visual(nn.Module):
            def forward(self, video):
                return video.mean(dim=(-1, -2)).repeat(1, 6, 1)
        backbone = SimpleNamespace(feature_extractor_audio=nn.Conv1d(4, 6, 1),
                                   feature_extractor_video=Visual())
        adapter = FrozenAVHubertBranches(backbone)
        adapter.train()
        a, v = adapter(torch.randn(2, 4, 5), torch.randn(2, 1, 5, 8, 8))
        self.assertEqual(a.shape, (2, 5, 6))
        self.assertEqual(v.shape, (2, 5, 6))
        self.assertFalse(a.requires_grad)
        self.assertFalse(adapter.training)
