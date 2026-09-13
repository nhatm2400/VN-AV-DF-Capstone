import copy
import unittest
import torch

from src.models.detector import AudioVisualDetector
from src.training.steps import train_step
from src.evaluation.predict import predict_batch


class DetectorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = AudioVisualDetector(3, 5, hidden_dim=8)
        self.audio, self.visual = torch.randn(2, 6, 3), torch.randn(2, 6, 5)
        self.lengths = torch.tensor([6, 3])

    def batch(self):
        return dict(audio=self.audio, visual=self.visual, paired_audio=self.audio + 0.2,
                    paired_visual=self.visual - 0.2, lengths=self.lengths, labels=torch.tensor([0., 1.]),
                    sample_ids=['a:0', 'b:0'], paired_sample_ids=['a:0', 'b:0'])

    def test_padding_does_not_change_scores_and_order_is_preserved(self):
        expected = self.model(self.audio, self.visual, self.lengths)
        audio, visual = self.audio.clone(), self.visual.clone()
        audio[1, 3:] = 999
        visual[1, 3:] = -999
        torch.testing.assert_close(expected, self.model(audio, visual, self.lengths))
        for i, length in enumerate(self.lengths):
            single = self.model(audio[i:i+1, :length], visual[i:i+1, :length], length.reshape(1))
            torch.testing.assert_close(expected[i:i+1], single)

    def test_both_modalities_receive_gradients(self):
        self.audio.requires_grad_()
        self.visual.requires_grad_()
        self.model(self.audio, self.visual, self.lengths).sum().backward()
        self.assertGreater(self.audio.grad.abs().sum().item(), 0)
        self.assertGreater(self.visual.grad.abs().sum().item(), 0)
        self.assertEqual(self.audio.grad[1, 3:].abs().sum().item(), 0)

    def test_x_y_share_supervision_and_x_adds_only_consistency(self):
        batch = self.batch()
        y = copy.deepcopy(self.model)
        x = copy.deepcopy(self.model)
        before = y.classifier.weight.detach().clone()
        y_result = train_step(y, torch.optim.SGD(y.parameters(), lr=0.1), batch, 0)
        x_result = train_step(x, torch.optim.SGD(x.parameters(), lr=0.1), batch, 2)
        self.assertAlmostEqual(y_result['supervised'], x_result['supervised'])
        self.assertAlmostEqual(y_result['loss'], y_result['supervised'])
        self.assertAlmostEqual(x_result['loss'], x_result['supervised'] + 2*x_result['consistency'], places=6)
        self.assertFalse(torch.equal(before, y.classifier.weight))

    def test_invalid_pairs_and_lengths_fail(self):
        batch = self.batch()
        batch['paired_sample_ids'] = ['different:0', 'b:0']
        with self.assertRaisesRegex(ValueError, 'identity'):
            train_step(self.model, torch.optim.SGD(self.model.parameters(), lr=0.1), batch)
        with self.assertRaises(ValueError):
            self.model(self.audio, self.visual, torch.tensor([6, 0]))

    def test_predictions_are_detached_and_preserve_model_mode(self):
        self.model.train()
        result = predict_batch(self.model, self.audio, self.visual, self.lengths)
        self.assertTrue(self.model.training)
        self.assertFalse(result['fake_score'].requires_grad)
        self.assertEqual(result['fake_score'].shape, (2,))
        self.assertTrue(torch.all((result['fake_score'] >= 0) & (result['fake_score'] <= 1)))
        torch.testing.assert_close(result['prediction'], (result['fake_score'] >= 0.5).long())


if __name__ == '__main__':
    unittest.main()
