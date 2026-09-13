"""Run to check model -> train step -> prediction using SYNTHETIC features only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    if '--help' in sys.argv:
        print(__doc__)
        return
    import copy
    import torch
    from src.models.detector import AudioVisualDetector
    from src.training.steps import train_step
    from src.evaluation.predict import predict_batch

    torch.manual_seed(42)
    audio, visual = torch.randn(4, 12, 16), torch.randn(4, 12, 24)
    batch = dict(audio=audio, visual=visual, paired_audio=audio + 0.05 * torch.randn_like(audio),
                 paired_visual=visual + 0.05 * torch.randn_like(visual),
                 lengths=torch.tensor([12, 10, 8, 6]), labels=torch.tensor([0., 1., 0., 1.]),
                 sample_ids=['fixture_a:0', 'fixture_b:0', 'fixture_c:0', 'fixture_d:0'],
                 paired_sample_ids=['fixture_a:0', 'fixture_b:0', 'fixture_c:0', 'fixture_d:0'])
    initial = AudioVisualDetector(16, 24)
    print('SYNTHETIC SMOKE ONLY: random features and noise are NOT AV-HuBERT or real compression data.')
    for method, weight in [('Y', 0.0), ('X', 1.0)]:
        model = copy.deepcopy(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(3):
            result = train_step(model, optimizer, batch, consistency_weight=weight)
        predictions = predict_batch(model, audio, visual, batch['lengths'])
        print(method, result, 'prediction shape:', tuple(predictions['prediction'].shape))
    print('PASS: forward/backward/prediction completed. No research accuracy or checkpoint produced.')


if __name__ == '__main__':
    main()
