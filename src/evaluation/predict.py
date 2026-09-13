"""Batch predictions over prepared audio-visual features; no threshold tuning."""
import torch


@torch.inference_mode()
def predict_batch(model, audio, visual, lengths, threshold=0.5):
    """0.5 is a plumbing default; select the research threshold on validation only."""
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be between 0 and 1')
    was_training = model.training
    model.eval()
    try:
        scores = model(audio, visual, lengths).sigmoid()
        return {'fake_score': scores.cpu(), 'prediction': (scores >= threshold).long().cpu()}
    finally:
        model.train(was_training)
