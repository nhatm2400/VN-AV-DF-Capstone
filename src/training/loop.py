"""Epoch execution over prepared pairs. Validation uses only supervised loss."""
import torch
from torch.nn import functional as F

from src.training.steps import train_step


def run_epoch(model, loader, optimizer=None, consistency_weight=0.0):
    device = next(model.parameters()).device
    totals = {'loss': 0.0, 'supervised': 0.0, 'consistency': 0.0}
    count = 0
    for source in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in source.items()}
        if optimizer is not None:
            metrics = train_step(model, optimizer, batch, consistency_weight)
        else:
            model.eval()
            with torch.no_grad():
                a = model(batch['audio'], batch['visual'], batch['lengths'])
                b = model(batch['paired_audio'], batch['paired_visual'], batch['lengths'])
                loss = (F.binary_cross_entropy_with_logits(a, batch['labels'])
                        + F.binary_cross_entropy_with_logits(b, batch['labels'])) / 2
                metrics = dict(loss=loss.item(), supervised=loss.item(),
                               consistency=F.mse_loss(a.sigmoid(), b.sigmoid()).item())
        size = len(batch['labels'])
        count += size
        for key in totals:
            totals[key] += metrics[key] * size
    if not count:
        raise ValueError('Empty epoch')
    return {key: value / count for key, value in totals.items()}
