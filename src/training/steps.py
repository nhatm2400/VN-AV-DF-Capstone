"""Shared Y/X train step. Both see the same paired views; X adds consistency loss."""
import torch
from torch.nn import functional as F


def train_step(model, optimizer, batch, consistency_weight=0.0):
    """Each pair is two compression views of the SAME labeled clip/window.

    batch: audio, visual, paired_audio, paired_visual, lengths, labels,
    sample_ids, paired_sample_ids. IDs must identify the clip AND window.
    Features must already be aligned, on the model device, and extracted from
    each actual view. This function does not load or generate features.
    """
    if consistency_weight < 0:
        raise ValueError('consistency_weight must be non-negative')
    labels = batch['labels']
    ids, paired_ids = list(batch['sample_ids']), list(batch['paired_sample_ids'])
    if ids != paired_ids or len(ids) != labels.numel() or not all(ids):
        raise ValueError('Each compression pair must have the same clip/window identity')
    if labels.ndim != 1 or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('Labels must be [B], with real=0 and fake=1')
    if batch['audio'].shape != batch['paired_audio'].shape or batch['visual'].shape != batch['paired_visual'].shape:
        raise ValueError('Paired views must use the same window and aligned feature shapes')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    original = model(batch['audio'], batch['visual'], batch['lengths'])
    paired = model(batch['paired_audio'], batch['paired_visual'], batch['lengths'])
    labels = labels.to(dtype=original.dtype, device=original.device)
    supervised = (F.binary_cross_entropy_with_logits(original, labels)
                  + F.binary_cross_entropy_with_logits(paired, labels)) / 2
    consistency = F.mse_loss(original.sigmoid(), paired.sigmoid())
    loss = supervised + consistency_weight * consistency
    loss.backward()
    optimizer.step()
    return {'loss': loss.item(), 'supervised': supervised.item(), 'consistency': consistency.item()}
