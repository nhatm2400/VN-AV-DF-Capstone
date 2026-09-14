"""Frozen pre-fusion branches of an already loaded official AVHubertModel.

Not the fused Transformer output. No checkpoint loading or media preprocessing.
Audio must be checkpoint-compatible stacked acoustic features [B,F,T]; video
must be normalized grayscale mouth crops [B,1,T,H,W] on the SAME timeline.
Reference: facebookresearch/av_hubert, avhubert/hubert.py, SubModel.
"""
import torch
from torch import nn


class FrozenAVHubertBranches(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.audio_branch = backbone.feature_extractor_audio
        self.visual_branch = backbone.feature_extractor_video
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, audio, video):
        if (audio.ndim != 3 or video.ndim != 5 or video.shape[1] != 1
                or audio.shape[0] != video.shape[0] or audio.shape[2] != video.shape[2]):
            raise ValueError('Expected aligned audio [B,F,T] and video [B,1,T,H,W]')
        # Frozen extraction must stay deterministic even if a parent calls train().
        self.eval()
        a = self.audio_branch(audio).transpose(1, 2).contiguous()
        v = self.visual_branch(video).transpose(1, 2).contiguous()
        if a.shape[:2] != v.shape[:2] or a.shape[1] != audio.shape[2]:
            raise ValueError('Backbone changed the shared timeline')
        return a, v
