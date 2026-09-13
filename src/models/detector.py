"""Small audio-visual classifier over aligned features, not a video/AV-HuBERT loader."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class AudioVisualDetector(nn.Module):
    """Inputs: audio [B,T,Da], visual [B,T,Dv], valid lengths [B]. Output: fake logits [B]."""

    def __init__(self, audio_dim, visual_dim, hidden_dim=64):
        super().__init__()
        self.audio_projection = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.ReLU())
        self.visual_projection = nn.Sequential(nn.Linear(visual_dim, hidden_dim), nn.ReLU())
        self.temporal = nn.GRU(2 * hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, audio, visual, lengths):
        if audio.ndim != 3 or visual.ndim != 3 or audio.shape[:2] != visual.shape[:2]:
            raise ValueError('Audio and visual must share batch size and aligned time steps')
        if (lengths.shape != (audio.shape[0],) or lengths.is_floating_point()
                or torch.any(lengths < 1) or torch.any(lengths > audio.shape[1])):
            raise ValueError('Lengths must be integer valid-frame counts, one per sample')
        fused = torch.cat((self.audio_projection(audio), self.visual_projection(visual)), dim=-1)
        packed = pack_padded_sequence(fused, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.temporal(packed)
        return self.classifier(hidden[-1]).squeeze(-1)
