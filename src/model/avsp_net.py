"""
avsp_net.py — AVSP-Net (Audio-Visual Sync + Prosody Network)

Hiện thực theo MODEL_PROPOSAL.md §3–§5:

  Audio  w2v [B,Ta,768] ──proj──► A [B,Ta,D] ─┐
  Mouth ROI [B,Tv,96,96] ─CNN+Transformer──► V [B,Tv,D] ─┤ CrossAttn(Q=A, KV=V)
  Prosody   [B,Tp,4] ────Conv+BiGRU────► P [B,Tp,Dp] ─┐  │
                                                      ▼  ▼
                                    concat(av_pool, p_pool) ► MLP ► logit real/fake
                                    av_pool ► offset head (7 lớp: ±15/±7/±3/0 frames)

Nguyên tắc giữ từ PoC PAMF: audio làm Query đi tìm khẩu hình (Key/Value) — nhưng
trên MOUTH ROI thay vì full-frame (chống leak identity/background), và thêm nhánh
PROSODY để bắt fake 03_pitch_flatten (audio thuần, lip-sync vẫn khớp).

Khác PoC: dùng BCEWithLogitsLoss (không Sigmoid trong forward — ổn định số học),
thêm head phụ offset (multi-task, khớp 01_temporal_desync) + consistency loss.

Toggle nhánh qua `branches` để chạy ablation §7:
  ("audio",) / ("visual",) / ("audio","visual") / ("audio","visual","prosody")
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 7 lớp offset khớp 01_temporal_desync (âm = audio sớm) ; real & fake khác = 0
OFFSET_CLASSES = [-15, -7, -3, 0, 3, 7, 15]


def offset_to_class(shift_frames):
    """Map shift (int) -> index lớp; mọi giá trị lạ -> lớp 0-offset (index 3)."""
    try:
        return OFFSET_CLASSES.index(int(shift_frames))
    except (ValueError, TypeError):
        return OFFSET_CLASSES.index(0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):                          # [B,T,D]
        return x + self.pe[:, :x.size(1)]


class MouthEncoder(nn.Module):
    """uint8-normalized mouth ROI [B,T,1,96,96] -> [B,T,D] (2D CNN + temporal transformer)."""

    def __init__(self, d_model=256, n_layers=2, n_heads=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),    # 48
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),   # 24
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(), # 12
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),# 6
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model,
                                           dropout=0.1, batch_first=True)
        self.temporal = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x):                          # [B,T,1,96,96] float in [0,1]
        B, T = x.shape[:2]
        f = self.cnn(x.flatten(0, 1)).flatten(1)   # [B*T,128]
        f = self.proj(f).view(B, T, -1)
        return self.temporal(self.pos(f))          # [B,T,D]


class ProsodyEncoder(nn.Module):
    """[B,T,4] (f0_z, delta_f0, energy_z, voiced) -> [B,T',Dp] (Conv1d + BiGRU)."""

    def __init__(self, d_out=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.ReLU(),
        )
        self.gru = nn.GRU(64, d_out // 2, batch_first=True, bidirectional=True)

    def forward(self, x):                          # [B,T,4]
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)   # [B,T/2,64]
        out, _ = self.gru(h)
        return out                                 # [B,T/2,Dp]


class AttentivePool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x):                          # [B,T,D] -> [B,D]
        w = torch.softmax(self.score(x), dim=1)
        return (w * x).sum(dim=1)


class AVSPNet(nn.Module):
    def __init__(self, d_model=256, d_prosody=128, n_heads=4,
                 d_audio_in=768, branches=("audio", "visual", "prosody")):
        super().__init__()
        self.branches = tuple(branches)
        assert any(b in self.branches for b in ("audio", "visual")), \
            "cần ít nhất 1 nhánh audio hoặc visual"

        if "audio" in self.branches:
            self.audio_norm = nn.LayerNorm(d_audio_in)
            self.audio_proj = nn.Linear(d_audio_in, d_model)
        if "visual" in self.branches:
            self.visual_enc = MouthEncoder(d_model, n_layers=2, n_heads=n_heads)
        if "audio" in self.branches and "visual" in self.branches:
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                    dropout=0.1, batch_first=True)
            self.attn_norm = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.ReLU(),
                                    nn.Dropout(0.1), nn.Linear(2 * d_model, d_model))
            self.ff_norm = nn.LayerNorm(d_model)
            # projector cho consistency loss (audio vs visual pooled)
            self.cons_a = nn.Linear(d_model, 128)
            self.cons_v = nn.Linear(d_model, 128)
        if "prosody" in self.branches:
            self.prosody_enc = ProsodyEncoder(d_prosody)
            self.prosody_pool = AttentivePool(d_prosody)

        self.av_pool = AttentivePool(d_model)
        d_cls = d_model + (d_prosody if "prosody" in self.branches else 0)
        self.cls_head = nn.Sequential(
            nn.Linear(d_cls, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
        self.offset_head = nn.Linear(d_model, len(OFFSET_CLASSES))

    def forward(self, w2v=None, mouth=None, prosody=None):
        """
        w2v     [B,Ta,768] float  (None nếu nhánh audio tắt)
        mouth   [B,Tv,96,96] float [0,1] (None nếu nhánh visual tắt)
        prosody [B,Tp,4] float    (None nếu nhánh prosody tắt)
        """
        a = v = None
        if "audio" in self.branches and w2v is not None:
            a = self.audio_proj(self.audio_norm(w2v))            # [B,Ta,D]
        if "visual" in self.branches and mouth is not None:
            v = self.visual_enc(mouth.unsqueeze(2))              # [B,Tv,D]

        cons_a = cons_v = None
        if a is not None and v is not None:
            att, _ = self.cross_attn(a, v, v)                    # Q=A, K=V, V=V
            h = self.attn_norm(a + att)
            h = self.ff_norm(h + self.ff(h))                     # [B,Ta,D]
            cons_a = F.normalize(self.cons_a(a.mean(dim=1)), dim=-1)
            cons_v = F.normalize(self.cons_v(v.mean(dim=1)), dim=-1)
        else:
            h = a if a is not None else v                        # unimodal

        av = self.av_pool(h)                                     # [B,D]
        feats = [av]
        if "prosody" in self.branches and prosody is not None:
            feats.append(self.prosody_pool(self.prosody_enc(prosody)))
        z = torch.cat(feats, dim=-1)

        return {
            "logit": self.cls_head(z).squeeze(-1),               # [B]
            "offset_logits": self.offset_head(av),               # [B,7]
            "cons_a": cons_a, "cons_v": cons_v,
        }


def compute_losses(out, label, offset_cls, w_offset=0.5, w_cons=0.1, pos_weight=None):
    """
    MODEL_PROPOSAL §5:  L = BCEWithLogits + 0.5*CE(offset) + 0.1*consistency
    consistency: real -> kéo audio/visual embedding lại gần; fake -> đẩy xa (margin).
    """
    losses = {}
    losses["bce"] = F.binary_cross_entropy_with_logits(
        out["logit"], label.float(), pos_weight=pos_weight)
    losses["offset"] = F.cross_entropy(out["offset_logits"], offset_cls)
    total = losses["bce"] + w_offset * losses["offset"]
    if out["cons_a"] is not None:
        sim = (out["cons_a"] * out["cons_v"]).sum(dim=-1)        # cosine [-1,1]
        m_pos, m_neg = 0.5, 0.3
        cons = torch.where(label.bool(),
                           F.relu(sim - m_neg),                  # fake: sim phải < 0.3
                           F.relu(m_pos - sim))                  # real: sim phải > 0.5
        losses["cons"] = cons.mean()
        total = total + w_cons * losses["cons"]
    losses["total"] = total
    return losses
