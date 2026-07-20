"""
dataset.py — Dataset đọc labels.csv (stage 05) + feature .pt (stage 04) cho AVSP-Net.

Mỗi sample trả dict tensor kích thước CỐ ĐỊNH (pad 0 / truncate) để collate mặc
định stack được:
  w2v     [T_A, 768]  float32   (zeros nếu thiếu / nhánh tắt)
  mouth   [T_V, 96, 96] float32 [0,1]
  prosody [T_P, 4]    float32
  label   ()  int64  — 0 real / 1 fake
  offset  ()  int64  — index lớp offset (khớp model.OFFSET_CLASSES)

TODO sau này: random temporal crop khi train (hiện truncate từ đầu), mask pad.
"""

import os
import re
import csv
import sys
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import gaussian_blur

# import offset map từ src/model (namespace package, chạy từ repo root)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.avsp_net import offset_to_class  # noqa: E402

T_AUDIO = 200      # ~4s wav2vec2 (50Hz)
T_MOUTH = 100      # ~4s @ 25fps
T_PROSODY = 400    # ~4s @ 100Hz
MOUTH_SIZE = 96

_SHIFT_RE = re.compile(r"shift=(-?\d+)f")


def _pad_trunc(x, T):
    """[t, ...] -> [T, ...] pad 0 cuối / cắt đầu."""
    if x.shape[0] >= T:
        return x[:T]
    pad = torch.zeros((T - x.shape[0],) + tuple(x.shape[1:]), dtype=x.dtype)
    return torch.cat([x, pad], dim=0)


class AVSPDataset(Dataset):
    def __init__(self, labels_csv, features_dir, split, branches=("audio", "visual", "prosody"),
                 real_blur_aug_p=0.0, blur_sigma=(2.0, 8.0)):
        # real_blur_aug_p: xác suất blur mouth ROI của clip REAL khi train -> chống leak
        #   "mờ = fake". Đặt ≈ tỉ lệ anon trong fake (~0.25) để P(blur|real)≈P(blur|fake).
        #   CHỈ bật cho split train (val/test giữ nguyên để đo trung thực).
        self.features_dir = features_dir
        self.branches = tuple(branches)
        self.real_blur_aug_p = real_blur_aug_p
        self.blur_sigma = blur_sigma
        self.rows = []
        with open(labels_csv, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("split") != split:
                    continue
                pt = os.path.join(features_dir, r["clip_id"] + ".pt")
                if os.path.isfile(pt):
                    r["_pt"] = pt
                    self.rows.append(r)
        if not self.rows:
            raise RuntimeError(
                f"Không có sample split='{split}' trong {labels_csv} có feature tại {features_dir}. "
                f"Chạy 04_extract_features + 05_build_labels trước.")

    def __len__(self):
        return len(self.rows)

    def pos_weight(self):
        """n_real/n_fake — cho BCEWithLogitsLoss cân bằng lớp."""
        n_fake = sum(1 for r in self.rows if r["label"] == "1")
        n_real = len(self.rows) - n_fake
        return torch.tensor(max(n_real, 1) / max(n_fake, 1), dtype=torch.float32)

    def __getitem__(self, i):
        r = self.rows[i]
        obj = torch.load(r["_pt"], map_location="cpu", weights_only=False)

        # audio (w2v)
        if "audio" in self.branches and obj.get("w2v") is not None:
            w2v = _pad_trunc(obj["w2v"].float(), T_AUDIO)
        else:
            w2v = torch.zeros(T_AUDIO, 768)

        # visual (mouth ROI)
        if "visual" in self.branches and obj.get("mouth") is not None:
            mouth = _pad_trunc(obj["mouth"].float() / 255.0, T_MOUTH)
            # blur đối xứng: làm mờ ROI của một phần REAL -> "mờ" hết là chỉ báo của fake
            if int(r["label"]) == 0 and self.real_blur_aug_p > 0 \
                    and random.random() < self.real_blur_aug_p:
                s = random.uniform(*self.blur_sigma)
                mouth = gaussian_blur(mouth.unsqueeze(1), kernel_size=9, sigma=s).squeeze(1)
        else:
            mouth = torch.zeros(T_MOUTH, MOUTH_SIZE, MOUTH_SIZE)

        # prosody
        if "prosody" in self.branches and obj.get("prosody") is not None:
            pros = _pad_trunc(obj["prosody"].float(), T_PROSODY)
        else:
            pros = torch.zeros(T_PROSODY, 4)

        # offset class từ param (chỉ temporal_desync có shift=±Nf; còn lại -> 0)
        m = _SHIFT_RE.search(r.get("param", "") or "")
        off = offset_to_class(m.group(1) if m else 0)

        return {
            "w2v": w2v,
            "mouth": mouth,
            "prosody": pros,
            "label": torch.tensor(int(r["label"]), dtype=torch.long),
            "offset": torch.tensor(off, dtype=torch.long),
            "method": r.get("method", ""),
            "clip_id": r.get("clip_id", ""),
        }


def collate(batch):
    out = {}
    for k in ("w2v", "mouth", "prosody", "label", "offset"):
        out[k] = torch.stack([b[k] for b in batch])
    out["method"] = [b["method"] for b in batch]
    out["clip_id"] = [b["clip_id"] for b in batch]
    return out
