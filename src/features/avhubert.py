"""Frozen pre-fusion branches of an already loaded official AVHubertModel.

Not the fused Transformer output. The loader below supports base_vox_iter5.pt.
Audio must be checkpoint-compatible stacked acoustic features [B,F,T]; video
must be normalized grayscale mouth crops [B,1,T,H,W] on the SAME timeline.
Reference: facebookresearch/av_hubert, avhubert/hubert.py, SubModel.
"""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_source(path, name):
    """Import an explicit upstream file without importing its Fairseq package."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Missing upstream source: {path}. See src/features/README.md')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ProjectionBranch(nn.Module):
    """Equivalent to upstream SubModel for sub_encoder_layers=0 only."""
    def __init__(self, input_dim, output_dim, resnet=None):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, inputs):
        if self.resnet is not None:
            inputs = self.resnet(inputs)
        return self.proj(inputs.transpose(1, 2)).transpose(1, 2)


class _DictionaryMetadata:
    """Inert holder for the checkpoint's unused Fairseq training dictionary.

    Only tensor weights and cfg are consumed below. No Fairseq dictionary code
    is executed, and no model layer/weight is substituted by this metadata type.
    """
    pass


def load_checkpoint(checkpoint, upstream, device='cpu'):
    """Load the two exact pre-fusion branches, not the fused Transformer.

    The official Base file stores plain cfg and an unused Fairseq dictionary.
    Read with weights_only and narrowly allow that metadata class, without
    importing Fairseq. No unrestricted pickle or fallback to random weights.
    The upstream ResEncoder is reused; Fairseq itself is not needed for these
    branches because Base has no per-modality Transformer layers.
    """
    checkpoint, upstream = Path(checkpoint), Path(upstream)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Place base_vox_iter5.pt at: {checkpoint.resolve()}')
    if tuple(int(part) for part in torch.__version__.split('.')[:2]) < (2, 6):
        raise RuntimeError('Checkpoint loader requires PyTorch >= 2.6')
    with torch.serialization.safe_globals([
            (_DictionaryMetadata, 'fairseq.data.dictionary.Dictionary')]):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    cfg = state.get('cfg')
    if cfg is None:
        raise ValueError('Expected official pretrained checkpoint with cfg/model; finetuned files are unsupported')
    model_cfg, task_cfg = cfg['model'], cfg['task']
    if (model_cfg.get('_name') != 'av_hubert'
            or model_cfg.get('sub_encoder_layers', 0) != 0
            or model_cfg.get('audio_feat_dim') != 104
            or model_cfg.get('encoder_embed_dim', 768) != 768):
        raise ValueError('This loader supports AV-HuBERT Base pre-fusion branches with 104 audio inputs only')
    from src.features.media import MediaConfig
    config = MediaConfig.from_task(task_cfg)
    resnet_path = upstream / 'avhubert/resnet.py'
    resnet = load_source(resnet_path, '_avhubert_resnet').ResEncoder(
        relu_type=model_cfg.get('resnet_relu_type', 'prelu'), weights=None)
    audio = _ProjectionBranch(104, 768)
    visual = _ProjectionBranch(resnet.backend_out, 768, resnet)
    for name, branch in [('audio', audio), ('video', visual)]:
        prefix = f'feature_extractor_{name}.'
        branch_state = {key[len(prefix):]: value for key, value in state['model'].items()
                        if key.startswith(prefix)}
        # Strict loading rejects incomplete or incompatible branches.
        branch.load_state_dict(branch_state, strict=True)
    adapter = FrozenAVHubertBranches(SimpleNamespace(
        feature_extractor_audio=audio, feature_extractor_video=visual)).to(device)
    provenance = dict(checkpoint_sha256=sha256(checkpoint),
                      resnet_sha256=sha256(resnet_path), mode='base_pre_fusion_v1',
                      loaded_parameters=sum(p.numel() for p in adapter.parameters()))
    return adapter, config, provenance
