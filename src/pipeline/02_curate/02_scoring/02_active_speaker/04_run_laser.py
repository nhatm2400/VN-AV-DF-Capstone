"""Run the official LoCoNet+LASER checkpoint on selective request bins.

The official repository is loaded from an explicit pinned checkout.  This
adapter reuses the same 25 fps InsightFace tracks as the Light-ASD stage, but
keeps LoCoNet's own 112x112 face preprocessing and VGGish audio frontend.
LASER was trained with lip landmarks, while its consistency objective permits
landmark-free inference; therefore this runner supplies zero landmark features.

Output is an immutable sidecar bundle consumed by ``05_apply_laser_scores.py``.
It never mutates source media and never invents scores for failed bins.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import yaml
from scipy.io import wavfile


TARGET_FPS = 25
TARGET_SR = 16000
FACE_SIZE = 112
# The official full-video demo uses 20 frames as a track-quality heuristic.
# This adapter scores 200 ms selective bins, so a track covering one complete
# bin (5 frames at 25 fps) is the minimum useful unit. Clip-level rejection
# still requires a much longer bad interval in the temporal policy.
MIN_TRACK_FRAMES = 5


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: str | Path, expected: str, label: str) -> str:
    expected = str(expected).strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"{label} expected SHA-256 is invalid")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected}, actual={actual}"
        )
    return actual


def canonical_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_revision(path: str | Path) -> str:
    resolved = str(Path(path).resolve())
    result = subprocess.run(
        ["git", "-c", f"safe.directory={resolved}", "-C", resolved,
         "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def git_model_sources_are_clean(path: str | Path) -> bool:
    resolved = str(Path(path).resolve())
    result = subprocess.run(
        ["git", "-c", f"safe.directory={resolved}", "-C", resolved,
         "diff", "--quiet", "HEAD", "--", "*.py", "*.yaml", "*.yml"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def load_score_preprocessing_module():
    path = Path(__file__).with_name("01_score.py")
    spec = importlib.util.spec_from_file_location("active_speaker_score_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load active-speaker preprocessing: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def as_attr_dict(value):
    if isinstance(value, dict):
        return AttrDict({key: as_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [as_attr_dict(item) for item in value]
    return value


def load_request_bundle(request_dir: str | Path):
    root = Path(request_dir)
    request_path = root / "laser_requests.csv"
    config_path = root / "request_config.json"
    if not request_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("LASER request bundle requires CSV and request_config.json")
    config = json.loads(config_path.read_text("utf-8"))
    if config.get("schema") != "laser_request_v1":
        raise ValueError("Unsupported LASER request schema")
    with request_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"clip_id", "bin_index", "start_ms", "end_ms", "file_path"}
    if not rows or not required <= set(rows[0]):
        raise ValueError("LASER request CSV is empty or missing required columns")
    seen = set()
    for row in rows:
        key = (str(row["clip_id"]), int(row["bin_index"]))
        if key in seen:
            raise ValueError(f"Duplicate LASER request: {key}")
        if int(row["start_ms"]) < 0 or int(row["end_ms"]) <= int(row["start_ms"]):
            raise ValueError(f"Invalid LASER interval: {key}")
        if not str(row["file_path"]).strip():
            raise ValueError(f"LASER request has empty file_path: {key}")
        seen.add(key)
    if len(rows) != int(config.get("requested_bins", -1)):
        raise ValueError("LASER request count differs from request_config.json")
    if len({str(row["clip_id"]) for row in rows}) != int(
        config.get("requested_clips", -1)
    ):
        raise ValueError("LASER requested clip count differs from request_config.json")
    timeline_hash = str(config.get("source_timeline_sha256", ""))
    if len(timeline_hash) != 64:
        raise ValueError("LASER request bundle lacks a valid source timeline hash")
    return rows, config, request_path, config_path


def crop_loconet_face(frame: np.ndarray, bbox, padding: float = 0.775) -> np.ndarray:
    """Match the official demo's square crop and zero border behavior."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    radius = max(x2 - x1, y2 - y1) * padding
    left, top = int(center_x - radius), int(center_y - radius)
    right, bottom = int(center_x + radius), int(center_y + radius)
    if right <= left or bottom <= top:
        raise ValueError("invalid_loconet_face_bbox")
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - width), max(0, bottom - height)
    padded = cv2.copyMakeBorder(
        frame, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    left += pad_left
    right += pad_left
    top += pad_top
    bottom += pad_top
    crop = padded[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("empty_loconet_face_crop")
    crop = cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def build_track_faces(frames: list[np.ndarray], tracks: list[dict]) -> list[dict]:
    prepared = []
    for track in tracks:
        frame_ids = np.asarray(track["frames"], dtype=np.int64)
        if len(frame_ids) < MIN_TRACK_FRAMES:
            continue
        try:
            faces = np.stack([
                crop_loconet_face(frames[int(frame_id)], bbox)
                for frame_id, bbox in zip(frame_ids, track["bbox"])
            ])
        except (ValueError, cv2.error, IndexError):
            continue
        prepared.append({
            "track_id": int(track["track_id"]),
            "frames": frame_ids,
            "faces": faces,
        })
    return prepared


def align_context(primary: dict, candidate: dict) -> np.ndarray:
    by_frame = {
        int(frame): crop for frame, crop in zip(candidate["frames"], candidate["faces"])
    }
    return np.stack([
        by_frame.get(int(frame), np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.uint8))
        for frame in primary["frames"]
    ])


def visual_context(primary: dict, tracks: list[dict]) -> np.ndarray:
    primary_frames = set(int(value) for value in primary["frames"])
    candidates = []
    for track in tracks:
        if track["track_id"] == primary["track_id"]:
            continue
        overlap = len(primary_frames.intersection(int(value) for value in track["frames"]))
        if overlap >= len(primary_frames) / 2:
            candidates.append((overlap, -int(track["track_id"]), track))
    candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
    contexts = [align_context(primary, item[2]) for item in candidates[:2]]
    if not contexts:
        contexts = [primary["faces"], primary["faces"]]
    elif len(contexts) == 1:
        contexts.append(primary["faces"])
    return np.stack([primary["faces"], contexts[0], contexts[1]])


def extract_pcm16(path: str, wav_path: str) -> np.ndarray:
    result = subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", path, "-vn", "-ac", "1", "-ar", str(TARGET_SR),
        "-c:a", "pcm_s16le", wav_path,
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("laser_audio_decode_failed: " + result.stderr[-500:])
    sample_rate, audio = wavfile.read(wav_path)
    if sample_rate != TARGET_SR or audio.size == 0 or audio.dtype != np.int16:
        raise RuntimeError("laser_audio_decode_empty_or_wrong_format")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.int16)
    return audio


class LoCoNetLaser:
    N_CHANNEL = 4
    LAYER = 1

    def __init__(self, repo: str, weights: str, expected_revision: str,
                 expected_weights_sha256: str):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("LoCoNet+LASER runner requires CUDA")
        root = Path(repo).resolve()
        loconet_root = root / "LoCoNet"
        config_path = loconet_root / "configs" / "multi.yaml"
        source_path = loconet_root / "landmark_loconet.py"
        if not config_path.is_file() or not source_path.is_file():
            raise FileNotFoundError("Pinned LASER checkout lacks LoCoNet source/config")
        actual_revision = git_revision(root)
        if actual_revision == "UNAVAILABLE" or actual_revision != expected_revision:
            raise ValueError(
                f"LASER checkout revision mismatch: expected={expected_revision}, "
                f"actual={actual_revision}"
            )
        if not git_model_sources_are_clean(root):
            raise ValueError("LASER checkout has modified Python/config source files")
        weights_path = Path(weights).resolve()
        if not weights_path.is_file():
            raise FileNotFoundError("LoCoNet+LASER checkpoint not found")
        weights_sha256 = require_sha256(
            weights_path, expected_weights_sha256, "LoCoNet+LASER checkpoint"
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        landmark_weight = state.get("model.module.landmark_bottleneck.weight")
        bottle_weight = state.get("model.module.bottle_neck.weight")
        if tuple(getattr(landmark_weight, "shape", ())) != (4, 164, 1, 1):
            raise ValueError("Checkpoint is not the expected n_channel=4 LASER model")
        if tuple(getattr(bottle_weight, "shape", ())) != (64, 68, 1, 1):
            raise ValueError("Checkpoint is not the expected layer=1 LASER model")

        cfg = as_attr_dict(yaml.safe_load(config_path.read_text("utf-8")))
        sys.path.insert(0, str(loconet_root))
        try:
            from landmark_loconet import loconet
            from torchvggish import vggish_input
            # The complete LASER checkpoint contains every VGGish parameter.
            # Suppress the constructor's unrelated, unpinned bootstrap download.
            with mock.patch("torch.hub.load_state_dict_from_url", return_value={}):
                wrapper = loconet(cfg, n_channel=self.N_CHANNEL, layer=self.LAYER)
        finally:
            sys.path.pop(0)
        mapped = {
            key.replace("model.module.", "model.", 1): value
            for key, value in state.items()
        }
        incompatibility = wrapper.load_state_dict(mapped, strict=False)
        if incompatibility.missing_keys or incompatibility.unexpected_keys:
            raise ValueError(
                "LASER checkpoint/source mismatch: "
                f"missing={incompatibility.missing_keys}, "
                f"unexpected={incompatibility.unexpected_keys}"
            )
        self.wrapper = wrapper.eval()
        self.vggish_input = vggish_input
        self.torch = torch
        self.repo_revision = actual_revision
        self.repo = root
        self.weights = weights_path
        self.weights_sha256 = weights_sha256
        self.config_path = config_path

    def score_track(self, audio: np.ndarray, primary: dict, tracks: list[dict]) -> np.ndarray:
        torch = self.torch
        visual = visual_context(primary, tracks)
        frame_count = visual.shape[1]
        start_sample = int(int(primary["frames"][0]) * TARGET_SR / TARGET_FPS)
        end_sample = start_sample + int(frame_count * TARGET_SR / TARGET_FPS)
        segment = audio[start_sample:end_sample]
        expected_samples = end_sample - start_sample
        if len(segment) < expected_samples:
            segment = np.pad(segment, (0, expected_samples - len(segment)))
        audio_feature = self.vggish_input.waveform_to_examples(
            segment, TARGET_SR, frame_count, TARGET_FPS, return_tensor=False
        )
        input_audio = torch.as_tensor(
            audio_feature, dtype=torch.float32, device="cuda"
        ).unsqueeze(0).unsqueeze(0)
        input_visual = torch.as_tensor(
            visual, dtype=torch.float32, device="cuda"
        ).unsqueeze(0)
        batch, speakers, frames = input_visual.shape[:3]
        landmark_feature = torch.zeros(
            (batch * speakers * frames, self.N_CHANNEL, 28, 28),
            dtype=input_visual.dtype,
            device=input_visual.device,
        )
        core = self.wrapper.model
        with torch.no_grad():
            audio_embed = core.model.forward_audio_frontend(input_audio)
            visual_flat = input_visual.view(batch * speakers, *input_visual.shape[2:])
            visual_embed = core.forward_visual_frontend(visual_flat, landmark_feature)
            audio_embed = audio_embed.repeat(speakers, 1, 1)
            audio_embed, visual_embed = core.model.forward_cross_attention(
                audio_embed, visual_embed
            )
            fused = core.model.forward_audio_visual_backend(
                audio_embed, visual_embed, batch, speakers
            )
            primary_fused = fused.view(batch, speakers, frames, -1)[:, 0]
            logits = core.lossAV.FC(primary_fused.reshape(batch * frames, -1))
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
        result = probabilities.detach().cpu().numpy().astype(np.float32)
        if len(result) != frame_count or not np.isfinite(result).all():
            raise RuntimeError("LASER returned invalid frame probabilities")
        return result


def aggregate_requested_bins(request_rows: list[dict], track_scores: list[dict]):
    """Max over per-track median frame probabilities for each requested bin."""
    output, missing = [], []
    for request in request_rows:
        start_frame = math.floor(int(request["start_ms"]) * TARGET_FPS / 1000)
        end_frame = math.ceil(int(request["end_ms"]) * TARGET_FPS / 1000)
        candidates = []
        for track in track_scores:
            frame_ids = np.asarray(track["frames"])
            scores = np.asarray(track["scores"], dtype=np.float32)
            mask = (frame_ids >= start_frame) & (frame_ids < end_frame)
            if mask.any():
                candidates.append(float(np.median(scores[mask])))
        key = (str(request["clip_id"]), int(request["bin_index"]))
        if candidates:
            output.append({
                "clip_id": key[0],
                "bin_index": key[1],
                "laser_score": max(candidates),
            })
        else:
            missing.append(key)
    return output, missing


def write_bundle(out_dir: Path, scores: list[dict], failures: list[dict], metadata: dict):
    out_dir.mkdir(parents=True, exist_ok=False)
    score_tmp = out_dir / "laser_scores.jsonl.partial"
    failure_tmp = out_dir / "failures.csv.partial"
    metadata_tmp = out_dir / "laser_metadata.json.partial"
    with score_tmp.open("w", encoding="utf-8") as handle:
        for row in scores:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = ["clip_id", "bin_index", "error_type", "error_message"]
    with failure_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)
    metadata_tmp.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    os.replace(score_tmp, out_dir / "laser_scores.jsonl")
    os.replace(failure_tmp, out_dir / "failures.csv")
    os.replace(metadata_tmp, out_dir / "laser_metadata.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests_dir", required=True)
    parser.add_argument("--laser_repo", required=True)
    parser.add_argument("--laser_weights", required=True)
    parser.add_argument("--laser_weights_sha256", required=True)
    parser.add_argument("--laser_revision", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--detect_every", type=int, default=2)
    parser.add_argument("--det_size", type=int, default=640)
    args = parser.parse_args()

    output = Path(args.out_dir)
    if output.exists():
        raise FileExistsError(f"Immutable LASER output exists: {output}")
    if args.detect_every <= 0 or args.det_size <= 0:
        raise ValueError("detect_every and det_size must be positive")
    requests, request_config, request_path, request_config_path = load_request_bundle(
        args.requests_dir
    )
    by_clip = {}
    for row in requests:
        by_clip.setdefault(str(row["clip_id"]), []).append(row)

    preprocessing = load_score_preprocessing_module()
    tracker = preprocessing.FaceTracker(args.det_size, args.detect_every)
    model = LoCoNetLaser(
        args.laser_repo, args.laser_weights, args.laser_revision,
        args.laser_weights_sha256,
    )
    started = time.time()
    all_scores, failures = [], []
    for index, (clip_id, clip_requests) in enumerate(sorted(by_clip.items()), 1):
        media_paths = {str(row["file_path"]) for row in clip_requests}
        if len(media_paths) != 1:
            raise ValueError(f"One clip maps to multiple media paths: {clip_id}")
        path = next(iter(media_paths))
        try:
            frames = preprocessing.decode_25fps(path)
            tracks = build_track_faces(frames, tracker.track(frames))
            if not tracks:
                raise RuntimeError("no_eligible_loconet_face_track")
            with tempfile.TemporaryDirectory(prefix="laser_audio_") as temp_dir:
                audio = extract_pcm16(path, os.path.join(temp_dir, "audio.wav"))
            track_scores = []
            for track in tracks:
                probabilities = model.score_track(audio, track, tracks)
                track_scores.append({
                    "track_id": track["track_id"],
                    "frames": track["frames"],
                    "scores": probabilities,
                })
            scores, missing = aggregate_requested_bins(clip_requests, track_scores)
            all_scores.extend(scores)
            for _, bin_index in missing:
                failures.append({
                    "clip_id": clip_id,
                    "bin_index": bin_index,
                    "error_type": "LaserBinUnscored",
                    "error_message": "No eligible tracked face overlaps the requested bin",
                })
        except Exception as error:
            for row in clip_requests:
                failures.append({
                    "clip_id": clip_id,
                    "bin_index": int(row["bin_index"]),
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:500],
                })
        print(f"[{index}/{len(by_clip)}] scores={len(all_scores)} failures={len(failures)}")

    scored_keys = {(row["clip_id"], int(row["bin_index"])) for row in all_scores}
    requested_keys = {
        (str(row["clip_id"]), int(row["bin_index"])) for row in requests
    }
    model_identity = {
        "backbone": "LoCoNet+LASER",
        "official_repo": "https://github.com/plnguyen2908/LASER_ASD",
        "model_git_sha": model.repo_revision,
        "weights_sha256": model.weights_sha256,
        "official_config_sha256": sha256_file(model.config_path),
        "runner_sha256": sha256_file(__file__),
        "source_timeline_sha256": request_config["source_timeline_sha256"],
        "request_csv_sha256": sha256_file(request_path),
        "request_config_sha256": sha256_file(request_config_path),
        "preprocessing": {
            "fps": TARGET_FPS,
            "sample_rate": TARGET_SR,
            "face_size": FACE_SIZE,
            "face_crop": "official_demo_square_padding_0.775",
            "tracker": "insightface_greedy_iou_0.25_interp_v1",
            "detect_every": args.detect_every,
            "det_size": args.det_size,
            "min_track_frames": MIN_TRACK_FRAMES,
            "landmarks_at_inference": False,
            "speaker_context": "primary_plus_two_overlap_deterministic_v1",
            "bin_aggregation": "max_of_per_track_median_probability_v1",
            "logit_conversion": "two_class_softmax_speaking_probability",
        },
    }
    metadata = {
        "schema": "laser_sidecar_v1",
        **model_identity,
        "config_hash": canonical_hash(model_identity),
        "requested_bins": len(requested_keys),
        "scored_bins": len(scored_keys),
        "missing_bins": len(requested_keys - scored_keys),
        "requested_clips": len(by_clip),
        "failure_rows": len(failures),
        "coverage_passed": scored_keys == requested_keys,
        "elapsed_seconds": round(time.time() - started, 3),
        "score_semantics": (
            "maximum across face tracks of the median per-frame speaking "
            "probability inside each requested 200 ms bin"
        ),
    }
    write_bundle(output, all_scores, failures, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
