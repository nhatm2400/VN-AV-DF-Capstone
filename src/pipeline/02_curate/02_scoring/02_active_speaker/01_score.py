"""Temporal active-speaker measurement for source real clips.

This stage is intentionally placed after ``all_manifest.csv`` and before
``04_curate.py``.  It never mutates media and never silently drops a clip.

Light-ASD is loaded from an explicit checkout of the official repository. Bins
that need LASER are marked ``laser_requested`` and enriched in the separate
``05_apply_laser_scores.py`` step. Until then they remain manual, never reject.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.io import wavfile

from policy import TemporalPolicy, summarize_timeline


TARGET_FPS = 25
TARGET_SR = 16000
FACE_SIZE = 112
# Five video frames (200 ms) yield fewer than the 20 MFCC steps required for
# five Light-ASD outputs. Six frames are the shortest valid track at 25 fps.
MIN_LIGHT_TRACK_FRAMES = 6
ALIGN_TEMPLATE = np.array(
    [[38.3, 51.7], [73.5, 51.5], [56.0, 71.7], [41.5, 92.4], [70.7, 92.2]],
    dtype=np.float32,
)
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def git_revision(path: str) -> str:
    result = subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", "utf-8")
    os.replace(tmp, path)


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def decode_25fps(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("video_open_failed")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(source_fps) or source_fps <= 0:
        cap.release()
        raise RuntimeError("invalid_fps")
    source = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        source.append(frame)
    cap.release()
    if not source:
        raise RuntimeError("video_decode_empty")
    duration = len(source) / source_fps
    target_n = max(1, int(round(duration * TARGET_FPS)))
    indices = np.clip(
        np.round(np.arange(target_n) * source_fps / TARGET_FPS).astype(int),
        0,
        len(source) - 1,
    )
    return [source[index] for index in indices]


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Convert decoded PCM to the [-1, 1] float range expected by Silero."""
    if np.issubdtype(audio.dtype, np.integer):
        scale = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def extract_audio(path: str, wav_path: str) -> np.ndarray:
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", path, "-vn", "-ac", "1", "-ar", str(TARGET_SR), "-c:a", "pcm_s16le",
        wav_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("audio_decode_failed: " + result.stderr[-500:])
    sr, audio = wavfile.read(wav_path)
    if sr != TARGET_SR or audio.size == 0:
        raise RuntimeError("audio_decode_empty_or_wrong_rate")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return normalize_audio(audio)


def resolve_silero_onnx(repo: str) -> Path:
    """Resolve the pinned, 16 kHz Silero model without importing torchaudio."""
    if not repo:
        raise ValueError("--silero_repo is required; remote/unpinned VAD is disabled")
    root = Path(repo)
    candidates = [
        root / "src" / "silero_vad" / "data" / "silero_vad_16k_op15.onnx",
        root / "src" / "silero_vad" / "data" / "silero_vad.onnx",
        root / "files" / "silero_vad.onnx",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Pinned Silero checkout does not contain a supported 16 kHz ONNX model"
    )


class SileroVAD:
    """Small ONNX adapter for Silero VAD's official 16 kHz model.

    The official hub module imports ``torchaudio`` even though media has already
    been decoded by ffmpeg.  Running the bundled ONNX file keeps the inference
    pinned and removes that unnecessary binary dependency.
    """

    WINDOW = 512
    CONTEXT = 64

    def __init__(self, repo: str):
        import onnxruntime as ort

        self.model_path = resolve_silero_onnx(repo)
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"],
            sess_options=options,
        )

    def probabilities(self, audio: np.ndarray) -> list[float]:
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, self.CONTEXT), dtype=np.float32)
        values = np.asarray(audio, dtype=np.float32).reshape(-1)
        probabilities = []
        for start in range(0, len(values), self.WINDOW):
            chunk = values[start:start + self.WINDOW]
            if len(chunk) < self.WINDOW:
                chunk = np.pad(chunk, (0, self.WINDOW - len(chunk)))
            model_input = np.concatenate((context, chunk.reshape(1, -1)), axis=1)
            output, state = self.session.run(
                None,
                {
                    "input": model_input.astype(np.float32, copy=False),
                    "state": state,
                    "sr": np.asarray(TARGET_SR, dtype=np.int64),
                },
            )
            probabilities.append(float(np.asarray(output).reshape(-1)[0]))
            context = model_input[:, -self.CONTEXT:]
        return probabilities

    @staticmethod
    def speech_timestamps(
        probabilities: list[float], audio_samples: int,
        threshold: float = 0.5, neg_threshold: float = 0.35,
        min_speech_ms: int = 250, min_silence_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> list[dict[str, int]]:
        """Convert 32 ms probabilities into speech spans using Silero defaults."""
        min_speech = TARGET_SR * min_speech_ms / 1000
        min_silence = TARGET_SR * min_silence_ms / 1000
        pad = int(TARGET_SR * speech_pad_ms / 1000)
        triggered = False
        start_sample = 0
        possible_end = None
        spans = []
        for index, probability in enumerate(probabilities):
            current = index * SileroVAD.WINDOW
            if probability >= threshold:
                if not triggered:
                    triggered = True
                    start_sample = current
                possible_end = None
                continue
            if triggered and probability < neg_threshold:
                if possible_end is None:
                    possible_end = current
                if current - possible_end >= min_silence:
                    if possible_end - start_sample > min_speech:
                        spans.append({"start": start_sample, "end": possible_end})
                    triggered = False
                    possible_end = None
        if triggered and audio_samples - start_sample > min_speech:
            spans.append({"start": start_sample, "end": audio_samples})

        for index, span in enumerate(spans):
            span["start"] = max(0, span["start"] - pad)
            span["end"] = min(audio_samples, span["end"] + pad)
            if index and span["start"] < spans[index - 1]["end"]:
                midpoint = (span["start"] + spans[index - 1]["end"]) // 2
                spans[index - 1]["end"] = midpoint
                span["start"] = midpoint
        return spans

    def bins(self, audio: np.ndarray, count: int, bin_ms: int) -> list[bool]:
        timestamps = self.speech_timestamps(
            self.probabilities(audio), len(audio),
        )
        out = []
        samples_per_bin = int(TARGET_SR * bin_ms / 1000)
        for index in range(count):
            start, end = index * samples_per_bin, (index + 1) * samples_per_bin
            overlap = any(int(item["start"]) < end and int(item["end"]) > start
                          for item in timestamps)
            out.append(overlap)
        return out


class FaceTracker:
    """Dense multi-face tracks with five-point landmark interpolation."""

    def __init__(self, det_size=640, detect_every=2):
        try:
            import onnxruntime as ort
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
        except Exception:
            pass
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        detector = self.app.models.get("detection")
        self.detector_model_path = Path(getattr(detector, "model_file", ""))
        if not self.detector_model_path.is_file():
            raise FileNotFoundError("InsightFace detection model file is unavailable")
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))
        self.detect_every = detect_every

    def track(self, frames: list[np.ndarray]) -> list[dict]:
        tracks = []
        for frame_index in range(0, len(frames), self.detect_every):
            detections = []
            for face in self.app.get(frames[frame_index]):
                if float(getattr(face, "det_score", 1.0)) < 0.5:
                    continue
                kps = getattr(face, "kps", None)
                if kps is None or np.asarray(kps).shape != (5, 2):
                    continue
                detections.append({
                    "frame": frame_index,
                    "bbox": np.asarray(face.bbox, dtype=np.float32),
                    "kps": np.asarray(kps, dtype=np.float32),
                })

            candidates = []
            for track_index, track in enumerate(tracks):
                gap = frame_index - track["points"][-1]["frame"]
                if gap <= max(8, self.detect_every * 4):
                    for detection_index, detection in enumerate(detections):
                        score = iou(track["points"][-1]["bbox"], detection["bbox"])
                        if score >= 0.25:
                            candidates.append((score, track_index, detection_index))
            used_tracks, used_detections = set(), set()
            for _, track_index, detection_index in sorted(candidates, reverse=True):
                if track_index in used_tracks or detection_index in used_detections:
                    continue
                tracks[track_index]["points"].append(detections[detection_index])
                used_tracks.add(track_index)
                used_detections.add(detection_index)
            for detection_index, detection in enumerate(detections):
                if detection_index not in used_detections:
                    tracks.append({"track_id": len(tracks), "points": [detection]})

        dense = []
        for track in tracks:
            points = track["points"]
            if len(points) < 2:
                continue
            start, end = points[0]["frame"], points[-1]["frame"]
            xp = np.asarray([point["frame"] for point in points])
            frame_ids = np.arange(start, end + 1)
            bbox = np.stack([
                np.interp(frame_ids, xp, [point["bbox"][axis] for point in points])
                for axis in range(4)
            ], axis=1)
            kps = np.empty((len(frame_ids), 5, 2), dtype=np.float32)
            for landmark in range(5):
                for axis in range(2):
                    kps[:, landmark, axis] = np.interp(
                        frame_ids, xp, [point["kps"][landmark, axis] for point in points]
                    )
            dense.append({"track_id": track["track_id"], "frames": frame_ids,
                          "bbox": bbox, "kps": kps})
        return dense


def aligned_face_and_mouth(frame, bbox, kps):
    transform, _ = cv2.estimateAffinePartial2D(
        np.asarray(kps, np.float32), ALIGN_TEMPLATE, method=cv2.LMEDS
    )
    if transform is None:
        raise RuntimeError("landmark_alignment_failed")
    aligned = cv2.warpAffine(frame, transform, (FACE_SIZE, FACE_SIZE),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    mouth = gray[72:110, 25:87]

    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    x1, x2 = x1 - 0.15 * width, x2 + 0.15 * width
    y1, y2 = y1 - 0.15 * height, y2 + 0.15 * height
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("invalid_face_crop")
    face = cv2.resize(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY),
                      (FACE_SIZE, FACE_SIZE))
    return face, mouth


class LightASD:
    def __init__(self, repo: str, weights: str):
        if not os.path.isfile(os.path.join(repo, "ASD.py")):
            raise FileNotFoundError("Light-ASD checkout missing ASD.py")
        if not os.path.isfile(weights):
            raise FileNotFoundError("Light-ASD weights not found")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Light-ASD official implementation requires CUDA")
        sys.path.insert(0, os.path.abspath(repo))
        try:
            from ASD import ASD
            self.model = ASD()
            self.model.loadParameters(weights)
            self.model.eval()
        finally:
            sys.path.pop(0)

    def score(self, audio: np.ndarray, faces: np.ndarray, start_frame: int) -> np.ndarray:
        import python_speech_features
        import torch

        start_sample = int(start_frame * TARGET_SR / TARGET_FPS)
        end_sample = start_sample + int(len(faces) * TARGET_SR / TARGET_FPS)
        segment = audio[start_sample:end_sample]
        mfcc = python_speech_features.mfcc(
            segment, TARGET_SR, numcep=13, winlen=0.025, winstep=0.010
        )
        usable = min(len(faces), len(mfcc) // 4)
        if usable < 5:
            raise RuntimeError("light_asd_track_too_short")
        input_a = torch.as_tensor(mfcc[:usable * 4], dtype=torch.float32,
                                  device="cuda").unsqueeze(0)
        input_v = torch.as_tensor(faces[:usable], dtype=torch.float32,
                                  device="cuda").unsqueeze(0)
        with torch.no_grad():
            audio_embed = self.model.model.forward_audio_frontend(input_a)
            visual_embed = self.model.model.forward_visual_frontend(input_v)
            output = self.model.model.forward_audio_visual_backend(audio_embed, visual_embed)
            scores = self.model.lossAV.forward(output, labels=None)
        return np.asarray(scores, dtype=np.float32)


def score_clip(clip_id, path, tracker, light, vad, laser, policy):
    frames = decode_25fps(path)
    bin_frames = int(TARGET_FPS * policy.bin_ms / 1000)
    bin_count = max(1, math.ceil(len(frames) / bin_frames))
    with tempfile.TemporaryDirectory(prefix="asd_audio_") as temp_dir:
        audio = extract_audio(path, os.path.join(temp_dir, "audio.wav"))
    speech = vad.bins(audio, bin_count, policy.bin_ms)
    tracks = tracker.track(frames)

    per_frame = [[] for _ in frames]
    tracked_face_present = [False for _ in frames]
    for track in tracks:
        faces, mouths, valid_frames = [], [], []
        for frame_id, bbox, kps in zip(track["frames"], track["bbox"], track["kps"]):
            tracked_face_present[int(frame_id)] = True
            face, mouth = aligned_face_and_mouth(frames[int(frame_id)], bbox, kps)
            faces.append(face)
            mouths.append(mouth)
            valid_frames.append(int(frame_id))
        if len(faces) < MIN_LIGHT_TRACK_FRAMES:
            continue
        scores = light.score(audio, np.asarray(faces), valid_frames[0])
        usable = min(len(scores), len(valid_frames))
        previous_mouth = None
        for index in range(usable):
            motion = None if previous_mouth is None else float(np.mean(
                cv2.absdiff(mouths[index], previous_mouth)
            ))
            previous_mouth = mouths[index]
            per_frame[valid_frames[index]].append({
                "track_id": track["track_id"],
                "light_asd_score": float(scores[index]),
                "mouth_motion": motion,
            })

    timeline = []
    for bin_index in range(bin_count):
        start = bin_index * bin_frames
        end = min(len(frames), (bin_index + 1) * bin_frames)
        candidates = {}
        for frame_candidates in per_frame[start:end]:
            for item in frame_candidates:
                candidates.setdefault(item["track_id"], []).append(item)
        track_scores = []
        for track_id, items in candidates.items():
            light_score = float(np.median([item["light_asd_score"] for item in items]))
            motions = [item["mouth_motion"] for item in items if item["mouth_motion"] is not None]
            motion = float(np.median(motions)) if motions else None
            track_scores.append((light_score, track_id, motion))
        track_scores.sort(reverse=True)
        best = track_scores[0] if track_scores else (None, -1, None)
        competing = len(track_scores) > 1 and track_scores[1][0] is not None and (
            best[0] is None or abs(best[0] - track_scores[1][0]) < policy.light_margin
        )
        near = best[0] is not None and abs(best[0] - policy.light_active_threshold) < policy.light_margin
        motion_conflict = (
            best[0] is not None and best[2] is not None
            and best[0] >= policy.light_active_threshold + policy.light_margin
            and best[2] <= policy.mouth_freeze_threshold
        )
        needs_laser = bool(speech[bin_index] and (near or competing or motion_conflict))
        laser_score = laser.get((clip_id, bin_index)) if needs_laser else None
        disagreement = bool(laser_score is not None and best[0] is not None and (
            (best[0] >= policy.light_active_threshold + policy.light_margin
             and laser_score <= policy.laser_active_threshold - policy.laser_margin)
            or (best[0] <= policy.light_active_threshold - policy.light_margin
                and laser_score >= policy.laser_active_threshold + policy.laser_margin)
        ))
        timeline.append({
            "clip_id": clip_id,
            "bin_index": bin_index,
            "start_ms": bin_index * policy.bin_ms,
            "end_ms": (bin_index + 1) * policy.bin_ms,
            "speech": bool(speech[bin_index]),
            "face_visible": bool(track_scores),
            "selected_track_id": int(best[1]),
            "face_track_count": len(track_scores),
            "mouth_motion": best[2],
            "mouth_frozen": best[2] is not None and best[2] <= policy.mouth_freeze_threshold,
            "light_asd_score": best[0],
            "laser_score": laser_score,
            "laser_requested": needs_laser,
            "asd_disagreement": disagreement,
            "multiple_competing_faces": competing,
            "inference_failure": bool(
                speech[bin_index]
                and any(tracked_face_present[start:end])
                and not track_scores
            ),
        })
    return summarize_timeline(timeline, policy), timeline


def write_outputs(output_dir, summaries, timeline, failures, config):
    output_dir.mkdir(parents=True, exist_ok=False)
    scores_path = output_dir / "asd_clip_scores.csv"
    timeline_path = output_dir / "asd_timeline.jsonl.gz"
    failures_path = output_dir / "failures.csv"
    pd.DataFrame(summaries).to_csv(str(scores_path) + ".partial", index=False)
    with gzip.open(str(timeline_path) + ".partial", "wt", encoding="utf-8") as handle:
        for row in timeline:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = ["clip_id", "file_path", "error_type", "error_message"]
    with open(str(failures_path) + ".partial", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)
    atomic_json(output_dir / "run_config.json", config)
    os.replace(str(scores_path) + ".partial", scores_path)
    os.replace(str(timeline_path) + ".partial", timeline_path)
    os.replace(str(failures_path) + ".partial", failures_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/01_collect/cut_clips/all_manifest.csv")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--runs_root", default="data/02_curate/runs")
    parser.add_argument("--path_col", default="file_path")
    parser.add_argument("--batch_start", type=int, default=0)
    parser.add_argument("--batch_end", type=int, default=None)
    parser.add_argument("--shard_id", default="")
    parser.add_argument("--light_asd_dir", required=True)
    parser.add_argument("--light_weights", required=True)
    parser.add_argument("--light_revision", default="",
                        help="explicit commit SHA when checkout has no .git metadata")
    parser.add_argument("--silero_repo", default="")
    parser.add_argument("--silero_revision", default="")
    parser.add_argument("--allow_unpinned_models", action="store_true",
                        help="development smoke only; never use for calibration/full publish")
    parser.add_argument("--detect_every", type=int, default=2)
    parser.add_argument("--det_size", type=int, default=640)
    parser.add_argument("--policy", default="", help="JSON policy produced by calibration")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, low_memory=False)
    required = {"clip_id", args.path_col}
    missing = sorted(required - set(manifest.columns))
    if missing or manifest.empty or manifest["clip_id"].isna().any() or manifest["clip_id"].duplicated().any():
        raise ValueError(f"Invalid manifest; missing={missing}")
    end = len(manifest) if args.batch_end is None else args.batch_end
    if not (0 <= args.batch_start < end <= len(manifest)):
        raise ValueError("Invalid batch range")
    batch = manifest.iloc[args.batch_start:end].copy()
    if len(batch) > 5000:
        raise ValueError("A scoring shard may contain at most 5,000 clips")
    absent = [str(path) for path in batch[args.path_col] if not os.path.isfile(str(path))]
    if absent:
        raise FileNotFoundError(f"{len(absent)} media files missing; examples={absent[:3]}")

    run_root = Path(args.runs_root) / args.run_id
    output_dir = run_root / "shards" / args.shard_id if args.shard_id else run_root
    if output_dir.exists():
        raise FileExistsError(f"Immutable output already exists: {output_dir}")

    policy_values = {}
    if args.policy:
        policy_doc = json.loads(Path(args.policy).read_text("utf-8"))
        if policy_doc.get("schema") != "active_speaker_policy_v1":
            raise ValueError("Unknown temporal policy schema")
        policy_values = policy_doc["policy"]
    policy = TemporalPolicy(**policy_values)
    preprocessing_config = {
        "schema": "active_speaker_preprocessing_v1",
        "target_fps": TARGET_FPS,
        "target_sample_rate": TARGET_SR,
        "bin_ms": policy.bin_ms,
        "detector": "insightface_buffalo_l",
        "det_size": args.det_size,
        "detect_every": args.detect_every,
        "tracker": "greedy_iou_0.25_interp_v1",
        "alignment": "insightface_5point_affine_v1",
        "vad_adapter": "silero_onnx_16k_hysteresis_v1",
        "scorer_sha256": sha256_file(Path(__file__).resolve()),
        "temporal_policy_code_sha256": sha256_file(
            Path(__file__).with_name("policy.py").resolve()
        ),
    }
    preprocessing_config_hash = canonical_hash(preprocessing_config)
    light_revision = args.light_revision or git_revision(args.light_asd_dir)
    silero_revision = args.silero_revision or (
        git_revision(args.silero_repo) if args.silero_repo else "UNPINNED"
    )
    silero_model_path = resolve_silero_onnx(args.silero_repo)
    unpinned = (
        not args.silero_repo
        or light_revision in {"", "UNAVAILABLE", "UNPINNED"}
        or silero_revision in {"", "UNAVAILABLE", "UNPINNED"}
    )
    if unpinned and not args.allow_unpinned_models:
        raise ValueError(
            "Model revisions are not pinned. Clone repos with .git metadata or pass "
            "--light_revision/--silero_revision; --allow_unpinned_models is smoke-only."
        )

    print(f"Scoring {len(batch)} clips [{args.batch_start}:{end}] -> {output_dir}")
    started = time.time()
    tracker = FaceTracker(args.det_size, args.detect_every)
    model_versions = {
        "light_asd_repo": os.path.abspath(args.light_asd_dir),
        "light_asd_git_sha": light_revision,
        "light_weights_sha256": sha256_file(args.light_weights),
        "laser_model": None,
        "silero_source": os.path.abspath(args.silero_repo) if args.silero_repo else "snakers4/silero-vad",
        "silero_git_sha": silero_revision,
        "silero_weights_sha256": sha256_file(silero_model_path),
        "insightface_package_version": importlib.metadata.version("insightface"),
        "onnxruntime_package_version": importlib.metadata.version("onnxruntime-gpu"),
        "insightface_detector_file": tracker.detector_model_path.name,
        "insightface_detector_sha256": sha256_file(tracker.detector_model_path),
        "preprocessing_config_hash": preprocessing_config_hash,
    }
    inference_config = {
        "schema": "active_speaker_inference_v1",
        "manifest_sha256": sha256_file(args.manifest),
        "detect_every": args.detect_every,
        "det_size": args.det_size,
        "policy": asdict(policy),
        "model_versions": model_versions,
        "preprocessing": preprocessing_config,
    }
    config_hash = canonical_hash(inference_config)
    config_payload = {
        "schema": "active_speaker_run_v1",
        "run_id": args.run_id,
        "manifest_sha256": inference_config["manifest_sha256"],
        "manifest_rows": len(manifest),
        "batch_start": args.batch_start,
        "batch_end": end,
        "shard_id": args.shard_id,
        "detect_every": args.detect_every,
        "det_size": args.det_size,
        "policy": inference_config["policy"],
        "policy_document": os.path.abspath(args.policy) if args.policy else None,
        "policy_document_sha256": sha256_file(args.policy) if args.policy else None,
        "policy_gate_passed": bool(policy_doc.get("gate_passed", False)) if args.policy else False,
        "model_versions": model_versions,
        "preprocessing": preprocessing_config,
        "config_hash": config_hash,
    }
    light = LightASD(args.light_asd_dir, args.light_weights)
    vad = SileroVAD(args.silero_repo)
    laser = {}
    summaries, timeline, failures = [], [], []
    versions_json = json.dumps(model_versions, sort_keys=True, separators=(",", ":"))
    for number, row in enumerate(batch.itertuples(index=False), 1):
        clip_id = str(row.clip_id)
        path = str(getattr(row, args.path_col))
        try:
            summary, clip_timeline = score_clip(
                clip_id, path, tracker, light, vad, laser, policy
            )
        except Exception as error:
            failures.append({"clip_id": clip_id, "file_path": path,
                             "error_type": type(error).__name__, "error_message": str(error)})
            clip_timeline = [{
                "clip_id": clip_id, "bin_index": 0, "start_ms": 0,
                "end_ms": policy.bin_ms, "speech": True, "face_visible": False,
                "inference_failure": True,
            }]
            summary = summarize_timeline(clip_timeline, policy)
        summary.update(clip_id=clip_id, model_versions=versions_json,
                       config_hash=config_hash)
        summaries.append(summary)
        timeline.extend(clip_timeline)
        if number % 25 == 0 or number == len(batch):
            print(f"[{number}/{len(batch)}] failures={len(failures)}")

    config_payload.update({
        "elapsed_seconds": round(time.time() - started, 3),
        "output_rows": len(summaries),
        "timeline_rows": len(timeline),
        "failures": len(failures),
        "coverage_passed": len(summaries) == len(batch)
                           and len({row["clip_id"] for row in summaries}) == len(batch),
    })
    if not config_payload["coverage_passed"]:
        raise RuntimeError("Scorer coverage failed; refusing to publish")
    write_outputs(output_dir, summaries, timeline, failures, config_payload)
    print(json.dumps({key: config_payload[key] for key in (
        "run_id", "output_rows", "timeline_rows", "failures", "coverage_passed"
    )}, indent=2))


if __name__ == "__main__":
    main()
