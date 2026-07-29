"""Stage 04 Cut Clips dùng chung cho Tier 1/2/3.

Khác notebook cũ:

- CUDA decode thất bại sẽ retry software decode, có kiểm return code/stderr.
- NVENC cut thất bại sẽ retry libx264 và chỉ publish file đã probe hợp lệ.
- clip_id dựa trên source + biên thời gian, không phụ thuộc accepted order.
- mỗi video input có đúng một terminal row trong ``video_status.csv``.
- mỗi batch ghi vào run directory bất biến và tự kiểm coverage trước khi publish.

Notebook Kaggle chỉ cài dependency, checkout đúng Git SHA rồi gọi ``main()``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


ACCEPTED_FIELDS = [
    "clip_id", "source_video", "start_time", "end_time", "duration",
    "face_ratio", "speech_ratio", "snr", "file_path", "tier",
    "decode_backend", "cut_backend", "run_id",
]
REJECT_FIELDS = [
    "video", "filename", "start", "end", "reason", "detail", "run_id", "tier",
]
STATUS_FIELDS = [
    "filename", "video_id", "status", "accepted_count", "rejected_count",
    "decode_backend", "decode_attempts", "cut_backends", "terminal_detail",
    "run_id", "tier",
]


@dataclass
class CutConfig:
    tier: str
    dataset_dir: str
    input_csv: str
    output_root: str = "/kaggle/working"
    run_id: str = ""
    start_index: int = 0
    end_index: int = 0
    expected_input_count: int = 0
    num_workers: int = 4
    face_model: str = "yolov8n-face.pt"
    use_hwaccel_decode: bool = True
    use_nvenc: bool = True
    create_zip: bool = True
    hash_media: bool = True
    vad_min_duration: float = 2.0
    vad_max_duration: float = 12.0
    window_size: float = 5.0
    window_step: float = 4.0
    tail_min_duration: float = 2.5
    yolo_pass_ratio: float = 0.7
    min_speech_ratio: float = 0.4
    histogram_correlation_threshold: float = 0.6
    min_snr: float = -999.0
    audio_bitrate: str = "128k"
    nvenc_cq: int = 23
    nvenc_preset: str = "p4"
    x264_crf: int = 18
    x264_preset: str = "veryfast"
    yolo_batch: int = 256
    sample_fps: float = 1.0
    frame_size: int = 320
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        known = {field.name for field in cls.__dataclass_fields__.values()}
        config = {key: value for key, value in payload.items() if key in known}
        config["extra"] = {
            key: value for key, value in payload.items() if key not in known
        }
        return cls(**config)

    def validate(self):
        if self.tier not in {"tier1", "tier2", "tier3"}:
            raise ValueError(f"tier không hợp lệ: {self.tier}")
        if not self.run_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.run_id):
            raise ValueError("run_id bắt buộc và chỉ gồm chữ/số/._-")
        for name in ("dataset_dir", "input_csv", "output_root"):
            value = str(getattr(self, name))
            if not value or value.startswith("__"):
                raise ValueError(f"{name} chưa được cấu hình: {value}")
        if self.start_index < 0 or self.end_index < 0:
            raise ValueError("batch index không được âm")
        if self.end_index and self.end_index <= self.start_index:
            raise ValueError("end_index phải lớn hơn start_index hoặc bằng 0")
        if self.expected_input_count <= 0:
            raise ValueError("expected_input_count phải được khóa từ Stage 03")
        if self.num_workers < 1:
            raise ValueError("num_workers phải >= 1")


@dataclass
class Attempt:
    backend: str
    returncode: int
    stderr: str
    valid: bool


@dataclass
class DecodeResult:
    frames: list
    backend: str
    attempts: list[Attempt]


@dataclass
class CutResult:
    ok: bool
    backend: str
    attempts: list[Attempt]


def _stderr(value, limit=600):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value or "").strip().replace("\r", " ").replace("\n", " ")[-limit:]


def _run(command):
    return subprocess.run(command, capture_output=True, check=False)


def stable_clip_id(source_video, start_sec, end_sec):
    """ID không phụ thuộc thứ tự xử lý hay số cửa sổ được accept trước đó."""
    source = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source_video)).strip("_")
    if not source:
        raise ValueError("source_video rỗng")
    start_ms = int(round(float(start_sec) * 1000))
    end_ms = int(round(float(end_sec) * 1000))
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError(f"biên clip không hợp lệ: {start_sec}..{end_sec}")
    return f"{source}_s{start_ms:010d}_e{end_ms:010d}"


def build_chunks(start_sec, end_sec, config):
    duration = end_sec - start_sec
    if duration < config.vad_min_duration:
        return []
    if duration <= config.vad_max_duration:
        return [(start_sec, end_sec)]
    chunks = []
    start = start_sec
    while start + config.window_size <= end_sec:
        chunks.append((start, start + config.window_size))
        start += config.window_step
    if end_sec - start >= config.tail_min_duration:
        chunks.append((start, end_sec))
    return chunks


def _raw_to_frames(raw, size):
    frame_bytes = size * size * 3
    if not raw or len(raw) % frame_bytes:
        return []
    return [
        np.frombuffer(raw[offset:offset + frame_bytes], dtype=np.uint8)
        .reshape(size, size, 3).copy()
        for offset in range(0, len(raw), frame_bytes)
    ]


def decode_frames_with_fallback(
        video_path,
        gpu_id,
        config,
        runner: Callable = _run):
    """Decode 1 FPS; CUDA lỗi/rỗng/sai byte thì retry CPU."""
    attempts = []
    backends = ["cuda", "cpu"] if config.use_hwaccel_decode else ["cpu"]
    for backend in backends:
        hw = (
            ["-hwaccel", "cuda", "-hwaccel_device", str(gpu_id)]
            if backend == "cuda" else ["-threads", "2"]
        )
        command = [
            "ffmpeg", "-y", *hw, "-i", str(video_path),
            "-vf", f"fps={config.sample_fps},scale={config.frame_size}:{config.frame_size}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-loglevel", "error", "pipe:1",
        ]
        result = runner(command)
        frames = _raw_to_frames(result.stdout, config.frame_size)
        valid = result.returncode == 0 and bool(frames)
        attempts.append(Attempt(
            backend=backend,
            returncode=result.returncode,
            stderr=_stderr(result.stderr),
            valid=valid,
        ))
        if valid:
            return DecodeResult(frames, backend, attempts)
    return DecodeResult([], "", attempts)


def probe_media(path, runner: Callable = _run):
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type:format=duration", "-of", "json", str(path),
    ]
    result = runner(command)
    if result.returncode != 0:
        return False, _stderr(result.stderr)
    try:
        payload = json.loads(
            result.stdout.decode("utf-8") if isinstance(result.stdout, bytes)
            else result.stdout
        )
        stream_types = {row.get("codec_type") for row in payload.get("streams", [])}
        duration = float(payload.get("format", {}).get("duration", 0))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"ffprobe_parse_failed:{exc}"
    if not {"video", "audio"}.issubset(stream_types) or duration <= 0:
        return False, f"invalid_streams_or_duration:{stream_types}/{duration}"
    return True, ""


def _unlink_own_partial(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def cut_clip_with_fallback(
        video_path,
        clip_path,
        start_sec,
        end_sec,
        gpu_id,
        config,
        runner: Callable = _run,
        validator: Callable = probe_media):
    """NVENC lỗi thì retry CPU/libx264; output chỉ publish sau probe."""
    clip_path = Path(clip_path)
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    partial = clip_path.with_name(clip_path.name + f".{os.getpid()}.partial.mp4")
    if clip_path.exists():
        raise FileExistsError(f"Từ chối ghi đè clip đã publish: {clip_path}")
    attempts = []
    backends = ["nvenc", "libx264"] if config.use_nvenc else ["libx264"]
    for backend in backends:
        _unlink_own_partial(partial)
        if backend == "nvenc":
            hw = ["-hwaccel", "cuda", "-hwaccel_device", str(gpu_id)]
            codec = [
                "-c:v", "h264_nvenc", "-preset", config.nvenc_preset,
                "-rc", "vbr", "-cq", str(config.nvenc_cq),
                "-gpu", str(gpu_id), "-pix_fmt", "yuv420p",
            ]
        else:
            hw = ["-threads", "2"]
            codec = [
                "-c:v", "libx264", "-crf", str(config.x264_crf),
                "-preset", config.x264_preset, "-pix_fmt", "yuv420p",
            ]
        command = [
            "ffmpeg", "-y", *hw, "-ss", f"{start_sec:.6f}",
            "-i", str(video_path), "-t", f"{end_sec - start_sec:.6f}",
            *codec, "-c:a", "aac", "-b:a", config.audio_bitrate,
            "-movflags", "+faststart", "-loglevel", "error", str(partial),
        ]
        result = runner(command)
        valid = False
        detail = _stderr(result.stderr)
        if result.returncode == 0 and partial.is_file() and partial.stat().st_size:
            valid, probe_detail = validator(partial)
            detail = probe_detail or detail
        attempts.append(Attempt(backend, result.returncode, detail, valid))
        if valid:
            os.replace(partial, clip_path)
            return CutResult(True, backend, attempts)
    _unlink_own_partial(partial)
    return CutResult(False, "", attempts)


def extract_audio(video_path, wav_path, runner: Callable = _run):
    result = runner([
        "ffmpeg", "-y", "-threads", "2", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-loglevel", "error", str(wav_path),
    ])
    valid = (
        result.returncode == 0
        and Path(wav_path).is_file()
        and Path(wav_path).stat().st_size > 44
    )
    return valid, _stderr(result.stderr)


def check_scene_cut(frames, threshold):
    previous = None
    for frame in frames:
        histogram = cv2.calcHist(
            [frame], [0, 1, 2], None, [8, 8, 8],
            [0, 256, 0, 256, 0, 256],
        )
        cv2.normalize(
            histogram, histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
        )
        if (
            previous is not None
            and cv2.compareHist(previous, histogram, cv2.HISTCMP_CORREL) < threshold
        ):
            return True
        previous = histogram
    return False


def speech_ratio_in_window(start, end, vad_segments):
    duration = end - start
    if duration <= 0:
        return 0.0
    overlap = 0.0
    for segment in vad_segments:
        low = max(start, segment["start"] / 16000.0)
        high = min(end, segment["end"] / 16000.0)
        if high > low:
            overlap += high - low
    return overlap / duration


def build_snr_context(waveform, vad_segments, sample_rate=16000):
    power = waveform ** 2
    speech_mask = np.zeros(waveform.size, dtype=bool)
    for segment in vad_segments:
        start = max(0, int(segment["start"]))
        end = min(waveform.size, int(segment["end"]))
        if end > start:
            speech_mask[start:end] = True
    noise = power[~speech_mask]
    noise_floor = (
        float(np.percentile(power, 5))
        if noise.size < int(sample_rate * 0.1)
        else float(np.mean(noise))
    )
    return power, speech_mask, max(noise_floor, 1e-10)


def compute_snr(context, start, end, sample_rate=16000):
    power, speech_mask, noise_floor = context
    low, high = int(start * sample_rate), int(end * sample_rate)
    signal = power[low:high][speech_mask[low:high]]
    if not signal.size:
        return -math.inf
    return 10.0 * math.log10(float(np.mean(signal)) / noise_floor)


def detect_faces_batched(face_model, frames, gpu_id, batch):
    import torch

    flags = []
    device = gpu_id if torch.cuda.is_available() else "cpu"
    for index in range(0, len(frames), batch):
        results = face_model.predict(
            frames[index:index + batch],
            verbose=False,
            imgsz=320,
            half=torch.cuda.is_available(),
            device=device,
            stream=False,
        )
        flags.extend(len(result.boxes) > 0 for result in results)
    return flags


_CONFIG = None
_FACE_MODEL = None
_VAD_MODEL = None
_GET_SPEECH = None
_READ_AUDIO = None
_GPU_ID = 0


def init_worker(config_dict, gpu_counter, num_gpus):
    global _CONFIG, _FACE_MODEL, _VAD_MODEL, _GET_SPEECH, _READ_AUDIO, _GPU_ID
    import torch
    from ultralytics import YOLO

    _CONFIG = CutConfig(**config_dict)
    with gpu_counter.get_lock():
        index = gpu_counter.value
        gpu_counter.value += 1
    _GPU_ID = index % max(num_gpus, 1)
    device = torch.device(
        f"cuda:{_GPU_ID}" if torch.cuda.is_available() else "cpu"
    )
    _FACE_MODEL = YOLO(_CONFIG.face_model)
    _FACE_MODEL.model = _FACE_MODEL.model.to(device)
    _FACE_MODEL.model.eval()
    vad_model, utils = torch.hub.load(
        "snakers4/silero-vad",
        "silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    _VAD_MODEL = vad_model.to(device).eval()
    _GET_SPEECH, _, _READ_AUDIO, _, _ = utils


def _attempts_text(attempts):
    return json.dumps([asdict(attempt) for attempt in attempts], ensure_ascii=False)


def _terminal(filename, video_id, reason, detail=""):
    return {
        "filename": filename,
        "video_id": video_id,
        "status": reason,
        "accepted_count": 0,
        "rejected_count": 0,
        "decode_backend": "",
        "decode_attempts": "[]",
        "cut_backends": "",
        "terminal_detail": detail,
        "run_id": _CONFIG.run_id,
        "tier": _CONFIG.tier,
    }


def process_one_video(filename):
    import torch

    config = _CONFIG
    video_path = Path(config.dataset_dir) / filename
    video_id = Path(filename).stem
    accepted, rejected = [], []
    wav_path = Path(tempfile.gettempdir()) / (
        f"cut04_{video_id}_{os.getpid()}.wav"
    )
    try:
        audio_ok, audio_error = extract_audio(video_path, wav_path)
        if not audio_ok:
            return accepted, rejected, _terminal(
                filename, video_id, "audio_extract_failed", audio_error
            )
        device = torch.device(
            f"cuda:{_GPU_ID}" if torch.cuda.is_available() else "cpu"
        )
        try:
            waveform = _READ_AUDIO(str(wav_path)).to(device)
            vad_segments = _GET_SPEECH(
                waveform, _VAD_MODEL, sampling_rate=16000
            )
        except Exception as exc:
            return accepted, rejected, _terminal(
                filename, video_id, "vad_failed", repr(exc)
            )
        if not vad_segments:
            return accepted, rejected, _terminal(
                filename, video_id, "no_speech_detected"
            )

        snr_context = build_snr_context(
            waveform.detach().cpu().numpy().astype(np.float32).flatten(),
            vad_segments,
        )
        decoded = decode_frames_with_fallback(video_path, _GPU_ID, config)
        if not decoded.frames:
            status = _terminal(
                filename,
                video_id,
                "decode_both_failed",
                _attempts_text(decoded.attempts),
            )
            status["decode_attempts"] = _attempts_text(decoded.attempts)
            return accepted, rejected, status
        try:
            face_flags = detect_faces_batched(
                _FACE_MODEL, decoded.frames, _GPU_ID, config.yolo_batch
            )
        except Exception as exc:
            status = _terminal(
                filename, video_id, "face_detect_failed", repr(exc)
            )
            status["decode_backend"] = decoded.backend
            status["decode_attempts"] = _attempts_text(decoded.attempts)
            return accepted, rejected, status

        cut_backends = Counter()
        for segment in vad_segments:
            segment_start = segment["start"] / 16000.0
            segment_end = segment["end"] / 16000.0
            if segment_end - segment_start < config.vad_min_duration:
                rejected.append({
                    "video": video_id, "filename": filename,
                    "start": round(segment_start, 3),
                    "end": round(segment_end, 3),
                    "reason": "segment_too_short", "detail": "",
                    "run_id": config.run_id, "tier": config.tier,
                })
                continue
            for start, end in build_chunks(segment_start, segment_end, config):
                start_index = int(start * config.sample_fps)
                end_index = max(start_index + 1, int(end * config.sample_fps))
                chunk_frames = decoded.frames[start_index:end_index]
                reason = ""
                detail = ""
                if not chunk_frames:
                    reason = "frame_extract_failed"
                elif check_scene_cut(
                        chunk_frames, config.histogram_correlation_threshold):
                    reason = "scene_cut_detected"
                else:
                    chunk_faces = face_flags[start_index:end_index]
                    if not chunk_faces:
                        reason = "face_index_oob"
                    else:
                        face_ratio = sum(chunk_faces) / len(chunk_faces)
                        speech_ratio = speech_ratio_in_window(
                            start, end, vad_segments
                        )
                        snr = compute_snr(snr_context, start, end)
                        if face_ratio < config.yolo_pass_ratio:
                            reason = "face_density_low"
                        elif speech_ratio < config.min_speech_ratio:
                            reason = "speech_low"
                        elif snr < config.min_snr:
                            reason = "snr_low"
                            detail = f"snr={snr:.2f}"
                if reason:
                    rejected.append({
                        "video": video_id, "filename": filename,
                        "start": round(start, 3), "end": round(end, 3),
                        "reason": reason, "detail": detail,
                        "run_id": config.run_id, "tier": config.tier,
                    })
                    continue

                clip_id = stable_clip_id(video_id, start, end)
                clip_path = (
                    Path(config.output_root)
                    / f"cut_{config.run_id}"
                    / config.tier
                    / f"{config.start_index}_{config.end_index or 'end'}"
                    / "media"
                    / f"{clip_id}.mp4"
                )
                cut = cut_clip_with_fallback(
                    video_path, clip_path, start, end, _GPU_ID, config
                )
                if not cut.ok:
                    rejected.append({
                        "video": video_id, "filename": filename,
                        "start": round(start, 3), "end": round(end, 3),
                        "reason": "ffmpeg_cut_both_failed",
                        "detail": _attempts_text(cut.attempts),
                        "run_id": config.run_id, "tier": config.tier,
                    })
                    continue
                cut_backends[cut.backend] += 1
                accepted.append({
                    "clip_id": clip_id,
                    "source_video": video_id,
                    "start_time": round(start, 3),
                    "end_time": round(end, 3),
                    "duration": round(end - start, 3),
                    "face_ratio": round(face_ratio, 3),
                    "speech_ratio": round(speech_ratio, 3),
                    "snr": round(snr, 2),
                    "file_path": str(clip_path.resolve()),
                    "tier": config.tier,
                    "decode_backend": decoded.backend,
                    "cut_backend": cut.backend,
                    "run_id": config.run_id,
                })

        status = {
            "filename": filename,
            "video_id": video_id,
            "status": "completed" if accepted else "completed_no_clips",
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "decode_backend": decoded.backend,
            "decode_attempts": _attempts_text(decoded.attempts),
            "cut_backends": json.dumps(cut_backends, sort_keys=True),
            "terminal_detail": "",
            "run_id": config.run_id,
            "tier": config.tier,
        }
        return accepted, rejected, status
    finally:
        _unlink_own_partial(wav_path)


def _write_csv_atomic(path, rows, fields):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def _write_json_atomic(path, payload):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(partial, path)


def validate_batch_coverage(input_filenames, statuses, accepted, media_paths):
    errors = []
    inputs = list(input_filenames)
    status_names = [row["filename"] for row in statuses]
    if len(inputs) != len(set(inputs)):
        errors.append("input filename trùng")
    if len(status_names) != len(set(status_names)):
        errors.append("video_status filename trùng")
    if set(inputs) != set(status_names):
        errors.append(
            f"coverage input/status lệch: missing={sorted(set(inputs)-set(status_names))[:3]}, "
            f"extra={sorted(set(status_names)-set(inputs))[:3]}"
        )
    clip_ids = [row["clip_id"] for row in accepted]
    if len(clip_ids) != len(set(clip_ids)):
        errors.append("accepted clip_id trùng")
    accepted_paths = {str(Path(row["file_path"]).resolve()) for row in accepted}
    actual_paths = {str(Path(path).resolve()) for path in media_paths}
    if accepted_paths != actual_paths:
        errors.append(
            f"accepted/media lệch: missing={len(accepted_paths-actual_paths)}, "
            f"orphan={len(actual_paths-accepted_paths)}"
        )
    status_counts = {
        row["video_id"]: int(row["accepted_count"]) for row in statuses
    }
    actual_counts = Counter(row["source_video"] for row in accepted)
    for video_id, count in status_counts.items():
        if actual_counts[video_id] != count:
            errors.append(
                f"accepted_count sai cho {video_id}: "
                f"{count}!={actual_counts[video_id]}"
            )
            break
    if errors:
        raise ValueError("; ".join(errors))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_text(command):
    result = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    return (result.stdout or result.stderr).strip()


def _environment(config):
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "ffmpeg": _command_text(["ffmpeg", "-version"]).splitlines()[:1],
        "ffprobe": _command_text(["ffprobe", "-version"]).splitlines()[:1],
        "git_sha": _command_text(["git", "rev-parse", "HEAD"]),
    }
    face_model = Path(config.face_model)
    payload["face_model"] = str(face_model.resolve())
    payload["face_model_sha256"] = (
        _sha256(face_model) if face_model.is_file() else ""
    )
    return payload


def _load_input(config):
    with open(config.input_csv, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "filename" not in rows[0]:
        raise ValueError(f"input CSV rỗng/thiếu filename: {config.input_csv}")
    filenames = [row["filename"].strip() for row in rows]
    if any(not value for value in filenames) or len(filenames) != len(set(filenames)):
        raise ValueError("input CSV có filename rỗng/trùng")
    video_ids = [Path(filename).stem for filename in filenames]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError(
            "input CSV có source stem trùng; stable clip_id sẽ xung đột"
        )
    if len(rows) != config.expected_input_count:
        raise ValueError(
            f"input count lệch config: {len(rows)} != "
            f"{config.expected_input_count}"
        )
    end = config.end_index or len(rows)
    if config.start_index >= len(rows) or end > len(rows):
        raise ValueError(
            f"batch {config.start_index}:{end} ngoài input {len(rows)}"
        )
    selected = rows[config.start_index:end]
    missing = [
        row["filename"] for row in selected
        if not (Path(config.dataset_dir) / row["filename"]).is_file()
    ]
    if missing:
        raise ValueError(
            f"thiếu {len(missing)} input media, ví dụ {missing[:3]}"
        )
    return rows, selected


def run_batch(config):
    config.validate()
    all_input, selected = _load_input(config)
    end = config.end_index or len(all_input)
    config.end_index = end
    batch_dir = (
        Path(config.output_root)
        / f"cut_{config.run_id}"
        / config.tier
        / f"{config.start_index}_{end}"
    )
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(
            f"Batch output đã tồn tại; dùng run_id mới: {batch_dir}"
        )
    media_dir = batch_dir / "media"
    media_dir.mkdir(parents=True)

    inventory = []
    for index, row in enumerate(selected, config.start_index):
        path = (Path(config.dataset_dir) / row["filename"]).resolve()
        inventory.append({
            **row,
            "input_index": index,
            "resolved_path": str(path),
            "size_bytes": path.stat().st_size,
        })
    inventory_fields = list(inventory[0])
    _write_csv_atomic(
        batch_dir / "input_inventory.csv", inventory, inventory_fields
    )
    _write_json_atomic(batch_dir / "config.json", asdict(config))
    _write_json_atomic(batch_dir / "environment.json", _environment(config))

    import torch
    from tqdm import tqdm

    mp.set_start_method("spawn", force=True)
    torch.hub.load(
        "snakers4/silero-vad",
        "silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    num_gpus = max(torch.cuda.device_count(), 1)
    counter = mp.Value("i", 0)
    accepted, rejected, statuses = [], [], []
    started = time.time()
    config_dict = asdict(config)
    config_dict.pop("extra", None)
    with mp.Pool(
        config.num_workers,
        initializer=init_worker,
        initargs=(config_dict, counter, num_gpus),
    ) as pool:
        iterator = pool.imap_unordered(
            process_one_video, [row["filename"] for row in selected]
        )
        for clips, rejects, status in tqdm(
                iterator, total=len(selected), desc="videos"):
            accepted.extend(clips)
            rejected.extend(rejects)
            statuses.append(status)

    statuses.sort(key=lambda row: row["filename"])
    accepted.sort(key=lambda row: row["clip_id"])
    rejected.sort(
        key=lambda row: (row["filename"], float(row.get("start") or -1))
    )
    media_paths = sorted(media_dir.glob("*.mp4"))
    validate_batch_coverage(
        [row["filename"] for row in selected],
        statuses,
        accepted,
        media_paths,
    )
    for path in media_paths:
        valid, detail = probe_media(path)
        if not valid:
            raise ValueError(f"media contract fail {path}: {detail}")

    _write_csv_atomic(
        batch_dir / "accepted_clips.csv", accepted, ACCEPTED_FIELDS
    )
    _write_csv_atomic(
        batch_dir / "rejected_windows.csv", rejected, REJECT_FIELDS
    )
    _write_csv_atomic(
        batch_dir / "video_status.csv", statuses, STATUS_FIELDS
    )
    summary = {
        "schema": "cut_clips_run_v1",
        "run_id": config.run_id,
        "tier": config.tier,
        "input_total_tier": len(all_input),
        "batch_start": config.start_index,
        "batch_end": end,
        "batch_inputs": len(selected),
        "terminal_statuses": len(statuses),
        "accepted_clips": len(accepted),
        "rejected_windows": len(rejected),
        "terminal_reasons": dict(Counter(row["status"] for row in statuses)),
        "decode_backends": dict(Counter(
            row["decode_backend"] for row in statuses if row["decode_backend"]
        )),
        "cut_backends": dict(Counter(
            row["cut_backend"] for row in accepted
        )),
        "elapsed_seconds": round(time.time() - started, 3),
        "coverage_passed": True,
    }
    _write_json_atomic(batch_dir / "run_summary.json", summary)

    checksum_paths = [
        path for path in batch_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    if config.hash_media:
        checksum_paths.extend(media_paths)
    checksum_rows = [
        f"{_sha256(path)}  {path.relative_to(batch_dir).as_posix()}"
        for path in sorted(checksum_paths)
    ]
    sums = batch_dir / "SHA256SUMS"
    partial = sums.with_suffix(".partial")
    partial.write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")
    os.replace(partial, sums)

    if config.create_zip:
        archive_base = str(batch_dir / f"{config.tier}_{config.start_index}_{end}_media")
        shutil.make_archive(archive_base, "zip", media_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output bất biến: {batch_dir}")
    return batch_dir


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_dir")
    parser.add_argument("--input_csv")
    parser.add_argument("--output_root")
    parser.add_argument("--run_id")
    parser.add_argument("--start_index", type=int)
    parser.add_argument("--end_index", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--no_zip", action="store_true")
    parser.add_argument("--no_hash_media", action="store_true")
    args = parser.parse_args(argv)

    config = CutConfig.from_json(args.config)
    for name in (
        "dataset_dir", "input_csv", "output_root", "run_id",
        "start_index", "end_index", "num_workers",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    if args.no_zip:
        config.create_zip = False
    if args.no_hash_media:
        config.hash_media = False
    return run_batch(config)


if __name__ == "__main__":
    main()
