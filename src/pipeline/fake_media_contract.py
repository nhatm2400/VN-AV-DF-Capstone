"""Shared media contract for repaired non-temporal fake generators."""

import json
import os
import subprocess
from fractions import Fraction


AUDIO_TARGET_RATE = 16000


def _round_fraction(value):
    return int(value + Fraction(1, 2))


def probe_media(path):
    """Return exact-enough paired audio/video fields used by Stage 05."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,codec_name,avg_frame_rate,r_frame_rate,time_base,"
         "start_time,duration,duration_ts,sample_rate,channels,nb_read_frames:"
         "format=duration", "-of", "json", path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
        video = next(s for s in data["streams"] if s.get("codec_type") == "video")
        audio = next(s for s in data["streams"] if s.get("codec_type") == "audio")
        fps_text = video.get("r_frame_rate") or video.get("avg_frame_rate")
        fps = Fraction(fps_text)
        if fps <= 0:
            fps = Fraction(video["avg_frame_rate"])
        frames = int(video["nb_read_frames"])
        if video.get("duration_ts") is not None and video.get("time_base"):
            video_duration = float(
                Fraction(int(video["duration_ts"])) * Fraction(video["time_base"])
            )
        else:
            video_duration = float(video["duration"])
        if audio.get("duration_ts") is not None and audio.get("time_base"):
            audio_duration_fraction = (
                Fraction(int(audio["duration_ts"])) * Fraction(audio["time_base"])
            )
        else:
            audio_duration_fraction = Fraction(audio["duration"])
        target_samples = _round_fraction(
            audio_duration_fraction * AUDIO_TARGET_RATE
        )
        if fps <= 0 or frames <= 0 or video_duration <= 0 or target_samples <= 0:
            return None
        return {
            "video_frames": frames,
            "video_fps": fps,
            "video_duration": video_duration,
            "video_start": float(video.get("start_time", 0) or 0),
            "video_codec": str(video.get("codec_name", "")),
            "audio_target_samples": target_samples,
            "audio_duration": target_samples / AUDIO_TARGET_RATE,
            "audio_start": float(audio.get("start_time", 0) or 0),
            "audio_codec": str(audio.get("codec_name", "")),
            "audio_sample_rate": int(audio.get("sample_rate", 0) or 0),
            "audio_channels": int(audio.get("channels", 0) or 0),
        }
    except (KeyError, ValueError, IndexError, StopIteration, ZeroDivisionError,
            json.JSONDecodeError):
        return None


def media_contract_matches(actual, source, *, audio_tolerance_samples=0):
    if actual is None or source is None:
        return False
    audio_error = abs(
        actual["audio_target_samples"] - source["audio_target_samples"]
    )
    return (
        actual["video_frames"] == source["video_frames"]
        and actual["video_fps"] == source["video_fps"]
        and abs(actual["video_duration"] - source["video_duration"]) <= 1e-3
        and abs(actual["video_start"]) <= 1e-3
        and abs(actual["audio_start"]) <= 1e-3
        and audio_error <= audio_tolerance_samples
    )


def is_valid_repaired_output(path, source_media):
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        return False
    return media_contract_matches(probe_media(path), source_media)


def remove_if_exists(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def publish_validated(partial_path, out_path, source_media):
    """Validate a complete temporary media file, then atomically publish it."""
    if not is_valid_repaired_output(partial_path, source_media):
        remove_if_exists(partial_path)
        return False
    try:
        os.replace(partial_path, out_path)
    except OSError:
        remove_if_exists(partial_path)
        return False
    if not is_valid_repaired_output(out_path, source_media):
        remove_if_exists(out_path)
        return False
    return True
