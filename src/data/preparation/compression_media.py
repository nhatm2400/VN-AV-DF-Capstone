"""Media normalization primitives retained from the tested SNVSM implementation."""
import os
import json
import subprocess
from fractions import Fraction

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

def _round_fraction(value):
    return int(value + Fraction(1, 2))

def audio_target_samples(path):
    """Số sample 16 kHz theo timeline audio khai báo, bỏ decoder padding AAC."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=duration,duration_ts,time_base",
         "-of", "json", path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        audio = json.loads(proc.stdout)["streams"][0]
        if audio.get("duration_ts") is not None and audio.get("time_base"):
            samples = _round_fraction(
                Fraction(int(audio["duration_ts"]))
                * Fraction(audio["time_base"])
                * AUDIO_SAMPLE_RATE
            )
        else:
            samples = _round_fraction(
                Fraction(audio["duration"]) * AUDIO_SAMPLE_RATE
            )
        return samples if samples > 0 else None
    except (ValueError, KeyError, IndexError, ZeroDivisionError,
            json.JSONDecodeError):
        return None

def video_contract(path):
    """Frame/timeline contract that SNVSM must preserve from its input."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries",
         "stream=nb_read_frames,r_frame_rate,avg_frame_rate,duration,duration_ts,time_base,start_time",
         "-of", "json", path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        video = json.loads(proc.stdout)["streams"][0]
        frames = int(video["nb_read_frames"])
        fps = Fraction(video.get("r_frame_rate") or video["avg_frame_rate"])
        if video.get("duration_ts") is not None and video.get("time_base"):
            duration = float(Fraction(int(video["duration_ts"]))
                             * Fraction(video["time_base"]))
        else:
            duration = float(video["duration"])
        if frames <= 0 or fps <= 0 or duration <= 0:
            return None
        return {
            "frames": frames,
            "fps": str(fps),
            "duration": duration,
            "start": float(video.get("start_time", 0) or 0),
        }
    except (ValueError, KeyError, IndexError, ZeroDivisionError,
            json.JSONDecodeError):
        return None

def video_contract_matches(actual, expected):
    if actual is None or expected is None:
        return False
    return (actual["frames"] == expected["frames"]
            and Fraction(actual["fps"]) == Fraction(expected["fps"])
            and abs(actual["duration"] - expected["duration"]) <= 1e-3
            and abs(actual["start"]) <= 1e-3)

def decoded_audio_samples(path):
    """PCM samples exposed by the same 16 kHz mono decode used by Stage 04."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-map", "0:a:0",
         "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0 or len(proc.stdout) % 2:
        return None
    return len(proc.stdout) // 2

def is_valid_output(path, expected_samples=None, expected_video=None):
    """Validate media plus enough decoded PCM for Stage 04's manifest trim."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,start_time,sample_rate,channels,duration_ts,time_base:"
         "format=duration",
         "-of", "json", path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return False
    try:
        data = json.loads(proc.stdout)
        streams = data.get("streams", [])
        video = next(s for s in streams if s.get("codec_type") == "video")
        audio = next(s for s in streams if s.get("codec_type") == "audio")
        duration = float(data.get("format", {}).get("duration", 0))
        output_samples = _round_fraction(
            Fraction(int(audio["duration_ts"]))
            * Fraction(audio["time_base"])
            * AUDIO_SAMPLE_RATE
        )
        metadata_ok = (video.get("codec_name") == "h264"
                       and audio.get("codec_name") == "aac"
                       and int(audio.get("sample_rate", 0)) == AUDIO_SAMPLE_RATE
                       and int(audio.get("channels", 0)) == AUDIO_CHANNELS
                       and abs(float(audio.get("start_time", 0) or 0)) <= 1e-3
                       and duration > 0
                       and (expected_samples is None
                            or abs(output_samples - expected_samples) <= 1024))
        if not metadata_ok or expected_samples is None:
            return metadata_ok
        if expected_video is None:
            return False
        # AAC frames may expose trailing decoder padding. Stage 04 trims that tail
        # to expected_samples; reject files that would instead require silence pad.
        decoded_samples = decoded_audio_samples(path)
        return (decoded_samples is not None
                and decoded_samples >= expected_samples
                and video_contract_matches(video_contract(path), expected_video))
    except (ValueError, KeyError, StopIteration, ZeroDivisionError,
            json.JSONDecodeError):
        return False

def compress(in_path, out_path, crf, preset, encoder, target_samples=None,
             expected_video=None):
    """
    Re-encode video @CRF + audio AAC 128k/16 kHz/mono. CÙNG pipeline real+fake.
      libx264   : -crf N (CPU, chuẩn)
      h264_nvenc: -cq N  (GPU NVENC — nhanh, không chiếm CPU; CQ≈CRF)
    """
    if target_samples is None:
        target_samples = audio_target_samples(in_path)
    expected_video = expected_video or video_contract(in_path)
    if target_samples is None or target_samples <= 0 or expected_video is None:
        return False
    if encoder == "h264_nvenc":
        venc = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    else:
        venc = ["-c:v", "libx264", "-crf", str(crf), "-preset", preset]

    partial_path = out_path + ".part.mp4"
    try:
        if os.path.exists(partial_path):
            os.remove(partial_path)
    except OSError:
        return False

    audio_filter = (
        f"aresample={AUDIO_SAMPLE_RATE},"
        "aformat=sample_fmts=fltp:channel_layouts=mono,"
        f"apad=whole_len={target_samples},atrim=end_sample={target_samples},"
        "asetpts=PTS-STARTPTS"
    )
    cmd = ["ffmpeg", "-y", "-i", in_path,
           "-map", "0:v:0", "-map", "0:a:0?", *venc, "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-ar", str(AUDIO_SAMPLE_RATE),
           "-ac", str(AUDIO_CHANNELS), "-af", audio_filter,
           "-movflags", "+faststart", "-loglevel", "error", partial_path]
    proc = subprocess.run(cmd, capture_output=True)
    ok = (proc.returncode == 0
          and is_valid_output(partial_path, target_samples, expected_video))
    try:
        if ok:
            os.replace(partial_path, out_path)
        elif os.path.exists(partial_path):
            os.remove(partial_path)
    except OSError:
        ok = False
        try:
            if os.path.exists(partial_path):
                os.remove(partial_path)
        except OSError:
            pass
    return ok
