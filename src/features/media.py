"""A shared 25 Hz timeline, aligned mouth crops and checkpoint acoustic inputs."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np
import torch
from scipy.io import wavfile

from src.features.avhubert import load_source, sha256


@dataclass(frozen=True)
class MediaConfig:
    fps: int = 25
    stack_order_audio: int = 4
    normalize: bool = True
    image_crop_size: int = 88
    image_mean: float = 0.421
    image_std: float = 0.165

    @classmethod
    def from_task(cls, task):
        config = cls(fps=int(task['sample_rate']),
                     stack_order_audio=int(task['stack_order_audio']),
                     normalize=bool(task['normalize']),
                     image_crop_size=int(task.get('image_crop_size', 88)),
                     image_mean=float(task.get('image_mean', 0.421)),
                     image_std=float(task.get('image_std', 0.165)))
        if (config.fps != 25 or config.stack_order_audio != 4
                or not 1 <= config.image_crop_size <= 96
                or not np.isfinite([config.image_mean, config.image_std]).all()
                or config.image_std <= 0):
            raise ValueError('Unsupported checkpoint preprocessing configuration')
        return config


def run_media_command(command):
    result = subprocess.run([str(part) for part in command], capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace')[-3000:])
    return result.stdout


def decode_window(path, start_frame, num_frames, directory):
    """Window is relative to the container start, in ticks of 1/25 second.

    FFmpeg retains the audio/video relative timestamps. Never reset each stream
    independently with STARTPTS, nor estimate/correct lip/audio synchronization.
    Lossless intermediate video avoids adding another lossy compression level.
    """
    if start_frame < 0 or not 1 <= num_frames <= 250:
        raise ValueError('Use start_frame >= 0 and 1..250 frames per window (up to 10 seconds)')
    path, directory = Path(path).resolve(), Path(directory)
    if not path.is_file():
        raise FileNotFoundError(path)
    info = json.loads(run_media_command([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', path]))
    streams = [next((s for s in info['streams'] if s['codec_type'] == kind), None)
               for kind in ('video', 'audio')]
    if any(stream is None for stream in streams):
        raise ValueError('Video must contain both video and audio streams')
    start, end = start_frame / 25, (start_frame + num_frames) / 25
    origin = float(info['format'].get('start_time', 0))
    for stream in streams:
        begin = float(stream.get('start_time', origin)) - origin
        duration = stream.get('duration')
        if begin > start + 0.04 or (duration is not None and begin + float(duration) < end - 0.04):
            raise ValueError('Requested window is outside audio/video coverage')
    video_path, audio_path = directory / 'video.mkv', directory / 'audio.wav'
    run_media_command([
        'ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', path,
        '-map', '0:v:0', '-an', '-vf',
        f'trim=start={start}:end={end},setpts=PTS-{start}/TB,fps=25:start_time=0:round=near',
        '-c:v', 'ffv1', '-pix_fmt', 'bgr0', video_path,
        '-map', '0:a:0', '-vn', '-af',
        f'atrim=start={start}:end={end},asetpts=PTS-{start}/TB,aresample=16000:first_pts=0',
        '-ac', '1', '-c:a', 'pcm_s16le', audio_path])
    rate, waveform = wavfile.read(audio_path)
    if rate != 16000 or waveform.ndim != 1 or abs(len(waveform) - num_frames * 640) > 640:
        raise ValueError('Decoded audio does not cover the requested timeline')
    return video_path, waveform


def make_inputs(waveform, mouth_bgr, config):
    """Official logfbank -> stack -> match video length -> optional layer norm.

    Visual eval transform: grayscale, /255, center crop, (x-mean)/std.
    Mouth frames remain in memory; no additional CRF-20 ROI re-encoding.
    """
    from python_speech_features import logfbank
    if (waveform.ndim != 1 or waveform.dtype != np.int16 or len(waveform) == 0
            or mouth_bgr.ndim != 4 or mouth_bgr.shape[1:] != (96, 96, 3)
            or len(mouth_bgr) == 0 or mouth_bgr.dtype != np.uint8):
        raise ValueError('Expected mono PCM16 audio and nonempty uint8 BGR mouth crops [T,96,96,3]')
    audio = logfbank(waveform, samplerate=16000).astype(np.float32)
    stack = config.stack_order_audio
    audio = np.pad(audio, ((0, (-len(audio)) % stack), (0, 0)))
    audio = audio.reshape(-1, 26 * stack)
    count = len(mouth_bgr)
    if abs(len(audio) - count) > 1:
        raise ValueError('Acoustic features and video differ by more than one frame')
    audio = np.pad(audio, ((0, max(0, count - len(audio))), (0, 0)))[:count]
    audio = torch.from_numpy(audio)
    if config.normalize:
        audio = torch.nn.functional.layer_norm(audio, audio.shape[1:])
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in mouth_bgr])
    size = config.image_crop_size
    offset = (96 - size) // 2
    gray = gray[:, offset:offset + size, offset:offset + size].astype(np.float32) / 255
    video = torch.from_numpy((gray - config.image_mean) / config.image_std)
    return audio.T.unsqueeze(0).contiguous(), video.unsqueeze(0).unsqueeze(0)


class MediaProcessor:
    def __init__(self, upstream, predictor, mean_face, config):
        for path in (predictor, mean_face):
            if not Path(path).is_file():
                raise FileNotFoundError(f'Missing face preprocessing asset: {path}')
        import dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(predictor))
        self.mean_face = np.load(mean_face, allow_pickle=False)
        if self.mean_face.shape != (68, 2) or not np.isfinite(self.mean_face).all():
            raise ValueError('Expected finite mean face with shape [68,2]')
        align_path = Path(upstream) / 'avhubert/preparation/align_mouth.py'
        self.align = load_source(align_path, '_avhubert_align')
        self.config = config
        self.provenance = dict(preprocessing='aligned_mouth_logfbank_v1', config=asdict(config),
                               landmark_detector='dlib_hog_single_face', max_missing_run=3,
                               max_missing_fraction=0.1, mouth_size=96, smoothing_frames=12,
                               roi_reencode=False, align_sha256=sha256(align_path),
                               predictor_sha256=sha256(predictor), mean_face_sha256=sha256(mean_face))

    def __call__(self, path, start_frame, num_frames):
        with tempfile.TemporaryDirectory(prefix='avhubert_') as temporary:
            video_path, waveform = decode_window(path, start_frame, num_frames, temporary)
            cap = cv2.VideoCapture(str(video_path))
            landmarks, missing, longest, streak = [], 0, 0, 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.detector(gray, 1)
                    if len(faces) > 1:
                        raise ValueError('Multiple faces: review/select the speaking person before extraction')
                    if len(faces) == 0:
                        landmarks.append(None)
                        missing += 1
                        streak += 1
                        longest = max(longest, streak)
                    else:
                        shape = self.predictor(gray, faces[0])
                        landmarks.append(np.array([(shape.part(i).x, shape.part(i).y)
                                                   for i in range(68)], dtype=np.float32))
                        streak = 0
            finally:
                cap.release()
            if len(landmarks) != num_frames:
                raise ValueError(f'Expected {num_frames} video frames; decoded {len(landmarks)}')
            if longest > 3 or missing / num_frames > 0.1:
                raise ValueError('Insufficient landmarks; refusing to substitute whole-frame ROI')
            landmarks = self.align.landmarks_interpolate(landmarks)
            crops = self.align.crop_patch(str(video_path), landmarks, self.mean_face,
                                          [33, 36, 39, 42, 45], (256, 256), 12, 48, 68, 96, 96)
            if crops is None or len(crops) != num_frames:
                raise ValueError('Mouth cropping changed the timeline')
            audio, visual = make_inputs(waveform, crops, self.config)
            return audio, visual, dict(interpolated_frames=missing, frames=num_frames,
                                      audio_samples=len(waveform))
