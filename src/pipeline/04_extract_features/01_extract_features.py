"""
01_extract_features.py — Trích feature 3 nhánh cho AVSP-Net (stage 04)

Theo MODEL_PROPOSAL.md §4 + §9 Phase 2 — "Replace PoC Feature Extraction":
KHÔNG dùng full-frame (leak identity/background/codec); thay bằng:

  1. MOUTH ROI  : YOLOv8n-face detect mặt -> crop nửa dưới khuôn mặt (vùng miệng)
                  -> grayscale 96x96, sample ~25fps  -> uint8 [T_v, 96, 96]
                  Mỗi sampled-frame LUÔN có 1 ROI (carry-forward + backward-fill khi
                  detect fail) -> chuỗi hình không co/lệch với audio. ANON (mặt mờ,
                  YOLO fail): dùng chuỗi box của REAL ghép cặp (source_clip) áp lên
                  anon theo timestamp -> crop môi chặt, không phụ thuộc detect trên mờ.
  2. AUDIO      : wav2vec2-base-vietnamese-250h (frozen) trên wav 16k mono.
                  Với SNVSM V2, cắt phần đệm AAC đúng snvsm_target_samples trước
                  khi tạo wav/prosody/w2v -> float16 [T_a, 768].
  3. PROSODY    : F0 (parselmouth, fallback librosa.pyin), delta-F0, energy RMS,
                  voiced flag — hop 10ms -> float32 [T_p, 4]
                  (nhánh bắt fake 03_pitch_flatten — đặc thù tiếng Việt)

Mỗi clip -> 1 file .pt trong data/04_features/:
  {clip_id, label, method, speaker_id, mouth: uint8[T,96,96],
   w2v: float16[T,768] | None, wave: int16[N] | None, prosody: float32[T,4], meta}

Input: đọc CẢ real (all_clean.csv) lẫn fake (data/03_fake/labels.csv) — feature
không phụ thuộc split. Với SNVSM V2 phải chạy gate Stage 05 trước để fail-fast
coverage/audio/video/CRF rồi mới launch extraction dài.

⚠️ SNVSM: nén CRF đối xứng real+fake nên áp TRƯỚC bước này (trên .mp4). Manifest
SNVSM phải giữ snvsm_target_samples để bước này loại trailing AAC padding trước
khi model nhìn thấy audio.

Ví dụ:
  python src/pipeline/04_extract_features/01_extract_features.py --limit 5 --no_w2v
  python src/pipeline/04_extract_features/01_extract_features.py            # full, GPU
"""

import os
import sys
import csv
import json
import hashlib
import argparse
import tempfile
import subprocess
from fractions import Fraction

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

try:
    # pyrefly: ignore [missing-import]
    import torch
except Exception:
    print("Thiếu torch. Cài: pip install torch")
    sys.exit(1)

try:
    # pyrefly: ignore [missing-import]
    import cv2
    # pyrefly: ignore [missing-import]
    from ultralytics import YOLO
except Exception as e:
    print(f"Thiếu thư viện visual: {e}. Cài: pip install ultralytics opencv-python")
    sys.exit(1)

INDEX_FIELDS = ["clip_id", "feature_path", "label", "method", "speaker_id",
                "t_mouth", "t_w2v", "t_prosody", "status"]

HOP_SEC = 0.010          # 10ms cho prosody
F0_FLOOR, F0_CEIL = 75.0, 500.0   # dải F0 tiếng Việt (khớp 03_pitch_flatten)
SNVSM_MARKERS = ("snvsm_version", "snvsm_config_id", "snvsm_encoder",
                 "snvsm_preset", "snvsm_audio", "snvsm_sample_rate",
                 "snvsm_channels", "snvsm_target_samples", "snvsm_mode",
                 "snvsm_crf_set", "snvsm_seed", "snvsm_pair_key",
                 "snvsm_video_frames", "snvsm_video_fps",
                 "snvsm_video_duration_s")
EXPECTED_SNVSM_VERSION = "snvsm_v2_h264_aac16k_mono_exactdur"
EXPECTED_SNVSM_AUDIO = "aac_128k_16khz_mono"
FEATURE_SCHEMA_VERSION = "avsp_feature_v2_contract"
W2V_MODEL_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"


def expected_snvsm_config_id(row, crf_set):
    """Rebuild the Stage-03 normalization hash instead of trusting provenance."""
    config = {
        "normalization_version": EXPECTED_SNVSM_VERSION,
        "video_encoder": row["snvsm_encoder"].strip(),
        "video_preset": row["snvsm_preset"].strip(),
        "pixel_format": "yuv420p",
        "audio_codec": "aac",
        "audio_bitrate": "128k",
        "audio_sample_rate": 16000,
        "audio_channels": 1,
        "crfs": crf_set,
        "mode": row["snvsm_mode"].strip(),
        "seed": int(row["snvsm_seed"]),
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_config_id(fps, mouth_size, detect_every, conf, face_model_hash,
                      no_w2v=False, save_wave=False):
    config = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "fps": float(fps),
        "mouth_size": int(mouth_size),
        "detect_every": int(detect_every),
        "confidence": float(conf),
        "face_model_sha256": face_model_hash,
        "w2v_model": None if no_w2v else W2V_MODEL_NAME,
        "save_wave": bool(save_wave),
        "hop_sec": HOP_SEC,
        "f0_floor": F0_FLOOR,
        "f0_ceil": F0_CEIL,
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def csv_has_rows(path):
    if not os.path.isfile(path):
        return False
    with open(path, newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle), None) is not None


# ---------------------------------------------------------------- audio helpers
def extract_wav(mp4, wav, target_samples=None):
    """Decode 16 kHz mono; trim AAC padding to the manifest contract when present."""
    try:
        if os.path.exists(wav):
            os.remove(wav)
    except OSError:
        return False
    cmd = ["ffmpeg", "-y", "-i", mp4, "-vn", "-map", "0:a:0"]
    if target_samples is not None:
        if target_samples <= 0:
            return False
        audio_filter = (
            "aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono,"
            f"atrim=end_sample={target_samples},"
            "asetpts=PTS-STARTPTS"
        )
        cmd += ["-af", audio_filter]
    cmd += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-loglevel", "error", wav]
    proc = subprocess.run(cmd, capture_output=True)
    ok = proc.returncode == 0 and os.path.exists(wav) and os.path.getsize(wav) > 0
    if not ok:
        try:
            if os.path.exists(wav):
                os.remove(wav)
        except OSError:
            pass
    return ok


def read_wav_int16(path):
    import wave
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).copy()


def snvsm_target_samples(row):
    """Return the Stage-04 PCM length contract; reject incomplete SNVSM rows."""
    raw = str(row.get("snvsm_target_samples", "") or "").strip()
    is_snvsm = bool(raw) or any(str(row.get(field, "") or "").strip()
                                for field in SNVSM_MARKERS)
    if not raw:
        if is_snvsm:
            raise ValueError("snvsm_target_samples_missing")
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("snvsm_target_samples_invalid") from exc
    if value <= 0:
        raise ValueError("snvsm_target_samples_invalid")
    if is_snvsm:
        try:
            crf_set = [int(item) for item in row["snvsm_crf_set"].split(",")]
            valid = (
                row["snvsm_version"].strip() == EXPECTED_SNVSM_VERSION
                and row["snvsm_audio"].strip() == EXPECTED_SNVSM_AUDIO
                and row["snvsm_encoder"].strip() in ("libx264", "h264_nvenc")
                and str(row["snvsm_preset"]).strip()
                and row["snvsm_config_id"].strip()
                == expected_snvsm_config_id(row, crf_set)
                and int(row["snvsm_sample_rate"]) == 16000
                and int(row["snvsm_channels"]) == 1
                and row["snvsm_mode"].strip() in ("random", "all")
                and len(crf_set) > 0
                and len(set(crf_set)) == len(crf_set)
                and int(row["crf"]) in crf_set
                and int(row["snvsm_seed"]) >= 0
                and str(row["snvsm_pair_key"]).strip()
                and int(row["snvsm_video_frames"]) > 0
                and Fraction(row["snvsm_video_fps"]) > 0
                and float(row["snvsm_video_duration_s"]) > 0
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            valid = False
        if not valid:
            raise ValueError("snvsm_contract_invalid")
    return value


def is_valid_existing_feature(path, target_samples, expected, require_w2v=True):
    """Resume only from a complete feature object matching the PCM contract."""
    if not os.path.exists(path):
        return False
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict):
            return False
        mouth = obj.get("mouth")
        prosody = obj.get("prosody")
        w2v = obj.get("w2v")
        wave = obj.get("wave")
        mouth_size = int(expected["mouth_size"])
        identity_ok = (
            str(obj.get("clip_id", "")) == str(expected["clip_id"])
            and int(obj.get("label", -1)) == int(expected["label"])
            and str(obj.get("method", "")) == str(expected["method"])
        )
        if (not identity_ok
                or not torch.is_tensor(mouth) or mouth.dtype != torch.uint8
                or mouth.ndim != 3 or tuple(mouth.shape[1:]) != (mouth_size, mouth_size)
                or mouth.numel() == 0
                or not torch.is_tensor(prosody) or prosody.ndim != 2
                or prosody.dtype != torch.float32
                or prosody.shape[-1] != 4 or prosody.numel() == 0):
            return False
        if (require_w2v
                and (not torch.is_tensor(w2v) or w2v.ndim != 2
                     or w2v.dtype != torch.float16 or w2v.shape[-1] != 768
                     or w2v.numel() == 0)):
            return False
        if (expected.get("require_wave")
                and (not torch.is_tensor(wave) or wave.ndim != 1
                     or wave.dtype != torch.int16 or wave.numel() == 0)):
            return False
        meta = obj.get("meta", {})
        try:
            source_ok = (os.path.normcase(os.path.abspath(meta.get("src", "")))
                         == os.path.normcase(os.path.abspath(expected["src"])))
            config_ok = (
                meta.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
                and meta.get("feature_config_id") == expected["feature_config_id"]
                and str(meta.get("snvsm_config_id", "") or "")
                == str(expected.get("snvsm_config_id", "") or "")
            )
            audio_samples = int(meta.get("audio_samples", -1))
            audio_ok = (audio_samples > 0 if target_samples is None else
                        audio_samples == target_samples
                        and int(meta.get("audio_target_samples", -1)) == target_samples)
            wave_ok = (not expected.get("require_wave")
                       or wave.numel() == audio_samples)
            return source_ok and config_ok and audio_ok and wave_ok
        except (TypeError, ValueError):
            return False
    except (OSError, RuntimeError, TypeError, ValueError, EOFError):
        return False


def prosody_features(wav_path, sig_i16):
    """[T,4]: f0_z (0 tại unvoiced), delta_f0, energy_z (log RMS), voiced."""
    n_hop = int(16000 * HOP_SEC)
    f0 = None
    try:
        import parselmouth
        snd = parselmouth.Sound(wav_path)
        pitch = snd.to_pitch(time_step=HOP_SEC, pitch_floor=F0_FLOOR, pitch_ceiling=F0_CEIL)
        f0 = pitch.selected_array["frequency"]          # 0 = unvoiced
    except Exception:
        try:
            import librosa
            y = sig_i16.astype(np.float32) / 32768.0
            f0_pyin, _, _ = librosa.pyin(y, fmin=F0_FLOOR, fmax=F0_CEIL,
                                         sr=16000, hop_length=n_hop)
            f0 = np.nan_to_num(f0_pyin, nan=0.0)
        except Exception as e:
            print(f"  ! prosody bỏ qua (thiếu parselmouth lẫn librosa): {e}")
            return None

    T = len(f0)
    voiced = (f0 > 0).astype(np.float32)
    # z-norm log-F0 trong vùng voiced (chuẩn hóa theo NGƯỜI NÓI của chính clip)
    logf0 = np.zeros(T, dtype=np.float32)
    if voiced.sum() > 3:
        lv = np.log(f0[f0 > 0])
        mu, sd = lv.mean(), max(lv.std(), 1e-4)
        logf0[f0 > 0] = (lv - mu) / sd
    d_f0 = np.zeros(T, dtype=np.float32)
    d_f0[1:] = logf0[1:] - logf0[:-1]

    # energy: log RMS theo hop, z-norm
    y = sig_i16.astype(np.float32) / 32768.0
    n_frames = max(1, len(y) // n_hop)
    rms = np.array([np.sqrt((y[i * n_hop:(i + 1) * n_hop] ** 2).mean() + 1e-10)
                    for i in range(n_frames)], dtype=np.float32)
    loge = np.log(rms)
    loge = (loge - loge.mean()) / max(loge.std(), 1e-4)
    # khớp độ dài với f0
    L = min(T, len(loge))
    out = np.stack([logf0[:L], d_f0[:L], loge[:L], voiced[:L]], axis=1)
    return out.astype(np.float32)


# ---------------------------------------------------------------- visual helper
def _crop_mouth(frame, box, size):
    """Crop nửa dưới bbox (vùng miệng), nới ngang 10%, -> grayscale size×size. None nếu suy biến."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    mx1 = int(max(0, x1 - 0.10 * bw))
    mx2 = int(min(frame.shape[1], x2 + 0.10 * bw))
    my1 = int(y1 + 0.55 * bh)
    my2 = int(min(frame.shape[0], y2 + 0.10 * bh))
    if mx2 <= mx1 or my2 <= my1:
        return None
    g = cv2.cvtColor(frame[my1:my2, mx1:mx2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (size, size))


def detect_and_crop(model, mp4, target_fps, size, detect_every, conf):
    """
    MỘT lần decode: detect (carry-forward + backward-fill) + crop mouth. Trả
    (boxes, mouth[T,size,size]); (None, None) nếu cả clip không có mặt.
    Mỗi output-frame LUÔN có 1 box -> chuỗi ROI không co/lệch với audio. `boxes`
    trả ra để anon ghép cặp tái dùng (crop trên video anon, không detect mặt mờ).
    """
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        return None, None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    boxes, rois, pending = [], [], []      # pending: (idx, frame) chưa có box -> backfill sau
    box, fi, oi, det_i = None, 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        while fi >= round(oi * src_fps / target_fps):      # emit output-frame (target_fps thật)
            if det_i % max(1, detect_every) == 0:
                res = model.predict(frame, verbose=False, conf=conf)[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    box = xyxy[int(areas.argmax())][:4]    # fail -> giữ box cũ (carry-forward)
            det_i += 1
            boxes.append(box)
            if box is None:
                pending.append((oi, frame.copy())); rois.append(None)   # backfill khi có box đầu
            else:
                rois.append(_crop_mouth(frame, box, size))
            oi += 1
        fi += 1
    cap.release()
    first = next((b for b in boxes if b is not None), None)
    if first is None:
        return None, None                                  # cả clip không bắt được mặt
    for k, fr in pending:                                  # backward-fill các None ở đầu
        boxes[k] = first
        rois[k] = _crop_mouth(fr, first, size)
    rois = [r for r in rois if r is not None]
    if not rois:
        return boxes, None
    return boxes, np.stack(rois).astype(np.uint8)


def crop_from_boxes(mp4, boxes, target_fps, size):
    """
    Crop mouth trên VIDEO này theo chuỗi box cho sẵn (anon dùng box của real ghép cặp).
    Cùng sampler timestamp target_fps -> output-frame khớp box theo thời gian; frame dư
    cuối (anon lệch 1–4 frame do encode) dùng box CUỐI. Trả uint8 [T,size,size] | None.
    """
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    rois, fi, oi = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        while fi >= round(oi * src_fps / target_fps):
            c = _crop_mouth(frame, boxes[min(oi, len(boxes) - 1)], size)
            if c is not None:
                rois.append(c)
            oi += 1
        fi += 1
    cap.release()
    if not rois:
        return None
    return np.stack(rois).astype(np.uint8)


# ---------------------------------------------------------------- w2v helper
class W2V:
    def __init__(self, device):
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        name = W2V_MODEL_NAME
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(name)
        self.model = Wav2Vec2Model.from_pretrained(name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, sig_i16):
        y = sig_i16.astype(np.float32) / 32768.0
        inp = self.fe(y, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        out = self.model(inp).last_hidden_state[0]          # [T, 768]
        return out.half().cpu()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_csv", default="data/02_curate/all_clean.csv")
    ap.add_argument("--fake_labels", default="data/03_fake/labels.csv")
    ap.add_argument("--out_dir", default="data/04_features")
    ap.add_argument("--face_model", default="yolov8n-face.pt")
    ap.add_argument("--fps", type=float, default=25.0, help="fps sample mouth ROI")
    ap.add_argument("--mouth_size", type=int, default=96)
    ap.add_argument("--detect_every", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no_w2v", action="store_true", help="bỏ wav2vec2 (nhanh, nhưng nhánh audio cần nó)")
    ap.add_argument("--save_wave", action="store_true", help="lưu cả waveform int16 (cho fine-tune sau)")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false",
                    help="buộc trích lại kể cả khi .pt đã tồn tại")
    ap.add_argument("--allow_real_only", action="store_true",
                    help="cho phép extract chỉ real (mặc định từ chối fake rỗng/thiếu)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not csv_has_rows(args.real_csv):
        print(f"LỖI: real manifest trống/thiếu: {args.real_csv}")
        sys.exit(1)
    if not args.allow_real_only and not csv_has_rows(args.fake_labels):
        print(f"LỖI: fake manifest trống/thiếu: {args.fake_labels}. "
              "Từ chối extract real-only; chỉ dùng --allow_real_only khi thật sự chủ ý.")
        sys.exit(1)

    if not os.path.isfile(args.face_model):
        print(f"Không tìm thấy {args.face_model} (chạy từ repo root)"); sys.exit(1)
    extraction_config_id = feature_config_id(
        args.fps, args.mouth_size, args.detect_every, args.conf,
        file_sha256(args.face_model), args.no_w2v, args.save_wave,
    )
    yolo = YOLO(args.face_model)

    w2v = None
    if not args.no_w2v:
        try:
            w2v = W2V(device)
            print("wav2vec2-base-vietnamese-250h: OK")
        except Exception as e:
            # FAIL-FAST: không im lặng chạy tiếp ra feature THIẾU nhánh audio (phí cả run dài).
            # Muốn cố ý bỏ audio thì truyền --no_w2v.
            print(f"LỖI: không nạp được wav2vec2 ({e}). "
                  f"Dừng để tránh trích feature thiếu nhánh audio. "
                  f"Nếu CỐ Ý bỏ nhánh audio, chạy lại với --no_w2v.")
            sys.exit(1)

    # gộp real (label=0) + fake (label=1)
    rows = []
    real_path_by_orig = {}          # orig_clip_id (=source_clip của fake) -> .mp4 real trên đĩa
    with open(args.real_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["_label"], r["_method"] = 0, "real"
            key = r.get("orig_clip_id") or r.get("clip_id", "")   # SNVSM: orig_clip_id; thô: clip_id
            if key:
                real_path_by_orig[key] = r.get("file_path", "")
            rows.append(r)
    fake_count = 0
    if os.path.isfile(args.fake_labels):
        with open(args.fake_labels, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["_label"], r["_method"] = 1, r.get("method", "fake")
                rows.append(r)
                fake_count += 1
    if fake_count == 0 and not args.allow_real_only:
        print(f"LỖI: fake manifest trống/thiếu: {args.fake_labels}. "
              "Từ chối extract real-only; chỉ dùng --allow_real_only khi thật sự chủ ý.")
        sys.exit(1)
    if fake_count == 0:
        print("CẢNH BÁO: extract real-only theo yêu cầu --allow_real_only.")
    if args.limit:
        rows = rows[:args.limit]
    print(f"Tổng {len(rows)} clip cần trích")

    index_path = os.path.join(args.out_dir, "features_index.csv")
    new_idx = not os.path.exists(index_path) or os.path.getsize(index_path) == 0
    idx_f = open(index_path, "a", newline="", encoding="utf-8")
    idx_w = csv.DictWriter(idx_f, fieldnames=INDEX_FIELDS)
    if new_idx:
        idx_w.writeheader()

    tmpdir = tempfile.mkdtemp(prefix="feat_")
    wav_tmp = os.path.join(tmpdir, "a.wav")
    box_cache = {}                  # orig_clip_id -> chuỗi box của real (anon ghép cặp tái dùng)
    done = skipped = failed = 0
    for i, r in enumerate(rows):
        cid = r.get("clip_id", f"clip{i:06d}")
        mp4 = r.get("file_path", "")
        out_pt = os.path.join(args.out_dir, cid + ".pt")
        if not mp4 or not os.path.isfile(mp4):
            idx_w.writerow({"clip_id": cid, "feature_path": "", "label": r["_label"],
                            "method": r["_method"], "speaker_id": r.get("speaker_id", ""),
                            "t_mouth": 0, "t_w2v": 0, "t_prosody": 0, "status": "missing_mp4"})
            failed += 1
            continue

        try:
            target_samples = snvsm_target_samples(r)
        except Exception as e:
            idx_w.writerow({"clip_id": cid, "feature_path": "", "label": r["_label"],
                            "method": r["_method"], "speaker_id": r.get("speaker_id", ""),
                            "t_mouth": 0, "t_w2v": 0, "t_prosody": 0,
                            "status": str(e)[:80]})
            failed += 1
            continue
        expected_feature = {
            "clip_id": cid,
            "label": r["_label"],
            "method": r["_method"],
            "src": mp4,
            "mouth_size": args.mouth_size,
            "feature_config_id": extraction_config_id,
            "snvsm_config_id": r.get("snvsm_config_id", ""),
            "require_wave": args.save_wave,
        }
        if (args.skip_existing
                and is_valid_existing_feature(
                    out_pt, target_samples, expected_feature,
                    require_w2v=not args.no_w2v
                )):
            skipped += 1
            continue
        partial_pt = out_pt + ".part"
        stale_remove_failed = False
        for stale_path in (out_pt, partial_pt):
            try:
                if os.path.exists(stale_path):
                    os.remove(stale_path)
            except OSError as exc:
                idx_w.writerow({"clip_id": cid, "feature_path": "", "label": r["_label"],
                                "method": r["_method"], "speaker_id": r.get("speaker_id", ""),
                                "t_mouth": 0, "t_w2v": 0, "t_prosody": 0,
                                "status": f"stale_feature_remove:{exc}"[:80]})
                failed += 1
                stale_remove_failed = True
                break
        if stale_remove_failed:
            continue

        try:
            if r["_method"] == "anonymization":
                # anon: mặt đã mờ -> KHÔNG detect trên anon; dùng chuỗi box của real ghép cặp
                src = r.get("source_clip", "")
                boxes = box_cache.get(src)
                if boxes is None:                            # real bị skip/drop -> detect lại trên real
                    real_mp4 = real_path_by_orig.get(src, "")
                    if real_mp4 and os.path.isfile(real_mp4):
                        boxes, _ = detect_and_crop(yolo, real_mp4, args.fps, args.mouth_size,
                                                   args.detect_every, args.conf)
                if boxes is None:                            # không tra được real -> đành thử trên anon
                    boxes, _ = detect_and_crop(yolo, mp4, args.fps, args.mouth_size,
                                               args.detect_every, args.conf)
                mouth = crop_from_boxes(mp4, boxes, args.fps, args.mouth_size) if boxes is not None else None
            else:
                # real + fake không-anon: 1 lần decode vừa detect vừa crop (video của chính clip)
                boxes, mouth = detect_and_crop(yolo, mp4, args.fps, args.mouth_size,
                                               args.detect_every, args.conf)
                if r["_method"] == "real" and boxes is not None:
                    box_cache[r.get("orig_clip_id") or cid] = boxes   # cache cho anon ghép cặp
            if not extract_wav(mp4, wav_tmp, target_samples):
                raise RuntimeError("ffmpeg_wav_failed")
            sig = read_wav_int16(wav_tmp)
            if target_samples is not None and len(sig) != target_samples:
                raise RuntimeError(
                    f"audio_sample_contract:{len(sig)}!={target_samples}"
                )
            pros = prosody_features(wav_tmp, sig)
            feat_w2v = w2v(sig) if w2v is not None else None
            if mouth is None or len(mouth) == 0:
                raise RuntimeError("no_face_detected")
            if pros is None or len(pros) == 0:
                raise RuntimeError("prosody_failed")

            obj = {
                "clip_id": cid,
                "label": r["_label"],
                "method": r["_method"],
                "speaker_id": r.get("speaker_id", ""),
                "mouth": torch.from_numpy(mouth),                      # uint8 [T,96,96]
                "w2v": feat_w2v,                                       # float16 [T,768] | None
                "wave": torch.from_numpy(sig) if args.save_wave else None,
                "prosody": torch.from_numpy(pros) if pros is not None else None,
                "meta": {"fps": args.fps, "hop_sec": HOP_SEC, "src": mp4,
                         "feature_schema_version": FEATURE_SCHEMA_VERSION,
                         "feature_config_id": extraction_config_id,
                         "snvsm_config_id": r.get("snvsm_config_id", ""),
                         "audio_target_samples": target_samples,
                         "audio_samples": len(sig)},
            }
            torch.save(obj, partial_pt)
            os.replace(partial_pt, out_pt)
            idx_w.writerow({"clip_id": cid, "feature_path": out_pt, "label": r["_label"],
                            "method": r["_method"], "speaker_id": r.get("speaker_id", ""),
                            "t_mouth": len(mouth), "t_w2v": 0 if feat_w2v is None else len(feat_w2v),
                            "t_prosody": 0 if pros is None else len(pros), "status": "ok"})
            done += 1
        except Exception as e:
            try:
                if os.path.exists(partial_pt):
                    os.remove(partial_pt)
            except OSError:
                pass
            idx_w.writerow({"clip_id": cid, "feature_path": "", "label": r["_label"],
                            "method": r["_method"], "speaker_id": r.get("speaker_id", ""),
                            "t_mouth": 0, "t_w2v": 0, "t_prosody": 0, "status": str(e)[:80]})
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}  ok={done} skip={skipped} fail={failed}")
            idx_f.flush()

    idx_f.close()
    try:
        os.remove(wav_tmp); os.rmdir(tmpdir)
    except OSError:
        pass
    print(f"\nXong. ok={done} | skip(đã có)={skipped} | fail={failed}")
    print(f"  Feature -> {args.out_dir}/  | index -> {index_path}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
