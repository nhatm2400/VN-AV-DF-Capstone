"""
01_extract_features.py — Trích feature 3 nhánh cho AVSP-Net (stage 04)

Theo MODEL_PROPOSAL.md §4 + §9 Phase 2 — "Replace PoC Feature Extraction":
KHÔNG dùng full-frame (leak identity/background/codec); thay bằng:

  1. MOUTH ROI  : YOLOv8n-face detect mặt -> crop nửa dưới khuôn mặt (vùng miệng)
                  -> grayscale 96x96, sample ~25fps  -> uint8 [T_v, 96, 96]
  2. AUDIO      : wav2vec2-base-vietnamese-250h (frozen) trên wav 16k mono
                  -> float16 [T_a, 768]   (tùy chọn --no_w2v để bỏ)
  3. PROSODY    : F0 (parselmouth, fallback librosa.pyin), delta-F0, energy RMS,
                  voiced flag — hop 10ms -> float32 [T_p, 4]
                  (nhánh bắt fake 03_pitch_flatten — đặc thù tiếng Việt)

Mỗi clip -> 1 file .pt trong data/04_features/:
  {clip_id, label, method, speaker_id, mouth: uint8[T,96,96],
   w2v: float16[T,768] | None, wave: int16[N] | None, prosody: float32[T,4], meta}

Input: đọc CẢ real (all_clean.csv) lẫn fake (data/03_fake/labels.csv) — feature
không phụ thuộc split nên chạy trước/sau 05_build_labels đều được.

⚠️ SNVSM: nén CRF đối xứng real+fake nên áp TRƯỚC bước này (trên .mp4), hoặc
augment ở dataloader — nếu không, codec khác nhau giữa real/fake sẽ leak vào
mouth ROI. (TODO: script nén riêng.)

Ví dụ:
  python src/pipeline/04_extract_features/01_extract_features.py --limit 5 --no_w2v
  python src/pipeline/04_extract_features/01_extract_features.py            # full, GPU
"""

import os
import sys
import csv
import json
import argparse
import tempfile
import subprocess

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

try:
    import torch
except Exception:
    print("Thiếu torch. Cài: pip install torch")
    sys.exit(1)

try:
    import cv2
    from ultralytics import YOLO
except Exception as e:
    print(f"Thiếu thư viện visual: {e}. Cài: pip install ultralytics opencv-python")
    sys.exit(1)

INDEX_FIELDS = ["clip_id", "feature_path", "label", "method", "speaker_id",
                "t_mouth", "t_w2v", "t_prosody", "status"]

HOP_SEC = 0.010          # 10ms cho prosody
F0_FLOOR, F0_CEIL = 75.0, 500.0   # dải F0 tiếng Việt (khớp 03_pitch_flatten)


# ---------------------------------------------------------------- audio helpers
def extract_wav(mp4, wav):
    subprocess.run(["ffmpeg", "-y", "-i", mp4, "-vn", "-ac", "1", "-ar", "16000",
                    "-loglevel", "error", wav], capture_output=True)
    return os.path.exists(wav) and os.path.getsize(wav) > 0


def read_wav_int16(path):
    import wave
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).copy()


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
def mouth_rois(model, mp4, target_fps, size, detect_every, conf):
    """uint8 [T, size, size] — crop vùng miệng (nửa dưới bbox mặt to nhất)."""
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / target_fps))
    rois, box, fi, det_i = [], None, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % step == 0:
            if det_i % max(1, detect_every) == 0:
                res = model.predict(frame, verbose=False, conf=conf)[0]
                box = None
                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    box = xyxy[int(areas.argmax())][:4]
            det_i += 1
            if box is not None:
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                # vùng miệng: nửa dưới mặt, nới ngang 10%
                mx1 = int(max(0, x1 - 0.10 * bw))
                mx2 = int(min(frame.shape[1], x2 + 0.10 * bw))
                my1 = int(y1 + 0.55 * bh)
                my2 = int(min(frame.shape[0], y2 + 0.10 * bh))
                if mx2 > mx1 and my2 > my1:
                    crop = frame[my1:my2, mx1:mx2]
                    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    rois.append(cv2.resize(g, (size, size)))
        fi += 1
    cap.release()
    if not rois:
        return None
    return np.stack(rois).astype(np.uint8)


# ---------------------------------------------------------------- w2v helper
class W2V:
    def __init__(self, device):
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        name = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
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
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not os.path.isfile(args.face_model):
        print(f"Không tìm thấy {args.face_model} (chạy từ repo root)"); sys.exit(1)
    yolo = YOLO(args.face_model)

    w2v = None
    if not args.no_w2v:
        try:
            w2v = W2V(device)
            print("wav2vec2-base-vietnamese-250h: OK")
        except Exception as e:
            print(f"CẢNH BÁO: không nạp được wav2vec2 ({e}) — tiếp tục không có nhánh audio.")

    # gộp real (label=0) + fake (label=1)
    rows = []
    with open(args.real_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["_label"], r["_method"] = 0, "real"
            rows.append(r)
    if os.path.isfile(args.fake_labels):
        with open(args.fake_labels, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["_label"], r["_method"] = 1, r.get("method", "fake")
                rows.append(r)
    else:
        print(f"CẢNH BÁO: {args.fake_labels} chưa có — chỉ trích real.")
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
    done = skipped = failed = 0
    for i, r in enumerate(rows):
        cid = r.get("clip_id", f"clip{i:06d}")
        mp4 = r.get("file_path", "")
        out_pt = os.path.join(args.out_dir, cid + ".pt")
        if args.skip_existing and os.path.exists(out_pt):
            skipped += 1
            continue
        if not mp4 or not os.path.isfile(mp4):
            idx_w.writerow({"clip_id": cid, "feature_path": "", "label": r["_label"],
                            "method": r["_method"], "speaker_id": r.get("speaker_id", ""),
                            "t_mouth": 0, "t_w2v": 0, "t_prosody": 0, "status": "missing_mp4"})
            failed += 1
            continue

        try:
            mouth = mouth_rois(yolo, mp4, args.fps, args.mouth_size,
                               args.detect_every, args.conf)
            if not extract_wav(mp4, wav_tmp):
                raise RuntimeError("ffmpeg_wav_failed")
            sig = read_wav_int16(wav_tmp)
            pros = prosody_features(wav_tmp, sig)
            feat_w2v = w2v(sig) if w2v is not None else None
            if mouth is None:
                raise RuntimeError("no_face_detected")

            obj = {
                "clip_id": cid,
                "label": r["_label"],
                "method": r["_method"],
                "speaker_id": r.get("speaker_id", ""),
                "mouth": torch.from_numpy(mouth),                      # uint8 [T,96,96]
                "w2v": feat_w2v,                                       # float16 [T,768] | None
                "wave": torch.from_numpy(sig) if args.save_wave else None,
                "prosody": torch.from_numpy(pros) if pros is not None else None,
                "meta": {"fps": args.fps, "hop_sec": HOP_SEC, "src": mp4},
            }
            torch.save(obj, out_pt)
            idx_w.writerow({"clip_id": cid, "feature_path": out_pt, "label": r["_label"],
                            "method": r["_method"], "speaker_id": r.get("speaker_id", ""),
                            "t_mouth": len(mouth), "t_w2v": 0 if feat_w2v is None else len(feat_w2v),
                            "t_prosody": 0 if pros is None else len(pros), "status": "ok"})
            done += 1
        except Exception as e:
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


if __name__ == "__main__":
    main()
