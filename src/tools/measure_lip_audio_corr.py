"""
measure_lip_audio_corr.py — Tương quan CỬ ĐỘNG MIỆNG vs NĂNG LƯỢNG ÂM.

Dùng clip đã gán nhãn tay làm ground truth để trả lời: có tự động hoá được việc phát
hiện lồng tiếng / voice-over không, hay bắt buộc phải review tay?

Đo trên ROI preview (đã dựng sẵn, chính là vùng miệng stage 04 cắt ra):
  mouth_motion[t] = mean|ROI[t] - ROI[t-1]|
  audio_rms[t]    = RMS của audio trong cửa sổ tương ứng
  corr            = Pearson giữa hai chuỗi
Giả thuyết: người đang nói thật -> miệng động khi có tiếng, im khi ngắt -> corr cao;
lời bình đè lên B-roll -> hai chuỗi không liên quan -> corr ~ 0.

KẾT QUẢ ĐÃ ĐO (batch 60 clip rubric v2, 2026-07-27): **AUC 0,544** — giả thuyết KHÔNG
đúng trên phép đo này. Pearson giữa pixel-motion thô và audio RMS không tách được
keep/reject. Lưu lại làm bằng chứng cho quyết định "phải review tay".

Phạm vi kết luận: chỉ nói về ĐÚNG phép đo này trên ĐÚNG batch này. Không suy ra được
rằng mọi mô hình active-speaker / lip-sync (SyncNet, TalkNet, Light-ASD) đều vô dụng —
những mô hình đó so khớp ở mức âm vị, không phải mức năng lượng thô.

CÁCH DÙNG (từ thư mục gốc dự án):
  python src/tools/measure_lip_audio_corr.py
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FPS = 25.0


def audio_rms(path, n_frames):
    """RMS audio theo từng frame video (16k mono, cửa sổ = 1/FPS giây)."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", "-"],
        capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    x = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(16000 / FPS)
    out = []
    for i in range(n_frames):
        seg = x[i * hop:(i + 1) * hop]
        out.append(float(np.sqrt((seg ** 2).mean())) if len(seg) else 0.0)
    return np.array(out)


def mouth_motion(path):
    cap = cv2.VideoCapture(path)
    prev, diffs = None, []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.abs(g - prev).mean()))
        prev = g
    cap.release()
    return np.array(diffs) if len(diffs) >= 10 else None


def main():
    import collections
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",
                    default="data/02_curate/manual/manual_all_clean_review_v2.csv",
                    help="file quyết định tay, dùng làm ground truth")
    ap.add_argument("--roi_dir", default="data/02_curate/roi_preview")
    ap.add_argument("--out", default="data/02_curate/measurements/lipcorr.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.labels, encoding="utf-8")))
    print(f"Ground truth: {args.labels}  ({len(rows)} quyết định)")
    out = []
    for i, r in enumerate(rows, 1):
        p = os.path.join(args.roi_dir, r["clip_id"] + ".mp4")
        if not os.path.isfile(p):
            continue
        mm = mouth_motion(p)
        if mm is None:
            continue
        ar = audio_rms(p, len(mm) + 1)
        if ar is None:
            continue
        ar = ar[1:len(mm) + 1]
        n = min(len(mm), len(ar))
        mm, ar = mm[:n], ar[:n]
        if mm.std() < 1e-6 or ar.std() < 1e-6:
            corr = 0.0
        else:
            corr = float(np.corrcoef(mm, ar)[0, 1])
        out.append({
            "clip_id": r["clip_id"], "decision": r["decision"], "reason": r["reason"],
            "corr": corr,
            "mouth_motion_med": float(np.median(mm)),
            "audio_rms_med": float(np.median(ar)),
            "n": n,
        })
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    if not out:
        raise SystemExit("[LỖI] không đo được clip nào (thiếu ROI preview?)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"\nĐo được {len(out)} clip -> {args.out}")

    byd = collections.defaultdict(list)
    for r in out:
        byd[r["decision"]].append(r["corr"])
    print("\n-- corr theo quyết định --")
    for k, v in byd.items():
        v = np.array(v)
        print(f"  {k:8s} n={len(v):3d}  median={np.median(v):+.3f}  mean={v.mean():+.3f}")

    byr = collections.defaultdict(list)
    for r in out:
        byr[r["reason"] or "(keep)"].append(r["corr"])
    print("\n-- corr theo lý do --")
    for k, v in sorted(byr.items()):
        v = np.array(v)
        print(f"  {k:12s} n={len(v):3d}  median={np.median(v):+.3f}")

    y = np.array([1 if r["decision"] == "keep" else 0 for r in out])
    s = np.array([r["corr"] for r in out])
    if len(set(y)) < 2:
        print("\nChỉ có một nhãn -> không tính được AUC")
        return
    print(f"\nAUC (corr dự báo KEEP): {roc_auc_score(y, s):.3f}   (0.5 = vô dụng)")
    mask = np.array([r["decision"] == "keep" or r["reason"] in ("dubbed", "voiceover")
                     for r in out])
    if mask.sum() > 10 and len(set(y[mask])) == 2:
        print(f"AUC (chỉ keep vs dubbed/voiceover): "
              f"{roc_auc_score(y[mask], s[mask]):.3f}  n={int(mask.sum())}")


if __name__ == "__main__":
    main()
