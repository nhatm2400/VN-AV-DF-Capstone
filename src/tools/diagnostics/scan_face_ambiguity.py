"""Quét toàn bộ all_clean: luật 'mặt to nhất' của stage 04 có đáng tin không?

Lấy 5 khung rải đều mỗi clip, chạy YOLO-face, đo:
  n_faces_med   số mặt phát hiện được (median trên 5 khung)
  ratio_med     diện tích mặt NHÌ / mặt NHẤT (median). Càng gần 1 = càng dễ chọn nhầm.
  cx_spread     độ lệch tâm box được chọn giữa các khung (box có nhảy người không)
"""
import argparse
import csv
import json

import cv2
import numpy as np
from ultralytics import YOLO


def probe(model, path, n_frames, conf):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1
    if total < n_frames:
        cap.release()
        return None
    idxs = np.linspace(0, total - 1, n_frames).astype(int)

    counts, ratios, cxs, areas = [], [], [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        res = model.predict(frame, verbose=False, conf=conf)[0]
        n = 0 if res.boxes is None else len(res.boxes)
        counts.append(n)
        if n == 0:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        a = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        order = np.argsort(-a)
        b = xyxy[order[0]]
        cxs.append(float((b[0] + b[2]) / 2 / W))
        areas.append(float(a[order[0]] / (W * H)))
        ratios.append(float(a[order[1]] / a[order[0]]) if n > 1 else 0.0)
    cap.release()
    if not counts:
        return None
    return {
        "n_faces_med": float(np.median(counts)),
        "n_faces_max": int(np.max(counts)),
        "ratio_med": float(np.median(ratios)) if ratios else 0.0,
        "ratio_max": float(np.max(ratios)) if ratios else 0.0,
        "cx_spread": float(np.max(cxs) - np.min(cxs)) if len(cxs) > 1 else 0.0,
        "area_med": float(np.median(areas)) if areas else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/02_curate/manifests/all_clean.csv")
    ap.add_argument("--face_model", default="yolov8n-face.pt")
    ap.add_argument("--n_frames", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out",
                    default="data/02_curate/measurements/face_ambiguity.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    model = YOLO(args.face_model)

    out, fail = [], 0
    for i, r in enumerate(rows, 1):
        m = probe(model, r["file_path"], args.n_frames, args.conf)
        if m is None:
            fail += 1
            continue
        m.update(clip_id=r["clip_id"], source_video=r["source_video"],
                 speaker_id=r.get("speaker_id", ""))
        out.append(m)
        if i % 250 == 0:
            print(f"  {i}/{len(rows)}  fail={fail}", flush=True)

    json.dump(out, open(args.out, "w", encoding="utf-8"))
    print(f"\nXong {len(out)}/{len(rows)} clip (fail {fail}) -> {args.out}")

    nf = np.array([r["n_faces_med"] for r in out])
    rt = np.array([r["ratio_med"] for r in out])
    print("\n-- so mat phat hien duoc (median tren 5 khung) --")
    for k in (1, 2, 3, 4):
        sel = (nf >= k)
        print(f"  >= {k} mat: {int(sel.sum()):5d}/{len(nf)} ({100*sel.mean():.1f}%)")
    print("\n-- ti le dien tich mat NHI/NHAT (cang cao cang de chon nham) --")
    for t in (0.4, 0.5, 0.6, 0.7, 0.8):
        sel = rt > t
        print(f"  > {t}: {int(sel.sum()):5d}/{len(rt)} ({100*sel.mean():.1f}%)")
    risky = (nf >= 2) & (rt > 0.6)
    print(f"\n  RUI RO (>=2 mat VA ti le >0.6): {int(risky.sum())}/{len(rt)} "
          f"({100*risky.mean():.1f}%)")


if __name__ == "__main__":
    main()
