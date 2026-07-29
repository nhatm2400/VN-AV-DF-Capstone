"""
evaluate.py — Đánh giá AVSP-Net trên split test (speaker-disjoint).

Metrics (MODEL_PROPOSAL.md §9 Phase 5 — tối thiểu 4 metrics, KHÔNG cosine sim):
  accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
  method-wise recall + F1 (từng loại fake vs toàn bộ real) — bắt buộc để biết
  model bắt được kênh tấn công nào (desync/reverse/pitch/anon).

Ví dụ:
  python src/eval/evaluate.py --ckpt experiments/avsp_audio_visual_prosody/best.pt
  python src/eval/evaluate.py --ckpt ... --split val --thresh 0.4
"""

import os
import sys
import json
import argparse
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.avsp_net import AVSPNet                          # noqa: E402
from train.dataset import AVSPDataset, collate              # noqa: E402
from train.train import auc_score                           # noqa: E402


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1e-9)
    r = tp / max(tp + fn, 1e-9)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--labels", default="data/05_labels/labels.csv")
    ap.add_argument("--features", default="data/04_features")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--out_json", default=None, help="mặc định: <ckpt_dir>/eval_<split>.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    branches = tuple(ckpt.get("branches", ("audio", "visual", "prosody")))
    model = AVSPNet(branches=branches).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Checkpoint: {args.ckpt} (epoch {ckpt.get('epoch')}) | branches={branches}")

    ds = AVSPDataset(args.labels, args.features, args.split, branches)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, collate_fn=collate)
    print(f"{args.split}: {len(ds)} sample")

    labels, scores, methods = [], [], []
    with torch.no_grad():
        for batch in dl:
            out = model(w2v=batch["w2v"].to(device),
                        mouth=batch["mouth"].to(device),
                        prosody=batch["prosody"].to(device))
            scores += torch.sigmoid(out["logit"]).cpu().tolist()
            labels += batch["label"].tolist()
            methods += batch["method"]

    preds = [int(s > args.thresh) for s in scores]
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    precision, recall, f1 = prf(tp, fp, fn)
    acc = (tp + tn) / max(len(labels), 1)
    auc = auc_score(labels, scores)

    print(f"\n===== TỔNG THỂ ({args.split}, thresh={args.thresh}) =====")
    print(f"accuracy : {acc:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    # ---- method-wise: mỗi method fake vs TOÀN BỘ real ----
    by_m = defaultdict(lambda: {"scores": [], "preds": []})
    real_preds = [p for p, l in zip(preds, labels) if l == 0]
    real_scores = [s for s, l in zip(scores, labels) if l == 0]
    for s, p, l, m in zip(scores, preds, labels, methods):
        if l == 1:
            by_m[m]["scores"].append(s)
            by_m[m]["preds"].append(p)

    method_stats = {}
    print(f"\n===== THEO METHOD (vs {len(real_preds)} real) =====")
    print(f"{'method':18} | {'n':>5} | {'recall':>7} | {'F1':>7} | {'AUC':>7}")
    for m in sorted(by_m):
        mp = by_m[m]["preds"]
        m_tp = sum(mp)
        m_fn = len(mp) - m_tp
        m_fp = sum(real_preds)                       # real bị đoán fake
        _, m_rec, m_f1 = prf(m_tp, m_fp, m_fn)
        m_auc = auc_score([0] * len(real_scores) + [1] * len(by_m[m]["scores"]),
                          real_scores + by_m[m]["scores"])
        method_stats[m] = {"n": len(mp), "recall": m_rec, "f1": m_f1, "auc": m_auc}
        print(f"{m:18} | {len(mp):5} | {m_rec:7.4f} | {m_f1:7.4f} | {m_auc:7.4f}")

    fpr_real = fp / max(fp + tn, 1)
    print(f"\nFalse-positive rate trên real: {fpr_real:.4f}")

    result = {"split": args.split, "thresh": args.thresh, "n": len(labels),
              "accuracy": acc, "precision": precision, "recall": recall,
              "f1": f1, "roc_auc": auc, "fpr_real": fpr_real,
              "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
              "method_wise": method_stats, "ckpt": args.ckpt, "branches": list(branches)}
    out_json = args.out_json or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)), f"eval_{args.split}.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu -> {out_json}")


if __name__ == "__main__":
    main()
