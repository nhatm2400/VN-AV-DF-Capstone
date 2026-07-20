"""
train.py — Training loop AVSP-Net (chạy từ repo root).

Baselines §7 MODEL_PROPOSAL.md qua --branches:
  python src/train/train.py --branches audio                    # 1. audio-only
  python src/train/train.py --branches visual                   # 2. visual-only (mouth ROI)
  python src/train/train.py --branches audio,visual             # 3. AV fusion
  python src/train/train.py --branches audio,visual,prosody     # 4. AVSP-Net full (mặc định)

Yêu cầu trước đó: 04_extract_features (w2v bật) + 05_build_labels.
Checkpoint + history ghi vào experiments/<run_name>/ (best theo val AUC).
"""

import os
import sys
import json
import time
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.avsp_net import AVSPNet, compute_losses          # noqa: E402
from train.dataset import AVSPDataset, collate              # noqa: E402


def auc_score(labels, scores):
    """ROC-AUC không cần sklearn (Mann-Whitney U / rank)."""
    pairs = sorted(zip(scores, labels))
    ranks, i = {}, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = sum(ranks[k] for k, (_, l) in enumerate(pairs) if l == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@torch.no_grad()
def evaluate_split(model, loader, device):
    model.eval()
    labels, scores = [], []
    for batch in loader:
        out = model(w2v=batch["w2v"].to(device),
                    mouth=batch["mouth"].to(device),
                    prosody=batch["prosody"].to(device))
        scores += torch.sigmoid(out["logit"]).cpu().tolist()
        labels += batch["label"].tolist()
    auc = auc_score(labels, scores)
    acc = sum((s > 0.5) == bool(l) for s, l in zip(scores, labels)) / max(len(labels), 1)
    return {"auc": auc, "acc": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/05_labels/labels.csv")
    ap.add_argument("--features", default="data/04_features")
    ap.add_argument("--branches", default="audio,visual,prosody")
    ap.add_argument("--run_name", default=None, help="mặc định: avsp_<branches>")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=7, help="early stop theo val AUC")
    ap.add_argument("--real_blur_aug_p", type=float, default=0.25,
                    help="xác suất blur mouth ROI của REAL (train) chống leak 'mờ=fake'; "
                         "≈ tỉ lệ anon trong fake. 0 = tắt")
    ap.add_argument("--amp", action="store_true", help="mixed precision (GPU)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    branches = tuple(b.strip() for b in args.branches.split(",") if b.strip())
    run_name = args.run_name or ("avsp_" + "_".join(branches))
    out_dir = os.path.join("experiments", run_name)
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Run: {run_name} | branches={branches} | device={device}")

    ds_tr = AVSPDataset(args.labels, args.features, "train", branches,
                        real_blur_aug_p=args.real_blur_aug_p)   # blur aug CHỈ ở train
    ds_va = AVSPDataset(args.labels, args.features, "val", branches)   # val: không aug
    print(f"train={len(ds_tr)} val={len(ds_va)} | pos_weight={ds_tr.pos_weight().item():.3f} "
          f"| real_blur_aug_p={args.real_blur_aug_p}")
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                       num_workers=args.workers, collate_fn=collate, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False,
                       num_workers=args.workers, collate_fn=collate)

    model = AVSPNet(branches=branches).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tham số trainable: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=args.amp)
    pos_weight = ds_tr.pos_weight().to(device)

    best_auc, best_epoch, history = -1.0, -1, []
    for epoch in range(args.epochs):
        model.train()
        t0, run_loss, n_b = time.time(), 0.0, 0
        for batch in dl_tr:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.split(":")[0], enabled=args.amp):
                out = model(w2v=batch["w2v"].to(device),
                            mouth=batch["mouth"].to(device),
                            prosody=batch["prosody"].to(device))
                losses = compute_losses(out, batch["label"].to(device),
                                        batch["offset"].to(device), pos_weight=pos_weight)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            run_loss += losses["total"].item()
            n_b += 1
        sched.step()

        val = evaluate_split(model, dl_va, device)
        history.append({"epoch": epoch, "train_loss": run_loss / max(n_b, 1),
                        "val_auc": val["auc"], "val_acc": val["acc"],
                        "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)})
        print(f"[{epoch+1:3}/{args.epochs}] loss={history[-1]['train_loss']:.4f} "
              f"val_auc={val['auc']:.4f} val_acc={val['acc']:.4f} ({history[-1]['sec']}s)")

        ckpt = {"model": model.state_dict(), "branches": branches,
                "epoch": epoch, "val": val, "args": vars(args)}
        torch.save(ckpt, os.path.join(out_dir, "last.pt"))
        if val["auc"] > best_auc:
            best_auc, best_epoch = val["auc"], epoch
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))
            print(f"  ↳ best mới (val_auc={best_auc:.4f}) -> {out_dir}/best.pt")
        with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)

        if epoch - best_epoch >= args.patience:
            print(f"Early stop (val AUC không cải thiện {args.patience} epoch).")
            break

    print(f"\nXong. Best val_auc={best_auc:.4f} @ epoch {best_epoch+1} -> {out_dir}/best.pt")
    print(f"Đánh giá test:  python src/eval/evaluate.py --ckpt {out_dir}/best.pt")


if __name__ == "__main__":
    main()
