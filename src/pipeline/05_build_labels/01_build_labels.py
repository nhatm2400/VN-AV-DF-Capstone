"""
01_build_labels.py — Gộp real + fake thành labels.csv thống nhất + chia split SPEAKER-DISJOINT

Stage 05 của pipeline (xem MODEL_PROPOSAL.md §9 Phase 1 — "Fix Data Contract").

Input:
  - REAL: data/02_curate/all_clean.csv  (label=0, có speaker_id từ 04_curate)
  - FAKE: data/03_fake/labels.csv       (label=1, schema chung 4 method 03_fake)

Output: data/05_labels/labels.csv — schema:
  clip_id, file_path, label, method, param, source_clip, source_video,
  speaker_id, tier, split

Quy tắc chia split (chống leakage — QUAN TRỌNG NHẤT của cả project):
  1. Đơn vị chia là NHÓM speaker (speaker_id; fallback source_video) — mọi clip
     của cùng một người nói nằm trọn trong 1 split -> speaker-disjoint.
  2. FAKE luôn đi theo split của source_clip real sinh ra nó (không bao giờ để
     fake ở test trong khi real gốc ở train — leak cả identity lẫn nội dung).
  3. Tỉ lệ 70/15/15 tính trên số CLIP REAL (greedy bin-packing nhóm lớn trước,
     deterministic theo --seed).
  4. Cuối script tự VERIFY: không speaker nào xuất hiện ở 2 split; in cảnh báo
     nếu method phân bố lệch giữa các split.

Chỉ dùng thư viện chuẩn.

Ví dụ:
  python src/pipeline/05_build_labels/01_build_labels.py
  python src/pipeline/05_build_labels/01_build_labels.py --ratios 0.8,0.1,0.1
"""

import os
import sys
import csv
import random
import argparse
from collections import defaultdict, Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT_FIELDS = ["clip_id", "file_path", "label", "method", "param",
              "source_clip", "source_video", "speaker_id", "tier", "split"]
SPLITS = ["train", "val", "test"]


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def group_key(row):
    """Đơn vị chia split: speaker_id nếu có, fallback source_video, fallback clip_id."""
    sid = (row.get("speaker_id") or "").strip()
    if sid not in ("", "-1"):
        return f"spk_{sid}"
    sv = (row.get("source_video") or "").strip()
    if sv:
        return f"vid_{sv}"
    return f"clip_{row.get('clip_id', '')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_csv", default="data/02_curate/all_clean.csv")
    ap.add_argument("--fake_labels", default="data/03_fake/labels.csv")
    ap.add_argument("--out", default="data/05_labels/labels.csv")
    ap.add_argument("--ratios", default="0.70,0.15,0.15", help="train,val,test (tính trên clip REAL)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check_files", action="store_true", help="verify file_path tồn tại (chậm hơn)")
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6, "--ratios phải là 3 số tổng = 1"

    real_rows = read_csv(args.real_csv)
    fake_rows = read_csv(args.fake_labels)
    if not real_rows:
        print(f"LỖI: không đọc được real từ {args.real_csv}")
        sys.exit(1)
    if not fake_rows:
        print(f"CẢNH BÁO: chưa có fake ({args.fake_labels} trống/thiếu) — labels chỉ có real.")

    # ---------- 1) Gom nhóm speaker trên tập REAL ----------
    groups = defaultdict(list)                      # key -> [real row]
    for r in real_rows:
        groups[group_key(r)].append(r)

    # ---------- 2) Greedy bin-packing: nhóm to trước, nhét vào split đang thiếu nhiều nhất ----------
    n_real = len(real_rows)
    targets = [ratios[i] * n_real for i in range(3)]
    filled = [0, 0, 0]
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))   # to trước, tie-break tên
    rng = random.Random(args.seed)
    # xáo trong từng "bậc" cùng kích thước để seed có tác dụng nhưng vẫn deterministic
    by_size = defaultdict(list)
    for k, v in order:
        by_size[len(v)].append((k, v))
    order = []
    for size in sorted(by_size, reverse=True):
        bucket = by_size[size]
        rng.shuffle(bucket)
        order.extend(bucket)

    split_of_group = {}
    for k, rows in order:
        # split còn "đói" nhất theo tỉ lệ
        deficits = [(targets[i] - filled[i]) / max(targets[i], 1e-9) for i in range(3)]
        i = max(range(3), key=lambda j: deficits[j])
        split_of_group[k] = SPLITS[i]
        filled[i] += len(rows)

    # ---------- 3) Gán split cho từng clip real + index tra cứu cho fake ----------
    out_rows = []
    split_of_clip = {}
    for r in real_rows:
        sp = split_of_group[group_key(r)]
        cid = r.get("clip_id", "")
        split_of_clip[cid] = sp
        out_rows.append({
            "clip_id": cid,
            "file_path": r.get("file_path", ""),
            "label": 0,
            "method": "real",
            "param": "",
            "source_clip": "",
            "source_video": r.get("source_video", ""),
            "speaker_id": r.get("speaker_id", ""),
            "tier": r.get("tier", ""),
            "split": sp,
        })

    # ---------- 4) FAKE đi theo split của source_clip ----------
    orphan = 0
    for r in fake_rows:
        src = r.get("source_clip", "")
        sp = split_of_clip.get(src)
        if sp is None:                              # real gốc không nằm trong tập sạch
            sp = split_of_group.get(group_key(r))   # thử theo speaker
            if sp is None:
                orphan += 1
                continue                            # bỏ fake mồ côi (an toàn hơn là đoán)
        row = {f: r.get(f, "") for f in OUT_FIELDS[:-1]}
        row["label"] = 1
        row["split"] = sp
        out_rows.append(row)

    # ---------- 5) (tùy chọn) verify file tồn tại ----------
    if args.check_files:
        before = len(out_rows)
        out_rows = [r for r in out_rows if os.path.isfile(r["file_path"])]
        if len(out_rows) < before:
            print(f"CẢNH BÁO: loại {before - len(out_rows)} dòng có file_path không tồn tại")

    # ---------- 6) Ghi ----------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    # ---------- 7) Thống kê + VERIFY chống leakage ----------
    print(f"\nĐã ghi {len(out_rows)} dòng -> {args.out}")
    if orphan:
        print(f"  (bỏ {orphan} fake mồ côi — source_clip không có trong tập real sạch)")

    stat = defaultdict(Counter)
    spk_in_split = defaultdict(set)
    for r in out_rows:
        stat[r["split"]][f"label{r['label']}"] += 1
        stat[r["split"]][r["method"]] += 1
        key = group_key(r)
        spk_in_split[key].add(r["split"])

    print(f"\n{'split':6} | {'real':>6} | {'fake':>6} | theo method")
    for sp in SPLITS:
        c = stat[sp]
        methods = {m: n for m, n in c.items() if not m.startswith("label") and m != "real"}
        print(f"{sp:6} | {c['label0']:6} | {c['label1']:6} | {dict(sorted(methods.items()))}")

    leaks = {k: v for k, v in spk_in_split.items() if len(v) > 1}
    if leaks:
        print(f"\n❌ LEAKAGE: {len(leaks)} nhóm speaker xuất hiện ở >1 split! Ví dụ: "
              f"{list(leaks.items())[:3]}")
        sys.exit(1)
    print("\n✅ Verify speaker-disjoint: không nhóm speaker nào nằm ở 2 split.")


if __name__ == "__main__":
    main()
