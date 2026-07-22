"""
01_build_labels.py — Gộp real + fake thành labels.csv thống nhất + chia split SPEAKER-DISJOINT

Stage 05 của pipeline (xem MODEL_PROPOSAL.md §9 Phase 1 — "Fix Data Contract").

Input:
  - REAL: data/02_curate/all_clean.csv  (label=0, có speaker_id từ 04_curate)
  - FAKE: data/03_fake/labels.csv       (label=1, schema chung 4 method 03_fake)

Output: data/05_labels/labels.csv — schema chính:
  clip_id, file_path, label, method, param, source_clip, source_video,
  speaker_id, tier, split; nếu input là SNVSM V2 thì giữ thêm provenance SNVSM.

Quy tắc chia split (chống leakage — QUAN TRỌNG NHẤT của cả project):
  1. Đơn vị chia là CONNECTED COMPONENT của đồ thị (speaker_id ∪ source_video):
     hai clip chung speaker_id HOẶC chung source_video -> buộc cùng 1 split.
     Lý do: 02_curate over-cluster (dist=0.6) chẻ 1 người thật thành nhiều
     speaker_id; nếu chỉ gom theo speaker_id thì 1 video có thể trải nhiều split
     -> cùng một người (nhiều ID) rơi vào train lẫn test = leak identity đội lốt
     "speaker-disjoint". Gom theo component khoá cả hai chiều speaker + video.
  2. FAKE luôn đi theo split của source_clip real sinh ra nó (không bao giờ để
     fake ở test trong khi real gốc ở train — leak cả identity lẫn nội dung).
  3. Tỉ lệ 70/15/15 tính trên số CLIP REAL (greedy bin-packing nhóm lớn trước,
     deterministic theo --seed).
  4. Cuối script tự VERIFY: không speaker VÀ không source_video nào xuất hiện ở
     2 split; in cảnh báo nếu method phân bố lệch giữa các split.

Chỉ dùng thư viện chuẩn.

Ví dụ:
  python src/pipeline/05_build_labels/01_build_labels.py
  python src/pipeline/05_build_labels/01_build_labels.py --ratios 0.8,0.1,0.1
"""

import os
import sys
import csv
import json
import hashlib
import random
import argparse
from collections import defaultdict, Counter
from fractions import Fraction

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT_FIELDS = ["clip_id", "file_path", "label", "method", "param",
              "source_clip", "source_video", "speaker_id", "tier", "crf",
              "orig_clip_id", "snvsm_version", "snvsm_config_id",
              "snvsm_encoder", "snvsm_preset", "snvsm_audio",
              "snvsm_sample_rate", "snvsm_channels", "snvsm_target_samples",
              "snvsm_mode", "snvsm_crf_set", "snvsm_seed", "snvsm_pair_key",
              "snvsm_video_frames", "snvsm_video_fps",
              "snvsm_video_duration_s",
              "split"]
SPLITS = ["train", "val", "test"]
EXPECTED_SNVSM_VERSION = "snvsm_v2_h264_aac16k_mono_exactdur"
EXPECTED_SNVSM_AUDIO = "aac_128k_16khz_mono"
EXPECTED_FAKE_METHODS = {
    "temporal_desync", "frame_reverse", "pitch_flatten", "anonymization"
}
SNVSM_CONTRACT_FIELDS = (
    "snvsm_version", "snvsm_config_id", "snvsm_encoder", "snvsm_preset",
    "snvsm_audio", "snvsm_sample_rate", "snvsm_channels",
    "snvsm_target_samples", "snvsm_mode", "snvsm_crf_set", "snvsm_seed",
    "snvsm_pair_key", "snvsm_video_frames", "snvsm_video_fps",
    "snvsm_video_duration_s",
)


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


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def validate_required_inputs(real_rows, fake_rows, allow_real_only=False):
    """Fail closed unless the caller explicitly requests a real-only manifest."""
    if not real_rows:
        raise ValueError("Không đọc được real")
    if not fake_rows and not allow_real_only:
        raise ValueError("Fake manifest trống/thiếu; từ chối tạo labels real-only")


def validate_file_paths(rows):
    """Missing media is a contract failure, never a row-filtering operation."""
    missing = [row.get("clip_id", "") for row in rows
               if not os.path.isfile(str(row.get("file_path", "") or ""))]
    if missing:
        raise ValueError(
            f"Thiếu file media ở {len(missing)} dòng; ví dụ {missing[:3]}"
        )
    return len(rows)


def write_rows_atomic(path, rows, overwrite=False):
    """Publish labels only after every contract/leakage check has passed."""
    if os.path.exists(path) and not overwrite:
        raise ValueError(
            f"Output đã tồn tại, từ chối ghi đè: {path}. "
            "Dùng path versioned mới hoặc --overwrite khi thật sự chủ ý."
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    partial = path + ".part"
    try:
        if os.path.exists(partial):
            os.remove(partial)
        with open(partial, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(partial, path)
    except Exception:
        try:
            if os.path.exists(partial):
                os.remove(partial)
        except OSError:
            pass
        raise


def _spk_node(row):
    sid = (row.get("speaker_id") or "").strip()
    return f"spk_{sid}" if sid not in ("", "-1") else None


def _vid_node(row):
    sv = (row.get("source_video") or "").strip()
    return f"vid_{sv}" if sv else None


def primary_node(row):
    """Node đại diện 1 clip: speaker nếu có, fallback video, fallback clip_id."""
    return _spk_node(row) or _vid_node(row) or f"clip_{row.get('clip_id', '')}"


def validate_snvsm_contract(real_rows, fake_rows):
    """Nếu một phía là SNVSM V2, bắt buộc hai phía dùng cùng normalization config."""
    rows = real_rows + fake_rows
    if not any(str(row.get(field, "") or "").strip()
               for row in rows for field in SNVSM_CONTRACT_FIELDS):
        return None
    clip_ids = [str(row.get("clip_id", "") or "").strip() for row in rows]
    if any(not clip_id for clip_id in clip_ids) or len(set(clip_ids)) != len(clip_ids):
        raise ValueError("SNVSM clip_id trống hoặc trùng")
    fields = ("snvsm_version", "snvsm_config_id", "snvsm_encoder",
              "snvsm_preset", "snvsm_audio", "snvsm_sample_rate",
              "snvsm_channels", "snvsm_mode", "snvsm_crf_set",
              "snvsm_seed")
    signatures = []
    for name, side in (("real", real_rows), ("fake", fake_rows)):
        missing = [row.get("clip_id", "") for row in side
                   if any(not str(row.get(field, "") or "").strip()
                          for field in fields)]
        if missing:
            raise ValueError(
                f"{name} thiếu provenance SNVSM ở {len(missing)} dòng; "
                f"ví dụ {missing[:3]}"
            )
        invalid = []
        for row in side:
            try:
                crf_set = [int(value) for value in row["snvsm_crf_set"].split(",")]
                valid = (
                    row["snvsm_version"].strip() == EXPECTED_SNVSM_VERSION
                    and row["snvsm_audio"].strip() == EXPECTED_SNVSM_AUDIO
                    and row["snvsm_encoder"].strip()
                    in ("libx264", "h264_nvenc")
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
                    and int(row["snvsm_target_samples"]) > 0
                    and str(row.get("snvsm_pair_key", "") or "").strip()
                    and int(row["snvsm_video_frames"]) > 0
                    and Fraction(row["snvsm_video_fps"]) > 0
                    and float(row["snvsm_video_duration_s"]) > 0
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                valid = False
            if not valid:
                invalid.append(row.get("clip_id", ""))
        if invalid:
            raise ValueError(
                f"{name} có provenance/contract SNVSM invalid ở "
                f"{len(invalid)} dòng; ví dụ {invalid[:3]}"
            )
        side_signatures = {
            tuple(str(row[field]).strip() for field in fields) for row in side
        }
        if len(side_signatures) != 1:
            raise ValueError(f"{name} có nhiều cấu hình SNVSM: {side_signatures}")
        signatures.append(next(iter(side_signatures)))
    if signatures[0] != signatures[1]:
        raise ValueError(
            f"SNVSM real/fake không cùng cấu hình: {signatures[0]} != {signatures[1]}"
        )
    return dict(zip(fields, signatures[0]))


def validate_snvsm_pair_targets(real_rows, fake_rows):
    """Every fake must preserve its paired real's PCM, visual and CRF contracts."""
    target_by_real = {}
    video_by_real = {}
    crfs_by_real = defaultdict(set)
    crf_count_by_real = Counter()
    for row in real_rows:
        target = int(row["snvsm_target_samples"])
        source = row["snvsm_pair_key"].strip()
        if source != str(row.get("orig_clip_id", "") or "").strip():
            raise ValueError(f"SNVSM real pair_key sai lineage: {row.get('clip_id', '')}")
        video = (int(row["snvsm_video_frames"]),
                 Fraction(row["snvsm_video_fps"]),
                 float(row["snvsm_video_duration_s"]))
        previous = target_by_real.get(source)
        if previous is not None and previous != target:
            raise ValueError(f"Real source {source} có hai snvsm_target_samples")
        previous_video = video_by_real.get(source)
        if (previous_video is not None
                and (previous_video[:2] != video[:2]
                     or abs(previous_video[2] - video[2]) > 1e-3)):
            raise ValueError(f"Real source {source} có hai video contract")
        target_by_real[source] = target
        video_by_real[source] = video
        crfs_by_real[source].add(int(row["crf"]))
        crf_count_by_real[source] += 1

    missing = []
    mismatched = []
    video_mismatched = []
    crfs_by_fake = defaultdict(set)
    crf_count_by_fake = Counter()
    methods_by_source = defaultdict(set)
    for row in fake_rows:
        source = str(row.get("source_clip", "") or "").strip()
        if str(row.get("snvsm_pair_key", "") or "").strip() != source:
            missing.append(row.get("clip_id", ""))
            continue
        expected = target_by_real.get(source)
        if expected is None:
            missing.append(row.get("clip_id", ""))
            continue
        actual = int(row["snvsm_target_samples"])
        if actual != expected:
            mismatched.append((row.get("clip_id", ""), actual, expected))
        actual_video = (int(row["snvsm_video_frames"]),
                        Fraction(row["snvsm_video_fps"]),
                        float(row["snvsm_video_duration_s"]))
        expected_video = video_by_real[source]
        if (actual_video[:2] != expected_video[:2]
                or abs(actual_video[2] - expected_video[2]) > 1e-3):
            video_mismatched.append(
                (row.get("clip_id", ""), actual_video, expected_video)
            )
        crfs_by_fake[(source, row.get("method", ""))].add(int(row["crf"]))
        crf_count_by_fake[(source, row.get("method", ""))] += 1
        methods_by_source[source].add(row.get("method", ""))
    if missing:
        raise ValueError(
            f"SNVSM fake thiếu real ghép cặp ở {len(missing)} dòng; ví dụ {missing[:3]}"
        )
    if mismatched:
        raise ValueError(
            "SNVSM target lệch real-fake ở "
            f"{len(mismatched)} dòng; ví dụ {mismatched[:3]}"
        )
    if video_mismatched:
        raise ValueError(
            "SNVSM video contract lệch real-fake ở "
            f"{len(video_mismatched)} dòng; ví dụ {video_mismatched[:3]}"
        )
    coverage_mismatched = [
        (source, sorted(methods_by_source.get(source, set())))
        for source in target_by_real
        if methods_by_source.get(source, set()) != EXPECTED_FAKE_METHODS
    ]
    unexpected_sources = sorted(set(methods_by_source) - set(target_by_real))
    if coverage_mismatched or unexpected_sources:
        raise ValueError(
            "SNVSM thiếu/thừa method theo source; "
            f"ví dụ {coverage_mismatched[:3]}, source lạ {unexpected_sources[:3]}"
        )
    duplicate_real_crfs = [
        source for source, count in crf_count_by_real.items()
        if count != len(crfs_by_real[source])
    ]
    duplicate_fake_crfs = [
        key for key, count in crf_count_by_fake.items()
        if count != len(crfs_by_fake[key])
    ]
    if duplicate_real_crfs or duplicate_fake_crfs:
        raise ValueError(
            "SNVSM trùng CRF trong source/method; "
            f"real={duplicate_real_crfs[:3]}, fake={duplicate_fake_crfs[:3]}"
        )
    mode = real_rows[0]["snvsm_mode"].strip()
    if mode == "random":
        bad_real = [source for source, count in crf_count_by_real.items()
                    if count != 1]
        bad_fake = [key for key, count in crf_count_by_fake.items()
                    if count != 1]
        if bad_real or bad_fake:
            raise ValueError(
                "SNVSM mode=random phải có đúng 1 CRF/source-method; "
                f"real={bad_real[:3]}, fake={bad_fake[:3]}"
            )
    if mode == "all":
        declared = {int(value) for value in real_rows[0]["snvsm_crf_set"].split(",")}
        incomplete_real = [
            source for source, values in crfs_by_real.items() if values != declared
        ]
        if incomplete_real:
            raise ValueError(
                f"SNVSM mode=all thiếu CRF real ở source {incomplete_real[:3]}"
            )
    crf_mismatched = [
        (source, method, sorted(values), sorted(crfs_by_real[source]))
        for (source, method), values in crfs_by_fake.items()
        if values != crfs_by_real[source]
    ]
    if crf_mismatched:
        raise ValueError(
            "SNVSM CRF policy lệch real-fake ở "
            f"{len(crf_mismatched)} nhóm; ví dụ {crf_mismatched[:3]}"
        )
    return len(fake_rows)


class UnionFind:
    """Gom clip vào connected component qua các cạnh (speaker_id ∪ source_video)."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:          # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def key(self, row):
        """Khoá nhóm = root component của clip (cùng component -> cùng split)."""
        return self.find(primary_node(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_csv", default="data/02_curate/all_clean.csv")
    ap.add_argument("--fake_labels", default="data/03_fake/labels.csv")
    ap.add_argument("--out", default="data/05_labels/labels.csv")
    ap.add_argument("--ratios", default="0.70,0.15,0.15", help="train,val,test (tính trên clip REAL)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check_files", dest="check_files", action="store_true",
                    help="verify file_path tồn tại (mặc định bật)")
    ap.add_argument("--no_check_files", dest="check_files", action="store_false",
                    help="bỏ kiểm file_path; chỉ dùng cho audit manifest offline")
    ap.add_argument("--allow_real_only", action="store_true",
                    help="cho phép tạo labels chỉ có real (mặc định từ chối)")
    ap.add_argument("--overwrite", action="store_true",
                    help="cho phép thay output đã tồn tại; mặc định bắt buộc path mới")
    ap.set_defaults(check_files=True)
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6, "--ratios phải là 3 số tổng = 1"

    real_rows = read_csv(args.real_csv)
    fake_rows = read_csv(args.fake_labels)
    validate_required_inputs(real_rows, fake_rows, args.allow_real_only)
    if args.check_files:
        validate_file_paths(real_rows + fake_rows)
    if not fake_rows:
        print(f"CẢNH BÁO: tạo labels real-only theo yêu cầu --allow_real_only.")
    snvsm_contract = validate_snvsm_contract(real_rows, fake_rows) if fake_rows else None
    if snvsm_contract:
        paired = validate_snvsm_pair_targets(real_rows, fake_rows)
        print(f"SNVSM contract: {snvsm_contract['snvsm_version']} | "
              f"config={snvsm_contract['snvsm_config_id']} | "
              f"{snvsm_contract['snvsm_encoder']}/{snvsm_contract['snvsm_preset']} | "
              f"{snvsm_contract['snvsm_audio']} | paired_target={paired}")

    # ---------- 1) Gom connected component (speaker_id ∪ source_video) trên REAL ----------
    uf = UnionFind()
    for r in real_rows:                             # nối speaker <-> video của cùng 1 clip
        uf.union(primary_node(r), _vid_node(r) or primary_node(r))
    groups = defaultdict(list)                      # component root -> [real row]
    for r in real_rows:
        groups[uf.key(r)].append(r)

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
        sp = split_of_group[uf.key(r)]
        cid = r.get("clip_id", "")
        for lookup_id in (cid, r.get("orig_clip_id", "")):
            if lookup_id:
                previous = split_of_clip.get(lookup_id)
                if previous is not None and previous != sp:
                    raise ValueError(f"Real lookup ID {lookup_id} nằm ở hai split")
                split_of_clip[lookup_id] = sp
        row = {field: r.get(field, "") for field in OUT_FIELDS}
        row.update({"clip_id": cid, "label": 0, "method": "real",
                    "param": "", "source_clip": "", "split": sp})
        out_rows.append(row)

    # ---------- 4) FAKE đi theo split của source_clip ----------
    for r in fake_rows:
        src = r.get("source_clip", "")
        sp = split_of_clip.get(src)
        if sp is None:
            raise ValueError(
                f"Fake mồ côi, không tìm thấy real nguồn: "
                f"{r.get('clip_id', '')} -> {src}"
            )
        row = {f: r.get(f, "") for f in OUT_FIELDS[:-1]}
        row["label"] = 1
        row["split"] = sp
        out_rows.append(row)

    # ---------- 5) Thống kê + VERIFY chống leakage ----------

    stat = defaultdict(Counter)
    spk_in_split = defaultdict(set)                  # speaker_id  -> {split}
    vid_in_split = defaultdict(set)                  # source_video -> {split}
    for r in out_rows:
        stat[r["split"]][f"label{r['label']}"] += 1
        stat[r["split"]][r["method"]] += 1
        sid = (r.get("speaker_id") or "").strip()
        if sid not in ("", "-1"):
            spk_in_split[sid].add(r["split"])
        sv = (r.get("source_video") or "").strip()
        if sv:
            vid_in_split[sv].add(r["split"])

    print(f"\n{'split':6} | {'real':>6} | {'fake':>6} | theo method")
    for sp in SPLITS:
        c = stat[sp]
        methods = {m: n for m, n in c.items() if not m.startswith("label") and m != "real"}
        print(f"{sp:6} | {c['label0']:6} | {c['label1']:6} | {dict(sorted(methods.items()))}")

    spk_leaks = {k: v for k, v in spk_in_split.items() if len(v) > 1}
    vid_leaks = {k: v for k, v in vid_in_split.items() if len(v) > 1}
    if spk_leaks or vid_leaks:
        if spk_leaks:
            print(f"\n❌ LEAKAGE: {len(spk_leaks)} speaker_id ở >1 split! Ví dụ: "
                  f"{list(spk_leaks.items())[:3]}")
        if vid_leaks:
            print(f"\n❌ LEAKAGE: {len(vid_leaks)} source_video ở >1 split! Ví dụ: "
                  f"{list(vid_leaks.items())[:3]}")
        sys.exit(1)
    print("\n✅ Verify: không speaker_id NÀO và không source_video NÀO nằm ở 2 split.")

    # ---------- 6) Chỉ publish sau khi tất cả gate đã đạt ----------
    write_rows_atomic(args.out, out_rows, overwrite=args.overwrite)
    print(f"Đã ghi atomic {len(out_rows)} dòng -> {args.out}")


if __name__ == "__main__":
    main()
