"""
02_frame_reverse.py — Pseudo-fake: ĐẢO NGƯỢC một đoạn VIDEO (Frame Reverse)

Tuần 2 (Pseudo-fake Engineering) — kỹ thuật thứ 2 theo docs/Pipeline.

Ý tưởng: chọn NGẪU NHIÊN một cửa sổ dài 0.3–1.0s trong clip, đảo ngược thứ tự
frame của RIÊNG đoạn đó (video chạy lùi trong cửa sổ), phần còn lại giữ xuôi;
AUDIO GIỮ NGUYÊN hoàn toàn. Kết quả: trong cửa sổ đó khẩu hình miệng chạy ngược
so với tiếng -> lệch pha cục bộ -> deepfake. Đây là loại fake THỊ GIÁC THUẦN
(đối ngẫu với 03_pitch_flatten là AUDIO thuần).

Khác với 01_temporal_desync (lệch toàn cục, đều): 02 lệch CỤC BỘ và mạnh trong
một khoảng ngắn -> bổ sung đa dạng kiểu lệch pha cho dataset.

Chống học-tủ (rất quan trọng):
  - AUDIO copy nguyên (-c:a copy) -> khác biệt nằm ở HÌNH, không phải audio.
  - Vị trí + độ dài cửa sổ NGẪU NHIÊN -> tránh model học "đảo đúng giữa clip".
  - Video buộc phải re-encode (concat + reverse không copy được). Điều này tạo
    artifact nén CHỈ trên fake -> nếu không xử lý sẽ leak codec. Bước nén SNVSM
    V2 áp ĐỐI XỨNG real+fake để giảm shortcut codec.
  - V2 đảo theo chỉ số frame, bỏ -shortest và chỉ publish nếu frame/FPS/duration
    video cùng audio target vẫn khớp source.

Chỉ dùng thư viện chuẩn + ffmpeg/ffprobe trong PATH.

Ví dụ:
  python 02_frame_reverse.py --input_csv data/02_curate/manifests/all_clean.csv \\
      --out_dir data/fake --labels data/labels.csv
  # đổi dải độ dài cửa sổ đảo:
  python 02_frame_reverse.py --input_csv ... --min_sec 0.3 --max_sec 1.0
  # thử 5 clip:
  python 02_frame_reverse.py --input_csv ... --limit 5
"""

import os
import sys
import csv
import random
import argparse
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pipeline.fake_media_contract import (
    is_valid_repaired_output,
    probe_media,
    publish_validated,
    remove_if_exists,
    same_file_path,
)
from src.pipeline.timeline_contract import (
    TIMELINE_FIELDS,
    build_timeline_contract,
    validate_timeline_contract,
)

try:                                  # in được tiếng Việt trên console Windows (cp1252)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

GENERATOR_VERSION = "frame_reverse_v2_exact_timeline_v1"
DEFAULT_OUT_DIR = "data/03_fake/frame_reverse_v2"
DEFAULT_LABELS = "data/03_fake/manifests/v2/frame_reverse.csv"
LABEL_FIELDS = ["clip_id", "file_path", "label", "method", "param",
                "source_clip", "source_video", "speaker_id", "tier",
                *TIMELINE_FIELDS]


def assert_manifest_compatible(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {}
    for row in rows:
        clip_id = row.get("clip_id", "")
        if (not clip_id or clip_id in by_id
                or row.get("method") != "frame_reverse"
                or f"generator={GENERATOR_VERSION}" not in row.get("param", "")):
            raise ValueError(f"Manifest frame_reverse V2 không tương thích: {path}")
        contract = {
            field: row.get(field, "") for field in TIMELINE_FIELDS
        }
        validate_timeline_contract(contract, "frame_reverse")
        by_id[clip_id] = row
    return by_id


def make_reverse(in_path, out_path, start_frame, end_frame, source_media=None):
    """
    Đảo ngược video theo frame [start_frame, end_frame), audio copy nguyên.
    Output chỉ được publish nếu media contract vẫn khớp source.
    """
    source_media = source_media or probe_media(in_path)
    if source_media is None:
        return False
    total = source_media["video_frames"]
    if not 0 < start_frame < end_frame < total:
        return False
    vf = (
        "[0:v]split=3[pre][mid][post];"
        f"[pre]trim=start_frame=0:end_frame={start_frame},setpts=PTS-STARTPTS[a];"
        f"[mid]trim=start_frame={start_frame}:end_frame={end_frame},"
        f"reverse,setpts=PTS-STARTPTS[b];"
        f"[post]trim=start_frame={end_frame}:end_frame={total},setpts=PTS-STARTPTS[c];"
        f"[a][b][c]concat=n=3:v=1:a=0[v]"
    )
    partial_path = out_path + ".part.mp4"
    remove_if_exists(partial_path)
    def run(filter_graph, audio_map, audio_codec):
        remove_if_exists(partial_path)
        cmd = ["ffmpeg", "-y", "-i", in_path,
               "-filter_complex", filter_graph,
               "-map", "[v]", "-map", audio_map,
               "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
               "-pix_fmt", "yuv420p", "-c:a", audio_codec,
               "-r", str(source_media["video_fps"]),
               "-frames:v", str(total),
               "-loglevel", "error", partial_path]
        proc = subprocess.run(cmd, capture_output=True)
        return (proc.returncode == 0
                and publish_validated(partial_path, out_path, source_media))

    if run(vf, "0:a:0", "copy"):
        return True
    lossless_graph = (
        vf + f";[0:a:0]atrim=end_sample={source_media['audio_native_samples']},"
        "asetpts=PTS-STARTPTS[aout]"
    )
    return run(lossless_graph, "[aout]", "alac")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="data/02_curate/manifests/all_clean.csv",
                    help="CSV clip real (cần cột file_path) — mặc định tập sạch từ 04_curate")
    ap.add_argument("--path_col", default="file_path")
    ap.add_argument("--id_col", default="clip_id")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--min_sec", type=float, default=0.3, help="độ dài tối thiểu cửa sổ đảo (s)")
    ap.add_argument("--max_sec", type=float, default=1.0, help="độ dài tối đa cửa sổ đảo (s)")
    ap.add_argument("--n_per_clip", type=int, default=1, help="số fake sinh mỗi clip")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="chỉ xử lý N clip đầu (để test)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    existing_rows = assert_manifest_compatible(args.labels)

    with open(args.input_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]
    print(f"Đọc {len(rows)} clip real từ {args.input_csv}")

    new_file = not os.path.exists(args.labels) or os.path.getsize(args.labels) == 0
    os.makedirs(os.path.dirname(os.path.abspath(args.labels)) or ".", exist_ok=True)
    lf = open(args.labels, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(lf, fieldnames=LABEL_FIELDS)
    if new_file:
        writer.writeheader()

    made = resumed = repaired = skipped = failed = 0
    for i, r in enumerate(rows):
        src_path = r.get(args.path_col, "")
        src_id = r.get(args.id_col, f"clip{i:06d}")
        if not src_path or not os.path.isfile(src_path):
            skipped += 1
            continue

        media = probe_media(src_path)
        if media is None:
            skipped += 1
            continue
        common_duration = min(media["audio_duration"], media["video_duration"])

        for k in range(args.n_per_clip):
            # RNG theo (seed, src_id, k) chứ KHÔNG phải random.seed() toàn cục: với seed
            # toàn cục, cửa sổ đảo của một clip phụ thuộc thứ tự nó được xử lý, nên
            # --limit, resume hay đổi thứ tự manifest đều cho ra tập fake khác.
            rng = random.Random(f"{args.seed}:{src_id}:{k}")
            d = rng.uniform(args.min_sec, args.max_sec)
            if common_duration <= d + 0.4:   # không đủ chỗ chừa 2 đầu -> bỏ
                skipped += 1
                continue
            fps = media["video_fps"]
            start_frame = max(1, round(rng.uniform(
                0.1, common_duration - d - 0.1
            ) * float(fps)))
            length_frames = max(2, round(d * float(fps)))
            end_frame = min(
                start_frame + length_frames,
                int(common_duration * float(fps)),
                media["video_frames"] - 1,
            )
            t0 = start_frame / float(fps)
            t1 = end_frame / float(fps)
            fake_id = f"{src_id}_revv2f{start_frame}-{end_frame}" + (f"_{k}" if args.n_per_clip > 1 else "")
            out_path = os.path.abspath(os.path.join(args.out_dir, fake_id + ".mp4"))
            expected_param = (f"reverse_frames=[{start_frame},{end_frame});"
                              f"generator={GENERATOR_VERSION}")

            existing = existing_rows.get(fake_id)
            if existing:
                recorded_path = os.path.abspath(existing.get("file_path", ""))
                if (not same_file_path(recorded_path, out_path)
                        or existing.get("param", "") != expected_param):
                    raise ValueError(f"Resume contract sai cho {fake_id}")
                if is_valid_repaired_output(out_path, media):
                    resumed += 1
                    continue
            if make_reverse(src_path, out_path, start_frame, end_frame, media):
                if existing:
                    repaired += 1
                    continue
                output_row = {
                    "clip_id": fake_id,
                    "file_path": out_path,
                    "label": 1,
                    "method": "frame_reverse",
                    "param": expected_param,
                    "source_clip": src_id,
                    "source_video": r.get("source_video", ""),
                    "speaker_id": r.get("speaker_id", ""),
                    "tier": r.get("tier", ""),
                }
                output_row.update(build_timeline_contract(
                    media["audio_duration"], media["video_duration"],
                    manipulation_scope="local", manipulation=(t0, t1),
                ))
                writer.writerow(output_row)
                made += 1
            else:
                failed += 1

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}  made={made} resume={resumed} "
                  f"repair={repaired} skip={skipped} fail={failed}")

    lf.close()
    print(f"\nXong. Fake tạo được: {made} | resume: {resumed} | "
          f"sửa file: {repaired} | bỏ qua: {skipped} | lỗi: {failed}")
    print(f"  Video -> {args.out_dir}/  | nhãn (append) -> {args.labels}")
    print("Lưu ý: video re-encode -> nén SNVSM 4 mức CRF ở bước sau sẽ đồng bộ codec real+fake.")
    if skipped or failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
