"""
01_temporal_desync.py — Pseudo-fake: LỆCH PHA audio-visual (Temporal Desync)

Tuần 2 (Pseudo-fake Engineering) — kỹ thuật ƯU TIÊN LÀM ĐẦU theo docs/Pipeline.

Ý tưởng: dịch audio so với video một khoảng = 3 / 7 / 15 frames (frame-based, đổi
ra giây theo fps thật của clip). Đây là loại fake đúng mục tiêu PAMF: tiếng và
khẩu hình lệch pha.

Chống học-tủ (rất quan trọng):
  - Video bitstream được COPY nguyên (không re-encode). Audio trung gian khác
    codec và có circular-wrap; sau SNVSM đối xứng, timing trong miền valid mới
    là tín hiệu mục tiêu. Model không được dùng seam/mask shape làm bằng chứng.
  - Audio được XOAY VÒNG theo đúng số sample thay vì chỉ sửa timestamp bằng
    -itsoffset. Cả hai hướng giữ nguyên duration/frame count, không chèn silence
    và không dùng -shortest. Phần wrap ở biên được ghi rõ theo cả tọa độ audio
    và visual bằng structured timeline fields để V2a tạo mask đúng miền thời gian.
  - Audio trung gian dùng ALAC lossless. Bước SNVSM sau đó mới encode AAC giống
    nhau cho cả real/fake, tránh fake temporal bị thêm một lần nén AAC riêng.
  - Độ lệch + hướng (audio sớm/muộn) chọn NGẪU NHIÊN -> tránh model học "đúng
    0.4s = fake". (Bước nén SNVSM 4 mức CRF ở sau sẽ đồng bộ codec cho cả real
    lẫn fake để giảm chênh lệch do re-encode audio.)

Chỉ dùng thư viện chuẩn + ffmpeg/ffprobe trong PATH (không cần ffmpeg-python).

Ví dụ:
  # Mỗi clip 1 fake, lệch ngẫu nhiên trong {3,7,15} frames, hướng ngẫu nhiên
  python 01_temporal_desync.py --input_csv data/clips/tier1_clean.csv \\
      --out_dir data/03_fake/temporal_v2 \\
      --labels data/03_fake/manifests/v2/temporal_desync.csv

  # Sinh cả 3 mức lệch cho mỗi clip
  python 01_temporal_desync.py --input_csv ... --mode all

  # Thử 5 clip để kiểm
  python 01_temporal_desync.py --input_csv ... --limit 5
"""

import os
import sys
import csv
import json
import random
import argparse
import subprocess
from fractions import Fraction

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pipeline.timeline_contract import (
    BOUNDARY_CIRCULAR_WRAP,
    TIMELINE_FIELDS,
    build_timeline_contract,
    validate_timeline_contract,
)

try:                                  # in được tiếng Việt trên console Windows (cp1252)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SHIFTS = [3, 7, 15]   # frames — theo pipeline
GENERATOR_VERSION = "temporal_v2_structured_timeline_v1"
DEFAULT_OUT_DIR = "data/03_fake/temporal_v2"
DEFAULT_LABELS = "data/03_fake/manifests/v2/temporal_desync.csv"
REQUIRED_PARAM_KEYS = (
    f"generator={GENERATOR_VERSION}",
)
LABEL_FIELDS = ["clip_id", "file_path", "label", "method", "param",
                "source_clip", "source_video", "speaker_id", "tier",
                *TIMELINE_FIELDS]


def _round_fraction(value):
    """Round Fraction dương về int, không qua float."""
    return int(value + Fraction(1, 2))


def assert_manifest_compatible(path):
    """Không cho append V2 vào manifest đang chứa temporal V1."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    rows_by_id = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(LABEL_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest {path} thiếu cột bắt buộc: {sorted(missing)}"
            )
        for row in reader:
            clip_id = row.get("clip_id", "")
            if not clip_id or clip_id in rows_by_id:
                raise ValueError(
                    f"Manifest {path} có clip_id rỗng hoặc trùng: {clip_id!r}"
                )
            rows_by_id[clip_id] = row
            if row.get("method") != "temporal_desync":
                continue
            param = row.get("param", "")
            if any(key not in param for key in REQUIRED_PARAM_KEYS):
                raise ValueError(
                    "Từ chối trộn temporal V2 vào manifest có temporal V1/schema cũ: "
                    f"{path}. Hãy dùng manifest V2 mới, ví dụ {DEFAULT_LABELS}."
                )
            validate_timeline_contract(row, "temporal_desync")
    return rows_by_id


def ffprobe_media(path):
    """Probe FPS chính xác + số sample audio cần cho phép xoay vòng."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-count_packets", "-show_entries",
             "stream=codec_type,codec_name,avg_frame_rate,time_base,start_time,duration,"
             "duration_ts,sample_rate,channels,nb_read_packets:format=duration",
             "-of", "json", path],
            capture_output=True, text=True
        )
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout)
        streams = data.get("streams", [])
        video = next(s for s in streams if s.get("codec_type") == "video")
        audio = next(s for s in streams if s.get("codec_type") == "audio")

        fps = Fraction(video["avg_frame_rate"])
        sample_rate = int(audio["sample_rate"])
        if fps <= 0 or sample_rate <= 0:
            return None

        if audio.get("duration_ts") is not None and audio.get("time_base"):
            audio_samples = _round_fraction(
                Fraction(int(audio["duration_ts"]))
                * Fraction(audio["time_base"])
                * sample_rate
            )
            sample_count_source = "duration_ts"
        elif audio.get("duration"):
            audio_samples = _round_fraction(
                Fraction(audio["duration"]) * sample_rate
            )
            sample_count_source = "duration_fallback"
        else:
            return None

        if audio_samples <= 0:
            return None
        return {
            "fps": fps,
            "fps_text": str(video["avg_frame_rate"]),
            "video_duration": float(video.get("duration")
                                    or data.get("format", {}).get("duration", 0)),
            "video_duration_ts": (int(video["duration_ts"])
                                  if video.get("duration_ts") is not None else None),
            "video_time_base": video.get("time_base", ""),
            "video_packets": int(video.get("nb_read_packets", 0)),
            "audio_duration": audio_samples / sample_rate,
            "audio_samples": audio_samples,
            "sample_rate": sample_rate,
            "channels": int(audio.get("channels", 0)),
            "video_codec": video.get("codec_name", ""),
            "audio_codec": audio.get("codec_name", ""),
            "audio_start": float(audio.get("start_time", 0) or 0),
            "sample_count_source": sample_count_source,
        }
    except (ValueError, ZeroDivisionError, KeyError, StopIteration,
            json.JSONDecodeError):
        return None


def is_valid_output(path, source_media=None):
    """Kiểm tra file temporal đủ stream/duration trước khi resume."""
    media = ffprobe_media(path)
    if (media is None or media["audio_codec"] != "alac"
            or not media["video_codec"] or media["video_packets"] <= 0):
        return False
    if abs(media["audio_start"]) > 1e-3:
        return False
    if source_media is None:
        return True
    duration_ts_ok = True
    if (media["video_duration_ts"] is not None
            and source_media["video_duration_ts"] is not None):
        duration_ts_ok = (
            media["video_time_base"] == source_media["video_time_base"]
            and abs(media["video_duration_ts"]
                    - source_media["video_duration_ts"]) <= 1
        )
    return (media["fps"] == source_media["fps"]
            and media["video_codec"] == source_media["video_codec"]
            and media["video_packets"] == source_media["video_packets"]
            and duration_ts_ok
            and media["audio_samples"] == source_media["audio_samples"]
            and media["sample_rate"] == source_media["sample_rate"]
            and media["channels"] == source_media["channels"]
            and abs(media["video_duration"]
                    - source_media["video_duration"]) <= 0.05)


def make_desync(in_path, out_path, shift_frames, media=None):
    """
    Xoay vòng audio đúng số sample. shift>0: audio MUỘN hơn; shift<0: audio
    SỚM hơn. Video luôn copy nguyên. Trả (ok, metadata/error).

    Xoay vòng là boundary policy duy nhất dùng được chỉ từ một clip mà vẫn giữ
    đủ video, đủ duration và không bịa silence. Vùng wrap được trả về để V2a
    không dùng nó làm vùng alignment hợp lệ.
    """
    media = media or ffprobe_media(in_path)
    if media is None:
        return False, {"error": "ffprobe_media_failed"}
    if shift_frames == 0:
        return False, {"error": "zero_shift_not_fake"}

    signed_shift = Fraction(int(shift_frames), 1) / media["fps"]
    shift_samples = _round_fraction(abs(signed_shift) * media["sample_rate"])
    total_samples = media["audio_samples"]
    if shift_samples <= 0 or shift_samples >= total_samples:
        return False, {"error": "shift_outside_audio"}

    if shift_frames > 0:
        split = total_samples - shift_samples
        audio_filter = (
            "[0:a:0]asplit=2[a0][a1];"
            f"[a0]atrim=start_sample={split}:end_sample={total_samples},"
            "asetpts=PTS-STARTPTS[tail];"
            f"[a1]atrim=start_sample=0:end_sample={split},"
            "asetpts=PTS-STARTPTS[head];"
            "[tail][head]concat=n=2:v=0:a=1[outa]"
        )
        valid_start = float(signed_shift)
        valid_end = media["audio_duration"]
    else:
        audio_filter = (
            "[0:a:0]asplit=2[a0][a1];"
            f"[a0]atrim=start_sample={shift_samples}:end_sample={total_samples},"
            "asetpts=PTS-STARTPTS[rest];"
            f"[a1]atrim=start_sample=0:end_sample={shift_samples},"
            "asetpts=PTS-STARTPTS[prefix];"
            "[rest][prefix]concat=n=2:v=0:a=1[outa]"
        )
        valid_start = 0.0
        valid_end = media["audio_duration"] - float(abs(signed_shift))

    visual_valid = (
        (0.0, media["video_duration"] - float(signed_shift))
        if shift_frames > 0
        else (float(abs(signed_shift)), media["video_duration"])
    )
    common_start = max(valid_start, visual_valid[0])
    common_end = min(valid_end, visual_valid[1])
    if common_end <= common_start:
        return False, {"error": "shift_leaves_no_common_valid_range"}
    timeline = build_timeline_contract(
        media["audio_duration"], media["video_duration"],
        boundary=BOUNDARY_CIRCULAR_WRAP,
        audio_valid=(valid_start, valid_end),
        visual_valid=visual_valid,
        manipulation_scope="global",
        manipulation=(common_start, common_end),
    )

    partial_path = out_path + ".part.mp4"
    try:
        if os.path.exists(partial_path):
            os.remove(partial_path)
    except OSError:
        return False, {"error": "cannot_remove_partial"}

    cmd = ["ffmpeg", "-y", "-i", in_path,
           "-filter_complex", audio_filter,
           "-map", "0:v:0", "-map", "[outa]",
           "-c:v", "copy", "-c:a", "alac", "-movflags", "+faststart",
           "-loglevel", "error", partial_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = (proc.returncode == 0 and os.path.exists(partial_path)
          and os.path.getsize(partial_path) > 0
          and is_valid_output(partial_path, media))
    if ok:
        try:
            os.replace(partial_path, out_path)
        except OSError as exc:
            ok = False
            error = f"atomic_replace_failed:{exc}"
    else:
        error = (proc.stderr or "ffmpeg_or_contract_failed").strip()[-500:]
    if not ok:
        try:
            if os.path.exists(partial_path):
                os.remove(partial_path)
        except OSError:
            pass
        return False, {"error": error}
    return True, {
        "shift_sec": float(signed_shift),
        "shift_samples": shift_samples if shift_frames > 0 else -shift_samples,
        "audio_valid_start": valid_start,
        "audio_valid_end": valid_end,
        "visual_valid_start": visual_valid[0],
        "visual_valid_end": visual_valid[1],
        "boundary": BOUNDARY_CIRCULAR_WRAP,
        "timeline": timeline,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="data/02_curate/all_clean.csv",
                    help="CSV clip real (cần cột file_path) — mặc định tập sạch từ 04_curate")
    ap.add_argument("--path_col", default="file_path")
    ap.add_argument("--id_col", default="clip_id")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--labels", default=DEFAULT_LABELS,
                    help="manifest temporal V2 riêng; không được trỏ vào labels.csv V1")
    ap.add_argument("--mode", choices=["random", "all"], default="random",
                    help="random: 1 fake/clip lệch ngẫu nhiên; all: 1 fake cho mỗi mức 3/7/15")
    ap.add_argument("--shifts", default="3,7,15", help="danh sách frame lệch, vd '3,7,15'")
    ap.add_argument("--both_dirs", action="store_true",
                    help="mode all: sinh cả hai hướng cho mỗi mức (mode random luôn chọn ngẫu nhiên)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="chỉ xử lý N clip đầu (để test)")
    args = ap.parse_args()

    random.seed(args.seed)
    shifts = [int(s) for s in args.shifts.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    existing_rows = assert_manifest_compatible(args.labels)

    with open(args.input_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]
    print(f"Đọc {len(rows)} clip real từ {args.input_csv}")

    # mở labels (ghi tiếp nếu đã có, thêm header nếu mới)
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

        media = ffprobe_media(src_path)
        if media is None:
            skipped += 1
            continue

        if args.mode == "all":
            signed_choices = [sign * n for n in shifts
                              for sign in ((1, -1) if args.both_dirs else (1,))]
        else:
            n = random.choice(shifts)
            signed_choices = [random.choice((1, -1)) * n]
        for signed_frames in signed_choices:
            n = abs(signed_frames)
            t = Fraction(n, 1) / media["fps"]
            if float(t) >= media["audio_duration"]:
                skipped += 1
                continue
            sign = 1 if signed_frames > 0 else -1
            tag = f"desyncv2r2{'p' if sign > 0 else 'm'}{n}f"
            fake_id = f"{src_id}_{tag}"
            out_path = os.path.abspath(os.path.join(args.out_dir, fake_id + ".mp4"))

            existing = existing_rows.get(fake_id)
            if existing:
                recorded_path = os.path.abspath(existing.get("file_path", ""))
                if os.path.normcase(recorded_path) != os.path.normcase(out_path):
                    raise ValueError(
                        f"{fake_id} đã có trong manifest nhưng file_path khác: "
                        f"{recorded_path} != {out_path}"
                    )
                if is_valid_output(out_path, media):
                    resumed += 1
                    continue

            ok, details = make_desync(src_path, out_path, signed_frames, media)
            if ok:
                if existing:
                    repaired += 1
                else:
                    output_row = {
                        "clip_id": fake_id,
                        "file_path": out_path,
                        "label": 1,
                        "method": "temporal_desync",
                        "param": (
                            f"shift={signed_frames}f;"
                            f"generator={GENERATOR_VERSION}"
                        ),
                        "source_clip": src_id,
                        "source_video": r.get("source_video", ""),
                        "speaker_id": r.get("speaker_id", ""),
                        "tier": r.get("tier", ""),
                    }
                    output_row.update(details["timeline"])
                    writer.writerow(output_row)
                    made += 1
            else:
                print(f"  ! {fake_id}: {details.get('error', 'unknown_error')}")
                failed += 1

        if (i + 1) % 200 == 0:
            lf.flush()
            print(f"  {i+1}/{len(rows)}  made={made} resume={resumed} "
                  f"repair={repaired} skip={skipped} fail={failed}")

    lf.close()
    print(f"\nXong. Fake tạo được: {made} | resume: {resumed} | "
          f"sửa file thiếu: {repaired} | bỏ qua: {skipped} | lỗi: {failed}")
    print(f"  Video -> {args.out_dir}/  | nhãn (append) -> {args.labels}")
    print("Lưu ý: nén SNVSM 4 mức CRF ở bước sau sẽ đồng bộ codec cho cả real+fake.")


if __name__ == "__main__":
    main()
