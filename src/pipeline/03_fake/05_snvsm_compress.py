"""
05_snvsm_compress.py — SNVSM: đồng bộ codec real+fake bằng nén H.264 đa mức CRF

Stage cầu nối giữa 03_fake và 04_extract_features (chạy SAU khi sinh fake, TRƯỚC
khi trích feature). KHÔNG sinh fake — đây là bước NORMALIZE áp ĐỐI XỨNG cho cả
clip real lẫn fake.

Vấn đề nó giải quyết (chống leakage codec):
  - 02_frame_reverse re-encode video bằng libx264; 04_anonymization xuất mpeg4;
    01/03 và real thì giữ codec gốc (H.264 tải từ YouTube). => real và fake có
    "vân codec" khác nhau. Nếu để nguyên, model học "codec lạ = fake" thay vì học
    bản chất lệch pha/prosody.
  - Cách xử lý: đưa MỌI clip (real + mọi loại fake) qua CÙNG một pipeline libx264
    ở các mức CRF {23,30,35,40}, đồng thời encode audio AAC 128k, 16 kHz mono
    cho mọi nhãn.
    Sau bước này real và fake chia sẻ cùng codec + dải nén, không còn nhánh
    copy-audio riêng theo method. Việc này giảm shortcut codec nhưng không chứng
    minh đã xóa mọi dấu vết từ các lần transcode trước đó.

Cách dùng (chạy 2 lần — real và fake với cùng tham số):
  # REAL
  python src/pipeline/03_fake/05_snvsm_compress.py \
      --input_csv data/02_curate/all_clean.csv \
      --out_dir data/03_fake/snvsm_v2/real \
      --out_manifest data/03_fake/snvsm_v2/real_snvsm.csv
  # FAKE
  python src/pipeline/03_fake/05_snvsm_compress.py \
      --input_csv data/03_fake/manifests/v2/fake_all.csv \
      --out_dir data/03_fake/snvsm_v2/fake \
      --out_manifest data/03_fake/snvsm_v2/fake_snvsm.csv

Rồi trỏ 04/05 vào manifest SNVSM:
  python src/pipeline/05_build_labels/01_build_labels.py \
      --real_csv data/03_fake/snvsm_v2/real_snvsm.csv \
      --fake_labels data/03_fake/snvsm_v2/fake_snvsm.csv
  python src/pipeline/04_extract_features/01_extract_features.py \
      --real_csv data/03_fake/snvsm_v2/real_snvsm.csv \
      --fake_labels data/03_fake/snvsm_v2/fake_snvsm.csv

Manifest ra giữ mọi cột input; clip_id chứa version + hash cấu hình normalization
+ CRF, file_path trỏ clip nén, và manifest ghi encoder/preset/audio, CRF policy,
pair key cùng audio/video contract.
speaker_id/source_video/… copy nguyên nên split speaker-disjoint ở 05 vẫn đúng.
ID versioned ngăn --skip_existing tái dùng file V1 hoặc file V2 từ cấu hình khác.

--mode random (mặc định): 1 CRF ngẫu nhiên theo real nguồn; real + mọi fake ghép
cặp dùng cùng CRF -> ×1 dung lượng (khuyến nghị).
--mode all: đủ 4 mức/clip -> ×4 (augmentation tối đa, tốn đĩa/thời gian).

Chỉ dùng thư viện chuẩn + ffmpeg trong PATH.
"""

import os
import sys
import csv
import json
import hashlib
import random
import argparse
import subprocess
from fractions import Fraction

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pipeline.timeline_contract import (
    TIMELINE_FIELDS,
    TIMELINE_SCHEMA_VERSION,
    build_timeline_contract,
    validate_timeline_against_media,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CRFS_DEFAULT = [23, 30, 35, 40]
SNVSM_VERSION = "snvsm_v2_h264_aac16k_mono_exactdur"
DEFAULT_OUT_DIR = "data/03_fake/snvsm_v2"
LEGACY_SNVSM_DIR = "data/03_fake/snvsm"
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1


def normalization_config(encoder, preset, crfs=None, mode="random", seed=42):
    effective_preset = "p4" if encoder == "h264_nvenc" else preset
    crfs = list(CRFS_DEFAULT if crfs is None else crfs)
    config = {
        "normalization_version": SNVSM_VERSION,
        "video_encoder": encoder,
        "video_preset": effective_preset,
        "pixel_format": "yuv420p",
        "audio_codec": "aac",
        "audio_bitrate": "128k",
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
        "audio_channels": AUDIO_CHANNELS,
        "timeline_schema_version": TIMELINE_SCHEMA_VERSION,
        "crfs": crfs,
        "mode": mode,
        "seed": int(seed),
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config["config_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
    return config


def assert_v2_destination(out_dir, out_manifest, input_csv=None):
    """Không cho V2 ghi vào cây manifest/media SNVSM V1 bất biến."""
    legacy = os.path.normcase(os.path.abspath(LEGACY_SNVSM_DIR))
    for path in (out_dir, out_manifest):
        target = os.path.normcase(os.path.abspath(path))
        try:
            inside_legacy = os.path.commonpath([legacy, target]) == legacy
        except ValueError:
            inside_legacy = False
        if inside_legacy:
            raise ValueError(
                f"Từ chối ghi SNVSM V2 vào cây V1 bất biến: {path}. "
                f"Hãy dùng path dưới {DEFAULT_OUT_DIR}."
            )
    if (input_csv is not None
            and os.path.normcase(os.path.abspath(input_csv))
            == os.path.normcase(os.path.abspath(out_manifest))):
        raise ValueError(f"Từ chối ghi đè input manifest: {input_csv}")


def _round_fraction(value):
    return int(value + Fraction(1, 2))


def audio_target_samples(path):
    """Số sample 16 kHz theo timeline audio khai báo, bỏ decoder padding AAC."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=duration,duration_ts,time_base",
         "-of", "json", path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        audio = json.loads(proc.stdout)["streams"][0]
        if audio.get("duration_ts") is not None and audio.get("time_base"):
            samples = _round_fraction(
                Fraction(int(audio["duration_ts"]))
                * Fraction(audio["time_base"])
                * AUDIO_SAMPLE_RATE
            )
        else:
            samples = _round_fraction(
                Fraction(audio["duration"]) * AUDIO_SAMPLE_RATE
            )
        return samples if samples > 0 else None
    except (ValueError, KeyError, IndexError, ZeroDivisionError,
            json.JSONDecodeError):
        return None


def video_contract(path):
    """Frame/timeline contract that SNVSM must preserve from its input."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries",
         "stream=nb_read_frames,r_frame_rate,avg_frame_rate,duration,duration_ts,time_base,start_time",
         "-of", "json", path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        video = json.loads(proc.stdout)["streams"][0]
        frames = int(video["nb_read_frames"])
        fps = Fraction(video.get("r_frame_rate") or video["avg_frame_rate"])
        if video.get("duration_ts") is not None and video.get("time_base"):
            duration = float(Fraction(int(video["duration_ts"]))
                             * Fraction(video["time_base"]))
        else:
            duration = float(video["duration"])
        if frames <= 0 or fps <= 0 or duration <= 0:
            return None
        return {
            "frames": frames,
            "fps": str(fps),
            "duration": duration,
            "start": float(video.get("start_time", 0) or 0),
        }
    except (ValueError, KeyError, IndexError, ZeroDivisionError,
            json.JSONDecodeError):
        return None


def video_contract_matches(actual, expected):
    if actual is None or expected is None:
        return False
    return (actual["frames"] == expected["frames"]
            and Fraction(actual["fps"]) == Fraction(expected["fps"])
            and abs(actual["duration"] - expected["duration"]) <= 1e-3
            and abs(actual["start"]) <= 1e-3)


def decoded_audio_samples(path):
    """PCM samples exposed by the same 16 kHz mono decode used by Stage 04."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-map", "0:a:0",
         "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0 or len(proc.stdout) % 2:
        return None
    return len(proc.stdout) // 2


def is_valid_output(path, expected_samples=None, expected_video=None):
    """Validate media plus enough decoded PCM for Stage 04's manifest trim."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,start_time,sample_rate,channels,duration_ts,time_base:"
         "format=duration",
         "-of", "json", path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return False
    try:
        data = json.loads(proc.stdout)
        streams = data.get("streams", [])
        video = next(s for s in streams if s.get("codec_type") == "video")
        audio = next(s for s in streams if s.get("codec_type") == "audio")
        duration = float(data.get("format", {}).get("duration", 0))
        output_samples = _round_fraction(
            Fraction(int(audio["duration_ts"]))
            * Fraction(audio["time_base"])
            * AUDIO_SAMPLE_RATE
        )
        metadata_ok = (video.get("codec_name") == "h264"
                       and audio.get("codec_name") == "aac"
                       and int(audio.get("sample_rate", 0)) == AUDIO_SAMPLE_RATE
                       and int(audio.get("channels", 0)) == AUDIO_CHANNELS
                       and abs(float(audio.get("start_time", 0) or 0)) <= 1e-3
                       and duration > 0
                       and (expected_samples is None
                            or abs(output_samples - expected_samples) <= 1024))
        if not metadata_ok or expected_samples is None:
            return metadata_ok
        if expected_video is None:
            return False
        # AAC frames may expose trailing decoder padding. Stage 04 trims that tail
        # to expected_samples; reject files that would instead require silence pad.
        decoded_samples = decoded_audio_samples(path)
        return (decoded_samples is not None
                and decoded_samples >= expected_samples
                and video_contract_matches(video_contract(path), expected_video))
    except (ValueError, KeyError, StopIteration, ZeroDivisionError,
            json.JSONDecodeError):
        return False


def compress(in_path, out_path, crf, preset, encoder, target_samples=None,
             expected_video=None):
    """
    Re-encode video @CRF + audio AAC 128k/16 kHz/mono. CÙNG pipeline real+fake.
      libx264   : -crf N (CPU, chuẩn)
      h264_nvenc: -cq N  (GPU NVENC — nhanh, không chiếm CPU; CQ≈CRF)
    """
    if target_samples is None:
        target_samples = audio_target_samples(in_path)
    expected_video = expected_video or video_contract(in_path)
    if target_samples is None or target_samples <= 0 or expected_video is None:
        return False
    if encoder == "h264_nvenc":
        venc = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    else:
        venc = ["-c:v", "libx264", "-crf", str(crf), "-preset", preset]

    partial_path = out_path + ".part.mp4"
    try:
        if os.path.exists(partial_path):
            os.remove(partial_path)
    except OSError:
        return False

    audio_filter = (
        f"aresample={AUDIO_SAMPLE_RATE},"
        "aformat=sample_fmts=fltp:channel_layouts=mono,"
        f"apad=whole_len={target_samples},atrim=end_sample={target_samples},"
        "asetpts=PTS-STARTPTS"
    )
    cmd = ["ffmpeg", "-y", "-i", in_path,
           "-map", "0:v:0", "-map", "0:a:0?", *venc, "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-ar", str(AUDIO_SAMPLE_RATE),
           "-ac", str(AUDIO_CHANNELS), "-af", audio_filter,
           "-movflags", "+faststart", "-loglevel", "error", partial_path]
    proc = subprocess.run(cmd, capture_output=True)
    ok = (proc.returncode == 0
          and is_valid_output(partial_path, target_samples, expected_video))
    try:
        if ok:
            os.replace(partial_path, out_path)
        elif os.path.exists(partial_path):
            os.remove(partial_path)
    except OSError:
        ok = False
        try:
            if os.path.exists(partial_path):
                os.remove(partial_path)
        except OSError:
            pass
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True,
                    help="CSV clip (real all_clean.csv hoặc fake manifests/v2/fake_all.csv)")
    ap.add_argument("--path_col", default="file_path")
    ap.add_argument("--id_col", default="clip_id")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--out_manifest", required=True, help="CSV manifest cho 04/05 đọc tiếp")
    ap.add_argument("--crfs", default="23,30,35,40")
    ap.add_argument("--mode", choices=["random", "all"], default="random",
                    help="random: 1 CRF theo real nguồn (×1); all: đủ 4 mức/clip (×4)")
    ap.add_argument("--preset", default="veryfast", help="chỉ áp cho libx264")
    ap.add_argument("--encoder", choices=["libx264", "h264_nvenc"], default="libx264",
                    help="h264_nvenc: encode GPU (nhanh, không chiếm CPU)")
    ap.add_argument("--skip_existing", action="store_true", default=True,
                    help="bỏ encode clip đã có -> RESUME (manifest vẫn ghi đủ)")
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    crfs = [int(c) for c in args.crfs.split(",") if c.strip()]
    if not crfs or len(set(crfs)) != len(crfs):
        raise ValueError("--crfs phải có ít nhất một giá trị và không được trùng")
    if args.seed < 0:
        raise ValueError("--seed phải >= 0")
    assert_v2_destination(args.out_dir, args.out_manifest, args.input_csv)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.input_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        in_fields = reader.fieldnames or []
        rows = list(reader)
    if args.limit:
        rows = rows[:args.limit]
    print(f"Đọc {len(rows)} clip từ {args.input_csv} | mode={args.mode} | crfs={crfs}")

    config = normalization_config(args.encoder, args.preset, crfs,
                                  args.mode, args.seed)
    out_fields = list(in_fields)
    for extra in ("crf", "orig_clip_id", "snvsm_version", "snvsm_config_id",
                  "snvsm_encoder", "snvsm_preset", "snvsm_audio",
                  "snvsm_sample_rate", "snvsm_channels",
                  "snvsm_target_samples", "snvsm_mode",
                  "snvsm_crf_set", "snvsm_seed", "snvsm_pair_key",
                  "snvsm_video_frames", "snvsm_video_fps",
                  "snvsm_video_duration_s", *TIMELINE_FIELDS):
        if extra not in out_fields:
            out_fields.append(extra)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_manifest)) or ".", exist_ok=True)
    mf = open(args.out_manifest, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(mf, fieldnames=out_fields)
    writer.writeheader()

    made = resumed = skipped = failed = 0
    for i, r in enumerate(rows):
        src_path = r.get(args.path_col, "")
        src_id = r.get(args.id_col, f"clip{i:06d}")
        if not src_path or not os.path.isfile(src_path):
            skipped += 1
            continue
        target_samples = audio_target_samples(src_path)
        expected_video = video_contract(src_path)
        if target_samples is None or expected_video is None:
            skipped += 1
            continue

        audio_duration = target_samples / AUDIO_SAMPLE_RATE
        visual_duration = expected_video["duration"]
        method = str(r.get("method", "") or "").strip()
        try:
            if method in ("temporal_desync", "frame_reverse",
                          "pitch_flatten", "anonymization"):
                timeline = validate_timeline_against_media(
                    r, audio_duration, visual_duration, method
                )
            else:
                timeline = build_timeline_contract(
                    audio_duration, visual_duration,
                    manipulation_scope="none",
                )
                timeline = validate_timeline_against_media(
                    timeline, audio_duration, visual_duration, "real"
                )
        except ValueError as exc:
            print(f"  ! {src_id}: timeline_contract:{exc}")
            skipped += 1
            continue

        # Ghép CRF theo real nguồn để real + mọi fake cùng source có cùng mức nén.
        pair_key = (r.get("source_clip") or r.get("orig_clip_id") or src_id).strip()
        if not pair_key:
            skipped += 1
            continue
        if args.mode == "all":
            chosen = crfs
        else:
            chosen = [random.Random(f"{args.seed}:{pair_key}").choice(crfs)]
        for crf in chosen:
            new_id = f"{src_id}_snvsmv2_{config['config_id']}_crf{crf}"
            out_path = os.path.abspath(os.path.join(args.out_dir, new_id + ".mp4"))
            exists = (args.skip_existing
                      and is_valid_output(out_path, target_samples, expected_video))
            ok = exists or compress(src_path, out_path, crf, args.preset,
                                    args.encoder, target_samples, expected_video)
            if ok:
                row = dict(r)
                row[args.id_col] = new_id
                row[args.path_col] = out_path
                row["crf"] = crf
                row["orig_clip_id"] = src_id
                row["snvsm_version"] = SNVSM_VERSION
                row["snvsm_config_id"] = config["config_id"]
                row["snvsm_encoder"] = config["video_encoder"]
                row["snvsm_preset"] = config["video_preset"]
                row["snvsm_audio"] = "aac_128k_16khz_mono"
                row["snvsm_sample_rate"] = AUDIO_SAMPLE_RATE
                row["snvsm_channels"] = AUDIO_CHANNELS
                row["snvsm_target_samples"] = target_samples
                row["snvsm_mode"] = config["mode"]
                row["snvsm_crf_set"] = ",".join(str(value) for value in config["crfs"])
                row["snvsm_seed"] = config["seed"]
                row["snvsm_pair_key"] = pair_key
                row["snvsm_video_frames"] = expected_video["frames"]
                row["snvsm_video_fps"] = expected_video["fps"]
                row["snvsm_video_duration_s"] = f"{expected_video['duration']:.9f}"
                row.update(timeline)
                writer.writerow(row)               # manifest luôn ghi đủ (kể cả clip resume)
                if exists:
                    resumed += 1
                else:
                    made += 1
            else:
                failed += 1

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}  made={made} resume={resumed} skip={skipped} fail={failed}")
            mf.flush()

    mf.close()
    print(f"\nXong. Clip nén: {made} | resume(đã có): {resumed} | bỏ qua: {skipped} | lỗi: {failed}")
    print(f"  Video -> {args.out_dir}/  | manifest -> {args.out_manifest}")
    print("Nhớ chạy CẢ real LẪN fake với CÙNG --crfs/--preset để giảm shortcut codec.")
    if skipped or failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
