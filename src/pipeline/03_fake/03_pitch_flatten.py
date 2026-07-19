"""
03_pitch_flatten.py — Pseudo-fake: LÀM PHẲNG F0 (Pitch Flatten) — đặc thù tiếng Việt

Tuần 2 (Pseudo-fake Engineering) — kỹ thuật thứ 3 theo docs/Pipeline.

Ý tưởng: tiếng Việt là ngôn ngữ THANH ĐIỆU; nghĩa của từ nằm ở đường F0 (cao độ).
Ta resynthesize audio sao cho F0 bị KÉO PHẲNG về một giá trị không đổi (mean/median),
xóa biến thiên cao độ -> các thanh (đặc biệt HỎI/NGÃ vốn có quỹ đạo F0 phức tạp)
bị phá. VIDEO GIỮ NGUYÊN (-c:v copy) -> đây là loại fake AUDIO THUẦN, đối ngẫu với
02_frame_reverse (visual thuần).

Vì sao quan trọng cho PAMF: nhánh khớp môi–tiếng có thể KHÔNG bắt được kiểu này
(khẩu hình vẫn khớp nội dung/timing, chỉ cao độ sai). Pitch Flatten là fake nhắm
thẳng vào NHÁNH PROSODY/F0 của model -> buộc kiến trúc phải có nhánh F0, không chỉ
dựa lip-sync. Một âm tấn công mà "mắt" không thấy nhưng "tai" (prosody) phải thấy.

Lưu ý về phạm vi: lý tưởng là chỉ làm phẳng F0 trên các đoạn thanh HỎI/NGÃ (cần
forced-alignment để định vị âm tiết) — ngoài phạm vi script này. Ở đây làm phẳng
TOÀN PHÁT NGÔN (monotone), vẫn là đòn prosody mạnh và đặc thù tiếng Việt; chỉ làm
phẳng vùng HỮU THANH (PSOLA chỉ đụng voiced), vùng vô thanh giữ nguyên.

Chống học-tủ:
  - VIDEO copy nguyên -> khác biệt nằm ở AUDIO, không phải hình.
  - Audio xuất 16kHz mono — trùng đúng tiền xử lý của wav2vec2 (model resample cả
    real lẫn fake về 16k mono) -> không tạo "vân" sample-rate phân biệt real/fake.

Phụ thuộc: praat-parselmouth  ->  pip install praat-parselmouth
Cần ffmpeg/ffprobe trong PATH.

Ví dụ:
  python 03_pitch_flatten.py --input_csv data/02_curate/all_clean.csv \\
      --out_dir data/fake --labels data/labels.csv
  python 03_pitch_flatten.py --input_csv ... --target median --limit 5
"""

import os
import sys
import csv
import math
import random
import tempfile
import argparse
import subprocess

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    # pyrefly: ignore [missing-import]
    import parselmouth
    # pyrefly: ignore [missing-import]
    from parselmouth.praat import call
except Exception:
    print("Thiếu praat-parselmouth. Cài: pip install praat-parselmouth")
    sys.exit(1)

# Schema nhãn DÙNG CHUNG với 01/02/04.
LABEL_FIELDS = ["clip_id", "file_path", "label", "method", "param",
                "source_clip", "source_video", "speaker_id", "tier"]

# Dải F0 tiếng Việt: nam ~75–200, nữ ~150–400, kèm excursion thanh hỏi/ngã -> 75–500.
F0_FLOOR = 75.0
F0_CEIL = 500.0


def extract_wav(in_path, wav_path):
    """Tách audio -> 16kHz mono WAV (khớp tiền xử lý wav2vec2)."""
    cmd = ["ffmpeg", "-y", "-i", in_path, "-vn", "-ac", "1", "-ar", "16000",
           "-loglevel", "error", wav_path]
    subprocess.run(cmd, capture_output=True)
    return os.path.exists(wav_path) and os.path.getsize(wav_path) > 0


def flatten_f0(wav_in, wav_out, target="mean"):
    """
    Làm phẳng F0 về một giá trị không đổi qua PSOLA (Praat).
    Trả giá trị F0 đích (Hz) nếu thành công, None nếu không đo được cao độ.
    """
    snd = parselmouth.Sound(wav_in)
    pitch = snd.to_pitch(pitch_floor=F0_FLOOR, pitch_ceiling=F0_CEIL)
    if target == "median":
        f0 = call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")
    else:
        f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    if f0 is None or math.isnan(f0) or f0 <= 0:
        return None                       # clip vô thanh / không đo được F0 -> bỏ

    manip = call(snd, "To Manipulation", 0.01, F0_FLOOR, F0_CEIL)
    ptier = call(manip, "Extract pitch tier")
    call(ptier, "Remove points between", 0, snd.xmax)
    # 1 điểm duy nhất -> PitchTier coi là hằng số trên toàn miền -> F0 phẳng tuyệt đối
    call(ptier, "Add point", snd.xmax / 2.0, f0)
    call([ptier, manip], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    out.save(wav_out, "WAV")
    return f0 if os.path.exists(wav_out) else None


def mux(in_video, flat_wav, out_path):
    """Ghép VIDEO copy nguyên + AUDIO đã làm phẳng F0."""
    cmd = ["ffmpeg", "-y", "-i", in_video, "-i", flat_wav,
           "-map", "0:v", "-map", "1:a",
           "-c:v", "copy", "-c:a", "aac",
           "-shortest", "-loglevel", "error", out_path]
    subprocess.run(cmd, capture_output=True)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="data/02_curate/all_clean.csv",
                    help="CSV clip real (cần cột file_path) — mặc định tập sạch từ 04_curate")
    ap.add_argument("--path_col", default="file_path")
    ap.add_argument("--id_col", default="clip_id")
    ap.add_argument("--out_dir", default="data/03_fake")
    ap.add_argument("--labels", default="data/03_fake/labels.csv")
    ap.add_argument("--target", choices=["mean", "median"], default="mean",
                    help="làm phẳng F0 về trung bình hay trung vị của phát ngôn")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="chỉ xử lý N clip đầu (để test)")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

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

    made = skipped = failed = 0
    tmpdir = tempfile.mkdtemp(prefix="pitchflat_")
    for i, r in enumerate(rows):
        src_path = r.get(args.path_col, "")
        src_id = r.get(args.id_col, f"clip{i:06d}")
        if not src_path or not os.path.isfile(src_path):
            skipped += 1
            continue

        wav_in = os.path.join(tmpdir, "in.wav")
        wav_out = os.path.join(tmpdir, "out.wav")
        if not extract_wav(src_path, wav_in):
            skipped += 1
            continue

        try:
            f0 = flatten_f0(wav_in, wav_out, target=args.target)
        except Exception as e:
            f0 = None
            print(f"  ! parselmouth lỗi ở {src_id}: {e}")
        if f0 is None:
            skipped += 1
            continue

        fake_id = f"{src_id}_flat{int(round(f0))}hz"
        out_path = os.path.abspath(os.path.join(args.out_dir, fake_id + ".mp4"))
        if mux(src_path, wav_out, out_path):
            writer.writerow({
                "clip_id": fake_id,
                "file_path": out_path,
                "label": 1,
                "method": "pitch_flatten",
                "param": f"flatten_{args.target}_F0={f0:.0f}Hz",
                "source_clip": src_id,
                "source_video": r.get("source_video", ""),
                "speaker_id": r.get("speaker_id", ""),
                "tier": r.get("tier", ""),
            })
            made += 1
        else:
            failed += 1

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}  made={made} skip={skipped} fail={failed}")

    lf.close()
    for f in ("in.wav", "out.wav"):
        try:
            os.remove(os.path.join(tmpdir, f))
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    print(f"\nXong. Fake tạo được: {made} | bỏ qua: {skipped} | lỗi: {failed}")
    print(f"  Video -> {args.out_dir}/  | nhãn (append) -> {args.labels}")
    print("Lưu ý: đây là fake AUDIO thuần — cần model có nhánh prosody/F0 mới phát hiện được.")


if __name__ == "__main__":
    main()
