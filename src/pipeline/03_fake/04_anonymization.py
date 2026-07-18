"""
04_anonymization.py — Pseudo-fake: NHIỄU/BLUR KHUÔN MẶT (Anonymization)

Tuần 2 (Pseudo-fake Engineering) — kỹ thuật thứ 4 theo docs/Pipeline.

Ý tưởng: phát hiện khuôn mặt từng frame rồi làm méo vùng mặt — Gaussian blur
(kernel ≥51px) hoặc pixelate (ô vuông to). AUDIO GIỮ NGUYÊN (-c:a copy). Mục tiêu
mô phỏng các fake đời thực hay che/méo mặt: "CapCut noise" (méo/nhiễu khuôn mặt)
và nén H.264 mạnh làm mặt mờ-nhiễu để né detection (xem CLAUDE.md, mục Tạo fake).

Khẩu hình bị xóa/mờ -> nhánh khớp môi–tiếng mất tín hiệu visual đáng tin -> tín
hiệu deepfake kiểu "che dấu vết". Bổ sung kênh tấn công VISUAL-IDENTITY, khác với
02 (visual-motion) và 03 (audio-prosody).

Detector: dùng LẠI yolov8n-face.pt sẵn có ở root (ultralytics) — KHÔNG thêm phụ
thuộc MediaPipe như proposal gốc, để đồng bộ stack với 01_collect/03_quality_gate.

⚠️ CẢNH BÁO LEAKAGE (quan trọng): nếu blur CHỈ xuất hiện ở fake, model học tắt
"mờ mặt = fake" thay vì học bản chất. BẮT BUỘC ở bước train phải áp blur ĐỐI XỨNG
lên một phần REAL (augmentation, nhãn giữ real), hoặc trộn cùng SNVSM. Method này
chỉ SINH fake; phần augment đối xứng do dataloader/train chịu trách nhiệm.

Phụ thuộc: ultralytics, opencv-python, numpy (đã có trong requirements).
Cần ffmpeg trong PATH. yolov8n-face.pt đặt ở root (hoặc trỏ qua --face_model).

Ví dụ:
  python 04_anonymization.py --input_csv data/02_curate/all_clean.csv \\
      --out_dir data/fake --labels data/labels.csv
  python 04_anonymization.py --input_csv ... --mode pixelate --limit 5
  python 04_anonymization.py --input_csv ... --blur_kernel 71 --detect_every 3
"""

import os
import sys
import csv
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
    import cv2
    # pyrefly: ignore [missing-import]
    from ultralytics import YOLO
except Exception as e:
    print(f"Thiếu thư viện: {e}. Cài: pip install ultralytics opencv-python")
    sys.exit(1)

# Schema nhãn DÙNG CHUNG với 01/02/03.
LABEL_FIELDS = ["clip_id", "file_path", "label", "method", "param",
                "source_clip", "source_video", "speaker_id", "tier"]


def odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def blur_region(frame, x1, y1, x2, y2, mode, kernel):
    """Làm méo vùng [x1:x2, y1:y2] tại chỗ. kernel<=0 -> tự thích nghi theo cỡ mặt."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    bw, bh = x2 - x1, y2 - y1
    if mode == "pixelate":
        # thu nhỏ ~12px chiều dài rồi phóng lại bằng NEAREST -> ô vuông to
        small = cv2.resize(roi, (max(1, bw // 16), max(1, bh // 16)),
                           interpolation=cv2.INTER_LINEAR)
        frame[y1:y2, x1:x2] = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
    else:  # blur
        k = odd(kernel) if kernel and kernel > 0 else odd(max(51, max(bw, bh) // 2))
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def anonymize(model, in_path, tmp_video, mode, kernel, margin, detect_every, conf):
    """
    Đọc video, blur/pixelate mặt từng frame, ghi ra tmp_video (KHÔNG audio).
    Trả True nếu ghi được & có ít nhất 1 frame bắt được mặt.
    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        return False, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))
    if not out.isOpened():
        cap.release()
        return False, 0

    boxes = []            # tái dùng box giữa các lần detect (mặt di chuyển chậm)
    fi = face_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % max(1, detect_every) == 0:
            res = model.predict(frame, verbose=False, conf=conf)[0]
            boxes = []
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes.xyxy.cpu().numpy():
                    boxes.append(b[:4])
        if boxes:
            face_frames += 1
            for (x1, y1, x2, y2) in boxes:
                bw, bh = x2 - x1, y2 - y1
                mx, my = int(bw * margin), int(bh * margin)
                blur_region(frame, int(x1) - mx, int(y1) - my,
                            int(x2) + mx, int(y2) + my, mode, kernel)
        out.write(frame)
        fi += 1

    cap.release()
    out.release()
    ok = os.path.exists(tmp_video) and os.path.getsize(tmp_video) > 0
    return (ok and face_frames > 0), face_frames


def mux_audio(tmp_video, src_audio, out_path):
    """Ghép VIDEO đã blur + AUDIO gốc copy nguyên."""
    def run(acodec):
        cmd = ["ffmpeg", "-y", "-i", tmp_video, "-i", src_audio,
               "-map", "0:v", "-map", "1:a",
               "-c:v", "copy", "-c:a", acodec,
               "-shortest", "-loglevel", "error", out_path]
        subprocess.run(cmd, capture_output=True)
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0

    # copy audio nguyên; nếu codec lạ khiến copy fail thì re-encode aac
    return run("copy") or run("aac")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="data/02_curate/all_clean.csv",
                    help="CSV clip real (cần cột file_path) — mặc định tập sạch từ 04_curate")
    ap.add_argument("--path_col", default="file_path")
    ap.add_argument("--id_col", default="clip_id")
    ap.add_argument("--out_dir", default="data/03_fake")
    ap.add_argument("--labels", default="data/03_fake/labels.csv")
    ap.add_argument("--face_model", default="yolov8n-face.pt",
                    help="đường dẫn YOLO face (mặc định ở root khi chạy từ repo root)")
    ap.add_argument("--mode", choices=["blur", "pixelate"], default="blur")
    ap.add_argument("--blur_kernel", type=int, default=0,
                    help="kernel Gaussian (lẻ, ≥51). 0 = tự thích nghi theo cỡ mặt")
    ap.add_argument("--margin", type=float, default=0.15, help="nới box mặt theo tỉ lệ")
    ap.add_argument("--detect_every", type=int, default=3,
                    help="detect mỗi N frame, tái dùng box ở giữa (tăng tốc)")
    ap.add_argument("--conf", type=float, default=0.25, help="ngưỡng confidence YOLO")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="chỉ xử lý N clip đầu (để test)")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    if not os.path.isfile(args.face_model):
        print(f"Không tìm thấy face model: {args.face_model} (chạy từ repo root hoặc dùng --face_model)")
        sys.exit(1)
    model = YOLO(args.face_model)

    with open(args.input_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]
    print(f"Đọc {len(rows)} clip real từ {args.input_csv} | mode={args.mode}")

    new_file = not os.path.exists(args.labels) or os.path.getsize(args.labels) == 0
    os.makedirs(os.path.dirname(os.path.abspath(args.labels)) or ".", exist_ok=True)
    lf = open(args.labels, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(lf, fieldnames=LABEL_FIELDS)
    if new_file:
        writer.writeheader()

    made = skipped = failed = 0
    tmpdir = tempfile.mkdtemp(prefix="anon_")
    tmp_video = os.path.join(tmpdir, "v.mp4")
    for i, r in enumerate(rows):
        src_path = r.get(args.path_col, "")
        src_id = r.get(args.id_col, f"clip{i:06d}")
        if not src_path or not os.path.isfile(src_path):
            skipped += 1
            continue

        if os.path.exists(tmp_video):
            try:
                os.remove(tmp_video)
            except OSError:
                pass
        ok, face_frames = anonymize(model, src_path, tmp_video, args.mode,
                                    args.blur_kernel, args.margin,
                                    args.detect_every, args.conf)
        if not ok:
            skipped += 1               # không đọc được / không bắt được mặt nào
            continue

        fake_id = f"{src_id}_anon{args.mode[:3]}"
        out_path = os.path.abspath(os.path.join(args.out_dir, fake_id + ".mp4"))
        if mux_audio(tmp_video, src_path, out_path):
            writer.writerow({
                "clip_id": fake_id,
                "file_path": out_path,
                "label": 1,
                "method": "anonymization",
                "param": f"{args.mode}_k={args.blur_kernel or 'auto'}_faceframes={face_frames}",
                "source_clip": src_id,
                "source_video": r.get("source_video", ""),
                "speaker_id": r.get("speaker_id", ""),
                "tier": r.get("tier", ""),
            })
            made += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}  made={made} skip={skipped} fail={failed}")

    lf.close()
    try:
        if os.path.exists(tmp_video):
            os.remove(tmp_video)
        os.rmdir(tmpdir)
    except OSError:
        pass

    print(f"\nXong. Fake tạo được: {made} | bỏ qua: {skipped} | lỗi: {failed}")
    print(f"  Video -> {args.out_dir}/  | nhãn (append) -> {args.labels}")
    print("⚠️  Nhớ áp blur ĐỐI XỨNG lên một phần REAL ở bước train để tránh leakage 'mờ=fake'.")


if __name__ == "__main__":
    main()
