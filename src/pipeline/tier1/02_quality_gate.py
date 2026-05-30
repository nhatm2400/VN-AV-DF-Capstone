import os
import cv2
import csv
import subprocess
import tempfile
import numpy as np
import librosa
from ultralytics import YOLO
import static_ffmpeg
static_ffmpeg.add_paths()

def get_project_root():
    """Trả về đường dẫn gốc của dự án (VN-AV-DF-Capstone)"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

def check_video_metadata(video_path, min_height=480, min_fps=24):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Không thể đọc video"
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if height < min_height:
        return False, f"Độ phân giải thấp: {height}p"
    if fps < min_fps:
        return False, f"FPS thấp: {fps:.2f}"
    return True, f"Pass (Height: {height}p, FPS: {fps:.2f})"

def check_audio_snr(video_path, min_snr=15):
    temp_audio_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
    try:
        cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_audio_path, '-y', '-loglevel', 'error']
        subprocess.run(cmd, check=True)
        y, sr = librosa.load(temp_audio_path, sr=16000)
        if len(y) == 0:
            return False, "File âm thanh rỗng"
        signal_power = np.mean(y**2)
        noise_power = np.percentile(y**2, 5)
        if noise_power == 0: noise_power = 1e-10
        snr = 10 * np.log10(signal_power / noise_power)
        if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
        if snr < min_snr: return False, f"SNR thấp: {snr:.2f} dB"
        return True, f"Pass (SNR: {snr:.2f} dB)"
    except subprocess.CalledProcessError:
        return False, "Không tìm thấy luồng âm thanh"
    except Exception as e:
        return False, f"Lỗi xử lý âm thanh: {str(e)}"

def check_face_presence(video_path, model, num_samples=5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False, "Không có khung hình"
    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    face_detected = False
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        results = model.predict(frame, verbose=False)
        if len(results[0].boxes) > 0:
            face_detected = True
            break
    cap.release()
    if face_detected: return True, "Pass (Tìm thấy khuôn mặt)"
    return False, "Không phát hiện khuôn mặt"

def process_tier_gate(tier_id, project_root, yolo_model):
    tier_name = f"tier{tier_id}"
    raw_dir = os.path.join(project_root, 'data', 'raw', tier_name)
    output_csv = os.path.join(project_root, 'data', f'{tier_name}_quality_gate_passed.csv')
    
    if not os.path.exists(raw_dir):
        print(f"Bỏ qua: Không tìm thấy thư mục {raw_dir}")
        return

    video_files = [f for f in os.listdir(raw_dir) if f.endswith(('.mp4', '.mkv', '.webm'))]
    if not video_files:
        print(f"Không có video nào trong {raw_dir}.")
        return

    print(f"\n--- Bắt đầu chạy Quality Gate cho {tier_name} ({len(video_files)} video) ---")
    passed_videos = []

    for idx, filename in enumerate(video_files):
        video_path = os.path.join(raw_dir, filename)
        print(f"[{idx+1}/{len(video_files)}] Đang kiểm tra: {filename}")
        
        meta_pass, meta_msg = check_video_metadata(video_path)
        if not meta_pass:
            print(f"  -> Loại bỏ: {meta_msg}"); continue
            
        audio_pass, audio_msg = check_audio_snr(video_path)
        if not audio_pass:
            print(f"  -> Loại bỏ: {audio_msg}"); continue
            
        face_pass, face_msg = check_face_presence(video_path, yolo_model)
        if not face_pass:
            print(f"  -> Loại bỏ: {face_msg}"); continue
            
        print("  -> PASSED")
        passed_videos.append({'filename': filename, 'metadata': meta_msg, 'audio_snr': audio_msg})

    if passed_videos:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'metadata', 'audio_snr'])
            writer.writeheader()
            writer.writerows(passed_videos)
        print(f"Hoàn tất {tier_name}! {len(passed_videos)}/{len(video_files)} video đạt tiêu chuẩn.")
        print(f"Lưu kết quả tại: {output_csv}")
    else:
        print(f"Hoàn tất {tier_name}! Không có video nào đạt tiêu chuẩn.")

def main():
    project_root = get_project_root()
    
    print("="*50)
    print("HỆ THỐNG QUALITY GATE VN-AV-DF")
    print("="*50)
    print("1. Kiểm duyệt Tier 1")
    print("2. Kiểm duyệt Tier 2")
    print("3. Kiểm duyệt Tier 3")
    print("0. Kiểm duyệt tất cả các Tier")
    print("="*50)
    
    while True:
        choice = input("Vui lòng nhập lựa chọn của bạn (0-3): ").strip()
        if choice in ['0', '1', '2', '3']: break
        print("Lựa chọn không hợp lệ, vui lòng nhập lại.")

    try:
        print("Đang khởi tạo mô hình YOLOv8-Face...")
        yolo_model = YOLO('../../model/yolov8n-face.pt')
    except Exception as e:
        print(f"Lỗi tải mô hình YOLO: {e}")
        return

    tiers_to_run = ['1', '2', '3'] if choice == '0' else [choice]
    
    for t in tiers_to_run:
        process_tier_gate(t, project_root, yolo_model)

if __name__ == "__main__":
    main()