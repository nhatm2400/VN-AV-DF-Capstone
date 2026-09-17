import os
import cv2
import csv
import subprocess

def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))

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

def check_audio_exists(video_path):
    # Quality gate chỉ kiểm tra video CÓ luồng âm thanh hay không.
    # SNR/chất lượng âm thanh được đánh giá per-clip ở bước 03_cut_clips.
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a',
               '-show_entries', 'stream=index', '-of', 'csv=p=0', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout.strip():
            return True, "Pass (Có luồng âm thanh)"
        return False, "Không tìm thấy luồng âm thanh"
    except Exception as e:
        return False, f"Lỗi kiểm tra âm thanh: {str(e)}"

def process_tier_gate(tier_id, project_root):
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

        audio_pass, audio_msg = check_audio_exists(video_path)
        if not audio_pass:
            print(f"  -> Loại bỏ: {audio_msg}"); continue

        print("  -> PASSED")
        passed_videos.append({'filename': filename, 'metadata': meta_msg, 'audio': audio_msg})

    if passed_videos:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'metadata', 'audio'])
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

    tiers_to_run = ['1', '2', '3'] if choice == '0' else [choice]

    for t in tiers_to_run:
        process_tier_gate(t, project_root)

if __name__ == "__main__":
    main()