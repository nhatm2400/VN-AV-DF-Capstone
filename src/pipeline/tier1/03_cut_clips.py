import os
import csv
import subprocess
import cv2
from datetime import datetime

def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0: return frame_count / fps
    return 0

def cut_video_into_clips(video_path, output_dir, video_id, clip_duration=5, overlap=1):
    duration = get_video_duration(video_path)
    if duration < clip_duration:
        print(f"  -> Video quá ngắn ({duration:.1f}s), bỏ qua.")
        return []

    step = clip_duration - overlap
    start_time = 0
    clip_idx = 0
    clips_metadata = []

    while (start_time + clip_duration) <= duration:
        clip_filename = f"{video_id}_clip{clip_idx:04d}_t{int(start_time):05d}.mp4"
        clip_path = os.path.join(output_dir, clip_filename)

        cmd = [
            'ffmpeg', '-y', '-ss', str(start_time), '-i', video_path, 
            '-t', str(clip_duration), '-c:v', 'libx264', '-crf', '18', 
            '-preset', 'fast', '-c:a', 'aac', '-b:a', '128k', 
            '-loglevel', 'error', clip_path
        ]

        try:
            subprocess.run(cmd, check=True)
            clips_metadata.append({
                'clip_id': clip_filename.replace('.mp4', ''),
                'source_video': video_id,
                'start_time': start_time,
                'end_time': start_time + clip_duration,
                'file_path': clip_path
            })
            clip_idx += 1
        except subprocess.CalledProcessError as e:
            print(f"  -> Lỗi khi cắt clip tại {start_time}s: {e}")
            break

        start_time += step

    return clips_metadata

def process_tier_cut(tier_id, project_root):
    tier_name = f"tier{tier_id}"
    raw_dir = os.path.join(project_root, 'data', 'raw', tier_name)
    clips_dir = os.path.join(project_root, 'data', 'clips', tier_name)
    input_csv = os.path.join(project_root, 'data', f'{tier_name}_quality_gate_passed.csv')
    
    date_str = datetime.now().strftime("%Y%m%d")
    log_csv = os.path.join(project_root, 'data', f'{tier_name}_cut_log_{date_str}.csv')

    if not os.path.exists(input_csv):
        print(f"Bỏ qua: Không tìm thấy file {input_csv}. Chạy 02_quality_gate.py trước.")
        return

    if not os.path.exists(clips_dir):
        os.makedirs(clips_dir)

    passed_videos = []
    with open(input_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            passed_videos.append(row['filename'])

    if not passed_videos:
        print(f"Danh sách video đầu vào của {tier_name} trống.")
        return

    print(f"\n--- Bắt đầu cắt {len(passed_videos)} video của {tier_name} ---")
    all_clips_log = []

    for idx, filename in enumerate(passed_videos):
        video_path = os.path.join(raw_dir, filename)
        video_id = os.path.splitext(filename)[0]
        
        print(f"[{idx+1}/{len(passed_videos)}] Đang xử lý: {filename}")
        
        if not os.path.exists(video_path):
            print(f"  -> Không tìm thấy file vật lý {video_path}, bỏ qua.")
            continue

        clip_data = cut_video_into_clips(video_path, clips_dir, video_id)
        if clip_data:
            all_clips_log.extend(clip_data)
            print(f"  -> Hoàn thành: {len(clip_data)} clips.")

    if all_clips_log:
        with open(log_csv, mode='w', newline='', encoding='utf-8') as f:
            fieldnames = ['clip_id', 'source_video', 'start_time', 'end_time', 'file_path']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_clips_log)
            
        print(f"\nHoàn tất {tier_name}! Tổng số clips tạo ra: {len(all_clips_log)}")
        print(f"Metadata lưu tại: {log_csv}")
    else:
        print(f"\nKhông có clip nào được tạo ra cho {tier_name}.")

def main():
    project_root = get_project_root()
    
    print("="*50)
    print("HỆ THỐNG CẮT CLIP VN-AV-DF")
    print("="*50)
    print("1. Cắt clip Tier 1")
    print("2. Cắt clip Tier 2")
    print("3. Cắt clip Tier 3")
    print("0. Cắt clip tất cả các Tier")
    print("="*50)
    
    while True:
        choice = input("Vui lòng nhập lựa chọn của bạn (0-3): ").strip()
        if choice in ['0', '1', '2', '3']: break
        print("Lựa chọn không hợp lệ, vui lòng nhập lại.")

    tiers_to_run = ['1', '2', '3'] if choice == '0' else [choice]
    
    for t in tiers_to_run:
        process_tier_cut(t, project_root)

if __name__ == "__main__":
    main()