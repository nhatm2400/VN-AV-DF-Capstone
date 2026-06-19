import os
import csv
import time
import yt_dlp
from pathlib import Path

def get_project_root():
    # file ở src/pipeline/01_collect/tier2/ -> repo root: lùi 4 cấp
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))

def retry_failed_downloads(csv_file_path, output_dir, failed_log_path='download_errors.log'):
    # 1. Đọc danh sách video đã tải
    downloaded_ids = set()
    for f in Path(output_dir).glob('*.mp4'):
        video_id = f.stem
        downloaded_ids.add(video_id)

    # 2. Đọc danh sách video cần thử lại từ file log lỗi
    failed_ids = set()
    if os.path.exists(failed_log_path):
        with open(failed_log_path, 'r') as f:
            for line in f:
                if line.strip():
                    failed_ids.add(line.strip())

    if not failed_ids:
        print("Không tìm thấy video lỗi trong file log.")
        return

    # 3. Đọc URL từ CSV và lọc những video trong failed_ids và chưa tải
    urls_to_retry = []
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_id = row['video_id']
            if video_id in failed_ids and video_id not in downloaded_ids:
                urls_to_retry.append(row['url'])

    if not urls_to_retry:
        print("Không có video lỗi nào cần thử lại.")
        return

    print(f"Sẽ thử lại {len(urls_to_retry)} video.")

    # 4. Cấu hình yt-dlp (giống với bản cập nhật)
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'cookiefile': 'cookies.txt',  # Đảm bảo file này đã được tạo
        'ignoreerrors': True,
        'quiet': False,
        'no_warnings': True,
        'merge_output_format': 'mp4',
        'sleep_interval': 30,
        'max_sleep_interval': 45,
        'retries': 10,
        'fragment_retries': 10,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(urls_to_retry)
        print(f"--- Hoàn thành thử lại cho thư mục {output_dir} ---\n")
    except Exception as e:
        print(f"Lỗi trong quá trình thử lại: {str(e)}")

if __name__ == "__main__":
    project_root = get_project_root()
    output_path = os.path.join(project_root, 'data', 'raw', 'tier2')
    csv_path = os.path.join(project_root, 'data', 'youtube_tier2_urls.csv')
    # File log sẽ được tạo tự động bởi yt-dlp khi có lỗi, bạn có thể điều chỉnh tên file cho phù hợp
    retry_failed_downloads(csv_path, output_path, failed_log_path='download_errors.log')