import os
import csv
import yt_dlp
from pathlib import Path
import static_ffmpeg
static_ffmpeg.add_paths()

def get_project_root():
    """Trả về đường dẫn gốc của dự án (VN-AV-DF-Capstone)"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))

def download_videos_for_tier(csv_file_path, output_dir):
    # Tạo thư mục nếu chưa tồn tại
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Đọc danh sách video đã tải thành công
    downloaded_ids = set()
    for f in Path(output_dir).glob('*.mp4'):
        # Lấy tên file (không bao gồm phần mở rộng)
        video_id = f.stem
        downloaded_ids.add(video_id)

    # 2. Đọc danh sách URL từ CSV và lọc
    urls_to_download = []
    total_urls = 0
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            total_urls += 1
            video_id = row['video_id']
            if video_id not in downloaded_ids:
                urls_to_download.append(row['url'])

    if not urls_to_download:
        print(f"Không có video mới để tải trong {csv_file_path}.")
        return

    print(f"Tìm thấy {len(urls_to_download)} video mới (đã bỏ qua {len(downloaded_ids)} video cũ).")

    # 3. Cấu hình yt-dlp (quan trọng: thay đổi cookiefile nếu cần)
    cookie_path = os.path.join(get_project_root(), 'cookies.txt')
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        #THAY ĐỔI ĐƯỜNG DẪN NÀY NẾU CẦN 
        'cookiefile': cookie_path,  # Sử dụng file cookies.txt đã xuất ở thư mục gốc
        'ignoreerrors': True,          # Bỏ qua lỗi của video và tiếp tục
        'quiet': False,
        'no_warnings': True,
        'merge_output_format': 'mp4',
        # Cấu hình JavaScript runtime và Remote Challenge Solver để bypass bot check / n challenge
        'js_runtimes': {'node': {}, 'deno': {}},
        'remote_components': ['ejs:github'],
        # Tăng thời gian chờ để tránh bị chặn IP
        'sleep_interval': 90,          # Nghỉ 90 giây giữa các video
        'max_sleep_interval': 45,      # Nghỉ ngẫu nhiên 30-45 giây
        'retries': 10,
        'fragment_retries': 10,
        'throttledratelimit': 1000000, # 1 MB/s tối thiểu
    }

    try:
        print(f"\n--- Bắt đầu tải {len(urls_to_download)} video vào {output_dir} ---")
        # Sử dụng with statement để tự động quản lý tài nguyên
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(urls_to_download)
        print(f"--- Hoàn thành tải cho thư mục {output_dir} ---\n")
    except Exception as e:
        print(f"Lỗi khi xử lý {csv_file_path}: {str(e)}")

def main():
    project_root = get_project_root()
    
    print("="*50)
    print("HỆ THỐNG TẢI DỮ LIỆU VN-AV-DF")
    print("="*50)
    print("1. Tải Tier 1 (YouTube CC-BY)")
    print("2. Tải Tier 2 (YouTube Standard)")
    print("3. Tải Tier 3 (TikTok)")
    print("0. Tải tất cả các Tier")
    print("="*50)
    
    while True:
        choice = input("Vui lòng nhập lựa chọn của bạn (0-3): ").strip()
        if choice in ['0', '1', '2', '3']:
            break
        print("Lựa chọn không hợp lệ, vui lòng nhập lại.")

    tiers_config = [
        {'id': '1', 'csv': 'youtube_tier1_urls.csv', 'out': os.path.join('raw', 'tier1')},
        {'id': '2', 'csv': 'youtube_tier2_urls.csv', 'out': os.path.join('raw', 'tier2')},
        {'id': '3', 'csv': 'tiktok_tier3_urls.csv', 'out': os.path.join('raw', 'tier3')}
    ]

    for tier in tiers_config:
        if choice == '0' or choice == tier['id']:
            csv_path = os.path.join(project_root, 'data', '01_collect', tier['csv'])
            output_path = os.path.join(project_root, 'data', tier['out'])

            if not os.path.exists(csv_path):
                print(f"Bỏ qua Tier {tier['id']}: Không tìm thấy file {csv_path}")
                continue

            download_videos_for_tier(csv_path, output_path)

if __name__ == "__main__":
    main()