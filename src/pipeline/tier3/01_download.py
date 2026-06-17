import os
import csv
import yt_dlp

def get_project_root():
    level_1 = os.path.dirname(os.path.abspath(__file__))
    level_2 = os.path.dirname(level_1)
    level_3 = os.path.dirname(level_2)
    root_dir = os.path.dirname(level_3)

    return root_dir

def download_videos_for_tier(csv_file_path, output_dir):
    """Đọc URL từ file CSV và tải video về thư mục đích bằng yt-dlp."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'ignoreerrors': True,
        'quiet': False,
        'no_warnings': True,
        'merge_output_format': 'mp4',
        'sleep_interval': 5,
        'max_sleep_interval': 15
    }

    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            urls = [row['url'] for row in reader if 'url' in row]
            
            if not urls:
                print(f"Không có URL nào trong {csv_file_path}.")
                return

            print(f"\n--- Bắt đầu tải {len(urls)} video vào {output_dir} ---")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download(urls)
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
            csv_path = os.path.join(project_root, 'data', tier['csv'])
            output_path = os.path.join(project_root, 'data', tier['out'])
            
            if not os.path.exists(csv_path):
                print(f"Bỏ qua Tier {tier['id']}: Không tìm thấy file {csv_path}")
                continue
                
            download_videos_for_tier(csv_path, output_path)

if __name__ == "__main__":
    main()