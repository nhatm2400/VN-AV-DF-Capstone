import csv
import time
import os
import yt_dlp

def get_project_root():
    level_1 = os.path.dirname(os.path.abspath(__file__))
    level_2 = os.path.dirname(level_1)
    level_3 = os.path.dirname(level_2)
    root_dir = os.path.dirname(level_3)

    return root_dir

# Danh sách các kênh TikTok mục tiêu
TARGET_USERS = [
    "vtv24news",
    "galaxyplayofficial",
    "saigonteu4",
    "vtv3official",
    "bundau_chamchao",
    "lemkcr",
    "khanhbnq",
    "swan_meii"
]

def fetch_tiktok_videos(username, max_results=30):
    """Sử dụng yt-dlp để lấy danh sách URL video từ một kênh TikTok mà không tải video ngay"""
    url = f"https://www.tiktok.com/@{username}"
    
    ydl_opts = {
        'extract_flat': True,  # Chỉ trích xuất metadata, không tải video
        'quiet': True,
        'playlistend': max_results,
        # Nếu TikTok chặn, bỏ comment dòng dưới và dùng trình duyệt bạn đang đăng nhập TikTok
        # 'cookiesfrombrowser': ('chrome',), 
    }
    
    videos = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            if 'entries' in info:
                for entry in info['entries']:
                    if not entry.get('url'):
                        continue
                    
                    videos.append({
                        "video_id": entry.get('id', ''),
                        "url": entry.get('url', ''),
                        "title": entry.get('title', 'No Title').replace('\n', ' '),
                        "channel": username,
                        "published_at": "", # TikTok qua yt-dlp extract_flat thường không trả về ngày
                        "query": f"user:{username}"
                    })
        except Exception as e:
            print(f"Lỗi khi cào dữ liệu kênh @{username}: {e}")
            
    return videos

def main():
    project_root = get_project_root()
    output_csv = os.path.join(project_root, 'data', 'tiktok_tier3_urls.csv')
    
    # Đảm bảo thư mục data tồn tại
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    all_videos = []
    seen_urls = set()

    print("="*50)
    print("BẮT ĐẦU CÀO URL TIKTOK (TIER 3)")
    print("="*50)

    for user in TARGET_USERS:
        print(f"Đang quét kênh: @{user}...")
        results = fetch_tiktok_videos(user, max_results=200) # Điều chỉnh số lượng video mỗi kênh
        
        new_videos = [v for v in results if v["url"] not in seen_urls]
        seen_urls.update(v["url"] for v in new_videos)
        all_videos.extend(new_videos)
        
        print(f"  → Đã lấy {len(new_videos)} URL (Tổng: {len(all_videos)})")
        time.sleep(2) # Nghỉ 2s để tránh bị TikTok block rate limit

    if all_videos:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "url", "title", "channel", "published_at", "query"])
            writer.writeheader()
            writer.writerows(all_videos)
        print(f"\nTuyệt vời! Đã lưu {len(all_videos)} URL video TikTok vào -> {output_csv}")
    else:
        print("\nKhông tìm thấy video nào.")

if __name__ == "__main__":
    main()