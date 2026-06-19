# Tạo file download_tier2.py để người dùng cuối tự tải video
import os
import csv

def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))

def generate_download_script():
    project_root = get_project_root()
    csv_path = os.path.join(project_root, 'data', 'youtube_tier2_urls.csv')
    if not os.path.exists(csv_path):
        print("Chưa tìm thấy youtube_tier2_urls.csv. Hãy chạy 01_fetch_youtube_urls.py trước.")
        return

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        urls = [row['url'] for row in reader if 'url' in row]

    script_content = f'''#!/usr/bin/env python3
# Auto-generated download script for YouTube Tier 2 (Standard License)
# Số lượng video: {len(urls)}
# Hướng dẫn: python download_tier2.py
# 
# Script này tự động thêm delay giữa các video để tránh rate limit của YouTube.
# Nếu vẫn gặp lỗi rate limit, hãy tăng sleep_interval lên 10 hoặc 15 giây.

import yt_dlp
import os
import random
import time

# Tạo thư mục đích
output_dir = os.path.join("data", "raw", "tier2")
os.makedirs(output_dir, exist_ok=True)

# Cấu hình yt-dlp với delay và retry
ydl_opts = {{
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
    'ignoreerrors': True,
    'quiet': False,
    'no_warnings': True,
    'merge_output_format': 'mp4',
    'cookiesfrombrowser': ('chrome',),
    'sleep_interval': 15,
    'max_sleep_interval': 30,
    'retries': 10,
    'fragment_retries': 10,
}}

urls = {urls}

print(f"Bắt đầu tải {{len(urls)}} video vào {{output_dir}}...")
print("Mỗi video sẽ có khoảng nghỉ 5-10 giây để tránh bị YouTube chặn.")
print("Quá trình này có thể mất nhiều giờ. Kiên nhẫn nhé!\\n")

success = 0
failed = 0

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for idx, url in enumerate(urls, 1):
        print(f"[{{idx}}/{{len(urls)}}] Đang tải: {{url}}")
        try:
            ydl.download([url])
            success += 1
        except Exception as e:
            print(f"Lỗi không thể tải {{url}}: {{e}}")
            failed += 1
        # Delay thêm 1-2 giây sau mỗi video (ngoài sleep_interval của yt-dlp)
        if idx < len(urls):
            extra_sleep = random.uniform(1, 3)
            print(f"Nghỉ {{extra_sleep:.1f}} giây trước video tiếp theo...")
            time.sleep(extra_sleep)

print(f"\\nHoàn thành tải. Thành công: {{success}}, Thất bại: {{failed}}")
if failed > 0:
    print("Một số video không tải được do lỗi vĩnh viễn (region block, private, ...).")
'''
    script_path = os.path.join(project_root, 'download_tier2.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    print(f"Đã tạo script tải video tại: {script_path}")
    print(f"User có thể chạy: python download_tier2.py")

if __name__ == "__main__":
    generate_download_script()