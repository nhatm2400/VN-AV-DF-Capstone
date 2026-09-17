import csv
import time
import re
import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise ValueError("Không tìm thấy YOUTUBE_API_KEY trong file .env")

QUERIES = [
    # Nhóm 1: Tin tức - Thời sự
    "tin tức hôm nay -karaoke -nhạc -bé -game",
    "bản tin thời sự -karaoke -nhạc -bé -game",
    "phỏng vấn -karaoke -nhạc -bé -game",
    "thời sự VTV -karaoke -nhạc -bé -game",
    "thời sự HTV -karaoke -nhạc -bé -game",
    "tin nóng Việt Nam -karaoke -nhạc -bé -game",
    # Nhóm 2: Giáo dục
    "bài giảng -karaoke -nhạc -bé -game",
    "học trực tuyến -karaoke -nhạc -bé -game",
    "kiến thức phổ thông -karaoke -nhạc -bé -game",
    "luyện thi -karaoke -nhạc -bé -game",
    # Nhóm 3: Kinh tế
    "kinh doanh -karaoke -nhạc -bé -game",
    "chứng khoán -karaoke -nhạc -bé -game",
    "khởi nghiệp -karaoke -nhạc -bé -game",
    "bất động sản -karaoke -nhạc -bé -game",
    # Nhóm 4: Sức khỏe
    "sức khỏe -karaoke -nhạc -bé -game",
    "dinh dưỡng -karaoke -nhạc -bé -game",
    "y học thường thức -karaoke -nhạc -bé -game",
    # Nhóm 5: Công nghệ - Khoa học
    "công nghệ -karaoke -nhạc -bé -game",
    "khoa học -karaoke -nhạc -bé -game",
    "AI -karaoke -nhạc -bé -game",
    "lập trình -karaoke -nhạc -bé -game",
    # Nhóm 6: Văn hóa - Du lịch - Ẩm thực
    "văn hóa Việt Nam -karaoke -nhạc -bé -game",
    "du lịch -karaoke -nhạc -bé -game",
    "ẩm thực -karaoke -nhạc -bé -game",
    "làng nghề truyền thống -karaoke -nhạc -bé -game",
    # Nhóm 7: Miền Trung (giữ nguyên)
    "phóng sự đài PTTH Nghệ An -karaoke -nhạc -bé",
    "tin tức thời sự Đà Nẵng -karaoke -nhạc -bé",
    "thời sự TRT Huế -karaoke -nhạc -bé",
    "talkshow miền Trung -karaoke -nhạc -bé",
]

BAD_KEYWORDS_PATTERN = r'\b(karaoke|beat|tone nam|tone nữ|nhạc|ca khúc|cover|remix|mashup|dân ca|chèo|cải lương|tân cổ|thiếu nhi|hoạt hình|bé|trẻ em|đồ chơi|kids|mầm non|game|liên quân|pubg|free fire|parody|hài chế|nhạc chế|lồng tiếng|tụng kinh|chú đại bi)\b'

def check_licenses(youtube, video_ids):
    """Trả về dict {video_id: license_type}"""
    if not video_ids:
        return {}
    # Mỗi lần gọi videos.list tối đa 50 ID
    licenses = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        try:
            request = youtube.videos().list(part="status", id=",".join(batch))
            response = request.execute()
            for item in response.get("items", []):
                licenses[item["id"]] = item["status"].get("license", "unknown")
        except Exception as e:
            print(f"Lỗi check license batch: {e}")
    return licenses

def search_standard_videos(youtube, query, regex_filter, max_results=100):
    videos = []
    next_page_token = None
    retrieved = 0

    while retrieved < max_results:
        try:
            search_res = youtube.search().list(
                q=query,
                part="snippet",
                type="video",
                relevanceLanguage="vi",
                videoDuration="medium",
                videoDefinition="high",
                maxResults=min(50, max_results - retrieved),
                pageToken=next_page_token,
            ).execute()

            video_ids = [item["id"]["videoId"] for item in search_res.get("items", [])]
            licenses = check_licenses(youtube, video_ids)

            for item in search_res.get("items", []):
                title = item["snippet"]["title"]
                video_id = item["id"]["videoId"]
                if regex_filter.search(title):
                    continue
                # Chỉ giữ video có license = "youtube" (Standard)
                if licenses.get(video_id) != "youtube":
                    continue

                videos.append({
                    "video_id": video_id,
                    "url": f"https://youtu.be/{video_id}",
                    "title": title,
                    "channel": item["snippet"]["channelTitle"],
                    "published_at": item["snippet"]["publishedAt"],
                    "query": query,
                })
                retrieved += 1
                if retrieved >= max_results:
                    break

            next_page_token = search_res.get("nextPageToken")
            if not next_page_token:
                break
            time.sleep(0.5)
        except Exception as e:
            print(f"Lỗi API: {e}")
            break
    return videos

def main():
    youtube = build("youtube", "v3", developerKey=API_KEY)
    all_videos = []
    seen_ids = set()
    regex = re.compile(BAD_KEYWORDS_PATTERN, re.IGNORECASE)

    for query in QUERIES:
        print(f"Đang tìm kiếm: '{query}' (max_results=100)...")
        results = search_standard_videos(youtube, query, regex, max_results=100)
        new = [v for v in results if v["video_id"] not in seen_ids]
        seen_ids.update(v["video_id"] for v in new)
        all_videos.extend(new)
        print(f"Thêm {len(new)} video mới (Tổng: {len(all_videos)})")
        time.sleep(1)

    # Lưu CSV nội bộ (KHÔNG PUBLIC)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    output_csv = os.path.join(project_root, 'data', 'youtube_tier2_urls.csv')
    if all_videos:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_videos[0].keys())
            writer.writeheader()
            writer.writerows(all_videos)
        print(f"\nĐã lưu {len(all_videos)} video Tier 2 (Standard License) vào:\n{output_csv}")
    else:
        print("\nKhông tìm thấy video nào.")

if __name__ == "__main__":
    main()
