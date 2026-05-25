import csv, time, re
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
    # Miền Bắc
    "tin tức hôm nay -karaoke -nhạc -bé -game",
    "bản tin thời sự -karaoke -nhạc -bé -game",
    "phỏng vấn -karaoke -nhạc -bé -game",
    "bài giảng đại học -karaoke -nhạc -bé -game",
    "diễn thuyết tiếng việt -karaoke -nhạc -bé -game",

    # Miền Nam
    "vlog tiếng Việt -karaoke -nhạc -bé -game",
    "chia sẻ kinh nghiệm -karaoke -nhạc -bé -game",
    "review sản phẩm -karaoke -nhạc -bé -game",

    # Miền Trung 
    "phóng sự đài PTTH Nghệ An -karaoke -nhạc -bé",
    "tin tức thời sự Đà Nẵng -karaoke -nhạc -bé",
    "thời sự TRT Huế -karaoke -nhạc -bé",
    "talkshow miền Trung -karaoke -nhạc -bé"
]

BAD_KEYWORDS_PATTERN = r'\b(karaoke|beat|tone nam|tone nữ|nhạc|ca khúc|cover|remix|mashup|dân ca|chèo|cải lương|tân cổ|thiếu nhi|hoạt hình|bé|trẻ em|đồ chơi|kids|mầm non|game|liên quân|pubg|free fire|parody|hài chế|nhạc chế|lồng tiếng|tụng kinh|chú đại bi)\b'

def search_cc_videos(youtube, query, regex_filter, max_results=50):
    videos = []
    next_page = None

    while len(videos) < max_results:
        try:
            request = youtube.search().list(
                q=query,
                part="snippet",
                type="video",
                videoLicense="creativeCommon",
                relevanceLanguage="vi",
                videoDuration="medium",
                videoDefinition="high",
                maxResults=min(50, max_results - len(videos)),
                pageToken=next_page,
            )
            response = request.execute()

            for item in response.get("items", []):
                title = item["snippet"]["title"]
                
                if regex_filter.search(title):
                    continue
                    
                videos.append({
                    "video_id":     item["id"]["videoId"],
                    "url":          f"https://youtu.be/{item['id']['videoId']}",
                    "title":        title,
                    "channel":      item["snippet"]["channelTitle"],
                    "published_at": item["snippet"]["publishedAt"],
                    "query":        query,
                })

            next_page = response.get("nextPageToken")
            if not next_page:
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
    
    project_root = get_project_root()
    output_csv = os.path.join(project_root, 'data', 'youtube_tier1_urls.csv')

    for query in QUERIES:
        print(f"Tìm: '{query}'...")
        results = search_cc_videos(youtube, query, regex, max_results=50)

        new = [v for v in results if v["video_id"] not in seen_ids]
        seen_ids.update(v["video_id"] for v in new)
        all_videos.extend(new)
        print(f"  → Thêm {len(new)} video đạt chuẩn (Tổng cộng: {len(all_videos)})")
        time.sleep(1)

    if all_videos:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_videos[0].keys())
            writer.writeheader()
            writer.writerows(all_videos)
        print(f"\nTuyệt vời! Đã lọc sạch và lưu {len(all_videos)} video chất lượng cao vào -> {output_csv}")
    else:
        print("\nKhông tìm thấy video nào đạt yêu cầu.")

if __name__ == "__main__":
    main()