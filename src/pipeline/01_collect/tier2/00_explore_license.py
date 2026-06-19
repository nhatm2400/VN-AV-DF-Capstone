import csv, time, re, os
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

# Sử dụng một số query đầu tiên từ Tier 1 để kiểm tra
TEST_QUERIES = [
    "tin tức hôm nay -karaoke -nhạc -bé -game",
    "bản tin thời sự -karaoke -nhạc -bé -game",
    "phỏng vấn -karaoke -nhạc -bé -game",
]

BAD_KEYWORDS_PATTERN = r'\b(karaoke|beat|tone nam|tone nữ|nhạc|ca khúc|cover|remix|mashup|dân ca|chèo|cải lương|tân cổ|thiếu nhi|hoạt hình|bé|trẻ em|đồ chơi|kids|mầm non|game|liên quân|pubg|free fire|parody|hài chế|nhạc chế|lồng tiếng|tụng kinh|chú đại bi)\b'
regex = re.compile(BAD_KEYWORDS_PATTERN, re.IGNORECASE)

def check_license(video_ids):
    """Lấy thông tin license cho danh sách video_ids (tối đa 50 ID mỗi lần gọi)"""
    if not video_ids:
        return {}
    try:
        request = youtube.videos().list(
            part="status",
            id=",".join(video_ids)
        )
        response = request.execute()
        licenses = {}
        for item in response.get("items", []):
            # Giá trị license có thể là 'youtube' (standard) hoặc 'creativeCommon'[reference:2]
            licenses[item["id"]] = item["status"].get("license", "unknown")
        return licenses
    except Exception as e:
        print(f"Lỗi khi kiểm tra license: {e}")
        return {}

def search_and_check_license(query, max_results=50):
    print(f"\n=== Kiểm tra query: '{query}' ===")
    videos = []
    next_page_token = None
    quota_used_search = 0
    quota_used_license = 0

    while len(videos) < max_results:
        try:
            # Search.list: 100 quota mỗi lần gọi[reference:3]
            search_response = youtube.search().list(
                q=query,
                part="snippet",
                type="video",
                relevanceLanguage="vi",
                videoDuration="medium",
                videoDefinition="high",
                maxResults=min(50, max_results - len(videos)),
                pageToken=next_page_token,
            ).execute()
            quota_used_search += 100

            video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
            # Videos.list: 1 quota cho tối đa 50 video[reference:4]
            licenses = check_license(video_ids)
            quota_used_license += 1

            for item in search_response.get("items", []):
                title = item["snippet"]["title"]
                video_id = item["id"]["videoId"]
                if regex.search(title):
                    continue
                license_type = licenses.get(video_id, "unknown")
                videos.append({
                    "video_id": video_id,
                    "title": title,
                    "channel": item["snippet"]["channelTitle"],
                    "license": license_type,
                })
            next_page_token = search_response.get("nextPageToken")
            if not next_page_token:
                break
            time.sleep(1)
        except Exception as e:
            print(f"Lỗi: {e}")
            break

    # Thống kê kết quả
    standard_count = sum(1 for v in videos if v['license'] == 'youtube')
    cc_count = sum(1 for v in videos if v['license'] == 'creativeCommon')
    unknown_count = len(videos) - standard_count - cc_count
    print(f"\n📊 Kết quả cho query '{query}':")
    print(f"  - Tổng video đã lọc (sau BAD_KEYWORDS): {len(videos)}")
    print(f"    ✅ Standard License: {standard_count} ({standard_count/len(videos)*100:.1f}%)")
    print(f"    🔓 CC License: {cc_count} ({cc_count/len(videos)*100:.1f}%)")
    print(f"    ❓ Unknown: {unknown_count}")
    print(f"💰 Quota sử dụng ước tính: {quota_used_search + quota_used_license} units")
    return videos

def main():
    all_results = {}
    for query in TEST_QUERIES:
        all_results[query] = search_and_check_license(query)
    print("\n🎉 Hoàn tất kiểm tra!")

if __name__ == "__main__":
    main()