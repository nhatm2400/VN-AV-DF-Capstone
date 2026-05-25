Mục tiêu: Xây dựng pipeline thu thập YouTube Tier 2 (Standard License) – không được phép distribute raw video, chỉ cung cấp danh sách YouTube ID + script tải (theo precedent FF++/Celeb-DF).
Dữ liệu Tier 2 cần đạt 220–250 video (~2.000 clips) sau quality gate và cắt 

Phân tích: 
Tier 1 dùng videoLicense="creativeCommon" → Tier 2 không filter theo license đó.
Tier 1 có thể xuất bản CSV chứa URL trực tiếp → Tier 2 chỉ được phép release ID + download script.
Các bước 01_download.py, 02_quality_gate.py, 03_cut_clips.py có thể tận dụng logic, nhưng cần điều chỉnh:
01_download.py phải chạy ở phía người dùng cuối, không nằm trong code release chính thức?
→ Thực tế: vẫn cần script để chính mình tải video về xử lý nội bộ, nhưng khi release dataset thì chỉ gửi ID + script. => thêm một script generate_download_script.py để tạo file .sh hoặc .py mà users chạy để tự tải video từ danh sách ID.
Quality gate và cut clips vẫn áp dụng như nhau cho video đã tải.

Đã làm:
Thẩm định khả năng thu thập YouTube cho Tier 2 
Kiểm tra xem YouTube Data API có cho phép tìm kiếm video “Standard” (không CC) hay không.
Chạy một script kiểm tra nhỏ để xác nhận và đo tỷ lệ CC trong kết quả tìm kiếm với các từ khóa mong muốn. (tier2.py, kết quả:
1. Xác nhận API Key
https://www.googleapis.com/youtube/v3/videos?part=snippet&chart=mostPopular&regionCode=VN&key=AIzaSyDF_4l6 đã trả về json 
{
  "kind": "youtube#videoListResponse",
  "etag": "jNsYgCX8GyRhi19as4bs_ntnrw4",
  "items": [
    {
      "kind": "youtube#video",
      "etag": "ir6PEu1yzZyMwg1MHrNZSQV6NCg",
      "id": "Kxr0kflgdtk",
...
Xác định hạn mức quota (quota limit) và Kiểm tra các API đã được bật: YouTube Data API Key được cung cấp sẵn từ code base nên các xác định này là đảm bảo, tôi không tự kiểm tra được, nên hãy kế thừa từ tài nguyên cũ
2. Xây dựng kế hoạch kiểm tra với Script đánh giá
kết quả chạy script
PS L:\FPT\Side project\Capstone\resources\VN-AV-DF-Capstone\src\pre-testing> python tier2.py
=== Kiểm tra query: 'tin tức hôm nay -karaoke -nhạc -bé -game' ===
📊 Kết quả cho query 'tin tức hôm nay -karaoke -nhạc -bé -game':
  - Tổng video đã lọc (sau BAD_KEYWORDS): 50
    ✅ Standard License: 50 (100.0%)
    🔓 CC License: 0 (0.0%)
    ❓ Unknown: 0
💰 Quota sử dụng ước tính: 101 units
=== Kiểm tra query: 'bản tin thời sự -karaoke -nhạc -bé -game' ===
📊 Kết quả cho query 'bản tin thời sự -karaoke -nhạc -bé -game':
  - Tổng video đã lọc (sau BAD_KEYWORDS): 50
    ✅ Standard License: 50 (100.0%)
    🔓 CC License: 0 (0.0%)
    ❓ Unknown: 0
💰 Quota sử dụng ước tính: 101 units
=== Kiểm tra query: 'phỏng vấn -karaoke -nhạc -bé -game' ===
📊 Kết quả cho query 'phỏng vấn -karaoke -nhạc -bé -game':
  - Tổng video đã lọc (sau BAD_KEYWORDS): 50
    ✅ Standard License: 49 (98.0%)
    🔓 CC License: 1 (2.0%)
    ❓ Unknown: 0
💰 Quota sử dụng ước tính: 101 units
🎉 Hoàn tất kiểm tra!)
Tận dụng code hiện có:
00_fetch_youtube_urls.py:
Bỏ videoLicense="creativeCommon".
Thêm bước kiểm tra license của từng video (gọi API videos.list với part status). Chỉ giữ video có license = "youtube" (standard).
Output sẽ là CSV chứa video_id, url, title, channel, published_at, query – csv này chỉ dùng nội bộ để tải, không release.
01_download.py: giữ nguyên (chỉ tải từ csv nội bộ).
02_quality_gate.py & 03_cut_clips.py: không thay đổi logic.
Điều chỉnh code hoàn chỉnh cho Tier 2
Viết 00_fetch_youtube_tier2_urls.py với cơ chế lọc standard license.
Viết script generate_download_script.py để tạo file .sh hoặc .py mà users chạy để tự tải video từ danh sách ID.
Đảm bảo các bước chạy nội bộ của bạn (fetch → download → quality gate → cut clips) hoạt động mượt mà.
Chạy toàn bộ luồng (chi tiết từng lệnh, file cấu hình cần có).
Đánh giá kết quả – dự kiến số lượng video thực tế, các vấn đề về license, tỷ lệ trùng lặp, v.v.


