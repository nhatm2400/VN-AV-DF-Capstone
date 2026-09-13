# Cấu hình và mẫu đầu vào

- Cấu hình dùng hằng ngày nằm ở `src/data/preparation/settings.py`; mở file chạy có số để Run trong IDE.
- `cut_dataset.example.json`: mẫu nếu chạy cutter trực tiếp bằng CLI. Pipeline bước 03 tự đọc số nguồn/đường dẫn từ batch tải nên không cần điền file JSON này. Giữ trường tier như nhãn nhóm nguồn, không suy giấy phép từ trường này.
- `templates/videos.csv`: danh sách video tự tuyển chọn, không tự tìm ngẫu nhiên theo keyword.
- `templates/rights.csv`: hồ sơ theo video, tùy chọn; downloader chỉ kiểm tra trạng thái khi chủ động truyền --rights.
- `templates/reviewed_clips.csv`: trường tối thiểu sau review để chia split.

Các CSV chỉ có header. Đặt bản làm việc vào `data/sources/<version>/` hoặc `data/manifests/<version>/`. Các đường dẫn trong mẫu chưa có dữ liệu; chạy mẫu nguyên trạng phải thất bại rõ ràng.

Các config X/Y/features/generator chưa được tạo vì model chưa tích hợp. CRF không có mặc định trong CLI nén để tránh vô tình sử dụng mức final test khi phát triển.
