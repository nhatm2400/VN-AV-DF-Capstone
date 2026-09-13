# Generators — tạo video fake

Trạng thái: mới tạo khung thư mục, chưa tích hợp generator.

Chứa code gọi công cụ chỉnh môi, dự kiến thử Wav2Lip/MuseTalk và một generator giữ riêng cho test. Nhận clip real đã chia split và audio đầu vào; xuất video fake cùng thông tin nguồn, generator, checkpoint, cấu hình và lỗi. Audio thay lời không được vượt split.

Media đầu ra nằm trong `data/generated/<dataset_version>/`, bảng nguồn/nhãn nằm trong `data/manifests/<dataset_version>/`; không lưu video hoặc weights trong thư mục source này.

Generator **tạo fake**; detector trong `src/models/` **phát hiện fake**. Khi tích hợp xong, thêm file chạy có số ngay trong thư mục generators.
