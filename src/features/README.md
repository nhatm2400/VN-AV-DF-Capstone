# Features — chuẩn bị đầu vào cho model

Trạng thái: có `avhubert.py`, adapter nhận AVHubertModel đã được nạp và lấy hai nhánh
audio/video trước fusion. Adapter đóng băng tham số và tắt dropout. Đây KHÔNG phải
đầu ra Transformer hợp nhất của AV-HuBERT. Mới kiểm tra bằng backbone giả lập,
chưa nạp checkpoint thật và chưa có preprocessing video/audio hoàn chỉnh.

Audio đầu vào là acoustic features đã stack theo checkpoint `[B,F,T]`, không phải
waveform WAV; video là mouth ROI grayscale đã chuẩn hóa `[B,1,T,H,W]`. Hai luồng
phải có cùng timestamps. Adapter trả hai tensor `[B,T,D]`. Chạy extraction từng
clip/window chưa padding; cache mỗi bản nén riêng, không dùng lại feature bản gốc.

API đối chiếu: https://github.com/facebookresearch/av_hubert/blob/main/avhubert/hubert.py
(`feature_extractor_audio`, `feature_extractor_video`). Việc lấy pre-fusion features
là lựa chọn thử nghiệm, không chứng minh ưu thế so với Transformer đầy đủ hoặc
khả năng hoạt động tốt trên tiếng Việt. Không tự tải weights khi import.

Chứa phần chuẩn bị audio và vùng miệng cùng đoạn thời gian, rồi lấy đặc trưng bằng backbone đã chọn. AV-HuBERT là ứng viên cần thử nghiệm. Phần định nghĩa/nạp backbone dùng chung thuộc `src/models/`; thư mục này điều khiển xử lý các clip và lưu đặc trưng khi cần.

Khác với `src/data/` (tải, cắt và quản lý dataset), phần này phụ thuộc yêu cầu đầu vào cụ thể của model. ROI preview đang có chỉ phục vụ review, chưa phải đầu vào AV-HuBERT đã xác minh.

Nếu lưu đặc trưng, đặt trong `cache/features/<dataset_version>/` kèm định danh checkpoint và cấu hình xử lý. Không để feature arrays hoặc weights trong source.
