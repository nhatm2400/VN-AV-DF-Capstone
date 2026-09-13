# Features — chuẩn bị đầu vào cho model

Trạng thái: mới tạo khung thư mục, chưa có preprocessing/extractor cho model mới.

Chứa phần chuẩn bị audio và vùng miệng cùng đoạn thời gian, rồi lấy đặc trưng bằng backbone đã chọn. AV-HuBERT là ứng viên cần thử nghiệm. Phần định nghĩa/nạp backbone dùng chung thuộc `src/models/`; thư mục này điều khiển xử lý các clip và lưu đặc trưng khi cần.

Khác với `src/data/` (tải, cắt và quản lý dataset), phần này phụ thuộc yêu cầu đầu vào cụ thể của model. ROI preview đang có chỉ phục vụ review, chưa phải đầu vào AV-HuBERT đã xác minh.

Nếu lưu đặc trưng, đặt trong `cache/features/<dataset_version>/` kèm định danh checkpoint và cấu hình xử lý. Không để feature arrays hoặc weights trong source.
