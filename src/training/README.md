# Training — huấn luyện detector

Trạng thái: mới tạo khung thư mục, chưa có chương trình train cho model mới.

Chứa cách đọc manifest/đặc trưng thành batch, vòng lặp train, cách tính lỗi, cập nhật model, validation và lưu checkpoint. Phần khuyến khích dự đoán nhất quán giữa các bản nén của phương pháp X sẽ nằm ở đây.

Nhận split đã khóa từ luồng data; không tự chia ngẫu nhiên lại clip. X/Y dùng cùng dữ liệu, bản nén và ngân sách train. Chỉ chọn cấu hình/ngưỡng trên validation, không dùng final test để chọn model.

Model được import từ `src/models/`. Log, cấu hình, kết quả validation và checkpoint ghi vào `experiments/<run_id>/`, không vào source. File chạy có số sẽ được thêm ngay trong thư mục training khi phần train được triển khai.
