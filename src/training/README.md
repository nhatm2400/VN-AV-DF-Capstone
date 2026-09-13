# Training — huấn luyện detector

Trạng thái: có hàm `train_step` trong `steps.py`, chưa có loader dataset thật hoặc chương trình train nhiều epoch/checkpoint.

Y dùng `consistency_weight=0`; X dùng trọng số lớn hơn 0. Cả hai đều học nhãn real=0/fake=1 trên cả hai view. X cộng thêm lỗi chênh lệch score giữa hai view. Trọng số X phải chọn bằng validation, không lấy giá trị trong smoke làm cấu hình nghiên cứu.

Batch có audio, visual, paired_audio, paired_visual, lengths, labels, sample_ids và paired_sample_ids. Mỗi ID phải xác định cùng clip và cửa sổ; cặp chỉ khác mức nén, không phải real ghép với fake. Feature phải được trích thật từ từng view bằng cùng cấu hình. Hàm kiểm tra ID/hình dạng, nhưng không thể xác minh nguồn gốc media chỉ từ tensor.

Chứa cách đọc manifest/đặc trưng thành batch, vòng lặp train, cách tính lỗi, cập nhật model, validation và lưu checkpoint. Phần khuyến khích dự đoán nhất quán giữa các bản nén của phương pháp X sẽ nằm ở đây.

Nhận split đã khóa từ luồng data; không tự chia ngẫu nhiên lại clip. X/Y dùng cùng dữ liệu, bản nén và ngân sách train. Chỉ chọn cấu hình/ngưỡng trên validation, không dùng final test để chọn model.

Model được import từ `src/models/`. Log, cấu hình, kết quả validation và checkpoint ghi vào `experiments/<run_id>/`, không vào source. File chạy có số sẽ được thêm ngay trong thư mục training khi phần train được triển khai.
