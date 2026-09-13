# Evaluation — đánh giá model

Hiện có `check_shortcuts.py`: thử dự đoán bằng metadata của media để phát hiện dấu hiệu phụ. Đây chưa phải chương trình đánh giá detector audio–visual.

Phần sẽ triển khai: nạp detector/checkpoint, dự đoán trên split đã chọn, tính chỉ số và tách kết quả theo generator đã thấy/chưa thấy, mức nén, cũng như tỷ lệ báo nhầm real. So sánh X/Y trên cùng protocol.

Ngưỡng được chọn trên validation và giữ cố định khi đánh giá final test. Dự đoán, chỉ số và cấu hình đánh giá nằm trong `experiments/<run_id>/`, không trong source. Không dùng chẩn đoán metadata làm kết quả của detector chính.
