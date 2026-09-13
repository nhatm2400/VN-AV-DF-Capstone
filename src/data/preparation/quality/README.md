# Đo chất lượng tùy chọn

Giữ công cụ đo mặt/embedding, gom speaker, active-speaker và phát hiện trường hợp nhiều mặt. Đây là công cụ curation tái sử dụng, không phải detector lip-sync và không phải bước bắt buộc cho mọi pilot.

- `face_quality.py`: đo mặt, tạo embedding; cần InsightFace/ONNX runtime.
- `curate.py`: gom người, lọc chất lượng và cân bằng; ngưỡng phải được kiểm chứng trên nguồn mới. ID cluster cần rà thủ công xuyên chương trình.
- `scan_face_ambiguity.py`: chẩn đoán nhiều mặt bằng YOLO.
- `active_speaker/`: tạo pool nguồn độc lập, chấm/ghép timeline, tích hợp Light-ASD/LASER và calibrate policy nếu nhóm cần. CLI tham số nguồn/đầu ra trỏ vùng mới. `tier` chỉ còn là tên nhóm nguồn; số nhóm không cố định bằng ba.

Chỉ test phần logic bằng fixture trong đợt chuyển đổi. Chưa tải/chạy lại model ngoài trên nguồn mới; không lấy ngưỡng hoặc trạng thái NO-GO cũ làm trạng thái dataset mới.
