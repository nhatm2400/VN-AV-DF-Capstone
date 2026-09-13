# Công cụ review

**Cách chạy hiện hành:** mở các file có số 01–05 trong [review/](review/README.md) rồi Run Python File. Chúng dùng dataset_v1 và tên reviewer trong src/data/preparation/settings.py. Các lệnh dưới đây là cách gọi riêng các module xử lý khi cần tùy chỉnh.

`review/` giữ giao diện review có audio, ROI preview, phân công và gộp kết quả. Không có detector AI thật/giả trong giao diện này.

1. Chuẩn bị CSV clip có clip_id, file_path, source_video và nhãn nhóm nguồn nếu có. Để xem trực tiếp không cần chạy bộ lọc cũ.
2. Nếu cần ROI preview, chạy `python src/tools/review/build_roi_preview.py --csv <clips.csv> --out_dir <preview-dir> --limit 3` trên mẫu nhỏ. Cần YOLO/weights sẵn; crop này phục vụ review, chưa phải preprocessing AV-HuBERT đã xác minh.
3. `python src/tools/review/build_review_assignments.py --manifest <clips.csv> --reviewers A B C --no_shared_calibration --out_dir <assignments-dir>` chia việc cho pilot. Khi cần calibration chung, cung cấp CSV qua --calibration thay vì dùng dữ liệu cũ.
4. `python src/tools/review/clip_review.py --csv <assignment.csv> --roi_dir <preview-dir> --out <review.csv> --reviewer A` mở server local theo cổng CLI. Có thể review video gốc khi chưa có ROI.
5. `merge_review_results.py --help` chỉ cách gộp, báo thiếu review/xung đột; final manifest có decision=keep. Bổ sung speaker_id/canonical_source_id đã rà soát trước khi chia split.

`build_review_manifest.py` hỗ trợ ghép metadata/phép đo nếu đã có, không phải bước bắt buộc. `export_review_batch.py` tạo gói media cho reviewer khi quyền sử dụng/phạm vi chia sẻ phù hợp.

Các công cụ giữ rubric review cũ có kiểm tra interval. Ngưỡng curation/calibration cũ không tự có hiệu lực với nguồn mới. Trong pilot, gán người nói và nghe/xem trực tiếp là bước độc lập với detector.
