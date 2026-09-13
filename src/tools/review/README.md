# Review — mở file có số rồi Run

Thực hiện sau `src/data/04_build_manifest.py`. Dùng interpreter đã chọn cho pipeline. Cấu hình tên nhóm và tên của bạn trong [settings.py](../../data/preparation/settings.py).

| File | Tác dụng |
|---|---|
| [01_build_roi_preview.py](01_build_roi_preview.py) | Tùy chọn: dựng video vùng miệng có tiếng. Có thể bỏ qua và xem clip gốc. |
| [02_assign_reviewers.py](02_assign_reviewers.py) | Chia clips.csv cho REVIEWERS; mỗi clip một reviewer. Mặc định chưa có tập calibration chung. |
| [03_review_clips.py](03_review_clips.py) | Mở giao diện local cho REVIEWER, lưu quyết định vào reviews/results/review_<tên>.csv. |
| [04_merge_reviews.py](04_merge_reviews.py) | Kiểm tra đủ quyết định, ghi reviewed_clips.csv; nếu thiếu/xung đột thì dừng và xuất danh sách cần xử lý. |
| [05_export_batch_optional.py](05_export_batch_optional.py) | Tùy chọn sau khi phân công: gom media của một reviewer để chuyển máy; không cần đợi gộp kết quả. |

File không có số là phần xử lý mà các bước này gọi. `build_review_manifest.py` chỉ dùng nếu cần ghép thêm điểm đo từ các công cụ quality. `mouth_roi.py` là hàm crop, không phải bước chạy độc lập. Công cụ quality/active-speaker là nhánh tùy chọn, không bắt buộc chạy trước review thủ công.

Trước bước 05 của pipeline, mở `data/manifests/dataset_v1/reviewed_clips.csv`, bổ sung `speaker_id` nhất quán giữa mọi tập (và speaker_ids nếu liên quan nhiều người). Giao diện hiện chỉ review chất lượng, **chưa tự gán danh tính người nói**. Giữ source_video/canonical_source_id để kiểm tra leakage. Review pass chưa đồng nghĩa clip đã sẵn sàng train.

Đường dẫn kết quả nằm trong `data/manifests/dataset_v1/reviews/`; preview trong `cache/previews/dataset_v1/`. Chuyển media cho người khác không tự chuyển đường dẫn tuyệt đối của máy: giao diện hỗ trợ `--media_root <thư-mục-media>` khi cần. Bộ rubric cũ được giữ nhưng ngưỡng chất lượng phải rà trên nguồn mới.
