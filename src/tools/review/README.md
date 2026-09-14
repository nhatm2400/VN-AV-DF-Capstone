# Review — mở file có số rồi Run

Thực hiện sau `src/data/04_build_manifest.py`. Dùng interpreter đã chọn cho pipeline. Cấu hình tên nhóm và tên của bạn trong [settings.py](../../data/preparation/settings.py).

| File | Tác dụng |
|---|---|
| [01_build_roi_preview.py](01_build_roi_preview.py) | Tùy chọn: dựng video vùng miệng có tiếng. Có thể bỏ qua và xem clip gốc. |
| [02_assign_reviewers.py](02_assign_reviewers.py) | Chia clips.csv cho REVIEWERS; mỗi clip một reviewer. Mặc định chưa có tập calibration chung. |
| [03_review_clips.py](03_review_clips.py) | Mở giao diện local, lưu vào reviews/exports/clips/<tên>/review_<tên>.csv. |
| [04_merge_reviews.py](04_merge_reviews.py) | Kiểm tra đủ quyết định, ghi reviewed_clips.csv; nếu thiếu/xung đột thì dừng và xuất danh sách cần xử lý. |
| [05_export_batch_optional.py](05_export_batch_optional.py) | Tùy chọn sau khi phân công: gom media của một reviewer để chuyển máy; không cần đợi gộp kết quả. |

File không có số là phần xử lý mà các bước này gọi. `build_review_manifest.py` chỉ dùng nếu cần ghép thêm điểm đo từ các công cụ quality. `mouth_roi.py` là hàm crop, không phải bước chạy độc lập. Công cụ quality/active-speaker là nhánh tùy chọn, không bắt buộc chạy trước review thủ công.

Trước bước 05 của pipeline, mở `data/manifests/dataset_v1/reviewed_clips.csv`, bổ sung `speaker_id` nhất quán giữa mọi tập (và speaker_ids nếu liên quan nhiều người). Giao diện hiện chỉ review chất lượng, **chưa tự gán danh tính người nói**. Giữ source_video/canonical_source_id để kiểm tra leakage. Review pass chưa đồng nghĩa clip đã sẵn sàng train.

## Review và đồng bộ bằng Git

Pull nhánh `codex/research-reset`. Giải nén video từ Drive vào
`data/manifests/dataset_v1/reviews/exports/clips/<tên>/`, cạnh assignment có sẵn.
Không tạo thêm một folder tên người bên trong folder đó. Có thể đặt video vào
`media/` và ROI vào `roi/` bên trong folder của mình. Không cần chạy lại bước 01/02/05.

Từ root repo, Linh chạy:

```powershell
python src/tools/review/03_review_clips.py --reviewer nguyenvanlinh
```

Lâm Anh dùng `--reviewer nguyenlamanh`, Nhật dùng `--reviewer nguyenminhnhat`.
Không cần sửa settings.py. Giao diện tự lưu; khi nghỉ, dừng bằng Ctrl+C rồi commit:

```powershell
git add data/manifests/dataset_v1/reviews/exports/clips/nguyenvanlinh/review_nguyenvanlinh.csv
git commit -m "Review clips by nguyenvanlinh"
git pull --rebase origin codex/research-reset
git push origin codex/research-reset
```

Mỗi người thay cả đường dẫn và tên bằng tên mình, chỉ sửa kết quả của mình.
Nếu push báo nhánh remote có commit mới, pull --rebase rồi push lại; không force-push.
Git theo dõi CSV/metadata; media, ZIP và data/real được ignore. Không xóa hay
ghi đè CSV kết quả khi giải nén lại. Khi cả nhóm xong, người tổng hợp pull và chạy
`python src/tools/review/04_merge_reviews.py`; không cần chép kết quả thủ công.
