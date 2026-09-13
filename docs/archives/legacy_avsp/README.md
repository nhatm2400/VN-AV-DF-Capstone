# Archive nghiên cứu AVSP-Net / pseudo-fake

Đóng mốc ngày 13/09/2026, trước khi chuyển sang hướng lip-sync audio–visual. Các báo cáo giữ nguyên kết luận lịch sử; số clip, AUC và NO-GO chỉ áp dụng population/model được nêu trong từng báo cáo.

| Thư mục | Nội dung |
|---|---|
| reports/ | Pilot V1, review V1/V2, temporal smoke, cut hotfix, active-speaker implementation. |
| architecture/ | MODEL_PROPOSAL của AVSP-Net V2. |
| logs/ | Nhật ký curation, multi-reviewer và rebuild. |
| manifests/data/ | Bản CSV/JSON và checksum của population cũ, giữ nguyên đường dẫn tương đối. |
| experiments/ | Config, history, metrics và source state của pilot đã chạy. |
| environment/ | requirements và config cutter cũ. |
| project_snapshot/ | README/PROJECT/CLAUDE cũ, chỉ mục/hướng dẫn cũ, notebook, working-tree patch, source_state, inventory và cleanup_result. |
| 2026_report_checkpoints/ | Các mốc báo cáo trước đây, giữ nguyên file. |

Commit trước chuyển đổi: `0178ef43685e126e38e9a40be683db92b24452e9`. Thay đổi chưa commit của người dùng được giữ trong bản backup và các snapshot/patch. Source code cũ lấy từ backup hoặc lịch sử Git; archive không chứa thêm một cây source song song.

Bản backup local đầy đủ đã đối chiếu tại `F:/capstone_temp/VN-AV-DF-Capstone`. Kiểm kê 68.136 file, 167.031.386.412 byte; đối chiếu size toàn cây, hash 110 file code/tài liệu và ba clip mẫu. Không hash toàn bộ media. Bản backup không bị sửa trong đợt chuyển đổi.

Liên kết nội bộ của báo cáo được điều chỉnh khi có đích tài liệu đã chuyển. Đường dẫn code/media lịch sử không còn trong workspace chỉ được dùng để tra bản backup; chúng không phải lệnh chạy hiện hành. Bảng quyền/metadata lịch sử không chứng minh toàn bộ nguồn được phép phát hành.

Điểm bắt đầu hiện hành: [PROJECT](../../../../PROJECT.md).
