# Chuyển cấu trúc repo — 13/09/2026

Nhánh `codex/research-reset`, từ commit `0178ef43685e126e38e9a40be683db92b24452e9`. Đã chuyển báo cáo và bằng chứng cũ vào `docs/archives/legacy_avsp/`; source và media đầy đủ còn trong backup ở ổ F. Không sửa backup, không commit/push trong đợt này.

- Đối chiếu backup trước dọn: 68.136 file, 155,56 GiB; cùng dung lượng theo file, hash 110 file code/tài liệu và ba clip mẫu khớp. Chênh lệch duy nhất là ref nội bộ Codex 41 byte. Không hash toàn bộ media.
- Dọn 66.580 file/output/code cũ, tổng dung lượng logic 166.907.383.391 byte (~155,44 GiB). Đây là lượng đã xóa, không phải dung lượng trống ròng sau khi tạo archive.
- Giữ 214 file metadata từ data cũ, đã so hash bản archive với nguồn trước khi xóa. Có inventory, snapshot tài liệu, patch thay đổi chưa commit và source state trong archive.
- Chuyển công cụ cắt/gộp/quality/timeline sang `src/data/`, metadata diagnostic sang `src/evaluation/`; giữ review và tách cropper khỏi feature AVSP-Net. Không còn cây `src/pipeline/` hoặc model/train AVSP-Net trong code hiện hành.
- Tách split real khỏi ràng buộc bốn fake; cập nhật đường dẫn, nhãn nhóm nguồn, manifest quyền sử dụng và CLI nén. Không triển khai detector/generator mới trong đợt dọn.
- Chuyển `mmb.md` thành hướng nghiên cứu trong `docs/research/`; chuyển kế hoạch pilot vào `docs/planning/`. Viết lại README/PROJECT/CLAUDE, chỉ mục và `.gitignore`.

## Kiểm chứng

Trước chuyển đổi: 49 test nền qua. Sau chuyển đổi: `python -m unittest discover -s tests -q` chạy 58 test qua, gồm split người/nguồn, review, media và nén. Media của test được tổng hợp bằng FFmpeg và xóa khi test kết thúc; không phải dataset nghiên cứu hay bằng chứng accuracy.

Môi trường kiểm tra: Python 3.10.20 của môi trường data sẵn có. Chưa cài lại môi trường từ đầu; chưa tải nguồn mới, gọi YouTube API, tải/chạy AV-HuBERT hoặc generator, train Y/X hay dựng demo detector.

Ngưỡng active-speaker giữ lại là cấu hình tham khảo, cần kiểm chứng trên nguồn mới; việc test logic pass không chứng nhận auto gate. File kế hoạch dọn mô tả cả module tương lai; chỉ `PROJECT.md` ghi trạng thái hiện hành.

Tiếp theo: pilot nguồn mới và chạy thử backbone trên một tập nhỏ, theo [kế hoạch pilot](../../planning/KE_HOACH_PILOT_LIP_SYNC_VI.md).

Kiểm tra bàn giao: 51 file Python parse được; 9 CLI --help chạy được; liên kết tài liệu hiện hành không thiếu đích; đã cập nhật 47 liên kết tài liệu lịch sử; git diff --check đạt. Sau sửa số mẫu calibration, test calibration đã chạy lại và qua.
