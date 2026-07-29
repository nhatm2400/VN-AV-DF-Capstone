# src/tools — công cụ độc lập

Khác `src/pipeline/`: pipeline chạy theo thứ tự stage 01→05 để sinh dataset; các file ở
đây là công cụ rời, chạy khi cần, không nằm trong đường chạy chính.

## Quy trình lọc tay (manual curation)

Sáu file dưới đây là **một quy trình liền mạch**, chạy theo đúng thứ tự này. Xem hướng
dẫn đầy đủ ở [nhật ký 2026-07-28](../../docs/logs/2026-07-28_MULTI_REVIEWER_PLAN_AND_DATA_DISTRIBUTION.md#6-hướng-dẫn-chạy).

| Thứ tự | File | Việc |
|---|---|---|
| 1 | `build_review_manifest.py` | Gộp `all_clean.csv` với các phép đo phụ (motion, nhập nhằng nhiều mặt, kênh) → `manifests/all_clean_review.csv`. Tái lập được, fail nếu một nguồn phủ dưới 95%. |
| 2 | `build_roi_preview.py` | Dựng video preview vùng miệng **kèm audio gốc**, dùng đúng `detect_and_crop` của stage 04. Đây là thứ để lộ lồng tiếng / cắt nhầm mặt / ảnh tĩnh. ~1,7 giây/clip. |
| 3 | `export_review_batch.py` | Gom clip gốc của manifest vào một thư mục phẳng `<clip_id>.mp4` để phát cho reviewer. |
| 4 | `build_review_assignments.py` | Chia manifest cho nhiều reviewer: primary disjoint cân bằng theo tier + calibration set dùng chung để đo đồng thuận. |
| 5 | `clip_review.py` | **Công cụ chính.** Web UI xem video gốc + ô ROI có tiếng, chấm KEEP / REJECT(+lý do) / UNCERTAIN. Chỉ dùng thư viện chuẩn Python. `--media_root` để chạy trên máy khác. |
| 6 | `merge_review_results.py` | Gộp kết quả nhiều người, fail-closed. Tách clip bất đồng sang `needs_adjudication.csv`. `--allow_partial` khi chủ ý dừng sớm. |

## Phép đo phụ trợ

Chạy một lần để trả lời một câu hỏi cụ thể; kết quả nằm ở `data/02_curate/measurements/`.

| File | Đo gì | Kết luận đã rút ra |
|---|---|---|
| `scan_face_ambiguity.py` | Số khuôn mặt và tỉ lệ diện tích mặt nhì/mặt nhất | 22% clip có ≥2 mặt xấp xỉ nhau → luật "chọn mặt to nhất" của stage 04 không đáng tin ở nhóm này |
| `measure_lip_audio_corr.py` | Tương quan cử động miệng ↔ năng lượng âm | AUC 0,544 trên 60 clip có nhãn tay — **không** tách được clip tốt/xấu. Giữ lại làm bằng chứng cho quyết định phải review tay. |

## Khác

| File | Việc |
|---|---|
| `download_data.py` | Tải dataset từ nguồn ngoài, có khôi phục khi file ZIP hỏng. Dùng ở giai đoạn thu thập, không liên quan curation. |

## Lưu ý

- Mọi lệnh chạy từ **thư mục gốc dự án**, đường dẫn mặc định là tương đối so với gốc.
- Python phải gọi bằng đường dẫn tuyệt đối của env (xem `CLAUDE.md`), trừ `clip_review.py`
  chạy trên máy reviewer thì Python nào cũng được vì chỉ dùng thư viện chuẩn.
