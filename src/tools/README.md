# src/tools — công cụ độc lập

Khác `src/pipeline/`: pipeline chạy theo thứ tự stage 01→05 để sinh dataset; các file ở
đây là công cụ rời, chạy khi cần, không nằm trong đường chạy chính. Chúng được chia thành
`review/`, `diagnostics/` và `data_admin/` để tránh một thư mục phẳng khó đọc.

## Quy trình lọc tay (manual curation)

Sáu file dưới đây là **một quy trình liền mạch**, chạy theo đúng thứ tự này. Xem hướng
dẫn đầy đủ ở [nhật ký 2026-07-28](../../docs/logs/2026-07-28_MULTI_REVIEWER_PLAN_AND_DATA_DISTRIBUTION.md#6-hướng-dẫn-chạy).

| Thứ tự | File | Việc |
|---|---|---|
| 1 | `review/build_review_manifest.py` | Gộp `all_clean.csv` với các phép đo phụ (motion, nhập nhằng nhiều mặt, kênh) → `manifests/all_clean_review.csv`. Tái lập được, fail nếu một nguồn phủ dưới 95%. |
| 2 | `review/build_roi_preview.py` | Dựng video preview vùng miệng **kèm audio gốc**, dùng đúng `detect_and_crop` của stage 04. Đây là thứ để lộ lồng tiếng / cắt nhầm mặt / ảnh tĩnh. ~1,7 giây/clip. |
| 3 | `review/export_review_batch.py` | Gom clip gốc của manifest vào một thư mục phẳng `<clip_id>.mp4` để phát cho reviewer. |
| 4 | `../pipeline/02_curate/02_scoring/02_active_speaker/05_build_calibration_manifest.py` | Chọn 450 clip source-disjoint, cân bằng 3 tier và nhóm rủi ro; khóa 300 tune + 150 validation. Đây là bước pipeline nên không còn nằm trong `tools/`. |
| 5 | `review/build_review_assignments.py` | Chia manifest cho reviewer: calibration dùng chung hoặc primary disjoint cân bằng theo tier. |
| 6 | `review/clip_review.py` | **Công cụ chính.** Rubric v3 đánh KEEP / REJECT / UNCERTAIN và các interval lỗi `start_ms/end_ms/reason`. `--media_root` để chạy trên máy khác. |
| 7 | `review/merge_review_results.py` | Gộp kết quả fail-closed; so interval với sai số 200 ms, xuất `consensus_labels_v3.csv`, đẩy bất đồng sang adjudication. |

## Phép đo phụ trợ

Chạy một lần để trả lời một câu hỏi cụ thể; kết quả nằm ở `data/02_curate/measurements/`.

| File | Đo gì | Kết luận đã rút ra |
|---|---|---|
| `diagnostics/scan_face_ambiguity.py` | Số khuôn mặt và tỉ lệ diện tích mặt nhì/mặt nhất | 22% clip có ≥2 mặt xấp xỉ nhau → luật "chọn mặt to nhất" của stage 04 không đáng tin ở nhóm này |
| `diagnostics/measure_lip_audio_corr.py` | Tương quan cử động miệng ↔ năng lượng âm | AUC 0,544 trên 60 clip có nhãn tay — **không** tách được clip tốt/xấu. Giữ lại làm bằng chứng cho quyết định phải review tay. |

## Quản trị dữ liệu

| File | Việc |
|---|---|
| `data_admin/download_data.py` | Tải dataset từ nguồn ngoài, có khôi phục khi file ZIP hỏng. |
| `data_admin/recover_cut_input_inventory.py` | Khôi phục inventory đầu vào Stage 04 từ các manifest nguồn. |
| `data_admin/snapshot_cut_hotfix_baseline.py` | Chụp provenance/checksum baseline trước hotfix; không copy media nặng. |

## Lưu ý

- Mọi lệnh chạy từ **thư mục gốc dự án**, đường dẫn mặc định là tương đối so với gốc.
- Python phải gọi bằng đường dẫn tuyệt đối của env (xem `CLAUDE.md`), trừ `review/clip_review.py`
  chạy trên máy reviewer thì Python nào cũng được vì chỉ dùng thư viện chuẩn.
