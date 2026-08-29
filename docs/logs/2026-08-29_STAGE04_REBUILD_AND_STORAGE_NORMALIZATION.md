# Nhật ký 2026-08-29 — Hoàn tất Stage 04 và chuẩn hóa dung lượng

**Ngày:** 2026-08-29

**Phạm vi:** tập hợp đủ output Cut Clips của ba tier, audit Gate C, xử lý outlier 4K và bàn giao sang P4 curation

**Trạng thái:** Stage 04 **đã hoàn tất**; P4/P5 chưa chạy; manual review mới, sinh fake, extract feature và train vẫn **NO-GO**

## 1. Population Stage 04 hiện tại

| Tier | Video nguồn | Batch | Accepted clip/MP4 | Rejected window | Dung lượng hiện tại |
|---|---:|---:|---:|---:|---:|
| Tier 1 | 472 | 10 | 30.623 | 57.575 | 67,496 GiB |
| Tier 2 | 292 | 4 | 20.512 | 36.081 | 45,719 GiB |
| Tier 3 | 2.274 | 3 | 14.487 | 26.531 | 42,062 GiB |
| **Tổng** | **3.038** | **17** | **65.622** | **120.187** | **155,277 GiB** |

Audit Gate C xác nhận:

- toàn bộ 3.038 video nguồn có terminal status;
- range batch liên tục, không thiếu hoặc overlap;
- `65.622` accepted row khớp 1–1 với `65.622` MP4;
- không missing/orphan/zero-byte/trùng `clip_id` trong hoặc giữa tier;
- checksum metadata và media mẫu đạt trước khi xử lý dung lượng;
- sau downscale, toàn bộ file bị tác động được probe và hash lại, khớp
  `SHA256SUMS` hiện hành.

Terminal status theo tier:

- Tier 1: `432 completed`, `39 completed_no_clips`, `1 no_speech_detected`;
- Tier 2: `285 completed`, `7 completed_no_clips`;
- Tier 3: `1.813 completed`, `253 completed_no_clips`, `208 no_speech_detected`.

## 2. Chuẩn hóa nhóm 4K

Audit độ phân giải trên một MP4 đại diện cho mỗi nguồn accepted tìm thấy đúng
14 nguồn 4K, tất cả là `3840×2160`. Chúng tạo 1.200 clip — chỉ 1,83% số clip
nhưng chiếm `46,381 GiB`, tương đương 23,37% dung lượng Stage 04 trước xử lý.

Theo quyết định lưu trữ, 1.200 file được downscale **trực tiếp tại chỗ**:

- video: `libx264`, CRF 18, preset `veryfast`, resize Lanczos;
- độ phân giải: `3840×2160 → 1920×1080`;
- audio: stream-copy, không encode lại;
- publish: encode ra file tạm, probe video/timeline/audio và so hash audio packet,
  sau đó mới atomic replace;
- checksum media trong 8 batch bị tác động được cập nhật sau khi đủ coverage.

Kết quả đo thực tế:

| Chỉ số | Trước | Sau |
|---|---:|---:|
| Nhóm 1.200 clip | 46,381 GiB | 3,191 GiB |
| Toàn Stage 04 | 198,467 GiB | 155,277 GiB |
| Dung lượng giải phóng | — | **43,190 GiB** |

Audit độc lập sau encode đạt `1.200/1.200`: đúng H.264 1920×1080, hash file
khớp checksum mới, accepted/media toàn bộ vẫn 1–1 và không còn file `.part`.

## 3. Quyết định provenance và giới hạn rollback

Thư mục provenance tạm của phép downscale gồm progress JSONL, summary JSON và
bản sao checksum cũ đã được xóa theo yêu cầu dọn dữ liệu. `SHA256SUMS` hiện hành
trong từng batch vẫn được giữ và mô tả đúng MP4 1080p hiện tại.

Hệ quả cần ghi rõ:

- media 4K gốc đã bị ghi đè và không được Git track;
- không thể rollback về pixel 4K từ Git hoặc checksum cũ;
- số liệu trước/sau và cấu hình encode được cố định trong nhật ký này;
- `cut_backend` trong accepted CSV vẫn mô tả backend cắt ban đầu; bước downscale
  sau cut được mô tả ở đây, không sửa lại identity hay lineage của clip.

## 4. Dọn artifact lịch sử

Đợt cleanup hiện tại đã loại PoC cũ, archive media/manifest Pilot V1 và Phase 0,
measurement/EDA/manual/assignment của population cũ. Ba `all_clean*.csv` còn lại
chỉ để tham khảo lịch sử `6.888 → 3.001`; không được đưa vào fake/training mới.

Run model Pilot V1 trong `experiments/` và các báo cáo Markdown vẫn được giữ để
tham chiếu kết quả, nhưng không đủ để tái lập toàn bộ pilot vì media/feature nguồn
đã được dọn.

## 5. Việc tiếp theo

1. Chạy `src/pipeline/02_curate/01_prep_manifest.py`; kỳ vọng manifest mới có
   đúng 65.622 dòng và không có media lỗi.
2. Chạy score/motion, curate và EDA theo Gate P4; không tái dùng measurement cũ.
3. Dựng `all_clean_review.csv` và ROI preview từ population mới.
4. Chia primary/calibration cho ba reviewer, export và verify gói review ở P5.
5. Dừng để bàn giao assignment; chỉ tiếp tục fake V2 sau khi manual review và
   merge/adjudication hoàn tất.
