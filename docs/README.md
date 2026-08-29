# Chỉ mục tài liệu

Tài liệu ở root chỉ giữ vai trò quy tắc và điểm vào. Nội dung chuyên môn được tổ chức trong `docs/`.

## Kiến trúc

- [MODEL_PROPOSAL.md](architecture/MODEL_PROPOSAL.md) — AVSP-Net V2, gồm V2a và V2b, loss, code layout, output contract và roadmap.

## Báo cáo hiện tại

- [CUT_CLIPS_HOTFIX_AND_REBUILD_PLAN.md](reports/CUT_CLIPS_HOTFIX_AND_REBUILD_PLAN.md) — audit lỗi giải mã ở Stage 04 Cut Clips, phạm vi dữ liệu bị ảnh hưởng và thứ tự dựng lại toàn bộ downstream.
- [PILOT_REPORT.md](reports/PILOT_REPORT.md) — báo cáo chi tiết quá trình pilot V1.
- [PILOT_V1_REVIEW_AND_V2_PLAN.md](reports/PILOT_V1_REVIEW_AND_V2_PLAN.md) — review V1 sau pilot, fact-check, blocking issue và kế hoạch V2.
- [TEMPORAL_DESYNC_PHASE0_SMOKE.md](reports/TEMPORAL_DESYNC_PHASE0_SMOKE.md) — code repair, test matrix và smoke dữ liệu thật cho temporal V2.

## Nhật ký làm việc

- [2026-08-29 — Hoàn tất Stage 04 và chuẩn hóa dung lượng](logs/2026-08-29_STAGE04_REBUILD_AND_STORAGE_NORMALIZATION.md) — audit đủ ba tier, downscale 1.200 clip 4K và trạng thái bàn giao sang P4/P5.
- [2026-07-27 — Manual curation và audit dung lượng](logs/2026-07-27_MANUAL_CURATION_AND_STORAGE_AUDIT.md) — tóm tắt phiên phân tích manual review, giới hạn của mẫu 60 clip và phân loại dữ liệu có thể dọn.
- [2026-07-28 — Multi-reviewer và phân phối dữ liệu](logs/2026-07-28_MULTI_REVIEWER_PLAN_AND_DATA_DISTRIBUTION.md) — tooling chia việc cho ba reviewer, số clip thực sự cần review, và bộ 7,3 GiB đã gom sẵn để phát.

## Thứ tự ưu tiên khi đọc

1. [`../CLAUDE.md`](../CLAUDE.md)
2. [`../PROJECT.md`](../PROJECT.md)
3. [Kế hoạch hotfix Stage 04 và dựng lại dữ liệu](reports/CUT_CLIPS_HOTFIX_AND_REBUILD_PLAN.md)
4. [Báo cáo review V1/V2](reports/PILOT_V1_REVIEW_AND_V2_PLAN.md)
5. [Báo cáo Phase 0 temporal](reports/TEMPORAL_DESYNC_PHASE0_SMOKE.md)
6. [Kiến trúc V2](architecture/MODEL_PROPOSAL.md)
7. [Báo cáo pilot gốc](reports/PILOT_REPORT.md)
