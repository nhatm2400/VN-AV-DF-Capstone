# Chỉ mục tài liệu

Tài liệu ở root chỉ giữ vai trò quy tắc và điểm vào. Nội dung chuyên môn được tổ chức trong `docs/`.

## Kiến trúc

- [MODEL_PROPOSAL.md](architecture/MODEL_PROPOSAL.md) — AVSP-Net V2, gồm V2a và V2b, loss, code layout, output contract và roadmap.

## Báo cáo hiện tại

- [PILOT_REPORT.md](reports/PILOT_REPORT.md) — báo cáo chi tiết quá trình pilot V1.
- [PILOT_V1_REVIEW_AND_V2_PLAN.md](reports/PILOT_V1_REVIEW_AND_V2_PLAN.md) — review V1 sau pilot, fact-check, blocking issue và kế hoạch V2.
- [TEMPORAL_DESYNC_PHASE0_SMOKE.md](reports/TEMPORAL_DESYNC_PHASE0_SMOKE.md) — code repair, test matrix và smoke dữ liệu thật cho temporal V2.

## Báo cáo lịch sử

Các checkpoint cũ được giữ trong [`archives/2026_report_checkpoints/`](archives/2026_report_checkpoints/) để truy vết lịch sử; không dùng chúng thay cho trạng thái hiện tại trong `PROJECT.md`.

## Nhật ký làm việc

- [2026-07-27 — Manual curation và audit dung lượng](logs/2026-07-27_MANUAL_CURATION_AND_STORAGE_AUDIT.md) — tóm tắt phiên phân tích manual review, giới hạn của mẫu 60 clip và phân loại dữ liệu có thể dọn.

## Thứ tự ưu tiên khi đọc

1. [`../CLAUDE.md`](../CLAUDE.md)
2. [`../PROJECT.md`](../PROJECT.md)
3. [Báo cáo review V1/V2](reports/PILOT_V1_REVIEW_AND_V2_PLAN.md)
4. [Báo cáo Phase 0 temporal](reports/TEMPORAL_DESYNC_PHASE0_SMOKE.md)
5. [Kiến trúc V2](architecture/MODEL_PROPOSAL.md)
6. [Báo cáo pilot gốc](reports/PILOT_REPORT.md)
