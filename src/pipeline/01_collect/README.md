# Stage 01 Collect

`04_cut_clips.ipynb` là notebook Kaggle duy nhất cho cả ba tier.
`cut_clips_core.py` chứa logic có test; config theo tier nằm trong `configs/`.

Hai notebook cũ tại `tier1/` và `tier3/` đã bị loại vì copy-paste sai cấu hình:
Tier 3 trong repo thực tế vẫn trỏ Tier 1, còn Tier 2 không có notebook.

Quy trình:

1. khóa input inventory từ output Stage 03;
2. chọn config tier, Git SHA, run ID và batch range trong notebook;
3. chạy smoke trước;
4. chạy các batch không overlap;
5. chỉ merge khi `run_summary.json` của mọi batch có `coverage_passed=true`.

Không tái sử dụng run ID và không ghi output mới vào
`data/01_collect/cut_clips/` trước khi toàn bộ Gate C đạt.
