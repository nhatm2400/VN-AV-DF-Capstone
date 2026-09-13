# REPORT — Tái cấu trúc Pipeline & Dọn dữ liệu

Ngày: 20/06/2026

---

## 1. Dọn 810 file hỏng

- Trên đĩa có 7.698 `.mp4` nhưng chỉ **6.888 clip hợp lệ**; 810 file dư là **rác 0 byte** (ffmpeg ghi dở, `moov atom not found`).
- Kiểm chứng bằng ffprobe: **810/810 không mở được**, đến từ **14 video "chết"** (mỗi video chỉ phun ra `clip0000` hỏng, không có clip tốt nào lẫn vào).
- Đã **xóa an toàn 810 file** (điều kiện kép: ngoài manifest + 0 byte). Đĩa còn đúng **6.888** = khớp manifest. CSV (clip tốt + rejects) không bị ảnh hưởng.

## 2. Tái cấu trúc `src/pipeline/` — stage đánh số + tier con

Quy ước: **folder stage đánh số** (`01_`, `02_`…); **file trong stage đánh số lại từ `01_`**; tier nào cần code riêng thì tách `tierN/`, không thì để phẳng.

```
src/pipeline/
  01_collect/        ← thu thập + cắt (tách tier vì nguồn khác nhau)
    tier1/  01_fetch_youtube_urls  02_download  03_quality_gate  04_cut_clips.ipynb
    tier2/  00_explore_license  01_fetch_youtube_urls  02_download
            03_generate_download_script  04_retry_failed_downloads  05_clean_temp_files
    tier3/  01_fetch_tiktok_urls  02_download  03_quality_gate  04_cut_clips.ipynb
  02_curate/         01_prep_manifest  02_score_clips  03_sync_score  04_curate
  03_fake/           01_temporal_desync
  04_extract_features/   (placeholder)
  05_build_labels/       (placeholder)
```

- Git nhận diện **100% là rename** -> giữ nguyên lịch sử.
- Đồng bộ tên file tham chiếu trong code; chuẩn hóa `03-cut-clips.ipynb` (tier3) -> `04_cut_clips.ipynb`.

## 3. Dọn tier2 + sửa bug

- 3 helper tải tier2 được **đánh số 03/04/05** (không đổi logic).
- `src/pre-testing/tier2.py` -> `01_collect/tier2/00_explore_license.py` (khảo sát license CC/Standard + quota trước khi fetch). Giữ folder `pre-testing/` trống.
- **Sửa `get_project_root`** ở 3 file path-sensitive: lùi đúng 4 cấp về repo root (lần move làm sai số tầng).

## 4. Cấu trúc `data/` theo thiết kế nhóm

- Tạo `data/fake/`, `data/features/` (+ `.gitkeep`), `.gitignore` chặn nội dung — giữ folder rỗng.
- Quyết định **giữ tên folder mô tả, KHÔNG đánh số ở cấp tier** trong `01_collect` (đồng bộ với cách cả nhóm + `model/train/eval`).

## 5. Commit

| Commit | Nội dung |
|---|---|
| `4355e39` | refactor cấu trúc stage đánh số + tier con |
| `ac6b7db` | đánh số helper tier2 + chuyển explore script + sửa get_project_root |

---

## Việc còn treo

- **Nhắc nhóm `git pull`/rebase** — đợt restructure đụng đường dẫn cũ của tier2/tier3 (code của thành viên khác).
- Sinh fake hàng loạt nên chạy **sau** khi curate (04_curate) ra tập sạch.
