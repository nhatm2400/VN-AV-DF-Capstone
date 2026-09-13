# REPORT — Kế hoạch Curate + EDA & chốt nơi chạy

Ngày: 20/06/2026 · Bối cảnh: chuẩn bị **Report 2 — Data Tasks (Collection + Cleaning + EDA)**, hạn 21/06.

---

## 1. Quyết định hướng đi


Chọn **chạy curate + EDA trên 6888 clip hiện có**, KHÔNG thu thập thêm trước hạn.

- Report 2 chấm **Cleaning + EDA**, không phải "đủ số clip". 6888 clip / 246 video / 3 tier là đủ.
- Thu thập thêm = download + cắt lại (Kaggle GPU) → rủi ro trượt hạn, mà không thêm điểm cho 2 mục đang nộp.
- **EDA chính là thứ chỉ ra đang thiếu đa dạng ở đâu** → sprint sau thu thập *có mục tiêu*, không đoán mò.

## 2. Làm rõ thứ tự pipeline (gỡ hiểu nhầm)

```
collect → cut → CURATE (clean) → [ Report 2: Collection + Cleaning + EDA ]  ← đang ở đây
                                       ↓ (sau khi nộp)
                                  fake → feature extraction → train → eval
```

- **EDA = phân tích bộ real clip đã làm sạch.** KHÔNG cần sinh fake hay feature extraction trước (đó là phase modeling sau).
- Với pseudo-fake temporal-desync: video copy nguyên byte → mọi trục visual của fake **trùng** real; chỉ trục **sync** khác. Phần này để mục *Label design + leakage audit* trong report (phân tích trên giấy, không cần chạy).

## 3. Thêm bước EDA vào pipeline

Tạo **`src/pipeline/02_curate/05_eda.py`** — bước cuối stage curate:

- Input: `tier1_scored_all.csv` (bắt buộc) + `all_clean.csv` (tùy chọn, để so trước/sau curate).
- Output: biểu đồ PNG rời (duration, snr, face_ratio, det_ratio, mean_face_area, embed_consistency, tier balance, face presence, top video, clip/speaker) + `eda_summary.md` (bảng copy thẳng vào report).
- Chỉ dùng pandas + numpy + matplotlib; tự ép numeric cột rỗng; tự bỏ qua cột thiếu (sync/speaker).

**Pipeline curate đầy đủ:** `01_prep_manifest → 02_score_clips → (03_sync_score) → 04_curate → 05_eda`.

## 4. Gỡ loạt hiểu nhầm kỹ thuật

| Hiểu nhầm | Thực tế |
|---|---|
| Dataset 22GB đụng cap 20GB | Cap 20GB là **/kaggle/working (output)**; input dataset read-only, không tính vào |
| Chạy code làm dataset giảm dung lượng | Không. Input bất biến; output curate chỉ vài chục MB |
| Curate tạo bộ clip mới nhỏ hơn | Không — chỉ ra **CSV liệt kê** clip đạt (`file_path` trỏ về clip gốc); không copy/xóa .mp4 |
| Phải hardcode từng batch tier1 (5 batch) | Không — glob `**` đệ quy tự quét mọi batch/thư mục con; chỉ cần trỏ vào gốc tier |
| Local 4050 6GB "không nổi" | Đủ cho curate + fake + feature (model nhỏ, inference). Chỉ **training dài** mới cần cloud |

## 5. Phân biệt Sync vs Option A (hay nhầm)

| | Sync (`03_sync_score.py`) | Option A (Label design + leakage audit) |
|---|---|---|
| Bản chất | bước **chạy code đo** (SyncNet) | **mục viết** trong report |
| Cần SyncNet/GPU? | Có (dễ fail khi tải model) | Không |
| Ra cái gì | cột số `sync_conf` | bảng + lập luận |
| Vai trò sync | đo thực tế → histogram | chứng minh bằng *cách dựng* |

→ Sync là **tùy chọn**, fail thì bỏ, `04`/`05` tự chạy không cần nó.

## 6. Chốt nơi chạy từng bước

| Việc | Chạy ở | Lý do |
|---|---|---|
| curate (score/cluster/eda) | **local** | data sẵn, model nhỏ, output nhỏ |
| sinh fake (`01_temporal_desync`) | **local** | chỉ ffmpeg `-c:v copy`, không GPU, output lớn (tránh cap 20GB) |
| feature extraction | **local** | inference nhẹ, 4050 đủ, output lớn dùng tại chỗ |
| sync (SyncNet) | Kaggle / skip | Windows khó setup, không bắt buộc cho Report 2 |
| training dài | **cloud** | chỗ duy nhất 6GB hụt hơi |

Lưu ý local: dùng **venv dự án** (tránh xung đột numpy Anaconda); cài `insightface onnxruntime-gpu opencv-python pandas scikit-learn matplotlib`; nhớ override `--out_dir` (mặc định đang `/kaggle/working`).

---

## Việc tiếp theo

- Chạy `01→02→04→05` (local) ra `tier1_scored_all.csv`, `all_clean.csv`, `eda_figs/`.
- Viết mục **Label design + leakage audit** vào report (Option A — phân tích, không sinh fake).
- (Sau hạn) pilot: sinh fake subset → sync → histogram real vs fake; thu thập thêm theo lỗ hổng EDA chỉ ra.
