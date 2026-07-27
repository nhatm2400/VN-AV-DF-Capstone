# 02_curate — Lọc & chuẩn hóa clip (Data Cleaning + EDA)

Stage này biến tập clip thô (sau bước cắt) thành **tập sạch + metadata + biểu đồ EDA**.
Triết lý: *đo mọi thứ, chỉ loại rác rõ ràng, giữ phần còn lại làm metadata; calibrate
trước khi đặt ngưỡng*. Mọi gate phải áp **đối xứng** real/fake (chống leakage).

Chạy tuần tự `01 → 02 → (03) → 04 → 05`. Mặc định đã wire path local; chạy thẳng
`python <file>` từ thư mục gốc dự án là được (xem docstring đầu mỗi file để biết tham số
và biến thể Kaggle).

---

## Bố cục dữ liệu

`data/02_curate/` được chia theo vai trò để tránh nhầm các bảng có tên gần giống nhau:

- `measurements/`: số đo tự động trên toàn bộ clip, face embedding và các phép đo chẩn
  đoán như `lipcorr.json`; đây chưa phải quyết định giữ/loại.
- `manifests/`: các bảng đầu vào/đầu ra quyết định của curation. `all_clean.csv` là tập code
  giữ, `all_clean_rejects.csv` là tập code loại, còn `all_clean_review.csv` là scope đưa vào
  công cụ review.
- `manual/`: quyết định do người review ghi ra; tách khỏi manifest do code tạo.
- `calibration/`: kết quả thử ngưỡng SyncNet, không phải dữ liệu train.
- `logs/`: log console của các lần chạy curation/preview; có thể tái sinh và không commit.
- `eda_figs/` và `roi_preview/`: artifact trực quan phục vụ phân tích/review.

---

## Các file

### 01_prep_manifest.py — gộp tier + verify
Quét mọi `.mp4` của 3 tier (glob `**` đệ quy, tự bắt mọi batch con), ghép metadata từ
CSV bước cắt, **verify từng file bằng ffprobe** (loại file hỏng 0-byte), remap `file_path`
về đĩa thật, gộp thành 1 manifest.
- **Input:** `data/clips/{tier1,tier2,tier3}/**` (+ các `*_v3_clips_*.csv`)
- **Output:** `data/clips/all_manifest.csv`

### 02_score_clips.py — đo mặt + embedding (GPU)
Mỗi clip lấy 9 frame rải đều, chạy **InsightFace (buffalo_l)** đo `det_ratio`,
`mean_face_area`, `embed_consistency` và trích **face embedding 512 chiều**. Không loại
clip nào — chỉ đo.
- **Input:** `data/clips/all_manifest.csv`
- **Output:** `data/02_curate/measurements/tier1_scored_all.csv` (manifest + face stats),
  `data/02_curate/measurements/embeddings_all.npy` (N×512, đã L2-norm)

### 03_sync_score.py — đo khớp môi-tiếng (SyncNet, tùy chọn)
Dùng **SyncNet** (repo joonson/syncnet_python) đo độ đồng bộ môi-tiếng: `LSE-C`
(confidence, cao = khớp) và `LSE-D` (min dist, thấp = khớp). Hai chế độ: `--calibrate`
(xem phân bố) và `--full` (thêm cột `sync_conf`). **Tùy chọn** — chậm trên CPU và phải
tránh siết sync cho tập real; 04/05 chạy không cần nó.
- **Input:** `data/02_curate/measurements/tier1_scored_all.csv` + `--syncnet_dir <repo>`
- **Output:** `data/02_curate/calibration/calibrate_sync_results.csv` (calibrate) hoặc CSV có thêm
  `sync_conf`/`sync_min_dist` (full)

### 04_curate.py — cluster → gate → cân bằng
Gom speaker bằng agglomerative clustering (cosine) → gate loại rác (mặt nhỏ/không
mặt/lẫn người) → cân bằng số clip mỗi speaker → xuất tập sạch. Luôn `--calibrate` trước
để chọn ngưỡng.
- **Input:** `data/02_curate/measurements/tier1_scored_all.csv` +
  `data/02_curate/measurements/embeddings_all.npy`
- **Output:** `data/02_curate/manifests/all_clean.csv` (tập sạch, có `speaker_id`),
  `data/02_curate/manifests/all_clean_rejects.csv` (clip bị gate loại)
- **Ngưỡng đang dùng:** `cluster_dist=0.6`, `min_det_ratio=0.6`, `min_face_area=0.01`,
  `min_consistency=0.3`, `cap_per_speaker=30`

### 05_eda.py — phân tích khám phá
Đọc scored CSV (+ clean CSV) → xuất biểu đồ PNG rời + bảng markdown tóm tắt cho báo cáo.
- **Input:** `data/02_curate/measurements/tier1_scored_all.csv`
  (+ `data/02_curate/manifests/all_clean.csv`)
- **Output:** `data/02_curate/eda_figs/` (11 biểu đồ `.png` + `eda_summary.md`)

---

## Kết quả (lần chạy gần nhất)

Phễu làm sạch: **6.888** clip đo được → **5.356** qua gate → **3.001** clip sạch / **674** speaker.

| File trong `data/02_curate/` | Mô tả |
|---|---|
| `measurements/tier1_scored_all.csv` | 6.888 dòng — manifest + face stats |
| `measurements/embeddings_all.npy` | (6888, 512) face embedding |
| `manifests/all_clean.csv` | 3.001 clip sạch (có `speaker_id`) |
| `manifests/all_clean_rejects.csv` | 1.532 clip bị gate loại |
| `calibration/calibrate_sync_results.csv`, `calibration/sync_calibrate_log.txt` | mẫu calibrate sync (30 clip) |
| `eda_figs/` | 11 biểu đồ + `eda_summary.md` |

---

## Môi trường

Conda env `vn_av_df` (Python 3.10): `insightface`, `onnxruntime-gpu` (+ `nvidia-cudnn-cu12==9.8.0.87`
để chạy GPU trên Windows), `opencv-python`, `pandas`, `scikit-learn`, `matplotlib`,
`ffmpeg`/`ffprobe` trong PATH. Bước 03 cần thêm `scenedetect`, `python_speech_features` và
repo SyncNet + model (`syncnet_v2.model`, `sfd_face.pth`).
