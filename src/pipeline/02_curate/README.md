# 02_curate — Lọc & chuẩn hóa clip (Data Cleaning + EDA)

Stage này biến tập clip thô (sau bước cắt) thành **tập sạch + metadata + biểu đồ EDA**.
Triết lý: *đo mọi thứ, chỉ loại rác rõ ràng, giữ phần còn lại làm metadata; calibrate
trước khi đặt ngưỡng*. Curation chỉ chạy trên real nguồn; mọi fake sinh sau kế thừa quyết định
của source real, không được gate riêng (chống leakage).

Chạy tuần tự `01 → 02_scoring → (03_diagnostics_optional) → 04 → 05`. Calibration Active-Speaker
nằm trong `02_scoring/02_active_speaker/`. Mặc định đã wire path local; chạy bằng
`D:\Anaconda\envs\vn_av_df\python.exe <file>` từ thư mục gốc dự án (xem docstring đầu mỗi file để biết tham số
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
- `runs/<run_id>/`: shard và output hợp nhất bất biến của temporal active-speaker scoring.
- `assignments/v3/`: workload cố định cho từng reviewer; calibration chung, primary không
  chồng lặp.
- `calibration/`: kết quả thử ngưỡng SyncNet, không phải dữ liệu train.
- `logs/`: log console của các lần chạy curation/preview; có thể tái sinh và không commit.
- `eda_figs/` và `roi_preview/`: artifact trực quan phục vụ phân tích/review.

---

## Bố cục code

```text
02_curate/
├── 01_prep_manifest.py
├── 02_scoring/
│   ├── 01_face_quality.py
│   └── 02_active_speaker/
│       ├── 00_build_candidate_pool.py
│       ├── 01_score.py
│       ├── 02_merge_shards.py
│       ├── 03_export_laser_requests.py
│       ├── 04_run_laser.py
│       ├── 05_apply_laser_scores.py
│       ├── 06_build_calibration_manifest.py
│       ├── 07_calibrate.py
│       └── policy.py
├── 03_diagnostics_optional/
│   ├── 01_motion_score.py
│   └── 02_sync_score.py
├── 04_curate.py
└── 05_eda.py
```

Các file có số là bước chạy trực tiếp. `policy.py` là logic dùng chung, không phải một job độc lập.
Thư mục `03_diagnostics_optional` chỉ phục vụ phân tích; motion/SyncNet không được dùng làm auto gate chính.

## Các file

### 01_prep_manifest.py — gộp tier + verify
Đọc `accepted_clips.csv` của mọi batch làm nguồn chân lý, đối chiếu 1–1 với `.mp4`,
**verify từng file bằng ffprobe**, remap `file_path` về đĩa thật rồi gộp ba tier.
Thiếu media, media mồ côi, ID trùng hoặc file hỏng đều làm bước này dừng.
- **Input:** `data/01_collect/cut_clips/{tier1,tier2,tier3}/**/accepted_clips.csv`
  và media tương ứng
- **Output:** `data/01_collect/cut_clips/all_manifest.csv` (atomic, mặc định không ghi đè)

### 02_scoring/01_face_quality.py — đo mặt + embedding (GPU)
Mỗi clip lấy 9 frame rải đều, chạy **InsightFace (buffalo_l)** đo `det_ratio`,
`mean_face_area`, `embed_consistency` và trích **face embedding 512 chiều**. Không loại
clip nào — chỉ đo.
- **Input:** `data/01_collect/cut_clips/all_manifest.csv`
- **Output:** `data/02_curate/measurements/tier1_scored_all.csv` (manifest + face stats),
  `data/02_curate/measurements/embeddings_all.npy` (N×512, đã L2-norm). Hai output
  được publish atomic và mặc định không ghi đè.

### 03_diagnostics_optional/02_sync_score.py — đo khớp môi-tiếng (SyncNet, tùy chọn)
Dùng **SyncNet** (repo joonson/syncnet_python) đo độ đồng bộ môi-tiếng: `LSE-C`
(confidence, cao = khớp) và `LSE-D` (min dist, thấp = khớp). Hai chế độ: `--calibrate`
(xem phân bố) và `--full` (thêm cột `sync_conf`). **Tùy chọn** — chậm trên CPU và phải
tránh siết sync cho tập real; 04/05 chạy không cần nó.
- **Input:** `data/02_curate/measurements/tier1_scored_all.csv` + `--syncnet_dir <repo>`
- **Output:** `data/02_curate/calibration/calibrate_sync_results.csv` (calibrate) hoặc CSV có thêm
  `sync_conf`/`sync_min_dist` (full)

### 02_scoring/02_active_speaker/ — đo active speaker theo thời gian

`00_build_candidate_pool.py` chọn trước 750 clip, gồm 250 clip mỗi tier và tối đa một
clip cho mỗi `source_video`. Thứ tự chọn dựa trên SHA-256 của seed/source/clip nên không
phụ thuộc thứ tự dòng trong manifest. Pool này chỉ là population để preliminary scoring,
không phải nhãn calibration và không phải tập train.

- **Input:** `data/01_collect/cut_clips/all_manifest.csv`.
- **Output bất biến:** `data/02_curate/calibration/active_speaker_candidate_pool_750_v1.csv`
  và file `_summary.json` chứa hash input/output, seed và kiểm tra source-disjoint.

Giải mã audio tạm thành mono 16 kHz, chuẩn hóa PCM rồi chạy Silero ONNX từ checkout đã pin theo bin 200 ms, track nhiều mặt bằng
InsightFace, ổn định vùng miệng theo năm landmark rồi chạy Light-ASD. Bin mơ hồ được đánh dấu
`laser_requested`; `04_run_laser.py` chạy checkpoint LoCoNet+LASER chính thức và sidecar
được ghép fail-closed bằng `05_apply_laser_scores.py` thành một run mới. Thiếu bằng chứng cần thiết thì clip vào manual,
không auto-reject. Các script không ghi đè video.

Light-ASD chỉ nhận face-track từ 6 frame. Ở 25 fps, track 5 frame chưa tạo đủ 20 bước
MFCC cho tối thiểu 5 output model; track phụ quá ngắn được bỏ riêng, không làm exception
toàn clip. Nếu một bin có speech và có detection nhưng không còn track đủ dài, bin đó ghi
`inference_failure=true` và clip được chuyển manual.

`03_export_laser_requests.py` xuất đúng các bin cần chạy cùng `source_timeline_sha256`.
`04_run_laser.py` kiểm tra Git SHA của checkout, SHA-256 checkpoint, chạy CUDA rồi trả
JSONL `{clip_id, bin_index, laser_score}` với score là xác suất active lớn nhất trong bin.
Runner dùng face-track tối thiểu 5 frame, tương ứng trọn một bin 200 ms; điều kiện này khác
ngưỡng 20 frame mang tính lọc track của demo full-video upstream. Không ghép face `track_id`
giữa hai pipeline độc lập.

- **Input:** `all_manifest.csv`, checkout/weights Light-ASD đã pin, checkout Silero đã pin có ONNX 16 kHz, tùy chọn sidecar LASER.
- **Output bất biến:** `data/02_curate/runs/<run_id>/` hoặc
  `runs/<run_id>/shards/<shard_id>/` gồm `asd_clip_scores.csv`,
  `asd_timeline.jsonl.gz`, `failures.csv`, `run_config.json`.
- **Merge:** `02_merge_shards.py` chỉ publish khi config hash đồng nhất và score +
  timeline phủ chính xác 100% manifest.

Chuỗi chạy LASER selective sau khi đã có một base run hợp nhất:

```powershell
# Checkout/weight để ngoài repository; revision và checksum dưới đây là bản đã smoke.
git clone https://github.com/plnguyen2908/LASER_ASD C:\models\LASER_ASD
git -C C:\models\LASER_ASD checkout 3703d3f396cc7b29aa704364f8a9a5ab0c8c1fb9
D:\Anaconda\envs\vn_av_df\python.exe -m gdown 1IrntlKqzw5EYAVbyDupr5tk-H3q9kkoW -O C:\models\LoCoNet_LASER.model
# SHA-256 checkpoint phải là 1702df2cc9a6976e4193dcd78d468d3a0f3afc7a926891e0376cc6d2ea72cc1f.

D:\Anaconda\envs\vn_av_df\python.exe src\pipeline\02_curate\02_scoring\02_active_speaker\03_export_laser_requests.py --manifest data\01_collect\cut_clips\all_manifest.csv --timeline data\02_curate\runs\<base_run>\asd_timeline.jsonl.gz --out_dir data\02_curate\runs\<request_run>
D:\Anaconda\envs\vn_av_df\python.exe src\pipeline\02_curate\02_scoring\02_active_speaker\04_run_laser.py --requests_dir data\02_curate\runs\<request_run> --laser_repo C:\models\LASER_ASD --laser_weights C:\models\LoCoNet_LASER.model --laser_weights_sha256 1702df2cc9a6976e4193dcd78d468d3a0f3afc7a926891e0376cc6d2ea72cc1f --laser_revision 3703d3f396cc7b29aa704364f8a9a5ab0c8c1fb9 --out_dir data\02_curate\runs\<laser_run>
D:\Anaconda\envs\vn_av_df\python.exe src\pipeline\02_curate\02_scoring\02_active_speaker\05_apply_laser_scores.py --base_run data\02_curate\runs\<base_run> --laser_scores data\02_curate\runs\<laser_run>\laser_scores.jsonl --laser_metadata data\02_curate\runs\<laser_run>\laser_metadata.json --out_run data\02_curate\runs\<enriched_run>
```

Mọi tên `<...>` phải được thay bằng run ID mới. Các thư mục output là bất biến; không dùng
lại tên đã tồn tại. `04_run_laser.py` cần CUDA. Gói `resampy` là dependency runtime của
VGGish; `gdown` chỉ dùng để tải checkpoint liên kết từ README upstream.

### 02_scoring/02_active_speaker/07_calibrate.py — khóa temporal policy

Sau khi preliminary scoring + LASER enrichment phủ đủ candidate pool,
`06_build_calibration_manifest.py` tạo 450 clip source-disjoint (150/tier), gồm 300 tune và
150 locked validation. Sau khi hai reviewer gán interval theo rubric v3 và người thứ ba phân xử,
`07_calibrate.py` grid-search chỉ trên tune rồi kiểm tra validation. Auto gate chỉ được publish khi recall
static/voice-over/mixed ≥95%, false-reject clean ≤2% toàn bộ và ≤3% từng tier.

### 04_curate.py — temporal gate → face gate → cluster/cân bằng
Gom speaker bằng agglomerative clustering (cosine) → gate loại rác (mặt nhỏ/không
mặt/lẫn người) → cân bằng số clip mỗi speaker → xuất tập sạch. Luôn `--calibrate` trước
để chọn ngưỡng.
- **Input:** `data/02_curate/measurements/tier1_scored_all.csv` +
  `data/02_curate/measurements/embeddings_all.npy`, thêm `--temporal_scores` và
  `--temporal_policy` khi policy đã calibrate.
- **Output:** `data/02_curate/manifests/all_clean.csv` (tập sạch, có `speaker_id`),
  `all_clean_rejects.csv` (clip bị gate loại), `all_clean_balance_dropped.csv`
  (clip bị bỏ khi cân bằng), và `all_clean_config.json` (tham số, SHA-256 input,
  số dòng từng partition). Output atomic và mặc định không ghi đè.
- Temporal decision chỉ là gate nhị phân. Policy chưa đạt validation chỉ được dùng để ưu tiên
  manual; continuous ASD score không đi vào `quality_score`.
- **Baseline lịch sử để calibrate lại:** `cluster_dist=0.6`, `min_det_ratio=0.6`,
  `min_face_area=0.01`, `min_consistency=0.3`, `cap_per_speaker=30`

### 05_eda.py — phân tích khám phá
Đọc scored CSV (+ clean CSV) → xuất biểu đồ PNG rời + bảng markdown tóm tắt cho báo cáo.
- **Input:** `data/02_curate/measurements/tier1_scored_all.csv`
  (+ `data/02_curate/manifests/all_clean.csv`)
- **Output:** `data/02_curate/eda_figs/` (11 biểu đồ `.png` + `eda_summary.md`)

---

## Kết quả lịch sử trước hotfix Stage 04

Phễu làm sạch: **6.888** clip đo được → **5.356** qua gate → **3.001** clip sạch / **674** speaker.
Các số này thuộc cut run có lỗi decode và chỉ được giữ làm provenance; không dùng
làm scope review/train mới.

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

---

## Manual review rubric v3

Reviewer đánh dấu từng khoảng lỗi bằng `start_ms`, `end_ms`, `reason`; reason cấp clip là lý do
của interval dài nhất. Hai reviewer gán 450 clip calibration độc lập, người thứ ba adjudicate.
Biên interval lệch tối đa 200 ms vẫn được xem là đồng thuận.

Không chia trực tiếp bằng vị trí dòng hoặc tự copy CSV. Dùng builder để giữ đúng coverage:

```bash
D:\Anaconda\envs\vn_av_df\python.exe src/tools/review/build_review_assignments.py \
  --manifest data/02_curate/calibration/active_speaker_450_v3.csv \
  --calibration data/02_curate/calibration/active_speaker_450_v3.csv \
  --reviewers <reviewer_1> <reviewer_2> \
  --out_dir data/02_curate/assignments/v3/calibration_450
```

Hai assignment calibration đều chứa đủ 450 clip. Reviewer thứ ba chỉ xử lý file
`needs_adjudication.csv`. Sau khi policy pass và `04_curate` tạo scope manual mới, chạy builder
lần nữa với `--no_shared_calibration --manifest <temporal-pass-review.csv> --reviewers <3 ID>`
để chia primary disjoint cho cả ba người; không trộn 60 nhãn v2 lịch sử vào scope này.

Mỗi người chạy đúng file mang reviewer ID của mình:

```bash
D:\Anaconda\envs\vn_av_df\python.exe src/tools/review/clip_review.py \
  --csv data/02_curate/assignments/v3/assignment_<reviewer>.csv \
  --reviewer <reviewer>
```

`clip_review.py` dừng ngay nếu `--reviewer` không khớp assignment. Output mặc định cũng
chứa reviewer ID, nên không ghi đè kết quả của người khác.

Sau khi đủ hai file kết quả calibration, chạy merger; nếu có bất đồng thì người thứ ba điền
adjudication rồi chạy lại:

```bash
D:\Anaconda\envs\vn_av_df\python.exe src/tools/review/merge_review_results.py \
  --manifest data/02_curate/calibration/active_speaker_450_v3.csv \
  --assignments "data/02_curate/assignments/v3/calibration_450/assignment_*.csv" \
  --results "data/02_curate/manual/manual_assignment_*_v3_*.csv" \
  --out_dir data/02_curate/manual/merged_calibration_v3 \
  --final_clean data/02_curate/calibration/active_speaker_450_keep_v3.csv
```

Nếu thiếu coverage, có `uncertain` hoặc calibration không đồng thuận, script dừng và ghi
`data/02_curate/manual/merged_v3/needs_adjudication.csv`. Điền
`final_decision`, `final_reason`, `final_bad_intervals_json`, `adjudicator`, sau đó chạy lại với
`--adjudication <file>`. Chỉ khi đủ scope đang review và không còn case cần phân xử, script mới
xuất `data/02_curate/manifests/manual_clean_v3.csv` và `consensus_labels_v3.csv`.

Quy trình đầy đủ, trạng thái đã/chưa chạy và giới hạn LASER được ghi tại
[`ACTIVE_SPEAKER_CURATION_IMPLEMENTATION.md`](../../../docs/reports/ACTIVE_SPEAKER_CURATION_IMPLEMENTATION.md).
