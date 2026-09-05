# PROJECT.md — VN-AV-DF-Capstone

## Tổng quan dự án

Dự án phát hiện **Deepfake âm thanh-hình ảnh tiếng Việt** (Vietnamese Audio-Visual Deepfake Detection). Mục tiêu là xây dựng dataset và huấn luyện mô hình phát hiện video giả mạo (deepfake) đặc thù cho người Việt, tập trung vào sự lệch pha giữa âm thanh và khẩu hình miệng.

Mô hình hiện đã chạy pilot: **AVSP-Net V1** — mouth ROI + Wav2Vec + prosody, hợp nhất bằng Cross-Attention. Kiến trúc mục tiêu mới là **AVSP-Net V2**, gồm V2a (local temporal core, giữ cùng loại feature nhưng repaired pilot dùng store mới) và V2b (mở rộng để tổng quát hóa sang deepfake thực tế), xem [MODEL_PROPOSAL.md](docs/architecture/MODEL_PROPOSAL.md).

Trạng thái hiện tại: **Stage 04 Cut Clips đã dựng lại xong và đạt Gate C; P4.1 `all_manifest.csv` đã dựng đủ 65.622 clip; candidate pool 750 clip đã preliminary-score và enrich bằng Light-ASD + LoCoNet/LASER với coverage 100%; manifest calibration 450 clip source-disjoint đã tạo xong và đang chờ một reviewer duy nhất gán nhãn rubric v3**. Light-ASD phủ `750/750` clip và `16.039` bin; LASER chấm đủ `3.856/3.856` bin được yêu cầu, không thiếu score. Một clip lỗi Light-ASD được chuyển `manual/inference_failure` đúng fail-closed. Calibration gồm 150 clip/tier, chia 300 tune + 150 locked validation và không thiếu media. Đây vẫn chỉ là kiểm tra kỹ thuật, chưa phải bằng chứng accuracy. Ba quality manifest gồm `472/292/2.274 = 3.038` video nguồn đều có terminal status. Output có `65.622` accepted clip khớp 1–1 với `65.622` MP4, không thiếu/mồ côi/trùng `clip_id`; `120.187` cửa sổ bị reject có lý do. Tổng media sau khi downscale trực tiếp 1.200 clip 4K xuống 1080p là `155,277 GiB`. Auto gate vẫn **NO-GO** cho đến khi reviewer hoàn tất 450 clip, policy đạt locked validation và smoke GPU 300 clip đủ ba tier; sinh fake, extract feature và train cũng chưa được mở. Ba file `all_clean*.csv` còn lại trong `data/02_curate/manifests/` thuộc population cũ `6.888 → 3.001`, chỉ có giá trị lịch sử. Xem [triển khai Active-Speaker](docs/reports/ACTIVE_SPEAKER_CURATION_IMPLEMENTATION.md), [kế hoạch hotfix](docs/reports/CUT_CLIPS_HOTFIX_AND_REBUILD_PLAN.md) và [nhật ký Stage 04](docs/logs/2026-08-29_STAGE04_REBUILD_AND_STORAGE_NORMALIZATION.md).

Các kết quả lịch sử vẫn được giữ nguyên: **03_fake V1 từng sinh 12.004 fake**, **PILOT V1 đã chạy xong** trên 2.700 clip với test AUC **0.809**, bốn generator V2/SNVSM/timeline contract đã implement/test, và stratified smoke `v2r6` đã đạt metadata gate max AUC 0,546. Các con số này không chứng minh tập nguồn hiện tại đã sạch và không được dùng để bỏ qua hotfix. Sau khi cut → curate → manual review được dựng lại, lộ trình tiếp tục từ fake V2/SNVSM/Stage 05/metadata gate đến repaired pilot V2a; full training vẫn **NO-GO**. Xem [báo cáo Phase 0](docs/reports/TEMPORAL_DESYNC_PHASE0_SMOKE.md) và [đánh giá V1/V2](docs/reports/PILOT_V1_REVIEW_AND_V2_PLAN.md).

---

## Cấu trúc thư mục

> Đã sắp xếp lại theo stage đánh số cho cả `src/pipeline/` lẫn `data/` (01_collect → 02_curate → 03_fake → 04_features). `data/` và `src/pipeline/` khớp số với nhau.

```
.
├── src/
│   ├── pipeline/                       # Pipeline xử lý dữ liệu (stage đánh số)
│   │   ├── 01_collect/                 # Thu thập + cắt clip — tách theo tier (nguồn)
│   │   │   ├── 04_cut_clips.ipynb      # Driver Kaggle canonical dùng chung ba tier
│   │   │   ├── cut_clips_core.py       # Core Stage 04 có fallback + contract test
│   │   │   ├── configs/                # Config tier1/tier2/tier3; placeholder fail-closed nếu chưa khóa
│   │   │   ├── tier1/                  # YouTube CC: fetch, download, quality gate
│   │   │   ├── tier2/                  # YouTube Std: fetch, download/retry/cleanup
│   │   │   └── tier3/                  # TikTok: fetch, download, quality gate
│   │   ├── 02_curate/                  # Lọc clip tự động (dùng chung mọi tier)
│   │   │   ├── 01_prep_manifest.py     # remap path + gộp tier -> all_manifest.csv
│   │   │   ├── 02_scoring/             # các phép đo bắt buộc, không tự loại clip
│   │   │   │   ├── 01_face_quality.py  # mặt + embedding 512-d (InsightFace buffalo_l)
│   │   │   │   └── 02_active_speaker/  # VAD + face track + Light-ASD/LASER
│   │   │   │       ├── 00_build_candidate_pool.py
│   │   │   │       ├── 01_score.py
│   │   │   │       ├── 02_merge_shards.py
│   │   │   │       ├── 03_export_laser_requests.py
│   │   │   │       ├── 04_run_laser.py
│   │   │   │       ├── 05_apply_laser_scores.py
│   │   │   │       ├── 06_build_calibration_manifest.py
│   │   │   │       ├── 07_calibrate.py
│   │   │   │       └── policy.py       # policy thuần, dùng chung và test độc lập
│   │   │   ├── 03_diagnostics_optional/ # motion/SyncNet chỉ để chẩn đoán
│   │   │   │   ├── 01_motion_score.py
│   │   │   │   └── 02_sync_score.py
│   │   │   ├── 04_curate.py            # cluster speaker -> gate rác -> cân bằng
│   │   │   ├── 05_eda.py               # thống kê + xuất eda_figs/
│   │   │   └── README.md
│   │   ├── 03_fake/                    # Sinh pseudo-fake (4 method, cùng schema labels.csv)
│   │   │   ├── 01_temporal_desync.py   # lệch pha audio-visual 3/7/15 frames (video copy)
│   │   │   ├── 02_frame_reverse.py     # đảo ngược 0.3–1.0s video (audio copy) — visual thuần
│   │   │   ├── 03_pitch_flatten.py     # làm phẳng F0 (video copy) — audio thuần, parselmouth
│   │   │   ├── 04_anonymization.py     # blur/pixelate mặt (audio copy) — YOLOv8n-face
│   │   │   ├── 05_snvsm_compress.py    # nén H.264 + AAC đối xứng real+fake
│   │   │   └── 06_build_fake_manifest_v2.py # thay temporal V1; chưa chứng nhận timing 3 method cũ
│   │   ├── 04_extract_features/
│   │   │   └── 01_extract_features.py  # mouth-ROI (YOLO) + wav2vec2 + prosody F0 -> .pt/clip
│   │   └── 05_build_labels/
│   │       └── 01_build_labels.py      # gộp real+fake -> labels.csv + split SPEAKER-DISJOINT
│   ├── tools/
│   │   ├── review/                     # manifest, preview, assignment, UI và merge review
│   │   │   ├── build_review_manifest.py
│   │   │   ├── build_review_assignments.py
│   │   │   ├── build_roi_preview.py
│   │   │   ├── export_review_batch.py
│   │   │   ├── clip_review.py
│   │   │   └── merge_review_results.py
│   │   ├── diagnostics/                # phép đo thử nghiệm, không phải gate chính
│   │   │   ├── scan_face_ambiguity.py
│   │   │   └── measure_lip_audio_corr.py
│   │   └── data_admin/                 # download, recovery và snapshot provenance
│   │       ├── download_data.py
│   │       ├── recover_cut_input_inventory.py
│   │       └── snapshot_cut_hotfix_baseline.py
│   ├── model/
│   │   └── avsp_net.py                 # AVSP-Net: mouth-CNN+Transformer, prosody BiGRU, cross-attn, 2 head
│   ├── train/
│   │   ├── dataset.py                  # AVSPDataset: labels.csv + .pt -> tensor cố định
│   │   └── train.py                    # training loop (--branches cho ablation), best theo val AUC
│   ├── eval/
│   │   └── evaluate.py                 # acc/P/R/F1/AUC + method-wise + confusion -> eval_<split>.json
│   ├── utils/                          # (placeholder test.py)
│   └── pre-testing/                    # (trống) — thử nghiệm nhanh
├── data/                               # (KHÔNG commit phần nặng: raw, cut_clips, fake, features)
│   ├── 01_collect/
│   │   ├── youtube_tier1_urls.csv, youtube_tier2_urls.csv
│   │   ├── tier1_quality_gate_passed.csv, tier2_quality_gate_passed.csv
│   │   └── cut_clips/                  # Stage 04 mới: 65.622 MP4 + metadata/checksum theo batch
│   │       ├── tier1/ tier2/ tier3/    # accepted_clips.csv, video_status.csv, SHA256SUMS, media/
│   │       └── all_manifest.csv        # đã dựng đủ 65.622 clip; input canonical P4
│   ├── 02_curate/
│   │   ├── measurements/               # P4 sẽ dựng lại score/embedding cho population mới
│   │   ├── manifests/                  # bảng quyết định của curation
│   │   │   └── all_clean*.csv          # 3 manifest lịch sử; P4 sẽ thay bằng output mới
│   │   ├── calibration/                # P4 sẽ tạo lại nếu tiếp tục dùng SyncNet
│   │   ├── eda_figs/                   # P4 sẽ tạo lại
│   │   └── manual/                     # P5/manual review sẽ tạo version mới
│   ├── 03_fake/                        # output fake .mp4 + labels.csv (hiện trống, .gitkeep)
│   ├── 04_features/                    # output feature .pt + features_index.csv (hiện trống)
│   ├── 05_labels/                      # labels.csv thống nhất real+fake + cột split
│   ├── raw/tier{N}/                    # video gốc tải về
│   └── label.csv                       # nhãn legacy (từ PoC)
├── configs/                            # (placeholder test.py)
├── docs/                               # Tài liệu kiến trúc, báo cáo, proposal dữ liệu và tài liệu tham khảo
│   ├── architecture/
│   │   └── MODEL_PROPOSAL.md           # AVSP-Net V2a/V2b + roadmap + output contract
│   └── reports/
│       ├── PILOT_REPORT.md             # Báo cáo pilot gốc
│       ├── PILOT_V1_REVIEW_AND_V2_PLAN.md
│       └── TEMPORAL_DESYNC_PHASE0_SMOKE.md
├── experiments/exp001_baseline/
├── notebooks/                          # (trống)
├── README.md
├── yolov8n-face.pt                     # Đặt ở root — quality gate + anonymization
└── requirements.txt
```

---

## Kiến trúc mô hình

Code hiện hành dùng **AVSP-Net V1** trong `src/model/avsp_net.py`: mouth ROI, Wav2Vec tiếng Việt và prosody hợp nhất bằng cross-attention. Pilot cho thấy global fusion/offset loss chưa đủ, nên kiến trúc mục tiêu là V2a/V2b trong [MODEL_PROPOSAL.md](docs/architecture/MODEL_PROPOSAL.md). Thư mục proof-of-concept PAMF cũ đã được dọn khỏi repository; số đo pilot V1 vẫn được giữ trong báo cáo và run bất biến dưới `experiments/`.

---

## Pipeline dữ liệu

Chạy theo đúng thứ tự từ **root repo**. Stage `data/` khớp số với `src/pipeline/`.

### 01_collect — Thu thập & cắt clip

**Fetch URLs (theo tier):**
```bash
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/01_collect/tier1/01_fetch_youtube_urls.py   # tier1 (YouTube CC)
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/01_collect/tier2/01_fetch_youtube_urls.py   # tier2 (YouTube Std)
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/01_collect/tier3/01_fetch_tiktok_urls.py    # tier3 (TikTok)
```
Yêu cầu `YOUTUBE_API_KEY` trong `.env`. Script ghi mặc định ra `data/…_urls.csv`; output cuối đã gom vào `data/01_collect/`.

**Download:**
```bash
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/01_collect/tier{N}/02_download.py     # yt-dlp -> data/raw/tier{N}/
```

**Quality Gate** (`03_quality_gate.py`) — video phải pass cả 3:
| Tiêu chí | Ngưỡng |
|---|---|
| Độ phân giải | ≥ 480p |
| FPS | ≥ 24 |
| Luồng âm thanh | Có (đọc bằng ffprobe) |

Output: `data/01_collect/tier{N}_quality_gate_passed.csv`. (Ngưỡng SNR/face trong tài liệu là chủ trương thiết kế; code hiện gate độ phân giải + FPS + có audio.)

**Cắt clip** — `04_cut_clips.ipynb` gọi core dùng chung `cut_clips_core.py` trên Kaggle GPU. Pipeline cắt theo VAD (Silero), kiểm scene/face/speech, có CUDA→CPU decode fallback và NVENC→libx264 cut fallback; **đo SNR nhưng KHÔNG gate** (`min_snr=-999`). Mỗi batch bất biến chứa `accepted_clips.csv`, `rejected_windows.csv`, `video_status.csv`, `run_summary.json`, config/environment, `SHA256SUMS` và `media/`. Stage 04 mới đã hoàn tất `3.038` input → `65.622` accepted clip; 14 nguồn 4K được downscale trực tiếp thành 1080p sau cut, audio stream-copy và checksum hiện hành đã cập nhật.

### 02_curate — Curation tự động (`src/pipeline/02_curate/`)

Lọc population mới 65.622 clip trước manual review. Triết lý: **đo mọi thứ, chỉ loại rác rõ ràng, giữ phần còn lại làm metadata; calibrate trước khi đặt ngưỡng**. Temporal Active-Speaker được đặt sau `all_manifest.csv` và trước `04_curate.py`; không recut media.

```bash
# 1) Gộp accepted CSV ba tier + remap file_path về đĩa thật + verify ffprobe
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/02_curate/01_prep_manifest.py
# -> data/01_collect/cut_clips/all_manifest.csv; kỳ vọng 65.622 dòng
# 2) Đo mặt + embedding (GPU, InsightFace) -> data/02_curate/measurements/
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/02_curate/02_scoring/01_face_quality.py \
    --input_csv data/01_collect/cut_clips/all_manifest.csv --tag all
# 3) Tạo candidate pool 750 clip: 250/tier, đúng một clip/source
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/02_curate/02_scoring/02_active_speaker/00_build_candidate_pool.py
# 3a) Temporal ASD: VAD 200 ms + face track + mouth motion + Light-ASD;
#     chạy trên candidate pool, chưa chạy full khi policy chưa pass.
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/02_curate/02_scoring/02_active_speaker/01_score.py ...
# 3b) Xuất request, chạy LoCoNet+LASER cho đúng các bin mơ hồ, rồi enrich run
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/02_curate/02_scoring/02_active_speaker/03_export_laser_requests.py ...
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/02_curate/02_scoring/02_active_speaker/04_run_laser.py ...
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/02_curate/02_scoring/02_active_speaker/05_apply_laser_scores.py ...
# 4) Tạo 450 clip source-disjoint; 300 tune + 150 locked validation; một reviewer làm rubric v3
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/02_curate/02_scoring/02_active_speaker/06_build_calibration_manifest.py ...
# 5) Grid-search tune, sau đó gate validation khóa
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/02_curate/02_scoring/02_active_speaker/07_calibrate.py ...
# Chẩn đoán khớp môi-tiếng (SyncNet) — tùy chọn, không phải quality gate chính
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/02_curate/03_diagnostics_optional/02_sync_score.py \
    --input_csv data/02_curate/measurements/tier1_scored_all.csv \
    --syncnet_dir <repo> --calibrate
# 6) Chỉ sau policy pass: temporal gate -> face gate -> cân bằng
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/02_curate/04_curate.py \
    --temporal_scores data/02_curate/runs/<run_id>/asd_clip_scores.csv \
    --temporal_policy data/02_curate/calibration/<policy>.json
# 7) EDA: thống kê + xuất data/02_curate/eda_figs/
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/02_curate/05_eda.py
```

**Ngưỡng chính:** cluster_dist=0.6, min_det_ratio=0.6, min_face_area=0.01, min_consistency=0.3, cap_per_speaker=30; `quality_score = 0.4·det + 0.3·norm(face_area) + 0.15·consistency` (chuẩn hóa p5–p95).

⚠️ **Chống leakage:** active-speaker gate chỉ chạy trên real nguồn trước khi sinh fake. Fake sinh sau đó kế thừa đúng quyết định của `source_clip`; không chạy ASD riêng trên fake và không đưa curation score vào feature model. Đây là cách áp quyết định đối xứng theo cặp real–fake. Các script inference full chạy trên **Kaggle GPU**.

**Lọc tay (BẮT BUỘC trước khi sinh fake):** 60 clip rubric v2 lịch sử từng loại **34/60** — chủ yếu `dubbed` (14), `cut` (7), `static` (6), cho thấy gate cũ false-accept nghiêm trọng. Mẫu này chỉ là development seed. Calibration mới có 450 clip rubric v3 và do một reviewer duy nhất gán nhãn toàn bộ; nhãn `uncertain` phải được chính reviewer xem lại và chốt trước calibration.

```bash
D:\Anaconda\envs\vn_av_df\python.exe src/tools/review/build_review_manifest.py   # 1) gộp số đo -> manifests/all_clean_review.csv
D:\Anaconda\envs\vn_av_df\python.exe src/tools/review/build_roi_preview.py       # 2) dựng ô ROI+tiếng cho all_clean mới
D:\Anaconda\envs\vn_av_df\python.exe src/tools/review/clip_review.py             # 3) mở http://127.0.0.1:8000
```

`clip_review.py` rubric v3 cho phép đặt đầu/cuối từng interval lỗi và chọn `static`, `voiceover`, `wrong_face`, `dubbed`...; output thêm `bad_intervals_json`. Vì chỉ có một reviewer, không có bước consensus/adjudication; mọi nhãn `uncertain` phải được reviewer mở lại và sửa thành `keep` hoặc `reject` trước khi khóa nhãn.

**Đã thử và thất bại** (đừng làm lại): `motion_median` và số khuôn mặt không dự báo được keep/reject (60% vs 55%); tương quan pixel-motion ROI với audio RMS cho **AUC 0,544**. Lọc theo kênh chỉ nên dùng để **phân tầng/ưu tiên**, không auto-reject — mẫu theo kênh quá nhỏ (Fisher exact p≈0,12 cho kênh tệ nhất).

⚠️ Chỉ loại **rác rõ ràng**, KHÔNG lọc theo chất lượng lip-sync. Real bị loại thì loại **cả cụm** (1 real + 4 fake sinh từ nó).

### 03_fake — Sinh pseudo-fake (`src/pipeline/03_fake/`)

V1 lịch sử đã sinh bốn method vào `data/03_fake/labels.csv`; không dùng lại cho repaired pilot. Bốn generator V2 ghi media/manifest versioned riêng dưới `data/03_fake/*_v2` và `data/03_fake/manifests/v2/`. Builder chỉ nhận đúng generator version + timeline contract V2 của cả bốn method, không còn lấy ba method từ manifest V1.

> **`data/03_fake/` hiện trống (chỉ còn `.gitkeep`).** Media/manifest Fake V1 và các
> smoke Phase 0 cũ đã được dọn khỏi repository; báo cáo kết quả vẫn còn trong `docs/`,
> còn checkpoint pilot V1 nằm trong `experiments/`. Repaired pilot sẽ ghi mới vào
> `data/03_fake/` sau khi P4/P5 và manual review hoàn tất.

```bash
# Mặc định V2: media -> data/03_fake/temporal_v2/
#               manifest -> data/03_fake/manifests/v2/temporal_desync.csv
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/03_fake/01_temporal_desync.py

# Tạo master composition không còn temporal V1; mọi media V2 phải qua paired contract
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/03_fake/06_build_fake_manifest_v2.py
# -> data/03_fake/manifests/v2/fake_all.csv
```

| Method | Kênh tấn công | Cơ chế | Stream giữ nguyên | Phụ thuộc thêm |
|---|---|---|---|---|
| `01_temporal_desync` | timing | V2 xoay audio đúng sample ±3/±7/±15 frame; circular-wrap có valid range | video copy | ffmpeg |
| `02_frame_reverse` | visual-motion | đảo ngược cửa sổ 0.3–1.0s video | **audio copy** | ffmpeg |
| `03_pitch_flatten` | audio-prosody | làm phẳng F0 (PSOLA, đặc thù tiếng Việt) | **video copy** | **parselmouth** |
| `04_anonymization` | visual-identity | blur ≥51px / pixelate vùng mặt | **audio copy** | ultralytics + cv2 |

⚠️ **Bốn lưu ý bắt buộc trước khi train:**
1. **Đồng bộ codec (SNVSM):** `02`/`04` re-encode video → khác codec với real. `05_snvsm_compress.py` V2 nén 4 mức CRF (23/30/35/40) và encode audio AAC 128k, **16 kHz mono đối xứng cả real+fake**, trước khi trích feature để giảm shortcut codec/sample-format. Việc cùng output format không chứng minh xóa hết dấu vết transcode trước đó. Output V1 cũ chưa có normalization audio mới; phải chạy vào path versioned mới.
2. **Anonymization có hai vấn đề tách biệt:** lỗi anon bị drop vì YOLO không thấy mặt **đã xử lý** bằng chuỗi box của REAL ghép cặp. Leakage "mờ = fake" mới chỉ **được giảm thiểu, chưa được chứng minh đã hết**: `train.py --real_blur_aug_p` (mặc định 0.25, chỉ ở train) blur mouth-ROI một phần real, nhưng phân phối blur augmentation chưa chắc khớp anonymization thật và vẫn phải kiểm tra bằng trivial baseline/ablation.
3. **Temporal V1 không dùng lại:** fake cũ có artifact `-itsoffset/-shortest`. Generator V2 dùng sample-exact circular shift + ALAC; manifest ghi `audio_valid_*` và `visual_valid_*`. V2a phải loại wrap khỏi mọi local/global evidence, đồng thời dùng cửa sổ cố định hoặc edge-mask đối xứng cho mọi nhãn để độ dài mask không trở thành shortcut.
4. **Ba method V1 có timing artifact:** `frame_reverse`, `pitch_flatten`, `anonymization` đều từng dùng `-shortest`. Code V2 đã bỏ cơ chế này, publish atomic và fail nếu frame/FPS/video-duration/audio-target lệch source. Synthetic smoke và stratified smoke thật `v2r6` đều đạt; media V1 vẫn không được dùng lại cho repaired pilot.

### 03_fake/05 — SNVSM đồng bộ codec (chạy sau 4 method, trước 04)

```bash
# REAL và FAKE cùng --crfs/--preset -> H.264 + AAC 128k/16 kHz/mono cùng pipeline
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/03_fake/05_snvsm_compress.py --input_csv data/02_curate/manifests/all_clean.csv \
    --out_dir data/03_fake/snvsm_v2/real --out_manifest data/03_fake/snvsm_v2/real_snvsm.csv
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/03_fake/05_snvsm_compress.py --input_csv data/03_fake/manifests/v2/fake_all.csv \
    --out_dir data/03_fake/snvsm_v2/fake --out_manifest data/03_fake/snvsm_v2/fake_snvsm.csv
# rồi 04/05 trỏ vào manifest SNVSM:
#   ...05_build_labels/01_build_labels.py --real_csv .../real_snvsm.csv --fake_labels .../fake_snvsm.csv
#   ...04_extract_features/01_extract_features.py --real_csv .../real_snvsm.csv --fake_labels .../fake_snvsm.csv
```

`--mode random` (mặc định): 1 CRF ngẫu nhiên/clip (×1 dung lượng). `--mode all`: đủ 4 mức (×4, augmentation tối đa). Manifest giữ nguyên mọi cột (speaker_id, source_video…) nên split speaker-disjoint ở 05 vẫn đúng.

SNVSM V2 thêm version, config hash, encoder/preset/audio và CRF policy (`crf_set`, `mode`, `seed`) vào manifest/ID `<clip_id>_snvsmv2_<config-id>_crf<N>`. Mode random khóa CRF theo real nguồn để real + bốn fake ghép cặp có cùng chất lượng nén. Manifest còn giữ `snvsm_target_samples`, pair key và visual contract (frame/FPS/duration). Stage 04 trim trailing AAC padding về đúng target trước feature; resume kiểm decoded PCM cùng visual contract, encode qua file tạm rồi atomic-replace và CLI fail khác 0 nếu thiếu row. Cây `data/03_fake/snvsm/` V1 được guard không cho V2 ghi vào.

### 04_extract_features — Trích feature 3 nhánh

Theo data contract V1 đã triển khai: **KHÔNG dùng full-frame** (giảm leak identity/background) — mỗi clip (real + fake) trích 1 file `.pt` vào `data/04_features/`:
- `mouth`: uint8 `[T,96,96]` — YOLOv8n-face detect, crop nửa dưới bbox (vùng miệng), ~25fps. **Mỗi sampled-frame LUÔN có 1 ROI** (carry-forward khi detect fail giữa clip, backward-fill khi fail ở đầu) → chuỗi hình không co/lệch với audio. **ANON** (mặt mờ, YOLO fail ~18-25%): dùng chuỗi box của **REAL ghép cặp** (`source_clip`→`orig_clip_id`, cache khi xử lý real trước) áp lên anon theo timestamp — crop môi chặt, không phụ thuộc detect trên mặt mờ, không tạo shortcut "static-crop = anon".
- `w2v`: float16 `[T,768]` — wav2vec2-base-vietnamese-250h frozen (tắt bằng `--no_w2v`)
- `prosody`: float32 `[T,4]` — f0_z, delta_f0, energy_z, voiced @100Hz (parselmouth, fallback librosa.pyin)

```bash
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/04_extract_features/01_extract_features.py            # full (GPU khuyến nghị)
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/04_extract_features/01_extract_features.py --limit 5 --no_w2v   # test nhanh
```

Feature không phụ thuộc split, nhưng với SNVSM V2 phải chạy Stage 05 trước như contract gate để không tốn extraction trên manifest thiếu/lệch. Index: `data/04_features/features_index.csv`.

### 05_build_labels — Data contract + split speaker-disjoint

```bash
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/05_build_labels/01_build_labels.py    # -> data/05_labels/labels.csv
```

V1 mặc định gộp real `all_clean.csv` + fake `data/03_fake/labels.csv`. Run V2 **phải truyền rõ** hai manifest `snvsm_v2`; không dùng default V1. Quy tắc chia 70/15/15:
1. Đơn vị chia = **connected component của (speaker_id ∪ source_video)** — clip chung speaker HOẶC chung video buộc cùng split. (Chỉ gom theo speaker_id là chưa đủ: 02_curate over-cluster chẻ 1 người thành nhiều speaker_id → 1 video trải nhiều split = leak identity đội lốt "speaker-disjoint".)
2. **Fake luôn cùng split với source_clip real** sinh ra nó.
3. Với SNVSM V2, mọi dòng phải đủ provenance semantic, real-fake phải cùng codec/CRF policy; mỗi source phải đủ đúng 4 method và cùng CRF-set/count với real; target audio cùng frame/FPS/duration video phải khớp real. Fake rỗng/mồ côi, media thiếu, thiếu method, duplicate, partial manifest hoặc lệch timing đều fail trước khi ghi labels. File được ghi atomic sau gate và mặc định từ chối overwrite.
4. Greedy bin-packing deterministic (`--seed`); cuối script **tự verify** không speaker_id VÀ không source_video nào ở 2 split (exit 1 nếu leak).

### Train / Eval (AVSP-Net — `src/model/avsp_net.py`)

Kiến trúc đang có trong code là **AVSP-Net V1**: audio Query × mouth-ROI Key/Value (cross-attention), nhánh prosody BiGRU, 2 head (real/fake `BCEWithLogitsLoss` + offset 7 lớp `CE`) + consistency loss. Đây là baseline pilot; kiến trúc mục tiêu mới nằm tại [MODEL_PROPOSAL.md](docs/architecture/MODEL_PROPOSAL.md).

```bash
# Baselines §7 (chạy đủ trước khi claim fusion tốt):
D:\Anaconda\envs\vn_av_df\python.exe src/train/train.py --branches audio                    # 1. audio-only
D:\Anaconda\envs\vn_av_df\python.exe src/train/train.py --branches visual                   # 2. visual-only (mouth ROI)
D:\Anaconda\envs\vn_av_df\python.exe src/train/train.py --branches audio,visual             # 3. AV fusion
D:\Anaconda\envs\vn_av_df\python.exe src/train/train.py --branches audio,visual,prosody     # 4. AVSP-Net full (mặc định)
# -> experiments/avsp_<branches>/{best.pt,last.pt,history.json} (best theo val AUC, early stop)

D:\Anaconda\envs\vn_av_df\python.exe src/eval/evaluate.py --ckpt experiments/avsp_audio_visual_prosody/best.pt
# -> acc/precision/recall/F1/ROC-AUC + method-wise recall/F1/AUC + FPR real -> eval_test.json
```

### PILOT (đã chạy 2026-07-21) — de-risk trước full run

Pilot = subset **speaker/video-disjoint** 540 real + 2160 fake (4 method ghép cặp) = **2700 clip**, split 378/81/81. Mục đích: xác nhận model học được TRƯỚC khi bỏ ~4.4h extract full. Output để ở path `_pilot` **cô lập** khỏi production (pilot fail thì xóa, không nhiễm `data/04_features/`).

> **Đường dẫn lệnh dưới đây là lịch sử và không còn chạy lại được nguyên trạng.**
> Manifest/media/feature Pilot V1 đã được dọn khỏi repository; run kết quả bất biến
> `experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/` và các báo cáo vẫn được giữ.

```bash
# manifest pilot: data/03_fake/snvsm/pilot_{real,fake}_snvsm.csv + data/05_labels/labels_pilot.csv
# (KHÔNG dùng --limit: code append toàn bộ real trước rồi mới slice -> ra 2700 real, không ghép cặp)
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/04_extract_features/01_extract_features.py \
    --real_csv data/03_fake/snvsm/pilot_real_snvsm.csv \
    --fake_labels data/03_fake/snvsm/pilot_fake_snvsm.csv \
    --out_dir data/04_features_pilot --detect_every 4        # ~1h, 2700/2700 ok, 0 fail
D:\Anaconda\envs\vn_av_df\python.exe src/train/train.py --labels data/05_labels/labels_pilot.csv \
    --features data/04_features_pilot --run_name pilot_v1_<timestamp>_<git-sha>_<config-hash> --epochs 30 --amp
# Không tái dùng run ID đã hoàn tất. Pilot V1 lịch sử hiện được khóa tại:
# experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/
D:\Anaconda\envs\vn_av_df\python.exe src/eval/evaluate.py --ckpt experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/best.pt \
    --labels data/05_labels/labels_pilot.csv --features data/04_features_pilot --split test
```

**Kết quả đo được:** best val AUC **0.813** @epoch 21 (early-stop 28); test AUC **0.809**, F1 0.789, precision 0.94, recall 0.68, FPR real 0.173. Val và test gần nhau trong cùng phân phối pseudo-fake; điều này **không đủ để kết luận không overfit artifact**. Audit metadata xác nhận split không trùng `speaker_id`/`source_video`, nhưng `speaker_id` là clustering tự động nên không phải ground-truth danh tính tuyệt đối.

| Method | Recall | F1 | AUC | Ghi chú |
|---|---|---|---|---|
| `pitch_flatten` | 1.000 | 0.921 | **0.990** | phát hiện tốt phép biến đổi F0 mạnh; chưa chứng minh hiểu thanh điệu Việt |
| `anonymization` | 0.975 | 0.908 | **0.960** | extractor cứu đủ anon; shortcut blur vẫn rất mạnh |
| `temporal_desync` | 0.543 | 0.633 | 0.750 | đang nhiễm artifact biên/độ dài từ generator; chưa đo thuần AV sync |
| `frame_reverse` | 0.198 | 0.288 | **0.535** | ⚠️ **gần như random** |

⚠️ **Điểm yếu đã biết và trạng thái xử lý:**
1. **`frame_reverse` AUC 0.535 ≈ chance** — kiến trúc thiếu nhạy thứ tự thời gian vẫn là giả thuyết chính: đảo ngược cửa sổ 0.3–1s trong clip ~4s là tín hiệu visual-motion cục bộ+ngắn, còn cross-attention thiên về khớp nội dung. Tuy nhiên audit sau pilot đã phát hiện timing artifact do `-shortest`, nên **chưa thể loại trừ lỗi data/shortcut** cho đến khi regenerate sạch và chạy lại. Hướng model vẫn là thêm frame-delta, motion-consistency loss và head cục bộ nhạy temporal-order.
2. **Temporal V1 có artifact blocking** — positive shift tạo khoảng trống đầu audio; negative shift có thể làm ngắn video/mouth do `-itsoffset` + `-shortest`. **Cơ chế generator V2 đã sửa và smoke đạt; structured timeline schema/fixed-common-window đã khóa**, nhưng loader/model V2a chưa tiêu thụ mask và repaired pilot chưa chạy, nên vẫn chưa được full.
3. **Offset head không học được shift** — accuracy bằng majority-zero baseline; consistency loss còn áp giả định sai cho fake vẫn đồng bộ.
4. **FPR real 0.173** — 17% real bị gắn fake; cần chọn threshold trên validation và kiểm tra calibration.
5. **Loader chỉ lấy 4 giây đầu và không có padding mask** — gây mất local anomaly và có thể tạo duration shortcut.
6. **Ba generator không-temporal V1 có `-shortest`** — code V2 đã repair; stratified smoke thật đạt 45/45 output và metadata gate tổng thể đạt, nhưng `pitch_flatten` logistic AUC 0,649 sát ngưỡng 0,65 nên full-pilot gate vẫn bắt buộc.

Chi tiết bằng chứng và thứ tự xử lý: [PILOT_V1_REVIEW_AND_V2_PLAN.md](docs/reports/PILOT_V1_REVIEW_AND_V2_PLAN.md).

### Phase 0 temporal repair (smoke 2026-07-21)

- Generator mới xoay audio theo sample, giữ video packet/frame, duration và timestamp hai hướng; output dùng file tạm + atomic-replace, resume so packet-count/time-base/duration-tick và tự sửa file corrupt không tạo dòng manifest trùng.
- Audio trung gian ALAC; SNVSM encode AAC 128k, 16 kHz mono cho cả real/fake. Vì AAC có trailing decoder padding, Stage 04 dùng `snvsm_target_samples` để trim waveform trước prosody/Wav2Vec và fail nếu manifest SNVSM thiếu contract này.
- Synthetic: 5 FPS × 6 shift = 30/30 đạt.
- Dữ liệu thật: 3 tier × 6 shift = 18/18 đạt; lag error lớn nhất 0,0625 ms.
- CLI smoke r4: 6 temporal → master 24 fake đủ bốn method → SNVSM 6 real + 24 fake cùng config hash và duy nhất AAC 16 kHz mono. Container timeline đạt 30/30; raw AAC decode có padding ở 29/30, nhưng contract trim mới của Stage 04 đưa cả 30/30 về đúng `snvsm_target_samples`. Sáu lag sau SNVSM sai số tối đa 0 ms. Artifact Stage 05 30 dòng được tạo trước paired-target gate và chỉ được giữ làm lịch sử.
- Policy smoke r5 tại checkpoint `1c592a4`: 6 real + 24 fake đủ method, `0/24` fake lệch CRF nguồn, `30/30` media/PCM contract đạt. Gate Stage 05 sau đó cố ý reject media V1: 12 fake lệch audio target và 17 fake lệch visual contract; không sinh labels. Sau Bước 2, r5 còn bị code hiện tại reject sớm hơn vì thiếu `av_timeline_v1`; đây là artifact lịch sử, chưa phải repaired pilot.
- Bước 2 structured timeline: schema `av_timeline_v1`, policy `fixed_common_window_v1`, semantics theo method, propagation qua SNVSM/Stage 05/Stage 04 và `timeline_contract_id` trong feature đã có contract-test.
- Bước 4 stratified smoke `v2r6`: 15 nguồn thật (5/tier) → 15 output/method, raw paired contract 60/60, SNVSM 15 real + 60 fake, Stage 05 đủ 75 labels và không trùng speaker/source qua split. Metadata gate group-disjoint đạt: logistic AUC 0,530; random forest AUC 0,546; max 0,546 ≤ 0,65.
- Guard fail-closed: Stage 05 mặc định kiểm file và dừng nếu fake rỗng/mồ côi, media thiếu, coverage/CRF/audio/video lệch; Stage 04 từ chối fake rỗng, chỉ resume feature khớp identity/source/config + exact tensor contract, ghi `.pt` atomic và trả exit lỗi nếu còn clip fail; dataloader không drop labels thiếu/sai feature/nhánh.
- Báo cáo lịch sử ghi nhận Pilot V1 từng khớp 8/8 checksum; artifact nguồn tương ứng đã được dọn nên không thể kiểm lại đầy đủ từ checkout hiện tại.

Chi tiết: [TEMPORAL_DESYNC_PHASE0_SMOKE.md](docs/reports/TEMPORAL_DESYNC_PHASE0_SMOKE.md). Đây vẫn chưa phải pilot V2a. Bước kế tiếp là tạo repaired pilot và, vì SNVSM V2 đổi media của cả real lẫn mọi fake, normalize lại **540 real + 2.160 fake**, qua Stage 05 và metadata gate toàn bộ labels, rồi mới extract **2.700 feature** vào store versioned; không trộn với `.pt` V1.

### Quy ước experiment bất biến

Từ V2 trở đi, mọi pilot/full/test run phải nằm trong một thư mục run ID duy nhất dưới `experiments/`, chứa scope (`pilot`/`full`), model (`v2a`/`v2b`), timestamp, git SHA và config hash. Run hoàn tất không được ghi đè. Cấu trúc artifact đầy đủ nằm ở [MODEL_PROPOSAL.md §18](docs/architecture/MODEL_PROPOSAL.md#18-experiment-output-bất-biến).

---

## Cài đặt môi trường

```bash
# Env chính hiện dùng: conda "vn_av_df" (có GPU InsightFace, cv2, ultralytics)
conda activate vn_av_df
pip install -r requirements.txt      # đã gồm transformers + praat-parselmouth + torchvision
# GPU: torch/torchvision trong requirements là bản PyPI (CPU-safe). Muốn CUDA:
#   pip install --force-reinstall --no-deps torch==2.12.0+cu126 torchvision==0.27.0+cu126 \
#       --index-url https://download.pytorch.org/whl/cu126
```

**Yêu cầu hệ thống:**
- Python 3.10+
- `ffmpeg`/`ffprobe` có trong PATH (quality gate, mọi script 03_fake, PoC inference)
- `yolov8n-face.pt` đặt tại thư mục root
- File `.env` với `YOUTUBE_API_KEY=...`

**Thư viện chính:** PyTorch, transformers (Wav2Vec2), ultralytics, insightface, librosa, opencv-python, yt-dlp, praat-parselmouth.

---

## Lưu ý kỹ thuật quan trọng

### Đặc thù tiếng Việt
- **Không dùng stemming kiểu tiếng Anh** — sẽ làm mất ngữ nghĩa hoàn toàn do hệ thống dấu đặc thù.
- **Thanh hỏi/ngã** có khoảng trống âm học đặc biệt cần chú ý khi xử lý prosody features. `03_pitch_flatten` là fake nhắm thẳng vào tín hiệu này.
- Audio backbone phải là model **Vietnamese-specific** (`wav2vec2-base-vietnamese-250h`), không dùng English model.
- Biên độ tần số tiếng Việt rộng hơn tiếng Anh (12–30 Hz), cần xác nhận lại độ phù hợp của PAMF.

### Metrics đánh giá
Cần bổ sung ít nhất **4 metrics** vào báo cáo (xem docs). **Cosine similarity không phù hợp** cho bài toán này. Bắt buộc report **method-wise F1** (theo từng loại fake) trên test set **speaker-disjoint**.

### Baseline models
Đưa vào so sánh: `wav2vec`, `sadtalker`. Lưu ý: các model này hiện chỉ hoạt động tốt trên tiếng Anh.

### Tạo fake data
- **Code (4 method trong `03_fake/`):** desync / frame-reverse / pitch-flatten / anonymization.
- **VEO3:** tạo video AI với khuôn mặt người nổi tiếng/vlogger thực tế.
- **CapCut noise:** gây nhiễu khuôn mặt bằng hiệu ứng méo/nhiễu, thời lượng ~15s.
- Đối tượng thực tế dùng **H.264** để né detection (hình ảnh mờ và nhiễu hơn).

### Dữ liệu & nhãn
- Chiếm 80–90% khối lượng công việc — ưu tiên làm data và label trước.
- Nguồn: YouTube (tier1 CC, tier2 Std) và TikTok (tier3).
- **Không commit:** `data/raw/`, `data/01_collect/cut_clips/`, `data/03_fake/`, `data/04_features/` (dung lượng lớn). Trước P4, accepted CSV theo batch là nguồn chân lý; sau khi `01_prep_manifest.py` đạt gate, `all_manifest.csv` mới trở thành manifest path canonical cho curation.

---

## File quan trọng

| File | Mục đích |
|---|---|
| [docs/architecture/MODEL_PROPOSAL.md](docs/architecture/MODEL_PROPOSAL.md) | Đề xuất AVSP-Net V2a/V2b, code layout, output contract và roadmap |
| [docs/reports/CUT_CLIPS_HOTFIX_AND_REBUILD_PLAN.md](docs/reports/CUT_CLIPS_HOTFIX_AND_REBUILD_PLAN.md) | Audit lỗi Stage 04 Cut Clips và kế hoạch chạy lại toàn bộ downstream |
| [docs/reports/PILOT_REPORT.md](docs/reports/PILOT_REPORT.md) | Báo cáo chi tiết quá trình pilot gốc |
| [docs/reports/PILOT_V1_REVIEW_AND_V2_PLAN.md](docs/reports/PILOT_V1_REVIEW_AND_V2_PLAN.md) | Review V1 sau pilot, fact-check và quyết định NO-GO/roadmap V2 |
| [docs/reports/TEMPORAL_DESYNC_PHASE0_SMOKE.md](docs/reports/TEMPORAL_DESYNC_PHASE0_SMOKE.md) | Bằng chứng sửa generator temporal và smoke Phase 0 |
| [docs/reports/ACTIVE_SPEAKER_CURATION_IMPLEMENTATION.md](docs/reports/ACTIVE_SPEAKER_CURATION_IMPLEMENTATION.md) | Thiết kế, code contract, validation gate và trạng thái Temporal Active-Speaker P4 |
| [src/pipeline/timeline_contract.py](src/pipeline/timeline_contract.py) | Schema/validator timeline dùng chung và fixed-common-window policy |
| [src/pipeline/fake_media_contract.py](src/pipeline/fake_media_contract.py) | Probe/validator atomic cho paired frame/FPS/duration/audio target của generator V2 |
| [src/pipeline/03_fake/01_temporal_desync.py](src/pipeline/03_fake/01_temporal_desync.py) | Generator temporal V2 sample-exact, manifest riêng và structured valid-range |
| [src/pipeline/03_fake/06_build_fake_manifest_v2.py](src/pipeline/03_fake/06_build_fake_manifest_v2.py) | Loại temporal V1 và audit composition 4 method/source; paired timing được generator/SNVSM/Stage 05 kiểm riêng |
| [src/pipeline/03_fake/07_metadata_shortcut_gate.py](src/pipeline/03_fake/07_metadata_shortcut_gate.py) | Group-disjoint baseline chỉ dùng metadata media/container để chặn shortcut trước extract/train |
| [src/model/avsp_net.py](src/model/avsp_net.py) | AVSP-Net (cross-attn + prosody + 2 head) + compute_losses |
| [src/pipeline/05_build_labels/01_build_labels.py](src/pipeline/05_build_labels/01_build_labels.py) | Data contract + split speaker-disjoint + verify leakage |
| [src/tools/review/clip_review.py](src/tools/review/clip_review.py) | Web tool lọc tay (ô ROI+tiếng) + so sánh với lọc code |
| [src/tools/review/build_review_manifest.py](src/tools/review/build_review_manifest.py) | Dựng `all_clean_review.csv` (tái lập được) |
| [yolov8n-face.pt](yolov8n-face.pt) | Face detection model (root) |
| `data/01_collect/cut_clips/tier{1,2,3}/**/accepted_clips.csv` | Nguồn chân lý Stage 04 hiện tại; tổng 65.622 clip |
| `data/01_collect/cut_clips/all_manifest.csv` | Input canonical P4 đã dựng đủ 65.622 clip real nguồn |
| [data/02_curate/manifests/all_clean.csv](data/02_curate/manifests/all_clean.csv) | Manifest lịch sử 3.001 clip; không dùng cho population mới |
