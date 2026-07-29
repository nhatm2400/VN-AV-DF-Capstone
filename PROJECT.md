# PROJECT.md — VN-AV-DF-Capstone

## Tổng quan dự án

Dự án phát hiện **Deepfake âm thanh-hình ảnh tiếng Việt** (Vietnamese Audio-Visual Deepfake Detection). Mục tiêu là xây dựng dataset và huấn luyện mô hình phát hiện video giả mạo (deepfake) đặc thù cho người Việt, tập trung vào sự lệch pha giữa âm thanh và khẩu hình miệng.

Mô hình hiện đã chạy pilot: **AVSP-Net V1** — mouth ROI + Wav2Vec + prosody, hợp nhất bằng Cross-Attention. Kiến trúc mục tiêu mới là **AVSP-Net V2**, gồm V2a (local temporal core, giữ cùng loại feature nhưng repaired pilot dùng store mới) và V2b (mở rộng để tổng quát hóa sang deepfake thực tế), xem [MODEL_PROPOSAL.md](docs/architecture/MODEL_PROPOSAL.md).

Trạng thái hiện tại: **BLOCKED tại Stage 04 Cut Clips, NO-GO cho manual review mới, sinh fake, extract feature và train**. Audit 2026-07-29 xác nhận lần cắt cũ làm `1.413` video dừng ở `video_decode_failed` do CUDA decode không có CPU fallback; Tier 2 còn thiếu coverage `169/292` video quality-pass. Vì vậy `6.888 clip → 3.001 all_clean` hiện chỉ còn giá trị lịch sử và phải được dựng lại từ Stage 04. P0 đã snapshot lineage/media inventory cũ; P1 đã có core dùng chung, notebook Kaggle canonical, fallback CPU/libx264 và contract output fail-closed. P2 local đã xác nhận Tier 1 đủ `472/472`, Tier 2 manifest đủ `292` filename và phục hồi exact inventory 1.262 source Tier 3 từ cut logs; vẫn phải đối chiếu Tier 2/3 với raw media trên Kaggle trước smoke. Stage 03 trở về trước không chạy lại. Xem [kế hoạch hotfix và rebuild](docs/reports/CUT_CLIPS_HOTFIX_AND_REBUILD_PLAN.md).

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
│   │   ├── 02_curate/                  # Lọc clip tự động (dùng chung mọi tier) — file phẳng
│   │   │   ├── 01_prep_manifest.py     # remap path + gộp tier -> all_manifest.csv
│   │   │   ├── 02_score_clips.py       # đo mặt + embedding 512-d (InsightFace buffalo_l)
│   │   │   ├── 03_sync_score.py        # đo khớp môi-tiếng (SyncNet) — tùy chọn
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
│   │   ├── build_review_manifest.py    # all_clean + motion + face_ambiguity + channel -> all_clean_review.csv
│   │   ├── build_review_assignments.py # chia calibration chung + primary riêng cho từng reviewer
│   │   ├── build_roi_preview.py        # dựng ô ROI+tiếng (chạy đúng detect_and_crop của stage 04)
│   │   ├── merge_review_results.py     # audit coverage, disagreement -> manual_clean_v2.csv
│   │   ├── scan_face_ambiguity.py      # luật "mặt to nhất" của stage 04 có đáng tin không
│   │   ├── measure_lip_audio_corr.py   # kiểm giả thuyết tự động hoá (đã đo: AUC 0,544 = vô dụng)
│   │   ├── clip_review.py              # Web tool lọc tay clip (stdlib http.server) + so sánh với lọc code
│   │   └── download_data.py            # tải dataset từ Google Drive (gdown, stream-zip, resume) — tiện ích nhóm
│   ├── model/
│   │   └── avsp_net.py                 # AVSP-Net: mouth-CNN+Transformer, prosody BiGRU, cross-attn, 2 head
│   ├── train/
│   │   ├── dataset.py                  # AVSPDataset: labels.csv + .pt -> tensor cố định
│   │   └── train.py                    # training loop (--branches cho ablation), best theo val AUC
│   ├── eval/
│   │   └── evaluate.py                 # acc/P/R/F1/AUC + method-wise + confusion -> eval_<split>.json
│   ├── utils/                          # (placeholder test.py)
│   └── pre-testing/                    # (trống) — thử nghiệm nhanh
├── PoC/                                # Proof-of-concept PAMF (đã hoạt động)
│   ├── src/
│   │   ├── feature_extractor.py        # Wav2Vec2 + MobileNetV2
│   │   ├── fusion_model.py             # PAMF Cross-Attention
│   │   ├── train_and_eval.py
│   │   ├── evaluate.py
│   │   ├── inference.py
│   │   ├── extract_all.py
│   │   └── data_maker.py
│   ├── data/                           # test1.mp4, test2.mp4 (mẫu)
│   ├── checkpoints/                    # pamf_poc_model.pth
│   ├── confusion_matrix.png
│   └── requirements.txt
├── data/                               # (KHÔNG commit phần nặng: raw, cut_clips, fake, features)
│   ├── 01_collect/
│   │   ├── youtube_tier1_urls.csv, youtube_tier2_urls.csv
│   │   ├── tier1_quality_gate_passed.csv, tier2_quality_gate_passed.csv
│   │   └── cut_clips/                  # ĐÃ DỜI từ data/clips/ về đây
│   │       ├── all_manifest.csv        # nguồn chân lý path clip (6.888 clip)
│   │       └── tier1/ tier2/ tier3/    # các .mp4 đã cắt + tier{N}_v3_clips_*.csv
│   ├── 02_curate/
│   │   ├── measurements/               # kết quả đo tự động trước khi ra quyết định
│   │   │   ├── tier1_scored_all.csv, tier1_scored_motion.csv
│   │   │   └── embeddings_all.npy      # 6.888 × 512 (commit — ngoại lệ trong .gitignore)
│   │   ├── manifests/                  # bảng quyết định của curation
│   │   │   ├── all_clean.csv, all_clean_rejects.csv
│   │   │   └── all_clean_review.csv
│   │   ├── calibration/                # kết quả thử ngưỡng SyncNet
│   │   │   ├── calibrate_sync_results.csv
│   │   │   └── sync_calibrate_log.txt
│   │   ├── logs/                       # log chạy curation/preview (không commit)
│   │   ├── eda_figs/                   # 11 PNG + eda_summary.md
│   │   └── manual/                     # quyết định lọc tay từ src/tools/clip_review.py
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

## Kiến trúc mô hình PAMF

### Feature Extractors

| Modality | Backbone | Output dim | Checkpoint |
|---|---|---|---|
| Audio | Wav2Vec2 Base (Vietnamese) | `[B, T_A, 768]` | `nguyenvulebinh/wav2vec2-base-vietnamese-250h` |
| Visual | MobileNetV2 (features only) | `[B, T_V, 1280]` | ImageNet pretrained |

### Fusion Model (`PoC/src/fusion_model.py`)

```
Audio [B,T_A,768]  ──LayerNorm──Linear──► Query [B,T_A,512]
                                                    │
Visual [B,T_V,1280]──LayerNorm──Linear──► Key,Value [B,T_V,512]
                                                    │
                                      MultiheadAttention (4 heads)
                                                    │
                                        GlobalAvgPool → [B,512]
                                                    │
                                        MLP → Dropout(0.3) → Sigmoid
                                                    │
                                          0 = Real, 1 = Fake
```

**Nguyên lý:** Audio làm Query đi tìm khẩu hình miệng (Visual Key/Value) tương ứng. Nếu lệch pha → Deepfake.

> AVSP-Net V1 đã triển khai mouth ROI + prosody nhưng pilot cho thấy global fusion/offset loss chưa đủ. Kiến trúc thay thế V2a/V2b nằm tại [MODEL_PROPOSAL.md](docs/architecture/MODEL_PROPOSAL.md).

### Training (PoC)

- Loss: `BCELoss`
- Optimizer: `Adam(lr=5e-4)`
- Epochs: 50
- Gradient accumulation qua toàn bộ file trong epoch, clip grad norm = 1.0
- Chạy từ thư mục `PoC/`: `python src/train_and_eval.py`

### Inference

```bash
cd PoC
python src/inference.py
```

Checkpoint: `PoC/checkpoints/pamf_poc_model.pth`

---

## Pipeline dữ liệu

Chạy theo đúng thứ tự từ **root repo**. Stage `data/` khớp số với `src/pipeline/`.

### 01_collect — Thu thập & cắt clip

**Fetch URLs (theo tier):**
```bash
python src/pipeline/01_collect/tier1/01_fetch_youtube_urls.py   # tier1 (YouTube CC)
python src/pipeline/01_collect/tier2/01_fetch_youtube_urls.py   # tier2 (YouTube Std)
python src/pipeline/01_collect/tier3/01_fetch_tiktok_urls.py    # tier3 (TikTok)
```
Yêu cầu `YOUTUBE_API_KEY` trong `.env`. Script ghi mặc định ra `data/…_urls.csv`; output cuối đã gom vào `data/01_collect/`.

**Download:**
```bash
python src/pipeline/01_collect/tier{N}/02_download.py     # yt-dlp -> data/raw/tier{N}/
```

**Quality Gate** (`03_quality_gate.py`) — video phải pass cả 3:
| Tiêu chí | Ngưỡng |
|---|---|
| Độ phân giải | ≥ 480p |
| FPS | ≥ 24 |
| Luồng âm thanh | Có (đọc bằng ffprobe) |

Output: `data/01_collect/tier{N}_quality_gate_passed.csv`. (Ngưỡng SNR/face trong tài liệu là chủ trương thiết kế; code hiện gate độ phân giải + FPS + có audio.)

**Cắt clip** — `tier{N}/04_cut_clips.ipynb` (Kaggle GPU). Cắt theo VAD (Silero) + lọc cắt cảnh/mặt (YOLO)/speech; **đo SNR nhưng KHÔNG gate** (`MIN_SNR=-999`, giữ metadata). Output vào `data/01_collect/cut_clips/tier{N}/`: `tier{N}_v3_clips_*.csv` (+ `_rejects_*.csv`) + các `.mp4`. Schema clip: `clip_id, source_video, start_time, end_time, duration, face_ratio, speech_ratio, snr, file_path`.

### 02_curate — Curation tự động (`src/pipeline/02_curate/`)

Lọc ~6.9k clip thay cho lọc tay. Triết lý: **đo mọi thứ, chỉ loại rác rõ ràng, giữ phần còn lại làm metadata; calibrate trước khi đặt ngưỡng**. Chạy tuần tự:

```bash
# 1) Gộp tier + remap file_path về đĩa thật (CHẠY THEO ĐĨA, verify ffprobe)
python src/pipeline/02_curate/01_prep_manifest.py \
    --add tier1 "data/01_collect/cut_clips/tier1/**/*_v3_clips_*.csv" "data/01_collect/cut_clips/tier1" \
    --add tier2 "data/01_collect/cut_clips/tier2/**/*_v3_clips_*.csv" "data/01_collect/cut_clips/tier2" \
    --add tier3 "data/01_collect/cut_clips/tier3/**/*_v3_clips_*.csv" "data/01_collect/cut_clips/tier3" \
    --out data/01_collect/cut_clips/all_manifest.csv
# 2) Đo mặt + embedding (GPU, InsightFace) -> data/02_curate/measurements/
python src/pipeline/02_curate/02_score_clips.py \
    --input_csv data/01_collect/cut_clips/all_manifest.csv --tag all
# 3) Đo khớp môi-tiếng (SyncNet) — tùy chọn; calibrate trước
python src/pipeline/02_curate/03_sync_score.py \
    --input_csv data/02_curate/measurements/tier1_scored_all.csv \
    --syncnet_dir <repo> --calibrate
# 4) Quyết định: cluster speaker -> gate rác -> cân bằng -> manifests/all_clean.csv
python src/pipeline/02_curate/04_curate.py --calibrate
# 5) EDA: thống kê + xuất data/02_curate/eda_figs/
python src/pipeline/02_curate/05_eda.py
```

**Ngưỡng chính:** cluster_dist=0.6, min_det_ratio=0.6, min_face_area=0.01, min_consistency=0.3, cap_per_speaker=30; `quality_score = 0.4·det + 0.3·norm(face_area) + 0.15·consistency` (chuẩn hóa p5–p95).

⚠️ **Chống leakage:** không siết ngưỡng sync cho tập real (đẩy real về "sync cao" → model học tắt). Mọi gate sync/chất lượng phải áp **đối xứng** real/fake. Các script này thiết kế chạy trên **Kaggle** (GPU, path mặc định `/kaggle/working`).

**Lọc tay (BẮT BUỘC trước khi sinh fake):** gate tự động **không** xác nhận được người nhìn thấy có phải người phát ra tiếng hay không. Batch hiệu chuẩn 60 clip (rubric v2) loại **34/60** — chủ yếu `dubbed` (14), `cut` (7), `static` (6). Đây là false-accept nghiêm trọng của curation tự động, đủ để chặn pilot.

```bash
python src/tools/build_review_manifest.py   # 1) gộp số đo -> manifests/all_clean_review.csv
python src/tools/build_roi_preview.py       # 2) dựng ô ROI+tiếng (~85 phút, 3.001 clip)
python src/tools/clip_review.py             # 3) mở http://127.0.0.1:8000
```

Ô ROI phát **chuỗi mouth-ROI thật của stage 04 ghép audio gốc** — nhìn miệng + nghe tiếng cùng lúc mới lộ được lồng tiếng / voice-over / cắt nhầm mặt. Phím `1`–`7` reject kèm lý do, `K` keep, `C` chưa chắc; quyết định ghi kèm `reviewer_id` + `rubric_version`.

**Đã thử và thất bại** (đừng làm lại): `motion_median` và số khuôn mặt không dự báo được keep/reject (60% vs 55%); tương quan pixel-motion ROI với audio RMS cho **AUC 0,544**. Lọc theo kênh chỉ nên dùng để **phân tầng/ưu tiên**, không auto-reject — mẫu theo kênh quá nhỏ (Fisher exact p≈0,12 cho kênh tệ nhất).

⚠️ Chỉ loại **rác rõ ràng**, KHÔNG lọc theo chất lượng lip-sync. Real bị loại thì loại **cả cụm** (1 real + 4 fake sinh từ nó).

### 03_fake — Sinh pseudo-fake (`src/pipeline/03_fake/`)

V1 lịch sử đã sinh bốn method vào `data/03_fake/labels.csv`; không dùng lại cho repaired pilot. Bốn generator V2 ghi media/manifest versioned riêng dưới `data/03_fake/*_v2` và `data/03_fake/manifests/v2/`. Builder chỉ nhận đúng generator version + timeline contract V2 của cả bốn method, không còn lấy ba method từ manifest V1.

> **`data/03_fake/` hiện trống (chỉ còn `.gitkeep`).** Fake V1 nằm ở
> [`archive/pilot_v1/`](archive/pilot_v1/README.md); sáu vòng smoke Phase 0 của generator
> V2 (`phase0_*`, gồm cả `v2r6` và metadata gate) nằm ở
> [`archive/phase0_v2_smoke/`](archive/phase0_v2_smoke/README.md) từ 2026-07-29. Đường dẫn
> trong manifest của cả hai đã được viết lại và kiểm chứng toàn bộ. Repaired pilot sẽ ghi
> mới vào `data/03_fake/`.

```bash
# Mặc định V2: media -> data/03_fake/temporal_v2/
#               manifest -> data/03_fake/manifests/v2/temporal_desync.csv
python src/pipeline/03_fake/01_temporal_desync.py

# Tạo master composition không còn temporal V1; mọi media V2 phải qua paired contract
python src/pipeline/03_fake/06_build_fake_manifest_v2.py
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
python src/pipeline/03_fake/05_snvsm_compress.py --input_csv data/02_curate/manifests/all_clean.csv \
    --out_dir data/03_fake/snvsm_v2/real --out_manifest data/03_fake/snvsm_v2/real_snvsm.csv
python src/pipeline/03_fake/05_snvsm_compress.py --input_csv data/03_fake/manifests/v2/fake_all.csv \
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
python src/pipeline/04_extract_features/01_extract_features.py            # full (GPU khuyến nghị)
python src/pipeline/04_extract_features/01_extract_features.py --limit 5 --no_w2v   # test nhanh
```

Feature không phụ thuộc split, nhưng với SNVSM V2 phải chạy Stage 05 trước như contract gate để không tốn extraction trên manifest thiếu/lệch. Index: `data/04_features/features_index.csv`.

### 05_build_labels — Data contract + split speaker-disjoint

```bash
python src/pipeline/05_build_labels/01_build_labels.py    # -> data/05_labels/labels.csv
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
python src/train/train.py --branches audio                    # 1. audio-only
python src/train/train.py --branches visual                   # 2. visual-only (mouth ROI)
python src/train/train.py --branches audio,visual             # 3. AV fusion
python src/train/train.py --branches audio,visual,prosody     # 4. AVSP-Net full (mặc định)
# -> experiments/avsp_<branches>/{best.pt,last.pt,history.json} (best theo val AUC, early stop)

python src/eval/evaluate.py --ckpt experiments/avsp_audio_visual_prosody/best.pt
# -> acc/precision/recall/F1/ROC-AUC + method-wise recall/F1/AUC + FPR real -> eval_test.json
```

### PILOT (đã chạy 2026-07-21) — de-risk trước full run

Pilot = subset **speaker/video-disjoint** 540 real + 2160 fake (4 method ghép cặp) = **2700 clip**, split 378/81/81. Mục đích: xác nhận model học được TRƯỚC khi bỏ ~4.4h extract full. Output để ở path `_pilot` **cô lập** khỏi production (pilot fail thì xóa, không nhiễm `data/04_features/`).

> **Đường dẫn dưới đây là lịch sử.** Từ 2026-07-28 toàn bộ artifact V1 + pilot V1 đã
> chuyển sang [`archive/pilot_v1/`](archive/pilot_v1/README.md) và manifest đã được
> viết lại tương ứng: `data/03_fake/` → `archive/pilot_v1/03_fake/`,
> `data/04_features_pilot/` → `archive/pilot_v1/04_features_pilot/`,
> `data/05_labels/labels{,_pilot}.csv` → `archive/pilot_v1/05_labels/`.

```bash
# manifest pilot: data/03_fake/snvsm/pilot_{real,fake}_snvsm.csv + data/05_labels/labels_pilot.csv
# (KHÔNG dùng --limit: code append toàn bộ real trước rồi mới slice -> ra 2700 real, không ghép cặp)
python src/pipeline/04_extract_features/01_extract_features.py \
    --real_csv data/03_fake/snvsm/pilot_real_snvsm.csv \
    --fake_labels data/03_fake/snvsm/pilot_fake_snvsm.csv \
    --out_dir data/04_features_pilot --detect_every 4        # ~1h, 2700/2700 ok, 0 fail
python src/train/train.py --labels data/05_labels/labels_pilot.csv \
    --features data/04_features_pilot --run_name pilot_v1_<timestamp>_<git-sha>_<config-hash> --epochs 30 --amp
# Không tái dùng run ID đã hoàn tất. Pilot V1 lịch sử hiện được khóa tại:
# experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/
python src/eval/evaluate.py --ckpt experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/best.pt \
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
- Pilot V1 vẫn bất biến, checksum khớp 8/8.

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
- **Không commit:** `data/raw/`, `data/01_collect/cut_clips/`, `data/03_fake/`, `data/04_features/` (dung lượng lớn). `all_manifest.csv` là nguồn chân lý về path clip.

---

## File quan trọng

| File | Mục đích |
|---|---|
| [docs/architecture/MODEL_PROPOSAL.md](docs/architecture/MODEL_PROPOSAL.md) | Đề xuất AVSP-Net V2a/V2b, code layout, output contract và roadmap |
| [docs/reports/CUT_CLIPS_HOTFIX_AND_REBUILD_PLAN.md](docs/reports/CUT_CLIPS_HOTFIX_AND_REBUILD_PLAN.md) | Audit lỗi Stage 04 Cut Clips và kế hoạch chạy lại toàn bộ downstream |
| [docs/reports/PILOT_REPORT.md](docs/reports/PILOT_REPORT.md) | Báo cáo chi tiết quá trình pilot gốc |
| [docs/reports/PILOT_V1_REVIEW_AND_V2_PLAN.md](docs/reports/PILOT_V1_REVIEW_AND_V2_PLAN.md) | Review V1 sau pilot, fact-check và quyết định NO-GO/roadmap V2 |
| [docs/reports/TEMPORAL_DESYNC_PHASE0_SMOKE.md](docs/reports/TEMPORAL_DESYNC_PHASE0_SMOKE.md) | Bằng chứng sửa generator temporal và smoke Phase 0 |
| [PoC/src/fusion_model.py](PoC/src/fusion_model.py) | Định nghĩa PAMF_Fusion (Cross-Attention) |
| [PoC/src/feature_extractor.py](PoC/src/feature_extractor.py) | Wav2Vec2 + MobileNetV2 extractor |
| [src/pipeline/timeline_contract.py](src/pipeline/timeline_contract.py) | Schema/validator timeline dùng chung và fixed-common-window policy |
| [src/pipeline/fake_media_contract.py](src/pipeline/fake_media_contract.py) | Probe/validator atomic cho paired frame/FPS/duration/audio target của generator V2 |
| [src/pipeline/03_fake/01_temporal_desync.py](src/pipeline/03_fake/01_temporal_desync.py) | Generator temporal V2 sample-exact, manifest riêng và structured valid-range |
| [src/pipeline/03_fake/06_build_fake_manifest_v2.py](src/pipeline/03_fake/06_build_fake_manifest_v2.py) | Loại temporal V1 và audit composition 4 method/source; paired timing được generator/SNVSM/Stage 05 kiểm riêng |
| [src/pipeline/03_fake/07_metadata_shortcut_gate.py](src/pipeline/03_fake/07_metadata_shortcut_gate.py) | Group-disjoint baseline chỉ dùng metadata media/container để chặn shortcut trước extract/train |
| [src/model/avsp_net.py](src/model/avsp_net.py) | AVSP-Net (cross-attn + prosody + 2 head) + compute_losses |
| [src/pipeline/05_build_labels/01_build_labels.py](src/pipeline/05_build_labels/01_build_labels.py) | Data contract + split speaker-disjoint + verify leakage |
| [src/tools/clip_review.py](src/tools/clip_review.py) | Web tool lọc tay (ô ROI+tiếng) + so sánh với lọc code |
| [src/tools/build_review_manifest.py](src/tools/build_review_manifest.py) | Dựng `all_clean_review.csv` (tái lập được) |
| [yolov8n-face.pt](yolov8n-face.pt) | Face detection model (root) |
| [data/01_collect/cut_clips/all_manifest.csv](data/01_collect/cut_clips/all_manifest.csv) | Nguồn chân lý path 6.888 clip |
| [data/02_curate/manifests/all_clean.csv](data/02_curate/manifests/all_clean.csv) | 3.001 clip real sạch (kèm speaker_id) |
