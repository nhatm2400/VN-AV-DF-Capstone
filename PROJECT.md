# PROJECT.md — VN-AV-DF-Capstone

## Tổng quan dự án

Dự án phát hiện **Deepfake âm thanh-hình ảnh tiếng Việt** (Vietnamese Audio-Visual Deepfake Detection). Mục tiêu là xây dựng dataset và huấn luyện mô hình phát hiện video giả mạo (deepfake) đặc thù cho người Việt, tập trung vào sự lệch pha giữa âm thanh và khẩu hình miệng.

Mô hình hiện đã chạy pilot: **AVSP-Net V1** — mouth ROI + Wav2Vec + prosody, hợp nhất bằng Cross-Attention. Kiến trúc mục tiêu mới là **AVSP-Net V2**, gồm V2a (local temporal core, tái dùng feature hiện có) và V2b (mở rộng để tổng quát hóa sang deepfake thực tế), xem [MODEL_PROPOSAL.md](docs/architecture/MODEL_PROPOSAL.md).

Trạng thái hiện tại: **dữ liệu + curation đã xong** (6.888 clip → 3.001 clip sạch); **03_fake đã sinh đủ 12.004 fake (4 method), SNVSM đồng bộ codec 15.005 clip, `data/05_labels/labels.csv` đã build + split speaker/video-disjoint verified**; **PILOT V1 đã chạy xong** (2.700 clip, test AUC **0.809** — xem [PILOT](#pilot-đã-chạy-2026-07-21--de-risk-trước-full-run)); **chưa trích feature FULL, chưa train mô hình chính**. Trạng thái quyết định hiện tại là **NO-GO full V1**: phải sửa artifact của `temporal_desync`, nâng lên V2a và chạy lại pilot diagnostic trước. Xem [báo cáo đánh giá V1/V2](docs/reports/PILOT_V1_REVIEW_AND_V2_PLAN.md).

---

## Cấu trúc thư mục

> Đã sắp xếp lại theo stage đánh số cho cả `src/pipeline/` lẫn `data/` (01_collect → 02_curate → 03_fake → 04_features). `data/` và `src/pipeline/` khớp số với nhau.

```
.
├── src/
│   ├── pipeline/                       # Pipeline xử lý dữ liệu (stage đánh số)
│   │   ├── 01_collect/                 # Thu thập + cắt clip — tách theo tier (nguồn)
│   │   │   ├── tier1/                  # YouTube CC: 01_fetch_youtube_urls 02_download 03_quality_gate 04_cut_clips.ipynb
│   │   │   ├── tier2/                  # YouTube Std: 00_explore_license 01_fetch 02_download 03_generate_download_script 04_retry_failed 05_clean_temp
│   │   │   └── tier3/                  # TikTok: 01_fetch_tiktok_urls 02_download 03_quality_gate 04_cut_clips.ipynb
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
│   │   │   └── 05_snvsm_compress.py    # nén H.264 CRF đối xứng real+fake (đồng bộ codec)
│   │   ├── 04_extract_features/
│   │   │   └── 01_extract_features.py  # mouth-ROI (YOLO) + wav2vec2 + prosody F0 -> .pt/clip
│   │   └── 05_build_labels/
│   │       └── 01_build_labels.py      # gộp real+fake -> labels.csv + split SPEAKER-DISJOINT
│   ├── tools/
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
│   │   ├── tier1_scored_all.csv        # 6.888 clip + det_ratio, mean_face_area, embed_consistency
│   │   ├── embeddings_all.npy          # 6.888 × 512 (commit — có ngoại lệ trong .gitignore)
│   │   ├── all_clean.csv               # 3.001 clip sạch (kèm speaker_id)
│   │   ├── all_clean_rejects.csv       # 1.532 clip bị gate loại
│   │   ├── calibrate_sync_results.csv, sync_calibrate_log.txt
│   │   ├── eda_figs/                   # 11 PNG + eda_summary.md
│   │   └── manual/manual_review.csv    # kết quả lọc tay từ src/tools/clip_review.py
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
│       └── PILOT_V1_REVIEW_AND_V2_PLAN.md
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
# 2) Đo mặt + embedding (GPU, InsightFace)  -> tier1_scored_all.csv + embeddings_all.npy
python src/pipeline/02_curate/02_score_clips.py --input_csv all_manifest.csv --tag all
# 3) Đo khớp môi-tiếng (SyncNet) — tùy chọn; calibrate trước
python src/pipeline/02_curate/03_sync_score.py --input_csv tier1_scored_all.csv --syncnet_dir <repo> --calibrate
# 4) Quyết định: cluster speaker -> gate rác -> cân bằng -> tập sạch (all_clean.csv)
python src/pipeline/02_curate/04_curate.py --scored_csv ... --emb ... --calibrate
# 5) EDA: thống kê + xuất data/02_curate/eda_figs/
python src/pipeline/02_curate/05_eda.py
```

**Ngưỡng chính:** cluster_dist=0.6, min_det_ratio=0.6, min_face_area=0.01, min_consistency=0.3, cap_per_speaker=30; `quality_score = 0.4·det + 0.3·norm(face_area) + 0.15·consistency` (chuẩn hóa p5–p95).

⚠️ **Chống leakage:** không siết ngưỡng sync cho tập real (đẩy real về "sync cao" → model học tắt). Mọi gate sync/chất lượng phải áp **đối xứng** real/fake. Các script này thiết kế chạy trên **Kaggle** (GPU, path mặc định `/kaggle/working`).

**Lọc tay đối chiếu:** `python src/tools/clip_review.py` mở web tool xem từng clip (phím K=keep, X=reject, U=unset), xuất `data/02_curate/manual/manual_review.csv` và có nút so sánh lọc-tay vs lọc-code (ma trận keep/gate/balance-drop). Chỉ loại **rác rõ ràng**, KHÔNG lọc theo chất lượng lip-sync, áp đối xứng real/fake.

### 03_fake — Sinh pseudo-fake (`src/pipeline/03_fake/`)

4 method độc lập, mỗi cái đọc CSV clip real (mặc định `data/02_curate/all_clean.csv`), sinh `.mp4` fake và **append cùng một `labels.csv`** (schema chung: `clip_id, file_path, label, method, param, source_clip, source_video, speaker_id, tier`; `label=1`).

```bash
# Mặc định: --input_csv data/02_curate/all_clean.csv, --out_dir data/03_fake, --labels data/03_fake/labels.csv
python src/pipeline/03_fake/01_temporal_desync.py
python src/pipeline/03_fake/02_frame_reverse.py
python src/pipeline/03_fake/03_pitch_flatten.py     # cần: pip install praat-parselmouth
python src/pipeline/03_fake/04_anonymization.py
```

| Method | Kênh tấn công | Cơ chế | Stream giữ nguyên | Phụ thuộc thêm |
|---|---|---|---|---|
| `01_temporal_desync` | timing | dịch audio 3/7/15 frame, hướng ngẫu nhiên | video copy | ffmpeg |
| `02_frame_reverse` | visual-motion | đảo ngược cửa sổ 0.3–1.0s video | **audio copy** | ffmpeg |
| `03_pitch_flatten` | audio-prosody | làm phẳng F0 (PSOLA, đặc thù tiếng Việt) | **video copy** | **parselmouth** |
| `04_anonymization` | visual-identity | blur ≥51px / pixelate vùng mặt | **audio copy** | ultralytics + cv2 |

⚠️ **Hai lưu ý bắt buộc trước khi train:**
1. **Đồng bộ codec (SNVSM):** `02`/`04` re-encode video → khác codec với real. `05_snvsm_compress.py` nén 4 mức CRF (23/30/35/40) **đối xứng cả real+fake** trước khi trích feature — xóa vân codec (không thì model học "codec = fake"). Chạy 2 lần cùng tham số (real + fake) rồi trỏ `04`/`05` vào manifest SNVSM. **(đã chạy: real 3001 + fake 12004)**
2. **Anonymization leakage:** blur chỉ trên fake → model học "mờ = fake". **ĐÃ XỬ LÝ:** `train.py --real_blur_aug_p` (mặc định 0.25, CHỈ ở train) blur đối xứng mouth-ROI một phần real; vì anon = 1/4 fake nên P(blur|real)≈P(blur|fake). Ngoài ra anon blur mặt → YOLO detect fail ~18-25%; `04` xử lý bằng cách lấy chuỗi box của REAL ghép cặp áp lên anon (không detect trên mặt mờ) — xem mục 04.

### 03_fake/05 — SNVSM đồng bộ codec (chạy sau 4 method, trước 04)

```bash
# REAL và FAKE cùng --crfs/--preset -> codec khớp tuyệt đối (real+fake đều h264 fresh)
python src/pipeline/03_fake/05_snvsm_compress.py --input_csv data/02_curate/all_clean.csv \
    --out_dir data/03_fake/snvsm/real --out_manifest data/03_fake/snvsm/real_snvsm.csv
python src/pipeline/03_fake/05_snvsm_compress.py --input_csv data/03_fake/labels.csv \
    --out_dir data/03_fake/snvsm/fake --out_manifest data/03_fake/snvsm/fake_snvsm.csv
# rồi 04/05 trỏ vào manifest SNVSM:
#   ...05_build_labels/01_build_labels.py --real_csv .../real_snvsm.csv --fake_labels .../fake_snvsm.csv
#   ...04_extract_features/01_extract_features.py --real_csv .../real_snvsm.csv --fake_labels .../fake_snvsm.csv
```

`--mode random` (mặc định): 1 CRF ngẫu nhiên/clip (×1 dung lượng). `--mode all`: đủ 4 mức (×4, augmentation tối đa). Manifest giữ nguyên mọi cột (speaker_id, source_video…) nên split speaker-disjoint ở 05 vẫn đúng.

### 04_extract_features — Trích feature 3 nhánh

Theo data contract V1 đã triển khai: **KHÔNG dùng full-frame** (giảm leak identity/background) — mỗi clip (real + fake) trích 1 file `.pt` vào `data/04_features/`:
- `mouth`: uint8 `[T,96,96]` — YOLOv8n-face detect, crop nửa dưới bbox (vùng miệng), ~25fps. **Mỗi sampled-frame LUÔN có 1 ROI** (carry-forward khi detect fail giữa clip, backward-fill khi fail ở đầu) → chuỗi hình không co/lệch với audio. **ANON** (mặt mờ, YOLO fail ~18-25%): dùng chuỗi box của **REAL ghép cặp** (`source_clip`→`orig_clip_id`, cache khi xử lý real trước) áp lên anon theo timestamp — crop môi chặt, không phụ thuộc detect trên mặt mờ, không tạo shortcut "static-crop = anon".
- `w2v`: float16 `[T,768]` — wav2vec2-base-vietnamese-250h frozen (tắt bằng `--no_w2v`)
- `prosody`: float32 `[T,4]` — f0_z, delta_f0, energy_z, voiced @100Hz (parselmouth, fallback librosa.pyin)

```bash
python src/pipeline/04_extract_features/01_extract_features.py            # full (GPU khuyến nghị)
python src/pipeline/04_extract_features/01_extract_features.py --limit 5 --no_w2v   # test nhanh
```

Feature không phụ thuộc split — chạy trước/sau 05 đều được. Index: `data/04_features/features_index.csv`.

### 05_build_labels — Data contract + split speaker-disjoint

```bash
python src/pipeline/05_build_labels/01_build_labels.py    # -> data/05_labels/labels.csv
```

Gộp real (`all_clean.csv`, label=0) + fake (`data/03_fake/labels.csv`, label=1). Quy tắc chia 70/15/15:
1. Đơn vị chia = **connected component của (speaker_id ∪ source_video)** — clip chung speaker HOẶC chung video buộc cùng split. (Chỉ gom theo speaker_id là chưa đủ: 02_curate over-cluster chẻ 1 người thành nhiều speaker_id → 1 video trải nhiều split = leak identity đội lốt "speaker-disjoint".)
2. **Fake luôn cùng split với source_clip real** sinh ra nó.
3. Greedy bin-packing deterministic (`--seed`); cuối script **tự verify** không speaker_id VÀ không source_video nào ở 2 split (exit 1 nếu leak).

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

⚠️ **Điểm yếu đã biết (chưa xử lý):**
1. **`frame_reverse` AUC 0.535 ≈ chance** — vấn đề KIẾN TRÚC, không phải bug data: đảo ngược cửa sổ 0.3–1s trong clip ~4s là tín hiệu visual-motion cục bộ+ngắn; nhánh mouth (2D-CNN + temporal transformer) chưa nhạy **thứ tự thời gian**, và cross-attention thiên "khớp nội dung" nên đoạn đảo vẫn ~khớp audio. Hướng sửa: thêm feature delta giữa frame / motion-consistency loss / head nhạy temporal-order.
2. **`temporal_desync` generator có artifact blocking** — positive shift tạo khoảng trống đầu audio; negative shift có thể làm ngắn video/mouth do `-itsoffset` + `-shortest`. Phải sửa và regenerate trước full.
3. **Offset head không học được shift** — accuracy bằng majority-zero baseline; consistency loss còn áp giả định sai cho fake vẫn đồng bộ.
4. **FPR real 0.173** — 17% real bị gắn fake; cần chọn threshold trên validation và kiểm tra calibration.
5. **Loader chỉ lấy 4 giây đầu và không có padding mask** — gây mất local anomaly và có thể tạo duration shortcut.

Chi tiết bằng chứng và thứ tự xử lý: [PILOT_V1_REVIEW_AND_V2_PLAN.md](docs/reports/PILOT_V1_REVIEW_AND_V2_PLAN.md).

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
| [docs/reports/PILOT_REPORT.md](docs/reports/PILOT_REPORT.md) | Báo cáo chi tiết quá trình pilot gốc |
| [docs/reports/PILOT_V1_REVIEW_AND_V2_PLAN.md](docs/reports/PILOT_V1_REVIEW_AND_V2_PLAN.md) | Review V1 sau pilot, fact-check và quyết định NO-GO/roadmap V2 |
| [PoC/src/fusion_model.py](PoC/src/fusion_model.py) | Định nghĩa PAMF_Fusion (Cross-Attention) |
| [PoC/src/feature_extractor.py](PoC/src/feature_extractor.py) | Wav2Vec2 + MobileNetV2 extractor |
| [src/pipeline/03_fake/01_temporal_desync.py](src/pipeline/03_fake/01_temporal_desync.py) | Mẫu chuẩn của 4 method fake (schema labels chung) |
| [src/model/avsp_net.py](src/model/avsp_net.py) | AVSP-Net (cross-attn + prosody + 2 head) + compute_losses |
| [src/pipeline/05_build_labels/01_build_labels.py](src/pipeline/05_build_labels/01_build_labels.py) | Data contract + split speaker-disjoint + verify leakage |
| [src/tools/clip_review.py](src/tools/clip_review.py) | Web tool lọc tay + so sánh với lọc code |
| [yolov8n-face.pt](yolov8n-face.pt) | Face detection model (root) |
| [data/01_collect/cut_clips/all_manifest.csv](data/01_collect/cut_clips/all_manifest.csv) | Nguồn chân lý path 6.888 clip |
| [data/02_curate/all_clean.csv](data/02_curate/all_clean.csv) | 3.001 clip real sạch (kèm speaker_id) |
