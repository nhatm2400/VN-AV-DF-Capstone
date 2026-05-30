# CLAUDE.md — VN-AV-DF-Capstone

## Tổng quan dự án

Dự án phát hiện **Deepfake âm thanh-hình ảnh tiếng Việt** (Vietnamese Audio-Visual Deepfake Detection). Mục tiêu là xây dựng dataset và huấn luyện mô hình phát hiện video giả mạo (deepfake) đặc thù cho người Việt, tập trung vào sự lệch pha giữa âm thanh và khẩu hình miệng.

Mô hình cốt lõi: **PAMF (Prosody-Aligned Multi-modal Fusion)** — kết hợp audio và visual thông qua Cross-Attention.

---

## Cấu trúc thư mục

```
.
├── src/
│   ├── pipeline/           # 4 bước xử lý dữ liệu chính
│   │   ├── 00_fetch_youtube_urls.py
│   │   ├── 01_download.py
│   │   ├── 02_quality_gate.py
│   │   └── 03_cut_clips.py
│   ├── model/              # Định nghĩa mô hình chính (đang phát triển)
│   ├── train/              # Training pipeline chính
│   ├── eval/               # Evaluation
│   └── utils/
├── PoC/                    # Proof-of-concept PAMF (đã hoạt động)
│   ├── src/
│   │   ├── feature_extractor.py  # Wav2Vec2 + MobileNetV2
│   │   ├── fusion_model.py       # PAMF Cross-Attention
│   │   ├── train_and_eval.py
│   │   ├── inference.py
│   │   ├── extract_all.py
│   │   └── data_maker.py
│   ├── data/
│   │   ├── raw/            # Video thật (real_01..10.mp4)
│   │   ├── pseudo_fake/    # Video giả (fake_01..10.mp4)
│   │   └── features/       # Feature vectors đã trích xuất (.pt)
│   └── checkpoints/        # pamf_poc_model.pth
├── data/
│   ├── raw/                # Video gốc tải về (không commit)
│   ├── clips/              # Clip đã cắt (không commit)
│   ├── fake/               # Video deepfake
│   ├── youtube_tier1_urls.csv
│   ├── tier1_quality_gate_passed.csv
│   └── label.csv
├── configs/
├── docs/
├── experiments/
├── notebooks/
├── yolov8n-face.pt         # Đặt ở root, dùng cho quality gate
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

Chạy theo đúng thứ tự từ `src/pipeline/`:

### Bước 0 — Fetch YouTube URLs
```bash
python src/pipeline/00_fetch_youtube_urls.py
```
Yêu cầu `YOUTUBE_API_KEY` trong `.env`. Output: `data/youtube_tier1_urls.csv`.

### Bước 1 — Download video
```bash
python src/pipeline/01_download.py
```
Dùng `yt-dlp`. Hỏi tier khi chạy. Output vào `data/raw/tier{N}/`.

### Bước 2 — Quality Gate
```bash
python src/pipeline/02_quality_gate.py
```

Tiêu chí lọc (video phải pass cả 3):
| Tiêu chí | Ngưỡng |
|---|---|
| Độ phân giải | ≥ 480p |
| FPS | ≥ 24 |
| SNR âm thanh | ≥ 15 dB |
| Face detection | Có khuôn mặt (YOLOv8n-face, 5 frames mẫu) |

Output: `data/tier{N}_quality_gate_passed.csv`.

### Bước 3 — Cắt clip
```bash
python src/pipeline/03_cut_clips.py
```
Clip tiêu chuẩn: **5–15 giây**. Một video dài có thể cắt thành 3–4 clip.

---

## Cài đặt môi trường

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Yêu cầu hệ thống:**
- Python 3.10+
- `ffmpeg` có trong PATH (dùng cho quality gate và PoC inference)
- `yolov8n-face.pt` đặt tại thư mục root
- File `.env` với `YOUTUBE_API_KEY=...`

**Thư viện chính:** PyTorch 2.12, torchvision 0.27, transformers (Wav2Vec2), ultralytics 8.4, librosa 0.11, opencv-python 4.13, yt-dlp.

---

## Lưu ý kỹ thuật quan trọng

### Đặc thù tiếng Việt
- **Không dùng stemming kiểu tiếng Anh** — sẽ làm mất ngữ nghĩa hoàn toàn do hệ thống dấu đặc thù.
- **Thanh hỏi/ngã** có khoảng trống âm học đặc biệt cần chú ý khi xử lý prosody features.
- Audio backbone phải là model **Vietnamese-specific** (`wav2vec2-base-vietnamese-250h`), không dùng English model.
- Biên độ tần số tiếng Việt rộng hơn tiếng Anh (12–30 Hz), cần xác nhận lại độ phù hợp của PAMF.

### Metrics đánh giá
Cần bổ sung ít nhất **4 metrics** vào báo cáo (xem docs). **Cosine similarity không phù hợp** cho bài toán này.

### Baseline models
Đưa vào so sánh: `wav2vec`, `sadtalker`. Lưu ý: các model này hiện chỉ hoạt động tốt trên tiếng Anh.

### Tạo fake data
- **VEO3**: tạo video AI với khuôn mặt người nổi tiếng/vlogger thực tế.
- **CapCut noise**: gây nhiễu khuôn mặt bằng hiệu ứng méo/nhiễu, thời lượng ~15s.
- Đối tượng thực tế dùng **H.264** để né detection (hình ảnh mờ và nhiễu hơn).

### Dữ liệu & nhãn
- Chiếm 80–90% khối lượng công việc — ưu tiên làm data và label trước.
- Nguồn: YouTube và TikTok.
- `data/raw/` và `data/clips/` không được commit lên git.

---

## File quan trọng

| File | Mục đích |
|---|---|
| [PoC/src/fusion_model.py](PoC/src/fusion_model.py) | Định nghĩa PAMF_Fusion (Cross-Attention) |
| [PoC/src/feature_extractor.py](PoC/src/feature_extractor.py) | Wav2Vec2 + MobileNetV2 extractor |
| [PoC/src/inference.py](PoC/src/inference.py) | Chạy inference trên video .mp4 |
| [src/pipeline/02_quality_gate.py](src/pipeline/02_quality_gate.py) | Lọc video theo 3 tiêu chí |
| [yolov8n-face.pt](yolov8n-face.pt) | Face detection model (root) |
| [data/label.csv](data/label.csv) | File nhãn chính |