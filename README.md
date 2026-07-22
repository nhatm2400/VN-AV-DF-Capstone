# VN-AV-DF-Capstone

VN-AV-DF-Capstone xây dựng dữ liệu và mô hình phát hiện deepfake âm thanh–hình ảnh tiếng Việt, tập trung vào bằng chứng theo thời gian giữa tiếng nói, khẩu hình, chuyển động khuôn mặt và ngữ điệu.

## Đọc trước khi làm việc

- [CLAUDE.md](CLAUDE.md): quy tắc làm việc chung và quy ước bắt buộc của repository.
- [PROJECT.md](PROJECT.md): trạng thái hiện tại, cấu trúc source/data, pipeline và các lỗi đã biết.
- [Đề xuất AVSP-Net V2](docs/architecture/MODEL_PROPOSAL.md): kiến trúc V2a/V2b, loss, code layout, output contract và roadmap.
- [Báo cáo pilot gốc](docs/reports/PILOT_REPORT.md): toàn bộ quá trình chạy pilot V1.
- [Đánh giá V1 và kế hoạch V2](docs/reports/PILOT_V1_REVIEW_AND_V2_PLAN.md): diễn giải kết quả, blocking issues và thứ tự trước full.
- [Phase 0 temporal smoke](docs/reports/TEMPORAL_DESYNC_PHASE0_SMOKE.md): cơ chế sửa generator và bằng chứng test/smoke.

## Trạng thái ngắn

- Curation: 6.888 clip nguồn → 3.001 clip real sạch.
- Pseudo-fake V1: 4 method × 3.001 clip; temporal V1 được giữ làm lịch sử nhưng không dùng cho run mới.
- Split: real/fake ghép cặp cùng split; không trùng `speaker_id`/`source_video` theo metadata.
- Pilot AVSP-Net V1: 2.700 clip, test AUC 0,809.
- Temporal V2: generator + SNVSM H.264/AAC-16k-mono normalization đã implement; CRF ghép theo real nguồn, Stage 04 trim AAC padding, Stage 05 gate đủ method/audio/video/CRF. Synthetic 30/30, real smoke 18/18 và policy smoke r5 lịch sử đạt 30/30 media; r5 bị chặn ở paired timing tại checkpoint cơ chế và hiện còn bị reject vì thiếu structured timeline.
- Full feature/model: chưa chạy.
- Quyết định hiện tại: **NO-GO full**; structured schema đã khóa và code V2 của ba method không-temporal đã bỏ `-shortest`, qua synthetic media-contract test. Bước kế tiếp là stratified real-data smoke + metadata-shortcut gate; sau đó mới normalize + Stage 05 + metadata gate toàn pilot và extract 2.700 clip.

## Cấu trúc chính

```text
.
├── CLAUDE.md                    # Quy tắc làm việc
├── PROJECT.md                   # Tổng quan sống của project
├── README.md                    # Điểm vào tài liệu
├── docs/
│   ├── architecture/
│   │   └── MODEL_PROPOSAL.md    # AVSP-Net V2a/V2b
│   ├── reports/
│   │   ├── PILOT_REPORT.md
│   │   ├── PILOT_V1_REVIEW_AND_V2_PLAN.md
│   │   └── TEMPORAL_DESYNC_PHASE0_SMOKE.md
│   └── README.md                # Chỉ mục tài liệu
├── src/
│   ├── pipeline/                # Pipeline 01 → 05
│   ├── model/                   # Model hiện tại và V2 trong tương lai
│   ├── train/
│   ├── eval/
│   └── tools/
├── data/                        # Manifest và artifact dữ liệu
├── experiments/                 # Mỗi run pilot/full là một thư mục bất biến
└── PoC/                         # Proof of concept PAMF cũ
```

## Môi trường

Trên máy phát triển hiện tại, luôn gọi Python bằng đường dẫn:

```powershell
D:\Anaconda\envs\vn_av_df\python.exe
```

`ffmpeg` và `ffprobe` phải có trong `PATH`. Không tự chạy job dài như full extraction/full training nếu chưa có yêu cầu rõ; xem chi tiết trong [CLAUDE.md](CLAUDE.md).

## Pipeline

Pipeline được chạy từ root theo thứ tự:

```text
01_collect
-> 02_curate
-> 03_fake + 05_snvsm_compress
-> 05_build_labels (contract gate)
-> 04_extract_features
-> train/eval
```

Stage 04/05 phải dùng manifest SNVSM và Stage 05 phải pass trước khi launch extraction dài. Không dùng `--limit` để tạo pilot ghép cặp; phải tạo manifest pilot riêng. Các lệnh và data contract cụ thể nằm trong [PROJECT.md](PROJECT.md).

## Quy ước experiment

Mỗi lần chạy là một thư mục bất biến:

```text
experiments/<pilot|full>_<v1|v2a|v2b>_<timestamp>_<git-sha>_<config-hash>/
```

Run phải lưu config, checksum manifest, source state, log, checkpoint, metrics, predictions và plots. Không ghi đè run đã hoàn tất. Chi tiết tại [MODEL_PROPOSAL.md](docs/architecture/MODEL_PROPOSAL.md#18-experiment-output-bất-biến).
