# VN-AV-DF-Capstone

VN-AV-DF-Capstone xây dựng dữ liệu và mô hình phát hiện deepfake âm thanh–hình ảnh tiếng Việt, tập trung vào bằng chứng theo thời gian giữa tiếng nói, khẩu hình, chuyển động khuôn mặt và ngữ điệu.

## Đọc trước khi làm việc

- [CLAUDE.md](CLAUDE.md): quy tắc làm việc chung và quy ước bắt buộc của repository.
- [PROJECT.md](PROJECT.md): trạng thái hiện tại, cấu trúc source/data, pipeline và các lỗi đã biết.
- [Đề xuất AVSP-Net V2](docs/architecture/MODEL_PROPOSAL.md): kiến trúc V2a/V2b, loss, code layout, output contract và roadmap.
- [Báo cáo pilot gốc](docs/reports/PILOT_REPORT.md): toàn bộ quá trình chạy pilot V1.
- [Đánh giá V1 và kế hoạch V2](docs/reports/PILOT_V1_REVIEW_AND_V2_PLAN.md): diễn giải kết quả, blocking issues và thứ tự trước full.

## Trạng thái ngắn

- Curation: 6.888 clip nguồn → 3.001 clip real sạch.
- Pseudo-fake: 4 method × 3.001 clip; SNVSM đã đồng bộ codec real/fake.
- Split: real/fake ghép cặp cùng split; không trùng `speaker_id`/`source_video` theo metadata.
- Pilot AVSP-Net V1: 2.700 clip, test AUC 0,809.
- Full feature/model: chưa chạy.
- Quyết định hiện tại: **NO-GO full V1**; phải sửa `temporal_desync`, triển khai V2a và chạy lại pilot diagnostic.

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
│   │   └── PILOT_V1_REVIEW_AND_V2_PLAN.md
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
-> 04_extract_features
-> 05_build_labels
-> train/eval
```

Stage 04/05 phải dùng manifest SNVSM. Không dùng `--limit` để tạo pilot ghép cặp; phải tạo manifest pilot riêng. Các lệnh và data contract cụ thể nằm trong [PROJECT.md](PROJECT.md).

## Quy ước experiment

Mỗi lần chạy là một thư mục bất biến:

```text
experiments/<pilot|full>_<v1|v2a|v2b>_<timestamp>_<git-sha>_<config-hash>/
```

Run phải lưu config, checksum manifest, source state, log, checkpoint, metrics, predictions và plots. Không ghi đè run đã hoàn tất. Chi tiết tại [MODEL_PROPOSAL.md](docs/architecture/MODEL_PROPOSAL.md#18-experiment-output-bất-biến).
