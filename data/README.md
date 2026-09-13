# Dữ liệu chính thức đang xây dựng: dataset_v1

`pilot_v1` đã đổi thành `dataset_v1`, giữ nguyên videos.csv bạn đang điền. Tên phiên bản không phải chứng nhận dataset đã hoàn tất. Dữ liệu cũ vẫn ở backup, manifest lịch sử ở docs/archives/legacy_avsp/manifests.

```text
data/
├── sources/dataset_v1/
│   ├── videos.csv                  # Điền URL video/playlist đã chọn
│   ├── selected_videos.csv         # Bước 01: chuẩn hóa và mở rộng playlist
│   └── rights.csv                  # Tùy chọn, khi kiểm tra giấy phép sau này
├── raw/dataset_v1/download_001/
│   ├── <video_id>.mp4              # Bước 02: video gốc có audio
│   └── download_results.csv        # Trạng thái thành công/lỗi từng video
├── real/dataset_v1/
│   └── <thư mục batch do cutter tạo>/
│       ├── media/*.mp4             # Bước 03: clip real đã cắt
│       ├── accepted_clips.csv
│       ├── rejected_windows.csv
│       ├── video_status.csv
│       └── run_summary.json        # Cùng config, checksum và log của cutter
├── manifests/dataset_v1/
│   ├── clips.csv                   # Bước 04: clip + nguồn, chưa review
│   ├── reviews/
│   │   ├── assignments/            # assignment_<reviewer>.csv
│   │   ├── results/                # review_<reviewer>.csv
│   │   ├── merged/                 # Coverage, nhãn và mục cần phân xử
│   │   └── exports/                # Tùy chọn: media gom theo reviewer
│   ├── reviewed_clips.csv          # Clip giữ lại; bổ sung speaker_id trước split
│   ├── real_splits.csv             # Bước 05: train/val/test và group_id
│   └── masters.csv                 # Sau này: real/fake có nhãn/split/provenance
├── generated/dataset_v1/           # Dành cho lip-sync fake; generator chưa có
├── compressed/dataset_v1/
│   └── compression_001/            # Bước 06: media nén, variants.csv, summary.json
└── rights_evidence/                # Hồ sơ giấy phép khi nhóm bổ sung
```

Các thư mục nền đã tạo để dễ nhìn; những file đầu ra trong sơ đồ **chỉ xuất hiện sau khi chạy bước tương ứng**. Hiện videos.csv vẫn chỉ có header, chưa có video mới. generated/ và masters.csv là phần chờ triển khai generator.

Giống repo cũ ở trình tự thu thập → tải → cắt → review → chia dữ liệu → sinh fake → nén. Khác ở việc thư mục tách theo loại dữ liệu, có phiên bản; không còn bốn loại pseudo-fake làm cấu trúc trung tâm. Thứ tự chạy nằm trong `src/data/01_...06_...`, không cần nhớ thứ tự qua tên thư mục dữ liệu.

Nhóm nguồn vẫn giữ trong cột tier (podcast, presentation, lecture hoặc nhóm do bạn chọn); không mất tính đa dạng khi media đặt chung thư mục. Chống leakage vẫn theo người/nguồn. Bản nén/fake kế thừa split của real.

Cache preview/đặc trưng tính lại nằm ở cache/, weights ở weights/, kết quả train/eval tương lai ở experiments/. Media, hồ sơ riêng và manifest làm việc mặc định không commit. Hướng dẫn bắt đầu: [pipeline](../src/data/README.md).
