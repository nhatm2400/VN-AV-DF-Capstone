# Dữ liệu theo phiên bản: dataset_v1 và dataset_v2

`dataset_v1` là bộ sơ bộ đã cắt, review và chia split trên một người. Nguồn mở rộng hiện ở `dataset_v2`; để chạy các bước cho V2, đặt `DATASET_VERSION = 'dataset_v2'` trong `src/data/preparation/settings.py`. Máy kiểm kê đã đặt local nhưng chưa commit. Tên phiên bản không chứng nhận dataset đã hoàn tất. Bản AVSP-Net cũ vẫn ở backup/archive.

```text
data/
├── sources/<dataset_version>/
│   ├── videos.csv                  # URL video/playlist đã chọn
│   ├── selected_videos.csv         # Bước 01: chuẩn hóa và mở rộng playlist
│   └── rights.csv                  # Tùy chọn, khi kiểm tra giấy phép
├── raw/<dataset_version>/<download_run>/
│   ├── <video_id>.mp4              # Bước 02: video gốc có audio
│   └── download_results.csv        # Trạng thái thành công/lỗi từng video
├── real/<dataset_version>/
│   └── <thư mục batch do cutter tạo>/
│       ├── media/*.mp4             # Bước 03: clip real đã cắt
│       ├── accepted_clips.csv
│       ├── rejected_windows.csv
│       ├── video_status.csv
│       └── run_summary.json        # Cùng config, checksum và log của cutter
├── manifests/<dataset_version>/
│   ├── clips.csv                   # Bước 04: clip + nguồn, chưa review
│   ├── reviews/
│   │   ├── assignments/            # assignment_<reviewer>.csv
│   │   ├── results/                # review_<reviewer>.csv
│   │   ├── merged/                 # Coverage, nhãn và mục cần phân xử
│   │   └── exports/                # Tùy chọn: media gom theo reviewer
│   ├── reviewed_clips.csv          # Clip giữ lại; bổ sung speaker_id trước split
│   ├── real_splits.csv             # Bước 05: train/val/test và group_id
│   └── masters.csv                 # Sau này: real/fake có nhãn/split/provenance
├── generated/<dataset_version>/    # Các run lip-sync fake/preview riêng
├── compressed/<dataset_version>/
│   └── compression_001/            # Bước 06: media nén, variants.csv, summary.json
└── rights_evidence/                # Hồ sơ giấy phép khi nhóm bổ sung
```

Các nhánh `real/`, `manifests/`, `generated/`, `compressed/` của V2 chỉ xuất hiện khi chạy bước tương ứng; hiện V2 có `sources/` và `raw/download_001/`. Ngày 23/09/2026, V2 có 1.954 nguồn đã chọn; trạng thái tải còn failed/pending, chưa sẵn sàng cho bước cắt. V1 có 2.252 clip real, `reviewed_clips.csv`, `real_splits.csv` và nhiều run generator thử nghiệm. `masters.csv` real/fake đã duyệt vẫn chưa có.

Không đặt nguồn vào `data/dataset/_V2`: các file chạy dựng đường dẫn từ `data/sources|raw|real|manifests/<dataset_version>`. Với nguồn mới trong V2, chỉnh `videos.csv` rồi chạy bước 01. `raw/dataset_v2/download_001/download_sources.csv` đang khóa danh sách 1.954 video; nếu `selected_videos.csv` thay đổi, đặt `DOWNLOAD_RUN` mới thay vì trộn với đợt tải đang dở. Giữ file `.part` để tiếp tục tải. Xem [kiểm kê dọn dẹp](../docs/reports/2026-09-23_WORKSPACE_CLEANUP_AUDIT.md) trước khi xóa bản export review hay output generator V1.

Giống repo cũ ở trình tự thu thập → tải → cắt → review → chia dữ liệu → sinh fake → nén. Khác ở việc thư mục tách theo loại dữ liệu, có phiên bản; không còn bốn loại pseudo-fake làm cấu trúc trung tâm. Thứ tự chạy nằm trong `src/data/01_...06_...`, không cần nhớ thứ tự qua tên thư mục dữ liệu.

Nhóm nguồn vẫn giữ trong cột tier (podcast, presentation, lecture hoặc nhóm do bạn chọn); không mất tính đa dạng khi media đặt chung thư mục. Chống leakage vẫn theo người/nguồn. Bản nén/fake kế thừa split của real.

Cache preview/đặc trưng tính lại nằm ở cache/, weights ở weights/, kết quả train/eval tương lai ở experiments/. Media, hồ sơ riêng và manifest làm việc mặc định không commit. Hướng dẫn bắt đầu: [pipeline](../src/data/README.md).
