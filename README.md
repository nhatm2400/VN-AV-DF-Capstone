# VN-AV-DF-Capstone

Nghiên cứu phát hiện **lip-sync manipulation trong video một người nói tiếng Việt**, sử dụng audio và hình ảnh vùng miệng.

> Phương pháp X có cải thiện so với baseline Y trên người/nguồn độc lập, generator chưa thấy và video bị nén hay không?

Y là detector audio–visual; X dùng cùng detector, dữ liệu và ngân sách nhưng thêm yêu cầu nhất quán giữa các bản nén của cùng clip. AV-HuBERT là ứng viên cần kiểm chứng, chưa phải model đã chạy tốt trên tiếng Việt.

## Trạng thái

Repo đã chuyển sang nền xử lý dữ liệu cho hướng mới. Công cụ cắt, kiểm tra media, review, chia split và nén đã được tách khỏi pipeline AVSP-Net. `dataset_v1` đã qua review clip và chia split sơ bộ trên một người; Wav2Lip/MuseTalk mới có các run xem chất lượng, chưa duyệt làm dataset fake. Nguồn mở rộng đang tải trong `dataset_v2`. Hai nhánh AV-HuBERT đã nạp checkpoint và trích feature từ một clip thật thành công. **Chưa có thí nghiệm huấn luyện/đánh giá detector X/Y trên dataset thật.** Không có kết quả accuracy mới.

Xem [PROJECT.md](PROJECT.md) để phân biệt phần đã triển khai và phần dự kiến. Kết quả pseudo-fake/AVSP-Net nằm trong [archive](docs/archives/legacy_avsp/README.md).

## Bắt đầu

**Luồng chính: mở file có số rồi Run Python File trong IDE.** Chọn interpreter `vn_av_df`; đường dẫn dữ liệu do `src/data/preparation/settings.py` quyết định. Để chạy V2, đặt `DATASET_VERSION = 'dataset_v2'` (máy kiểm kê đã đặt local, chưa commit): URL nằm ở `data/sources/dataset_v2/videos.csv`, danh sách đã mở rộng ở `selected_videos.csv`, và bước 02 đang ghi `data/raw/dataset_v2/download_001/`. Khi tải đủ và xử lý lỗi, chạy `src/data/03_cut_clips.py` → `04_build_manifest.py`, review bằng các file có số trong `src/tools/review/`, rồi chia split ở bước 05. Bước 01/02 dùng để thu thập hoặc tiếp tục tải; không chạy lại bước 01 với danh sách khác rồi tiếp tục vào `download_001` đã khóa nguồn. Kiểm tra giấy phép hiện là bước riêng tùy chọn.

Xem [thứ tự và đầu vào/đầu ra từng bước](src/data/README.md), [cấu trúc data hoàn chỉnh](data/README.md) và [tests theo nhóm](tests/README.md). File cấu hình chung là [settings.py](src/data/preparation/settings.py). Các lệnh CLI bên dưới chỉ là lựa chọn khi cần thao tác terminal.

Chọn interpreter Python của môi trường data đã chuẩn bị; phiên bản tham chiếu và giới hạn kiểm chứng nằm trong [environments/README.md](environments/README.md). Lệnh bên dưới chạy từ root repo; không tự tải dataset/model.

```powershell
python -m unittest discover -s tests -q
python -m src.data.preparation.build_splits --help
python -m src.data.preparation.compress --help
python -m src.data.preparation.download --help
python src/data/preparation/cut_clips.py --help
python src/data/preparation/build_manifest.py --help
python src/tools/review/clip_review.py --help
```

Đọc [hướng dẫn công cụ data](src/data/README.md) trước khi chạy với dữ liệu thật. `configs/templates/` chứa CSV mẫu chỉ có header, không phải dataset đã duyệt. File config cắt mẫu chưa điền số nguồn nên cố ý từ chối chạy.

## Cấu trúc đang có

| Đường dẫn | Nội dung |
|---|---|
| `src/data/` | File chạy 01–06 ở ngoài cùng; code hỗ trợ và settings.py nằm trong `preparation/`. |
| `src/data/preparation/quality/` | Đo mặt, gom người và active-speaker tùy chọn; cần kiểm chứng ngưỡng trên nguồn mới. |
| `src/tools/review/` | Preview miệng có tiếng, review, chia/gộp công việc. |
| `src/generators/`, `src/features/` | Khung cho tạo lip-sync fake và chuẩn bị/trích đặc trưng audio–visual. |
| `src/models/`, `src/training/` | Detector audio–visual tối thiểu, train_step Y/X và smoke giả lập; chưa nối backbone/dataset thật. |
| `src/evaluation/check_shortcuts.py` | Chẩn đoán metadata trên nhóm người/nguồn, không phải detector chính. |
| `configs/`, `environments/` | Mẫu đầu vào và hướng dẫn môi trường. |
| `data/`, `cache/`, `weights/`, `experiments/` | Vùng dữ liệu mới, cache, weights local và run mới; media không commit. |
| `docs/` | Nghiên cứu hiện hành, planning, báo cáo và archive cũ. |

Các phiên bản dataset nằm song song theo loại dữ liệu: `data/sources/dataset_v2/`, `data/raw/dataset_v2/`, rồi `data/real/dataset_v2/` và `data/manifests/dataset_v2/` khi chạy các bước sau. `data/dataset/_V2` không thuộc cấu trúc mà pipeline hiện đọc. `dataset_v1` giữ clip real, quyết định review và split sơ bộ; tránh trộn hai phiên bản hoặc đổi tên thư mục V1. Generator có settings riêng hiện vẫn dùng V1. Xem [kiểm kê dọn workspace sau review 1](docs/reports/2026-09-23_WORKSPACE_CLEANUP_AUDIT.md) trước khi bỏ media/cache của các run thử.

Model/train/eval AVSP-Net cũ nằm trong backup và lịch sử Git. Bản model mới có detector, loader feature pairs, training Y/X và predict_batch; mở [01_smoke_test.py](src/models/01_smoke_test.py) để thử bằng tensor giả lập. [01_extract.py](src/features/01_extract.py) đã nối loader checkpoint Base pre-fusion và xử lý media vào cache cho training; xem [cách đặt weights và chạy extraction](src/features/README.md). Đã chạy end-to-end trên một clip thật với dlib/weights thật, chưa benchmark dataset; generator và evaluator theo protocol chưa tích hợp. Môi trường model tối thiểu ở [requirements-model.txt](environments/requirements-model.txt), bổ sung cho extractor ở [requirements-features.txt](environments/requirements-features.txt); chưa thử cài mới từ đầu. Xem [vai trò từng thư mục source](src/README.md).

## Dữ liệu và báo cáo

Nhóm nguồn phục vụ độ đa dạng; giấy phép được ghi riêng theo từng video. Không mặc định playlist VLR hoặc nội dung công khai có CC. Hồ sơ cần phân biệt nghiên cứu, cách lấy dữ liệu, tạo biến thể và phát hành media. Không phân phối lại video chỉ vì có file trong máy.

Mọi biến thể giữ nguồn và split của clip real. Người nói, tập nguồn và bản đăng lại liên quan phải cùng nhóm; cùng host có thể nối nhiều tập thành một nhóm lớn. Nén không làm clip real thành fake.

Tài liệu: [hướng nghiên cứu](docs/research/HUONG_NGHIEN_CUU_LIP_SYNC_VI.md), [protocol data](docs/research/DATA_PROTOCOL.md), [pilot](docs/planning/KE_HOACH_PILOT_LIP_SYNC_VI.md), [chỉ mục](docs/README.md).
