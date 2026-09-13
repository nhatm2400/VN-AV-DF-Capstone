# VN-AV-DF-Capstone

Nghiên cứu phát hiện **lip-sync manipulation trong video một người nói tiếng Việt**, sử dụng audio và hình ảnh vùng miệng.

> Phương pháp X có cải thiện so với baseline Y trên người/nguồn độc lập, generator chưa thấy và video bị nén hay không?

Y là detector audio–visual; X dùng cùng detector, dữ liệu và ngân sách nhưng thêm yêu cầu nhất quán giữa các bản nén của cùng clip. AV-HuBERT là ứng viên cần kiểm chứng, chưa phải model đã chạy tốt trên tiếng Việt.

## Trạng thái

Repo đã chuyển sang nền xử lý dữ liệu cho hướng mới. Công cụ cắt, kiểm tra media, review, chia split và nén đã được tách khỏi pipeline AVSP-Net. **Dataset mới, generator lip-sync, AV-HuBERT và detector X/Y chưa được chạy/huấn luyện.** Không có kết quả accuracy mới.

Xem [PROJECT.md](PROJECT.md) để phân biệt phần đã triển khai và phần dự kiến. Kết quả pseudo-fake/AVSP-Net nằm trong [archive](docs/archives/legacy_avsp/README.md).

## Bắt đầu

**Luồng chính: mở file có số rồi Run Python File trong IDE.** Chọn interpreter `vn_av_df`, điền URL vào `data/sources/dataset_v1/videos.csv`, rồi chạy `src/data/01_collect.py` → `02_download.py` → `03_cut_clips.py` → `04_build_manifest.py`. Review bằng các file có số trong `src/tools/review/`, sau đó mới chia split ở bước 05. Kiểm tra giấy phép hiện là bước riêng tùy chọn.

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
| `src/models/`, `src/training/` | Khung cho cấu trúc detector và huấn luyện X/Y; chưa có implementation. |
| `src/evaluation/check_shortcuts.py` | Chẩn đoán metadata trên nhóm người/nguồn, không phải detector chính. |
| `configs/`, `environments/` | Mẫu đầu vào và hướng dẫn môi trường. |
| `data/`, `cache/`, `weights/`, `experiments/` | Vùng dữ liệu mới, cache, weights local và run mới; media không commit. |
| `docs/` | Nghiên cứu hiện hành, planning, báo cáo và archive cũ. |

Model/train/eval AVSP-Net cũ nằm trong backup và lịch sử Git, không còn trong src hiện hành. Đã tạo khung `src/features/`, `src/generators/`, `src/models/`, `src/training/` với README và __init__.py; chưa triển khai model mới. `src/evaluation/` hiện chỉ có chẩn đoán shortcut, chưa có evaluator detector mới. Xem [vai trò từng thư mục source](src/README.md).

## Dữ liệu và báo cáo

Nhóm nguồn phục vụ độ đa dạng; giấy phép được ghi riêng theo từng video. Không mặc định playlist VLR hoặc nội dung công khai có CC. Hồ sơ cần phân biệt nghiên cứu, cách lấy dữ liệu, tạo biến thể và phát hành media. Không phân phối lại video chỉ vì có file trong máy.

Mọi biến thể giữ nguồn và split của clip real. Người nói, tập nguồn và bản đăng lại liên quan phải cùng nhóm; cùng host có thể nối nhiều tập thành một nhóm lớn. Nén không làm clip real thành fake.

Tài liệu: [hướng nghiên cứu](docs/research/HUONG_NGHIEN_CUU_LIP_SYNC_VI.md), [protocol data](docs/research/DATA_PROTOCOL.md), [pilot](docs/planning/KE_HOACH_PILOT_LIP_SYNC_VI.md), [chỉ mục](docs/README.md).
