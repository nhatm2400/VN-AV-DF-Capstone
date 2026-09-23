# Vai trò các thư mục source

`src/data/` chứa toàn bộ luồng dữ liệu: file chạy có số, cấu hình và code xử lý. Đã gộp thư mục pipeline vào đây.

Để tải, mở `data/02_download.py` rồi Run. Ngoài cùng data chỉ có các file Python có số; code hỗ trợ và cấu hình nằm trong `data/preparation/`, gồm `settings.py` và nhánh quality. Không cần chạy lần lượt các file hỗ trợ.

| Thư mục | Vai trò | Hiện trạng |
|---|---|---|
| [data/](data/README.md) | Các bước chạy 01–06, settings và logic thu thập, tải, cắt, manifest, split, nén, quality | Có code |
| [tools/review/](tools/review/README.md) | Xem/nghe clip, phân công và gộp review | Có code và file chạy có số |
| [generators/](generators/README.md) | Tạo lip-sync fake | Wav2Lip/MuseTalk đã chạy các batch xem chất lượng trên V1; output còn chờ review, chưa nối vào dataset huấn luyện |
| [features/](features/README.md) | Chuẩn bị audio/vùng miệng, trích đặc trưng cho model | 01_extract: loader Base pre-fusion, preprocessing, cache cho training; đã chạy một clip thật với pretrained/landmark thật |
| [models/](models/README.md) | Detector nhận audio/visual features | Có detector.py và 01_smoke_test.py; backbone chưa có |
| [training/](training/README.md) | Học nhãn và consistency cho X/Y | Loader paired features, train/validation theo epoch, best checkpoint; kiểm chứng giả lập |
| [evaluation/](evaluation/README.md) | Dự đoán và đánh giá | Có predict_batch, check_shortcuts.py; metrics/protocol chưa có |

Luồng dự kiến: dữ liệu đã chia split → tạo fake → tạo các bản nén → chuẩn bị audio/visual → train detector → đánh giá. Generator đã có adapter và run thử V1; extractor đã chạy một clip thật. Chưa có dataset fake đã duyệt hay thí nghiệm train/eval detector trên dataset thật. Để chạy pipeline data V2, đặt `DATASET_VERSION = 'dataset_v2'` trong `data/preparation/settings.py`; generator có cấu hình V1 riêng.

Video ở `data/`, đặc trưng tính sẵn ở `cache/`, weights tải về ở `weights/`, đầu ra train/eval ở `experiments/`. Các đường dẫn này nằm ở root repo, không nằm trong `src/`.
