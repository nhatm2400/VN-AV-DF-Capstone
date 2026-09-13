# Vai trò các thư mục source

`src/data/` chứa toàn bộ luồng dữ liệu: file chạy có số, cấu hình và code xử lý. Đã gộp thư mục pipeline vào đây.

Để tải, mở `data/02_download.py` rồi Run. Ngoài cùng data chỉ có các file Python có số; code hỗ trợ và cấu hình nằm trong `data/preparation/`, gồm `settings.py` và nhánh quality. Không cần chạy lần lượt các file hỗ trợ.

| Thư mục | Vai trò | Hiện trạng |
|---|---|---|
| [data/](data/README.md) | Các bước chạy 01–06, settings và logic thu thập, tải, cắt, manifest, split, nén, quality | Có code |
| [tools/review/](tools/review/README.md) | Xem/nghe clip, phân công và gộp review | Có code và file chạy có số |
| [generators/](generators/README.md) | Tạo lip-sync fake | Khung thư mục |
| [features/](features/README.md) | Chuẩn bị audio/vùng miệng, trích đặc trưng cho model | Khung thư mục |
| [models/](models/README.md) | Cấu trúc detector và backbone | Khung thư mục |
| [training/](training/README.md) | Đọc batch, train, validation, lưu checkpoint | Khung thư mục |
| [evaluation/](evaluation/README.md) | Đánh giá theo protocol | Có check_shortcuts.py; evaluator detector chưa có |

Luồng dự kiến của phần model: dữ liệu đã chia split → tạo fake → tạo các bản nén → chuẩn bị audio/visual → train detector → đánh giá. Các thư mục vừa tạo chỉ có README và __init__.py; chưa có model giả, checkpoint hay kết quả train.

Video ở `data/`, đặc trưng tính sẵn ở `cache/`, weights tải về ở `weights/`, đầu ra train/eval ở `experiments/`. Các đường dẫn này nằm ở root repo, không nằm trong `src/`.
