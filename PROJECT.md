# PROJECT.md — trạng thái hiện hành

Cập nhật 13/09/2026. Hướng hiện hành: **lip-sync audio–visual tiếng Việt**, so X/Y trên protocol tách người/nguồn, generator chưa thấy và nén.

## Đã chuyển đổi

- Nhánh làm việc: `codex/research-reset`; giữ lịch sử Git và thay đổi trước dọn trong archive/backup.
- Media, cache và population cũ đã được đưa ra khỏi workspace sau đối chiếu backup. Dữ liệu nguồn mới chưa được thu thập.
- Báo cáo, metadata và bằng chứng AVSP-Net cũ ở [legacy_avsp](docs/archives/legacy_avsp/README.md). Số đo cũ không áp dụng cho hướng mới.
- Công cụ cắt/review/quality được giữ, cập nhật đường dẫn. Preview không còn import bộ feature AVSP-Net.
- Split real theo nhóm người/nguồn được tách riêng, bỏ điều kiện đủ bốn fake. Có kiểm tra nguồn audio và generator giữ riêng cho biến thể.
- Nén dùng media primitives được giữ lại, CLI mới dùng output bất biến, CRF tường minh, nhãn/split/provenance và log lỗi.
- Đã tạo khung generators/features/models/training và tài liệu evaluation. Đây là bố trí thư mục, chưa phải triển khai generator, backbone, detector hoặc train loop; xem [src/README.md](src/README.md).

## Bảng tiến trình

| Hạng mục | Trạng thái |
|---|---|
| Kiểm tra giấy phép theo video | Có CLI lấy metadata và trạng thái pending; chưa gọi API trên nguồn mới. |
| Thu thập / downloader tuyển chọn | Có file chạy 01/02; nhận URL video/playlist, không bắt buộc kiểm tra giấy phép; chưa tải nguồn mới. |
| Cắt clip / manifest / review | Đã chuyển công cụ và kiểm tra kỹ thuật; chưa chứng nhận chất lượng nguồn mới. |
| Chia split / quan hệ biến thể | Đã triển khai kiểm tra tự động và test; chưa khóa dataset nghiên cứu. |
| Active-speaker / gom người tự động | Công cụ tùy chọn được giữ; chưa chạy lại model ngoài hoặc calibrate ngưỡng cho population mới. |
| Nén real/fake | Có CLI và smoke bằng media tổng hợp, không phải kết quả nghiên cứu. |
| Generator Wav2Lip/MuseTalk/ứng viên giữ riêng | Đề xuất; chưa tích hợp và benchmark. |
| AV-HuBERT | Ứng viên; chưa nạp checkpoint trên luồng mới. |
| Detector Y và cải tiến X | Thiết kế; chưa triển khai/train. |
| Final test và demo detector | Chưa thực hiện. |

## Quyết định nghiên cứu

- Giữ audio–visual làm trung tâm. Tìm tín hiệu hình ảnh hoặc âm đặc biệt là phân tích hỗ trợ.
- Khởi đầu với backbone giữ nguyên và bộ phân loại nhẹ. Đo riêng chỉ hình ảnh, chỉ audio và kết hợp trước khi thêm cross-attention.
- X/Y dùng cùng kiến trúc, các bản nén, dữ liệu và lịch train; X chỉ thêm yêu cầu nhất quán.
- Chọn model/tham số/ngưỡng trên validation. Pilot đã dùng chọn cấu hình là dữ liệu phát triển.
- Giữ một generator và mức nén cuối để đánh giá sau khi khóa phương pháp; không dò final test để chọn thắng lợi.
- Quy mô, nguồn cụ thể, giấy phép và GPU còn phải được xác minh bằng pilot. Chưa có bằng chứng tiết kiệm 70% công data.

## Pipeline đang chạy được

URL video/playlist đã chọn → 01 thu thập danh sách → 02 tải → 03 cắt clip → 04 gộp manifest → các bước review + xác định người/nguồn → 05 chia split. Các file chạy có số ở src/data và src/tools/review; cấu hình chính ở src/data/preparation/settings.py, mặc định dataset_v1. Kiểm tra giấy phép tách riêng, hiện không chặn download.

`src/data/preparation/compress.py` nhận master đã có nhãn/split để tạo bản nén. Chưa có generator nối tự động từ split tới fake. Tất cả lệnh phải nhận dataset/đường dẫn mới, không lấy manifest từ archive làm mặc định.

Hướng dẫn: [src/data/README.md](src/data/README.md), [review](src/tools/README.md), [DATA_PROTOCOL](docs/research/DATA_PROTOCOL.md).

## Việc kế tiếp

1. Chọn và rà một nhóm nguồn đa dạng, quyền sử dụng và host/người trùng.
2. Dựng pilot nhỏ với nguồn/người độc lập; chưa mở final test.
3. Dựng môi trường AV-HuBERT riêng và tích hợp một generator trên mẫu nhỏ.
4. Đo chi phí và các baseline đơn luồng/kết hợp; quyết định tiếp tục backbone trước X/Y.

[Báo cáo chuyển đổi](docs/reports/2026-09-13_REPO_RESET.md) ghi kiểm tra thực tế và giới hạn tại thời điểm reset. Cách chạy mới: [pipeline có số](src/data/README.md). Không tự mở full download/train từ kế hoạch này.
