# PROJECT.md — trạng thái hiện hành

Cập nhật 23/09/2026. Hướng hiện hành: **lip-sync audio–visual tiếng Việt**, so X/Y trên protocol tách người/nguồn, generator chưa thấy và nén. Giai đoạn sơ bộ `dataset_v1` chỉ có một người: dùng `source_disjoint_single_speaker` để chia theo nguồn; bổ sung người cho đánh giá speaker-disjoint sau. Nguồn mở rộng đang được thu thập riêng ở `dataset_v2`.

## Đã chuyển đổi

- Nhánh làm việc: `codex/research-reset`; giữ lịch sử Git và thay đổi trước dọn trong archive/backup.
- Media, cache và population cũ đã được đưa ra khỏi workspace sau đối chiếu backup. Dữ liệu mới đã được thu thập/cắt và đang review.
- Báo cáo, metadata và bằng chứng AVSP-Net cũ ở [legacy_avsp](docs/archives/legacy_avsp/README.md). Số đo cũ không áp dụng cho hướng mới.
- Công cụ cắt/review/quality được giữ, cập nhật đường dẫn. Preview không còn import bộ feature AVSP-Net.
- Split real theo nhóm người/nguồn được tách riêng, bỏ điều kiện đủ bốn fake. Có kiểm tra nguồn audio và generator giữ riêng cho biến thể.
- Nén dùng media primitives được giữ lại, CLI mới dùng output bất biến, CRF tường minh, nhãn/split/provenance và log lỗi.
- Đã tạo generators/features và bản detector/train_step/predict tối thiểu. Kiểm tra bằng tensor giả lập, chưa tích hợp generator/backbone/dataset thật; xem [src/README.md](src/README.md).

## Bảng tiến trình

| Hạng mục | Trạng thái |
|---|---|
| Kiểm tra giấy phép theo video | Có CLI lấy metadata và trạng thái pending; chưa gọi API trên nguồn mới. |
| Thu thập / downloader tuyển chọn | Có file chạy 01/02; nguồn mới đã được tải và đưa qua cắt clip/review. Kiểm tra giấy phép vẫn là bước riêng. |
| Cắt clip / manifest / review | Đã gộp đủ 2.252 quyết định của ba người: 1.143 Keep, 1.109 Reject, không còn uncertain; xuất reviewed_clips.csv. |
| Chia split / quan hệ biến thể | Đã chạy bước 05 cho bộ một người: 786 train, 192 val, 165 test; 18 nhóm nguồn, protocol source_disjoint_single_speaker. Chưa phải split nghiên cứu trên người chưa thấy. |
| Active-speaker / gom người tự động | Công cụ tùy chọn được giữ; chưa chạy lại model ngoài hoặc calibrate ngưỡng cho population mới. |
| Nén real/fake | Có CLI và smoke bằng media tổng hợp, không phải kết quả nghiên cứu. |
| Generator Wav2Lip/MuseTalk/ứng viên giữ riêng | MuseTalk 1.5 đã cài đủ và chạy xong smoke 5 clip. Bước 07 đã chạy `quality_10x2s_002`: 10 clip train × 2 giây, 1080p, chung frame/audio cho Wav2Lip GAN và MuseTalk; đủ 20 fake và 10 comparison, chờ review. Preprocessing vẫn inline; S3FD dùng ảnh nhỏ, tắt benchmark và giải phóng cache GPU thừa; ghép mềm vùng dưới mặt. 19 kiểm tra generator qua; chưa phải kết quả detector hay bằng chứng fake đủ tinh vi. Generator giữ riêng chưa tích hợp. |
| AV-HuBERT | Có 01_extract: loader Base pre-fusion, xử lý miệng/audio cùng timeline và xuất cache/pairs.csv. Đã cài dlib CPU 20.0.1, nạp checkpoint và chạy một clip thật: cửa sổ 2 giây, 50 frame, không nội suy landmark, hai feature [50,768] hữu hạn. Kết quả ở cache/features/dataset_v1/real_smoke_001. Chưa benchmark toàn bộ dữ liệu; không dùng Transformer hợp nhất. |
| Detector Y và cải tiến X | Có detector, loader feature pairs, train/validation theo epoch và lưu checkpoint; X thêm consistency. Chỉ kiểm chứng bằng feature giả lập, chưa train nghiên cứu. |
| Final test và demo detector | Chưa thực hiện. |
| Nguồn mở rộng `dataset_v2` | Máy kiểm kê đang đặt `src/data/preparation/settings.py` thành `dataset_v2` (thay đổi local chưa commit); `data/sources/dataset_v2/selected_videos.csv` và `data/raw/dataset_v2/download_001/` đã có. Ngày 23/09: 1.954 nguồn đã chọn, 186 downloaded, 475 failed, 1.293 pending theo `download_results.csv`; chưa cắt/review/chia split V2. Đây là ảnh chụp trạng thái khi audit, không phải tổng kết đợt tải. |

## Quyết định nghiên cứu

- Giữ audio–visual làm trung tâm. Tìm tín hiệu hình ảnh hoặc âm đặc biệt là phân tích hỗ trợ.
- Khởi đầu với backbone giữ nguyên và bộ phân loại nhẹ. Đo riêng chỉ hình ảnh, chỉ audio và kết hợp trước khi thêm cross-attention.
- X/Y dùng cùng kiến trúc, các bản nén, dữ liệu và lịch train; X chỉ thêm yêu cầu nhất quán.
- Chọn model/tham số/ngưỡng trên validation. Pilot đã dùng chọn cấu hình là dữ liệu phát triển.
- Giữ một generator và mức nén cuối để đánh giá sau khi khóa phương pháp; không dò final test để chọn thắng lợi.
- Quy mô, nguồn cụ thể, giấy phép và GPU còn phải được xác minh bằng pilot. Chưa có bằng chứng tiết kiệm 70% công data.

## Pipeline đang chạy được

URL video/playlist đã chọn → 01 thu thập danh sách → 02 tải → 03 cắt clip → 04 gộp manifest → các bước review + xác định người/nguồn → 05 chia split. Các file chạy có số ở src/data và src/tools/review; để chạy V2, đặt `DATASET_VERSION = 'dataset_v2'` trong `src/data/preparation/settings.py`. Máy kiểm kê đã đặt như vậy nhưng thay đổi này chưa commit. `dataset_v1` vẫn giữ manifest, clip và các run thử của giai đoạn sơ bộ. Generator Wav2Lip/MuseTalk có cấu hình riêng vẫn trỏ `dataset_v1`; không tự chuyển theo settings của pipeline data. Kiểm tra giấy phép tách riêng, hiện không chặn download.

`src/data/preparation/compress.py` nhận master đã có nhãn/split để tạo bản nén. Chưa có generator nối tự động từ split tới fake. Tất cả lệnh phải nhận dataset/đường dẫn mới, không lấy manifest từ archive làm mặc định.

Hướng dẫn: [src/data/README.md](src/data/README.md), [review](src/tools/README.md), [DATA_PROTOCOL](docs/research/DATA_PROTOCOL.md).

## Sau review 1: dữ liệu và dọn workspace

- Tiếp tục nguồn mới trong `data/sources/dataset_v2/` và `data/raw/dataset_v2/<DOWNLOAD_RUN>/`; các bước sau sẽ sinh `data/real/dataset_v2/`, `data/manifests/dataset_v2/` theo cùng tên phiên bản. Không dùng thư mục `data/dataset/_V2` vì code hiện không đọc đường dẫn đó.
- `download_001` đang khóa danh sách nguồn bằng `download_sources.csv`. Tiếp tục cùng danh sách thì giữ `DOWNLOAD_RUN`; nếu thêm/bớt nguồn trong `selected_videos.csv`, dùng run tải mới. Chưa chạy bước 03 khi `download_results.csv` còn failed/pending.
- Không coi bất kỳ output generator V1 nào là dataset fake đã duyệt: các candidates còn `awaiting_visual_review`, và `quality_10x2s_002` mới được kiểm tra kỹ thuật 40 video. Giữ run cần review trước khi dọn.
- [Kiểm kê dọn dẹp 23/09/2026](docs/reports/2026-09-23_WORKSPACE_CLEANUP_AUDIT.md) phân biệt bản nguồn/nhãn phải giữ, media xuất để review có thể tái tạo, run thử cũ và cache. Chưa xóa dữ liệu trong lần kiểm kê này.

## Việc kế tiếp

1. Review đã hoàn tất. Người dùng xác nhận mọi video cùng một người; bước 05 được cấu hình chia theo nguồn cho kết quả sơ bộ, không tuyên bố speaker-disjoint.
2. Rà nguồn trùng và quyền sử dụng trước khi tạo fake; bổ sung người sau kết quả sơ bộ AV-HuBERT và tạo phiên bản split nghiên cứu mới.
3. Bước kiểm tra extraction một clip thật đã qua; tiếp theo kiểm tra vùng miệng và tỷ lệ lỗi trên một nhóm nhỏ đa dạng trước khi chạy hàng loạt.
4. Tích hợp một generator trên mẫu phát triển nhỏ; đo chi phí và baseline đơn luồng/kết hợp trước X/Y. Chưa mở final test.

[Báo cáo chuyển đổi](docs/reports/2026-09-13_REPO_RESET.md) ghi kiểm tra thực tế và giới hạn tại thời điểm reset. Cách chạy mới: [pipeline có số](src/data/README.md). Không tự mở full download/train từ kế hoạch này.
