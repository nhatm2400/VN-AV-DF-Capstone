# VN-AV-DF-Capstone

Phát hiện **deepfake âm thanh–hình ảnh tiếng Việt** — nhận biết video giả mạo bằng cách đối chiếu tiếng nói với khẩu hình, chuyển động khuôn mặt và ngữ điệu theo thời gian, thay vì tìm dấu vết giả mạo trên từng khung hình riêng lẻ.

Đồ án tốt nghiệp, Đại học FPT.

## Bài toán

Phần lớn bộ phát hiện deepfake hiện có được huấn luyện trên dữ liệu tiếng Anh và chỉ nhìn hình ảnh. Với tiếng Việt, cách tiếp cận đó bỏ sót một tín hiệu quan trọng: tiếng Việt là **ngôn ngữ có thanh điệu** — cao độ (F0) không phải nét biểu cảm mà mang **nghĩa từ vựng**. "ma", "má", "mà", "mã", "mạ", "mả" khác nhau hoàn toàn chỉ bởi đường F0. Một hệ thống giả giọng làm sai đường thanh điệu sẽ tạo ra lỗi mà người Việt nghe ra ngay nhưng mô hình tiếng Anh không hề biết tới.

Trở ngại thứ hai: **không có bộ dữ liệu deepfake âm thanh–hình ảnh tiếng Việt nào công khai** để huấn luyện.

## Hướng tiếp cận

**Sinh pseudo-fake có kiểm soát.** Thay vì chờ dữ liệu deepfake thật, dự án tạo giả từ video thật bằng bốn phép biến đổi, mỗi phép tấn công đúng một kênh tín hiệu:

| Kênh | Phép biến đổi | Phá vỡ điều gì |
|---|---|---|
| Đồng bộ thời gian | `temporal_desync` — xoay vòng audio theo số sample chính xác | Quan hệ thời gian giữa tiếng và khẩu hình |
| Chuyển động hình | `frame_reverse` — đảo ngược một đoạn khung hình | Hướng chuyển động tự nhiên của môi |
| Ngữ điệu | `pitch_flatten` — làm phẳng F0 bằng PSOLA | Đường thanh điệu tiếng Việt |
| Danh tính hình | `anonymization` — làm mờ vùng mặt | Đặc trưng nhận dạng khuôn mặt |

Cách này cho **nhãn chính xác tuyệt đối** và cho phép đo riêng từng kênh — biết mô hình mạnh ở đâu, mù ở đâu, thay vì chỉ có một con số tổng.

**Chống học tắt.** Rủi ro lớn nhất của pseudo-fake là mô hình học đặc điểm phụ thay vì học bản chất. Dự án xử lý bằng ba lớp: chuẩn hoá lại codec **đối xứng** cho cả real lẫn fake (SNVSM) để xoá dấu vết nén; chia tập theo **thành phần liên thông** của `speaker_id ∪ source_video` để cùng một người không xuất hiện ở hai tập; và một **cổng kiểm tra metadata** — huấn luyện bộ phân loại chỉ dùng metadata container, nếu nó đạt AUC quá ngưỡng thì dữ liệu đã lộ đường tắt và pipeline dừng lại.

**Kiến trúc AVSP-Net.** Ba nhánh mã hoá — mouth ROI qua CNN + Transformer, tiếng nói qua wav2vec2 tiếng Việt, ngữ điệu qua Conv1D + BiGRU — hợp nhất bằng cross-attention với **audio làm Query**. Hai đầu ra: thật/giả, và phân loại độ lệch thời gian. Bản V1 có 2,29 triệu tham số.

## Trạng thái

Đã chạy pilot V1 trên 2.700 clip, đạt test ROC-AUC **0,809**. Nhưng phân tích theo từng kênh cho thấy con số tổng che giấu khoảng cách lớn: `pitch_flatten` 0,990 trong khi `frame_reverse` chỉ 0,535 — gần bằng đoán bừa. Kiểm tra đối kháng còn phát hiện các baseline tầm thường giải được hai kênh dễ, và một lỗi tạo tác trong generator temporal.

Kết luận hiện tại là **NO-GO** cho huấn luyện đầy đủ với V1. Stage 04 đã được cắt lại đủ ba tier và đạt coverage; bước kế tiếp là dựng lại curation, ROI review và assignment trên population mới trước khi tiếp tục fake V2.

Trạng thái chi tiết và luôn cập nhật: [PROJECT.md](PROJECT.md).

## Cấu trúc

```text
src/
├── pipeline/          # Pipeline dữ liệu 5 stage
│   ├── 01_collect/    #   Thu thập + cắt clip, tách theo tier nguồn (YouTube CC / YouTube / TikTok)
│   ├── 02_curate/     #   Lọc clip: đo mặt, gom speaker, loại rác
│   ├── 03_fake/       #   Sinh 4 loại pseudo-fake + chuẩn hoá codec + cổng metadata
│   ├── 04_extract_features/   # mouth ROI + wav2vec2 + F0 -> tensor mỗi clip
│   └── 05_build_labels/       # Gộp real+fake, chia tập chống rò rỉ danh tính
├── model/             # Kiến trúc AVSP-Net
├── train/  eval/      # Vòng huấn luyện và đánh giá
└── tools/             # Công cụ độc lập: lọc tay có preview ROI, các phép đo phụ trợ

data/                  # Manifest và artifact theo stage (media không commit)
docs/                  # Kiến trúc, báo cáo, nhật ký làm việc — xem docs/README.md
experiments/           # Mỗi lần chạy là một thư mục bất biến
tests/                 # Test cho generator và data contract
```

## Tài liệu

| Tài liệu | Nội dung |
|---|---|
| [PROJECT.md](PROJECT.md) | Trạng thái, pipeline, data contract, cạm bẫy đã gặp |
| [docs/README.md](docs/README.md) | Chỉ mục toàn bộ tài liệu |
| [docs/architecture/MODEL_PROPOSAL.md](docs/architecture/MODEL_PROPOSAL.md) | Đề xuất AVSP-Net V2a/V2b |
| [docs/reports/](docs/reports/) | Báo cáo pilot, đánh giá V1, bằng chứng smoke test |

## Giấy phép và dữ liệu

Video nguồn thu thập từ nội dung công khai và **không được phân phối kèm repository**. Chỉ manifest và artifact phái sinh được version-control.
