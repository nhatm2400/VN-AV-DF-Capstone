# Training — huấn luyện detector

Trạng thái: có train_step, loader cache feature, padding batch, vòng train/validation
và CLI `01_train.py` lưu best.pt cùng epochs.csv. Đã thiết kế để nhận feature thật,
nhưng hiện chỉ kiểm chứng trên dữ liệu giả lập. CLI nền chạy CPU.

CSV đầu vào: `sample_id,label,split,feature_path,paired_feature_path`. Path tương đối
với CSV; split dùng train/val/test; real=0, fake=1. sample_id phải chỉ rõ clip
(phân biệt bản real và bản fake) và cửa sổ thời gian. Mỗi cache .pt chứa audio
`[T,Da]`, visual `[T,Dv]`, sample_id, label, split, view_id, extractor_id, timeline_id.
extractor_id phải ghi nhận hash checkpoint, mode extraction và preprocessing;
timeline_id ghi nhận cùng timestamps. Hai view phải khác view_id và cùng metadata
còn lại. Đây là kiểm tra metadata, không tự chứng minh data không có leakage.

Chạy từ root repo SAU KHI có split và feature thật:

`src/features/01_extract.py` tạo `pairs.csv` theo đúng schema này. Hướng dẫn chuẩn bị checkpoint/media ở [features](../features/README.md). Các đường dẫn ví dụ dưới đây dùng output mặc định `extract_001`.

```powershell
python src/training/01_train.py --manifest cache/features/dataset_v1/extract_001/pairs.csv --out experiments/y_001 --consistency-weight 0
python src/training/01_train.py --manifest cache/features/dataset_v1/extract_001/pairs.csv --out experiments/x_001 --consistency-weight 1
```

Trọng số 1 chỉ là ví dụ. Cùng seed/dữ liệu/batch/epoch cho X và Y. Chương trình
không đọc feature test; chọn checkpoint bằng supervised loss validation cho cả
hai phương pháp. Cần kiểm tra speaker/source-disjoint bằng pipeline data trước;
loader không nhận diện cùng người từ sample_id. Không dùng reviewed_clips.csv
trực tiếp: đó chưa phải manifest real/fake với feature và split. Chạy --help để xem
tham số; không bấm Run trống rồi kỳ vọng chương trình tự tìm dữ liệu.

Y dùng `consistency_weight=0`; X dùng trọng số lớn hơn 0. Cả hai đều học nhãn real=0/fake=1 trên cả hai view. X cộng thêm lỗi chênh lệch score giữa hai view. Trọng số X phải chọn bằng validation, không lấy giá trị trong smoke làm cấu hình nghiên cứu.

Batch có audio, visual, paired_audio, paired_visual, lengths, labels, sample_ids và paired_sample_ids. Mỗi ID phải xác định cùng clip và cửa sổ; cặp chỉ khác mức nén, không phải real ghép với fake. Feature phải được trích thật từ từng view bằng cùng cấu hình. Hàm kiểm tra ID/hình dạng, nhưng không thể xác minh nguồn gốc media chỉ từ tensor.

Chứa cách đọc manifest/đặc trưng thành batch, vòng lặp train, cách tính lỗi, cập nhật model, validation và lưu checkpoint. Phần khuyến khích dự đoán nhất quán giữa các bản nén của phương pháp X sẽ nằm ở đây.

Nhận split đã khóa từ luồng data; không tự chia ngẫu nhiên lại clip. X/Y dùng cùng dữ liệu, bản nén và ngân sách train. Chỉ chọn cấu hình/ngưỡng trên validation, không dùng final test để chọn model.

Model được import từ `src/models/`. Log, cấu hình, kết quả validation và checkpoint ghi vào `experiments/<run_id>/`, không vào source. File chạy có số sẽ được thêm ngay trong thư mục training khi phần train được triển khai.
