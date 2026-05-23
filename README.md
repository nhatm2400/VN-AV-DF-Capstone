# VN-AV-DF-Capstone

VN-AV-DF-Capstone là repo phục vụ bài toán xây dựng dữ liệu và thử nghiệm mô hình cho dự án capstone liên quan đến video tiếng Việt. Repo hiện tập trung vào chuỗi xử lý dữ liệu gồm: thu thập URL video, tải video, quality gate, cắt clip, và một PoC riêng để huấn luyện/đánh giá mô hình PAMF.

## Tổng quan

Pipeline chính trong repo được chia thành các bước:

1. Thu thập URL video từ YouTube bằng YouTube Data API.
2. Tải video về máy bằng `yt-dlp`.
3. Lọc video theo quality gate dựa trên metadata, âm thanh và phát hiện khuôn mặt.
4. Cắt video thành các clip ngắn để phục vụ huấn luyện hoặc đánh giá.

Ngoài pipeline chính, thư mục `PoC/` chứa một proof-of-concept riêng với mô hình hợp nhất audio-visual.

## Cấu trúc thư mục

```text
.
├── configs/                 # File cấu hình phụ trợ
├── data/                    # CSV đầu vào/đầu ra của pipeline
├── docs/                    # Tài liệu tham khảo, hình ảnh, báo cáo
├── experiments/             # Kết quả hoặc cấu hình thí nghiệm
├── notebooks/               # Notebook thử nghiệm
├── PoC/                     # Proof-of-concept cho mô hình PAMF
├── src/
│   ├── pipeline/            # Các bước xử lý dữ liệu chính
│   ├── eval/
│   ├── model/
│   ├── train/
│   └── utils/
└── requirements.txt
```

## Yêu cầu hệ thống

- Python 3.10+.
- `ffmpeg` có sẵn trong `PATH`.
- Tài khoản / API key YouTube Data API v3.
- Môi trường cài đặt các gói trong `requirements.txt`.
- Mô hình YOLO face detection `yolov8n-face.pt` cho bước quality gate.

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Tạo file `.env` ở root dự án với nội dung tối thiểu:

```env
YOUTUBE_API_KEY=your_youtube_api_key_here
```

## Chạy pipeline dữ liệu

Các script trong `src/pipeline/` nên được chạy theo đúng thứ tự sau:

### 1. Lấy danh sách URL YouTube

```bash
python src/pipeline/00_fetch_youtube_urls.py
```

Script này dùng YouTube Data API để tìm video phù hợp và xuất CSV vào `data/youtube_tier1_urls.csv`.

### 2. Tải video

```bash
python src/pipeline/01_download.py
```

Chương trình sẽ hỏi tier cần tải và đọc các file CSV tương ứng trong `data/`.

### 3. Quality gate

```bash
python src/pipeline/02_quality_gate.py
```

Bước này kiểm tra:

- Độ phân giải và FPS của video.
- Chất lượng âm thanh thông qua SNR.
- Sự hiện diện của khuôn mặt bằng YOLOv8-Face.

Kết quả đạt chuẩn được lưu thành file CSV trong `data/`.

### 4. Cắt clip

```bash
python src/pipeline/03_cut_clips.py
```

Script sẽ cắt video đã qua quality gate thành các clip ngắn và sinh file log metadata cho từng clip.

## Dữ liệu đầu ra

Một số file quan trọng trong `data/`:

- `youtube_tier1_urls.csv`: danh sách URL video đã lọc.
- `tier1_quality_gate_passed.csv`: danh sách video đạt quality gate.
- `label.csv`: file nhãn hoặc mapping dùng cho các bước thí nghiệm.

Thư mục `data/raw/` và `data/clips/` được tạo ra trong quá trình chạy pipeline và thường không được commit lên git.

## Proof of Concept

Thư mục `PoC/` là một nhánh thử nghiệm riêng cho mô hình PAMF.

### Huấn luyện

```bash
cd PoC
python src/train_and_eval.py
```

### Suy luận

```bash
cd PoC
python src/inference.py
```

PoC sử dụng checkpoint tại `PoC/checkpoints/pamf_poc_model.pth`.

## Ghi chú

- Nếu chạy `00_fetch_youtube_urls.py`, hãy đảm bảo `.env` có `YOUTUBE_API_KEY` hợp lệ.
- Nếu chạy `02_quality_gate.py`, cần có `yolov8n-face.pt` ở thư mục gốc hoặc chỉnh lại đường dẫn trong code.
- Một số script phụ thuộc `ffmpeg`, `opencv-python`, `librosa`, `ultralytics`, và `yt-dlp`.

## Trạng thái repo

Repo này đang nghiêng về phần xây dựng dataset và pipeline xử lý dữ liệu cho capstone. Nếu bạn muốn, có thể mở rộng README bằng phần mô tả mô hình, dataset schema, hoặc hướng dẫn tái lập thí nghiệm chi tiết hơn khi các thành phần đó ổn định.