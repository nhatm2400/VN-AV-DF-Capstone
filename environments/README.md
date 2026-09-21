# Môi trường

Đợt chuyển đổi được kiểm tra trong môi trường data sẵn có: Python 3.10.20, FFmpeg/FFprobe trong PATH. `requirements.txt` ghi các phiên bản dependency trực tiếp đã quan sát. Chưa thử cài mới toàn bộ môi trường từ đầu.

| Công việc | Cần gì |
|---|---|
| CSV/split/quyền sử dụng dry-run | Phần lớn thư viện chuẩn Python. |
| Tải YouTube | yt-dlp[default] (gồm EJS), Node.js >= 22 trên PATH, FFmpeg/FFprobe. Đã thử tải thực tế với yt-dlp 2026.8.19, EJS 0.8.0 và Node 24.11.0; Python 3.10 vẫn chạy nhưng yt-dlp đã cảnh báo ngừng hỗ trợ trong tương lai. |
| Review, kiểm tra media và test | Dependency trong requirements; FFmpeg/FFprobe. |
| Cắt có VAD/YOLO, tạo ROI thực | Thêm Torch, Ultralytics và weights/local VAD. Môi trường hiện có Torch 2.12.0+cu126, Ultralytics 8.4.53; chưa cài lại/chạy full trong đợt này. |
| Face embedding / active speaker tùy chọn | InsightFace/ONNX runtime và repo/model ngoài theo CLI. Không tự tải/chạy từ smoke test. |
| AV-HuBERT Base pre-fusion | Cần Torch >=2.6, FFmpeg và requirements-features.txt; không cần Fairseq/OmegaConf cho checkpoint Base này. Ngày 16/09: dlib CPU 20.0.1 đã cài vào vn_av_df; đã nạp weights/landmark và trích feature một clip thật thành công. Xem src/features/README.md. |
| Wav2Lip | Bridge trong vn_av_df (librosa 0.11.0, Numba 0.65.1), hỗ trợ checkpoint TorchScript/state_dict. Đã qua forward batch 1/8 trên CUDA và tạo một cặp real/fake 2 giây trên RTX 4050 Laptop 6 GB; không đổi dependency. Xem src/generators/README.md. MuseTalk mới có bộ chuẩn bị local 5 × 2 giây, chưa chạy inference. |

Trong PowerShell, có thể gọi `& '<đường dẫn python của môi trường data>' -m unittest discover -s tests -q`. Không mặc định Python base là môi trường đúng.

File `docs/archives/legacy_avsp/environment/requirements.txt` là môi trường lịch sử, không là hướng dẫn cài model mới.

MuseTalk 1.5 có bộ chuẩn bị riêng cho **5 video × 2 giây chạy local**:
`src/generators/04_musetalk_setup.py` → `05_musetalk_check.py` → `06_musetalk_generate.py`.
Chọn interpreter `vn_av_df` để Run; setup chỉ cài vào `external/MuseTalk/.venv`.
`requirements-musetalk.txt` dành riêng cho venv này, không cài vào `vn_av_df`.
Đã tạo venv và đầu vào; dependencies/weights mới tải một phần, chưa kiểm chứng inference MuseTalk thật.
Chi tiết và output: [src/generators/README.md](../src/generators/README.md).

Cài dlib đã dùng Conda với `dlib=20.0.1=cpu*`, `libblas=*=*openblas`, `--freeze-installed`. Conda bổ sung thư viện CPU và cập nhật OpenSSL/zlib/CA; NumPy vẫn 2.2.6 nhưng chuyển sang bản conda-forge. Đã xác minh import cùng Torch 2.12.0+cu126, OpenCV 4.13.0 và SciPy 1.15.3. Không cài thêm CUDA cho dlib.
