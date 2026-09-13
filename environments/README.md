# Môi trường

Đợt chuyển đổi được kiểm tra trong môi trường data sẵn có: Python 3.10.20, FFmpeg/FFprobe trong PATH. `requirements.txt` ghi các phiên bản dependency trực tiếp đã quan sát. Chưa thử cài mới toàn bộ môi trường từ đầu.

| Công việc | Cần gì |
|---|---|
| CSV/split/quyền sử dụng dry-run | Phần lớn thư viện chuẩn Python. |
| Review, kiểm tra media và test | Dependency trong requirements; FFmpeg/FFprobe. |
| Cắt có VAD/YOLO, tạo ROI thực | Thêm Torch, Ultralytics và weights/local VAD. Môi trường hiện có Torch 2.12.0+cu126, Ultralytics 8.4.53; chưa cài lại/chạy full trong đợt này. |
| Face embedding / active speaker tùy chọn | InsightFace/ONNX runtime và repo/model ngoài theo CLI. Không tự tải/chạy từ smoke test. |
| AV-HuBERT | Chưa dựng; tạo môi trường riêng sau khi chốt checkpoint và thử preprocessing. |
| Wav2Lip/MuseTalk/generator giữ riêng | Chưa dựng; mỗi component được khóa phiên bản sau khi chạy mẫu. |

Trong PowerShell, có thể gọi `& '<đường dẫn python của môi trường data>' -m unittest discover -s tests -q`. Không mặc định Python base là môi trường đúng.

File `docs/archives/legacy_avsp/environment/requirements.txt` là môi trường lịch sử, không là hướng dẫn cài model mới.
