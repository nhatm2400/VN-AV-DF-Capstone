# Weights local

`face_detector/yolov8n-face.pt` được chuyển từ root repo, giữ nguyên bytes. Phục vụ công cụ cắt/review hiện có; không phải detector deepfake. Chưa xác minh lại nguồn phát hành/giấy phép của file này trong đợt chuyển đổi, không phân phối qua Git.

Đặt checkpoint AV-HuBERT Base vào `weights/avhubert/base_vox_iter5.pt`. Bộ xử lý miệng cần thêm `weights/face_landmarks/shape_predictor_68_face_landmarks.dat` và `weights/face_landmarks/20words_mean_face.npy`. Nguồn tải và cách chạy: [features](../src/features/README.md). Code không tự tải weights; lần extraction lưu SHA-256 trong `run.json`.

Checkpoint Base đã có trên máy; SHA-256 `dee8b0651dbea3a21cda0f2c52b956e3a878d5f6ee503d427d6a4044bb95c9cc`. Ngày 16/09: hai asset landmark đã tải; predictor đã giải nén thành `.dat`, giữ nguyên `.bz2`; mean face đọc được với shape `[68,2]`. Đã nạp predictor bằng dlib CPU 20.0.1 và trích feature từ một clip thật; hash của cả ba asset trong `cache/features/dataset_v1/real_smoke_001/run.json`. Generator mới chưa có. Mỗi weights thêm sau phải ghi nguồn, phiên bản, hash và điều kiện sử dụng. Checkpoint detector do nhóm train nằm trong experiments/<run_id>/checkpoints.
