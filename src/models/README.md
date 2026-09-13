# Models — cấu trúc detector

Trạng thái: mới tạo khung thư mục, chưa triển khai detector Y/X hoặc nạp AV-HuBERT.

Chứa phần định nghĩa/nạp backbone, kết hợp đặc trưng audio và hình ảnh, xử lý thông tin theo thời gian và tạo điểm dự đoán real/fake. Khởi đầu dự kiến là backbone giữ nguyên và bộ phân loại nhỏ; AV-HuBERT vẫn là ứng viên cần kiểm chứng.

Baseline Y và phương pháp X dùng cùng cấu trúc detector. Theo hướng hiện tại, khác biệt chính của X là thêm yêu cầu nhất quán giữa các bản nén trong lúc huấn luyện; phần đó thuộc `src/training/`, không cần tạo hai kiến trúc khác nhau chỉ vì có tên X/Y.

Vòng lặp train ở `src/training/`, tính chỉ số ở `src/evaluation/`. Checkpoint tải về nằm trong `weights/`; checkpoint của từng lần train nằm trong `experiments/<run_id>/checkpoints/`.
