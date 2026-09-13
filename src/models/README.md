# Models — cấu trúc detector

Trạng thái: có detector tối thiểu trong `detector.py`; chưa nạp AV-HuBERT hoặc train trên dữ liệu thật.

Mở `01_smoke_test.py` rồi Run để kiểm tra cả model, một vài bước train Y/X và dự đoán bằng feature giả lập. Test này không đọc video, không tải weights và không tạo kết quả nghiên cứu.

Đầu vào: audio `[batch, thời_gian, số_đặc_trưng_audio]`, visual `[batch, thời_gian, số_đặc_trưng_visual]` và lengths là số bước thời gian hợp lệ. Hai luồng phải được căn cùng cửa sổ và thời gian từ trước. Model chiếu mỗi luồng qua một lớp Linear, ghép chúng, chạy GRU và tạo một logit cho fake. Padding cuối chuỗi được bỏ qua. Đổi logit sang score bằng sigmoid; score chưa được hiệu chỉnh thành xác suất đáng tin cậy.

Chứa phần định nghĩa/nạp backbone, kết hợp đặc trưng audio và hình ảnh, xử lý thông tin theo thời gian và tạo điểm dự đoán real/fake. Khởi đầu dự kiến là backbone giữ nguyên và bộ phân loại nhỏ; AV-HuBERT vẫn là ứng viên cần kiểm chứng.

Baseline Y và phương pháp X dùng cùng cấu trúc detector. Theo hướng hiện tại, khác biệt chính của X là thêm yêu cầu nhất quán giữa các bản nén trong lúc huấn luyện; phần đó thuộc `src/training/`, không cần tạo hai kiến trúc khác nhau chỉ vì có tên X/Y.

Vòng lặp train ở `src/training/`, tính chỉ số ở `src/evaluation/`. Checkpoint tải về nằm trong `weights/`; checkpoint của từng lần train nằm trong `experiments/<run_id>/checkpoints/`.
