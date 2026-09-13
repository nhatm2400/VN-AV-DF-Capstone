# Phân công giai đoạn pilot

| Đầu mối | Trách nhiệm | Kiểm tra chéo |
|---|---|---|
| A — Data/generator | Nguồn/quyền sử dụng, review, manifest và thử generator trên mẫu nhỏ. | C rà leakage; B kiểm tra audio/crop. |
| B — Model/training | Môi trường AV-HuBERT, đặc trưng, baseline đơn luồng/kết hợp; sau đó Y/X. | C tái lập đánh giá. |
| C — Protocol/evaluation | Split, điều kiện nén, bảng kết quả, báo cáo; demo sau khi model ổn. | A rà media; B rà so sánh công bằng. |

Mốc hoàn thành đồ án đang giả định cuối tháng 12/2026; lịch report chưa biết. Không biến lịch pilot thành cam kết trước khi đo GPU/data thực tế. Tất cả thành viên cần hiểu X/Y/Z và có bằng chứng đóng góp riêng.
