# Tests theo chức năng

| Thư mục | Kiểm tra |
|---|---|
| `data/` | URL/thu thập/tải, cắt clip, manifest, split chống leakage, timeline và nén. |
| `quality/` | Face/curation, active-speaker, LASER, gộp điểm và calibration. |
| `review/` | Giao diện review, chia/gộp công việc, ROI preview. |
| `models/` | Hai luồng có gradient, bỏ qua padding, ghép đúng view, supervision X/Y và dự đoán. Cần PyTorch. |

Chạy tất cả từ root: `python -m unittest discover -s tests -q`.
Chạy một nhóm, ví dụ data: `python -m unittest discover -s tests/data -q`.

Tests kiểm tra kỹ thuật bằng fixture/mock/media tổng hợp; không phải bộ test accuracy của detector. `test_research_data_contract.py` kiểm tra các ràng buộc dùng chung giữa manifest/split/media. Tests model chỉ dùng tensor giả lập, chưa phải benchmark tiếng Việt.
