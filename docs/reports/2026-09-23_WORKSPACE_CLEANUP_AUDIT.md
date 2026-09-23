# Kiểm kê workspace sau review 1 — 23/09/2026

Phạm vi: root repo, source/config/tests/notebooks/docs, Git và các thư mục local `data/`, `cache/`, `tmp/`, `.tmp/`, `weights/`, `external/`, `experiments/`. Đây là kiểm kê đọc-only cho dữ liệu: **chưa xóa hay di chuyển file nào**. Dung lượng dưới đây là GiB/MiB nhị phân, làm tròn; trạng thái tải thay đổi theo thời gian.

## Trạng thái cần giữ đúng

- Git đang ở `codex/research-reset`. Trước khi kiểm kê đã có sửa local `src/data/preparation/settings.py` từ `dataset_v1` sang `dataset_v2` và hai slide trong `docs/presentations/` đang bị xóa. Không quy các thay đổi này cho việc dọn dẹp.
- Source hiện hành nằm trong `src/data/` (collect/download/cut/manifest/split/compress), `src/tools/review/`, `src/generators/`, `src/features/`, `src/models/`, `src/training/`, `src/evaluation/`; `tests/`, `configs/`, `environments/`, `docs/research/` phục vụ hợp đồng và nghiên cứu. `docs/archives/legacy_avsp/` cùng `notebooks/LA_draft/` là lịch sử, không phải output sinh tự động cần xóa theo dung lượng.
- `data/` khoảng 50,44 GiB: raw 36,14; real 9,06; manifests 4,54; generated 0,70. `external/` khoảng 10,23 GiB, `weights/` 1,58 GiB, `.tmp/` 2,44 GiB, `cache/` 0,79 GiB. Phần lớn dung lượng là media/venv/weights local, không nằm trong Git (`data/`, `external/`, `.tmp/` bị ignore).
- `dataset_v1` có `clips.csv`, `reviewed_clips.csv`, `real_splits.csv` và 2.252 clip real. Split hiện tại là `source_disjoint_single_speaker`; không đại diện cho đánh giá người chưa thấy. `dataset_v2` có `sources/videos.csv` (52 dòng playlist/video), `selected_videos.csv` (1.954 video) và `raw/download_001/` đang tải. Lúc đọc `download_results.csv`: 186 downloaded, 475 failed, 1.293 pending; 187 file `.mp4` và 12 file `.part` trong thư mục raw. Đếm file có thể khác đếm trạng thái vì trạng thái được ghi từng bước. Chưa có `real/dataset_v2` hay `manifests/dataset_v2`.

## Đề xuất dọn, theo thứ tự ưu tiên

| Vùng | Bằng chứng | Đề xuất / điều kiện |
|---|---|---|
| `data/manifests/dataset_v1/reviews/exports/clips/*/*.mp4` | 2.252 MP4, khoảng 4.53 GiB. Có đủ 2.252 tên và kích thước khớp với `data/real/dataset_v1/**/media/*.mp4`; một cặp lấy mẫu có SHA-256 trùng. `export_review_batch.py` tạo bản này bằng `shutil.copy2`. | **Ưu tiên cao nhất để giải phóng chỗ**, nhưng chỉ bỏ bản MP4 sau khi kiểm tra hash toàn bộ và xác nhận không cần phát lại gói review portable. Giữ mọi `assignment_*.csv` và `review_*.csv` trong cùng thư mục: `03_review_clips.py`/`04_merge_reviews.py` đọc trực tiếp chúng. Không xóa cả `reviews/exports/` hay `data/manifests/dataset_v1/`. |
| `data/generated/dataset_v1/quality_10x2s_001/` | Run thử dở, 8 MP4 đầu ra và input/intermediate, khoảng 231 MiB; `quality_10x2s_002` có 40 MP4 hợp lệ về frame/FPS. | Ứng viên bỏ sau khi nhóm xác nhận run 002 đã thay thế mục đích xem chất lượng của run 001 và lưu metadata/log nếu cần truy vết lỗi. |
| `data/generated/dataset_v1/musetalk_smoke_10x2s_001/`, `musetalk_smoke_input_001/` | Khoảng 10,8 MiB và 1,3 MiB; run đầu chỉ có `input`, `mapping.csv`, `smoke.yaml`, không có output fake. | Ứng viên bỏ sau khi xác nhận input không còn phục vụ so sánh hoặc tái chạy. Không coi là fake dataset. |
| `data/generated/dataset_v1/wav2lip_gpu_batch_001/` | Thư mục rỗng. | Có thể bỏ khi không cần giữ tên run. |
| `data/manifests/dataset_v1/generators/wav2lip_gpu_batch_001/` và `_002/` | Mỗi run có plan, candidates, summary `successful_pairs=10`, log; không có media tương ứng trong `data/generated/dataset_v1/`. Tổng dung lượng metadata nhỏ. | Chỉ bỏ sau khi chép/lưu bằng chứng thử nghiệm cần thiết; hiện hai manifest trỏ output không có nên không dùng làm dataset. Giữ run `_004` với 33 cặp thật/giả và metadata chờ review. |
| `cache/previews/dataset_v1/` | 2.252 MP4 preview, khoảng 0,48 GiB; nguồn real và manifest V1 vẫn còn. | Có thể tái tạo, nhưng giữ nếu nhóm còn sửa quyết định review hoặc cần xem ROI nhanh. Dọn sau khi khóa review. |
| `cache/pip_musetalk/`, `cache/numba_*`, `cache/slide_build/`, `cache/review_ui/` | Cache tải/cài/biên dịch hoặc artifact UI/slide, không là manifest nghiên cứu. Riêng pip khoảng 0,26 GiB. | Ứng viên dọn khi không còn cài lại/chạy lại phiên liên quan; nếu cài lại sẽ tốn thời gian tải/biên dịch. Kiểm tra từng thư mục, không xóa toàn `cache/`. |
| `tmp/` và các file probe nhỏ trong `.tmp/` | `tmp/` chỉ còn thư mục `pdfs/` rỗng; `.tmp/` có `musetalk_probe.part`, `torch_probe.part`, script phụ và `__pycache__`. | Có thể bỏ thư mục rỗng và probe nhỏ nếu không còn phiên kiểm tra đang chạy. **Không xóa toàn `.tmp/`**: `.tmp/wheels/torch-2.0.1+cu118-...whl` khoảng 2,44 GiB đang được `musetalk_setup.py` tham chiếu khi cài lại. |

## Giữ và không trộn phiên bản

- Giữ `data/sources/dataset_v2/`, `data/raw/dataset_v2/download_001/` gồm `download_sources.csv`, `download_results.csv` và `.part` để tiếp tục tải. `download_sources.csv` khóa tập 1.954 video của run này. `src/data/preparation/download.py` từ chối danh sách khác vào cùng run; thêm/bớt nguồn sau khi chạy lại bước 01 cần `DOWNLOAD_RUN` mới. Bước 03 cũng từ chối khi còn failed/pending.
- Giữ V1 `sources/`, `raw/`, `real/`, `manifests/` làm nguồn gốc clip/split và đối chiếu generator. Đặc biệt không xóa `reviewed_clips.csv`, `real_splits.csv`, các CSV quyết định reviewer và `data/generated/dataset_v1/wav2lip_gpu_batch_004/` trước khi review 33 cặp. `quality_10x2s_002/` (10 real + 20 fake + 10 comparison) và `musetalk_smoke_5x2s_001/` (5 fake) cũng đang chờ xem hình/tiếng. `verification.json` chỉ xác nhận kỹ thuật frame/FPS, không phải duyệt chất lượng/nhãn.
- Giữ `external/MuseTalk`, `external/Wav2Lip`, `weights/` khi còn chạy generator/extractor; dung lượng lớn không tự chứng minh dư thừa. `cache/features/dataset_v1/` nhỏ và gồm bằng chứng smoke AV-HuBERT; có thể bỏ các run cache lặp lại sau khi lưu kết quả cần báo cáo.
- Các file nguồn, test, config và archive đã track không thuộc nhóm dọn media. Kiểm tra Git trước bất kỳ cleanup nào để không nhập nhằng với hai slide xóa sẵn và settings V2 đang sửa local.

## Đặt nguồn mới ở đâu

Dùng tên **`dataset_v2`** theo cây hiện tại: `data/sources/dataset_v2/videos.csv` → `selected_videos.csv` → `data/raw/dataset_v2/<DOWNLOAD_RUN>/`; sau cắt/review là `data/real/dataset_v2/` và `data/manifests/dataset_v2/`. Không đặt vào `data/dataset/_V2`: code hiện ghép path theo `sources`, `raw`, `real`, `manifests` với `DATASET_VERSION`. Không đổi tên hoặc gộp V1 vào V2; khi thêm người/nguồn, lập manifest và split V2 mới, rà trùng nguồn với V1 nếu dùng cả hai trong nghiên cứu. Generator Wav2Lip có `src/generators/preparation/settings.py` riêng vẫn đặt `DATASET='dataset_v1'`; MuseTalk smoke/quality cũng còn path V1 trong source, nên thay phiên bản ở data settings không tự chuyển generator.
