# Tạo fake thử — mở file rồi Run

## MuseTalk 1.5: 5 video × 2 giây trên máy hiện tại

Trạng thái ngày 23/09/2026: run này đã sinh 5 video MuseTalk hợp lệ về 50 frame/25 fps
và audio 2 giây; `summary.json` còn ghi `awaiting_visual_review`. Đây là run V1 để xem
chất lượng, chưa phải fake được duyệt cho training. Bước 07 đã tạo run so sánh mới hơn
`quality_10x2s_002` với 10 clip, 40 video real/fake/comparison qua kiểm tra kỹ thuật;
vẫn chờ review hình/tiếng. Mỗi run giữ riêng, không tự nhập vào manifest nghiên cứu.

Mở file rồi bấm **Run Python File**, chọn interpreter `vn_av_df` (Python 3.10).
Các file tự gọi venv riêng `external/MuseTalk/.venv`; không cần đổi interpreter giữa các bước.

| Thứ tự | File Run | Thực hiện |
|---|---|---|
| 04, khi cài lại môi trường | [04_musetalk_setup.py](04_musetalk_setup.py) | Tải/kiểm tra checksum và cài PyTorch, thư viện, weights vào môi trường riêng. Không tạo video. |
| 05 | [05_musetalk_check.py](05_musetalk_check.py) | Kiểm tra đủ 5 đầu vào, checksum weights, CUDA/MMCV và nạp model tìm mặt/landmark. Chạy offline; không tự tải. |
| 06, cho run mới | [06_musetalk_generate.py](06_musetalk_generate.py) | Chạy MuseTalk 1.5, fp16, batch size 1, 5 video × 2 giây; đổi `RUN` trong `preparation/musetalk_smoke.py` và chuẩn bị input mới trước khi chạy lại. Output hiện có sẽ bị từ chối ghi đè. |

Tải/cài lần đầu ước tính 7–8 GB; nên có 15–20 GB trống để giải nén và giữ cache.
Mã nguồn MuseTalk đã clone ở commit `0a89dec45a0192b824e3cf4daf96c239440c5ed8`.
Python/CUDA dựa trên hướng dẫn upstream; dùng wheel MMCV 2.0.0 có sẵn cho Windows/Python 3.10/
PyTorch 2.0/CUDA 11.8. Danh sách dành riêng cho inference nằm trong
[requirements-musetalk.txt](../../environments/requirements-musetalk.txt).
Môi trường hiện tại đã chạy xong smoke, nhưng cần chạy bước 05 nếu cài lại hoặc chuyển máy.

Nếu cần cài lại, chạy 04 → 05 trước khi tạo run mới. Giữ lại
`.tmp/wheels/` nếu còn cần cài lại và `external/MuseTalk/models/` để chạy inference;
file cuối phải đúng checksum mới được cài/nạp.
Không chạy đồng thời hai bản 04. Log cài đặt: `cache/musetalk_setup/setup.log`.
Bản clone hiện có và S3FD từ Wav2Lip là prerequisites của setup này; đây chưa phải installer cho một máy trắng.

Các file của lần thử nằm trong `data/generated/dataset_v1/musetalk_smoke_5x2s_001/`:

- `input/`: 5 video nguồn 2 giây và 5 audio lái.
- `mapping.csv`, `smoke.yaml`: đối chiếu nguồn và cấu hình 5 clip.
- `output/v15/`: 5 fake đã xuất; cần xem kỹ trước khi chọn dùng tiếp.
- `comparison/`: 5 video ba cột, từ trái sang phải **Real → Wav2Lip GAN → MuseTalk**.
  Cả ba cột cùng resize để xem tổng thể; soi chi tiết pixel bằng video gốc ở `input/`,
  video Wav2Lip trong `mapping.csv`, và MuseTalk ở `output/v15/`.
  Audio của bản so sánh là audio thay lời; cột real chỉ đối chiếu hình, không đánh giá lip-sync với audio này.
- `generation.log`, `summary.json`, `provenance.json`, `upstream.patch`: log, kiểm tra số
  frame/audio, hashes đầu vào/weights và thay đổi upstream đã dùng.

Adapter chỉ sửa bước ghép audio để copy video stream, tránh nén hình lần hai; không dùng
face enhancer/sharpening. Output đã có sẽ không bị ghi đè. Nếu 06 bị ngắt, giữ run và log
để kiểm tra trước khi tạo run mới. Bộ này dùng xem chất lượng, chưa phải dataset huấn luyện.
Đánh giá độ rõ môi, viền ghép, răng, rung và giữ danh tính trên cả 5 mẫu; không chỉ chọn mẫu đẹp nhất.

## Wav2Lip (các bước 00–03)

Adapter dùng [Wav2Lip mã nguồn mở chính thức](https://github.com/Rudrabha/Wav2Lip), chưa fine-tune. Đã qua smoke 2 giây trên RTX 4050; batch GAN `wav2lip_gpu_batch_004` hiện có 33 cặp real/fake 4 giây đã chốt metadata sau khi dừng sớm. Video còn chờ review, chưa phải benchmark chất lượng fake hoặc accuracy detector. Tests dùng model giả lập kiểm tra hợp đồng dữ liệu; kết quả GPU thật được lưu riêng trong metadata run.

## Chuẩn bị một lần

Mã nguồn đã clone vào `external/Wav2Lip` trên máy hiện tại, thư mục này không lên Git. Máy khác chạy `git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git external/Wav2Lip` từ root repo.

Tải hai weights theo liên kết từ README chính thức:

| Tải | Đặt file ở đâu tính từ root repo |
|---|---|
| [Wav2Lip thường: Wav2Lip-SD-NOGAN.pt trong thư mục chính thức](https://drive.google.com/drive/folders/153HLrqlBNxzZcHi17PEvP09kkAfzRshM?usp=share_link) | `weights/wav2lip/wav2lip.pth` (đổi tên để khớp settings; nội dung vẫn là TorchScript) |
| [S3FD nhận diện mặt](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) | `external/Wav2Lip/face_detection/detection/sfd/s3fd.pth` (đổi tên file tải về) |

Chọn interpreter `vn_av_df`. Runtime cần torch, numpy, scipy, opencv-python, librosa, numba, tqdm và FFmpeg/FFprobe. Không cài requirements cũ của upstream đè lên môi trường AV-HuBERT. Bridge hỗ trợ cách gọi mel bằng keyword của librosa mới. Nếu dùng môi trường riêng, sửa `GENERATOR_PYTHON` trong settings. Loader nhận diện TorchScript theo nội dung archive và dùng `torch.jit.load`; checkpoint state_dict dùng `torch.load(weights_only=True)` và kiểm tra key nghiêm ngặt. Chỉ dùng TorchScript từ nguồn tin cậy như upstream chính thức; không tự chuyển sang pickle không giới hạn để bỏ qua lỗi.

Đã kiểm tra import bridge trên máy hiện tại: Torch 2.12.0+cu126, NumPy 2.2.6, librosa 0.11.0, OpenCV 4.13.0.92, SciPy 1.15.3, Numba 0.65.1. Không cài/thay dependency. Numba cần ghi cache khi import lần đầu; kiểm tra trong sandbox Codex bị chặn ghi cache đã được chạy lại ngoài sandbox thành công. Dòng `Using cuda` do upstream in lúc import chưa phản ánh override; trường `device` của bridge và settings mới là thiết bị thực sự được chọn.

Upstream giới hạn code/model cho nghiên cứu/cá nhân/phi thương mại; quyền đối với video nguồn vẫn là phần riêng.

## Trình tự

1. Run [00_check_setup.py](00_check_setup.py): kiểm tra mã nguồn, hai weights, import, nạp generator và forward với tensor thử batch 1/8 trên thiết bị đã chọn. Không tạo video. Kiểm tra này cũng chạy trước mỗi batch để checkpoint lỗi không tốn nhiều phút nhận diện mặt.
2. Run [01_prepare.py](01_prepare.py): lấy `COUNT` clip Keep thuộc train, đủ 2 giây cộng biên 0,2 giây, ưu tiên trải ra các video nguồn. Ghép audio từ một clip train khác, cùng người nhưng khác video (donor không bắt buộc nằm trong COUNT clip). Tạo `data/manifests/dataset_v1/generators/<RUN_ID>/plan.csv` và `plan.json`; không tạo media.
3. Run [02_generate.py](02_generate.py): gọi Wav2Lip theo từng clip, xuất COUNT fake và COUNT real đối chiếu nếu mọi mẫu thành công. Mỗi lần chạy in `[i/COUNT]`; log chi tiết trong `logs/`. `DEVICE = 'cuda'` đã được kiểm tra trên RTX 4050 Laptop 6 GB. Máy không có CUDA phải chọn CPU, sẽ chậm hơn. COUNT có thể là 1 để kiểm tra một clip.
4. Mở video real/fake để kiểm tra đúng người, miệng, âm thanh và độ dài. `candidates.csv` ghi `review_status=pending`; chưa phải manifest đã duyệt để train. Không dùng điểm detector để lựa fake. Ghi nhận cả mẫu lỗi trước khi mở rộng batch.

Muốn dừng có chủ ý sau N cặp: chờ terminal in `[N+1/COUNT]`, nhấn `Ctrl+C`, rồi Run [03_finalize_partial.py](03_finalize_partial.py). Bước 03 kiểm tra đủ hai file, FPS/frame/audio và chỉ ghi `candidates.csv` cho cặp hoàn chỉnh; summary ghi `partial=true` cùng số chưa chạy. Không chạy 02 lại vào cùng output. Nếu dừng giữa một clip, cặp đang xử lý không được công bố.

Các biến cần chỉnh tập trung ở [preparation/settings.py](preparation/settings.py): `RUN_ID`, `COUNT`, `NUM_FRAMES`, `DEVICE` và đường dẫn checkpoint/Python. Không phải đổi đường dẫn từng clip. Nếu chạy 00 báo thiếu weights, tải đủ rồi chạy lại; không cần đổi RUN_ID. Sau khi 01 đã tạo plan hoặc 02 đã tạo output, dùng RUN_ID mới khi thay cấu hình/chạy lại. Code không ghi đè run cũ, không tự resume batch dở.

## Output và cách tạo

- Video ở `data/generated/dataset_v1/<RUN_ID>/`, tên kết thúc `__real.mp4` hoặc `__fake.mp4`.
- Metadata ở `data/manifests/dataset_v1/generators/<RUN_ID>/`: `plan.csv`, `plan.json`, `generation.json`, `candidates.csv`, `summary.json`, `failures.csv` nếu có lỗi và `logs/`.
- Fake kế thừa split/group/người/nguồn từ real; audio cũng phải thuộc train. Manifest nguồn và plan có SHA-256 để phát hiện thay đổi sau chuẩn bị. Hash checkpoint, S3FD, mã nguồn upstream/adapter và cấu hình được ghi trong `generation.json`; nguồn video/audio có hash trong candidates.
- Input hình chuẩn hóa 25 fps lossless, audio 16 kHz. Audio lái có thêm 0,2 giây im lặng bên phải phục vụ cửa sổ mel của upstream; video cuối và audio cuối đều cắt đúng 2 giây. Không kéo dài/loop nguồn trong output cuối.
- Bridge giữ model, phát hiện mặt và vòng inference của upstream; đổi intermediate DIVX sang FFV1 và mux lossless. Real và fake cùng được encode cuối bằng H.264 CRF18/AAC128k. Đây là cấu hình adapter riêng, chưa tuyên bố tái lập chất lượng upstream hay xóa mọi dấu vết codec.
- Không tự tải weights, đổi môi trường hoặc dùng val/test. Mẫu sinh thành công vẫn cần review. Các clip nguồn khác URL chưa được xác minh trùng nội dung ở bước này.

Sau review generator: chuẩn bị master real/fake được duyệt, tạo hai mức nén của từng master rồi mới lập `media_pairs.csv` cho AV-HuBERT. Không ghép real với fake thành một cặp consistency. Chưa có bước tự động xuất media_pairs từ candidates.

## Chạy thử cải thiện chất lượng: 10 clip × 2 giây

Bấm Run [07_quality_preview.py](07_quality_preview.py) bằng environment `vn_av_df` đã chạy Wav2Lip.
Script tự chọn 10 clip Keep thuộc train, bỏ nguồn hình/audio trên 1080p (không xử lý 4K),
rồi chạy Wav2Lip GAN và MuseTalk lần lượt. MuseTalk vẫn dùng venv riêng; preprocessing
vẫn nằm trong tiến trình inference, không có bước preprocessing riêng để chạy.

- Hai generator dùng chung một file video lossless 50 frame/25 fps và waveform lái 16 kHz.
- Chỉ ảnh đưa vào S3FD được thu về cạnh dài tối đa 960 px; tọa độ đổi về ảnh gốc.
  Tắt cuDNN benchmark tại bộ dò. Không hạ độ phân giải ảnh sinh/ghép.
- Wav2Lip giữ mắt/kính gốc và ghép mềm phần dưới mặt. MuseTalk giữ mask semantic upstream,
  đồng thời giới hạn vùng da bị sửa bằng mask mềm. Đây là cấu hình thử, chưa bảo đảm tăng chất lượng.
- Output: `data/generated/dataset_v1/quality_10x2s_002/` chứa `input`, `output/real`,
  `output/wav2lip`, `output/musetalk/v15`, `comparison`, `logs`, `mapping.csv`, `plan.json`, `provenance.json`.
- Comparison trái → phải: hình gốc | Wav2Lip | MuseTalk; tiếng phát là audio lái fake.
  Mở `output/real` để nghe tiếng gốc; không dùng khung trái với tiếng lái để chấm lip-sync.
- Nếu bị ngắt, Run lại file 07 để kiểm tra hash đầu vào và bỏ qua output đã hoàn tất/hợp lệ.
  Cặp real/fake dở dang phải được kiểm tra trước khi chạy lại. Đổi `RUN_ID` khi đổi cấu hình/model;
  không ghi đè batch cũ. Không tự train, tải weights,
  hay dùng val/test. Output vẫn chờ review, không tự nhập vào manifest nghiên cứu.
