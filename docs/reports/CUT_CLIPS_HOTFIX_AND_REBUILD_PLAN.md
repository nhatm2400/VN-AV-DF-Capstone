# Kế hoạch hotfix Stage 04 Cut Clips và dựng lại dữ liệu downstream

**Ngày audit:** 2026-07-29  
**Phạm vi:** bắt đầu từ `04_cut_clips.ipynb`; không chạy lại bước thu thập, tải và quality gate từ Stage 03 trở về trước.  
**Trạng thái:** P0 đã snapshot; P1 đã triển khai trong source và đang chờ Kaggle smoke.
Chưa xóa, di chuyển hoặc ghi đè media hiện tại.

## 0. Nhật ký triển khai

### P0 — hoàn tất

- Snapshot bất biến nằm tại `archive/cut_clips_v1_decode_bug/`.
- Đã lưu checksum 65 file nhỏ, 15 cut CSV/log và inventory cho media cũ.
- Media cũ vẫn nguyên vị trí: 6.888 cut MP4, 3.001 MP4 batch review và 3.001
  ROI preview; không copy, move hoặc delete media.
- Commit checkpoint P0: `db80dda`.

### P1 — source local hoàn tất, chưa chạy Kaggle

- `src/pipeline/01_collect/cut_clips_core.py`: fallback CUDA → CPU và
  NVENC → libx264, stable clip ID, atomic publish, terminal status và coverage
  contract theo batch.
- `src/pipeline/01_collect/04_cut_clips.ipynb`: driver Kaggle canonical, checkout
  exact Git SHA; config theo tier nằm trong `configs/`.
- Hai notebook copy cũ của Tier 1/Tier 3 đã được bỏ khỏi source canonical.
- `01_prep_manifest.py` dùng accepted CSV làm nguồn chân lý và bắt buộc
  accepted/media 1–1.
- Các output measurement/curation mặc định không ghi đè; curation xuất đủ ba
  partition `gate rejected + balance dropped + clean = scored`, kèm config và
  SHA-256 input.
- Kiểm thử local: **34 pass, 1 skip**. Test bị skip là real-data smoke chỉ chạy
  khi bật `RUN_REAL_AV_AUDIT=1`; compile Python và compile 5 code cell notebook
  đều đạt.
- Chưa có bằng chứng Kaggle GPU/smoke, chưa có full cut mới và chưa thay canonical
  data. Tier 2 vẫn cần khóa Kaggle path; Tier 3 vẫn fail-closed với
  `expected_input_count=0` cho tới khi khôi phục manifest Stage 03.

## 1. Kết luận điều hành

Không được tiếp tục manual review, sinh fake, trích feature hoặc train trên tập
`6.888 → 3.001` hiện tại. Tập này được dựng từ một lần cắt clip có lỗi giải mã
video bằng CUDA và một lần chạy Tier 2 không đủ phạm vi.

Lỗi chính không nằm ở việc các ngưỡng face/speech quá gắt:

- `1.413` video đã dừng ngay ở `video_decode_failed`, trước khi được xét
  `face_ratio`, `speech_ratio` hay scene cut.
- Tier 2 chỉ xử lý `123/292` video quality-pass; còn thiếu hoàn toàn `169` video.
- Có thêm `810` cửa sổ bị `ffmpeg_cut_failed`. Chưa thể kết luận tất cả đều cứu
  được, nhưng code hiện tại cũng bỏ qua return code/stderr và không có fallback.
- Notebook Tier 3 trong repo vẫn mang toàn bộ cấu hình Tier 1; Tier 2 không có
  notebook Stage 04. Các batch từng chạy trên Kaggle vì vậy không tái lập được từ
  source hiện tại.

Thứ tự đúng là:

```text
đóng băng dữ liệu cũ
  → sửa và test Stage 04 Cut Clips
  → chạy lại đủ Tier 1/2/3 trên Kaggle
  → kiểm coverage và media contract
  → dựng lại manifest + toàn bộ curation
  → manual review lại trên scope mới
  → sinh fake V2 + SNVSM + metadata gate
  → build labels chống leakage
  → extract feature mới
  → repaired pilot V2a
```

Không train full ngay sau hotfix. Điểm dừng tiếp theo vẫn là repaired pilot V2a
và các gate đã quy định trong `MODEL_PROPOSAL.md`.

## 2. Phạm vi đã rà soát

### 2.1 Source

- `src/pipeline/01_collect/` — thu thập, quality gate, Stage 04 Cut Clips theo tier.
- `src/pipeline/02_curate/` — manifest, face/embedding, motion, sync, curate, EDA.
- `src/tools/` — face ambiguity, ROI preview, review manifest, chia và merge reviewer.
- `src/pipeline/03_fake/` — bốn generator V2, hợp nhất manifest, SNVSM, metadata gate.
- `src/pipeline/04_extract_features/` — mouth ROI, Wav2Vec2 và prosody.
- `src/pipeline/05_build_labels/` — contract, ghép cặp real-fake và split.
- `src/train/`, `src/eval/`, `src/model/` và các test hiện có.

### 2.2 Artifact hiện tại

| Artifact | Số file | Dung lượng | Trạng thái sau audit |
|---|---:|---:|---|
| `data/01_collect/cut_clips/` | 6.903 tổng, trong đó 6.888 MP4 | 22,400 GiB | Bị ảnh hưởng; không dùng làm nguồn mới |
| `data/01_collect/final_clips_batch1/` | 3.001 MP4 | 6,835 GiB | Batch review cũ; sẽ thay |
| `data/02_curate/` | 3.033 | 0,511 GiB | Toàn bộ là dẫn xuất từ cut cũ |
| `data/03_fake/` | chỉ `.gitkeep` | 0 | Không có media hiện hành cần xóa |
| `data/04_features/` | chỉ `.gitkeep` | 0 | Không có feature hiện hành cần xóa |
| `data/05_labels/` | chỉ `.gitkeep` | 0 | Không có labels hiện hành cần xóa |
| `experiments/` | 9 | 0,019 GiB | Lịch sử pilot bất biến; giữ nguyên |

`archive/pilot_v1/` và `archive/phase0_v2_smoke/` là bằng chứng lịch sử, không
được trộn vào lần dựng mới và không được xóa trong hotfix này.

## 3. Bằng chứng lỗi Stage 04

### 3.1 Coverage từ artifact đã chạy

| Tier | Video quality-pass/input biết được | Video đã xử lý | Nguồn có ≥1 clip nhận | Nguồn không có clip nhận | `video_decode_failed` |
|---|---:|---:|---:|---:|---:|
| Tier 1 | 472 | 472 | 70 | 402 | 386 |
| Tier 2 | 292 | 123 | 13 | 110 | 108 |
| Tier 3 | chưa có manifest quality-pass trong repo | 1.262 theo log cut | 163 | 1.099 | 919 |
| **Tổng đo được** | — | 1.857 | 246 | 1.611 | **1.413** |

Với Tier 2, `292 - 123 = 169` video quality-pass không xuất hiện trong accepted
log lẫn reject log. Đây là thiếu coverage do phạm vi chạy, không phải bị filter.

### 3.2 Dataset cũ được tạo ra như thế nào

| Tập | Clip | Source video |
|---|---:|---:|
| `all_manifest.csv` | 6.888 | 246 |
| Tier 1 trong manifest | 4.776 | 70 |
| Tier 2 trong manifest | 1.011 | 13 |
| Tier 3 trong manifest | 1.101 | 163 |
| `all_clean.csv` | 3.001 | 226 |

Con số “246 video” là số nguồn còn ít nhất một clip sau lần cắt lỗi, không phải
số video ban đầu thu thập hay số video đã qua quality gate.

### 3.3 Cơ chế gây mất video

Trong `src/pipeline/01_collect/tier1/04_cut_clips.ipynb`:

1. `USE_HWACCEL_DECODE = True`.
2. `get_all_frames_1fps()` luôn gọi FFmpeg với `-hwaccel cuda`.
3. Lệnh chỉ lấy `stdout`, không kiểm `returncode`, không lưu `stderr` và không
   retry bằng software decoder.
4. Khi FFmpeg trả về byte rỗng, hàm trả danh sách frame rỗng.
5. `process_one_video()` lập tức trả một reject terminal
   `video_decode_failed`.

Việc `ffmpeg -hwaccels` liệt kê `cuda` chỉ chứng minh build có giao diện CUDA;
không chứng minh mọi codec/profile/pixel format của nguồn đều giải mã được trên
GPU.

`cut_clip()` có cùng kiểu lỗi vận hành: dùng CUDA decode + NVENC encode, bỏ qua
return code/stderr và chỉ nhìn việc file có tồn tại hay không. Do đó
`ffmpeg_cut_failed` cũng cần fallback và validation, nhưng phải đo lại sau khi sửa
trước khi kết luận nguyên nhân cho từng cửa sổ.

### 3.4 Lỗi provenance và cấu hình

- Notebook Tier 1 được commit với `START_INDEX=0`, `END_INDEX=100`, nhưng dữ liệu
  hiện tại đến từ năm batch `0-100`, `100-300`, `300-400`, `400-440`,
  `440-9999`. Các chỉnh sửa batch trên Kaggle không được lưu vào source.
- `tier3/04_cut_clips.ipynb` vẫn có `TIER_NAME="tier1"`, dataset Tier 1 và input
  CSV Tier 1.
- `tier2/04_cut_clips.ipynb` không tồn tại.
- Repo chưa có `tier3_quality_gate_passed.csv`. Stage 03 Tier 3 hiện còn ghi
  output về `data/` thay vì `data/01_collect/`, không khớp layout hiện tại.

Kết luận: không sửa bằng cách đổi riêng `USE_HWACCEL_DECODE=False` rồi chạy.
Hotfix phải đồng thời khóa code, config, input inventory, output manifest và
coverage của mỗi batch.

## 4. Lỗi/điểm yếu downstream phải xử lý trước khi chạy lại

### 4.1 `01_prep_manifest.py`

- Default và hướng dẫn vẫn trỏ `data/clips/`, trong khi canonical path hiện tại
  là `data/01_collect/cut_clips/`.
- Script hiện quét MP4 trên đĩa trước, rồi mới ghép metadata. Một MP4 hợp lệ
  nhưng không nằm trong accepted CSV vẫn có thể được đưa vào manifest với
  `has_cut_meta=0`.
- Output được ghi trực tiếp, có thể ghi đè manifest đang dùng.

Hotfix cần đổi accepted CSV thành nguồn chân lý, yêu cầu mọi dòng có media hợp
lệ, và fail nếu có media mồ côi, CSV mồ côi, clip ID trùng hoặc
`has_cut_meta != 1`.

### 4.2 `02_score_clips.py` và tên artifact

- Default input còn là `data/clips/all_manifest.csv`.
- Tên `tier1_scored_all.csv` gây hiểu nhầm dù file chứa cả ba tier.
- CSV/NPY đang ghi trực tiếp vào canonical path.

Lần chạy mới phải truyền path/version rõ ràng và ghi vào run staging. Có thể giữ
tên file cũ để giảm phạm vi sửa, nhưng không được dựa vào default sai.

### 4.3 `04_curate.py`

Default trong code hiện là:

- `cluster_dist=0.5`
- `min_consistency=0.0`
- `cap_per_speaker=12`

Cấu hình thực tế đã dùng để tạo 3.001 clip theo report lịch sử là:

- `cluster_dist=0.6`
- `min_det_ratio=0.6`
- `min_face_area=0.01`
- `min_consistency=0.3`
- `cap_per_speaker=30`

Lần dựng lại phải lưu toàn bộ CLI/config vào run manifest. Trước tiên chạy
calibration trên population mới; cấu hình cũ chỉ là baseline so sánh, không được
âm thầm lấy default hiện tại hay tự đổi ngưỡng để đạt số clip mong muốn.

`all_clean_rejects.csv` hiện chỉ chứa clip bị gate loại. Clip bị bỏ ở bước cân
bằng speaker không có bảng riêng. Hotfix nên xuất thêm
`balance_dropped.csv` để bảo toàn lineage:

```text
all scored = gate rejected + balance dropped + all_clean
```

### 4.4 Manual review

Tất cả artifact sau đây gắn với `all_clean.csv` cũ:

- `tier1_scored_motion.csv`
- `face_ambiguity.json`
- `all_clean_review.csv`
- `roi_preview/`
- `assignments/v2/`
- `final_clips_batch1/`
- các CSV quyết định manual hiện có

Không được tiếp tục review trên assignment cũ. Clip ID hiện tại phụ thuộc
`clip_idx`, mà `clip_idx` chỉ tăng sau khi cut thành công. Khi fallback cứu thêm
một clip ở đầu video, ID của các clip sau có thể đổi dù nội dung của chúng không
đổi.

Quyết định manual cũ phải được giữ làm provenance. Chỉ được tái sử dụng một
quyết định nếu đồng thời khớp:

- `source_video`
- `start_time` và `end_time` theo đơn vị chính xác
- SHA-256 của media
- rubric version

Không tái sử dụng chỉ vì `clip_id` giống nhau.

### 4.5 Fake, feature, labels và experiment

Source V2 hiện có thể dùng tiếp sau khi real manifest mới được manual-review,
nhưng mọi media/manifest mới phải dùng run ID mới:

- bốn fake generator phải chạy lại từ real clean mới;
- `fake_all.csv`, SNVSM real/fake và metadata gate phải dựng lại;
- labels/split phải dựng lại vì speaker/source population đã đổi;
- feature phải extract vào store mới, không dùng `--skip_existing` trên store cũ;
- pilot/full experiment phải có run ID bất biến mới.

## 5. Thiết kế hotfix Stage 04 đề xuất

### 5.1 Một implementation chuẩn, cấu hình tách khỏi code

Không tiếp tục duy trì các notebook copy-paste khác nhau theo tier.

Đề xuất:

```text
src/pipeline/01_collect/
├── cut_clips_core.py             # logic có thể unit-test
├── 04_cut_clips.ipynb            # driver mỏng để chạy trên Kaggle
└── configs/
    ├── tier1.json
    ├── tier2.json
    └── tier3.json
```

Hai notebook cũ trong `tier1/` và `tier3/` được archive hoặc thay bằng file chỉ
dẫn đến notebook chuẩn. Tier 2 không cần một bản copy thứ ba.

Mỗi config phải khóa ít nhất:

- tier, dataset path, input CSV;
- `run_id`, `start_index`, `end_index`;
- số worker/GPU;
- toàn bộ ngưỡng VAD, face, scene, speech;
- decoder/encoder policy;
- phiên bản package, FFmpeg, model YOLO và checksum model;
- git SHA của source.

### 5.2 Decode và encode fallback

Decode frame 1 FPS:

1. thử CUDA;
2. kiểm `returncode`, `stderr`, số byte có chia hết cho frame size và số frame
   hợp lý so với duration;
3. nếu thất bại, retry bằng software decode;
4. chỉ ghi `decode_both_failed` khi cả hai đường đều thất bại;
5. ghi `decode_backend=cuda|cpu_fallback` và lỗi rút gọn vào log.

Cut/encode:

1. ghi ra file `.partial`;
2. thử NVENC nếu config yêu cầu;
3. nếu thất bại, xóa file tạm của chính lần thử đó và retry bằng `libx264`;
4. kiểm return code, video stream, audio stream, duration và khả năng decode;
5. chỉ `os.replace()` sang tên cuối sau khi đạt contract.

Audio extraction và VAD cũng phải ghi return code/error cụ thể, không dùng
“file tồn tại” làm bằng chứng duy nhất.

### 5.3 Clip ID ổn định

Không dùng số thứ tự accepted (`clip_idx`) làm identity chính. Dùng key ổn định
dựa trên nguồn và biên thời gian, ví dụ:

```text
<source_video>_s<start_ms>_e<end_ms>
```

Accepted order có thể vẫn lưu ở cột riêng để debug. `01_prep_manifest.py` phải
đọc source/timestamp từ accepted CSV, không phụ thuộc parse tên file.

### 5.4 Log và output contract mỗi batch

Mỗi batch Kaggle là một output bất biến:

```text
cut_<run_id>/<tier>/<start>_<end>/
├── input_inventory.csv
├── video_status.csv
├── accepted_clips.csv
├── rejected_windows.csv
├── config.json
├── environment.json
├── run_summary.json
├── media.zip
└── SHA256SUMS
```

`video_status.csv` phải có đúng một dòng terminal cho mỗi video input, gồm số
clip accepted, tổng reject theo loại, decode/encode backend và lỗi terminal nếu
có.

## 6. Test và gate trước khi full rerun

### Gate A — unit/contract test local

- Test decode CUDA fail → CPU thành công.
- Test cả hai decoder fail → log đúng và exit trạng thái lỗi.
- Test NVENC fail → libx264 thành công.
- Test file `.partial` không được publish.
- Test clip ID không đổi khi đảo thứ tự input hoặc khi một cửa sổ trước đó fail.
- Test batch range không overlap, không bỏ sót.
- Test accepted CSV và media 1–1.

### Gate B — Kaggle smoke

Chọn mẫu có chủ đích ở cả ba tier:

- video từng `video_decode_failed`;
- video từng decode thành công;
- nhiều codec/profile/pixel format;
- video ngắn/dài và dọc/ngang.

Gate đạt khi:

- video hợp lệ từng fail CUDA được CPU fallback xử lý;
- video healthy không giảm coverage do hotfix;
- không còn failure không có return code/stderr;
- mọi accepted media qua `ffprobe` và decode thử;
- chạy lại cùng config cho cùng manifest cho kết quả identity giống nhau.

Smoke chỉ chứng minh cơ chế sửa hoạt động; không dùng vài mẫu để ước lượng số
clip full.

### Gate C — full batch coverage

Với từng tier:

```text
input video
= video_status terminal duy nhất
= video có accepted clip hoặc có lý do terminal rõ ràng
```

Với toàn bộ batch:

- range liên tục, không overlap;
- số input unique khớp manifest Stage 03;
- không duplicate video, clip ID hoặc media;
- `accepted_clips.csv` ↔ MP4 là 1–1;
- mọi MP4 có video + audio, mở/decode được;
- không có `video_decode_failed` kiểu cũ;
- `decode_both_failed` phải được audit từng video, không bị gộp thành “lọc gắt”.

## 7. Thứ tự chạy lại đầy đủ

### P0 — đóng băng trước khi sửa

1. Tạm dừng review theo assignment hiện tại.
2. Thu mọi CSV kết quả reviewer đã làm dở, không ghi đè.
3. Xuất inventory + SHA-256 cho:
   - cut manifest/log cũ;
   - `all_manifest.csv`;
   - toàn bộ manifest/measurement/manual/assignment hiện tại.
4. Ghi snapshot config và git SHA.
5. Không di chuyển 22,4 GiB media cho đến khi có staging mới đạt Gate C.

### P1 — sửa Stage 04 và contract liên quan

1. Tạo core dùng chung + notebook Kaggle chuẩn + ba config tier.
2. Thêm decode/encode fallback, atomic publish và structured log.
3. Đổi clip identity sang key ổn định.
4. Sửa `01_prep_manifest.py`:
   - canonical path hiện tại;
   - accepted CSV là nguồn chân lý;
   - fail closed nếu CSV/media lệch;
   - output versioned/atomic.
5. Thêm test Stage 04 và chạy toàn bộ test hiện có.

Không đổi các ngưỡng face/speech/scene trong P1. Trước hết phải loại riêng lỗi
infrastructure khỏi lỗi chất lượng dữ liệu.

### P2 — preflight input Stage 03

1. Tier 1: xác nhận đủ `472` filename quality-pass và media trên Kaggle.
2. Tier 2: xác nhận đủ `292` filename quality-pass và media trên Kaggle.
3. Tier 3: khôi phục/export manifest quality-pass chuẩn từ output Stage 03; không
   chỉ dựa vào con số `1.262` suy ra từ log cut cũ.
4. Kiểm filename unique, media tồn tại, probe được và không có input ngoài
   manifest.
5. Lưu ba input inventory cùng checksum vào run.

Stage 01–03 không chạy lại. Việc khôi phục manifest Tier 3 là khóa provenance
cho output Stage 03 đã đúng, không phải thay đổi tập dữ liệu.

### P3 — chạy lại Cut Clips trên Kaggle

1. Chạy Gate B.
2. Chạy full Tier 1 theo batch bất biến.
3. Chạy full Tier 2 đủ cả `292` input.
4. Chạy full Tier 3 theo manifest vừa khóa.
5. Merge chỉ sau khi từng batch đạt contract.
6. Chạy Gate C cho từng tier và toàn bộ run.
7. Download vào thư mục staging local, chưa ghi đè canonical.

### P4 — dựng lại curation trong staging

Thứ tự bắt buộc:

1. `01_prep_manifest.py`
2. `02_score_clips.py`
3. `02b_motion_score.py`
4. `03_sync_score.py --calibrate` nếu tiếp tục dùng SyncNet; không dùng sync
   threshold làm gate riêng cho real
5. `04_curate.py --calibrate`
6. chốt config curate có log, rồi chạy `04_curate.py`
7. `05_eda.py`
8. `scan_face_ambiguity.py`
9. `build_review_manifest.py`
10. `build_roi_preview.py`
11. `build_review_assignments.py`
12. `export_review_batch.py`

Gate P4:

- số dòng nhất quán qua mọi phép join;
- embedding N×512 khớp đúng N dòng scored;
- không missing path, duplicate clip ID hoặc source/tier rỗng;
- `gate_rejected + balance_dropped + clean = scored`;
- ROI preview coverage đạt 100% hoặc mọi failure được chặn trước khi giao review;
- assignment primary phủ đúng một lần mỗi clip và calibration phủ đủ reviewer.

### P5 — manual review lại

1. Chọn calibration set từ manifest mới.
2. Reuse quyết định cũ chỉ qua exact media hash + exact interval.
3. Chia lại assignment theo tier, không chia theo dải liên tiếp.
4. Merge fail-closed và phân xử disagreement/uncertain.
5. Chỉ publish real clean mới khi coverage mục tiêu đã đạt và provenance đầy đủ.

### P6 — dựng lại fake V2 đến repaired pilot

1. Chạy `01_temporal_desync.py`.
2. Chạy `02_frame_reverse.py`.
3. Chạy `03_pitch_flatten.py`.
4. Chạy `04_anonymization.py`.
5. Chạy `06_build_fake_manifest_v2.py`.
6. Chạy SNVSM cho real và fake bằng cùng config.
7. Chạy `05_build_labels/01_build_labels.py` bằng manifest SNVSM được truyền rõ.
8. Chạy metadata shortcut gate trên toàn repaired-pilot scope.
9. Chỉ khi gate đạt mới chạy `04_extract_features/01_extract_features.py` vào
   feature store mới.
10. Chạy repaired pilot V2a trong `experiments/<run_id-bất-biến>/`.
11. Chỉ xem xét full training sau khi pilot qua các gate model/data đã định.

## 8. Kế hoạch giữ, archive, thay thế và xóa

| Đối tượng | Hành động | Thời điểm/điều kiện |
|---|---|---|
| `data/raw/`, URL CSV, quality-gate output Stage 03 | **Giữ** | Không chạy lại; bổ sung manifest Tier 3 chuẩn |
| Source Stage 01–03 | **Giữ logic** | Chỉ sửa path/provenance nếu cần tái lập; không thay data |
| Cut logs/manifest cũ | **Archive** | P0, kèm config, checksum và nhãn `decode_bug` |
| 6.888 MP4 cũ trong `cut_clips/` | **Giữ tạm**, sau đó archive hoặc xóa có điều kiện | Chỉ sau khi staging mới đạt Gate C và manifest cũ đã archive |
| `all_manifest.csv` cũ | **Archive rồi thay** | Sau khi cut mới đạt Gate C |
| Toàn bộ measurement/embedding/EDA cũ | **Archive rồi chạy lại** | Population clip đã đổi |
| `all_clean*.csv` cũ | **Archive rồi thay** | Không dùng làm real source mới |
| `roi_preview/` cũ (0,511 GiB tổng curate chủ yếu ở đây) | **Thay; xóa có điều kiện** | Sau khi ROI mới đạt coverage và được giao reviewer |
| `assignments/v2/` cũ | **Archive; tạo version mới** | Không overwrite giữa lúc có người review |
| CSV manual cũ/kết quả đang làm | **Giữ vĩnh viễn như provenance** | Có thể reuse chỉ bằng exact hash/interval |
| `final_clips_batch1/` cũ (6,835 GiB) | **Xóa có điều kiện** | Sau khi batch review mới đã export, verify và phân phối |
| `data/03_fake/`, `data/04_features/`, `data/05_labels/` hiện hành | **Không cần xóa** | Hiện chỉ có `.gitkeep`; lần mới dùng run/version mới |
| `archive/pilot_v1/`, `archive/phase0_v2_smoke/` | **Giữ nguyên** | Bằng chứng lịch sử |
| `experiments/pilot_v1_*` | **Giữ nguyên, bất biến** | Không dùng làm output lần mới |
| Báo cáo pilot V1 | **Giữ nhưng gắn trạng thái lịch sử** | Không sửa số đo đã quan sát |

Không thực hiện recursive delete trong lúc hotfix đang phát triển. Việc dọn
`22,4 + 6,835 + ~0,5 GiB` chỉ được làm bằng danh sách path tuyệt đối đã kiểm,
sau khi artifact thay thế đã được verify.

## 9. Tiêu chí hoàn tất hotfix

Hotfix Stage 04 chỉ được coi là xong khi:

- source có một implementation chuẩn, ba config tier và test fallback;
- input inventory Tier 1/2/3 được khóa;
- Tier 2 không còn thiếu 169 video do batch;
- mọi video input có terminal status đúng một lần;
- CUDA failure không còn âm thầm biến thành frame rỗng;
- accepted CSV và media khớp 1–1, không corrupt/orphan;
- rerun cùng input/config tạo cùng clip identity;
- cut run mới chưa ghi đè dữ liệu cũ trước khi qua gate.

Quá trình dựng lại chỉ được coi là xong khi:

- curation mới có đầy đủ lineage và manual review mới;
- fake V2/SNVSM/labels/feature đều trỏ real clean mới;
- split tiếp tục speaker/source-video-disjoint;
- metadata shortcut gate đạt;
- repaired pilot V2a có run bất biến và report riêng.

## 10. Việc cần làm ngay tiếp theo

Việc tiếp theo là **P0 + P1**, không phải chạy Kaggle ngay:

1. snapshot artifact cũ và thu kết quả reviewer đang làm dở;
2. sửa Stage 04 thành core testable + notebook/config chuẩn;
3. thêm test fallback và coverage;
4. chạy Kaggle smoke;
5. chỉ sau smoke đạt mới mở full rerun Tier 1/2/3.
