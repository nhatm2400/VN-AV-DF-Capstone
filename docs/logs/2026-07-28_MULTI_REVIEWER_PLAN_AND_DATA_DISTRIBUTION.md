# Nhật ký 2026-07-28 — Multi-reviewer và phân phối dữ liệu

**Ngày:** 2026-07-28

**Checkpoint:** `deeb0e2` (feat(review): --allow_partial, export batch, và assignment cho 3 reviewer)

**Phạm vi:** hoàn thiện tooling review nhiều người; tính lại số clip cần review cho repaired pilot; chuẩn bị bộ dữ liệu phát cho reviewer

**Trạng thái:** repaired pilot vẫn **tạm dừng**; manual review đang ở 60/3.001; dữ liệu đã sẵn sàng để phát

## 1. Đã commit trong phiên này

Bốn commit, tính từ `aa219c3`:

| Commit | Nội dung |
|---|---|
| `d059340` | `build_review_assignments.py`, `merge_review_results.py`, `test_manual_review_workflow.py`; sửa 4 khiếm khuyết của `clip_review.py` |
| `4129b41` | Nhật ký này (bản đầu) |
| `f31f349` | `--media_root` cho `clip_review.py` |
| `deeb0e2` | `--allow_partial` cho merge, `export_review_batch.py`, assignment cho 3 reviewer |

Toàn bộ 19 test trong `tests/` pass (1 skip) sau mỗi commit.

Bốn khiếm khuyết nêu ở mục 4 nhật ký 2026-07-27 đã xử lý xong:

1. `--exclude_channel` giờ áp **trước** `--sample`, không còn làm sample nhỏ hơn yêu cầu.
2. `diverse_order` nhận seed.
3. Docstring không còn hard-code `43%` / `1125` / `209-209`; thay bằng cảnh báo rằng tỉ lệ keep chỉ là ước lượng từ mẫu nhỏ và đa dạng hàng đợi không suy ra đa dạng tập keep.
4. `build_review_manifest.py` dựng lại `manifests/all_clean_review.csv` tái lập được — verify trùng khớp từng ô với bản tạo tay trước đó.

Ngoài ra `measurements/face_ambiguity.json` (3.001 clip, kết quả quét ~40 phút) trước đây chỉ nằm trong thư mục tạm, nay đã đưa vào `data/02_curate/measurements/` và `scan_face_ambiguity.py` mặc định ghi vào đó.

## 2. Hai điểm chặn đã gỡ

### 2.1 `--media_root` — review trên máy khác

`manifests/all_clean_review.csv` lưu đường dẫn tuyệt đối của máy dựng manifest (`E:\FPTU\PRJ\...`). Reviewer tải media về máy mình sẽ thấy **toàn bộ** clip báo thiếu file.

Cách giải: `--media_root` quét đệ quy một thư mục và tra theo tên file `<clip_id>.mp4`. Chọn tra theo tên file thay vì ghép lại tiền tố vì tên file đúng bằng `clip_id` và duy nhất trong cả 3.001 clip — reviewer sắp xếp thư mục kiểu gì cũng chạy.

Verify bằng mô phỏng máy lạ (manifest trỏ ổ `Z:` không tồn tại, media đặt trong thư mục phẳng):

| | Clip tìm thấy |
|---|---|
| Không `--media_root` | **0/8** |
| Có `--media_root` | 6/8 (2 clip cố ý không tải) |

Clip không có trong index giữ nguyên đường dẫn gốc nên vẫn hiện "thiếu file" chứ không âm thầm bỏ qua. `file_path` ghi vào CSV kết quả vẫn là đường dẫn gốc trong manifest, không phải đường dẫn cục bộ của reviewer — merge không lệch.

### 2.2 `--allow_partial` — dừng review sớm khi đã đủ keep

`merge_review_results.py` fail-closed: thiếu bất kỳ phán quyết nào đã giao là từ chối xuất `manual_clean_v2.csv`. Điều này mâu thuẫn với kế hoạch "review tới khi đủ số keep rồi dừng".

Verify bằng mô phỏng 3 người mỗi người review đúng một nửa assignment:

```
expected_judgements: 36
received_judgements: 18
missing_judgements:  18
-> exit 1, final manifest KHÔNG được tạo
```

`--allow_partial` cho phép xuất khi **chủ ý** dừng sớm, nhưng ghi rõ `partial: true` và số clip đã review theo từng reviewer trong summary. Manifest partial không đại diện cho toàn manifest, nên mọi tỉ lệ tính từ nó (keep-rate, phân bố tier/channel) chỉ áp cho phần đã review.

Clip bất đồng hoặc `uncertain` **vẫn chặn** — cờ này không bỏ qua chúng, phải phân xử trước.

## 3. Ý tưởng "1.000 clip cho 3 người": số học không đủ

Tỉ lệ keep đo trên 60 clip calibration: **26/60 = 43,3%**, khoảng tin cậy Wilson 95% **[32%, 56%]**.

| Clip review | Keep kỳ vọng | Cận dưới KTC | Cận trên KTC | Đủ 540 real? |
|---:|---:|---:|---:|---|
| 1.000 | 433 | 316 | 559 | **Không** |
| 1.200 | 520 | 379 | 671 | Không |
| 1.400 | 607 | 442 | 783 | Có thể |
| 1.600 | 693 | 505 | 894 | Có thể |
| 1.800 | 780 | 568 | 1.006 | Chắc chắn |

**1.000 clip cho kỳ vọng 433 keep — thiếu so với 540 real cần cho pilot.** Để cận dưới khoảng tin cậy vượt 540 cần review **≥ 1.710 clip**.

Ba lưu ý:

1. Tỉ lệ 43,3% đến từ mẫu 60 clip phân tầng theo motion, các clip chung nguồn không độc lập — sai số thực có thể rộng hơn khoảng Wilson.
2. Đủ 540 keep **chưa đủ điều kiện** publish pilot. Còn phải kiểm tier, speaker, source video, channel, connected-component split và khả năng sinh đủ bốn fake cho mỗi source.
3. Cách an toàn hơn là **giao hết 3.001 clip và bám mục tiêu theo số keep**, dừng khi đủ — nhờ `--allow_partial` giờ làm được. Không cần chốt trước con số N.

### 3.1 Khối lượng thực tế

Assignment đã tạo giao hết 3.001 clip: mỗi người ~980 primary + 60 calibration ≈ **1.040 clip**.

Tốc độ đo ở phiên calibration: median 6,0 giây/clip, **mean 9,8 giây/clip** (P90 = 22 giây). Dùng mean khi lập kế hoạch vì phân bố có đuôi dài.

Nếu review hết assignment: **1,7–2,8 giờ/người**. Nếu dừng khi cả nhóm đủ ~600 keep: khoảng **1,3 giờ/người**.

## 4. Assignment đã tạo

`data/02_curate/assignments/v2/` — chia bằng `build_review_assignments.py --seed 42`:

| Reviewer | primary | tier1 / tier2 / tier3 | source video | speaker |
|---|---:|---|---:|---:|
| `nguyenminhnhat` | 981 | 594 / 221 / 166 | 165 | 377 |
| `nguyenvanlinh` | 980 | 593 / 222 / 165 | 170 | 377 |
| `nguyenlamanh` | 980 | 593 / 222 / 165 | 160 | 393 |

Cộng 60 clip calibration dùng chung cho cả ba.

**Không chia theo dải số thứ tự liên tiếp.** `all_clean.csv` được sắp theo `speaker_id`, nên chia liên tiếp sẽ khiến mỗi người nhận một nhóm người nói hoàn toàn khác nhau — khác biệt do người review sẽ lẫn với khác biệt do nội dung, không tách được. Builder chia cân bằng theo tier với shuffle có seed để tránh việc đó.

Reviewer ID phải chốt cứng: merge dựa vào `reviewer_id`, đổi tên giữa chừng là hỏng coverage.

### 4.1 Lưu ý về 60 clip calibration của `nguyenminhnhat`

Output của `clip_review.py` đặt tên theo manifest, nên khi dùng file assignment sẽ ra `manual_assignment_nguyenminhnhat_v2_nguyenminhnhat.csv` — khác file cũ `manual_all_clean_review_v2.csv`. 60 quyết định cũ **không được nhận lại**, phải chấm lại.

Với calibration set thì chấm lại là **đúng ý đồ**: cần cả ba người chấm độc lập mới đo được đồng thuận. Nhưng cần biết trước để không tưởng là lỗi.

## 5. Dữ liệu phát cho reviewer

### 5.1 Đã gom xong

`export_review_batch.py` copy 3.001 clip từ 7 thư mục tier vào một thư mục phẳng tên `<clip_id>.mp4` — khớp cách `--media_root` tra file.

Đã chạy: **3.001 file, 6,83 GiB, 36 giây** → `data/01_collect/final_clips_batch1/`.

Lý do phải gom: 3.001 clip nằm rải trong `data/01_collect/cut_clips/` (23 GiB) lẫn với clip đã bị gate loại. Gửi cả ba thư mục tier là tải thừa ~16 GiB clip không ai review.

| Thư mục tier | Số clip trong `all_clean` |
|---|---:|
| `tier2/tier2_v3_clips_0_999-001` | 681 |
| `tier1/100-300/tier1_v3_clips_100_300` | 665 |
| `tier3/tier3_v3_clips_0_1262-001` | 508 |
| `tier1/300-400/tier1_v3_clips_300_400` | 462 |
| `tier1/0-100/tier1_v3_clips_0_100` | 368 |
| `tier1/440-9999/tier1_v3_clips_440_9999` | 168 |
| `tier1/400-440/tier1_v3_clips_400_440` | 149 |

### 5.2 Bộ cần upload

| Nội dung | Dung lượng |
|---|---:|
| `data/01_collect/final_clips_batch1/` | 6,83 GiB |
| `data/02_curate/roi_preview/` | 501 MiB |
| `data/02_curate/assignments/v2/` | 1,3 MiB |
| **Tổng** | **~7,3 GiB** |

Mỗi người tải cả bộ, chỉ dùng file assignment của mình. Nếu reviewer chưa clone repo thì gửi kèm `src/tools/clip_review.py` (chỉ dùng thư viện chuẩn Python, không cần cài gì).

### 5.3 Phải gửi cả clip gốc lẫn ROI

`roi_preview/` rẻ hơn 14 lần nhưng **không đủ một mình**. Ô ROI chỉ là vùng miệng 96×96; muốn phán `dubbed` phải nhìn được khuôn mặt và bối cảnh trong video gốc — ví dụ nhận ra người nói là người nước ngoài, hoặc cảnh là phóng sự B-roll. `clip_review.py` hiển thị song song hai luồng vì lý do đó.

### 5.4 `all_clean.csv` và `all_clean_review.csv` khác nhau

Cùng 3.001 dòng, cùng thứ tự, 18 cột chung giống hệt nhau. `all_clean_review.csv` có thêm **9 cột**: `motion_median`, `motion_p90`, `frac_near_static`, `n_faces_med`, `n_faces_max`, `ratio_med`, `ratio_max`, `cx_spread`, `channel`.

Chín cột đó hiển thị trong panel công cụ; thiếu chúng reviewer mất thông tin cảnh báo (clip nghi tĩnh, clip nhiều mặt). File assignment sinh từ `all_clean_review.csv` nên đã có đủ.

## 6. Lệnh cho reviewer

```powershell
python src/tools/clip_review.py ^
  --csv assignment_<tên>.csv ^
  --media_root <thư mục final_clips_batch1 trên máy họ> ^
  --roi_dir <thư mục roi_preview trên máy họ> ^
  --reviewer <tên>
```

Gộp kết quả:

```powershell
python src/tools/merge_review_results.py ^
  --assignments "data/02_curate/assignments/v2/assignment_*.csv" ^
  --results "<thư mục kết quả>/*.csv" ^
  --allow_partial
```

Bỏ `--allow_partial` nếu review hết assignment.

## 7. Chưa kiểm chứng

Workflow mới chỉ chạy qua test tổng hợp và mô phỏng, **chưa chạy thật với hai người trên hai máy**. Nên thử một vòng nhỏ (vài chục clip) trước khi cả ba bắt đầu nghiêm túc.

## 8. Còn treo

- **Lỗi P1 RNG** trong `01_temporal_desync.py:338`, `02_frame_reverse.py:151`, `03_pitch_flatten.py:178`: dùng `random.seed(args.seed)` một lần rồi gọi `random.choice`/`random.uniform` trong vòng lặp, nên đổi thứ tự xử lý clip là ra kết quả khác. `05_snvsm_compress.py:413` đã làm đúng với `random.Random(f"{seed}:{pair_key}")`. Phải sửa **trước** khi sinh fake V2, không thì tập fake không tái lập được.
- **`data/03_fake`**: 27 GiB media V1, khoảng 26,7 GiB có thể archive/xóa có điều kiện sau khi chốt nhu cầu reproducibility (xem nhật ký 2026-07-27 mục 7).
- **Push**: bốn commit của phiên này chưa push.

## 9. Kết luận

Tooling multi-reviewer đã đủ để phát dữ liệu: chia việc disjoint cân bằng theo tier, calibration chung để đo đồng thuận, merge fail-closed có đường thoát `--allow_partial` khi chủ ý dừng sớm, và `--media_root` để chạy được trên máy khác. Dữ liệu đã gom sẵn 7,3 GiB.

Điều chỉnh quan trọng nhất về kế hoạch: **1.000 clip không đủ** — kỳ vọng 433 keep so với 540 real cần. Assignment đã giao hết 3.001 clip; nên bám mục tiêu theo số keep và dừng khi đủ, thay vì chốt cứng số clip review.
