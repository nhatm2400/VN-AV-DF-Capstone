# REPORT — Chạy pipeline Curate (01-05) + Sync (03)

Ngày: 21/06/2026
Bối cảnh: Report 2 — Data Tasks (Data Collection and Cleaning + Exploratory Data Analysis).
Môi trường chạy: conda env `vn_av_df` (Python 3.10), GPU RTX 4050 6GB (InsightFace chạy CUDA), ffmpeg trong PATH.

Tóm tắt kết quả: 6888 clip đo được → 5356 qua gate chất lượng → 3001 clip sạch / 674 speaker.
Toàn bộ output nằm trong `data/curate/` và `data/clips/` (đều gitignore).

---

## 0. Trình tự và vai trò từng bước

```
01_prep_manifest  -> gộp 3 tier + verify file  -> all_manifest.csv (6888)
02_score_clips    -> đo mặt + embedding (GPU)   -> tier1_scored_all.csv + embeddings_all.npy
03_sync_score     -> đo khớp môi-tiếng (SyncNet)-> (tùy chọn, chỉ chạy calibrate mẫu)
04_curate         -> cluster -> gate -> cân bằng-> all_clean.csv (3001)
05_eda            -> biểu đồ + bảng tóm tắt     -> eda_figs/
```

Đã chạy FULL trên 6888 clip: 01, 02, 04, 05.
03 chỉ chạy calibrate mẫu 30 clip (không full — lý do ở Mục 3).

---

## 1. Bước 01 — prep_manifest (chuẩn bị manifest)

Mục đích: quét mọi file .mp4 trên đĩa của 3 tier, ghép metadata từ CSV bước cắt, verify từng
file bằng ffprobe (loại file hỏng), remap `file_path` về đường dẫn thật, gộp thành 1 manifest.

Cơ chế/tham số chính:
- Glob `**` đệ quy: chỉ cần trỏ vào thư mục gốc mỗi tier, tự quét mọi batch con (tier1 chia 5 batch).
- Verify ffprobe (mặc định bật): mở thử từng .mp4, loại file ghi dở ("moov atom not found").
- `has_cut_meta`: =1 nếu clip có trong CSV log bước cắt (đã qua đủ filter VAD/face/scene), =0 nếu chỉ có trên đĩa.

Terminal (rút gọn):

```
(khong co --add -> dung mac dinh 3 tier local: tier1/tier2/tier3)
[tier1]
    + data/clips/tier1/0-100/tier1_v3_clips_0_100.csv        (931 dong)
    + data/clips/tier1/100-300/tier1_v3_clips_100_300.csv    (1884 dong)
    + data/clips/tier1/300-400/tier1_v3_clips_300_400.csv    (1107 dong)
    + data/clips/tier1/400-440/tier1_v3_clips_400_440.csv    (458 dong)
    + data/clips/tier1/440-9999/tier1_v3_clips_440_9999.csv  (396 dong)
...
Tong: 6888 clip -> data/clips/all_manifest.csv
```

Kết quả: 6888 clip hợp lệ (tier1=4776, tier2=1011, tier3=1101) từ 246 video gốc.
Số 6888 = số clip còn tốt sau khi đã xóa 810 file hỏng 0-byte ở phiên trước.

Output: `data/clips/all_manifest.csv` (6888 dòng).

---

## 2. Bước 02 — score_clips (đo mặt + embedding)

Mục đích: với mỗi clip lấy 9 frame rải đều, chạy InsightFace (buffalo_l) để đo các chỉ số mặt
và trích face embedding. Bước này KHÔNG loại clip nào — chỉ đo và ghi metadata.

Giải thích tham số / chỉ số:
- `k_frames = 9`: số frame lấy mẫu mỗi clip (rải đều, tránh sát đầu/cuối). Nhiều frame -> chỉ số mịn và embedding ổn định hơn.
- `det_size = (640,640)`: kích thước ảnh đưa vào detector.
- `MIN_DET_SCORE = 0.5`: ngưỡng tin cậy để tính một khuôn mặt là "có".
- `det_ratio`: tỉ lệ frame phát hiện được mặt (0..1). =1.0 nghĩa là cả 9 frame đều có mặt.
- `mean_face_area`: diện tích bbox mặt lớn nhất / diện tích khung, trung bình các frame có mặt. Đo mặt to hay nhỏ trong khung.
- `embed_consistency`: cosine tương đồng trung bình giữa embedding các frame (0..1). Cao = cùng một người xuyên suốt; thấp = clip lẫn nhiều người / cắt cảnh.
- `embedding`: vector 512 chiều đại diện danh tính, đã L2-normalize (dùng để cluster speaker ở bước 04).

Terminal (rút gọn):

```
Đọc 6888 clip từ data/clips/all_manifest.csv  (k_frames=9)
Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'] ...
set det-size: (640, 640)
Xong 6888 clip trong 156.0 phút
  có mặt: 6872 | không mặt: 16
  CSV : data/curate/tier1_scored_all.csv
  NPY : data/curate/embeddings_all.npy  shape=(6888, 512)
```

Kiểm tra chất lượng output:

| Kiểm tra | Kết quả |
|---|---|
| CSV 6888 dòng ↔ NPY (6888, 512) | khớp dòng |
| Clip có mặt | 6872 / 6888 (99.8%) |
| det_ratio | mean 0.991, median 1.0 |
| mean_face_area | median 0.032 (3.2% khung) |
| embedding L2-norm | min=max=1.0 (chuẩn hóa đúng) |

Nhận xét: không có cảnh báo path (nếu file_path sai, mọi clip sẽ thành "không mặt"). Tỉ lệ
có mặt 99.8% xác nhận remap path đúng và dataset sạch về mặt khuôn mặt. Chạy 156 phút (chậm
hơn benchmark vì mỗi clip còn decode + seek 9 frame, không chỉ inference).

Output: `data/curate/tier1_scored_all.csv` (6888 dòng, thêm cột face stats), `data/curate/embeddings_all.npy` (14MB).

---

## 3. Bước 03 — sync_score (đo khớp môi-tiếng)

Đây là bước đo DUY NHẤT nối hình với tiếng theo thời gian. Bước cắt trước đó chỉ kiểm "có
tiếng" (VAD) và "có mặt" (YOLO) một cách riêng rẽ — không biết miệng trên hình có đúng là
nguồn của tiếng không. SyncNet so cử động môi với sóng âm, cho ra:
- `LSE-C` (confidence): cao hơn = khớp hơn.
- `LSE-D` (min distance): thấp hơn = khớp hơn.
Nó thêm metadata (cột `sync_conf`), không biến đổi dữ liệu.

Dựng 03 (đã hoàn chỉnh, chạy được):
- Clone `joonson/syncnet_python` (đặt ngoài repo dự án).
- Cài thêm: `scenedetect==0.6.7.1`, `python_speech_features==0.6`.
- Tải model VGG Oxford (HTTP trực tiếp, không dính Google Drive quota): `syncnet_v2.model` (54MB), `sfd_face.pth` (90MB).
- Vá `run_syncnet.py`: bản này không ghi `offsets.txt` (chỉ lưu pickle + log) -> sửa để ghi ra cho 03 đọc được.

Giải thích tham số pipeline SyncNet (run_pipeline.py):
- `min_track = 100`: track mặt phải dài tối thiểu 100 frame. Quy về 25fps = 4 giây. Clip ngắn hoặc bị cắt cảnh thành đoạn < 4s sẽ bị loại track.
- `min_face_size = 100`: mặt phải lớn hơn 100 px mới nhận track.
- `frame_rate = 25`: pipeline ép video về 25fps trước khi xử lý.
- `num_failed_det = 25`: cho phép mất tối đa 25 frame liên tiếp không thấy mặt trước khi ngắt track.
- `facedet_scale = 0.25`, `crop_scale = 0.40`: tỉ lệ thu nhỏ ảnh khi detect và tỉ lệ nới bbox khi crop.

Terminal — calibrate trên 30 clip ngẫu nhiên:

```
===== CALIBRATE: chạy syncnet trên 30 clip ngẫu nhiên =====
  ... (14 clip ra LSE-C/LSE-D, 16 clip FAIL "không có mặt/track")
Hoàn tất 30 clip trong 25.1 phút (50.1s/clip)

-- Kết quả --
  Thành công: 14 | Fail (không có mặt / lỗi): 16

-- Phân bố LSE-C (confidence: cao hơn = khớp hơn) --
count 14.00 | mean 3.38 | std 3.49 | min 0.40 | 25% 0.69 | 50% 1.09 | 75% 6.12 | max 9.43

-- Phân bố LSE-D (min dist: thấp hơn = khớp hơn) --
count 14.00 | mean 9.92 | min 6.99 | 50% 10.36 | max 12.23
```

Nhận xét:
- Tỉ lệ FAIL 53% (16/30): các clip này InsightFace đều thấy mặt (det_ratio ~1.0), nhưng pipeline SyncNet loại track do `min_track=100` (4s) và `min_face_size=100` quá chặt với clip ngắn / mặt vừa. Đây là giới hạn tham số, không phải clip rác.
- LSE-C lưỡng cực: một cụm thấp 0.4–1.2 (track mỏng/biên, nhiễu) và một cụm bình thường 3–9 (real talking-head khớp tốt). Phân bố này chưa đủ tin để đặt ngưỡng gate.
- Tốc độ ~50s/clip trên CPU (torch trong env là bản CPU). Full 3001 clip ≈ 42 giờ → không chạy full trong khuôn khổ Report 2.

Quyết định: KHÔNG dùng sync để curate lần này, vì (1) full quá chậm trên CPU, (2) fail-rate + lưỡng cực
chưa đáng tin, (3) nguyên tắc chống leakage: không được siết tập real theo sync (đẩy real về
"sync cao" sẽ khiến model học tắt). Sync để dành cho pilot leakage audit sau (subset, đối xứng
real/fake), khi đó nới `min_track`/`min_face_size` + dùng GPU.

Output (mẫu): `data/curate/calibrate_sync_results.csv`, `data/curate/sync_calibrate_log.txt`.

---

## 4. Bước 04 — curate (cluster -> gate -> cân bằng)

Mục đích: từ scored CSV + embeddings, gom speaker, loại rác rõ ràng, cân bằng số clip mỗi
speaker, xuất tập sạch. Luôn chạy `--calibrate` trước để xem phân bố rồi mới chốt ngưỡng.

Giải thích tham số:
- `cluster_dist`: ngưỡng khoảng cách cosine cho agglomerative clustering trên embedding. Nhỏ -> nhiều cụm nhỏ (tách quá mức); lớn -> ít cụm, gộp mạnh hơn.
- `min_det_ratio = 0.6`: gate — loại clip có tỉ lệ frame thấy mặt dưới 0.6.
- `min_face_area = 0.01`: gate — loại clip mặt nhỏ hơn 1% khung (khó đọc khẩu hình).
- `min_consistency = 0.3`: gate — loại clip embed_consistency dưới 0.3 (lẫn nhiều người rõ ràng).
- `cap_per_speaker`: số clip tối đa giữ lại mỗi speaker (để 1 người không áp đảo dataset).
- `quality_score`: điểm 0..1 để ưu tiên khi cân bằng, ghép từ det_ratio (0.4), mean_face_area (0.3), embed_consistency (0.15), và sync_conf nếu có.

Terminal — calibrate (phân bố để chọn ngưỡng):

```
Tổng clip: 6888 | có mặt: 6872 | không mặt: 16

det_ratio          : mean 0.991  median 1.000  p10 1.000
mean_face_area     : mean 0.0427 median 0.0319 p10 0.0053  p25 0.0126  max 0.4922
embed_consistency  : mean 0.785  median 0.857  p10 0.452

Số speaker theo ngưỡng cluster:
  dist=0.3: 1507 speaker | cụm to nhất 636 | cụm 1-clip 886
  dist=0.4: 1298 speaker | cụm to nhất 639 | cụm 1-clip 731
  dist=0.5: 1181 speaker | cụm to nhất 639 | cụm 1-clip 631
  dist=0.6: 1073 speaker | cụm to nhất 639 | cụm 1-clip 543
  dist=0.7:  863 speaker | cụm to nhất 639 | cụm 1-clip 329
```

Chọn ngưỡng: `cluster_dist=0.6` (1073 speaker, ít singleton hơn các dist nhỏ), `min_face_area=0.01`,
`min_consistency=0.3`, `cap_per_speaker=30`. Lưu ý có 1 cụm khổng lồ 639 clip (1 speaker/kênh áp
đảo, ~9% data) không đổi theo dist -> cần cap để không lệch.

Terminal — export (cap=30):

```
[B2] 1073 speaker từ 246 video
[B3] gate: giữ 5356/6888, loại 1532 (không mặt / mặt nhỏ / mặt thưa / lẫn người)
[B4] sau cân bằng (cap 30/speaker): 3001 clip
     clip/speaker: min=1 med=2 max=30
[B5] xuất 3001 clip sạch -> data/curate/all_clean.csv
```

So sánh cap: cap=15 -> 2487 clip; cap=30 -> 3001 clip (chỉ +514). Phần data mất khi cap tập
trung ở vài cụm lớn -> đây là đánh đổi volume vs cân bằng. Chọn cap=30 để giữ nhiều hơn mà vẫn
ghìm cụm 639.

Funnel làm sạch: 6888 (scored) → 5356 (qua gate) → 3001 (sau cân bằng).
Đối chiếu: 6888 = 3001 sạch + 1532 gate loại + 2355 bị cap (không log riêng).

Output: `data/curate/all_clean.csv` (3001), `data/curate/all_clean_rejects.csv` (1532 gate loại).

---

## 5. Bước 05 — EDA (phân tích khám phá)

Mục đích: mô tả bộ real clip đã làm sạch — phân bố chất lượng, cân bằng tier, định danh
speaker, so sánh trước/sau curate. Sinh 11 biểu đồ PNG + `eda_summary.md`.

### 5.1 Bảng tổng quan

| Chỉ số | Giá trị |
|---|---|
| Tổng clip (scored) | 6888 |
| Số video gốc | 246 |
| Clip có mặt | 6872 (99.8%) |
| Clip không mặt | 16 (0.2%) |
| Clip sau curate (tập sạch) | 3001 |
| Số speaker (tập sạch) | 674 |

### 5.2 Phân bố theo tier

| Tier | Scored | Tập sạch | Tỉ lệ giữ |
|---|---|---|---|
| tier1 | 4776 | 1812 | 38% |
| tier2 | 1011 | 681 | 67% |
| tier3 | 1101 | 508 | 46% |

### 5.3 Phân bố đặc trưng (toàn bộ scored)

| Đặc trưng | n | mean | std | min | p10 | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| Thời lượng (s) | 6888 | 4.35 | 1.93 | 2.01 | 2.30 | 4.09 | 6.78 | 12.00 |
| SNR (dB) | 6888 | 9.76 | 8.75 | -16.54 | 1.54 | 7.27 | 20.39 | 40.27 |
| face_ratio (cut) | 6888 | 0.994 | 0.035 | 0.700 | 1.000 | 1.000 | 1.000 | 1.000 |
| speech_ratio (cut) | 6888 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| det_ratio (đo) | 6888 | 0.991 | 0.067 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| mean_face_area | 6888 | 0.0427 | 0.0490 | 0.0000 | 0.0053 | 0.0319 | 0.0828 | 0.4922 |
| embed_consistency | 6888 | 0.785 | 0.193 | 0.000 | 0.452 | 0.857 | 0.954 | 1.000 |

### 5.4 Định danh speaker (tập sạch)

| Chỉ số | Giá trị |
|---|---|
| Số speaker | 674 |
| Clip/speaker (min/median/max) | 1 / 2 / 30 |
| Speaker chỉ 1 clip | 310 |

`speaker_id` dùng để chia train/val/test theo speaker-disjoint, chống identity leakage ở bước modeling.

### 5.5 Biểu đồ — giải thích và nhận xét

(Ảnh nằm ở `data/curate/eda_figs/`.)

1. Thời lượng clip
![duration](../../data/curate/eda_figs/hist_duration.png)
Phân bố lệch phải, đa số 2–7s, median 4.09s, có sàn cứng 2s (từ bước cắt). Nhận xét: độ dài
phù hợp cho phân tích audio-visual cửa sổ ngắn; không có clip quá dài gây mất cân bằng.

2. SNR
![snr](../../data/curate/eda_figs/hist_snr.png)
Phân bố rất rộng (min −16.5dB đến 40dB, p10 chỉ 1.5dB). Nhận xét: phản ánh đúng triết lý "đo
SNR nhưng KHÔNG gate" ở bước cắt — giữ cả clip ồn làm metadata. Đây là trục có thể lọc về sau,
nhưng phải áp đối xứng real/fake để tránh leakage.

3. face_ratio (đo ở bước cắt)
![face_ratio](../../data/curate/eda_figs/hist_face_ratio.png)
Gần như dồn hết về 1.0 (đã bị gate ở bước cắt). Nhận xét: ít thông tin, đưa vào cho đầy đủ.

4. speech_ratio (đo ở bước cắt)
![speech_ratio](../../data/curate/eda_figs/hist_speech_ratio.png)
Bằng 1.0 cho mọi clip (std=0). Nhận xét: suy biến — VAD đã gate ở bước cắt, mọi clip đều toàn tiếng nói.

5. det_ratio (đo lại bằng InsightFace)
![det_ratio](../../data/curate/eda_figs/hist_det_ratio.png)
Dồn về 1.0, đuôi rất mỏng về 0 (16 clip không mặt + vài clip thưa). Nhận xét: một detector khác
(InsightFace) xác nhận lại sự hiện diện khuôn mặt mà bước cắt đo bằng YOLO — kiểm chéo, tăng độ tin.

6. mean_face_area
![mean_face_area](../../data/curate/eda_figs/hist_mean_face_area.png)
Lệch phải mạnh, median 3.2% khung, đuôi nhỏ dưới 1%, vài clip tới ~49%. Nhận xét: khung hình
kiểu tin tức/vlog nên mặt cỡ vừa-nhỏ là chủ yếu; phần đuôi <1% bị gate ở bước 04 (mặt quá nhỏ khó đọc khẩu hình).

7. embed_consistency
![embed_consistency](../../data/curate/eda_figs/hist_embed_consistency.png)
Đa số cao (median 0.86), có đuôi trái xuống thấp (p10 0.45). Nhận xét: phần lớn clip một danh
tính ổn định; đuôi thấp là clip lẫn nhiều người / cắt cảnh, đã loại phần dưới 0.3 ở bước 04.

8. Số clip theo tier (scored vs sạch)
![tier](../../data/curate/eda_figs/bar_tier_counts.png)
tier1 áp đảo ở dữ liệu thô (4776) nhưng sau cap tỉ lệ giữ thấp nhất (38%) vì tier1 chứa nhiều
cụm lớn; tier2 giữ cao nhất (67%). Nhận xét: bước cân bằng kéo 3 tier về tỉ lệ đều hơn, giảm
thiên lệch nguồn.

9. Có mặt vs không mặt
![face_presence](../../data/curate/eda_figs/bar_face_presence.png)
6872 có mặt / 16 không mặt. Nhận xét: 99.8% — chất lượng dữ liệu cao và quan trọng là xác nhận
remap path đúng (nếu sai path, cột này sẽ toàn "không mặt").

10. Top 20 video gốc theo số clip
![top_videos](../../data/curate/eda_figs/barh_top_videos.png)
Một số video đóng góp rất nhiều clip (video dài). Nhận xét: cho thấy lý do cần cap theo speaker —
tránh để vài nguồn/người chiếm phần lớn dataset.

11. Số clip mỗi speaker (tập sạch)
![clips_per_speaker](../../data/curate/eda_figs/hist_clips_per_speaker.png)
Lệch phải kể cả sau cap (trần 30), median 2, 310/674 speaker chỉ 1 clip. Nhận xét: đa dạng danh
tính cao (nhiều speaker), nhưng mỗi speaker ít clip; cap đã làm phẳng cụm khổng lồ 639.

---

## 6. Tổng hợp file output

| Bước | File | Mô tả |
|---|---|---|
| 01 | `data/clips/all_manifest.csv` | 6888 dòng — manifest gộp 3 tier, đã verify |
| 02 | `data/curate/tier1_scored_all.csv` | 6888 dòng — manifest + face stats |
| 02 | `data/curate/embeddings_all.npy` | (6888, 512) face embedding |
| 03 | `data/curate/calibrate_sync_results.csv` | mẫu 30 clip (calibrate) |
| 03 | `data/curate/sync_calibrate_log.txt` | log calibrate sync |
| 04 | `data/curate/all_clean.csv` | 3001 clip sạch (có speaker_id) |
| 04 | `data/curate/all_clean_rejects.csv` | 1532 clip bị gate loại |
| 05 | `data/curate/eda_figs/` | 11 PNG + eda_summary.md |

Tất cả nằm trong `data/` (gitignore) — không commit lên repo.

---

## 7. Đánh giá chung và việc tiếp theo

Đánh giá:
- Dataset real sạch về mặt khuôn mặt (99.8% có mặt, det_ratio ~1.0) và lời nói (speech_ratio=1.0).
- Sau làm sạch + cân bằng: 3001 clip / 674 speaker, 3 tier về tỉ lệ đều hơn, không speaker nào áp đảo (trần 30).
- SNR và mean_face_area giữ phân bố rộng làm metadata (không gate cứng) — đúng triết lý "đo mọi thứ, chỉ loại rác rõ ràng".
- 03 (sync) đã dựng xong và chạy được, nhưng để lại cho pilot do chậm trên CPU + chưa đáng tin để gate + tránh leakage.

Việc tiếp theo:
- Viết mục Label design + leakage audit (Option A — phân tích, không cần sinh fake).
- Pilot (sau hạn): nới min_track/min_face_size + GPU torch -> chạy 03 trên subset real + fake để vẽ histogram sync tách lớp.
- Thu thập thêm theo lỗ hổng đa dạng mà EDA chỉ ra (speaker 1-clip nhiều, tier1 đuôi dài).

---

## 8. Lý giải lựa chọn tham số (vì sao chọn con số này)

Phân biệt: [mặc định] = số do tác giả script/SyncNet đặt, không tùy chỉnh theo data;
[tự chọn] = số tôi chốt trong run này dựa trên calibrate.

### Bước 02 — score_clips

- `k_frames = 9` [mặc định]: đánh đổi giữa độ mịn và chi phí. 5 frame quá thô (det_ratio chỉ nhảy bậc 0 / 0.2 / 0.4...); 9 frame cho bước ~0.11 nên det_ratio mịn hơn và trung bình embedding ổn định hơn. Số lẻ để có 1 frame chính giữa. Clip chỉ 2–12s nên trên ~10 frame lợi ích giảm dần mà chi phí decode tăng tuyến tính.
- `det_size = (640,640)` [mặc định]: kích thước đầu vào chuẩn của buffalo_l — cân bằng giữa bắt được mặt nhỏ và tốc độ. To hơn (1024) bắt mặt nhỏ tốt hơn nhưng chậm.
- `MIN_DET_SCORE = 0.5` [mặc định]: điểm giữa của thang tin cậy. Dưới 0.5 hay nhận nhầm vật thể thành mặt; trên 0.5 bắt đầu bỏ sót mặt nghiêng/mờ.
- `CONSISTENCY_AVG_MIN = 0.5` [mặc định]: ngưỡng quyết định "trung bình embedding các frame" (khi đồng nhất) hay "lấy medoid" (khi lẫn người). Cosine 0.5 xấp xỉ ranh giới cùng-một-người của ArcFace.

### Bước 03 — sync_score (toàn bộ là mặc định của SyncNet)

- `min_track = 100` frame, ở 25fps = 4 giây [mặc định]: SyncNet ước lượng offset bằng tương quan trượt giữa chuỗi môi và chuỗi âm; cần đủ số frame thì confidence mới ổn định. Track ngắn hơn cho LSE nhiễu/không tin được. Đây trực tiếp gây FAIL 53%: nhiều clip của ta chỉ 2–4s, hoặc bị cắt cảnh thành đoạn < 4s.
- `min_face_size = 100` px [mặc định]: bộ mã hóa môi cần đủ pixel ở vùng miệng; mặt nhỏ hơn 100px quá ít thông tin.
- `frame_rate = 25` [mặc định]: model SyncNet được huấn luyện trên 25fps (lưới căn audio-video 40ms/frame khớp với bước MFCC của audio). Phải resample về 25 thì kết quả mới đúng chuẩn.
- `num_failed_det = 25` [mặc định]: cho phép mất phát hiện mặt liên tiếp tối đa 25 frame (~1s, do chớp mắt/quay đầu/che) trước khi ngắt track — giữ track liền mạch qua che khuất ngắn.
- `facedet_scale = 0.25`, `crop_scale = 0.40` [mặc định]: thu nhỏ ảnh khi detect cho nhanh; nới bbox 40% khi crop để lấy đủ cằm/miệng quanh khuôn mặt.

### Bước 04 — curate

- `cluster_dist = 0.6` [tự chọn]: ngưỡng khoảng cách cosine cho agglomerative. Với embedding ArcFace đã L2-norm, cặp cùng-người thường có khoảng cách < ~0.6 (tương đồng > 0.4), khác-người > ~0.7. Calibrate thử 0.3–0.7: các mức 0.3–0.5 tách quá vụn (rất nhiều cụm 1-clip: 886/731/631), 0.7 gộp mạnh dễ nhập nhầm hai người vào một cụm. 0.6 là điểm cân bằng giữa tách quá mức và gộp quá mức. (Lưu ý: kể cả 0.6 vẫn over-segment — 1073 cụm cho 246 video — do embedding đổi theo góc/ánh sáng.)
- `min_det_ratio = 0.6` [mặc định]: đòi mặt xuất hiện ở ≥60% frame mẫu. Để mức nới (không đòi 100%) nhằm giữ clip chỉ thỉnh thoảng mất mặt — đúng tinh thần "chỉ loại rác rõ ràng". Vì data có median=1.0 nên ngưỡng này chỉ cắt phần đuôi thật sự kém.
- `min_face_area = 0.01` [mặc định]: mặt phải ≥1% diện tích khung. Dưới 1% thì vùng miệng quá nhỏ để đọc khẩu hình (và cho model lip-sync sau này). Calibrate cho p10=0.0053, p25=0.0126 nên 0.01 nằm giữa, loại ~15–20% clip mặt nhỏ nhất. (Có thể hạ về 0.005 nếu muốn giữ nhiều hơn.)
- `min_consistency = 0.3` [tự chọn; mặc định gốc = 0 = tắt]: embed_consistency dưới 0.3 gần như chắc chắn là clip lẫn ≥2 người / cắt cảnh mạnh. Chọn 0.3 chứ không cao hơn vì p10 = 0.452 — đặt 0.3 chỉ loại nhóm dưới 10% tệ nhất, tránh loại nhầm clip chỉ hơi đổi góc (over-cleaning).
- `cap_per_speaker = 30` [tự chọn]: số clip tối đa mỗi speaker. Cụm lớn nhất có 639 clip (~9% toàn bộ); để nguyên thì một danh tính áp đảo → model dễ học tủ người đó (shortcut). Thử cap=15 ra 2487 clip (quá mạnh tay: median speaker chỉ 2 clip nên cap chủ yếu chặt vài speaker năng suất hợp lệ); cap=30 ra 3001 clip, giữ nhiều hơn mà vẫn ghìm cụm 639 về 30 (~1%). 30 là dung hòa volume vs cân bằng.
- `quality_score` = 0.4·det_ratio + 0.3·mean_face_area + 0.15·embed_consistency [mặc định]: trọng số phản ánh mức quan trọng — hiện diện mặt (điều kiện cơ bản nhất) nặng nhất, kế đến mặt đủ to (đọc được miệng), consistency làm tiêu chí phụ. Dùng để xếp hạng clip khi cap, giữ clip tốt nhất mỗi speaker.
