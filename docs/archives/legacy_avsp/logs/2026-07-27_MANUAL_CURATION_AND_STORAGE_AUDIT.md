# Nhật ký 2026-07-27 — Manual curation và audit dung lượng

**Ngày kiểm tra:** 2026-07-27

**Checkpoint nền:** `871e69f`

**Phạm vi:** đọc lại log trao đổi về manual review, đối chiếu `clip_review.py` và dữ liệu hiện có; audit read-only `data/02_curate` và `data/03_fake`

**Trạng thái:** repaired pilot tiếp tục **tạm dừng**; chưa xóa dữ liệu và chưa chạy lại generator/SNVSM/feature/train

## 1. Mục tiêu

Phiên kiểm tra trả lời ba câu hỏi:

1. Manual review có phải gate bắt buộc trước repaired pilot không?
2. Những kết luận nào trong cuộc trao đổi với AI khác được dữ liệu hỗ trợ, và kết luận nào đang quá mạnh?
3. File nào trong `data/02_curate` và `data/03_fake` phải giữ, có thể archive hoặc có thể xóa có điều kiện?

## 2. Trạng thái manual curation đo được

Có hai artifact manual khác scope:

| Artifact | Scope | Trạng thái |
|---|---|---|
| `manual/manual_review.csv` | manifest 6.888 clip cũ | 47 quyết định; chủ yếu là clip balance-drop, không đại diện `all_clean` |
| `manual/manual_all_clean_review_v2.csv` | 3.001 clip trong `manifests/all_clean_review.csv` | 60 quyết định: 26 keep, 34 reject |

Phân bố 34 reject của batch V2:

| Lý do | Số clip |
|---|---:|
| `dubbed` — lồng tiếng | 14 |
| `cut` — cắt cảnh/B-roll | 7 |
| `static` — ảnh tĩnh/miệng không động | 6 |
| `mouth` — miệng không dùng được | 3 |
| `broken` — media/audio lỗi hoặc quá ít tiếng | 3 |
| `voiceover` — người ngoài hình nói | 1 |

Kết quả **đã đo được** là automatic curation giữ nhầm 34/60 clip theo rubric manual V2. Đây là contamination nghiêm trọng và đủ để chặn repaired pilot.

Kết quả **chưa được phép khẳng định** là 56,7% của toàn bộ 3.001 clip đều là rác. Mẫu mới có 60 dòng, 42 source video và được phân tầng theo motion; các clip chung nguồn không độc lập. Tỉ lệ keep 43,3% chỉ là ước lượng ban đầu.

## 3. Đánh giá cuộc trao đổi với AI khác

### 3.1 Những điểm đúng

- Manual review là gate tiên quyết: VAD, face detection, motion và scene-cut không xác nhận người nhìn thấy là người phát ra tiếng.
- `risk_score` hiện tại không tách được clip tốt/xấu trên batch 60: nhóm “rủi ro” reject 60%, nhóm còn lại 55%.
- ROI preview có tiếng là công cụ hữu ích cho người review phát hiện lồng tiếng, voice-over, cắt nhầm mặt và miệng không dùng được.
- Thứ tự round-robin theo `source_video` phù hợp hơn việc xếp theo risk score khi mục tiêu là thu tập pilot đa dạng nguồn.
- Không nên loại toàn bộ TikTok: 508 clip TikTok trải trên 152 source video, là phần lớn diversity nguồn.

### 3.2 Những điểm cần sửa cách diễn giải

1. AUC 0,544 chỉ chứng minh Pearson correlation giữa pixel-motion của mouth ROI và audio RMS không hữu ích trên batch 60; nó không chứng minh mọi active-speaker/lip-sync model đều vô dụng.
2. HYTV có 3 keep/13 và TikTok có 2 keep/12, nhưng mẫu theo channel quá nhỏ để auto-reject cả channel. Fisher exact so với phần còn lại lần lượt cho `p≈0,122` và `p≈0,052`; channel chỉ nên là metadata phân tầng/ưu tiên review.
3. Ước tính 1.125 clip trong hai giờ dùng median 6 giây/clip. Mean thực đo là 9,8 giây/clip, tương đương khoảng 3,1 giờ; cộng fatigue và adjudication nên dự trù 3–5 giờ.
4. Đạt 600 keep chưa đủ để publish pilot. Cần kiểm tier, speaker, source video, channel, connected-component split và khả năng sinh đủ bốn fake/source.
5. Diversity của hàng đợi đã xem không tự động bảo đảm diversity của tập keep; phải audit chính tập keep sau manual review.

## 4. Đánh giá `clip_review.py` hiện tại

Các phần đang đi đúng hướng:

- scope mặc định là `manifests/all_clean_review.csv` gồm 3.001 clip;
- reason, `uncertain`, reviewer ID và rubric version;
- ROI preview có audio;
- ghi quyết định atomic và resume;
- thứ tự `diverse` round-robin qua source video.

Các điểm cần xử lý trước khi khóa tool:

- `--exclude_channel` đang chạy sau `--sample`; dùng đồng thời có thể làm sample nhỏ hơn yêu cầu và thay đổi sampling design;
- không nên khuyên loại toàn bộ HYTV từ batch 13 clip;
- `diverse_order` chưa shuffle có seed trong từng source và chưa cân bằng tier/channel/speaker;
- docstring đang hard-code các ước lượng tạm thời `43%`, `1125` và `209/209`;
- cột `channel` được thêm bằng lệnh one-off vào CSV, chưa có builder tái lập;
- chưa có builder xuất manifest manual-clean, fail nếu thiếu coverage và kiểm diversity/split.

Lệnh được đề xuất trong log có `--exclude_channel "Truyền hình Hưng Yên - HYTV"`. Không nên dùng exclusion này làm mặc định. HYTV có thể được ưu tiên review hoặc phân tầng, nhưng quyết định keep/reject nên dựa trên nội dung clip.

## 5. Gate đề xuất trước repaired pilot

1. Team ba người dùng `build_review_assignments.py`: cùng review tập calibration, sau đó
   chia độc quyền phần primary khoảng 1.000 clip/người; không auto-reject theo channel.
2. Mở rộng gold set lên ít nhất 250–300 clip, trải theo tier/source/channel, rồi mới ước lượng lại contamination.
3. Khi có 600–650 keep, chạy builder chọn đúng 540 real theo connected component `speaker_id ∪ source_video`.
4. Builder phải kiểm diversity, media tồn tại, rubric version, không `uncertain` và đủ headroom để sinh bốn fake/source.
5. Dùng `merge_review_results.py` kiểm coverage, tách case cần adjudication và chỉ xuất
   `manifests/manual_clean_v2.csv` khi đủ 3.001 quyết định; không ghi đè `all_clean.csv`.
6. Chỉ sau khi gate đạt mới regenerate bốn fake V2, SNVSM, Stage 05 và metadata gate.

## 6. Audit `data/02_curate`

Tổng đo được: **3.024 file, 0,509 GiB**.

| Nhóm | Dung lượng | Quyết định hiện tại |
|---|---:|---|
| `roi_preview/` — 3.001 video | 0,489 GiB | **Giữ**: đang cần cho manual review |
| `measurements/embeddings_all.npy` | 13,453 MiB | **Giữ**: nguồn embedding, tránh chạy lại scoring |
| `measurements/tier1_scored_all.csv` | 1,711 MiB | **Giữ**: measurement source |
| `measurements/tier1_scored_motion.csv` | 2,018 MiB | Giữ đến khi manual pipeline được khóa; có thể regenerate |
| `measurements/lipcorr.json` | 60 clip | **Giữ**: phép đo tương quan môi–âm thanh trên đúng scope manual review V2 |
| `manifests/all_clean_review.csv` | 1,145 MiB | **Giữ**: scope review hiện tại; cần làm cách sinh file tái lập |
| `manifests/all_clean.csv`, rejects, EDA, sync calibration | dưới 2 MiB tổng chính | **Giữ**: source/provenance và báo cáo |
| ba CSV trong `manual/` | rất nhỏ | **Không xóa**: ground truth và lịch sử rubric |

Kết luận: `data/02_curate` không phải nguyên nhân tốn dung lượng. Chỉ `roi_preview/` đáng kể, nhưng chưa được xóa khi review chưa hoàn tất. Sau khi manual curation được khóa và backup, ROI preview có thể regenerate nên là ứng viên cleanup khoảng **0,489 GiB**.

## 7. Audit `data/03_fake`

Tổng đo được: **27.349 file, 27,065 GiB**.

| Nhóm | File | Dung lượng | Vai trò |
|---|---:|---:|---|
| Fake V1 trực tiếp ở root `data/03_fake/*.mp4` | 12.004 | 20,859 GiB | bốn fake/source V1; không dùng cho repaired pilot |
| `snvsm/fake/` V1 | 12.004 | 4,627 GiB | SNVSM fake V1 |
| `snvsm/real/` V1 | 3.001 | 1,206 GiB | SNVSM real V1 |
| Các smoke Phase 0 | 334 | 0,367 GiB | bằng chứng/debug r1–r6 |
| `labels.csv` và manifest SNVSM | 5 CSV | khoảng 7 MiB | provenance V1/pilot, phải giữ |

### 7.1 Có thể giải phóng nhiều nhất

Nếu không cần tái trích feature V1, hai nhóm media sau có thể archive rồi xóa:

- 12.004 raw fake V1: khoảng **20,859 GiB**;
- `snvsm/{real,fake}` V1: khoảng **5,833 GiB**.

Tổng dung lượng có thể thu hồi: khoảng **26,692 GiB**.

Việc xóa media này sẽ làm mất khả năng tái tạo feature V1 từ video. Checkpoint/eval V1 đã lưu vẫn còn, nhưng provenance không còn đầy đủ nếu không có bản archive.

### 7.2 Chưa nên xóa lúc này

- `data/03_fake/labels.csv`;
- `snvsm/*.csv`, đặc biệt hai manifest pilot;
- `phase0_stratified_smoke_v2r6/`, vì đây là bằng chứng smoke hiện hành;
- các smoke r4/r5 nếu còn cần tái kiểm báo cáo lịch sử;
- bất kỳ V2 media nào được tạo sau manual curation.

Các smoke cũ chỉ chiếm khoảng 0,17 GiB ngoài r6, nên xóa chúng không mang lại lợi ích đáng kể.

## 8. Quyết định cleanup

**Không xóa gì trong phiên này.**

Thứ tự an toàn nếu cần dọn sau:

1. Backup hoặc xác nhận không cần reproduce V1 extraction.
2. Giữ toàn bộ CSV manifest, manual labels, report và experiment V1.
3. Xóa có mục tiêu media V1, không xóa cả `data/03_fake/snvsm/` vì bên trong có manifest cần giữ.
4. Verify số file/dung lượng và các artifact giữ lại sau cleanup.
5. Chỉ dọn `roi_preview/` sau khi manual review hoàn tất và manifest manual-clean đã được khóa.

## 9. Kết luận

Manual curation hiện là blocker số một. Dữ liệu mới xác nhận automatic curation có false-accept nghiêm trọng trên batch 60, nhưng chưa đủ để auto-exclude channel hoặc ngoại suy chính xác toàn bộ 3.001 clip.

`data/02_curate` nên được giữ gần như nguyên trạng. Phần chiếm dung lượng thực sự là media V1 trong `data/03_fake`; khoảng 26,7 GiB có thể archive/xóa có điều kiện sau khi chốt nhu cầu reproducibility, không phải trước manual curation.
