# Nhật ký 2026-07-28 — Kế hoạch multi-reviewer và phân phối dữ liệu

**Ngày kiểm tra:** 2026-07-28

**Checkpoint:** `d059340` (feat(review): multi-reviewer assignment + merge cho manual curation)

**Phạm vi:** rà soát tooling review sau khi bổ sung assignment/merge; tính lại số clip cần review cho repaired pilot; xác định bộ dữ liệu cần đưa lên Drive cho reviewer

**Trạng thái:** repaired pilot vẫn **tạm dừng**; manual review đang ở 60/3.001

## 1. Đã commit trong phiên này

| File | Vai trò |
|---|---|
| `src/tools/build_review_assignments.py` | Chia primary disjoint theo tier + calibration set dùng chung |
| `src/tools/merge_review_results.py` | Gộp kết quả, fail-closed, tách clip cần phân xử |
| `tests/test_manual_review_workflow.py` | Test end-to-end disjoint / calibration / adjudication |
| `src/tools/clip_review.py` | Sửa thứ tự `--exclude_channel`/`--sample`, seed cho `diverse`, output tách theo reviewer |
| `src/pipeline/02_curate/README.md`, `PROJECT.md` | Cập nhật quy trình |

Toàn bộ 19 test trong `tests/` pass (1 skip).

Bốn khiếm khuyết nêu ở mục 4 nhật ký 2026-07-27 đã xử lý xong:

1. `--exclude_channel` giờ áp **trước** `--sample`, không còn làm sample nhỏ hơn yêu cầu.
2. `diverse_order` nhận seed.
3. Docstring không còn hard-code `43%` / `1125` / `209-209`; thay bằng cảnh báo rằng tỉ lệ keep chỉ là ước lượng từ mẫu nhỏ và đa dạng hàng đợi không suy ra đa dạng tập keep.
4. `build_review_manifest.py` dựng lại `manifests/all_clean_review.csv` tái lập được — đã verify trùng khớp từng ô với bản tạo tay trước đó.

Ngoài ra `measurements/face_ambiguity.json` (3.001 clip, kết quả quét ~40 phút) trước đây chỉ nằm trong thư mục tạm, nay đã đưa vào `data/02_curate/measurements/` và `scan_face_ambiguity.py` mặc định ghi vào đó.

## 2. Tooling multi-reviewer: đủ dùng, trừ một điểm chặn

### 2.1 Phần đã có

- Chia primary **disjoint** — mỗi clip đúng một reviewer, cân bằng theo tier, shuffle có seed. Builder fail nếu coverage khác đúng 1 lần/clip.
- **Calibration set dùng chung** cho mọi reviewer — cơ sở duy nhất để đo đồng thuận giữa người với người.
- Merge **fail-closed**: sai rubric, thiếu coverage, hoặc kết quả nằm ngoài assignment đều dừng. Chỉ xuất `manual_clean_v2.csv` khi đã phân xử hết.
- Output tách theo reviewer, chặn trộn nhãn của hai người vào cùng file.

### 2.2 Điểm chặn: `file_path` là đường dẫn tuyệt đối

`manifests/all_clean_review.csv` lưu đường dẫn dạng:

```
E:\FPTU\PRJ\VN-AV-DF-Capstone\data\01_collect\cut_clips\tier1\...\<clip>.mp4
```

`clip_review.py` chưa có tuỳ chọn đổi gốc đường dẫn (`--media_root` hoặc tương đương). Reviewer đặt dữ liệu ở ổ/thư mục khác sẽ thấy toàn bộ clip báo thiếu file.

Đây là điều kiện cần trước khi phát dữ liệu; sửa nhỏ nhưng chưa làm trong phiên này.

### 2.3 Chưa kiểm chứng

Workflow mới chỉ chạy qua test tổng hợp, **chưa chạy thật với hai người trên hai máy**. Nên thử một vòng nhỏ (vài chục clip) trước khi phát toàn bộ.

## 3. Ý tưởng "1.000 clip cho 3 người": số học không đủ

Tỉ lệ keep đo trên 60 clip calibration: **26/60 = 43,3%**, khoảng tin cậy Wilson 95% **[32%, 56%]**.

Số keep kỳ vọng theo số clip đưa vào review:

| Clip review | Keep kỳ vọng | Cận dưới KTC | Cận trên KTC | Đủ 540 real? |
|---:|---:|---:|---:|---|
| 1.000 | 433 | 316 | 559 | **Không** |
| 1.200 | 520 | 379 | 671 | Không |
| 1.400 | 607 | 442 | 783 | Có thể |
| 1.600 | 693 | 505 | 894 | Có thể |
| 1.800 | 780 | 568 | 1.006 | Chắc chắn |

**1.000 clip cho kỳ vọng 433 keep — thiếu so với 540 real cần cho pilot.** Ngay cả 1.200 clip vẫn chưa chắc chắn.

Để cận dưới khoảng tin cậy vượt 540 cần review **≥ 1.710 clip**.

Ba lưu ý về con số này:

1. Tỉ lệ 43,3% đến từ mẫu 60 clip phân tầng theo motion, các clip chung nguồn không độc lập — sai số thực có thể rộng hơn khoảng Wilson.
2. Đủ 540 keep **chưa đủ điều kiện** publish pilot. Còn phải kiểm tier, speaker, source video, channel, connected-component split và khả năng sinh đủ bốn fake cho mỗi source.
3. Nếu đặt mục tiêu theo **số keep** thay vì số clip review thì không cần chốt trước con số N — cứ review tới khi đủ rồi dừng. Đây là cách an toàn hơn.

### 3.1 Khối lượng cho mỗi người

Giả sử 1.710 clip primary chia ba, cộng calibration set 60 clip dùng chung:

- mỗi người: **570 primary + 60 calibration = 630 clip**
- tốc độ đo thật ở phiên calibration: median 6,0 giây/clip, **mean 9,8 giây/clip**
- ước tính: **1,0–1,7 giờ/người**, chưa tính mệt mỏi và thời gian phân xử

Dùng mean chứ không dùng median khi lập kế hoạch — phân bố có đuôi dài (P90 = 22 giây/clip).

## 4. Dữ liệu cần đưa lên Drive

### 4.1 Vị trí nguồn

3.001 clip gốc **không nằm trong một thư mục**, mà rải theo 7 thư mục tier:

| Thư mục | Số clip |
|---|---:|
| `data/01_collect/cut_clips/tier2/tier2_v3_clips_0_999-001` | 681 |
| `data/01_collect/cut_clips/tier1/100-300/tier1_v3_clips_100_300` | 665 |
| `data/01_collect/cut_clips/tier3/tier3_v3_clips_0_1262-001` | 508 |
| `data/01_collect/cut_clips/tier1/300-400/tier1_v3_clips_300_400` | 462 |
| `data/01_collect/cut_clips/tier1/0-100/tier1_v3_clips_0_100` | 368 |
| `data/01_collect/cut_clips/tier1/440-9999/tier1_v3_clips_440_9999` | 168 |
| `data/01_collect/cut_clips/tier1/400-440/tier1_v3_clips_400_440` | 149 |

Cả 3.001 clip đều có mặt trên đĩa. Thư mục `data/01_collect/` tổng cộng 23 GiB vì chứa cả clip đã bị loại; riêng 3.001 clip trong `all_clean` chỉ **6,83 GiB**.

### 4.2 Dung lượng

| Nội dung | Mỗi clip | 3.001 clip | 1.710 clip | 570 clip (1 người) |
|---|---:|---:|---:|---:|
| Clip gốc | 2,33 MiB | 6,83 GiB | 3,89 GiB | 1,30 GiB |
| `roi_preview/` | 171 KiB | 501 MiB | 285 MiB | 95 MiB |
| **Cả hai** | 2,50 MiB | **7,32 GiB** | 4,17 GiB | 1,39 GiB |

### 4.3 Phải gửi cả hai, không chỉ ROI

`roi_preview/` rẻ hơn 14 lần nhưng **không đủ một mình**. Ô ROI chỉ là vùng miệng 96×96; muốn phán `dubbed` phải nhìn được khuôn mặt và bối cảnh trong video gốc — ví dụ nhận ra người nói là người nước ngoài, hoặc cảnh là phóng sự B-roll. `clip_review.py` hiển thị song song hai luồng chính vì lý do đó.

### 4.4 Cách phát khả thi nhất

Gửi **theo phần** thay vì gửi cả bộ: mỗi người chỉ cần media của assignment mình (~1,4 GiB) thay vì 7,32 GiB. Cần một script đóng gói theo file assignment — hiện chưa có.

Kèm theo mỗi gói: file assignment CSV, thư mục media tương ứng, và hướng dẫn chạy `clip_review.py` với gốc đường dẫn cục bộ (phụ thuộc mục 2.2).

## 5. Việc cần làm trước khi phát dữ liệu

1. Thêm tuỳ chọn đổi gốc đường dẫn media cho `clip_review.py` (mục 2.2).
2. Viết script đóng gói media theo assignment (mục 4.4).
3. Chạy thử workflow với hai người trên hai máy, quy mô nhỏ.
4. Chốt reviewer ID ổn định — merge dựa vào `reviewer_id`, đổi tên giữa chừng sẽ hỏng coverage.
5. Quyết định mục tiêu theo **số keep** thay vì số clip review, và dự trù khả năng phải review vượt 1.710 clip.

## 6. Kết luận

Tooling multi-reviewer đã có phần khó nhất — chia việc disjoint, calibration chung, merge fail-closed và test. Còn thiếu hai mảnh nhỏ nhưng bắt buộc: đổi gốc đường dẫn media và đóng gói theo assignment.

Điều chỉnh quan trọng nhất về kế hoạch: **1.000 clip không đủ**. Kỳ vọng chỉ 433 keep so với 540 real cần cho pilot. Con số hợp lý là khoảng 1.700 clip, tương đương 630 clip mỗi người khi chia ba, và nên bám mục tiêu theo số keep thay vì chốt cứng số clip review.
