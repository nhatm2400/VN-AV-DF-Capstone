# archive/pilot_v1 — đợt sinh fake V1 và pilot V1

**Ngày lưu trữ:** 2026-07-28 · **Checkpoint:** `a228c69`

Toàn bộ artifact của **đợt sinh fake đầu tiên (V1)** và **pilot V1**. Giữ lại để phòng
ngừa và truy vết, **không dùng cho lượt chạy mới**.

Đây là **lưu trữ, không phải rác tạm.** Đừng xóa mà chưa đọc mục "Có được xóa không".

## Vì sao ở đây

Pilot V1 chạy xong cho test ROC-AUC 0,809, nhưng phân tích theo từng kênh cho thấy con
số tổng che giấu khoảng cách lớn (`pitch_flatten` 0,990 so với `frame_reverse` 0,535),
và kiểm tra đối kháng phát hiện baseline tầm thường giải được hai kênh dễ cùng một lỗi
tạo tác trong generator temporal. Quyết định là **NO-GO** cho huấn luyện đầy đủ với V1.

Kế hoạch hiện tại: kiểm duyệt lại nguồn real bằng tay, sinh lại fake bằng generator V2,
rồi dựng repaired pilot. Khi đó dữ liệu ở đây thành lỗi thời — nhưng vẫn giữ để đối
chiếu nếu kết quả mới có gì bất thường.

Chi tiết: [PILOT_REPORT](../../docs/reports/PILOT_REPORT.md) ·
[PILOT_V1_REVIEW_AND_V2_PLAN](../../docs/reports/PILOT_V1_REVIEW_AND_V2_PLAN.md)

## Có gì

| Đường dẫn | Nội dung | Dung lượng |
|---|---|---|
| `03_fake/*.mp4` | 12.004 fake V1 (4 method × 3.001 clip) | 20,86 GiB |
| `03_fake/snvsm/` | Media đã chuẩn hoá codec + 4 manifest | 5,84 GiB |
| `03_fake/labels.csv` | 12.004 nhãn fake V1 | 2,45 MiB |
| `04_features_pilot/` | 2.700 tensor `.pt` + `features_index.csv` | 3,29 GiB |
| `05_labels/labels.csv` | 15.005 nhãn real+fake kèm cột split | 3,26 MiB |
| `05_labels/labels_pilot.csv` | 2.700 nhãn của riêng pilot | 0,59 MiB |

Tổng **khoảng 30 GiB**, trong đó chỉ **11,4 MiB manifest CSV được commit** — media bị
`.gitignore` chặn qua luật `*.mp4` / `*.pt`.

## Không nằm ở đây

| Thứ | Ở đâu | Vì sao |
|---|---|---|
| Smoke test Phase 0 V2 (`phase0_*`) | vẫn ở `data/03_fake/` | Bằng chứng V2 **hiện hành**, không phải V1 |
| `labels_phase0_*.csv` | vẫn ở `data/05_labels/` | Cùng lý do trên |
| Checkpoint + metrics pilot V1 | `experiments/pilot_v1_20260720-214741_.../` | Là một thí nghiệm, giữ nguyên chỗ theo quy ước experiment bất biến |

## Đường dẫn đã được viết lại

Mọi manifest ở đây trước kia trỏ vào `data/03_fake/...` và `data/04_features_pilot/...`.
Nếu chuyển thư mục mà không sửa, **50.114 đường dẫn** sẽ trỏ vào chỗ trống và bản lưu
trữ thành vô dụng — nên tất cả đã được viết lại sang `archive/pilot_v1/...` và kiểm
chứng: toàn bộ 15.005 dòng của `05_labels/labels.csv` và 2.700 dòng của
`features_index.csv` đều trỏ tới file có thật.

**Lưu ý:** đường dẫn trong các manifest này là **tuyệt đối** (`E:\FPTU\PRJ\...`) trừ
`features_index.csv` dùng đường dẫn tương đối. Trên máy khác phải viết lại lần nữa.

**Sót một chỗ, đã sửa 2026-07-29.** Lượt lưu trữ này chỉ viết lại đường dẫn trong các
CSV *được chuyển đi*, nên bỏ quên ba file nằm lại trong `data/`:
`phase0_smoke_v2r{2,3,4}/fake_all.csv` — chúng trộn output temporal_v2 mới với fake V1
cũ, và **54 đường dẫn** trong đó trỏ vào `data/03_fake/*.mp4` đã bị dời sang đây. Nay
đã trỏ đúng vào `archive/pilot_v1/03_fake/`; ba file đó giờ nằm ở
[`archive/phase0_v2_smoke/`](../phase0_v2_smoke/README.md).

Bài học cho lần sau: khi dời media, phải quét **mọi** CSV trong repo tìm tham chiếu tới
đường dẫn cũ, không chỉ các CSV cùng đi theo.

## Bản khóa toàn vẹn của experiment — vẫn xác minh được

`experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/manifest_hashes.json` khóa 8
SHA-256. Bốn trong đó là **manifest input đã bị lượt lưu trữ này dời đi và viết lại**,
nên không còn khớp trực tiếp:

| Nhóm | Trạng thái |
|---|---|
| 4 artifact mô hình (`best.pt`, `last.pt`, `history.json`, `eval_test.json`) | **khớp trực tiếp** — kết quả thí nghiệm nguyên vẹn |
| 4 manifest input (`pilot_{real,fake}_snvsm.csv`, `labels_pilot.csv`, `features_index.csv`) | lệch, **chỉ do đổi tiền tố đường dẫn** |

`manifest_hashes.json` **cố ý không được sửa.** Cập nhật nó thành hash hôm nay sẽ xóa
mất khả năng phát hiện sửa đổi thật sự về sau — bản khóa mà tự động chạy theo dữ liệu
thì không còn là bản khóa.

Thay vào đó dùng [`verify_lock.py`](verify_lock.py): nó đảo ngược đúng phép thế tiền tố
trong bộ nhớ rồi đối chiếu. Chạy từ thư mục gốc repo:

```bash
D:\Anaconda\envs\vn_av_df\python.exe archive/pilot_v1/verify_lock.py
```

Kết quả 2026-07-29: **8/8 khớp** — 2.700 dòng `labels_pilot.csv` và 2.700 dòng
`features_index.csv` đảo ngược ra đúng byte gốc. Nghĩa là ngoài tiền tố đường dẫn,
không có gì trong bốn manifest bị thay đổi.

Nếu sau này script báo lệch, đó là tín hiệu thật: có ai đó đã sửa nội dung, không phải
chỉ đổi chỗ file.

## Có được xóa không

Chưa quyết. Câu hỏi quyết định: **có bao giờ cần tái tạo lại feature V1 từ video không?**

- **Không cần** → xóa được `03_fake/*.mp4` (20,86 GiB) và `03_fake/snvsm/` (5,84 GiB),
  giữ lại toàn bộ CSV manifest làm provenance. Thu hồi khoảng **26,7 GiB**.
- **Có cần** → giữ nguyên như hiện tại.

`04_features_pilot/` (3,29 GiB) tái tạo được từ media nếu media còn; mất media thì
không tái tạo được nữa.

Xem thêm [nhật ký 2026-07-27 mục 7](../../docs/logs/2026-07-27_MANUAL_CURATION_AND_STORAGE_AUDIT.md).
