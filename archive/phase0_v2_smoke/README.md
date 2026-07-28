# archive/phase0_v2_smoke — smoke test generator V2 (Phase 0)

**Ngày lưu trữ:** 2026-07-29 · **Checkpoint:** `fc3e75a`

Sáu vòng smoke test của **generator V2**, chạy 2026-07-21 → 2026-07-22. Đây là **bằng
chứng cơ chế V2 đã đạt**, khác hẳn [`archive/pilot_v1/`](../pilot_v1/README.md) vốn là
dữ liệu V1 đã bị bác bỏ.

Đây là **lưu trữ, không phải rác tạm.** Tổng 376 MB, trong đó 404 KB manifest/JSON
được commit làm provenance; media bị `.gitignore` chặn qua luật `*.mp4`.

## Vì sao ở đây

Sáu vòng này chỉ chạy trên vài clip để kiểm **cơ chế**, không phải dữ liệu huấn luyện.
Repaired pilot sẽ sinh lại fake V2 trên tập real đã lọc tay — khi đó số liệu ở đây
thành lịch sử. Giữ vì chúng là bằng chứng duy nhất cho hai điều dưới đây.

Kết luận đầy đủ: [TEMPORAL_DESYNC_PHASE0_SMOKE](../../docs/reports/TEMPORAL_DESYNC_PHASE0_SMOKE.md)

## Có gì

| Vòng | mp4 | Dung lượng | Nhãn ở `05_labels/` | Dòng |
|---|---|---|---|---|
| `phase0_smoke` | 30 | 30 M | `labels_v2_phase0_smoke.csv` | 12 |
| `phase0_smoke_v2r2` | 36 | 32 M | `labels_phase0_smoke_v2r2.csv` | 30 |
| `phase0_smoke_v2r3` | 36 | 31 M | `labels_phase0_smoke_v2r3.csv` | 30 |
| `phase0_smoke_v2r4` | 36 | 31 M | `labels_phase0_smoke_v2r4.csv` | 30 |
| `phase0_policy_smoke_v2r5` | 30 | 52 M | **không có** — xem dưới | — |
| `phase0_stratified_smoke_v2r6` | 135 | 203 M | `labels_phase0_stratified_smoke_v2r6.csv` | 75 |

Đánh số `r2 → r6` là các lần chạy lại sau mỗi lần sửa generator. Bốn vòng đầu chỉ dùng
một source video (`y3vBfCdY2tQ`); tới r6 mới mở ra 15 source.

## Hai artifact có giá trị nhất

**`phase0_stratified_smoke_v2r6/metadata_gate.json`** — cổng chống shortcut. Huấn luyện
logistic + random forest **chỉ** trên metadata container (kích thước file, bitrate, số
frame, sample rate…), CV `GroupKFold(group=source_clip)`. AUC cao nhất **0,546** so với
ngưỡng 0,65 → `passed: true`. Nghĩa là fake V2 không còn lộ dấu vết ở tầng đóng gói —
đúng lớp lỗi đã khiến pilot V1 bị NO-GO.

**`phase0_smoke/real_integration_audit.json`** — `lag_error_ms = 0.0` ở cả bốn mức dịch,
`aligned_corr > 0,9999999`. Chứng minh phần xoay vòng audio theo sample thật đã đúng,
khác V1 nơi hướng dương chỉ đổi timestamp và độ lệch biến mất khi Stage 04 decode.

**`phase0_policy_smoke_v2r5` cố ý không có file nhãn.** Vòng này dùng lại ba media V1 để
thử gate; Stage 05 phát hiện 12/24 fake lệch audio target, 17/24 lệch visual contract và
**dừng với exit 1** trước khi ghi labels. Thiếu file nhãn ở đây là *kết quả đúng*, không
phải chạy dở.

## Đường dẫn đã được viết lại

Manifest gốc do một agent khác sinh ra trong sandbox, nên trỏ vào
`C:\Users\CodexSandboxOffline\.codex\.sandbox\cwd\d3718c48404e7c63\...`. Chúng chạy được
**chỉ vì máy này có junction** trỏ ngược về repo — đã xác nhận cùng `(dev, ino)`, đúng
một file. Trên máy khác, hoặc khi junction mất, là gãy toàn bộ.

Đã viết lại **612 đường dẫn** trong 33 CSV, ba nhóm với ba đích khác nhau:

| Nhóm | Số dòng | Đích |
|---|---|---|
| Tiền tố sandbox | 414 | → vị trí archive này |
| Artifact phase0 dùng tiền tố repo | 144 | → vị trí archive này |
| Fake V1 phẳng ở gốc `data/03_fake/` | 54 | → `archive/pilot_v1/03_fake/` |
| `data/01_collect/cut_clips` | 15 | **giữ nguyên** — clip nguồn không di chuyển |

Nhóm 54 dòng **đang gãy sẵn** trước lượt này: ba file `fake_all.csv` của r2/r3/r4 trộn
output temporal_v2 mới với fake V1 cũ, và đợt lưu trữ V1 đã dời media đi mà không sửa
các CSV nằm lại trong `data/`. Nay đã sửa.

Kiểm chứng **toàn bộ, không lấy mẫu**: 627/627 đường dẫn trỏ tới file có thật.

**Lưu ý:** đường dẫn vẫn là **tuyệt đối** (`E:\FPTU\PRJ\...`). Trên máy khác phải viết
lại lần nữa.

## Hash không còn khớp bảng trong report

Việc viết lại đường dẫn làm đổi nội dung 29/33 CSV → hash trong mục 5.4 của report không
còn khớp. Bảng cũ **được giữ nguyên** làm bản khóa lịch sử; bảng hash sau khi viết lại
nằm ở mục 5.5 của cùng report.

Bốn artifact **không đổi** vì không chứa đường dẫn: ba file JSON và
`metadata_predictions.csv`. Hash của chúng vẫn khớp bảng gốc — tức bằng chứng số liệu
(gate AUC, lag error) chưa hề bị đụng tới.

## Có được xóa không

**Chưa nên.** Khác `archive/pilot_v1/`, chỗ này chỉ 376 MB và vẫn là bằng chứng hiện
hành cho việc generator V2 đạt yêu cầu. Sau khi repaired pilot chạy xong và có gate mới
trên dữ liệu đã lọc tay, có thể xét lại — khi đó vẫn nên giữ hai file JSON.

## Phụ lục — SHA-256 sau khi viết lại đường dẫn

Chốt ngày 2026-07-29, checkpoint `fc3e75a`. Đường dẫn tương đối tính từ thư mục này.
Ba mục **in đậm** là bốn artifact không chứa đường dẫn nên hash vẫn khớp bảng gốc ở
mục 5.4 của report.

| Artifact | SHA-256 |
|---|---|
| `03_fake/phase0_smoke/snvsm/fake_snvsm.csv` | `0A598ABFC0BA1B525E4B637F4E20F0BD92E814B2B61DA8B22AE1588FF6665B19` |
| `03_fake/phase0_smoke/snvsm/real_snvsm.csv` | `52C038D3326655FA4DA06F0809A319C2A32B9C2B28A2E4F3400B23BDDAB8303D` |
| `03_fake/phase0_smoke/snvsm_v2/fake_snvsm.csv` | `F8D822D839F2A2897581359CAD16C82326727C989FA041722A206C8EF67AD2D1` |
| `03_fake/phase0_smoke/snvsm_v2/real_snvsm.csv` | `61D54F8505573862A0F3ADDB44B2F817E9EF8A88ABE5A54B664C32E6D74BC8E5` |
| `03_fake/phase0_smoke/temporal_v2_labels.csv` | `E5298BC36BC30B558D33099A7832694DE2C89F7C05A08E36BF3C93033FD93020` |
| **`03_fake/phase0_smoke/real_integration_audit.json`** | `912A344FF354F46844A38D2BA12B8BD3900351C19A0B6D42EAFDEA615AA509DA` |
| `03_fake/phase0_smoke_v2r2/fake_all.csv` | `AF49D01D415151D0D269B6C5A57666B64E1A81B02E211040838CB775936CD365` |
| `03_fake/phase0_smoke_v2r2/snvsm/fake_snvsm.csv` | `D9EF8B48556F9A695558E4F343816B91EB4259426A307DE0598553F6C3D17AA1` |
| `03_fake/phase0_smoke_v2r2/snvsm/real_snvsm.csv` | `26DDCAF701107BC9D96604D91AED4F1A62CAC71C816E8F9DF7DB53252A28F49B` |
| `03_fake/phase0_smoke_v2r2/temporal_v2.csv` | `82AEF5211E81E799AD1A2F17175B98F9E5B7FD9C6C6B4E1D92FEAE4977B90843` |
| **`03_fake/phase0_smoke_v2r2/real_integration_audit.json`** | `912A344FF354F46844A38D2BA12B8BD3900351C19A0B6D42EAFDEA615AA509DA` |
| `03_fake/phase0_smoke_v2r3/fake_all.csv` | `928D374BC3A2DBB4711F9DB6C345ADCB314239B9CC564C898C14BDD917AE158B` |
| `03_fake/phase0_smoke_v2r3/snvsm/fake_snvsm.csv` | `0D21C640B0D613FFB92E115BCECE112DA47668864C6E199CF1157C6A413A7297` |
| `03_fake/phase0_smoke_v2r3/snvsm/real_snvsm.csv` | `D5EF22EE1694B0F634BDD8247321377A9540E2E084ABC12C49686F4A80D68D1F` |
| `03_fake/phase0_smoke_v2r3/temporal_v2.csv` | `31B449284983AD2FEA1F7609762DD886699C8187A1DCFF7BAF0C345C91B0EAFD` |
| `03_fake/phase0_smoke_v2r4/fake_all.csv` | `48BFD863DB9D93E022BA09A7CA191560026CF8E7F0E86374A1CB190AF4562C41` |
| `03_fake/phase0_smoke_v2r4/snvsm/fake_snvsm.csv` | `9D9AB0E83EE282E4485DD9963851B18BDA71E37E3CD78F3D6A69D25CD283E75E` |
| `03_fake/phase0_smoke_v2r4/snvsm/real_snvsm.csv` | `7D72C346D161DEB8C3FEA7AE64C794A5ACFE98C62EC8F83766389E1CFA917628` |
| `03_fake/phase0_smoke_v2r4/temporal_v2.csv` | `16DC7E667AE6E0B892A8ADDDFBD9E875C7A1D06C9D006ACDA9A001BB5F315E03` |
| `03_fake/phase0_policy_smoke_v2r5/snvsm/fake_snvsm.csv` | `8B1050D5BDAF36122818262E426174B7956E1D9A247E1AC29E1F239B664D1885` |
| `03_fake/phase0_policy_smoke_v2r5/snvsm/real_snvsm.csv` | `744EEE79408517CF08D483D481CD5CFE387C105AD6E535C1854035FA3C81F6F8` |
| `03_fake/phase0_stratified_smoke_v2r6/anonymization.csv` | `4CC4B02EFF14F01B0E0110A054BA15C5C167DFB475E7C2D8944571D595E2F955` |
| `03_fake/phase0_stratified_smoke_v2r6/fake_all.csv` | `646F8CBEEBA850FB68DB8D7D7F89BB96B03371BBF1AFDB4AA253FBC84080C78B` |
| `03_fake/phase0_stratified_smoke_v2r6/fake_snvsm.csv` | `1679BB6965BE0F625EACF0192B40740F5303D178E1BA3BBB17F35302E21A07DB` |
| `03_fake/phase0_stratified_smoke_v2r6/frame_reverse.csv` | `EF7C9427CEFC6190C892FD0DB854810780AB129D924D780A6656180835B8F120` |
| `03_fake/phase0_stratified_smoke_v2r6/input_real.csv` | `7EC2E3F0C34227C77DFC89A74D1FF75C0C5783488B98CDE5A021870BA8F37E9A` |
| `03_fake/phase0_stratified_smoke_v2r6/pitch_flatten.csv` | `C05CA24E0CB6AE1C8A6F0FD49C481ABB8DA62381A0DA152F5F7E86C42AC83991` |
| `03_fake/phase0_stratified_smoke_v2r6/real_snvsm.csv` | `B063FEB716561204BD3848E26E1F08E2D0D74404EF61F82E731099CD41E199A9` |
| `03_fake/phase0_stratified_smoke_v2r6/temporal_desync.csv` | `984257820003B19AED14394551AC5F0A92EEF864894F9CD6BBE54EE1AC8E9CBE` |
| **`03_fake/phase0_stratified_smoke_v2r6/metadata_gate.json`** | `43CD4C065D17C2F5F2A34815760CFF93042FD8B25971F0D4D7FE3E3911D3BA28` |
| **`03_fake/phase0_stratified_smoke_v2r6/metadata_predictions.csv`** | `4960C85F0E06641B30A3E5D5105DD468805A7481BA5895FE97A478ED5E958FDA` |
| `05_labels/labels_v2_phase0_smoke.csv` | `510EB1ABC6DD47BE45895D720E0FBB3246D71F3BC4E61ECC81C3B9FC89357B12` |
| `05_labels/labels_phase0_smoke_v2r2.csv` | `692C9D3B2C128D0E42E77DF0339056426F313BA317231CCB74B3B0FA4FC82995` |
| `05_labels/labels_phase0_smoke_v2r3.csv` | `435FB6F754B2232D18481F2C6D8C8F0560CFF406C7CCD7D7D170B4E406A1442A` |
| `05_labels/labels_phase0_smoke_v2r4.csv` | `6BA9E6BF5B9A42FC8A7177402E0609AD7B4CEBE0615A4754AF41D3ECE283E8A8` |
| `05_labels/labels_phase0_stratified_smoke_v2r6.csv` | `691002C350DB63FCB08AD8FF40EBB3180E832B9A201F016D7743A69052AD6D5B` |

**Một điểm không kiểm chứng được:** `stratified_smoke_v2r6/input_real.csv` có hash lệch
bảng gốc dù script báo *sửa 0 dòng* (15 đường dẫn của nó trỏ vào `cut_clips`, không đổi).
Vòng đọc/ghi CSV là trung thực — `metadata_predictions.csv` đi qua đúng vòng đó và hash
vẫn khớp — nên nhiều khả năng mục này đã lệch từ trước. Nhưng script ghi đè cả file
không có thay đổi và không giữ bản sao, **nên không còn cách chứng minh byte-for-byte.**
Nội dung vẫn đúng về mặt ngữ nghĩa: 15 dòng, 15 đường dẫn `cut_clips` đều tồn tại.
