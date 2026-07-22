# Báo cáo Phase 0 — sửa và smoke-test `temporal_desync`

**Ngày kiểm tra:** 2026-07-21
**Git base khi kiểm tra:** `73e6b95` (worktree có thay đổi Phase 0 chưa commit)
**Trạng thái:** cơ chế generator/SNVSM V2, guard manifest và smoke-test đã đạt; **chưa normalize/extract repaired pilot 2.700 clip, chưa train V2a**

## 1. Mục tiêu

Phase 0 xử lý lỗi blocking của dữ liệu temporal V1:

- hướng dương trước đây chỉ đổi timestamp audio; khi Stage 04 decode WAV, độ lệch gần như biến mất;
- hướng âm bị `-shortest` cắt video;
- hai hướng có shortcut khác nhau về khoảng trống audio, duration và số frame;
- SNVSM trước đây thường copy audio, nên codec audio chưa được chuẩn hóa thật sự.

Không thay đổi hoặc ghi đè bất kỳ artifact pilot V1 nào.

## 2. Cơ chế sửa

### 2.1 Dịch audio theo sample thật

[`01_temporal_desync.py`](../../src/pipeline/03_fake/01_temporal_desync.py) không còn dùng `-itsoffset` hoặc `-shortest`.

Audio được xoay vòng theo số sample tính từ FPS hữu tỉ chính xác:

```text
shift_sec     = shift_frames / exact_fps
shift_samples = round(abs(shift_sec) * audio_sample_rate)
```

- shift dương: đưa đoạn cuối audio lên đầu;
- shift âm: đưa đoạn đầu audio xuống cuối;
- video dùng stream-copy;
- audio trung gian dùng ALAC lossless;
- audio bắt đầu tại timestamp 0;
- encode qua `.part.mp4`, kiểm tra codec/audio sample cùng video packet-count/time-base/duration-tick so với source rồi atomic-replace; resume sửa file corrupt mà không nhân đôi manifest;
- video packet/frame và audio timeline khai báo được giữ nguyên; decode kiểu Stage 04 cho phép sai khác tối đa một AAC-frame tương đương do source AAC/resample.

Xoay vòng không chèn silence và không cắt dữ liệu, nhưng tạo một điểm nối wrap. Vì không thể dịch toàn bộ audio hữu hạn mà đồng thời giữ nguyên toàn bộ video/duration và không có vùng biên, generator ghi rõ:

```text
boundary=circular_wrap
audio_valid_start=<giây>
audio_valid_end=<giây>
visual_valid_start=<giây>
visual_valid_end=<giây>
generator=temporal_v2_circular_avmask_v1
```

trong `param`. `audio_valid_*` là miền hợp lệ trên audio output; `visual_valid_*` là miền visual tương ứng sau bù lag. V2a phải loại wrap khỏi attention, lag feature, temporal head, pooling và clip evidence, không chỉ một loss. Để mask length/side không thành shortcut nhãn hoặc dấu shift, phải dùng cửa sổ đồng bộ độ dài cố định trong miền hợp lệ hoặc áp edge mask/crop đối xứng cho real và mọi fake.

ID mới dùng hậu tố `desyncv2r2p...` hoặc `desyncv2r2m...`. Mặc định media nằm ở `data/03_fake/temporal_v2/`, manifest ở `data/03_fake/manifests/v2/temporal_desync.csv`; code từ chối append nếu manifest có temporal V1/schema cũ. Resume kiểm tra schema, ID duy nhất và probe media ALAC hoàn chỉnh thay vì chỉ tin file khác rỗng.

### 2.2 Chuẩn hóa audio trong SNVSM

[`05_snvsm_compress.py`](../../src/pipeline/03_fake/05_snvsm_compress.py) hiện luôn encode:

```text
video: H.264 theo cùng CRF/preset
audio: AAC 128 kbps, 16 kHz mono, timestamp bắt đầu từ 0
```

cho cả real và fake. Nhánh “audio copy nếu được” đã bỏ vì nó giữ dấu vân codec phụ thuộc method. Cách này kiểm soát/giảm shortcut codec; chưa có bằng chứng rằng nó xóa mọi forensic trace từ lịch sử transcode trước đó.

SNVSM V2 dùng ID `<clip_id>_snvsmv2_<config-id>_crf<N>`. Config hash hiện bao gồm encoder/preset/audio cùng toàn bộ CRF policy (`crf_set`, `mode`, `seed`); manifest giữ các field này, `snvsm_pair_key`, sample target và visual contract (`frames`, nominal FPS, duration). Stage 04/05 tự tính lại hash từ provenance thay vì chỉ tin chuỗi `snvsm_config_id`. Ở mode random, CRF được chọn từ khóa real nguồn (`source_clip`) nên real và cả bốn fake ghép cặp nhận cùng mức nén, thay vì random độc lập theo fake ID.

Audio được resample 16 kHz mono và timeline container được trim/pad về target của input. Do AAC mã hóa theo frame, decode PCM vẫn có thể lộ trailing padding dù `duration_ts` đúng; vì vậy Stage 04 bắt buộc trim waveform theo `snvsm_target_samples` trước prosody/Wav2Vec. Resume probe H.264 + AAC + format + container target, decode để bảo đảm có ít nhất target PCM, đồng thời đếm frame và đối chiếu nominal FPS/duration video với input. Encode đi qua file `.part.mp4` rồi atomic-replace; nếu bất kỳ row skip/fail thì CLI trả exit khác 0. Cây `data/03_fake/snvsm/` V1 được guard không cho V2 ghi vào.

Stage 04 từ chối fake manifest rỗng theo mặc định và chỉ resume `.pt` khớp clip/label/method/source, exact dtype/kích thước tensor, PCM, SNVSM config cùng `feature_config_id` (hash schema, tham số extraction và SHA-256 YOLO weights). File thiếu/sai contract bị thay atomic qua `.part`, và job trả exit khác 0 nếu còn bất kỳ clip fail. Dataloader cũng fail nếu labels thiếu feature, sai identity/kích thước hoặc thiếu nhánh đang bật. Stage 05 mặc định kiểm file tồn tại, từ chối fake manifest rỗng, fake mồ côi, thiếu media hoặc mode random có nhiều hơn một CRF/source-method; labels chỉ được publish atomic sau toàn bộ gate và mặc định không ghi đè output cũ.

### 2.3 Master composition không còn temporal V1

[`06_build_fake_manifest_v2.py`](../../src/pipeline/03_fake/06_build_fake_manifest_v2.py) đọc manifest V1 nhưng loại toàn bộ `temporal_desync` cũ, rồi ghép đúng một temporal V2 với ba method không-temporal cho mỗi source. Builder fail khi thiếu/trùng method, trùng ID, metadata lineage lệch, media mất hoặc output đè input; output được ghi atomic tại `data/03_fake/manifests/v2/fake_all.csv`.

## 3. Test tự động

File test: [`tests/test_temporal_desync.py`](../../tests/test_temporal_desync.py).

Lệnh:

```powershell
D:\Anaconda\envs\vn_av_df\python.exe -m unittest tests.test_temporal_desync -v
```

Matrix synthetic đã chạy:

- 5 dạng FPS: `24/1`, `25/1`, `2997/100`, `30000/1001`, `30/1`;
- 6 độ lệch: `-15`, `-7`, `-3`, `+3`, `+7`, `+15` frame;
- tổng cộng 30 trường hợp temporal;
- thêm test SNVSM real/fake qua đúng decode Stage 04, input audio 44,1 kHz stereo vs 16 kHz mono, guard V1, resume-idempotency và builder master V2.

Kết quả cuối: `10` test pass, `1` real-data test skip mặc định (`11` test được discover). Test real được bật riêng bằng biến môi trường và cũng đã pass ở mục 4.

Các invariant được kiểm tra:

- chuỗi packet H.264 và hash SHA-256 giống nhau;
- `framemd5` và số frame decode giống nhau;
- time base video giữ nguyên, duration sai khác tối đa một tick khi remux VFR;
- audio start tại 0 và giữ sample rate/số kênh;
- Stage-04-style WAV decode đo đúng dấu và độ lớn lag;
- không sinh digital silence ở đầu/cuối fixture;
- smoke không thấy single-sample jump lớn tại seam theo metric đạo hàm đã định; kiểm tra này không loại trừ seam đa-sample/phổ có thể học được;
- sau SNVSM, real/fake cùng H.264 + AAC 16 kHz mono và signed lag vẫn đúng qua decode Stage 04;
- Stage 04 từ chối manifest SNVSM thiếu/invalid `snvsm_target_samples` và trim PCM đã decode về đúng sample count;
- Stage 04 không reuse `.pt` sai identity/source/config, thiếu mouth/prosody/W2V theo mode, sai exact shape/dtype hoặc sample contract; có `--no_skip_existing` để buộc extract lại, `.pt` được publish atomic và job fail nếu fake rỗng/còn clip lỗi;
- dataloader từ chối labels thiếu `.pt` hoặc feature thiếu nhánh đang bật thay vì drop sample/điền zero âm thầm;
- SNVSM reject output bị cắt video dù audio còn đủ, config hash đổi khi CRF-set/mode/seed đổi, và Stage 05 reject CRF khác real;
- manifest V2 không trộn temporal V1, resume không tạo dòng trùng.
- temporal output corrupt bị regenerate qua file tạm; validator không chấp nhận output mất/thêm video packet dù duration vẫn gần đúng.

## 4. Integration smoke trên dữ liệu thật

Lệnh:

```powershell
$env:RUN_REAL_AV_AUDIT='1'
$env:REAL_AV_AUDIT_REPORT='data/03_fake/phase0_smoke_v2r2/real_integration_audit.json'
D:\Anaconda\envs\vn_av_df\python.exe -m unittest `
  tests.test_temporal_desync.TemporalDesyncContractTest.test_real_data_smoke -v
```

Cỡ mẫu đo được:

- 3 clip thật: `y3vBfCdY2tQ_clip0004_t00135` (tier1), `mHTLamr-Zqk_clip0081_t00799` (tier2), `7642303380873956616_clip0001_t00032` (tier3);
- FPS `25`, `30`, `29,97`;
- mỗi clip chạy đủ 6 độ lệch;
- tổng cộng 18 cặp real/fake.

Kết quả đo được:

| Kiểm tra | Kết quả |
|---|---:|
| Trường hợp đạt | 18/18 |
| Sai số lag lớn nhất | 0,0625 ms |
| Tương quan sau căn chỉnh nhỏ nhất | 0,990799 |
| Chênh decoded sample lớn nhất | 215 sample @16 kHz |
| Tỷ lệ seam-jump lớn nhất so với percentile 99,9% | 0,295240 |
| Cặp mất/thêm frame | 0/18 |
| Cặp khác packet/frame content | 0/18 |

Chênh decoded sample tối đa 215 nhỏ hơn một AAC frame tương đương khi resample về 16 kHz. Nó không phụ thuộc dấu hoặc độ lớn shift. Các số đo seam chỉ là smoke metric, không chứng minh circular-wrap vô hình với model.

## 5. CLI pipeline smoke r4 tách riêng

Generator:

```powershell
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/03_fake/01_temporal_desync.py `
  --input_csv data/02_curate/all_clean.csv `
  --out_dir data/03_fake/phase0_smoke_v2r4/generator `
  --labels data/03_fake/phase0_smoke_v2r4/temporal_v2.csv `
  --limit 6 --seed 42
```

Kết quả: `6/6` tạo thành công, `0` skip, `0` fail; ID dùng `desyncv2r2...`, đủ bốn trường valid-range và generator version mới.

Builder hiện tại:

```powershell
D:\Anaconda\envs\vn_av_df\python.exe src/pipeline/03_fake/06_build_fake_manifest_v2.py `
  --legacy_labels data/03_fake/labels.csv `
  --temporal_v2 data/03_fake/phase0_smoke_v2r4/temporal_v2.csv `
  --out data/03_fake/phase0_smoke_v2r4/fake_all.csv
```

Kết quả: `24` fake từ 6 source; mỗi method `6`, không có temporal V1, không ID trùng và tất cả media tồn tại.

SNVSM chạy trên 6 real + 24 fake vào `data/03_fake/phase0_smoke_v2r4/snvsm/`. Kết quả audit media/container 30/30:

- video `h264`;
- audio `aac`, 16 kHz mono duy nhất cho cả bốn method và real;
- audio start `0`;
- `snvsm_version=snvsm_v2_h264_aac16k_mono_exactdur`;
- cùng `snvsm_config_id=2ae4ac96b5`, `libx264/ultrafast`, `aac_128k_16khz_mono`;
- `0/30` vi phạm target sample count ở **container timeline**;
- không có file encode fail.

Audit bổ sung phát hiện raw AAC decode có trailing padding ở `29/30` file, dư `64–1008` sample tùy clip/method. Đây là lý do không được diễn giải `duration_ts` đúng thành PCM decode-exact. Sau khi Stage 04 trim theo manifest, `30/30` waveform có độ dài đúng `snvsm_target_samples`; không dòng SNVSM nào thiếu/invalid target. Đo lại sáu cặp temporal cho kết quả lag error lớn nhất `0 ms`, valid-range correlation nhỏ nhất `0,996682`, chênh decoded length giữa real/fake lớn nhất `0` sample.

Stage 05 ban đầu tạo `data/05_labels/labels_phase0_smoke_v2r4.csv` gồm 6 real + 24 fake, gate config SNVSM giống nhau và giữ provenance trên cả 30 dòng. Verify không có `speaker_id`/`source_video` xuyên split; sáu real cùng component nên toàn bộ 30 dòng ở train và đây chỉ là wiring smoke. Artifact labels r4 được giữ làm lịch sử trước các gate mới. Code hiện tại sẽ reject manifest r4 ngay vì thiếu CRF/visual-policy provenance; audit trực tiếp còn cho thấy 12/24 fake (`frame_reverse` và `pitch_flatten`) có target khác real. Không dùng artifact này cho pilot.

Audit này đồng thời phát hiện blocker kế tiếp trong **ba generator V1 không-temporal**: trên đúng 6 source smoke, video `frame_reverse` và `anonymization` đều ngắn hơn real ghép cặp 40 ms; `pitch_flatten` lệch 0–160 ms. Code V1 của cả ba còn dùng `-shortest`. Đây là số đo 6 source, chưa được ngoại suy thành tỷ lệ toàn bộ 3.001 source. SNVSM chuẩn hóa audio format nhưng không thể khôi phục frame video đã bị cắt; phải audit/repair contract frame-duration của ba method hoặc chứng minh loader fixed-window đối xứng loại hoàn toàn shortcut này trước repaired pilot.

### 5.1 Policy/visual-contract smoke r5 bằng code hiện tại

Một smoke tách biệt normalize lại cùng 6 real + 24 fake r4 vào `data/03_fake/phase0_policy_smoke_v2r5/snvsm/` để kiểm đúng code hiện tại. Kết quả cuối:

- 6 real + 24 fake, đủ `6` clip cho mỗi method, không skip/fail;
- một config duy nhất `46c7f60176`, mode `random`, CRF-set `23,30,35,40`, seed `42`;
- `30/30` dòng qua semantic contract, gồm cả config hash được tính lại;
- `0/24` fake có CRF khác real nguồn;
- `30/30` media qua codec + decoded-audio + frame/FPS/duration validator;
- `30/30` waveform Stage 04 đúng target sau trim;
- test riêng video chỉ còn khoảng 1 giây nhưng giữ full audio bị validator reject.

Smoke này cố ý dùng lại ba media V1 để kiểm gate, **không phải repaired data**. Paired audit tìm thấy `12/24` fake lệch audio target (toàn bộ reverse + pitch) và `17/24` lệch visual contract (6 reverse + 5 pitch + 6 anonymization). Vì vậy lệnh Stage 05 trên r5 dừng với exit `1` trước khi ghi labels; `data/05_labels/labels_phase0_policy_smoke_v2r5.csv` không tồn tại. Đây là hành vi đúng: partial/method-drop, CRF lệch, audio lệch hoặc video lệch không thể âm thầm đi vào pilot.

### 5.2 Audit timing phân tầng cho ba method không-temporal

Để kiểm tra lỗi trên có chỉ nằm ở sáu clip tier1 hay không, audit read-only lấy đúng 5 source mỗi tier sau khi sort `clip_id`, tại các vị trí `0%, 25%, 50%, 75%, 100%`. Tổng cộng: 15 real và 45 fake ghép cặp. `ffprobe -count_frames` đo số frame cùng duration video/audio/format của từng cặp.

| Method | Cặp lệch số frame | Sai lệch video fake so với real | Sai lệch audio fake so với real |
|---|---:|---:|---:|
| `frame_reverse` | 6/15 | cả 15 ngắn hơn `20–40 ms` | cả 15 ngắn hơn `12–54 ms` |
| `pitch_flatten` | 8/15, mất tối đa 5 frame | `−166,9–0 ms` | `−60–+23 ms` |
| `anonymization` | 14/15, mất tối đa 3 frame | `−120–0 ms` | `0 ms` |

Exact source selection để tái lập:

| Tier | `clip_id` |
|---|---|
| tier1 | `-jDBwP6g2tA_clip0002_t00009` |
| tier1 | `EMFwAo7a4rg_clip0021_t00402` |
| tier1 | `XwMwDTPAyAo_clip0039_t00629` |
| tier1 | `jfQzsbl5zZc_clip0053_t00303` |
| tier1 | `ztblIi_pMPE_clip0003_t00213` |
| tier2 | `0PsGwmgAI-k_clip0001_t00039` |
| tier2 | `FufWLL1IIrQ_clip0048_t00281` |
| tier2 | `UnodrLYWJYI_clip0033_t00368` |
| tier2 | `mHTLamr-Zqk_clip0009_t00132` |
| tier2 | `wKaFsRX95YY_clip0049_t00569` |
| tier3 | `7204326198501969178_clip0000_t00000` |
| tier3 | `7583660080017444104_clip0007_t00062` |
| tier3 | `7634057975606365461_clip0001_t00052` |
| tier3 | `7638262561955269889_clip0001_t00004` |
| tier3 | `7644036486509153554_clip0003_t00050` |

Đây là mẫu phân tầng xác nhận lỗi tồn tại ở cả ba nguồn dữ liệu, **không phải ước lượng tỷ lệ lỗi trên 3.001 clip**. Cơ chế cũng khớp code: cả ba generator V1 vẫn dùng `-shortest`. Vì vậy reuse media V1 cho repaired pilot là NO-GO; SNVSM chỉ giữ/normalize timeline hiện có và không thể phục hồi frame đã mất.

### 5.3 Provenance của snapshot smoke

Raw media/manifests dưới `data/` bị gitignore theo quy tắc repo. Bảng dưới tách rõ: r4 là artifact lịch sử trước hardening; source/test và hai manifest policy-smoke r5 phản ánh code/contract hiện tại. Vì vậy `labels_phase0_smoke_v2r4.csv` không tái tạo được bằng code hiện tại nếu vẫn dùng ba media V1 lỗi, còn r5 chủ động không có labels vì paired gate đã dừng đúng.

Các CSV smoke chứa absolute path của workspace lúc chạy và chỉ là bằng chứng cục bộ; không dùng chúng làm input cho repaired pilot.

| Artifact | SHA-256 |
|---|---|
| `01_temporal_desync.py` | `C852D6BCF195BB9F428CFBF9810D789546347BE10FAC49BB8B47F6048CEBBEB3` |
| `05_snvsm_compress.py` | `906EFBCECCAEB5EA99377C48791F9B0DC30EE55C60BB7663B9199F4E908A7A27` |
| `06_build_fake_manifest_v2.py` | `1ABFC763D05B5E60E3956DDA19A48312087330F153C5FCBAC3D42DDDE47E40AF` |
| `04_extract_features/01_extract_features.py` | `51D85B009B0BEF424F5B9A9FFD5202B614B2BF2FFC73F04273151665F2E64705` |
| `01_build_labels.py` | `2B975BAF5C18ED2E49AD32B635948CD03423CF551D63AB6946FF3E670BFCDEED` |
| `train/dataset.py` | `8E1BCCEA1F108E8771C214F68AA113DDA7ABB3A410923D96FC66F6455B9756DF` |
| `tests/test_temporal_desync.py` | `2A915D8F40A6A8CE934C7DB981AE374FD73286190DAAB9A47FD60362C939D1ED` |
| `real_integration_audit.json` | `912A344FF354F46844A38D2BA12B8BD3900351C19A0B6D42EAFDEA615AA509DA` |
| `temporal_v2.csv` | `C35965A0C628F77065AAF6EC8A24181EB419DD7E4A44C2DD1C4BA919E4C4307F` |
| `fake_all.csv` | `B493E9D2DCFC8BF53438E812638AC39A69C0E268465C2EADEA58EE1F164B8A68` |
| `real_snvsm.csv` | `0F0C74938DB8983DBEA5D9DF8F1B52F317399A9BFBBAD21E916B895E549CA9E7` |
| `fake_snvsm.csv` | `572FB69FA37431CA1FEC1BB3E51AE36CF68A8C2199AA7C50A32DC4071679DDA3` |
| `labels_phase0_smoke_v2r4.csv` | `19A3D650A8DA1CEACAF06B5D706E8E6E6FAEDF88EC511FA9AA37B32F3A30A990` |
| `policy_smoke_v2r5/real_snvsm.csv` | `F11744A422CF00BD5C54CE36409998CE0ADF8DA76FCC7E086B0072E9AD6CE51E` |
| `policy_smoke_v2r5/fake_snvsm.csv` | `38DC581217073764D8282864DB1D9F1E42029BC6A4CCE58D1CA0DE6C24F579F9` |

## 6. Bảo toàn pilot V1

Đã đối chiếu lại toàn bộ 8 checksum trong:

```text
experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/manifest_hashes.json
```

Kết quả: 4 manifest/input và 4 artifact model vẫn khớp `8/8`. Không file V1 nào bị ghi đè.

## 7. Giới hạn và quyết định

Đã xác minh:

- lỗi sign-asymmetric do `-itsoffset/-shortest` được loại khỏi code V2;
- audio shift tồn tại thật sau cách decode của Stage 04;
- smoke không thấy shortcut silence, duration hoặc frame count;
- SNVSM mới chuẩn hóa cả video lẫn audio; Stage 04 loại trailing AAC padding theo sample-count contract trước feature extraction.
- Stage 05 fail-fast nếu fake rỗng/mồ côi, media thiếu, thiếu/thừa method, duplicate/partial manifest, CRF policy lệch, hoặc audio/video contract khác real; Stage 04 cũng trả lỗi nếu feature thiếu nhánh/clip. Composition chứa media V1 lỗi không thể lọt vào repaired pilot.

Chưa xác minh:

- 540 temporal clip của repaired pilot;
- SNVSM V2 và feature extraction mới trên đủ 2.700 clip repaired pilot;
- metadata-only baseline trên toàn pilot;
- audit/repair frame-duration của `frame_reverse`, `pitch_flatten`, `anonymization` V1;
- schema cột cấu trúc và V2a loader có thực sự dùng `audio_valid_*`/`visual_valid_*` đối xứng ở mọi nhánh;
- metric model sau repair.

Vì vậy trạng thái vẫn là **NO-GO full extraction/full training**. Bước tiếp theo là khóa structured schema/valid-range/mask V2a và test semantics trước; sau đó audit/sửa timing contract của ba method không-temporal rồi chạy metadata-shortcut gate trên smoke đã repair. Chỉ khi gate dữ liệu đạt mới tạo repaired pilot: regenerate các method bị ảnh hưởng, SNVSM lại 540 real + 2.160 fake, qua Stage 05 và metadata-only gate trên toàn labels, rồi mới extract 2.700 feature vào path versioned trước khi train.
