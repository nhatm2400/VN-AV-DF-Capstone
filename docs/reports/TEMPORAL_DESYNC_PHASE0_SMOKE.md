# Báo cáo Phase 0 — sửa và smoke-test `temporal_desync`

**Ngày kiểm tra:** 2026-07-21; cập nhật contract và stratified smoke ngày 2026-07-22

**Checkpoint Phase 0 cơ chế:** `1c592a4`; phần structured timeline là thay đổi của Bước 2 sau checkpoint này

**Trạng thái:** cơ chế của cả bốn generator V2, SNVSM, structured timeline, synthetic contract-test và stratified real-data smoke/metadata gate đã đạt; **chưa normalize/extract repaired pilot 2.700 clip, chưa train V2a**

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

Xoay vòng không chèn silence và không cắt dữ liệu, nhưng tạo một điểm nối wrap. Vì không thể dịch toàn bộ audio hữu hạn mà đồng thời giữ nguyên toàn bộ video/duration và không có vùng biên, generator ghi contract vào **các cột CSV có cấu trúc**, không nhét trong chuỗi `param`:

```text
timeline_schema_version=av_timeline_v1
timeline_mask_policy=fixed_common_window_v1
timeline_boundary=circular_wrap
timeline_duration_s=<giây>
audio_valid_start_s=<giây>
audio_valid_end_s=<giây>
visual_valid_start_s=<giây>
visual_valid_end_s=<giây>
manipulation_scope=global
manipulation_start_s=<giây>
manipulation_end_s=<giây>
```

`audio_valid_*` là miền hợp lệ trên audio output; `visual_valid_*` là miền visual tương ứng sau bù lag. `fixed_common_window_v1` chọn một cửa sổ đồng bộ có độ dài cố định nằm hoàn toàn trong giao của hai miền hợp lệ; như vậy số lượng/vị trí phần tử mask thô không được đưa thẳng cho model thành shortcut nhãn hay dấu shift. Module dùng chung [`timeline_contract.py`](../../src/pipeline/timeline_contract.py) kiểm version, boundary, miền hợp lệ, vùng thao tác và semantics theo method. V2a vẫn phải thực sự tiêu thụ contract này ở attention, lag feature, temporal head, pooling và clip evidence; Bước 2 mới khóa schema/parser/validator, chưa implement toàn bộ loader/model mask.

`param` giờ chỉ giữ mô tả con người đọc được như `shift=...;generator=...`; nguồn sự thật cho timeline là các cột cấu trúc. Mỗi feature `.pt` mới lưu `timeline_contract_id` cùng bản contract canonical để resume không tái sử dụng feature từ timeline khác.

ID mới dùng hậu tố `desyncv2r2p...` hoặc `desyncv2r2m...`. Mặc định media nằm ở `data/03_fake/temporal_v2/`, manifest ở `data/03_fake/manifests/v2/temporal_desync.csv`; code từ chối append nếu manifest có temporal V1/schema cũ. Resume kiểm tra schema, ID duy nhất và probe media ALAC hoàn chỉnh thay vì chỉ tin file khác rỗng.

### 2.2 Chuẩn hóa audio trong SNVSM

[`05_snvsm_compress.py`](../../src/pipeline/03_fake/05_snvsm_compress.py) hiện luôn encode:

```text
video: H.264 theo cùng CRF/preset
audio: AAC 128 kbps, 16 kHz mono, timestamp bắt đầu từ 0
```

cho cả real và fake. Nhánh “audio copy nếu được” đã bỏ vì nó giữ dấu vân codec phụ thuộc method. Cách này kiểm soát/giảm shortcut codec; chưa có bằng chứng rằng nó xóa mọi forensic trace từ lịch sử transcode trước đó.

SNVSM V2 dùng ID `<clip_id>_snvsmv2_<config-id>_crf<N>`. Config hash hiện bao gồm encoder/preset/audio, toàn bộ CRF policy (`crf_set`, `mode`, `seed`) và `timeline_schema_version`; manifest giữ các field này, `snvsm_pair_key`, sample target, visual contract (`frames`, nominal FPS, duration) cùng structured timeline. Real nhận contract `native`/`none` phủ toàn media; fake bắt buộc đã có contract hợp lệ và được đối chiếu với media probe. Stage 04/05 tự tính lại hash từ provenance thay vì chỉ tin chuỗi `snvsm_config_id`. Ở mode random, CRF được chọn từ khóa real nguồn (`source_clip`) nên real và cả bốn fake ghép cặp nhận cùng mức nén, thay vì random độc lập theo fake ID.

Audio được resample 16 kHz mono và timeline container được trim/pad về target của input. Do AAC mã hóa theo frame, decode PCM vẫn có thể lộ trailing padding dù `duration_ts` đúng; vì vậy Stage 04 bắt buộc trim waveform theo `snvsm_target_samples` trước prosody/Wav2Vec. Resume probe H.264 + AAC + format + container target, decode để bảo đảm có ít nhất target PCM, đồng thời đếm frame và đối chiếu nominal FPS/duration video với input. Encode đi qua file `.part.mp4` rồi atomic-replace; nếu bất kỳ row skip/fail thì CLI trả exit khác 0. Cây `data/03_fake/snvsm/` V1 được guard không cho V2 ghi vào.

Stage 04 từ chối fake manifest rỗng theo mặc định và chỉ resume `.pt` khớp clip/label/method/source, exact dtype/kích thước tensor, PCM, SNVSM config cùng `feature_config_id` (hash schema, tham số extraction và SHA-256 YOLO weights). File thiếu/sai contract bị thay atomic qua `.part`, và job trả exit khác 0 nếu còn bất kỳ clip fail. Dataloader cũng fail nếu labels thiếu feature, sai identity/kích thước hoặc thiếu nhánh đang bật. Stage 05 mặc định kiểm file tồn tại, từ chối fake manifest rỗng, fake mồ côi, thiếu media hoặc mode random có nhiều hơn một CRF/source-method; labels chỉ được publish atomic sau toàn bộ gate và mặc định không ghi đè output cũ.

### 2.3 Master composition không còn temporal V1

[`06_build_fake_manifest_v2.py`](../../src/pipeline/03_fake/06_build_fake_manifest_v2.py) chỉ ghép bốn manifest V2 riêng. Builder fail khi generator version sai, thiếu/trùng method, trùng ID, metadata lineage lệch, media mất, timeline contract thiếu/sai hoặc output đè input; output được ghi atomic tại `data/03_fake/manifests/v2/fake_all.csv`. Manifest V1 không còn là input của builder repaired.

### 2.4 Repair ba generator không-temporal

Code V2 của [`02_frame_reverse.py`](../../src/pipeline/03_fake/02_frame_reverse.py), [`03_pitch_flatten.py`](../../src/pipeline/03_fake/03_pitch_flatten.py) và [`04_anonymization.py`](../../src/pipeline/03_fake/04_anonymization.py) đã bỏ `-shortest`:

- `frame_reverse` chọn và đảo theo **chỉ số frame**, ép đúng FPS + tổng số frame source;
- `pitch_flatten` giữ video bằng stream-copy, mux audio ALAC 16 kHz mono đã trim/pad đúng sample target source;
- `anonymization` xử lý đủ toàn bộ video, ưu tiên audio stream-copy và fallback ALAC;
- cả ba ghi file `.part.mp4`, probe lại rồi chỉ atomic-replace khi số frame, FPS, video duration và audio target khớp real nguồn;
- cả ba dùng output/manifest V2 riêng, ID mới và ghi `av_timeline_v1`; resume chỉ chấp nhận đúng generator version + contract;
- utility dùng chung nằm tại [`fake_media_contract.py`](../../src/pipeline/fake_media_contract.py).

Phần code repair đã qua cả synthetic smoke lẫn Bước 4 stratified smoke thật ở mục 5.3. Mẫu 15 nguồn chỉ là gate trước pilot, không phải ước lượng tỷ lệ lỗi trên toàn bộ 3.001 clip.

## 3. Test tự động

File test: [`tests/test_temporal_desync.py`](../../tests/test_temporal_desync.py) và [`tests/test_non_temporal_generators.py`](../../tests/test_non_temporal_generators.py).

Lệnh:

```powershell
D:\Anaconda\envs\vn_av_df\python.exe -m unittest tests.test_temporal_desync -v
```

Matrix synthetic đã chạy:

- 5 dạng FPS: `24/1`, `25/1`, `2997/100`, `30000/1001`, `30/1`;
- 6 độ lệch: `-15`, `-7`, `-3`, `+3`, `+7`, `+15` frame;
- tổng cộng 30 trường hợp temporal;
- thêm test SNVSM real/fake qua đúng decode Stage 04, input audio 44,1 kHz stereo vs 16 kHz mono, guard V1, resume-idempotency và builder master V2.

Kết quả cuối sau Bước 4: `17` test pass, `1` real-data test skip mặc định (`18` test được discover). Test real temporal được bật riêng bằng biến môi trường và đã pass ở mục 4 tại checkpoint cơ chế.

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
- structured timeline đúng semantics cho real và đủ bốn method; fixed-common-window luôn nằm trong giao miền audio/visual hợp lệ và có độ dài cố định;
- SNVSM CLI tự thêm contract `native` hoàn chỉnh cho real, còn fake thiếu/sai contract bị fail-closed;
- Stage 04 gắn `timeline_contract_id` vào feature để chặn resume nhầm provenance.
- ba generator không-temporal giữ đúng paired media contract trên synthetic 25 FPS; `frame_reverse` còn được test thêm ở 30000/1001 FPS;
- builder chỉ nhận đúng generator version V2 của đủ bốn method và reject param V1.

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
  --input_csv data/02_curate/manifests/all_clean.csv `
  --out_dir data/03_fake/phase0_smoke_v2r4/generator `
  --labels data/03_fake/phase0_smoke_v2r4/temporal_v2.csv `
  --limit 6 --seed 42
```

Kết quả lịch sử: `6/6` tạo thành công, `0` skip, `0` fail; ID dùng `desyncv2r2...`. Manifest này có bốn valid-range sơ khai trong `param`; code Bước 2 hiện yêu cầu các cột `av_timeline_v1` nên không tái sử dụng trực tiếp artifact r4.

Builder tại thời điểm tạo smoke r4:

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

Stage 05 ban đầu tạo `data/05_labels/labels_phase0_smoke_v2r4.csv` gồm 6 real + 24 fake, gate config SNVSM giống nhau và giữ provenance trên cả 30 dòng. Verify không có `speaker_id`/`source_video` xuyên split; sáu real cùng component nên toàn bộ 30 dòng ở train và đây chỉ là wiring smoke. Artifact labels r4 được giữ làm lịch sử trước các gate mới. Code hiện tại reject manifest r4 vì thiếu CRF/visual-policy provenance và structured timeline; audit trực tiếp còn cho thấy 12/24 fake (`frame_reverse` và `pitch_flatten`) có target khác real. Không dùng artifact này cho pilot.

Audit này đồng thời phát hiện blocker kế tiếp trong **ba generator V1 không-temporal**: trên đúng 6 source smoke, video `frame_reverse` và `anonymization` đều ngắn hơn real ghép cặp 40 ms; `pitch_flatten` lệch 0–160 ms. Code V1 của cả ba còn dùng `-shortest`. Đây là số đo 6 source, chưa được ngoại suy thành tỷ lệ toàn bộ 3.001 source. SNVSM chuẩn hóa audio format nhưng không thể khôi phục frame video đã bị cắt; phải audit/repair contract frame-duration của ba method hoặc chứng minh loader fixed-window đối xứng loại hoàn toàn shortcut này trước repaired pilot.

### 5.1 Policy/visual-contract smoke r5 tại checkpoint `1c592a4`

Một smoke tách biệt đã normalize lại cùng 6 real + 24 fake r4 vào `data/03_fake/phase0_policy_smoke_v2r5/snvsm/` để kiểm code ở checkpoint cơ chế. Kết quả lúc đó:

- 6 real + 24 fake, đủ `6` clip cho mỗi method, không skip/fail;
- một config duy nhất `46c7f60176`, mode `random`, CRF-set `23,30,35,40`, seed `42`;
- `30/30` dòng qua semantic contract, gồm cả config hash được tính lại;
- `0/24` fake có CRF khác real nguồn;
- `30/30` media qua codec + decoded-audio + frame/FPS/duration validator;
- `30/30` waveform Stage 04 đúng target sau trim;
- test riêng video chỉ còn khoảng 1 giây nhưng giữ full audio bị validator reject.

Smoke này cố ý dùng lại ba media V1 để kiểm gate, **không phải repaired data**. Paired audit tìm thấy `12/24` fake lệch audio target (toàn bộ reverse + pitch) và `17/24` lệch visual contract (6 reverse + 5 pitch + 6 anonymization). Vì vậy lệnh Stage 05 trên r5 dừng với exit `1` trước khi ghi labels; `data/05_labels/labels_phase0_policy_smoke_v2r5.csv` không tồn tại. Đây là hành vi đúng: partial/method-drop, CRF lệch, audio lệch hoặc video lệch không thể âm thầm đi vào pilot.

Sau Bước 2, r5 là **artifact lịch sử**: các manifest này chưa có `av_timeline_v1` và config hash chưa bao gồm timeline schema. Stage 05 hiện tại reject chúng ngay ở provenance/structured-contract gate trước paired timing gate và vẫn không sinh output. Do đó các số `30/30` ở trên chỉ chứng minh validator media/PCM/CRF của checkpoint `1c592a4`, không phải tuyên bố rằng r5 tương thích với code hiện tại.

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

### 5.3 Stratified real-data smoke `v2r6` và metadata-shortcut gate

Đúng 15 source ở mục 5.2 được regenerate bằng toàn bộ code V2 hiện tại. Kết quả đo được:

- input: 15 real, gồm 5 source/tier;
- generator: `15/15` temporal, `15/15` frame reverse, `15/15` pitch flatten, `15/15` anonymization; raw paired media contract đạt `60/60`;
- SNVSM: `15/15` real + `60/60` fake; chỉ có một codec signature H.264/yuv420p + AAC/16 kHz/mono;
- Stage 05: `75/75` labels, gồm train 50, validation 15, test 10; không trùng `speaker_id` hoặc `source_video` qua split theo metadata;
- metadata gate dùng 18 thuộc tính quan sát được từ media/container, GroupKFold 5-fold theo `source_clip`; tuyệt đối không dùng method/path/ID/CRF làm feature.

| Model metadata-only | AUC | Group-bootstrap 95% CI | Kết luận với ngưỡng 0,65 |
|---|---:|---:|---|
| Logistic regression cân bằng | 0,530 | [0,503; 0,557] | Đạt |
| Random forest depth 3 cân bằng | 0,546 | [0,500; 0,614] | Đạt |

Method-wise AUC của logistic: anonymization 0,524; frame reverse 0,471; pitch flatten **0,649**; temporal 0,476. Random forest tương ứng: 0,544; 0,540; 0,607; 0,493. Max AUC tổng là `0,546 ≤ 0,65`, nên gate smoke **đạt**. Tuy nhiên pitch flatten sát ngưỡng và cỡ mẫu chỉ 15 source; đây là lý do metadata gate trên đủ repaired pilot vẫn là điều kiện bắt buộc.

Smoke còn phát hiện và sửa hai edge case dữ liệu thật mà fixture synthetic không cover:

- remux có thể làm `avg_frame_rate` thay đổi một tick dù cadence thật không đổi; validator chuyển sang ưu tiên `r_frame_rate`, trim audio concat đúng native sample và chỉ cho phép sai lệch một native sample khi target Stage 04 ở 16 kHz vẫn giống hệt;
- source tier3 `7204326198501969178_clip0000_t00000` có audio start 0,135 giây; frame reverse/anonymization dùng fallback ALAC lossless, trim đúng native sample và reset PTS. SNVSM chấp nhận sai lệch duration metadata trong tối đa một AAC frame nhưng vẫn yêu cầu decoded PCM đủ target chính xác.

Artifact cục bộ nằm tại `phase0_stratified_smoke_v2r6/` và `labels_phase0_stratified_smoke_v2r6.csv`; chúng chỉ dùng làm bằng chứng smoke, không tái dùng làm repaired pilot.

> **Vị trí artifact đã đổi (2026-07-29).** Toàn bộ sáu vòng smoke đã chuyển sang
> [`archive/phase0_v2_smoke/`](../../archive/phase0_v2_smoke/README.md). Các đường dẫn
> `data/03_fake/phase0_*` và `data/05_labels/labels_phase0_*` xuất hiện trong lệnh và mô
> tả ở trên được **giữ nguyên**: chúng ghi lại lệnh đã chạy lúc đó, không phải vị trí
> hiện tại. Đường dẫn *bên trong* các CSV thì đã được viết lại — xem mục 5.5.

### 5.4 Provenance của snapshot smoke

Bảng dưới khóa source/test cùng các artifact r4/r5 lịch sử và r6 hiện hành, **tại thời điểm chạy smoke (checkpoint `e49140f`)**. `labels_phase0_smoke_v2r4.csv` không tái tạo được bằng code hiện tại nếu vẫn dùng ba media V1 lỗi; r5 cũng không còn qua structured-contract gate và chủ động không có labels.

Các CSV smoke chứa absolute path của workspace lúc chạy và chỉ là bằng chứng cục bộ; không dùng chúng làm input cho repaired pilot.

**Bảng này là bản khóa lịch sử, cố ý không cập nhật.** Phần lớn hash đã lệch so với HEAD; mục 5.5 giải thích từng nguyên nhân.

| Artifact | SHA-256 |
|---|---|
| `fake_media_contract.py` | `8A14C9AC4316AAE9DEF352CD424177332BE2696919F34707E86A68F62F38000F` |
| `timeline_contract.py` | `D6C2D644A2DBDB39A7DB2AC82EA0C1190BDC2D9421DA0BC16071C07F624C04F3` |
| `01_temporal_desync.py` | `423D47BCE543DAEAEB89FAEACAB6A19136EA0F124C47F99B2F7BE75AE0DCB79E` |
| `02_frame_reverse.py` | `8603E944266189765A0B06ADD4BD3CFE2FEF3FFE4CA216F215C41BDB1C2CC7B2` |
| `03_pitch_flatten.py` | `3E1B3D87E0DFDD6579B5B60DECA01BC6190F69620CC808556988D0DFC1FBF165` |
| `04_anonymization.py` | `1B0BF53969CB14AF2FA8579ECD6C295BD8BB35FBE0D2E392B8F83CF834C7BB5A` |
| `05_snvsm_compress.py` | `D130720F006B8C408634654E0EF130DF750A91E189FEFF78102BDC3606ADF9B4` |
| `06_build_fake_manifest_v2.py` | `0BAA76BC3B25AB6D87658352C0E891C1BF5D7CD5F34ADB7666F58D3A908D82B3` |
| `07_metadata_shortcut_gate.py` | `200A01FE793E73435CDDBE7CCB500764860641181248C72C1FE54A81089CC419` |
| `04_extract_features/01_extract_features.py` | `49B5419146000C2A5DA8FFBA5103A33D0E6B59A7DFFE82812120716ECE16BD00` |
| `01_build_labels.py` | `F20942D33F0F9E2144B06448C7CC8226D10BFE81E87F60A2B3E7E7BEC8805DC7` |
| `train/dataset.py` | `8E1BCCEA1F108E8771C214F68AA113DDA7ABB3A410923D96FC66F6455B9756DF` |
| `tests/test_temporal_desync.py` | `E81AF770339F1A9A6E3D2012F12B38181204E0C12B06DEBFF3C981E344C961B6` |
| `tests/test_non_temporal_generators.py` | `CE500B35C01B00E77473D93CF8C9074792E08E1004C00DDC52476CFA50127FFF` |
| `real_integration_audit.json` | `912A344FF354F46844A38D2BA12B8BD3900351C19A0B6D42EAFDEA615AA509DA` |
| `temporal_v2.csv` | `C35965A0C628F77065AAF6EC8A24181EB419DD7E4A44C2DD1C4BA919E4C4307F` |
| `fake_all.csv` | `B493E9D2DCFC8BF53438E812638AC39A69C0E268465C2EADEA58EE1F164B8A68` |
| `real_snvsm.csv` | `0F0C74938DB8983DBEA5D9DF8F1B52F317399A9BFBBAD21E916B895E549CA9E7` |
| `fake_snvsm.csv` | `572FB69FA37431CA1FEC1BB3E51AE36CF68A8C2199AA7C50A32DC4071679DDA3` |
| `labels_phase0_smoke_v2r4.csv` | `19A3D650A8DA1CEACAF06B5D706E8E6E6FAEDF88EC511FA9AA37B32F3A30A990` |
| `policy_smoke_v2r5/real_snvsm.csv` | `F11744A422CF00BD5C54CE36409998CE0ADF8DA76FCC7E086B0072E9AD6CE51E` |
| `policy_smoke_v2r5/fake_snvsm.csv` | `38DC581217073764D8282864DB1D9F1E42029BC6A4CCE58D1CA0DE6C24F579F9` |
| `stratified_smoke_v2r6/input_real.csv` | `CFC8BEB2BF0E81D3863BD78105BCAF5BDBFCD81C1518B714B4A53F88AE722169` |
| `stratified_smoke_v2r6/temporal_desync.csv` | `F10A13D2AF51F8716E0392D90B9EE54FC924B769A3BCAC199C4FCADA75E44653` |
| `stratified_smoke_v2r6/frame_reverse.csv` | `8DE0126EA1411BA4D76529A4A6881340F62071F7253FC1425204FE30A6EFEB6A` |
| `stratified_smoke_v2r6/pitch_flatten.csv` | `64D5F77BFC7D77F4414791722FC43DBEF2C69EC5B76FADD6CE7E84C270647CA0` |
| `stratified_smoke_v2r6/anonymization.csv` | `EC004D1416C4CE436E89BA43E4C85CA2D9B920B73765E5EB753775375AEDC167` |
| `stratified_smoke_v2r6/fake_all.csv` | `8604397867A0EE12C5EB3FE7DF330ACDB8213149923E6080744C02C26B52AFC7` |
| `stratified_smoke_v2r6/real_snvsm.csv` | `AD2F7ABBDFE20CF99C9735D7AE0D654701F1CC5CBBD5F3E10B5B3C8C7F443C4B` |
| `stratified_smoke_v2r6/fake_snvsm.csv` | `B5B157D0E06481246A53328273F1FD114687E9EDE241802E39A95DC3A236E4F5` |
| `stratified_smoke_v2r6/metadata_gate.json` | `43CD4C065D17C2F5F2A34815760CFF93042FD8B25971F0D4D7FE3E3911D3BA28` |
| `stratified_smoke_v2r6/metadata_predictions.csv` | `4960C85F0E06641B30A3E5D5105DD468805A7481BA5895FE97A478ED5E958FDA` |
| `labels_phase0_stratified_smoke_v2r6.csv` | `4AD7BD45D2AD957AD0449CEFDD7323500CC35AB39542598ABBCAC94D22D2344C` |

### 5.5 Vì sao bảng 5.4 không còn khớp HEAD (cập nhật 2026-07-29)

Đối chiếu bằng máy toàn bộ 33 mục: **6 khớp, 24 lệch, 3 không tìm thấy** (ba mục cuối chỉ do đường dẫn trong bảng ghi tên file trần — `fake_media_contract.py` và `timeline_contract.py` nằm ở `src/pipeline/`, còn `01_build_labels.py` ở `src/pipeline/05_build_labels/`).

Lệch do **ba nguyên nhân độc lập**, không cái nào là hỏng dữ liệu:

| Nguyên nhân | Artifact bị ảnh hưởng |
|---|---|
| `e49140f` — chính commit chứa bảng, sửa file sau khi tính hash | `04_anonymization.py`, `05_snvsm_compress.py`, `01_extract_features.py` |
| `d33835d` — bản vá RNG P1 (seed theo clip thay `random.seed()` toàn cục) | `01_temporal_desync.py`, `02_frame_reverse.py`, `03_pitch_flatten.py` |
| `fc3e75a` + lượt lưu trữ 2026-07-29 — viết lại đường dẫn trong manifest | 20 CSV smoke |

Hệ quả quan trọng: **code ở HEAD không còn là code đã sinh ra bằng chứng r6.** Bản vá RNG P1 đổi hành vi chọn hướng desync, nên nếu chạy lại smoke bằng HEAD sẽ ra media khác. Bảng 5.4 vẫn là bản khóa đúng cho *bằng chứng đã có*.

Bốn artifact **không hề đổi** vì không chứa đường dẫn — và đây đúng là phần mang số liệu:

| Artifact | Trạng thái |
|---|---|
| `real_integration_audit.json` (phase0_smoke, v2r2) | khớp `912A344F…` |
| `stratified_smoke_v2r6/metadata_gate.json` | khớp `43CD4C06…` |
| `stratified_smoke_v2r6/metadata_predictions.csv` | khớp `4960C85F…` |

Nói cách khác, `lag_error_ms = 0.0` và gate AUC `0,546` vẫn được khóa bằng hash gốc.

Hash mới của toàn bộ artifact sau khi chuyển sang `archive/phase0_v2_smoke/` (đường dẫn tương đối tính từ thư mục đó) nằm trong [README của archive](../../archive/phase0_v2_smoke/README.md); 627/627 đường dẫn trong 33 CSV đã được kiểm chứng trỏ tới file có thật.

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
- V2a loader/model có thực sự tiêu thụ structured timeline và fixed-common-window đối xứng ở mọi nhánh;
- metric model sau repair.

Vì vậy trạng thái vẫn là **NO-GO full extraction/full training**. Structured schema, code repair bốn generator và stratified metadata gate đã đạt. Bước tiếp theo là tạo repaired pilot: regenerate đủ bốn method, SNVSM lại 540 real + 2.160 fake, qua Stage 05 và metadata-only gate trên toàn labels, rồi mới extract 2.700 feature vào path versioned trước khi train.
