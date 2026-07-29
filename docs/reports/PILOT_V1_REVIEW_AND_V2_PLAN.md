# Báo cáo đánh giá AVSP-Net V1 sau pilot và định hướng V2

**Ngày đánh giá:** 2026-07-21

**Phạm vi:** dữ liệu pilot, pipeline sinh fake/trích feature, kiến trúc AVSP-Net V1, kết quả đánh giá và kế hoạch V2
**Trạng thái quyết định:** **NO-GO full extraction/full training bằng V1 cho kết quả cuối**

> Báo cáo này tổng hợp kết quả kiểm tra trực tiếp repository. Các số liệu được ghi là “đo được” khi đã lấy từ artifact hoặc chạy kiểm tra trên dữ liệu hiện có; các ngưỡng V2 là đề xuất để ra quyết định, chưa phải kết quả thực nghiệm.

## 1. Kết luận điều hành

Pilot đã hoàn thành tốt vai trò kiểm tra pipeline:

- lỗi real và fake ghép cặp rơi vào hai split khác nhau đã được sửa;
- 2.700/2.700 clip pilot có feature hợp lệ;
- 540/540 clip anonymization vẫn được trích mouth ROI;
- checkpoint, lịch sử train và kết quả test đã được lưu;
- bảng theo từng phương pháp đã phơi bày điểm yếu mà AUC tổng che khuất.

Tuy nhiên, AVSP-Net V1 chưa đủ điều kiện để chạy full như mô hình cuối vì:

1. Full dataset chỉ có bốn loại pseudo-fake có kiểm soát, chưa đại diện cho deepfake thực tế.
2. `temporal_desync` đang chứa artifact ở biên và độ dài clip do cách dùng FFmpeg.
3. Loader luôn lấy bốn giây đầu, không dùng toàn bộ clip và không có padding mask.
4. Mô hình không có nhánh chuyển động cục bộ hay đầu ra theo frame/đoạn.
5. Offset head không học được offset; consistency loss áp một giả định sai cho nhiều loại fake.
6. AUC cao của pitch và anonymization phần lớn đến từ các dấu vết đơn giản.
7. Hệ thống output hiện chưa bất biến và chưa sinh đầy đủ biểu đồ, prediction hay metadata tái lập.

Schema/valid-range/fixed-common-window đã được khóa ở Bước 2; code repair ba generator không-temporal đã qua synthetic contract ở Bước 3; stratified real-data smoke + metadata-shortcut gate đã đạt ở Bước 4. Hướng tiếp theo là tạo repaired pilot mới qua SNVSM V2 và lặp lại gate trên đủ 2.700 labels. Sau đó mới implement V2a, chạy baseline/ablation nhiều seed và LOMO trước khi đóng băng contract để chạy full. V2b là giai đoạn mở rộng dữ liệu/encoder và external OOD để hướng đến deepfake thực tế.

**Cập nhật 2026-07-22:** generator temporal V2 đã chuyển sang sample-exact circular shift. SNVSM ép H.264 + AAC 16 kHz mono, ghép CRF theo real nguồn và ghi audio/visual contract; Stage 04 trim AAC padding theo `snvsm_target_samples`, kiểm đủ tensor và trả lỗi nếu còn clip fail; Stage 05 fail-fast khi fake rỗng/media thiếu, thiếu method hoặc lệch audio/video/CRF. Schema `av_timeline_v1` và policy `fixed_common_window_v1` đã propagate qua pipeline. Stratified smoke `v2r6` trên 15 nguồn thật đạt đủ 60/60 fake, 75/75 SNVSM media và 75/75 labels. Metadata-only gate GroupKFold theo nguồn đạt logistic AUC 0,530 và random-forest AUC 0,546 (max 0,546 ≤ 0,65); riêng `pitch_flatten` logistic AUC 0,649 sát ngưỡng. Gate này cho phép chuyển sang repaired pilot nhưng không thay thế gate trên đủ 2.700 labels; xem [báo cáo Phase 0](TEMPORAL_DESYNC_PHASE0_SMOKE.md).

## 2. Thuật ngữ sử dụng

| Thuật ngữ | Giải thích |
|---|---|
| ROI | “Region of Interest” — vùng ảnh được quan tâm; trong dự án là vùng miệng crop từ khuôn mặt. |
| Feature | Biểu diễn số đã trích từ video/audio để model sử dụng, ví dụ Wav2Vec, mouth ROI, F0. |
| AUC | Diện tích dưới đường ROC; có thể hiểu gần đúng là xác suất một fake ngẫu nhiên được cho điểm cao hơn một real ngẫu nhiên. |
| F1 | Trung bình điều hòa giữa precision và recall. |
| FPR | Tỷ lệ real bị báo nhầm là fake. |
| OOD | Dữ liệu ngoài phân phối huấn luyện, ví dụ generator hoặc nguồn dữ liệu chưa xuất hiện khi train. |
| Localization | Xác định chính xác thời điểm hoặc đoạn video bị chỉnh sửa. |
| Ablation | Tắt từng nhánh hoặc loss để đo đóng góp thực sự của thành phần đó. |
| LOMO | Leave-One-Method-Out — bỏ một phương pháp fake khỏi train và dùng nó để test khả năng gặp thao tác chưa thấy. |
| TCN | Mạng tích chập theo thời gian. |

## 3. Dữ liệu hiện tại cover đến đâu?

Full labels có 15.005 clip:

| Nhóm | Số clip | Tín hiệu chính |
|---|---:|---|
| Real | 3.001 | Video thật |
| `temporal_desync` | 3.001 | Lệch audio–video toàn cục |
| `frame_reverse` | 3.001 | Đảo chiều một đoạn hình ảnh |
| `pitch_flatten` | 3.001 | Biến đổi cao độ toàn câu |
| `anonymization` | 3.001 | Làm mờ/pixelate khuôn mặt |

Model hiện chỉ dự đoán nhị phân `real/fake`. Nó không dự đoán tên bốn phương pháp. Bảng method-wise trong evaluator chỉ nhóm kết quả bằng ground-truth metadata.

Các nhóm chưa có trong train/test:

- face swap;
- lip reenactment hoặc Wav2Lip;
- TTS và voice cloning;
- voice conversion;
- avatar/talking-head sinh hoàn toàn;
- partial edit ở mức từ hoặc câu;
- generator-disjoint và external cross-dataset.

Do đó kết quả hiện tại là **closed-set trên bốn công thức pseudo-fake**, không phải bằng chứng model phát hiện được mọi deepfake ngoài thực tế.

`anonymization` phù hợp hơn với corruption/robustness test hoặc tamper-quality head, không nên dùng như bằng chứng chính cho deepfake sinh tổng hợp. `pitch_flatten` là stress test prosody mạnh; nó chưa đại diện cho TTS/VC có ngữ điệu tự nhiên.

## 4. Những phần pipeline đã được xác minh tốt

### 4.1 Split và chống leakage

Audit hiện tại xác nhận:

- không có `speaker_id` xuyên split theo metadata;
- không có `source_video` xuyên split;
- fake luôn cùng split với real nguồn;
- mỗi real có đủ bốn fake ghép cặp.

Nên diễn đạt là “không leak theo metadata speaker/source”. `speaker_id` được tạo bằng clustering tự động nên chưa đủ để khẳng định tuyệt đối không trùng danh tính thật.

### 4.2 Anonymization không còn bị drop feature

Pilot có đủ 540/540 anonymization. Extractor dùng box từ real ghép cặp nên không phụ thuộc YOLO phải detect được khuôn mặt đã blur. Các clip anon cũ dùng biến thể blur khác nhau không làm hỏng cơ chế crop này.

### 4.3 Tính toàn vẹn của pilot artifact

- 2.700 labels;
- 2.700 dòng feature index;
- 2.700 file `.pt`;
- toàn bộ extraction status hợp lệ;
- checksum được ghi trong [PILOT_REPORT.md](PILOT_REPORT.md).

## 5. Kết quả pilot và cách diễn giải đúng

Pilot test gồm 405 clip: 81 real và 81 fake cho mỗi phương pháp.

### 5.1 Kết quả tổng thể

| Metric | Giá trị |
|---|---:|
| ROC-AUC | 0,808794 |
| F1 | 0,788530 |
| Precision | 0,940170 |
| Recall fake | 0,679010 |
| FPR real | 0,172840 |

### 5.2 Kết quả theo phương pháp

| Phương pháp | AUC | Recall | Diễn giải hợp lý |
|---|---:|---:|---|
| `pitch_flatten` | 0,990245 | 1,0000 | Phát hiện tốt phép biến đổi F0 rất mạnh. |
| `anonymization` | 0,960372 | 0,9753 | Blur là tín hiệu thị giác dễ phân biệt. |
| `temporal_desync` | 0,749581 | 0,5432 | Có tín hiệu nhưng đang nhiễm artifact generator. |
| `frame_reverse` | 0,534979 | 0,1975 | Gần mức ngẫu nhiên. |

Vì bốn method có cùng số lượng fake và dùng chung 81 real:

```text
overall AUC = mean(AUC của bốn method) = 0,808794
```

Nếu chỉ gộp hai thao tác thời gian `temporal_desync` và `frame_reverse`:

- macro AUC = 0,642280;
- recall = 60/162 = 0,370370.

`frame_reverse` có thể được giải bằng visual motion; không nên gọi toàn bộ nhóm này là khả năng AV synchronization thuần túy.

## 6. Các kiểm tra độc lập quan trọng

### 6.1 Shortcut anonymization

Một baseline chỉ dùng độ sắc nét mouth ROI, variance-of-Laplacian, đạt AUC khoảng **0,9941**, cao hơn AVSP-Net trên anonymization.

Điều này chưa chứng minh nhân quả rằng AVSP-Net chỉ học blur, nhưng chứng minh blur là shortcut đủ mạnh để gần như giải bài toán. Real blur augmentation hiện không cùng phân phối với anon thật: nó chỉ blur mouth ROI và dùng mức blur nhẹ hơn.

Khuyến nghị:

- bỏ anonymization khỏi positive class của bài toán deepfake chính;
- dùng blur/pixelate/compression đối xứng như augmentation cho mọi nhãn;
- đưa anonymization vào robustness suite hoặc quality/tamper head;
- cho phép output `không đủ bằng chứng hình ảnh` khi ROI quá mờ.

### 6.2 Shortcut pitch flatten

Một thống kê đơn giản của `delta_f0` đã đạt AUC khoảng **0,94–0,97**, tùy cách padding và thống kê. Fake thực tế có biến thiên delta cao hơn real do pitch tracking, tái tổng hợp audio và chuẩn hóa theo clip; không đơn giản là `delta_f0 = 0`.

Kết quả này cho thấy pitch task gần như được giải bằng một đặc trưng scalar. AUC 0,99 chưa chứng minh model hiểu sáu thanh tiếng Việt hoặc phát hiện được TTS/VC tự nhiên.

### 6.3 FPR real chưa được chứng minh là do blur

Trên 81 real test, fake score không tăng theo độ mờ; false-positive còn có xu hướng sắc nét hơn true-negative. Vì vậy chưa có bằng chứng quy FPR 17,3% cho anonymization. Cần counterfactual test bằng đúng pipeline blur của anon trước khi kết luận nhân quả.

## 7. Blocking issue: temporal_desync chứa artifact

Generator tại [`01_temporal_desync.py`](../../src/pipeline/03_fake/01_temporal_desync.py) dùng `-itsoffset` kết hợp `-shortest`.

Đo được:

- shift audio muộn `+3/+7/+15`: AUC khoảng `0,947/0,960/0,951`;
- shift audio sớm `-3/-7/-15`: AUC khoảng `0,536/0,546/0,482`;
- với 38 negative test, mouth sequence ngắn hơn real ghép cặp trung bình 10,21 frame;
- positive shift giữ video nhưng tạo khoảng trống hoặc năng lượng thấp ở đầu audio.

Loader lại không có padding mask. Model có thể học leading silence, thời lượng và lượng zero padding thay vì học môi–tiếng.

Vì vậy temporal AUC 0,75 hiện chưa thể xem là bằng chứng AV reasoning. Cần sửa generator để:

1. giữ nguyên duration và frame count ở cả hai hướng;
2. không tạo leading/trailing silence khác phân phối real;
3. không cắt stream bằng `-shortest` theo hướng bất đối xứng;
4. xác minh bằng `ffprobe`, frame count và waveform boundary trước khi extract.

## 8. Giới hạn bốn giây

[`dataset.py`](../../src/train/dataset.py) luôn lấy:

- 200 Wav2Vec timestep;
- 100 mouth frame;
- 400 prosody timestep;

tương ứng khoảng bốn giây đầu.

Trong full `frame_reverse`:

- 258/3.001 clip mất hoàn toàn đoạn reverse;
- 147/3.001 bị cắt một phần;
- riêng train có 195/2.100 clip mất hoàn toàn tín hiệu fake.

Nhưng đây không phải nguyên nhân duy nhất: trong pilot test, phần lớn reverse vẫn xuất hiện trong bốn giây mà AUC vẫn khoảng 0,53.

Các `.pt` đã lưu toàn bộ sequence, nên random windows, sliding-window inference, padding mask, frame difference và local head có thể thêm mà không extract lại feature hiện tại.

## 9. Đánh giá AVSP-Net V1

Kiến trúc hiện tại:

```text
Mouth ROI -> 2D CNN từng frame -> positional encoding -> Transformer
                                                       \
Wav2Vec -> LayerNorm -> Linear -------------------------> one-way cross-attention
                                                          -> attentive pooling
Prosody -> Conv1D -> BiGRU -> attentive pooling ---------/
                                                          -> binary classifier

AV pooled feature -> 7-class global offset head
```

Điểm mạnh:

- pipeline ba nhánh đã chạy end-to-end;
- dùng mouth ROI thay full frame;
- có temporal encoder cho visual và prosody;
- `BCEWithLogitsLoss` đúng về ổn định số;
- có hỗ trợ ablation bằng danh sách branch.

Điểm yếu:

- one-way cross-attention toàn cục, không khóa theo timestamp;
- audio không có positional/mask rõ ở tầng fusion;
- không có explicit motion/frame-difference branch;
- không có local lag/correlation;
- không có frame/segment output;
- không có padding mask;
- mouth-only bỏ qua artifact ở mắt, viền mặt, tóc và nền;
- Wav2Vec thiên về nội dung speech, có thể bỏ mất vocoder artifact;
- threshold 0,5 chưa calibrate;
- dataset âm thầm bỏ label thiếu `.pt`;
- checkpoint không đủ state để resume hoàn chỉnh;
- mới chạy một seed.

`AttentivePool` không phải average pooling cố định; về lý thuyết nó có thể chú ý anomaly ngắn. Nhưng với clip-level label và không có local supervision, tín hiệu reverse vẫn rất khó học.

## 10. Loss hiện tại có xung đột

### 10.1 Offset head

Đo trực tiếp checkpoint:

- offset accuracy = 80,49%;
- majority-zero baseline = 80,49%;
- temporal offset accuracy = 4,94%;
- 90,1% temporal được dự đoán zero.

Offset head hiện chưa học được shift. Offset loss còn được áp cho mọi method và cả unimodal ablation, trong khi visual-only không thể suy ra audio shift.

### 10.2 Consistency loss

Loss hiện ép mọi fake có audio–visual similarity thấp. Giả định này sai với:

- pitch flatten: timing vẫn đúng;
- anonymization: timing vẫn đúng;
- frame reverse: chỉ một đoạn ngắn bị đảo.

Nên thay bằng local aligned-vs-shifted objective trên real và các cặp shift có nhãn hợp lệ, không dùng binary fake label để đẩy xa mọi cặp AV.

## 11. Fact-check nhận xét AI bên ngoài

| Nhận xét | Đánh giá |
|---|---|
| AUC 0,81 che chênh lệch method | Đúng. AUC tổng chính xác là trung bình bốn method do tập cân bằng. |
| Anon AUC chứng minh model chỉ học blur | Quá mạnh về nhân quả; nhưng baseline sharpness 0,9941 xác nhận shortcut blur rất mạnh. |
| FPR 17,3% gần như chắc do blur | Không có bằng chứng; dữ liệu real test không cho thấy hướng này. |
| Pitch task gần tautology | Đúng về độ dễ; cơ chế `delta_f0≈0` không khớp feature đo được. |
| Offset head chủ động giết reverse | Chưa chứng minh; điều chắc chắn là head hiện chỉ bằng majority baseline. |
| Pooling không thể bắt anomaly cục bộ | Quá mạnh; attentive pooling có thể học trọng số nhưng thiếu local supervision. |
| Bốn giây làm mất reverse | Đúng một phần; 8,60% mất hoàn toàn và 13,50% bị cắt ít nhất một phần. Không phải nguyên nhân duy nhất. |
| Cần LOMO, trivial baseline, localization | Đúng và nên làm trước full. |
| Thêm frame difference tự nó bắt buộc re-extract | Sai; có thể tính từ mouth sequence trong cùng feature store. Tuy nhiên repaired pilot vẫn phải extract mới đủ 2.700 vì media đã qua SNVSM V2. |

Ngoài ra, `arXiv:2506.08493` không phải DiMoDif. DiMoDif đúng là `arXiv:2411.10193`.

## 12. Đề xuất AVSP-Net V2

Kiến trúc chi tiết, loss, code layout và roadmap nằm tại [MODEL_PROPOSAL.md](../architecture/MODEL_PROPOSAL.md).

### V2a — sửa bài toán pilot

V2a tái sử dụng **loại feature** hiện tại; repaired pilot dùng feature store mới, không tái dùng binary `.pt` V1. Trọng tâm:

- full-clip synchronized windows;
- padding masks;
- appearance + frame-difference motion branch;
- local AV alignment/lag correlation;
- temporal feature pyramid;
- frame/segment head;
- top-k + masked global aggregation;
- loss chỉ áp khi nhãn tương ứng hợp lệ.

### V2b — hướng tới fake thực tế

V2b mở rộng V2a bằng:

- lipreading-pretrained spatiotemporal encoder;
- real-only self-supervised AV pretraining;
- raw/log-mel audio forensic expert cho TTS/VC;
- full-face/high-frequency visual expert;
- quality/reliability gate;
- Vietnamese syllable/tone supervision khi có transcript;
- dữ liệu generator-disjoint và external OOD.

V2a và V2b không phải hai codebase độc lập. V2b mở rộng V2a bằng module/config, tránh copy kiến trúc và tạo drift.

## 13. Output experiment bất biến

Mọi pilot/full run cần có run ID duy nhất:

```text
experiments/<scope>_<model>_<date>_<git-sha>_<config-hash>/
├── config.json
├── manifest_hashes.json
├── environment.json
├── logs/train.log
├── checkpoints/{best.pt,last.pt}
├── metrics/{validation.json,test.json,method_wise.csv,predictions.csv,threshold.json}
└── plots/{training_curves.png,roc_pr.png,confusion_matrix.png,...}
```

`scope` phải là `pilot` hoặc `full`; `model` là `v1`, `v2a` hoặc `v2b`. Một run đã có `RUN_COMPLETE` thì không được ghi đè. Resume phải ghi tiếp trong cùng run và lưu đủ optimizer/scheduler/scaler state.

## 14. Thứ tự thực hiện và gate

1. ✅ Đã sửa cơ chế và smoke-test `temporal_desync`, tách/guard manifest V1, thêm builder master V2 và contract trim AAC.
2. ✅ Đã khóa schema `av_timeline_v1`, structured valid-range/localization và `fixed_common_window_v1`; contract test chạy trước khi regenerate media.
3. ✅ Đã repair code timing contract của `frame_reverse`, `pitch_flatten`, `anonymization`; synthetic paired media-contract test đạt.
4. ✅ Stratified real-data smoke `v2r6` đạt 15/15 source cho từng method; metadata baseline group-disjoint max AUC 0,546 ≤ 0,65.
5. Tạo master 540 real + 2.160 fake, normalize SNVSM V2, qua Stage 05; chạy lại metadata-only baseline trên toàn labels 2.700 rồi mới extract feature vào path versioned.
6. Implement loader multi-window/padding mask và model V2a.
7. Chạy trivial/unimodal baselines cùng loss/branch ablation.
8. Chạy tối thiểu ba seed và LOMO như unseen-pseudo-method test.
9. Chọn threshold hoàn toàn trên validation và áp các gate đã khóa trước khi train.
10. Chỉ khi V2a qua gate mới đóng băng feature/output contract và chạy full V2a; V2b, external OOD và final holdout nằm ở giai đoạn sau.

Gate đề xuất:

- frame-reverse AUC ≥ 0,70 và cận dưới confidence interval vượt 0,50;
- temporal-desync AUC ≥ 0,80 sau khi generator sạch artifact;
- FPR real ≤ 10% tại threshold chọn hoàn toàn trên validation;
- V2 thắng trivial và best-unimodal baseline;
- worst-method AUC ≥ 0,65 và cận dưới CI 95% > 0,50;
- LOMO macro AUC ≥ 0,60 và không left-out method nào có AUC ≤ 0,50;
- frame AUPRC cao hơn prevalence baseline ít nhất 0,05 và segment AP@IoU=0,5 ≥ 0,20;
- external OOD là report bắt buộc của V2b/final, không dùng final holdout để chọn V2a hay threshold.

## 15. Quyết định cuối

AVSP-Net V1 có giá trị như baseline và phép kiểm tra pipeline. Nó không nên được train full để dùng làm kết quả cuối ở trạng thái hiện tại.

Quy trình tiếp theo:

```text
structured schema + valid-range/mask semantics [đã khóa]
-> repair frame_reverse + pitch_flatten + anonymization [đã implement/test]
-> stratified smoke + metadata shortcut gate [đã đạt trên 15 nguồn]
-> master/SNVSM/Stage05 + full-pilot metadata gate + new 2,700-feature store [bước kế tiếp]
-> V2a loader/model
-> baselines + ablations + 3 seeds + LOMO + validation-only threshold
-> pass explicit gates and freeze feature/output contract
-> full V2a baseline
-> add real deepfake data and V2b experts
-> external OOD evaluation + sealed final holdout
```
