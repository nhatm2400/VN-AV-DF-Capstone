# AVSP-Net V2 — Đề xuất kiến trúc phát hiện deepfake âm thanh–hình ảnh tiếng Việt

**Phiên bản:** 2.0

**Ngày cập nhật:** 2026-07-21

**Trạng thái:** kiến trúc V2a/V2b chưa implement; cơ chế generator/SNVSM V2 đã smoke-test, data contract và repaired pilot còn pending
**Thay thế:** proposal AVSP-Net/VietTone-AVDF V1 trước pilot

## 1. Mục tiêu và phạm vi

AVSP-Net V2 là kiến trúc nhiều chuyên gia theo thời gian dành cho phát hiện và định vị deepfake audio–visual tiếng Việt.

Mục tiêu đầu ra:

1. Xác suất `real/fake` toàn clip.
2. Heatmap điểm nghi ngờ theo thời gian.
3. Các khoảng thời gian có khả năng bị chỉnh sửa.
4. Nhóm bằng chứng: AV-sync, chuyển động hình ảnh, audio/prosody, full-face artifact.
5. Độ tin cậy/chất lượng và trạng thái `không đủ bằng chứng`.

Không đưa bốn pseudo-fake hiện tại thành bốn lớp output bắt buộc. Fake chưa từng thấy có thể không thuộc bất kỳ method nào. Method metadata chỉ dùng cho phân tích và metric.

## 2. Vì sao proposal V1 đã cũ?

Proposal V1 được viết trước khi pilot chạy. Các giả định sau đã không còn phù hợp:

- global cross-attention và attentive pooling đủ để bắt anomaly cục bộ;
- offset head bảy lớp sẽ tự học global AV shift;
- mọi fake đều nên có audio–visual similarity thấp;
- bốn giây đầu đủ đại diện cho toàn clip;
- pitch flatten AUC cao chứng minh hiểu prosody tiếng Việt;
- anonymization có thể giữ như một positive deepfake nếu blur augmentation real được bật.

Pilot và các diagnostic sau pilot cho thấy:

- `frame_reverse` AUC khoảng 0,535;
- offset head chỉ bằng majority-zero baseline;
- consistency similarity chủ yếu tách anonymization;
- pitch/anonymization có trivial shortcut rất mạnh;
- dữ liệu `temporal_desync` V1 chứa artifact biên/độ dài (generator V2 đã sửa, pilot repaired chưa chạy);
- loader không có padding mask và luôn cắt bốn giây đầu.

Vì vậy V2 phải chuyển từ clip-level global fusion sang **local temporal evidence + reliability-aware multi-expert fusion**.

## 3. Nguyên tắc thiết kế

1. **Local trước, global sau:** sinh score theo frame/đoạn rồi mới tổng hợp thành clip score.
2. **Đồng bộ theo timestamp:** các modality phải có common timeline và padding mask.
3. **Mỗi expert giải một loại bằng chứng:** không giả định mọi fake phá mọi modality.
4. **Không dùng corruption làm nhãn deepfake:** blur/compression là nuisance hoặc quality signal.
5. **Tách representation learning khỏi fake recipe:** ưu tiên self-supervised learning trên real.
6. **V2b mở rộng V2a:** không copy thành hai codebase độc lập.
7. **Full chỉ chạy sau khi feature contract và output contract được đóng băng.**

## 4. Hai giai đoạn kiến trúc

### 4.1 V2a — Localized AVSP Core

V2a sửa các lỗi được pilot phát hiện và ưu tiên chạy được trên phần cứng hiện tại. Nó giữ **cùng loại feature** với V1:

- mouth ROI đầy đủ theo thời gian;
- Wav2Vec feature;
- prosody gồm F0, delta-F0, energy và voiced flag.

Các thử nghiệm loader/model-only trên media V1 có thể tái dùng `.pt` V1. Tuy nhiên repaired pilot sạch dùng SNVSM V2 cho cả real và mọi fake, nên phải normalize lại 2.700 media và extract mới toàn bộ 2.700 `.pt` vào feature store versioned. Không được ghép temporal/SNVSM V2 với feature V1. Frame difference, padding mask, random/sliding windows, local attention và frame head vẫn có thể tính từ feature store mới mà không cần thêm loại raw feature.

### 4.2 V2b — Generalization Extensions

V2b giữ nguyên local core của V2a và bổ sung:

- lipreading-pretrained spatiotemporal visual encoder;
- raw/log-mel audio forensic encoder;
- full-face/high-frequency visual forensic encoder;
- real-only self-supervised AV pretraining;
- Vietnamese syllable/tone supervision khi dữ liệu cho phép;
- reliability/quality gating hoàn chỉnh;
- dữ liệu real-world fake và external OOD benchmark.

V2b cần feature/data contract mới cho các nhánh raw audio và full face.

## 5. Kiến trúc tổng thể

```text
Audio/video clip đầy đủ
│
├─ Mouth ROI sequence
│  ├─ Appearance encoder
│  └─ Motion encoder trên [I_t, ΔI_t, |ΔI_t|]
│
├─ Speech representation
│  └─ Wav2Vec projection + temporal convolution
│
├─ Prosody sequence
│  └─ F0/energy/voicing temporal encoder
│
├─ V2b: raw/log-mel audio forensic expert
├─ V2b: full-face visual forensic expert
└─ Quality/reliability measurements
             │
      Resample về common timeline 25 Hz
      + timestamp alignment + length masks
             │
      Local AV alignment module
      ├─ banded cross-attention
      └─ lag correlation trong ±K frame
             │
      Multi-scale temporal feature pyramid
             │
   ┌─────────┼──────────┬────────────┐
AV-sync_t  motion_t  audio_t  face-forensic_t
   └─────────┼──────────┴────────────┘
       reliability-gated evidence fusion
             │
       frame fake score_t
       start/end boundary_t
             │
 top-k local aggregation + masked global aggregation
             │
├─ clip authenticity probability
├─ suspect intervals + heatmap
├─ evidence channel contribution
└─ uncertainty / insufficient quality
```

## 6. Input contract và timeline

### 6.1 Common timeline

Tất cả sequence được resample hoặc ánh xạ về lưới 25 Hz:

- mouth: giữ 25 FPS;
- Wav2Vec: từ khoảng 50 Hz xuống 25 Hz;
- prosody: từ 100 Hz xuống 25 Hz bằng local pooling/convolution;
- label interval: chuyển `start_sec/end_sec` thành frame mask 25 Hz.

Mỗi batch phải trả:

```text
mouth, audio, prosody
valid_length hoặc padding_mask
timestamp/grid metadata
clip label
optional frame mask / boundary labels / lag labels
```

Không được để zero padding đi qua attention/pooling như timestep hợp lệ.

Manifest repaired dùng schema `av_timeline_v1`, source of truth là các cột:

| Nhóm | Cột |
|---|---|
| Version/policy | `timeline_schema_version`, `timeline_mask_policy`, `timeline_boundary` |
| Timeline | `timeline_duration_s` |
| Miền hợp lệ | `audio_valid_start_s/end_s`, `visual_valid_start_s/end_s` |
| Localization | `manipulation_scope`, `manipulation_start_s/end_s` |

`timeline_mask_policy=fixed_common_window_v1`: train/eval chỉ chọn cửa sổ đồng bộ
có độ rộng cố định nằm trong giao miền audio-visual hợp lệ. Không đưa số timestep
hợp lệ, phía biên bị loại hoặc hình dạng raw mask vào clip classifier. Contract
được validate lại ở generator, SNVSM, Stage 05 và Stage 04; feature lưu
`timeline_contract_id` để resume không tái dùng nhầm schema cũ.

### 6.2 Window policy

Train:

- synchronized random windows 2–4 giây;
- ưu tiên lấy window giao với đoạn giả khi có localization label;
- vẫn lấy background/real windows để model không coi mọi window của fake clip là fake;
- có full/global context sampling theo tỷ lệ cấu hình.

Validation/test:

- sliding windows phủ toàn clip;
- overlap 50% hoặc theo config;
- hợp nhất score theo timestamp;
- không chỉ lấy bốn giây đầu.

## 7. Các encoder của V2a

### 7.1 Mouth appearance encoder

Giữ CNN nhẹ trên từng frame, nhưng trả sequence thay vì chỉ pooled vector:

```text
mouth [B,T,1,96,96]
-> 2D CNN
-> appearance sequence E_app [B,T,D]
```

Mục tiêu là giữ texture và hình dạng khẩu hình mà không dùng identity/background toàn frame.

### 7.2 Mouth motion encoder

Tính on-the-fly:

```text
ΔI_t = I_t - I_(t-1)
input_motion = [I_t, ΔI_t, |ΔI_t|]
```

Sau đó dùng CNN nhẹ + temporal convolution/TCN để bắt:

- thứ tự chuyển động;
- vận tốc mở/đóng miệng;
- discontinuity;
- local reverse.

Nhánh này trực tiếp xử lý failure của `frame_reverse` mà không cần extract lại mouth ROI.

### 7.3 Speech content encoder

V2a dùng Wav2Vec feature frozen đã trích:

```text
w2v [B,T,768]
-> LayerNorm
-> Linear projection
-> depthwise temporal convolution
-> audio sequence E_a [B,T,D]
```

Thêm positional/timestamp encoding và mask ở tầng fusion.

### 7.4 Prosody anomaly encoder

Input:

```text
f0_z, delta_f0, energy_z, voiced
```

Dùng temporal convolution hoặc TCN thay vì chỉ pooled BiGRU. Nhánh này trả prosody evidence theo thời gian.

Tên đúng ở V2a là **prosody anomaly expert**, chưa phải Vietnamese tone expert.

## 8. Local AV alignment

Global all-to-all attention cho phép audio ở thời điểm này match với mouth ở thời điểm xa. V2 giới hạn fusion trong vùng thời gian cục bộ.

### 8.1 Banded cross-attention

Với timestep `t`, audio chỉ attend visual trong `[t-K, t+K]`. Khởi đầu đề xuất `K=15` frame ở 25 Hz, tương đương ±0,6 giây; chọn lại bằng validation.

### 8.2 Lag correlation

Tính similarity cho từng lag:

```text
lag ∈ {-15, -7, -3, 0, +3, +7, +15}
```

Nhưng prediction là local/window-level, không phải một global offset class áp cho mọi clip.

Fusion feature tại mỗi thời điểm:

```text
[A_t, V_t, M_t, P_t, |A_t - V_t|, A_t ⊙ V_t, lag_scores_t]
```

Với temporal circular-wrap, manifest phải cung cấp riêng `audio_valid_*` và
`visual_valid_*`. Mask phải đi qua local attention, lag correlation, temporal
heads, pooling và clip evidence; không chỉ mask một loss. Để số timestep hợp lệ
hoặc phía biên bị mask không trở thành shortcut nhãn/dấu shift, train/eval dùng
cửa sổ đồng bộ độ dài cố định nằm trong miền hợp lệ, hoặc áp edge masking/crop
đối xứng tương đương cho real và mọi fake. Clip head không nhận mask count/shape
như feature phân lớp.

## 9. Multi-scale temporal evidence

Sau local alignment, dùng temporal feature pyramid với nhiều receptive field:

- mức ngắn: anomaly 0,2–0,5 giây;
- mức vừa: 0,5–2 giây;
- mức dài: toàn câu hoặc toàn clip.

Có thể dùng dilated TCN/MS-TCN nhẹ để phù hợp GPU hiện tại. Không cần Transformer lớn ở giai đoạn V2a.

## 10. Heads và aggregation

### 10.1 Evidence heads

Mỗi expert sinh một score theo thời gian:

```text
sync_score_t
motion_score_t
prosody_score_t
V2b: audio_forensic_score_t
V2b: face_forensic_score_t
```

### 10.2 Frame và boundary heads

```text
frame_fake_logit_t
start_logit_t
end_logit_t
```

`frame_reverse` có interval `[t0,t1]`; temporal shift có thể dùng window/global mask phù hợp; clip chỉ có weak label dùng Multiple Instance Learning.

### 10.3 Clip aggregation

Kết hợp:

- top-k hoặc log-sum-exp pooling cho anomaly ngắn;
- masked attentive/global mean cho attack toàn clip;
- reliability weights theo modality.

```text
clip_logit = MLP([local_topk, global_masked, expert_summary, quality])
```

## 11. Reliability và quality gate

Quality không được trực tiếp đồng nghĩa với fake.

Gate nhận các tín hiệu như:

- face/mouth detect confidence;
- ROI sharpness;
- audio SNR/voiced coverage;
- missing/corrupted modality;
- sequence coverage.

Gate chỉ điều chỉnh độ tin cậy của expert:

- mouth quá mờ -> giảm visual expert;
- audio thiếu -> giảm audio/prosody expert;
- cả hai không đủ -> trả `insufficient_quality`.

Anonymization được đưa vào robustness suite hoặc tamper/quality head riêng, không phải positive deepfake mặc định.

## 12. Loss V2

```text
L_total =
    L_clip
  + λ_frame    * L_frame
  + λ_boundary * L_boundary
  + λ_lag      * L_local_lag
  + λ_expert   * L_expert_aux
  + λ_inv      * L_corruption_invariance
  + λ_cal      * L_calibration
```

Quy tắc mask loss:

- `L_frame/L_boundary`: chỉ dùng khi có interval hoặc pseudo-mask đáng tin cậy;
- `L_local_lag`: real aligned và synthetic shift đã xác minh, cân bằng các lag;
- `L_expert_aux`: chỉ áp expert có target hợp lệ;
- `L_corruption_invariance`: blur/compression/noise áp đối xứng real/fake;
- không dùng quy tắc `mọi fake -> AV similarity thấp`;
- unimodal ablation phải tắt loss cần modality bị thiếu.

## 13. V2b extensions

### 13.1 Lipreading-pretrained visual encoder

Thay CNN V2a bằng hoặc bổ sung spatiotemporal encoder pretrained bằng lipreading/real talking faces. Mục tiêu là học động lực khẩu hình thay vì artifact của bốn fake recipe.

### 13.2 Real-only self-supervised pretraining

Trên video thật:

- local AV alignment;
- random lag prediction;
- temporal order/reverse discrimination;
- masked cross-modal reconstruction;
- augmentation consistency.

Giai đoạn này giảm phụ thuộc vào fake generator cụ thể.

### 13.3 Audio forensic expert

Wav2Vec thiên về nội dung ngôn ngữ. V2b thêm raw/log-mel encoder để giữ:

- vocoder artifact;
- phase/spectral discontinuity;
- TTS/voice-conversion dấu vết;
- local splice.

### 13.4 Full-face visual forensic expert

Mouth-only có thể bỏ qua face swap hoặc reenactment ngoài miệng. V2b dùng aligned-face crop hoặc patch high-frequency để quan sát:

- viền mặt;
- mắt và biểu cảm;
- texture/frequency artifact;
- face-background inconsistency.

Nhánh này phải có identity/domain regularization và augmentation đối xứng để tránh học speaker/background.

### 13.5 Vietnamese tone expert

Chỉ bật khi có:

- transcript tiếng Việt;
- forced alignment theo âm tiết;
- tone label từ dấu thanh;
- dữ liệu TTS/VC, tone swap và local tone corruption.

Tasks có thể gồm:

```text
tone class: ngang, sắc, huyền, hỏi, ngã, nặng
F0 contour reconstruction
tone–syllable consistency
local tone anomaly
```

Nếu chưa có các điều kiện này, không claim model hiểu thanh điệu; chỉ giữ prosody anomaly expert.

## 14. Data strategy cho generalization

Bốn pseudo-fake hiện tại là controlled probes, không phải final benchmark.

V2b cần bổ sung:

- lip-sync generator thực;
- face swap/reenactment;
- TTS và voice conversion;
- talking-head/avatar/full synthetic;
- partial/local edits;
- nhiều generator mỗi họ;
- nhiều mức codec, resize, noise và social-media processing áp đối xứng;
- hard real clips có blur, low SNR, natural AV lag.

Split/evaluation:

- source/speaker-component disjoint;
- generator-disjoint;
- attack-family-disjoint;
- cross-dataset/external OOD;
- corruption robustness;
- final holdout không dùng để chọn kiến trúc/threshold.

## 15. Metrics

Headline không chỉ dùng overall mixture AUC.

Bắt buộc:

- ROC-AUC, PR-AUC, precision, recall, F1;
- FPR real và operating point được chọn trên validation;
- macro method AUC/F1;
- worst-method AUC/F1;
- confidence interval bootstrap theo source/component;
- frame AUPRC;
- segment AP ở nhiều temporal IoU;
- calibration error/Brier score;
- external OOD và generator-disjoint metrics;
- kết quả tối thiểu ba seed.

## 16. Baselines và ablations

### 16.1 Trivial baselines

- sharpness/variance-of-Laplacian cho anonymization;
- `std(delta_f0)` và voiced coverage cho pitch;
- duration/padding/leading-silence baseline cho temporal;
- frame-difference energy cho reverse;
- SyncNet lag score cho AV desync.

### 16.2 Model ablations

- audio-only;
- visual appearance-only;
- visual motion-only;
- AV local fusion;
- AV + prosody;
- full V2a;
- `w_lag=0`;
- no frame/boundary supervision;
- no local attention;
- no reliability gate;
- V2b expert-by-expert.

### 16.3 Generalization tests

- LOMO trên pseudo-method;
- unseen generator trong cùng attack family;
- unseen attack family;
- external dataset;
- low-quality real counterfactual.

## 17. Code layout đề xuất

V2a và V2b nằm trong cùng package; V2b chỉ bật thêm module qua config.

```text
src/model/v2/
├── avsp_v2.py                  # top-level model và registry expert
├── timeline.py                 # resample, timestamp, padding masks
├── local_alignment.py          # banded attention + lag correlation
├── temporal_pyramid.py         # TCN/MS-TCN
├── aggregation.py              # top-k, log-sum-exp, masked global
├── reliability.py              # quality/reliability gate
├── losses.py                   # masked multi-task losses
├── heads.py                    # frame, boundary, clip, evidence
└── encoders/
    ├── mouth_appearance.py
    ├── mouth_motion.py
    ├── speech_content.py
    ├── prosody.py
    ├── audio_forensic.py       # V2b
    ├── face_forensic.py        # V2b
    └── vietnamese_tone.py      # V2b optional

configs/models/
├── avsp_v2a.yaml               # current features, local core
└── avsp_v2b.yaml               # V2a + new experts/pretraining
```

Không tạo `src/model/v2a/` và `src/model/v2b/` riêng vì sẽ lặp code và dễ lệch implementation.

## 18. Experiment output bất biến

Tên run:

```text
<scope>_<model>_<yyyymmdd-hhmmss>_<gitsha>_<config-hash>
```

Ví dụ:

```text
experiments/pilot_v2a_20260721-210000_a1b2c3d_cfg9f2e/
experiments/full_v2a_20260725-090000_a1b2c3d_cfg9f2e/
experiments/pilot_v2b_20260810-150000_b4c5d6e_cfg18aa/
```

Cấu trúc:

```text
run_dir/
├── RUNNING                      # tạo lúc bắt đầu
├── RUN_COMPLETE                 # chỉ tạo khi kết thúc thành công
├── config.json
├── manifest_hashes.json
├── environment.json
├── source_state.json            # git SHA + dirty state
├── logs/train.log
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── metrics/
│   ├── validation.json
│   ├── test.json
│   ├── method_wise.csv
│   ├── predictions.csv
│   └── threshold.json
└── plots/
    ├── training_curves.png
    ├── roc_pr.png
    ├── confusion_matrix.png
    ├── method_wise_ci.png
    ├── calibration.png
    └── temporal_heatmaps/
```

Quy tắc:

- không dùng run name mặc định có thể trùng;
- không ghi đè run có `RUN_COMPLETE`;
- resume giữ cùng run ID và ghi provenance;
- pilot/full bắt buộc khác `scope`;
- eval kiểm tra labels/features/config hash khớp checkpoint;
- raw feature store có feature version/config hash riêng.

## 19. Roadmap thực hiện

### Phase 0 — Data repair

1. ⚠️ Cơ chế `temporal_desync` đã sửa và smoke: sample-exact shift, giữ duration/frame count, không silence/truncation. SNVSM ghép CRF theo real nguồn; Stage 04 trim AAC padding và Stage 05 fail-fast khi thiếu method hoặc lệch audio/video/CRF.
2. ✅ Structured schema `av_timeline_v1`, valid-range/localization fields và `fixed_common_window_v1` đã implement + contract-test. Việc loader/model thực sự tiêu thụ mask thuộc Phase 1/2; shortcut mask-shape vẫn phải qua metadata gate.
3. ⏳ Repair/regenerate `frame_reverse`, `pitch_flatten`, `anonymization`; cả ba generator V1 còn `-shortest`, audit 15 source phân tầng (5/tier) đã xác nhận timing lệch ở cả tier1/2/3 nhưng chưa ước lượng tỷ lệ toàn bộ.
4. ⏳ Trên smoke đã repair, chạy baseline chỉ dùng metadata/timing/codec; nếu baseline còn tách nhãn tốt thì chưa được extract/train model.
5. ⏳ Tạo master composition đã qua timing audit, normalize SNVSM V2 cho đủ 2.700 clip pilot và chạy Stage 05; chạy lại metadata-only baseline trên toàn labels 2.700 trước khi extract mới toàn bộ feature và audit lại.

Bằng chứng smoke: [TEMPORAL_DESYNC_PHASE0_SMOKE.md](../reports/TEMPORAL_DESYNC_PHASE0_SMOKE.md).

### Phase 1 — V2a data loader

1. Full sequence length và padding masks.
2. Random synchronized windows.
3. Sliding-window validation/test.
4. Fail-fast nếu thiếu feature.

### Phase 2 — V2a model

1. Mouth motion branch.
2. Local AV alignment.
3. Temporal feature pyramid.
4. Frame/boundary/clip heads.
5. Masked losses và reliability gate cơ bản.

### Phase 3 — V2a pilot decision

1. Trivial baselines.
2. Branch/loss ablations.
3. Ba seed.
4. LOMO.
5. Chọn threshold trên validation.

Chỉ khi qua gate mới đóng băng V2a feature/output contract và chạy full V2a baseline.

### Phase 4 — V2b data và representation

1. Thu thập/sinh fake thực tế đa generator.
2. Real-only self-supervised pretraining.
3. Raw/log-mel audio expert.
4. Full-face visual expert.
5. Vietnamese tone expert khi đủ transcript/alignment.

### Phase 5 — Final evaluation

1. Generator-disjoint.
2. Attack-family-disjoint.
3. External OOD.
4. Localization, calibration và robustness.
5. Final holdout chỉ mở một lần sau khi khóa model/threshold.

## 20. Gate đề xuất trước full

V2a pilot nên đạt tối thiểu:

- frame-reverse AUC ≥ 0,70 và cận dưới CI vượt 0,50;
- temporal-desync AUC ≥ 0,80 sau data repair;
- FPR real ≤ 10% tại threshold chọn trên validation;
- thắng best trivial và unimodal baseline;
- không giảm nghiêm trọng trên attack toàn clip;
- có frame AUPRC/segment AP cho local anomaly;
- kết quả ổn định qua ba seed.

Các ngưỡng này là decision gate, không phải cam kết kết quả.

## 21. Claim khoa học phù hợp

### Sau V2a

```text
AVSP-Net V2a detects and temporally localizes controlled Vietnamese
audio-visual and prosodic inconsistencies under source/speaker-component
disjoint evaluation. Generalization to real-world deepfake generators
remains to be established.
```

### Sau V2b và external OOD

Chỉ khi có generator-disjoint/cross-dataset evidence mới có thể claim khả năng tổng quát hóa sang các họ deepfake thực tế đã đánh giá. Không claim phát hiện mọi deepfake chưa biết.

## 22. Tài liệu liên quan

- [Báo cáo pilot gốc](../reports/PILOT_REPORT.md)
- [Đánh giá V1 và kế hoạch V2](../reports/PILOT_V1_REVIEW_AND_V2_PLAN.md)
- [Tổng quan project](../../PROJECT.md)
- [LipForensics](https://openaccess.thecvf.com/content/CVPR2021/html/Haliassos_Lips_Dont_Lie_A_Generalisable_and_Robust_Approach_To_Face_CVPR_2021_paper.html)
- [RealForensics](https://openaccess.thecvf.com/content/CVPR2022/html/Haliassos_Leveraging_Real_Talking_Faces_via_Self-Supervision_for_Robust_Forgery_Detection_CVPR_2022_paper.html)
- [LAV-DF/BA-TFD](https://arxiv.org/abs/2204.06228)
- [DiMoDif](https://arxiv.org/abs/2411.10193)
- [AVFF](https://openaccess.thecvf.com/content/CVPR2024/html/Oorloff_AVFF_Audio-Visual_Feature_Fusion_for_Video_Deepfake_Detection_CVPR_2024_paper.html)
- [AV-Deepfake1M](https://arxiv.org/abs/2311.15308)
