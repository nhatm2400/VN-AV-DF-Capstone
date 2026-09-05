# Temporal Active-Speaker Curation — thiết kế và trạng thái triển khai

**Ngày cập nhật:** 2026-09-05

**Phạm vi dữ liệu:** 65.622 clip real nguồn trong `data/01_collect/cut_clips/all_manifest.csv`
**Trạng thái:** code và contract đã triển khai; pipeline Light-ASD + Silero + InsightFace + LoCoNet+LASER đã smoke end-to-end trên 1 clip thật với coverage 100%, nhưng **chưa có 450 nhãn chuẩn, chưa khóa ngưỡng, auto gate vẫn NO-GO**.

## 1. Vấn đề cần giải quyết

Gate cũ đo tiếng nói, khuôn mặt và chuyển động độc lập trên toàn clip. Vì vậy một clip 5 giây có thể có 3 giây đầu là người nói hợp lệ nhưng 2 giây cuối là ảnh tĩnh hoặc B-roll còn tiếng MC; phần tốt che phần lỗi khi lấy trung bình. Tầng mới chia clip thành các **bin 200 ms**, theo dõi mọi khuôn mặt và hỏi tại từng bin có tiếng: “có khuôn mặt nhìn thấy nào thực sự đang nói không?”.

Các lỗi mục tiêu:

- `static`: có tiếng nói nhưng vùng miệng đã ổn định bị đóng băng;
- `voiceover`: có tiếng nhưng không có khuôn mặt nào được xác định là người nói;
- `wrong_face`: có người nói nhưng hệ thống bám nhầm track/ROI;
- `dubbed`: môi chuyển động nhưng không tương ứng với audio;
- clip pha trộn: chỉ một đoạn clip bị một trong các lỗi trên.

`cut` chỉ mô tả chuyển cảnh, không được dùng thay cho `static` hoặc `voiceover`.

## 2. Kiến trúc đã triển khai

```text
all_manifest.csv
  ├─ 02_scoring/01_face_quality.py  face quality + embedding (nhánh hiện có)
  └─ 02_scoring/02_active_speaker/01_score.py
                                    VAD 200 ms + face track + mouth motion + Light-ASD
       └─ runs/<run_id>/shards/*
            ├─ asd_clip_scores.csv
            ├─ asd_timeline.jsonl.gz
            ├─ failures.csv
            └─ run_config.json
                 ↓ 02_merge_shards.py (coverage 100%, fail-closed)
       runs/<run_id>/{4 artifact hợp nhất}
                 ↓ 03_export_laser_requests.py → LASER chỉ trên bin laser_requested
       04_run_laser.py → runs/<laser_run_id>/{score + failure + metadata}
                 ↓ 05_apply_laser_scores.py → runs/<enriched_run_id>/{4 artifact}
                 ↓ 06_build_calibration_manifest.py
                 ↓ 07_calibrate.py
       active_speaker_policy_v1.json
                 ↓ 04_curate.py
       temporal gate → face-quality gate → cap speaker → manual review v3
```

### VAD và timeline

Audio được giải mã tạm thành mono 16 kHz, chuẩn hóa PCM về `[-1, 1]`; video gốc không bị ghi đè. Adapter chạy file Silero ONNX 16 kHz nằm trong checkout đã pin, không phụ thuộc `torchaudio`, rồi sinh cờ speech theo bin 200 ms. Quyết định không dựa vào tỷ lệ trung bình toàn clip mà dựa vào đoạn lỗi liên tục dài nhất và tổng thời lượng lỗi.

### Face track và vùng miệng ổn định

InsightFace phát hiện nhiều mặt và cung cấp năm landmark. Detection được nối track bằng IoU, sau đó bbox và landmark được nội suy giữa các lần detect. Năm landmark được dùng để affine-align khuôn mặt; chuyển động môi đo trên phần dưới của mặt đã ổn định nên camera pan/zoom không được dùng làm bằng chứng miệng chuyển động.

### Light-ASD và LASER

- Light-ASD chạy trên từng face track bằng API model của checkout chính thức. Code bắt buộc nhận đường dẫn repo và weight; `run_config.json` lưu Git SHA và SHA-256 weight.
- LASER chỉ chạy ở bin Light-ASD gần ngưỡng, mâu thuẫn với mouth-motion hoặc có nhiều mặt cạnh tranh. Adapter `04_run_laser.py` dùng backbone **LoCoNet+LASER**, checkout chính thức ở commit `3703d3f396cc7b29aa704364f8a9a5ab0c8c1fb9` và checkpoint có SHA-256 `1702df2cc9a6976e4193dcd78d468d3a0f3afc7a926891e0376cc6d2ea72cc1f`.
- Repo upstream không cung cấp sẵn CLI xuất score theo contract của dự án: demo để trống đường dẫn `loadParameters('')` và còn TODO ở forward loop. Adapter vì vậy kiểm tra revision/checkpoint, tái sử dụng decode 25 fps và face tracker của stage này, áp crop 112×112 theo demo chính thức, chạy VGGish audio 16 kHz và chuyển logits hai lớp thành xác suất bằng softmax.
- Checkpoint được huấn luyện với landmark môi, nhưng đường consistency inference của LASER cho phép chạy không cần landmark; adapter truyền landmark feature bằng 0 và ghi rõ `landmarks_at_inference=false` trong metadata. Đây là lựa chọn phải được kiểm chứng bằng calibration, không phải bằng chứng accuracy.
- Demo full-video upstream lọc track ngắn hơn 20 frame. Pipeline này cần chấm đúng bin 200 ms, nên adapter nhận track từ 5 frame (một bin ở 25 fps); policy cấp clip vẫn không thể auto-reject chỉ từ một bin ngắn.
- Thiếu score ở vùng cần LASER dẫn đến `manual`, không bao giờ tự suy thành `reject`.
- Sidecar có `clip_id`, `bin_index`, `laser_score`, trong đó score là xác suất active lớn nhất của mọi mặt trong bin. Không ghép `track_id` giữa hai pipeline vì ID track độc lập không tương đương. Metadata đi kèm phải có schema `laser_sidecar_v1`, `model_git_sha`, `weights_sha256`, `source_timeline_sha256`; script từ chối sidecar không sinh từ đúng base timeline.

Runner ghi `model_git_sha`, SHA-256 weight/config/code/request/timeline, số bin request/scored/missing và `coverage_passed`. `05_apply_laser_scores.py` chỉ nhận sidecar sinh từ đúng base timeline và tạo một run bất biến mới.

Backbone LoCoNet+LASER được chọn thay vì TalkNet+LASER vì paper báo cáo độ bền tốt hơn trong benchmark nhiễu và trực tiếp đánh giá các tình huống audio bị swap/shift. Đây mới là cơ sở chọn kỹ thuật; khả năng tổng quát trên video tiếng Việt vẫn phải được đo bằng tập calibration của dự án.

## 3. Policy quyết định

Module `02_scoring/02_active_speaker/policy.py` là logic thuần, tách khỏi torch/CV để test và grid-search độc lập.

Một clip chỉ được auto-reject khi có bằng chứng lỗi trên bin VAD-active và lỗi:

- liên tục ít nhất 800 ms; hoặc
- tổng cộng ít nhất 500 ms, đồng thời chiếm ít nhất 20% voiced duration.

Light-ASD tự đủ để xác nhận khi nằm ngoài margin độ tin cậy. Vùng gần ngưỡng, model bất đồng, landmark lỗi hoặc nhiều mặt cạnh tranh được đưa về manual. Mọi exception của một clip tạo một hàng `failures.csv`, một summary `temporal_decision=manual`, và vẫn tính vào coverage.

`04_curate.py` chỉ bật binary temporal gate khi JSON policy có `gate_passed=true` và policy trong JSON khớp chính xác policy đã dùng để scoring. Nếu validation chưa đạt, temporal score chỉ là metadata/ưu tiên manual. Điểm ASD liên tục không tham gia `quality_score` hay xếp hạng real.

## 4. Calibration 450 clip

`02_scoring/02_active_speaker/06_build_calibration_manifest.py` chọn 450 clip, 150 clip mỗi tier, tối đa một clip cho mỗi `source_video`. Mẫu được round-robin qua các nhóm rủi ro `clean_candidate`, `static`, `voiceover`, `mixed`, `multiple_faces` dựa trên preliminary scoring.

- 300 clip `tune`: dùng grid-search ngưỡng;
- 150 clip `locked_validation`: không được xem kết quả để sửa ngưỡng;
- hai reviewer gán nhãn độc lập;
- reviewer thứ ba phân xử decision, reason hoặc interval bất đồng;
- 60 nhãn v2 lịch sử chỉ là development seed, không nhập vào validation khóa.

Rubric v3 lưu `bad_intervals_json`, ví dụ:

```json
[{"start_ms":3000,"end_ms":5000,"reason":"voiceover"}]
```

`reason` cấp clip được suy ra từ interval dài nhất. Hai reviewer được xem là đồng thuận về interval khi cùng số đoạn, cùng reason và mỗi biên lệch không quá 200 ms. Kết quả hợp nhất đầy đủ nằm ở `consensus_labels_v3.csv`.

## 5. Validation gate bắt buộc

`02_scoring/02_active_speaker/07_calibrate.py` chỉ dùng 300 clip tune để chọn candidate tối đa hóa số auto-reject với false-reject clean không quá 2%. Candidate đã chọn mới được đánh giá một lần trên 150 clip khóa.

Policy chỉ được publish làm auto gate khi:

- recall `static`/`voiceover`/mixed ít nhất 95%;
- false-reject trên toàn bộ clip clean không quá 2%;
- false-reject clean của từng tier không quá 3%;
- coverage score và timeline đạt 100%;
- không có clip trùng/thiếu và failure không bị mất âm thầm.

Nếu fail bất kỳ điều kiện nào, output ghi `publish_mode=manual_priority_only`.

## 6. Batching và output bất biến

Mỗi shard dùng cùng `run_id`, manifest SHA-256, model/weight SHA và config hash; chỉ `batch_start`, `batch_end`, `shard_id` khác nhau. Giới hạn vận hành là tối đa 5.000 clip/shard và chọn cỡ sao cho dự báo dưới 6 giờ sau smoke.

`02_scoring/02_active_speaker/02_merge_shards.py` từ chối publish nếu:

- range có gap hoặc overlap;
- config/manifest hash khác nhau;
- score thiếu hoặc trùng `clip_id`;
- timeline không phủ đúng toàn manifest;
- failure trỏ đến clip ngoài manifest.

Output hợp nhất đúng contract:

```text
data/02_curate/runs/<run_id>/
├── asd_clip_scores.csv
├── asd_timeline.jsonl.gz
├── failures.csv
└── run_config.json
```

## 7. Thứ tự chạy tiếp theo

Không chạy full ngay. Thứ tự an toàn là:

1. Pin commit của Light-ASD, LASER, Silero; tải weight và ghi SHA-256.
2. Chạy development nhỏ để xác nhận preprocessing/model API.
3. Preliminary-score một candidate pool source-disjoint đủ lớn, enrich các bin mơ hồ bằng LASER rồi tạo manifest 450.
4. Hai người review toàn bộ 450 bằng rubric v3; người thứ ba adjudicate.
5. Grid-search trên 300 và mở validation khóa 150; chỉ giữ policy nếu đạt gate.
6. Smoke 300 clip đủ ba tier trên Kaggle; đo clip/s, VRAM, failure, tỷ lệ cần LASER và kiểm tra bước enrich.
7. Audit 100% disagreement, 10% auto-reject mỗi reason/tier và mọi false-reject.
8. Chia full run thành shard bất biến; merge fail-closed.
9. Chạy `04_curate.py` với `--temporal_scores` và `--temporal_policy`.
10. Chỉ tạo assignment manual từ temporal-pass/manual đã qua face gate; mục tiêu 3.000–6.000 là mục tiêu phụ, không ép bằng cách nới ngưỡng.

Mọi lệnh Python phải dùng `D:\Anaconda\envs\vn_av_df\python.exe` ở local. Full scoring chạy Kaggle GPU và phải chạy background/log theo quy tắc repo.

## 8. Kiểm thử đã chạy

Ngày 2026-09-05, `unittest discover` pass 65 test, skip 1 smoke dữ liệu thật có chủ đích. Các case policy tổng hợp đã cover:

1. người nói liên tục → pass;
2. 3 giây nói + 2 giây miệng tĩnh còn tiếng → reject `static`;
3. 3 giây nói + 2 giây B-roll còn tiếng → reject `voiceover`;
4. đuôi tĩnh nhưng không speech → pass;
5. camera motion không cứu mouth-freeze;
6. dừng môi một bin 200 ms → không reject;
7. nhiều mặt nhưng có một active track rõ → pass;
8. ambiguity đủ dài hoặc inference failure → manual;
9. bin yêu cầu LASER nhưng chưa có sidecar → manual;
10. bất đồng Light-ASD/LASER hoặc ambiguity cùng tồn tại với candidate reject → manual ưu tiên.

Adapter Light-ASD cũng đã được khởi tạo bằng weight `pretrain_AVA_CVPR.model` từ checkout chính thức và chạy tensor giả 1 giây: trả 24 score hữu hạn.

Smoke end-to-end bất biến `dev_asd_real1_20260905_01` chạy clip `-17kN1xdzBE_s0000098754_e0000103754` trong 17,834 giây: 1/1 summary, 25 bin, 0 failure, coverage 100%. `03_export_laser_requests.py` xuất 12 bin cần LASER cùng hash timeline. LASER smoke đầu tiên dùng ngưỡng demo 20 frame và chỉ phủ 11/12 bin; bin đầu thuộc hai track dài 13 frame nên được ghi failure, không mất âm thầm. Sau khi điều chỉnh ngưỡng adapter thành một bin 5 frame và bắt buộc khớp SHA-256 checkpoint/source revision, run `dev_laser_loconet_real1_20260905_03` chấm đủ 12/12 bin trong 9,195 giây, 0 failure, coverage 100%.

`05_apply_laser_scores.py` tạo run `dev_asd_laser_real1_20260905_02`: 25 timeline bin, 12/12 LASER score, 0 failure, coverage 100%. Quyết định của clip vẫn là `manual/ambiguous` và `asd_disagreement_ratio=0,12`; đây là kết quả hợp lý của policy phát triển, không được diễn giải là nhãn thật của clip. Các artifact smoke nằm trong `data/02_curate/runs/`, bị Git ignore và không phải input cho `04_curate.py` vì chỉ phủ một clip.

Các bài trên và smoke một clip chỉ xác nhận đường chạy, contract và fail-closed, **không phải bằng chứng accuracy của pretrained model trên data thật**. Bằng chứng đó chỉ có sau calibration 450 và smoke 300 clip đủ ba tier.

## 9. Chống leakage

Tầng này chỉ chạy trên **real nguồn trước khi sinh fake**. Sau này mỗi fake phải kế thừa quyết định của `source_clip`; không chạy ASD lại trên fake. Như vậy curation không chọn real “sync đẹp” nhưng giữ fake “sync xấu”, và score curation cũng không được đưa vào feature/model training.

## 10. Điểm mạnh và giới hạn

Điểm mạnh:

- bắt lỗi cục bộ theo thời gian thay vì để trung bình che khuất;
- theo dõi nhiều mặt, không cố định “mặt lớn nhất”;
- đo mouth motion sau ổn định landmark;
- failure/manual/reject là ba trạng thái tách biệt;
- artifact bất biến, có hash và kiểm tra coverage;
- validation khóa ngăn tune quá tay.

Giới hạn còn phải đo:

- InsightFace + Light-ASD có thể chậm hơn mục tiêu; batch size chỉ chốt sau smoke;
- Light-ASD domain AVA có thể lệch miền video tiếng Việt/dubbed;
- adapter LASER đã chạy end-to-end nhưng vẫn phải clone checkout và cung cấp checkpoint ngoài repo; chưa có installer/cache manager tự động;
- năm landmark đủ để ổn định tổng thể nhưng kém chi tiết hơn 82 lip landmarks của LASER;
- ngưỡng mặc định chỉ phục vụ development, không được dùng để publish gate;
- `wrong_face` và `dubbed` vẫn cần reviewer/nhánh LASER; không nên suy diễn rằng Light-ASD giải quyết hoàn toàn.

## 11. Tài liệu gốc

- [Light-ASD, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Liao_A_Light_Weight_Model_for_Active_Speaker_Detection_CVPR_2023_paper.html) và [mã nguồn](https://github.com/junhua-liao/light-asd)
- [LASER, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Nguyen_LASER_Lip_Landmark_Assisted_Speaker_Detection_for_Robustness_WACV_2026_paper.html)
- [LoCoNet, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_LoCoNet_Long-Short_Context_Network_for_Active_Speaker_Detection_CVPR_2024_paper.html)
- [AVA ActiveSpeaker](https://research.google/pubs/ava-activespeaker-an-audio-visual-dataset-for-active-speaker-detection/)
- [ASW](https://arxiv.org/abs/2108.07640)
- [Synchformer](https://arxiv.org/abs/2401.16423) — chỉ metadata/chẩn đoán, không phải gate chính.
