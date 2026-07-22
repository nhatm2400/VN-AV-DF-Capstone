# Báo cáo toàn bộ quá trình Pilot AVSP-Net

> **BÁO CÁO LỊCH SỬ V1 — KẾT LUẬN GO BÊN DƯỚI ĐÃ BỊ THAY THẾ.** Sau diagnostic
> ngày 21/07/2026, quyết định hiện tại là **NO-GO full** cho đến khi ba generator
> không-temporal hết timing artifact `-shortest`, data contract temporal V2, mask và
> repaired pilot đạt gate. Xem
> [đánh giá V1/V2](PILOT_V1_REVIEW_AND_V2_PLAN.md) và
> [smoke Phase 0](TEMPORAL_DESYNC_PHASE0_SMOKE.md). Số liệu/checksum V1 trong file
> này vẫn được giữ nguyên. Hành vi source sinh V1 thuộc snapshot/commit `467f606`;
> các link source tương đối hiện trỏ code V2 mới và không tái hiện generator V1.

> Snapshot được kiểm chứng ngày **20/07/2026** tại repository `VN-AV-DF-Capstone`.
>
> Trạng thái: **PILOT PASS CÓ ĐIỀU KIỆN** — pipeline dữ liệu/feature đủ an toàn để chạy full extraction; kết quả model chưa đủ để tuyên bố mô hình cuối cùng hoạt động tốt trên mọi phương pháp giả mạo.
>
> **Lưu ý:** dòng trạng thái ngay trên là kết luận lịch sử ngày 20/07 và đã bị banner đầu file thay thế; không dùng nó để quyết định run hiện tại.

## 1. Tóm tắt điều hành

Pilot (thử nghiệm quy mô nhỏ trước khi chạy toàn bộ) được thực hiện để trả lời bốn câu hỏi trước khi trích đặc trưng và huấn luyện toàn bộ 15.005 clip:

1. Sau khi sửa cách chia split (tập train/validation/test), real và fake ghép cặp có còn rơi vào các split khác nhau hoặc cùng người nói/video nguồn có bị rò sang nhiều split không?
2. Stage 04 có còn làm rơi toàn bộ anonymization fake (video giả được ẩn danh bằng cách làm mờ mặt) vì YOLO không nhận ra khuôn mặt đã bị làm mờ không?
3. Ba nhánh đặc trưng `mouth ROI + Wav2Vec2 + prosody` — vùng miệng, biểu diễn tiếng nói và đặc trưng ngữ điệu — có được trích đầy đủ, đúng trục thời gian và đủ ổn định để train không?
4. AVSP-Net có học được tín hiệu nào tốt hơn ngẫu nhiên trước khi tốn chi phí chạy toàn bộ dữ liệu không?

Kết quả ngắn gọn:

- Pilot dùng **540 real + 2.160 fake = 2.700 clip**, tương đương khoảng 18% tập full.
- Mỗi real có đúng bốn fake ghép cặp: `temporal_desync`, `frame_reverse`, `pitch_flatten`, `anonymization`.
- **0 speaker leak, 0 source-video leak, 0 real/fake pair lệch split**.
- **2.700/2.700** file feature tồn tại và load được; không có tensor rỗng, NaN hoặc Inf.
- **540/540 anonymization** có mouth ROI; lỗi anon bị drop đã được xử lý trong pilot thật.
- Best validation ROC-AUC (diện tích dưới đường ROC): **0,8126**; test ROC-AUC: **0,8088**. Gate pilot AUC > 0,70 đã đạt.
- Điểm yếu lớn nhất là `frame_reverse`: AUC **0,5350**, recall (tỷ lệ fake phát hiện được) **0,1975**, gần mức ngẫu nhiên.
- `anonymization` đạt AUC **0,9604**, nhưng điểm cao này chưa đủ chứng minh model không dựa vào kiểu blur.

Quyết định tại thời điểm pilot ngày 20/07 (đã bị thay thế): **có thể chạy full feature extraction**, nhưng phải tiếp tục ablation (thí nghiệm tắt từng nhánh) và điều tra `frame_reverse` trước khi đưa ra claim cuối cùng về AVSP-Net.

### 1.1. Từ điển thuật ngữ

Báo cáo giữ tên tiếng Anh trong code và file output để người đọc có thể đối chiếu trực tiếp. Phần dưới giải thích ý nghĩa bằng tiếng Việt. Trong dự án này, quy ước nhãn là **real = 0**, **fake = 1**, nên fake được xem là **lớp dương** khi tính precision, recall và F1.

#### Thuật ngữ dữ liệu và pipeline

| Thuật ngữ | Giải thích bằng tiếng Việt |
|---|---|
| **Pilot** | Lần thử nghiệm quy mô nhỏ trước khi chạy toàn bộ dữ liệu. Mục tiêu là phát hiện lỗi pipeline, ước lượng chi phí và kiểm tra model có học được tín hiệu hay không. |
| **Pipeline** | Chuỗi các bước xử lý nối tiếp nhau, ví dụ thu thập video → cắt clip → làm sạch → sinh fake → trích đặc trưng → huấn luyện → đánh giá. |
| **Stage** | Một công đoạn trong pipeline. Ví dụ Stage 04 là công đoạn trích đặc trưng. |
| **Clip** | Một đoạn video ngắn được cắt ra từ video nguồn dài hơn. |
| **Real** | Clip thật, không bị pipeline cố ý chỉnh sửa để tạo giả mạo. Nhãn của real là 0. |
| **Fake** | Clip giả hoặc đã bị chỉnh sửa theo một phương pháp tấn công. Nhãn của fake là 1. |
| **Deepfake** | Nội dung hình ảnh hoặc âm thanh giả mạo được tạo/chỉnh sửa bằng thuật toán để giống người hoặc sự kiện thật. Pilot hiện dùng pseudo-fake có kiểm soát, chưa phải toàn bộ các dạng deepfake thực tế. |
| **Pseudo-fake** | Fake được tạo có kiểm soát từ một clip real, thay vì lấy từ một deepfake thực tế không biết quy trình tạo. Nhờ vậy có thể biết chính xác loại can thiệp và real nguồn. |
| **Method** | Phương pháp sinh fake, gồm lệch tiếng–hình, đảo frame, làm phẳng cao độ và làm mờ mặt. |
| **Tier** | Nhóm nguồn dữ liệu. Tier 1/2 là các nhóm YouTube theo loại nguồn/giấy phép; Tier 3 là TikTok. |
| **Quality gate** | Bộ điều kiện chất lượng tối thiểu mà dữ liệu phải vượt qua, ví dụ đủ độ phân giải, FPS và có khuôn mặt/tiếng nói. |
| **Curation** | Quá trình chấm điểm, làm sạch, cân bằng và lựa chọn clip phù hợp để tạo tập real cuối cùng. |
| **Source video** | Video dài ban đầu dùng để cắt ra nhiều clip. Nhiều clip cùng `source_video` có thể chứa cùng một người hoặc cùng bối cảnh. |
| **Speaker ID** | Mã cụm người nói được suy ra từ embedding khuôn mặt. Đây là mã kỹ thuật để nhóm danh tính, không phải tên thật. |
| **Embedding** | Vector số cô đọng đại diện cho nội dung hoặc danh tính. Hai khuôn mặt cùng người thường có embedding gần nhau hơn hai người khác nhau. |
| **Manifest** | File CSV đóng vai trò danh mục dữ liệu: mỗi dòng mô tả một clip, đường dẫn file, nguồn, người nói và metadata liên quan. |
| **Artifact** | Sản phẩm được tạo và lưu lại sau một bước chạy, ví dụ manifest CSV, file feature `.pt`, checkpoint hoặc JSON metric. |
| **Metadata** | Thông tin mô tả clip nhưng không phải nội dung hình/tiếng trực tiếp, như `clip_id`, source video, speaker, method, FPS và đường dẫn. |
| **Full run/full extraction** | Chạy trên toàn bộ 15.005 clip, trái với pilot chỉ dùng 2.700 clip. |
| **Reproducibility — khả năng tái lập** | Khả năng người khác dùng cùng code, dữ liệu và cấu hình để tạo lại cùng kết quả hoặc kết quả tương đương. |
| **Checksum/SHA-256** | Chuỗi băm đại diện cho nội dung file. File chỉ cần đổi một phần nhỏ thì checksum sẽ đổi; dùng để xác nhận đúng snapshot đã được kiểm tra. |

#### Thuật ngữ hình ảnh, âm thanh và đặc trưng

| Thuật ngữ | Giải thích bằng tiếng Việt |
|---|---|
| **Feature — đặc trưng** | Biểu diễn số mà model nhận làm đầu vào, được rút ra từ video/audio thay vì đưa nguyên file `.mp4` vào model. |
| **Feature extraction — trích đặc trưng** | Quá trình chuyển clip thành các tensor mouth, Wav2Vec2 và prosody rồi lưu trong file `.pt`. |
| **ROI — Region of Interest** | “Vùng quan tâm”: phần nhỏ của ảnh được chọn để phân tích thay vì dùng toàn bộ khung hình. |
| **Mouth ROI** | Vùng quan tâm chứa miệng. Pipeline detect khuôn mặt, lấy phần dưới của khuôn mặt, chuyển xám và resize thành ảnh 96×96. |
| **YOLO face detector** | Mô hình phát hiện khuôn mặt. Output chính là hộp chữ nhật bao quanh mặt, gọi là face bounding box. |
| **Bounding box/box** | Tọa độ hình chữ nhật bao quanh khuôn mặt. Box được dùng để xác định vị trí crop mouth ROI. |
| **Crop** | Cắt một vùng nhỏ từ ảnh hoặc frame theo box. |
| **Anonymization — ẩn danh hóa** | Che dấu danh tính bằng cách làm mờ hoặc pixel hóa vùng mặt. Trong dataset này audio được giữ nguyên. |
| **Blur** | Phép làm mờ ảnh. Nếu chỉ fake bị mờ, model có thể học “mờ = fake” thay vì học sai lệch audio–visual. |
| **Frame** | Một ảnh đơn trong chuỗi video. Video 25 FPS có khoảng 25 frame mỗi giây. |
| **FPS — Frames Per Second** | Số khung hình mỗi giây. Mouth ROI của pipeline được lấy theo lưới 25 FPS. |
| **Temporal desync** | Làm lệch thời gian giữa tiếng nói và hình miệng. Pipeline dịch audio sớm hoặc muộn ±3/±7/±15 frame. |
| **Frame reverse** | Đảo ngược thứ tự frame trong một đoạn ngắn của video, làm chuyển động miệng cục bộ chạy ngược trong khi audio vẫn chạy xuôi. |
| **Pitch flatten** | Làm phẳng đường cao độ giọng nói, khiến F0 ít biến thiên và làm mất đặc trưng thanh điệu. |
| **Prosody — ngữ điệu** | Các đặc trưng cách câu nói được phát âm, như cao độ, biến thiên cao độ, năng lượng và đoạn hữu thanh. |
| **F0 — tần số cơ bản** | Đại lượng gần với cao độ cảm nhận của giọng nói, đo bằng Hz. F0 đặc biệt quan trọng với tiếng Việt vì tiếng Việt là ngôn ngữ thanh điệu. |
| **VAD — Voice Activity Detection** | Phát hiện khoảng thời gian có hoạt động tiếng nói, dùng để bỏ đoạn im lặng hoặc không có người nói. |
| **Wav2Vec2/W2V** | Mô hình học biểu diễn tiếng nói. Pipeline dùng checkpoint tiếng Việt để biến waveform audio thành chuỗi vector 768 chiều. |
| **Frozen model** | Mô hình chỉ dùng để suy diễn/trích đặc trưng, không cập nhật trọng số trong quá trình train model chính. Wav2Vec2 đang được dùng theo cách này. |
| **Waveform** | Dãy biên độ âm thanh theo thời gian sau khi giải mã audio. |
| **Tensor** | Mảng số nhiều chiều dùng trong PyTorch. Ví dụ mouth có dạng `[thời gian, 96, 96]`. |
| **Dtype** | Kiểu dữ liệu của tensor, như `uint8`, `float16`, `float32`; ảnh uint8 tiết kiệm dung lượng, số thực dùng cho feature/model. |
| **Timestep** | Một bước trên trục thời gian của chuỗi đặc trưng. Tần số timestep của mouth, W2V và prosody không giống nhau. |
| **NaN/Inf** | Giá trị số không hợp lệ: NaN là “không phải một số”, Inf là vô hạn. Chúng có thể làm loss và quá trình train hỏng. |
| **Codec/H.264** | Cách mã hóa và nén video. Codec khác nhau có thể để lại dấu vết khiến model học nhầm nguồn encode. |
| **Encode/decode** | Encode là mã hóa/nén ảnh và tiếng thành file video; decode là giải mã file video trở lại frame/audio để xử lý. |
| **CRF — Constant Rate Factor** | Tham số điều khiển chất lượng/nén khi encode H.264; CRF cao thường nén mạnh hơn và giảm chất lượng hơn. |
| **SNVSM** | Bước đưa real và fake qua cùng pipeline H.264/CRF để giảm “vân codec” khác nhau giữa hai lớp. |
| **Vân codec** | Dấu vết hình ảnh do bộ mã hóa/cấu hình nén tạo ra. Đây có thể trở thành tín hiệu tắt không mong muốn. |

#### Thuật ngữ chia dữ liệu và chống rò rỉ

| Thuật ngữ | Giải thích bằng tiếng Việt |
|---|---|
| **Split** | Phần dữ liệu được chia theo mục đích: train để học, validation để chọn model/cấu hình, test để đánh giá cuối. |
| **Train set** | Tập model trực tiếp dùng để cập nhật trọng số. |
| **Validation/val set** | Tập không dùng cập nhật trọng số, dùng theo dõi model và chọn checkpoint tốt nhất. |
| **Test set** | Tập giữ riêng để đánh giá cuối, không nên dùng lặp lại để tune model. |
| **Speaker-disjoint** | Một speaker không được xuất hiện trong nhiều split. |
| **Source-video-disjoint** | Các clip từ cùng một video nguồn không được rơi vào nhiều split. |
| **Connected component** | Thành phần liên thông của đồ thị. Ở đây clip chung speaker hoặc chung source video được nối lại và buộc đi cùng split. |
| **Data leakage — rò rỉ dữ liệu** | Thông tin từ train lọt sang validation/test, hoặc nhãn vô tình gắn với dấu hiệu phụ, khiến metric cao hơn năng lực thật. |
| **Shortcut — đường tắt học máy** | Tín hiệu dễ nhưng sai mục tiêu mà model lợi dụng, ví dụ học “mặt mờ = fake” hoặc “codec lạ = fake”. |
| **Pairing/ghép cặp** | Liên kết một fake với real nguồn đã sinh ra nó thông qua `source_clip`/`orig_clip_id`. |

#### Thuật ngữ huấn luyện và kiến trúc model

| Thuật ngữ | Giải thích bằng tiếng Việt |
|---|---|
| **Model** | Hàm có tham số học được, nhận feature và dự đoán clip là real hay fake. |
| **GPU** | Bộ xử lý đồ họa có khả năng tính toán song song, dùng để tăng tốc YOLO, Wav2Vec2 và huấn luyện model. |
| **VRAM** | Bộ nhớ riêng của GPU, chứa model, batch và tensor trong khi tính toán. |
| **Train/huấn luyện** | Quá trình cập nhật trọng số model để giảm hàm mất mát trên train set. |
| **Epoch** | Một lượt model đi qua gần như toàn bộ train set. |
| **Batch/batch size** | Nhóm sample xử lý cùng lúc trước một lần cập nhật trọng số. Pilot dùng batch size 16. |
| **Loss — hàm mất mát** | Con số biểu thị dự đoán sai đến đâu; optimizer cố làm loss giảm dần. |
| **Gradient** | Tín hiệu cho biết mỗi trọng số nên thay đổi theo hướng nào và bao nhiêu để loss giảm. |
| **BCE — Binary Cross-Entropy** | Loss cho bài toán phân loại hai lớp real/fake. |
| **Positive weight/`pos_weight`** | Trọng số của lớp dương trong BCE. Pilot đặt 0,25 vì fake nhiều gấp bốn lần real. |
| **Optimizer/AdamW** | Thuật toán cập nhật trọng số từ gradient. AdamW là optimizer được dùng trong pilot. |
| **Learning rate — tốc độ học** | Độ lớn mỗi bước cập nhật trọng số. Quá cao dễ mất ổn định, quá thấp học chậm. |
| **Weight decay** | Thành phần regularization hạn chế trọng số tăng quá lớn, giúp giảm overfitting. |
| **Regularization** | Các kỹ thuật hạn chế model học quá sát train set, nhằm cải thiện khả năng hoạt động trên dữ liệu chưa thấy. |
| **Overfitting — quá khớp** | Model nhớ tốt train set nhưng hoạt động kém trên validation/test hoặc dữ liệu mới. |
| **Gradient clipping** | Chặn độ lớn gradient vượt ngưỡng để train ổn định hơn. |
| **AMP — Automatic Mixed Precision** | Huấn luyện với độ chính xác số hỗn hợp để giảm VRAM và tăng tốc trên GPU. |
| **Checkpoint** | File lưu trọng số model và metadata tại một thời điểm train, ví dụ `best.pt` và `last.pt`. |
| **Early stopping** | Dừng sớm khi validation metric không cải thiện sau một số epoch, tránh train thừa hoặc overfit. |
| **Baseline** | Mốc so sánh đơn giản hơn, như audio-only hoặc visual-only. Model full chỉ có ý nghĩa khi so với baseline phù hợp. |
| **Ablation** | Thí nghiệm chủ động tắt/bỏ một nhánh để đo nhánh đó đóng góp bao nhiêu. |
| **Attention** | Cơ chế để model gán trọng số quan trọng khác nhau cho các timestep/đặc trưng. |
| **Cross-attention** | Attention giữa hai nguồn; ở AVSP-Net, audio truy vấn chuỗi visual để tìm chuyển động miệng tương ứng. |
| **Transformer** | Kiến trúc xử lý chuỗi dựa trên attention; nhánh mouth dùng Transformer để học quan hệ theo thời gian. |
| **BiGRU** | Mạng tuần tự đọc chuỗi theo cả hai hướng thời gian; nhánh prosody dùng BiGRU. |
| **Logit** | Điểm thô trước sigmoid. Logit được đổi thành xác suất/score fake để đặt threshold và tính metric. |
| **Sigmoid** | Hàm biến logit thành score trong khoảng 0–1. Score cao hơn biểu thị model nghiêng nhiều hơn về lớp fake. |
| **Class imbalance — mất cân bằng lớp** | Số real và fake không bằng nhau. Pilot có một real cho bốn fake nên accuracy thô có thể gây hiểu nhầm. |
| **Padding** | Thêm số 0 vào chuỗi ngắn để mọi sample trong batch có cùng chiều dài. |
| **Truncate/cắt ngắn** | Bỏ phần cuối của chuỗi dài hơn giới hạn. Pilot dùng tối đa khoảng bốn giây đầu. |
| **Mask** | Mặt nạ báo cho attention biết timestep nào là dữ liệu thật, timestep nào chỉ là padding. Model hiện chưa dùng padding mask. |

#### Thuật ngữ đánh giá

Trong các công thức dưới đây, fake là lớp dương:

| Thuật ngữ | Giải thích và cách đọc |
|---|---|
| **Threshold — ngưỡng quyết định** | Mốc biến score liên tục thành nhãn. Với threshold 0,5: score > 0,5 được đoán là fake. |
| **Score** | Mức tin tưởng liên tục của model rằng clip là fake, sau khi biến đổi logit bằng sigmoid. |
| **TP — True Positive** | Fake được dự đoán đúng là fake. |
| **TN — True Negative** | Real được dự đoán đúng là real. |
| **FP — False Positive** | Real bị báo nhầm thành fake. |
| **FN — False Negative** | Fake bị bỏ sót và dự đoán thành real. |
| **Accuracy — độ chính xác tổng thể** | `(TP + TN) / tổng số mẫu`. Dễ gây hiểu nhầm khi hai lớp mất cân bằng. |
| **Precision — độ chính xác của cảnh báo fake** | `TP / (TP + FP)`. Trong các clip model báo fake, có bao nhiêu clip thật sự là fake. |
| **Recall/TPR — độ bao phủ fake** | `TP / (TP + FN)`. Trong toàn bộ fake, model bắt được bao nhiêu. Recall thấp nghĩa là bỏ sót nhiều fake. |
| **F1-score** | Trung bình điều hòa của precision và recall. F1 chỉ cao khi cả precision và recall cùng tương đối tốt. |
| **FPR — False Positive Rate** | `FP / (FP + TN)`. Trong toàn bộ real, có bao nhiêu bị báo nhầm là fake; càng thấp càng tốt. |
| **ROC curve** | Đường biểu diễn trade-off giữa recall/TPR và FPR khi quét qua mọi threshold. |
| **ROC-AUC/AUC** | Diện tích dưới đường ROC. AUC ≈ 0,5 là gần đoán ngẫu nhiên; AUC = 1,0 là xếp hạng real/fake hoàn hảo. Có thể hiểu gần đúng là xác suất một fake ngẫu nhiên nhận score cao hơn một real ngẫu nhiên. |
| **PR-AUC** | Diện tích dưới đường Precision–Recall; hữu ích khi số mẫu hai lớp mất cân bằng và tập trung vào chất lượng phát hiện lớp fake. |
| **Balanced accuracy** | Trung bình của recall fake và tỷ lệ nhận đúng real. Hai lớp được coi trọng ngang nhau dù số lượng khác nhau. |
| **Confusion matrix — ma trận nhầm lẫn** | Bảng bốn ô TP/TN/FP/FN, cho biết model đúng và sai theo từng kiểu nào. |
| **Method-wise metric** | Metric tính riêng cho từng phương pháp fake, giúp tránh trường hợp AUC tổng cao nhưng một method cụ thể gần như không phát hiện được. |

## 2. Sơ đồ dữ liệu và sản phẩm đầu ra (artifact)

```text
YouTube tiếng Việt (Tier 1/2) + TikTok (Tier 3)
    │
    ├─ fetch/download/quality gate/cut clip
    ▼
data/01_collect/cut_clips/all_manifest.csv
    6.888 clip hợp lệ / 246 source video
    │
    ├─ face scoring + speaker embedding + curation
    ▼
data/02_curate/all_clean.csv
    3.001 real sạch / 226 source video / 674 speaker_id
    │
    ├─ 01 temporal desync
    ├─ 02 frame reverse
    ├─ 03 pitch flatten
    └─ 04 anonymization
    ▼
3.001 real + 12.004 fake
    │
    ├─ SNVSM: nén H.264/CRF đối xứng real và fake
    ▼
data/03_fake/snvsm/real_snvsm.csv
data/03_fake/snvsm/fake_snvsm.csv
    │
    ├─ connected-component split theo speaker_id ∪ source_video
    ▼
data/05_labels/labels.csv
    15.005 dòng, train/val/test = 70/15/15
    │
    ├─ chọn subset pilot khoảng 18%, giữ nguyên cặp real + 4 fake
    ▼
pilot_real_snvsm.csv + pilot_fake_snvsm.csv + labels_pilot.csv
    │
    ├─ Stage 04: mouth ROI + W2V + prosody
    ▼
data/04_features_pilot/*.pt (2.700 file, 3,29 GiB)
    │
    ├─ AVSP-Net full, early stopping theo validation AUC
    ▼
experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/{best.pt,last.pt,history.json,eval_test.json}
```

## 3. Nguồn dữ liệu ban đầu

### 3.1. Nguồn video

Dữ liệu real gốc là video có người nói tiếng Việt, lấy từ ba tier:

| Tier | Nguồn | Ý nghĩa | Clip hợp lệ sau cắt |
|---|---|---|---:|
| Tier 1 | YouTube, ưu tiên Creative Commons | Nguồn chính, dễ quản lý giấy phép hơn | 4.776 |
| Tier 2 | YouTube, Standard license | Bổ sung độ đa dạng nguồn/người nói | 1.011 |
| Tier 3 | TikTok | Video ngắn, đa dạng bối cảnh và người nói | 1.101 |
| **Tổng** |  | **246 source video** | **6.888** |

Các nhóm code thu thập/cắt dữ liệu:

- Tier 1: [`src/pipeline/01_collect/tier1/`](../../src/pipeline/01_collect/tier1/)
- Tier 2: [`src/pipeline/01_collect/tier2/`](../../src/pipeline/01_collect/tier2/)
- Tier 3: [`src/pipeline/01_collect/tier3/`](../../src/pipeline/01_collect/tier3/)
- Manifest sau khi hợp nhất và loại file hỏng: [`data/01_collect/cut_clips/all_manifest.csv`](../../data/01_collect/cut_clips/all_manifest.csv)

Quá trình trước curation gồm:

- lấy URL bằng YouTube Data API hoặc danh sách TikTok;
- tải video bằng `yt-dlp`;
- quality gate theo độ phân giải, FPS, audio/SNR và sự hiện diện khuôn mặt;
- dùng VAD để tìm đoạn có tiếng nói;
- dùng YOLO face để giữ đoạn có mặt;
- loại chuyển cảnh/rác rõ ràng;
- cắt thành clip khoảng 2–12 giây, phần lớn quanh 4–5 giây.

### 3.2. Làm sạch và lựa chọn real (curation)

Artifact real sạch cuối cùng là [`data/02_curate/all_clean.csv`](../../data/02_curate/all_clean.csv):

| Thuộc tính | Giá trị |
|---|---:|
| Real clip sạch | 3.001 |
| Source video | 226 |
| Speaker ID | 674 |
| Tier 1 | 1.812 |
| Tier 2 | 681 |
| Tier 3 | 508 |

Code curation nằm trong [`src/pipeline/02_curate/`](../../src/pipeline/02_curate/):

- `01_prep_manifest.py`: hợp nhất manifest và xác minh file trên đĩa;
- `02_score_clips.py`: detect/score mặt, sinh embedding nhận dạng speaker;
- `03_sync_score.py`: đo lip-sync khi cần calibrate;
- `04_curate.py`: gate rác rõ ràng, cluster `speaker_id`, cap số clip/speaker;
- `05_eda.py`: thống kê và EDA.

Từ 6.888 clip hợp lệ sau cắt, curation giữ lại 3.001 real có chất lượng phù hợp cho bài toán mouth/audio/prosody.

## 4. Cách sinh dữ liệu giả có kiểm soát (pseudo-fake)

Từ mỗi real sạch, pipeline sinh đúng một fake cho mỗi phương pháp. Vì vậy tập full có 3.001 real và 12.004 fake.

| Method | Code | Cơ chế | Số clip full | Rủi ro chính |
|---|---|---|---:|---|
| `temporal_desync` | [`01_temporal_desync.py`](../../src/pipeline/03_fake/01_temporal_desync.py) | Dịch audio ±3/±7/±15 frame so với video | 3.001 | Model có thể thiên về mức shift lớn, bỏ sót shift nhỏ |
| `frame_reverse` | [`02_frame_reverse.py`](../../src/pipeline/03_fake/02_frame_reverse.py) | Đảo thứ tự frame trong cửa sổ video 0,3–1,0 giây; audio giữ nguyên | 3.001 | Tín hiệu ngắn/cục bộ, dễ bị temporal pooling hoặc crop 4 giây làm loãng |
| `pitch_flatten` | [`03_pitch_flatten.py`](../../src/pipeline/03_fake/03_pitch_flatten.py) | Làm phẳng F0 bằng Parselmouth; video giữ nguyên | 3.001 | Chỉ nhánh prosody/audio có tín hiệu trực tiếp |
| `anonymization` | [`04_anonymization.py`](../../src/pipeline/03_fake/04_anonymization.py) | Gaussian blur/pixelate vùng mặt; audio giữ nguyên | 3.001 | YOLO fail trên mặt mờ; shortcut “mờ = fake” |

Trong pilot:

- temporal desync có đủ sáu hướng/mức shift, mỗi biến thể khoảng 81–98 clip;
- frame reverse có cửa sổ từ 0,3 đến 1,0 giây, trung bình khoảng 0,661 giây;
- pitch flatten có F0 đích từ 98 đến 343 Hz, trung bình khoảng 191,5 Hz;
- anonymization gồm 536 clip `blur_box` và 4 clip `blur_k` cũ.

### 4.1. Đồng bộ codec bằng SNVSM

Các pseudo-fake ban đầu không cùng đường encode: một số method copy video, một số re-encode. Nếu train trực tiếp, model có thể học codec thay vì học audio-visual inconsistency.

[`05_snvsm_compress.py`](../../src/pipeline/03_fake/05_snvsm_compress.py) đưa cả real và fake qua cùng pipeline H.264, chọn deterministic một trong bốn mức CRF `23/30/35/40` cho mỗi clip. Manifest sau SNVSM giữ `speaker_id`, `source_video`, thêm `crf` và `orig_clip_id`.

Nguồn full dùng cho pilot:

- [`data/03_fake/snvsm/real_snvsm.csv`](../../data/03_fake/snvsm/real_snvsm.csv): 3.001 real;
- [`data/03_fake/snvsm/fake_snvsm.csv`](../../data/03_fake/snvsm/fake_snvsm.csv): 12.004 fake.

Phân bố CRF của pilot khá cân bằng:

| Nhóm | CRF 23 | CRF 30 | CRF 35 | CRF 40 |
|---|---:|---:|---:|---:|
| 540 real | 137 | 130 | 138 | 135 |
| 2.160 fake | 525 | 544 | 549 | 542 |

## 5. Hai lỗi nghiêm trọng được xử lý trước pilot

### 5.1. Lỗi real/fake và identity rơi vào nhiều split

Thiết kế split cũ chỉ dựa vào `speaker_id`. Do curation có thể over-cluster một người thật thành nhiều `speaker_id`, cùng một source video có thể rơi vào train và test dưới các ID khác nhau. Audit trước fix tìm thấy 77 source video xuất hiện ở nhiều split.

Fix tại commit `a01c4d6` và code [`src/pipeline/05_build_labels/01_build_labels.py`](../../src/pipeline/05_build_labels/01_build_labels.py):

1. Xây đồ thị với node `speaker_id` và `source_video`.
2. Hai clip chung speaker **hoặc** chung source video thuộc cùng connected component.
3. Chia cả component vào một split bằng greedy bin-packing 70/15/15.
4. Fake bắt buộc đi theo split của real `source_clip` đã sinh ra nó.
5. Script fail nếu còn bất kỳ speaker hoặc source video nào xuyên split.

Tập full hiện tại trong [`data/05_labels/labels.csv`](../../data/05_labels/labels.csv):

| Split | Real | Mỗi fake method | Tổng |
|---|---:|---:|---:|
| Train | 2.100 | 2.100 × 4 | 10.500 |
| Validation | 451 | 451 × 4 | 2.255 |
| Test | 450 | 450 × 4 | 2.250 |
| **Tổng** | **3.001** | **3.001 × 4** | **15.005** |

Audit hiện tại xác nhận cả full labels và pilot labels đều có:

- 0 `speaker_id` nằm trong nhiều split;
- 0 `source_video` nằm trong nhiều split;
- 0 fake khác split với real gốc;
- mỗi real pilot có đúng bốn method fake.

### 5.2. Lỗi anonymization bị drop ở Stage 04

Lỗi ban đầu:

```text
Anon blur mặt → YOLO không detect → mouth=None → clip không có feature
```

Nếu giữ behavior này, toàn bộ 3.001 anonymization fake có nguy cơ biến mất khỏi train và không thể báo cáo method-wise F1.

Fix chính ở commit `45d33a5`, sau đó được tối ưu ở `467f606`, trong [`src/pipeline/04_extract_features/01_extract_features.py`](../../src/pipeline/04_extract_features/01_extract_features.py):

- Mỗi sampled frame của clip thường luôn sinh một ROI nếu clip từng detect được mặt.
- YOLO fail giữa clip: carry-forward box hợp lệ gần nhất.
- YOLO fail ở đầu clip: backward-fill bằng box đầu tiên detect được.
- Chỉ trả `mouth=None` nếu cả clip không có một box hợp lệ nào.
- Với anonymization, không dựa vào YOLO trên mặt đã blur. Code tra `source_clip` của fake, tìm real SNVSM có `orig_clip_id` tương ứng, detect chuỗi box trên real rồi áp chuỗi box đó lên video anon.
- Box được cache theo `orig_clip_id`, tránh detect lặp lại.
- Thiết kế dynamic box được giữ; static union-box bị loại vì union có thể phình mạnh và crop quá nhiều nền.

Kết quả pilot thật: **540/540 anon có mouth ROI**. Tập full có 10 clip anon cũ dạng `blur_k`; pilot chứa 4 clip trong số đó và cả 4 đều extract thành công. Một kiểm tra regression riêng trước pilot cũng đã chạy đủ 10 clip cũ; tuy nhiên log của kiểm tra nhỏ này không được lưu thành artifact độc lập.

### 5.3. Chống shortcut “blur = fake”

[`src/train/dataset.py`](../../src/train/dataset.py) blur on-the-fly mouth ROI của real train với xác suất `real_blur_aug_p=0.25`, sigma ngẫu nhiên 2–8. Vì anonymization chiếm khoảng 1/4 fake, mục tiêu là:

```text
P(blur | real) ≈ 0,25 ≈ P(blur | fake)
```

Validation/test không blur augmentation để đo trên dữ liệu nguyên bản.

Đây là biện pháp giảm leakage, chưa phải bằng chứng shortcut đã bị loại hoàn toàn. Blur augmentation trên mouth ROI của real và blur cả vùng mặt bằng FFmpeg của anon không hoàn toàn cùng phân bố.

### 5.4. Sửa sampler mouth ROI và hiệu năng

Sampler cũ dùng bước frame nguyên `round(src_fps/target_fps)`. Với video 30 FPS, bước này bằng 1 nên thực tế lấy gần 30 FPS nhưng metadata lại ghi 25 FPS, làm lệch mouth với audio 50 Hz và prosody 100 Hz.

Commit `467f606` thay bằng sampler theo timestamp:

```text
output frame j ← source frame round(j × source_fps / target_fps)
```

Kết quả là nguồn 24/25/30/50/60 FPS đều cho lưới mouth gần đúng 25 FPS. Code cũng gộp detect và crop clip thường vào một lần decode; lần decode riêng chỉ còn cần cho anon khi crop video fake bằng box từ real.

### 5.5. Fail-fast khi Wav2Vec2 không load được

Phiên bản trước có thể cảnh báo rồi tiếp tục, tạo hàng nghìn feature không có nhánh audio. Worktree hiện tại đã sửa [`01_extract_features.py`](../../src/pipeline/04_extract_features/01_extract_features.py) để thoát mã lỗi 1 nếu Wav2Vec2 không load được, trừ khi người chạy chủ động truyền `--no_w2v`.

Lưu ý: sửa đổi fail-fast này hiện vẫn là thay đổi chưa commit tại thời điểm lập báo cáo.

## 6. Thiết kế subset pilot

### 6.1. Quy mô

Pilot chọn khoảng 18% real của từng split full, rồi giữ đủ bốn fake ghép cặp cho mỗi real:

| Split | Real | Temporal | Reverse | Pitch | Anon | Tổng |
|---|---:|---:|---:|---:|---:|---:|
| Train | 378 | 378 | 378 | 378 | 378 | 1.890 |
| Validation | 81 | 81 | 81 | 81 | 81 | 405 |
| Test | 81 | 81 | 81 | 81 | 81 | 405 |
| **Tổng** | **540** | **540** | **540** | **540** | **540** | **2.700** |

### 6.2. Độ đa dạng real của pilot

| Split | Real | Speaker | Source video | Tier 1 | Tier 2 | Tier 3 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 378 | 211 | 83 | 253 | 75 | 50 |
| Validation | 81 | 34 | 15 | 43 | 31 | 7 |
| Test | 81 | 39 | 36 | 32 | 14 | 35 |

Pilot không chỉ lấy N dòng đầu của từng split và có phủ nhiều speaker/source video. Tuy nhiên repository hiện không có script hoặc metadata ghi chính xác thuật toán chọn 540 real. Do đó:

- chính pilot hiện tại vẫn tái lập được nếu giữ nguyên ba CSV pilot;
- việc **sinh lại đúng cùng subset** từ full labels chưa tái lập hoàn toàn bằng code;
- đây là một khoảng trống cần sửa trước báo cáo khoa học/final run.

Các manifest pilot:

- [`data/03_fake/snvsm/pilot_real_snvsm.csv`](../../data/03_fake/snvsm/pilot_real_snvsm.csv)
- [`data/03_fake/snvsm/pilot_fake_snvsm.csv`](../../data/03_fake/snvsm/pilot_fake_snvsm.csv)
- [`data/05_labels/labels_pilot.csv`](../../data/05_labels/labels_pilot.csv)

Không dùng `--limit 2700` của extractor để tạo pilot vì extractor nối toàn bộ real trước fake; dùng `--limit` như vậy có thể lấy lệch thành phần thay vì subset ghép cặp cân bằng.

## 7. Trích đặc trưng (feature) pilot

### 7.1. Code và lệnh

Code chính: [`src/pipeline/04_extract_features/01_extract_features.py`](../../src/pipeline/04_extract_features/01_extract_features.py).

Lệnh tương ứng với artifact hiện tại:

```powershell
D:\Anaconda\envs\vn_av_df\python.exe `
  src\pipeline\04_extract_features\01_extract_features.py `
  --real_csv data\03_fake\snvsm\pilot_real_snvsm.csv `
  --fake_labels data\03_fake\snvsm\pilot_fake_snvsm.csv `
  --out_dir data\04_features_pilot `
  --detect_every 4
```

Các tham số còn lại dùng giá trị mặc định:

| Tham số | Giá trị pilot | Ý nghĩa |
|---|---:|---|
| `face_model` | `yolov8n-face.pt` | YOLOv8n-face local tại root |
| `fps` | 25 | Tần số mouth ROI |
| `mouth_size` | 96 | ROI grayscale 96×96 |
| `detect_every` | 4 | Chạy YOLO mỗi bốn sampled frame, frame giữa dùng carry-forward |
| `conf` | 0,25 | Ngưỡng YOLO |
| W2V | bật | Không dùng `--no_w2v` |
| `save_wave` | tắt | Không lưu waveform vào `.pt` |
| `skip_existing` | bật | Resume theo sự tồn tại của file `.pt` |

Startup đã xác nhận `device=cuda` và `wav2vec2-base-vietnamese-250h: OK`.

### 7.2. Ba nhóm feature

Mỗi `.pt` chứa:

| Key | Dtype/shape | Nguồn |
|---|---|---|
| `mouth` | `uint8 [T,96,96]` | YOLO face → crop nửa dưới khuôn mặt → grayscale |
| `w2v` | `float16 [T,768]` | `nguyenvulebinh/wav2vec2-base-vietnamese-250h`, frozen, audio mono 16 kHz |
| `prosody` | `float32 [T,4]` | F0 z-score, delta-F0, energy z-score, voiced mask; hop 10 ms |
| `wave` | `None` | Pilot không bật `--save_wave` |
| metadata | scalar/dict | `clip_id`, label, method, speaker, fps, source path |

Parselmouth là backend chính cho F0; code có fallback `librosa.pyin`.

### 7.3. Sản phẩm feature được tạo ra

- Thư mục: [`data/04_features_pilot/`](../../data/04_features_pilot/)
- Index: [`data/04_features_pilot/features_index.csv`](../../data/04_features_pilot/features_index.csv)
- File `.pt`: **2.700**
- Dung lượng `.pt`: **3.532.245.266 byte ≈ 3,29 GiB**
- Thời gian theo timestamp file: khoảng **43,9 phút**
- Tốc độ quan sát: khoảng **0,975 giây/clip**
- Ngoại suy tuyến tính cho 15.005 clip full: khoảng **4,1 giờ**, nên dự phòng 4–5 giờ trên cùng máy/cấu hình.

Audit toàn bộ 2.700 file:

- tập ID trong labels, index và `.pt` khớp chính xác;
- index có 2.700 dòng, tất cả `status=ok`;
- không duplicate `clip_id`;
- không thiếu source video;
- không tensor rỗng;
- không NaN/Inf trong W2V hoặc prosody;
- metadata label/method/speaker trong `.pt` khớp labels;
- chiều dài trong `.pt` khớp `features_index.csv`;
- khoảng chiều dài quan sát:
  - mouth: 33–298 frame;
  - W2V: 72–596 timestep;
  - prosody: 143–1.190 timestep.

## 8. Bộ nạp dữ liệu (dataset loader) và tăng cường dữ liệu (augmentation)

Code: [`src/train/dataset.py`](../../src/train/dataset.py).

Loader đọc `labels_pilot.csv`, rồi tìm file theo quy ước:

```text
<features_dir>/<clip_id>.pt
```

Mỗi nhánh được pad/truncate thành chiều dài cố định khoảng bốn giây:

| Nhánh | Chiều dài train cố định |
|---|---:|
| W2V | 200 timestep |
| Mouth | 100 frame |
| Prosody | 400 timestep |

Augmentation chống blur leakage chỉ bật ở train và chỉ áp lên real với xác suất 0,25. Validation/test không augment.

Tập train có tỉ lệ real:fake = 1:4. `pos_weight` cho fake-positive được tính bằng `n_real/n_fake = 0,25`, giúp BCE không bị bốn lần fake lấn át hoàn toàn.

## 9. Mô hình (model) pilot

Code: [`src/model/avsp_net.py`](../../src/model/avsp_net.py).

Pilot dùng đủ ba nhánh `audio,visual,prosody`, tổng **2.292.298 tham số trainable**:

```text
W2V feature [T,768]
    └─ LayerNorm + Linear → audio sequence [T,256]

Mouth [T,96,96]
    └─ 2D CNN ×4 + positional encoding + Transformer ×2
       → visual sequence [T,256]

Audio query ── Multi-Head Cross Attention ── visual key/value
    └─ attentive pooling → AV embedding [256]

Prosody [T,4]
    └─ Conv1D + BiGRU + attentive pooling → prosody embedding [128]

concat(AV, prosody)
    └─ MLP → real/fake logit
    └─ auxiliary offset head → 7 lớp [-15,-7,-3,0,3,7,15]
```

Loss tổng:

```text
BCE(real/fake)
+ 0,5 × CrossEntropy(offset class)
+ 0,1 × audio/visual consistency loss
```

## 10. Cấu hình train

Code: [`src/train/train.py`](../../src/train/train.py).

Lệnh tái dựng từ `args` được lưu trực tiếp trong checkpoint. Riêng `--run_name` bên dưới đã được chuẩn hóa thành run ID bất biến sau pilot; checkpoint gốc vẫn ghi tên legacy `pilot_avsp_full`, được giữ trong `config.json` để bảo toàn provenance:

```powershell
D:\Anaconda\envs\vn_av_df\python.exe `
  src\train\train.py `
  --labels data\05_labels\labels_pilot.csv `
  --features data\04_features_pilot `
  --branches audio,visual,prosody `
  --run_name pilot_v1_20260720-214741_467f606_b8c61ed7 `
  --epochs 30 `
  --bs 16 `
  --lr 0.0003 `
  --wd 0.0001 `
  --workers 2 `
  --patience 7 `
  --real_blur_aug_p 0.25 `
  --amp `
  --seed 42
```

Chi tiết optimizer/train:

- AdamW;
- cosine annealing learning rate;
- mixed precision AMP;
- gradient clipping norm 1,0;
- early stopping theo validation ROC-AUC;
- checkpoint `best.pt` theo validation AUC, `last.pt` theo epoch cuối.

Kết quả train:

- chạy 28 epoch, từ epoch index 0 đến 27;
- tổng thời gian epoch ghi trong history: khoảng 1.187,5 giây, tức 19 phút 47,5 giây;
- best tại epoch index 20, tương đương epoch thứ 21;
- best validation AUC: **0,8125667**;
- best validation accuracy tại threshold 0,5: **0,7111111**;
- sau best có bảy epoch không cải thiện nên early stopping đúng theo `patience=7`; đây không phải crash.

Artifact train:

- [`experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/best.pt`](../../experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/best.pt)
- [`experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/last.pt`](../../experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/last.pt)
- [`experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/history.json`](../../experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/history.json)
- [`experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/config.json`](../../experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/config.json)
- [`experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/source_state.json`](../../experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/source_state.json)
- [`experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/manifest_hashes.json`](../../experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/manifest_hashes.json)
- [`experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/RUN_COMPLETE`](../../experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/RUN_COMPLETE)

Run ID được tạo từ scope/model, thời điểm bắt đầu suy ra từ history, commit base tồn tại lúc chạy và hash cấu hình hiệu dụng:

```text
pilot_v1_20260720-214741_467f606_b8c61ed7
```

Đây là run hoàn tất bất biến; không dùng lại ID này cho một lần train khác.

## 11. Đánh giá trên tập test

Code: [`src/eval/evaluate.py`](../../src/eval/evaluate.py).

Lệnh:

```powershell
D:\Anaconda\envs\vn_av_df\python.exe `
  src\eval\evaluate.py `
  --ckpt experiments\pilot_v1_20260720-214741_467f606_b8c61ed7\best.pt `
  --labels data\05_labels\labels_pilot.csv `
  --features data\04_features_pilot `
  --split test `
  --bs 16 `
  --thresh 0.5
```

Test có 405 sample: 81 real và 324 fake.

### 11.1. Chỉ số đánh giá (metric) tổng thể

| Metric | Giá trị |
|---|---:|
| Accuracy | 0,708642 |
| Precision | 0,940171 |
| Recall fake | 0,679012 |
| F1 fake | 0,788530 |
| ROC-AUC | **0,808794** |
| FPR trên real | 0,172840 |
| Balanced accuracy, kiểm tra độc lập | 0,753086 |
| PR-AUC, kiểm tra độc lập | 0,949445 |

Confusion matrix tại threshold 0,5:

|  | Dự đoán real | Dự đoán fake |
|---|---:|---:|
| Real | TN = 67 | FP = 14 |
| Fake | FN = 104 | TP = 220 |

Do test có tỉ lệ real:fake = 1:4, classifier luôn đoán fake sẽ có accuracy 80%. Vì vậy accuracy 70,86% không nên được đọc độc lập. ROC-AUC và balanced accuracy phản ánh pilot hợp lý hơn; class weighting đã chủ động đổi trade-off để tránh bỏ qua real.

### 11.2. Chỉ số theo từng phương pháp (method-wise metric)

Mỗi method được so với cùng 81 real test:

| Method | N fake | Recall | F1 | ROC-AUC | Nhận định |
|---|---:|---:|---:|---:|---|
| `anonymization` | 81 | **0,9753** | 0,9080 | **0,9604** | Anon không còn bị drop; rất dễ tách nhưng còn nguy cơ blur shortcut |
| `pitch_flatten` | 81 | **1,0000** | **0,9205** | **0,9902** | Nhánh prosody bắt rất mạnh |
| `temporal_desync` | 81 | 0,5432 | 0,6331 | 0,7496 | Có tín hiệu nhưng recall threshold 0,5 còn thấp |
| `frame_reverse` | 81 | **0,1975** | **0,2883** | **0,5350** | Gần random; điểm yếu nghiêm trọng nhất |

Kết quả lưu tại [`experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/eval_test.json`](../../experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/eval_test.json).

### 11.3. Xác minh metric độc lập

Checkpoint `best.pt` đã được load lại, inference lại toàn bộ 405 test sample trên GPU, rồi so ROC-AUC bằng `sklearn.metrics.roc_auc_score`.

- ROC-AUC chạy lại: `0,8087943910989179`;
- ROC-AUC trong JSON: `0,8087943910989178`;
- sai khác chỉ ở mức floating-point `1,11e-16`.

Do đó metric trong `eval_test.json` tái lập được từ checkpoint và feature hiện tại.

## 12. Môi trường chạy được xác minh

| Thành phần | Giá trị |
|---|---|
| OS/shell | Windows / PowerShell |
| Python executable | `D:\Anaconda\envs\vn_av_df\python.exe` |
| Python | 3.10.20 |
| PyTorch | 2.12.0+cu126 |
| CUDA runtime của PyTorch | 12.6 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Transformers | 5.14.1 theo `requirements.txt` |
| YOLO/Ultralytics | 8.4.53 theo `requirements.txt` |
| Parselmouth | 0.4.7 theo `requirements.txt` |

Các dependency đầy đủ nằm trong [`requirements.txt`](../../requirements.txt).

## 13. Điểm mạnh của pilot

1. **Hai lỗi dữ liệu nghiêm trọng đã được kiểm chứng bằng artifact thật.** Split hiện không leak speaker/video và anon không còn biến mất khỏi feature.
2. **Pairing được giữ chặt.** Mỗi real pilot có đủ bốn fake và tất cả cùng split, nên không có cùng nội dung ở train và test dưới nhãn khác nhau.
3. **Codec được normalize đối xứng.** Real và fake cùng SNVSM/CRF, giảm shortcut codec.
4. **Ba nhánh feature đều hoàn chỉnh.** Không có clip thiếu mouth, W2V hoặc prosody.
5. **Trục thời gian mouth đã được sửa về 25 FPS thật.** Không còn lỗi video 30 FPS bị ghi metadata 25 FPS.
6. **Feature extraction đủ nhanh để scale.** Pilot thực tế cho ước tính khoảng 4–5 giờ thay vì 14 giờ trên cùng máy.
7. **Train có bằng chứng học được tín hiệu.** Validation/test AUC đều trên 0,80 và vượt gate 0,70.
8. **Prosody branch có giá trị rõ ràng.** `pitch_flatten` đạt AUC gần 0,99, phù hợp mục tiêu dữ liệu tiếng Việt có thanh điệu.
9. **Checkpoint tự mô tả cấu hình train.** `best.pt` lưu branches, epoch, validation metric và toàn bộ args.
10. **Method-wise evaluation đã hoạt động.** Cả bốn phương pháp đều xuất hiện trong report, đặc biệt anonymization không còn bị mất.

## 14. Điểm yếu, rủi ro và phần chưa được chứng minh

### 14.1. Điểm yếu mô hình

1. **Frame reverse gần random.** AUC 0,535 và recall 0,198 cho thấy full model hiện hầu như chưa bắt được đảo frame cục bộ.
2. **Temporal desync recall còn thấp.** AUC 0,75 nhưng recall tại threshold 0,5 chỉ 0,543; shift nhỏ có thể khó phát hiện.
3. **FPR real 17,28% còn cao.** 14/81 real bị báo fake.
4. **Chưa chạy ablation bắt buộc.** Chưa có audio-only, visual-only, audio+visual để chứng minh full AVSP-Net tốt hơn unimodal baseline.
5. **Không có padding mask.** Sequence pad bằng zero nhưng Transformer/attention không nhận mask.
6. **Chỉ lấy bốn giây đầu.** Clip dài bị truncate từ đầu; cửa sổ reverse nằm sau giây thứ tư có thể biến mất khỏi input.
7. **Chưa random temporal crop.** Train luôn dùng đoạn đầu, giảm độ đa dạng và có thể là một nguyên nhân làm `frame_reverse` yếu.

### 14.2. Leakage/độ đại diện

1. **Blur shortcut chưa bị bác bỏ hoàn toàn.** Xác suất blur được cân bằng gần đúng, nhưng kiểu blur real augmentation khác pipeline anonymization thật.
2. **Anon AUC quá cao cần kiểm tra phản chứng.** Nên chạy visual-only và test trên real được blur bằng đúng pipeline FFmpeg/box như fake.
3. **Dữ liệu là pseudo-fake.** Kết quả không tự động chuyển thành hiệu năng trên deepfake thực tế ngoài phân phối.
4. **Test mỗi method chỉ có 81 fake.** Đủ làm gate pilot nhưng khoảng tin cậy vẫn rộng; chưa tính confidence interval.
5. **Class imbalance làm accuracy dễ gây hiểu nhầm.** Cần ưu tiên AUC, balanced accuracy, FPR và method-wise metric.

### 14.3. Reproducibility và vận hành

1. **Không có script sinh subset pilot.** Ba manifest pilot tồn tại nhưng thuật toán chọn subset chưa được version-control.
2. **Manifest pilot đang bị `.gitignore`.** `data/03_fake/*` và `data/05_labels/*` chỉ ngoại lệ cho full `labels.csv`, nên các CSV pilot hiện chỉ có trên máy local.
3. **Index và JSON pilot đang untracked.** `data/04_features_pilot/` và `experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/` xuất hiện trong `git status` do `features_index.csv`, `history.json` và `eval_test.json`. Toàn bộ `.pt`, gồm 3,29 GiB feature và checkpoint, đã được global-ignore bởi `*.pt`; chúng chỉ bị add nếu cưỡng bức bằng `git add -f`.
4. **Không có console log extraction/train đầy đủ.** Tham số train phục hồi được từ checkpoint, nhưng extractor không lưu `run_config.json` hoặc log immutable.
5. **Default extractor chưa an toàn cho full SNVSM.** Default vẫn trỏ `all_clean.csv`, `data/03_fake/labels.csv` và `detect_every=2`; full run phải truyền manifest SNVSM và `--detect_every 4` rõ ràng.
6. **`skip_existing` chỉ kiểm tra file tồn tại.** Nó không biết `.pt` cũ được tạo với FPS/detect interval/model nào; đổi config mà reuse cùng output có thể trộn feature khác cấu hình.
7. **Dataset loader âm thầm bỏ label không có `.pt`.** Pilot đã audit không mất mẫu, nhưng full run cần audit lại sau extraction.
8. **`features_index.csv` là index vận hành, không phải nguồn train trực tiếp.** Train tìm `.pt` theo `clip_id`; index cần được audit riêng để tránh hiểu nhầm rằng index tự khóa dataset.
9. **Sửa fail-fast W2V chưa commit.** Nếu checkout sang máy/branch khác mà không mang diff này, extractor có thể quay lại behavior tiếp tục không có audio.

## 15. Danh mục sản phẩm đầu ra (artifact inventory)

### 15.1. Dữ liệu và manifest

| Artifact | Vai trò | Quy mô/trạng thái |
|---|---|---|
| [`data/01_collect/cut_clips/all_manifest.csv`](../../data/01_collect/cut_clips/all_manifest.csv) | Manifest clip sau collect/cut | 6.888 clip |
| [`data/02_curate/all_clean.csv`](../../data/02_curate/all_clean.csv) | Real sạch trước fake | 3.001 clip |
| [`data/03_fake/snvsm/real_snvsm.csv`](../../data/03_fake/snvsm/real_snvsm.csv) | Real sau normalize codec | 3.001 clip |
| [`data/03_fake/snvsm/fake_snvsm.csv`](../../data/03_fake/snvsm/fake_snvsm.csv) | Bốn fake method sau normalize codec | 12.004 clip |
| [`data/05_labels/labels.csv`](../../data/05_labels/labels.csv) | Labels full đã sửa leakage | 15.005 dòng |
| [`data/03_fake/snvsm/pilot_real_snvsm.csv`](../../data/03_fake/snvsm/pilot_real_snvsm.csv) | Real input của extractor pilot | 540 dòng |
| [`data/03_fake/snvsm/pilot_fake_snvsm.csv`](../../data/03_fake/snvsm/pilot_fake_snvsm.csv) | Fake input của extractor pilot | 2.160 dòng |
| [`data/05_labels/labels_pilot.csv`](../../data/05_labels/labels_pilot.csv) | Train/val/test labels pilot | 2.700 dòng |

### 15.2. Feature và model output

| Artifact | Vai trò |
|---|---|
| [`data/04_features_pilot/`](../../data/04_features_pilot/) | 2.700 feature `.pt` |
| [`data/04_features_pilot/features_index.csv`](../../data/04_features_pilot/features_index.csv) | ID, feature path, label/method và chiều dài ba nhánh |
| [`experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/best.pt`](../../experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/best.pt) | Checkpoint tốt nhất theo validation AUC |
| [`experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/last.pt`](../../experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/last.pt) | Checkpoint epoch cuối trước early stop |
| [`experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/history.json`](../../experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/history.json) | Loss, validation AUC/accuracy, LR và thời gian từng epoch |
| [`experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/eval_test.json`](../../experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/eval_test.json) | Test metric tổng thể và theo method |

### 15.3. Code trực tiếp tham gia pilot

| Stage | File |
|---|---|
| Fake generation | [`src/pipeline/03_fake/01_temporal_desync.py`](../../src/pipeline/03_fake/01_temporal_desync.py) |
| Fake generation | [`src/pipeline/03_fake/02_frame_reverse.py`](../../src/pipeline/03_fake/02_frame_reverse.py) |
| Fake generation | [`src/pipeline/03_fake/03_pitch_flatten.py`](../../src/pipeline/03_fake/03_pitch_flatten.py) |
| Fake generation | [`src/pipeline/03_fake/04_anonymization.py`](../../src/pipeline/03_fake/04_anonymization.py) |
| Codec normalization | [`src/pipeline/03_fake/05_snvsm_compress.py`](../../src/pipeline/03_fake/05_snvsm_compress.py) |
| Labels/split | [`src/pipeline/05_build_labels/01_build_labels.py`](../../src/pipeline/05_build_labels/01_build_labels.py) |
| Feature extraction | [`src/pipeline/04_extract_features/01_extract_features.py`](../../src/pipeline/04_extract_features/01_extract_features.py) |
| Dataset/augmentation | [`src/train/dataset.py`](../../src/train/dataset.py) |
| Training | [`src/train/train.py`](../../src/train/train.py) |
| Model | [`src/model/avsp_net.py`](../../src/model/avsp_net.py) |
| Evaluation | [`src/eval/evaluate.py`](../../src/eval/evaluate.py) |

## 16. Mã kiểm tra tính toàn vẹn (checksum) của snapshot

Các SHA-256 dưới đây khóa snapshot artifact đã được audit. Nếu file thay đổi, hash sẽ khác.

| File | SHA-256 |
|---|---|
| `pilot_real_snvsm.csv` | `74C1ECF5654160D38BCB12CFF814AA782E960693E9C60E1AAAAD92F3A5511CA2` |
| `pilot_fake_snvsm.csv` | `6535BDCAAB5B9CE6F98101C0DA08D578C8FB31DAE4666A5AB7FED6FF78CEDACB` |
| `labels_pilot.csv` | `3E42A7DF5AE3B945A02AACD009B2294A214FD281C167697BA446A2AC5FF7ED5F` |
| `features_index.csv` | `21C84DD21C34CB7AA3E7B907EDFD1EC82B9190E3D63C5B80E745588A2B8FDA44` |
| `history.json` | `03E0FD9664F7667791D70B2955EE86A05845DE94CBC4770C5ABFAC1829028530` |
| `eval_test.json` | `BB80A7815D8137C19968083BEBF2AEBEDD12552B7868AFCE2DEF9CF57EA3FA78` |
| `best.pt` | `9F016A6FC490C589BAC9765811A07E55ACEFE70E9B4A9ED5CE2965DCD307C496` |

Không hash từng `.pt` trong tài liệu để tránh một bảng 2.700 dòng; `features_index.csv`, số lượng file và tensor audit được dùng làm kiểm tra tổng thể vận hành. Hash `eval_test.json` khác snapshot trước chỉ vì trường đường dẫn checkpoint được đổi sang run ID bất biến; toàn bộ score và metric giữ nguyên.

## 17. Git timeline liên quan

| Commit | Nội dung |
|---|---|
| `e0f36cc` | Hoàn thiện bốn fake method, stage 04/05 và AVSP-Net train/eval |
| `842b4d0` | Thêm SNVSM codec normalization |
| `dff4b1e` | Tối ưu anonymization bằng FFmpeg one-pass/resume |
| `6d1e230` | CRF deterministic, encoder và skip-existing cho SNVSM |
| `37f6551` | Tạo `labels.csv` full 15.005 dòng |
| `a01c4d6` | Sửa split bằng connected component speaker + source video |
| `45d33a5` | Cứu anon mouth ROI, giữ timeline và thêm real blur augmentation |
| `e6c2036` | Sửa lỗi import khi chạy trực tiếp `src/train/train.py` |
| `467f606` | Sửa sampler 25 FPS thật và detect/crop one-pass |

Worktree lúc lập báo cáo còn:

- modified: `src/pipeline/04_extract_features/01_extract_features.py` — pyrefly ignore + W2V fail-fast;
- untracked: `data/04_features_pilot/features_index.csv` (các `.pt` bên cạnh đã bị ignore);
- untracked: `experiments/pilot_v1_20260720-214741_467f606_b8c61ed7/history.json` và `eval_test.json` (checkpoint `.pt` đã bị ignore);
- các CSV pilot bị ignore bởi rule dữ liệu.
- untracked: chính `PILOT_REPORT.md` cho tới khi được add/commit.

## 18. Cách chạy full sau pilot

Không dùng default extractor hiện tại. Dùng manifest SNVSM và cấu hình pilot đã xác minh:

```powershell
D:\Anaconda\envs\vn_av_df\python.exe `
  src\pipeline\04_extract_features\01_extract_features.py `
  --real_csv data\03_fake\snvsm\real_snvsm.csv `
  --fake_labels data\03_fake\snvsm\fake_snvsm.csv `
  --out_dir data\04_features `
  --detect_every 4
```

Điều kiện startup bắt buộc:

```text
Device: cuda
wav2vec2-base-vietnamese-250h: OK
Tổng 15005 clip cần trích
```

Không truyền `--no_w2v`. Không dùng `--limit`. Không đổi FPS, mouth size, YOLO model hoặc detect interval giữa các lần resume vào cùng output directory.

Sau full extraction phải audit lại:

1. Có đúng 15.005 `.pt`.
2. Tập ID labels = index = `.pt`.
3. Mọi status là `ok`.
4. Không nhánh nào rỗng/NaN/Inf.
5. Đủ 3.001 clip cho từng method, đặc biệt anonymization.
6. Labels full vẫn 0 speaker/source-video leak.

## 19. Việc nên làm tiếp theo

### Trước hoặc song song với full extraction

1. Commit sửa W2V fail-fast.
2. Thêm script deterministic để sinh pilot subset hoặc lưu chính thức ba CSV pilot nhỏ.
3. Quyết định rõ artifact nào cần version-control: giữ global-ignore cho `.pt`; thêm ignore cho output JSON/CSV không cần commit hoặc chủ động commit index/metric nhỏ để làm bằng chứng.
4. Lưu `run_config.json` và console log cho mỗi extraction/train run.

### Trước khi claim model cuối

1. Chạy ba ablation trên cùng pilot:
   - audio-only;
   - visual-only;
   - audio+visual không prosody.
2. So full AVSP-Net với các baseline, không chỉ nhìn AUC tuyệt đối.
3. Điều tra `frame_reverse`:
   - random temporal crop thay vì luôn lấy bốn giây đầu;
   - bảo đảm cửa sổ reverse nằm trong crop;
   - thêm padding/attention mask;
   - kiểm tra visual-only có học được đảo frame không.
4. Kiểm tra blur leakage bằng real test được anonymize/blur bằng đúng pipeline của fake.
5. Báo cáo balanced accuracy, FPR real và confidence interval bên cạnh ROC-AUC/F1.
6. Sau khi full train, chỉ dùng test split cho đánh giá cuối; không tune threshold/hyperparameter lặp đi lặp lại trên test.

## 20. Kết luận

Pilot đã hoàn thành mục tiêu kỹ thuật chính:

- split leakage đã được sửa và audit sạch;
- anonymization không còn bị Stage 04 drop;
- pipeline feature ba nhánh chạy đủ 2.700 clip;
- AVSP-Net học được tín hiệu với test AUC khoảng 0,81, vượt gate 0,70;
- thời gian extraction đủ khả thi để scale lên full.

Điểm cần giữ thái độ thận trọng là pilot **không chứng minh mọi method đã được giải quyết**. `frame_reverse` vẫn gần random, `temporal_desync` recall còn trung bình, blur shortcut chưa bị loại bằng phản chứng và ablation chưa chạy. Vì vậy kết luận đúng là:

> **Kết luận lịch sử ngày 20/07, đã bị thay thế: GO cho full feature extraction; GO có điều kiện cho nghiên cứu tiếp; chưa GO cho claim mô hình cuối cùng.**
