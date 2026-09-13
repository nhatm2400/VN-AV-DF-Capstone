# Kế hoạch dọn dẹp và tái cấu trúc repo cho nghiên cứu lip-sync audio–visual tiếng Việt

> **Cập nhật 13/09/2026:** Đợt chuyển cấu trúc và dọn dữ liệu đã thực hiện trên nhánh `codex/research-reset`. Xem [báo cáo thực hiện](docs/reports/research/2026-09-13_REPO_RESET.md) và [PROJECT.md](PROJECT.md). Nội dung dưới giữ kế hoạch ban đầu; các module model/generator dự kiến chưa được triển khai.

**Tôi đề xuất dọn thành một repo làm việc mới cho hướng lip-sync audio–visual, giữ lại các công cụ nền có giá trị và một archive gọn về nghiên cứu cũ.** Bản backup ở ổ khác giữ toàn bộ trạng thái cũ; repo đang làm việc chỉ chứa những gì phục vụ hướng mới.

Ngày lập: 12/09/2026. Repo được đối chiếu: `E:\FPTU\PRJ\VN-AV-DF-Capstone`.

Tài liệu này chép lại đầy đủ các phần của phương án dọn repo đã trao đổi, mở rộng phần cấu trúc đích để giải thích từng thư mục và các file dự kiến. Nội dung được cập nhật theo trao đổi tiếp theo về độ đa dạng nguồn, giấy phép và mục tiêu viết paper.

**Trạng thái: kế hoạch đề xuất. Lượt này chỉ tạo file kế hoạch này; chưa xóa dữ liệu, chuyển báo cáo, sửa code, thay môi trường hoặc thực hiện tái cấu trúc.** Bản backup ở ổ khác chưa được kiểm tra trong lượt này. Các đường dẫn bên dưới, trừ khi ghi khác, được tính từ thư mục gốc repo.

## 1. Chốt ranh giới của lần dọn

Sau khi hoàn thành, repo cần thể hiện rõ:

- **Data:** video được tuyển chọn từ podcast, bài nói, thuyết trình và các playlist bổ sung; tiếp tục kiểm soát độ đa dạng và quyền sử dụng.
- **Fake:** lip-sync do generator tạo; bốn pseudo-fake cũ thuộc nghiên cứu lịch sử.
- **Model:** detector audio–visual mới; AV-HuBERT là ứng viên đang kiểm chứng.
- **Nghiên cứu chính:** so sánh X/Y, đánh giá riêng generator chưa thấy và mức nén.
- **Giữ nguyên nguyên tắc:** chống leakage, quản lý nguồn gốc, xử lý real/fake công bằng, báo cáo đúng những gì đã đo.

Câu hỏi trung tâm:

> Trong video một người nói tiếng Việt, phương pháp X có cải thiện khả năng phát hiện lip-sync manipulation so với baseline Y trên protocol Z hay không, với kết quả riêng cho generator chưa thấy và video bị nén?

| Thành phần | Ý nghĩa trong hướng mới |
|---|---|
| Y | Detector sử dụng audio và hình ảnh vùng miệng, học phân biệt có/không có AI chỉnh sửa môi. |
| X | Chính detector Y, thêm cách huấn luyện khuyến khích dự đoán nhất quán giữa các bản nén của cùng clip. |
| So sánh công bằng | X và Y cùng kiến trúc, cùng dữ liệu và cùng các bản nén; khác nhau ở yêu cầu nhất quán khi huấn luyện. |
| Z | Tách người nói và nguồn trước khi sinh fake; khóa validation/test; báo kết quả theo generator và mức nén. |
| Vai trò audio | Cung cấp lời nói để đối chiếu với hình ảnh; không mặc định audio của fake là giọng AI. |
| Điều cần kiểm chứng | AV-HuBERT có phù hợp với tập tiếng Việt của nhóm không; audio có giúp không; X có cải thiện so với Y không. |

Không mang sang trạng thái mới các con số cũ như 65.622 clip, AUC 0,809 hoặc curation NO-GO như thể chúng mô tả dataset/model mới. Chúng vẫn được lưu trong lịch sử. Kết quả cũ cũng không phải baseline trực tiếp của bài toán mới nếu dữ liệu, loại fake và protocol khác nhau.

### 1.1. Giữ mục tiêu của ba tier, cập nhật cách tổ chức

Ba tier ban đầu phục vụ ý định đa dạng nguồn và kiểm soát quyền sử dụng. Việc chuyển sang playlist tuyển chọn không có nghĩa bỏ hai mục tiêu đó.

Code hiện tại cần được phân biệt với ý định thiết kế:

- `src/pipeline/01_collect/tier1/01_fetch_youtube_urls.py` tìm video với `videoLicense="creativeCommon"`.
- `src/pipeline/01_collect/tier2/01_fetch_youtube_urls.py` kiểm tra giấy phép rồi giữ loại `youtube`, tức Standard YouTube License.
- Vì vậy, không ghi trong tài liệu mới rằng toàn bộ ba tier cũ đều đã được xác minh CC.

Đề xuất giữ nhãn nhóm nguồn nếu nhóm còn dùng nó để quản lý độ đa dạng, nhưng tách riêng các thông tin sau:

| Thông tin | Ví dụ / tác dụng |
|---|---|
| Nhóm nội dung | Podcast, bài giảng, thuyết trình; dùng để thống kê độ đa dạng. |
| Kênh, chương trình, tập | Biết video đến từ đâu và phát hiện nguồn liên quan nhau. |
| Người nói | Theo dõi số người độc lập và chống leakage do host xuất hiện nhiều lần. |
| Giấy phép ghi trên nguồn | CC BY, Standard YouTube License, chưa xác định. |
| Căn cứ sử dụng | Giấy phép đã kiểm tra, sự cho phép trực tiếp hoặc căn cứ khác đã được rà soát. |
| Phạm vi được sử dụng | Nghiên cứu, tạo biến thể lip-sync, đưa ví dụ vào paper/demo, phát hành media; ghi riêng theo hồ sơ thực tế. |

Không tự đổi tên ba tier thành ba mức giấy phép rồi coi đó là cách phân nhóm nguồn đã được thống nhất. Downloader và bộ cắt clip nên nhận manifest chung, trong khi nhãn tier/nhóm nguồn vẫn có thể được giữ trong metadata.

YouTube có giấy phép tiêu chuẩn và Creative Commons Attribution. Giấy phép cần được kiểm tra ở từng video; chọn cùng kênh/playlist với VLR không chứng minh video đó có CC. Nguồn không có CC cũng không được tự động đánh dấu là được phép chỉ vì dùng cho nghiên cứu. [Hướng dẫn giấy phép của YouTube](https://support.google.com/youtube/answer/2797468?hl=en).

Đối với nguồn dự định tạo fake và phát hành công khai, nhóm cần xác định phạm vi sử dụng tương ứng. Tài liệu này tổ chức cách lưu hồ sơ, không thay thế việc rà soát pháp lý cho từng trường hợp.

## 2. Backup trước, rồi mới dọn

Bản copy cần bao gồm cả file ẩn, `.git`, file chưa commit và file bị Git bỏ qua. Repo hiện có thay đổi chưa commit và tài liệu chưa được theo dõi, nên chỉ lưu một commit/tag là chưa đủ.

Trước khi xóa dữ liệu:

1. Xác định chính xác thư mục nguồn và thư mục backup ở ổ khác.
2. Đối chiếu số file, dung lượng và kiểm tra mở được một số media/tài liệu trong backup. Kiểm tra thêm hash của manifest, config và những bằng chứng quan trọng.
3. Nếu có liên kết thư mục, kiểm tra backup chứa dữ liệu thực hay chỉ trỏ về ổ cũ.
4. Ghi nhận trạng thái Git, commit hiện tại, thay đổi chưa commit và các file chưa được theo dõi.
5. Giữ lịch sử `.git` trong repo làm việc. Nên thực hiện chuyển đổi trên nhánh riêng, chẳng hạn `codex/research-reset`, để review được toàn bộ thay đổi.

Không cần tạo lại Git repository từ đầu. Không xóa thay đổi đang làm dở chỉ để có working tree sạch.

Đặc biệt, các chỉnh sửa hiện có trong `PROJECT.md`, công cụ ROI preview, báo cáo curation và các test liên quan phải có trong backup trước khi xử lý. Cấu hình bí mật và cookie nếu được backup vẫn là dữ liệu riêng; không đưa chúng vào archive công khai.

Khi thực sự thực hiện thao tác xóa/di chuyển, phải kiểm tra đường dẫn tuyệt đối nằm đúng trong repo làm việc, không trỏ nhầm vào bản backup. Dùng thao tác PowerShell với đường dẫn cụ thể; không xóa theo tên thư mục suy đoán.

## 3. Dữ liệu cũ: đưa ra khỏi nơi làm việc sau khi backup được kiểm tra

| Nhóm hiện có | Đề xuất |
|---|---|
| Video trong `data/raw/`, clip đã cắt trong `data/01_collect/cut_clips/` | Xóa khỏi repo làm việc sau khi backup đạt kiểm tra. |
| Media bốn pseudo-fake và các bản nén cũ | Xóa khỏi repo làm việc nếu còn tồn tại. |
| Feature, embedding, cache của dataset cũ | Xóa khỏi vùng dữ liệu hiện hành; không dùng lại cho backbone/dataset mới. |
| ROI preview, assignment, calibration, review output cũ | Đưa ra khỏi vùng dữ liệu hiện hành; giữ bản nhỏ cần truy vết trong archive. |
| Manifest, split, URL và nhãn cũ | Lưu những bản cần truy vết vào archive; phần đầy đủ nằm trong backup. |
| Log và số đo quan trọng của pilot | Giữ bản nhỏ trong archive. |
| File tạm, cache chạy thử, output lỗi | Xóa sau khi đã chọn giữ bằng chứng lỗi cần thiết. |

Danh sách trên là nhóm cần xử lý, không khẳng định thư mục nào cũng đang có media. Khi thực hiện phải lập danh sách thực tế trước; một số thư mục hiện chỉ có `.gitkeep`.

**Không nên chỉ thay video mà giữ nguyên manifest/calibration cũ.** Những dữ liệu này gắn với quần thể clip cũ và dễ khiến pipeline đọc nhầm. Các ngưỡng lọc học từ population cũ cũng không được mặc định phù hợp với nguồn mới.

Nguồn URL cũ có thể lưu làm tham khảo, nhưng danh sách nguồn mới phải có phiên bản và tiêu chí chọn riêng. Nếu sau này tận dụng lại một video cũ, đưa nó qua kiểm tra quyền sử dụng, chất lượng, trùng nguồn và split của bộ mới. Không nối trực tiếp nhãn cũ vào manifest mới.

Dataset audio riêng như VSASV, nếu còn lưu ở vị trí khác, không mặc định thuộc dữ liệu chính cho bài toán lip-sync video. Không xóa một dataset ngoài danh sách đã kiểm kê chỉ vì nó từng được nhắc trong cuộc trao đổi.

### 3.1. Dữ liệu mới phải được dựng theo luồng mới

```text
Nguồn tuyển chọn + hồ sơ quyền sử dụng
    -> video gốc
    -> cắt clip + kiểm tra người đang nói + review
    -> xác định nhóm người/nguồn và khóa split
    -> tạo lip-sync fake
    -> tạo bản nén cho cả real và fake
    -> trích đặc trưng audio/hình ảnh
    -> train và đánh giá
```

Cắt các clip ứng viên để review có thể diễn ra trước bước khóa split. Tuy nhiên, các clip chung nguồn/người và các đoạn chồng lấn phải được gom cùng nhóm; sau đó mới sinh fake, bản nén và cửa sổ dùng cho thí nghiệm.

Nguồn mới vẫn phải kiểm tra host chung giữa nhiều tập. Nếu nối người nói với tập nguồn tạo thành một nhóm rất lớn, cần bổ sung nguồn độc lập hoặc thu hẹp nguồn được chọn; không chia ngẫu nhiên clip để đạt tỷ lệ train/validation/test đẹp.

Mọi fake, bản nén và cửa sổ kế thừa split của clip nguồn. Nếu dùng audio từ clip khác để thay lời, nguồn audio cũng phải thuộc cùng split. Pilot đã dùng để chọn model là dữ liệu phát triển, không đổi tên thành final test về sau.

## 4. Code: giữ phần nền, thay phần gắn với bài toán cũ

Không cần viết lại mọi dòng code. Tận dụng theo chức năng, đồng thời kiểm tra lại đầu vào/đầu ra khi chuyển sang manifest mới.

| Phần hiện tại | Hướng xử lý | Vị trí đích dự kiến |
|---|---|---|
| Downloader YouTube | Tận dụng; đổi đầu vào sang URL/playlist tuyển chọn, giữ kiểm tra giấy phép riêng. | `src/data/collect_sources.py`, `check_licenses.py`, `download.py` |
| `src/pipeline/01_collect/cut_clips_core.py` | Giữ phần cắt, kiểm tra media và xử lý lỗi phù hợp; tách phụ thuộc thư mục tier khỏi logic cắt. | `src/data/cut_clips.py` |
| Công cụ preview/review | Giữ xem hình, nghe tiếng, ghi quyết định và chia việc; đơn giản hóa cho tập mới. | `src/tools/review/` |
| Lọc chất lượng mặt và người đang nói | Giữ phần cần dùng; đo lại trên nguồn mới. Không bắt pilot chạy hết hệ thống lọc cũ. | `src/data/quality_checks.py` |
| Logic gom người nói + video nguồn để chia split | Giữ và tách thành module riêng. | `src/data/build_splits.py` |
| `timeline_contract.py`, phần kiểm tra media của `fake_media_contract.py` | Giữ mốc thời gian, cửa sổ hợp lệ và kiểm tra audio/video; bỏ ràng buộc chỉ dành cho pseudo-fake. | `src/data/media_checks.py` |
| Nén real/fake, kiểm tra metadata | Tận dụng phần xử lý; cấu hình lại cho thí nghiệm mới. | `src/data/compress.py`, `src/evaluation/check_shortcuts.py` |
| Bốn script tạo pseudo-fake | Đưa ra khỏi code hiện hành; bản đầy đủ nằm trong backup/Git. | Không có trong luồng chính mới. |
| Bộ trích mouth/Wav2Vec2/prosody cũ | Thay bằng preprocessing và trích đặc trưng cho ứng viên backbone mới. | `src/features/` |
| AVSP-Net, dataset loader và loss cũ | Đưa ra khỏi luồng chạy hiện hành; chỉ tận dụng thành phần độc lập nếu còn phù hợp. | `src/models/`, `src/training/` |
| Training/evaluation | Tái sử dụng logging/metrics hữu ích; thay đầu vào và bổ sung bảng generator × mức nén. | `src/training/`, `src/evaluation/` |
| Script TikTok, phục hồi các đợt dữ liệu cũ | Đưa ra khỏi code hiện hành nếu không còn nhiệm vụ. | Backup/Git; ghi chỉ dẫn trong archive khi cần. |
| `configs/test.py`, `src/utils/test.py` và placeholder khác | Kiểm tra nội dung, xóa nếu thực sự không có nhiệm vụ. | Thay bằng config/module có chức năng cụ thể khi cần. |

Đặc biệt, `src/pipeline/05_build_labels/01_build_labels.py` đang gộp **logic chia split hữu ích** với **ràng buộc đủ bốn fake cũ**. Đây là nơi cần tách chức năng, không nên bê nguyên hoặc xóa toàn bộ.

Generator mới dùng weights có sẵn để tạo hoặc chỉnh môi. Các ứng viên trong hướng đi hiện có là Wav2Lip, MuseTalk và một generator giữ riêng như LatentSync. Chúng chưa được coi là đã tích hợp chỉ vì có tên trong cấu trúc dự kiến. Phiên bản, giấy phép, khả năng chạy và chi phí phải được kiểm tra trước khi chốt.

Fake chính ban đầu là video được generator tái tổng hợp môi bằng chính audio gốc. Việc này thay bốn pseudo-fake trong bộ train chính, nhưng chưa đại diện đầy đủ cho tình huống thay lời. Thử thay lời có thể bổ sung và báo kết quả riêng sau khi pipeline chính ổn.

Các test cũng đi theo chức năng: giữ/chuyển test cho phần được tái sử dụng; test riêng generator cũ nằm trong lịch sử. Sau khi chuyển cần kiểm tra chống leakage, quan hệ real–fake, phiên bản cache và đồng bộ thời gian cho pipeline mới.

## 5. Archive tài liệu: gom theo một mốc nghiên cứu

Repo đã có `docs/archives/`; dùng tiếp tên này thay vì tạo thêm cả `archive/` và `archives/` ở nhiều nơi.

Đề xuất một nhóm `docs/archives/legacy_avsp/`:

| Tài liệu hoặc bằng chứng hiện tại | Nơi đến trong `docs/archives/legacy_avsp/` |
|---|---|
| `docs/reports/PILOT_REPORT.md` | `reports/PILOT_REPORT.md` |
| `docs/reports/PILOT_V1_REVIEW_AND_V2_PLAN.md` | `reports/PILOT_V1_REVIEW_AND_V2_PLAN.md` |
| `docs/reports/TEMPORAL_DESYNC_PHASE0_SMOKE.md` | `reports/TEMPORAL_DESYNC_PHASE0_SMOKE.md` |
| `docs/reports/CUT_CLIPS_HOTFIX_AND_REBUILD_PLAN.md` | `reports/CUT_CLIPS_HOTFIX_AND_REBUILD_PLAN.md` |
| `docs/reports/ACTIVE_SPEAKER_CURATION_IMPLEMENTATION.md` | `reports/ACTIVE_SPEAKER_CURATION_IMPLEMENTATION.md` |
| `docs/architecture/MODEL_PROPOSAL.md` về AVSP-Net V2 | `architecture/MODEL_PROPOSAL.md` |
| Các nhật ký curation/rebuild trong `docs/logs/` | `logs/`, giữ nguyên tên file. |
| `docs/archives/2026_report_checkpoints/` | `2026_report_checkpoints/`, giữ nguyên nhóm và tên file bên trong. |
| Config, history, metrics và nguồn gốc của pilot trong `experiments/` | `experiments/<run_id>/` |
| Manifest/split/URL cần truy vết | `manifests/`, có chỉ mục giải thích thuộc đợt nào. |
| README/PROJECT/CLAUDE trước chuyển hướng | `project_snapshot/`, giữ bản mô tả trạng thái cũ. |
| `requirements.txt` và môi trường cũ cần tái lập | `environment/` |

Archive cần một `README.md` ngắn ghi:

- Đây là nghiên cứu trước khi chuyển hướng.
- Kết quả áp dụng cho dataset/model nào.
- Hạn chế đã biết và những kết quả đã bị bác bỏ hoặc chỉ có ý nghĩa pilot.
- Commit/trạng thái Git liên quan và cách tra bản backup đầy đủ. Đường dẫn ổ đĩa cá nhân có thể ghi trong hồ sơ nội bộ, không bắt buộc công khai.

**Giữ nguyên nội dung báo cáo lịch sử**, chỉ bổ sung chú thích trạng thái và sửa liên kết khi di chuyển. Không viết lại kết luận cũ để khớp hướng mới. Archive không chứa bản sao toàn bộ video, feature hoặc source code cũ vì chúng đã có trong backup/Git.

`ACTIVE_SPEAKER_CURATION_IMPLEMENTATION.md` vẫn có bài học áp dụng được. Đưa báo cáo triển khai cũ vào archive không có nghĩa bỏ kiểm tra người đang nói; phần hướng dẫn còn áp dụng sẽ được viết ngắn lại theo pipeline mới.

Hai tài liệu đang phục vụ hướng mới:

- `mmb.md`: đổi sang `docs/research/HUONG_NGHIEN_CUU_LIP_SYNC_VI.md` để có tên rõ nghĩa. Tại thời điểm kiểm tra, nội dung hướng đi nằm trong `mmb.md`; không có file `HUONG_NGHIEN_CUU_THAY_THE_LIP_SYNC_VI.md` ở root.
- `KE_HOACH_PILOT_LIP_SYNC_VI.md`: chuyển thành `docs/planning/KE_HOACH_PILOT_LIP_SYNC_VI.md` và cập nhật liên kết.

Nếu tìm lại được `LỘ TRÌNH NGHIÊN CỨU DEEPFAKE.md`, phân loại theo nội dung: lộ trình nghiên cứu cũ vào archive; nội dung đã được nhóm chọn cho hướng hiện hành mới đưa vào planning. File của thành viên là tài liệu tham khảo cho đến khi nhóm chốt, không tự trở thành kế hoạch chính.

Không duy trì nhiều bản hướng đi gần giống nhau ở root.

## 6. Phân vai rõ cho README, PROJECT và CLAUDE

| File | Nội dung nên giữ |
|---|---|
| `README.md` | Bài toán, phạm vi, cách bắt đầu, cấu trúc và đường dẫn tài liệu. |
| `PROJECT.md` | Trạng thái thực tế, quyết định đã chốt/chưa chốt, pipeline hiện hành, việc tiếp theo. |
| `CLAUDE.md` | Quy tắc làm việc với code/data, lệnh kiểm tra, môi trường và nguyên tắc chống leakage. |
| `docs/README.md` | Chỉ mục tài liệu hiện hành và một đường dẫn sang archive. |

Cần sửa cụ thể:

- Bỏ mô tả bốn pseudo-fake và AVSP-Net khỏi phần hướng hiện hành; trỏ sang archive nếu cần giải thích lịch sử.
- Bỏ các khẳng định quá mạnh như “nhãn chính xác tuyệt đối”, nén lại sẽ “xóa” dấu vết codec, hoặc model tiếng Anh mặc nhiên không dùng được cho tiếng Việt.
- Không gọi AV-HuBERT là model đã chốt hoặc đã hiệu quả khi chưa có pilot.
- Không chép đường dẫn Python cá nhân thành yêu cầu chung cho cả nhóm. Khi chuyển môi trường, ghi cách chọn interpreter và phiên bản cần thiết; không thay môi trường đang chạy một cách âm thầm.
- Tách quy tắc nghiên cứu còn áp dụng khỏi cạm bẫy của pipeline cũ.
- Chỉ đưa lệnh chạy đã được kiểm tra vào hướng dẫn khởi động. Module mới chưa triển khai phải được ghi rõ trạng thái.
- Không ghi “dataset CC” cho toàn bộ nguồn nếu chỉ một phần đã kiểm tra; không ghi “public dataset” khi chưa xác định quyền phát hành.

`PROJECT.md` mới nên có bảng trạng thái riêng cho: nguồn/quyền sử dụng, pilot data, generator, backbone, Y, X, evaluation và demo. Mỗi mục phân biệt “đề xuất”, “đã triển khai”, “đã chạy thử” và “đã có kết quả”.

`CLAUDE.md` giữ những quy tắc như: không trộn split, không ghi đè run cũ, không tự chạy job dài khi chưa được yêu cầu, không force-push, không đọc nhầm cache cũ. Các đường dẫn và lệnh Stage 01–05 chỉ còn trong hướng dẫn lịch sử khi code đã chuyển xong.

## 7. Sửa `.gitignore` và môi trường cùng đợt

`.gitignore` hiện chặn phần lớn `docs/` rồi mở ngoại lệ từng file. Nếu chỉ di chuyển report, tài liệu mới có thể bị bỏ qua ngoài ý muốn.

Nên chuyển sang:

- Theo dõi tài liệu Markdown, config, test và manifest nhỏ đã chọn, được phép chia sẻ.
- Bỏ qua media, weights, feature/cache và output nặng.
- Bỏ các ngoại lệ gắn với embedding/labels của dataset cũ.
- Nếu `CLAUDE.md` trở thành quy tắc chung cho nhóm, đưa nó vào Git; thông tin riêng từng máy để local.
- Giữ bí mật cấu hình ngoài Git; dùng `.env.example` chỉ chứa tên biến và giá trị mẫu.
- Bỏ qua toàn bộ thư mục bằng chứng quyền sử dụng nội bộ nếu có thư từ hoặc thông tin cá nhân; bản metadata công khai chỉ chứa thông tin phù hợp.

| Nên có trong Git | Để ngoài Git / bỏ qua |
|---|---|
| Source, test, config và tài liệu hiện hành | Video/audio gốc và các biến thể |
| Archive báo cáo và số đo nhỏ | Feature, embedding, preview/cache |
| Manifest đã rà soát quyền chia sẻ | Checkpoint, weights và môi trường cài đặt |
| Config, metrics, score theo ID nội bộ của run đã chọn | `.env`, token, cookie và hồ sơ cho phép riêng tư |

Không ignore toàn bộ `experiments/` nếu nhóm muốn lưu config và metrics trong Git. Chỉ bỏ qua phần nặng như checkpoint, log chi tiết quá lớn và media. Ngược lại, một file nhỏ cũng không tự động phù hợp để công khai nếu chứa thông tin riêng tư.

`requirements.txt` cũ nên lưu cùng môi trường lịch sử. Môi trường mới được dựng theo các thành phần thực sự dùng; không ép AV-HuBERT và mọi generator vào một môi trường ngay từ đầu.

Đề xuất `requirements.txt` cho phần code chính và `environments/` cho các môi trường phụ thực sự cần. Chỉ chốt phiên bản dependency sau khi kiểm tra cài đặt và chạy mẫu; tài liệu kế hoạch không tạo một bộ phiên bản giả định rồi coi là đã chạy được.

## 8. Cấu trúc đích và vai trò từng thư mục, từng file

Đây là cấu trúc đề xuất dưới `E:\FPTU\PRJ\VN-AV-DF-Capstone`. **Tên file mới thể hiện trách nhiệm dự kiến, không có nghĩa code đã tồn tại.** Chỉ tạo thư mục/module khi có bước triển khai dùng đến; không cần dựng trước toàn bộ cây rỗng.

Các thư mục có dạng `<dataset_version>`, `<run_id>`, `<generator>` hoặc `<config_id>` là chỗ điền tên thực tế. Không tạo nguyên văn tên có dấu `< >` trên Windows. Các file `__init__.py` sẽ được bổ sung tại package Python cần thiết khi triển khai, không liệt kê lặp lại trong cây.

### 8.1. Cây tổng thể

```text
VN-AV-DF-Capstone/
├── README.md
├── PROJECT.md
├── CLAUDE.md
├── KE_HOACH_DON_DEP_VA_TAI_CAU_TRUC_REPO.md
├── .gitignore
├── .env.example
├── requirements.txt
├── environments/
├── configs/
├── src/
│   ├── data/
│   ├── generators/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── tools/
├── tests/
├── data/
│   ├── sources/
│   ├── rights_evidence/
│   ├── manifests/
│   ├── raw/
│   ├── real/
│   ├── generated/
│   └── compressed/
├── cache/
├── weights/
├── experiments/
├── notebooks/                         # Chỉ giữ notebook đang dùng
└── docs/
    ├── README.md
    ├── research/
    ├── planning/
    ├── reports/
    └── archives/
        └── legacy_avsp/
```

`src/data/` là **code xử lý dữ liệu**. `data/` ở root là **dữ liệu và các bảng theo dõi**. `weights/` là trọng số tải sẵn; checkpoint detector do nhóm train nằm trong run tương ứng ở `experiments/`.

File kế hoạch này ở root theo yêu cầu hiện tại. Sau khi chuyển đổi hoàn tất, có thể chuyển nó vào `docs/planning/` như hồ sơ bàn giao, rồi cập nhật liên kết. Đó là bước sau, không thực hiện trong lượt tạo tài liệu.

### 8.2. File ở root và môi trường

| File/thư mục | Chứa gì, dùng để làm gì | Xử lý |
|---|---|---|
| `README.md` | Điểm vào cho thành viên mới; bài toán và cách chạy phần đã hoàn thành. | Viết lại từ file hiện có. |
| `PROJECT.md` | Trạng thái và quyết định hiện hành. | Lưu bản cũ vào archive rồi cập nhật. |
| `CLAUDE.md` | Quy ước chung cho AI/người sửa code. | Giữ quy tắc cần thiết, sửa phần gắn với pipeline cũ. |
| `.gitignore` | Quyết định loại file nào được theo dõi. | Sửa theo mục 7. |
| `.env.example` | Mẫu tên biến như API key và đường dẫn cục bộ; không chứa bí mật thật. | Tạo mới nếu cần. |
| `requirements.txt` | Dependency đã kiểm tra cho code chính. | Dựng lại có kiểm chứng, giữ bản cũ trong archive. |
| `environments/README.md` | Bản đồ component → môi trường; cách trao đổi file giữa các môi trường. | Tạo khi bắt đầu tích hợp model/generator. |
| `environments/avhubert.yml` | Môi trường trích đặc trưng AV-HuBERT nếu cần tách riêng. | Tạo sau khi chạy được ứng viên này. |
| `environments/wav2lip.yml`, `musetalk.yml`, `latentsync.yml` | Môi trường tương ứng từng generator được chọn. | Chỉ tạo cho generator thực sự tích hợp. |

Không xóa `.git`, `.agents`, `.codex`, `.vscode` hoặc `.env` theo một lệnh dọn chung. Đây là lịch sử/cấu hình làm việc, không phải dataset cũ. `.vscode` và `.env` vẫn có thể được giữ local và bỏ qua trong Git.

### 8.3. `configs/` — cấu hình chạy có thể lặp lại

Thư mục này chứa lựa chọn chạy, không chứa kết quả chạy hoặc API key.

| File dự kiến | Nội dung và tác dụng |
|---|---|
| `data.yaml` | Phiên bản dataset, manifest nguồn, tiêu chí clip, yêu cầu quyền sử dụng và đường dẫn đầu ra. |
| `protocol.yaml` | Quy tắc chia người/nguồn, seed, generator dùng cho train/validation và generator giữ riêng cho test. |
| `compression.yaml` | Codec, mức nén, chính sách audio và điều kiện dùng cho phát triển/test. |
| `features.yaml` | Backbone/checkpoint, preprocessing, cửa sổ audio–video và phiên bản cache. |
| `generators.yaml` | Đường dẫn weights, phiên bản và tham số gọi các generator; không dùng để train generator. |
| `baseline_y.yaml` | Cấu hình detector và huấn luyện baseline Y. |
| `method_x.yaml` | Cùng nền với Y, thêm cấu hình huấn luyện nhất quán qua nén; kiểm tra không lệch các yếu tố khác. |

Mỗi run lưu lại bản cấu hình đã giải quyết đầy đủ đường dẫn và giá trị vào thư mục run. Sửa config chung về sau không làm thay đổi bằng chứng của run cũ.

### 8.4. `src/` — code hiện hành của hướng mới

#### `src/data/` — từ nguồn video đến tập dữ liệu có thể train

| File dự kiến | Chức năng |
|---|---|
| `collect_sources.py` | Nhận danh sách video/playlist tuyển chọn, lập bảng nguồn và nhóm nội dung. |
| `check_licenses.py` | Ghi nhận giấy phép theo video, ngày kiểm tra và tình trạng hồ sơ; không tự suy ra quyền sử dụng pháp lý chỉ từ tên kênh. |
| `download.py` | Lấy video bằng cách phù hợp với nguồn/quyền truy cập, hỗ trợ tiếp tục và ghi lỗi. |
| `cut_clips.py` | Cắt clip từ nguồn, giữ ID và khoảng thời gian; tái sử dụng `cut_clips_core.py` khi phù hợp. |
| `quality_checks.py` | Kiểm tra mặt/miệng, cảnh và người đang nói; kết hợp kết quả review. |
| `media_checks.py` | Kiểm tra độ dài, audio/video, FPS, mốc thời gian và vùng hợp lệ dùng chung. |
| `build_manifest.py` | Tạo/kiểm tra các bảng nguồn, clip và biến thể; từ chối ID trùng hoặc thiếu quan hệ nguồn cần thiết. |
| `build_splits.py` | Gom người/nguồn liên quan, chia tập và kiểm tra rò rỉ; bỏ ràng buộc đủ bốn pseudo-fake. |
| `compress.py` | Tạo bản nén trực tiếp từ master cho cả real/fake theo cùng chính sách. |

Các bước ghi manifest đầu ra rõ ràng, không tự tìm một file “mới nhất” trong thư mục cũ. Quality check không dùng điểm detector để chọn fake dễ hoặc loại những clip detector đoán sai.

#### `src/generators/` — gọi model tạo lip-sync

| File dự kiến | Chức năng |
|---|---|
| `generate.py` | Đọc manifest/split, gọi generator phù hợp và ghi trạng thái cùng nguồn gốc output. |
| `wav2lip.py` | Phần kết nối với Wav2Lip: chuẩn bị input, gọi môi trường tương ứng, nhận output. |
| `musetalk.py` | Phần kết nối tương tự cho MuseTalk. |
| `latentsync.py` | Phần kết nối với ứng viên generator giữ riêng, nếu được chọn sau kiểm tra. |

Các file này là code tích hợp của nhóm, không phải bản sao toàn bộ repo tác giả. Mã và weights bên ngoài phải có phiên bản và giấy phép được ghi nhận. Bộ tạo dữ liệu tuân theo `protocol.yaml`, không đưa output của generator giữ riêng vào train/validation.

#### `src/features/` — chuẩn bị và trích đặc trưng hai luồng

| File dự kiến | Chức năng |
|---|---|
| `preprocess.py` | Crop miệng, chuẩn bị audio và cửa sổ đồng thời theo yêu cầu backbone. |
| `avhubert.py` | Nạp backbone ứng viên, cung cấp chế độ trích audio và hình ảnh riêng theo thiết kế. |
| `extract.py` | Chạy theo manifest, lưu đặc trưng, cache index và lỗi. |

Nếu AV-HuBERT không phù hợp sau pilot, thay phần tích hợp backbone và config; không làm mất các ID nguồn/split. Khi checkpoint hoặc preprocessing thay đổi, tạo cache mới.

#### `src/models/` — detector của nhóm

| File dự kiến | Chức năng |
|---|---|
| `fusion.py` | Kết hợp thông tin audio và hình ảnh theo thời gian. |
| `detector.py` | Phần tổng hợp theo thời gian và đầu phân loại; có chế độ chỉ audio, chỉ hình ảnh và cả hai để so sánh. |

Không tạo hai kiến trúc độc lập mang tên X và Y. Hai phương pháp dùng chung detector; khác biệt chính được đặt ở cách huấn luyện. Đầu ra là score cho cửa sổ/clip, chưa tự nhận là xác suất đã hiệu chuẩn hay vị trí chỉnh sửa chính xác.

#### `src/training/` — nạp dữ liệu và huấn luyện

| File dự kiến | Chức năng |
|---|---|
| `dataset.py` | Đọc manifest/cache đúng phiên bản, tạo batch và ghép các bản nén của cùng clip/cửa sổ. |
| `losses.py` | Mục tiêu phân loại và phần nhất quán dùng cho X. |
| `train.py` | Vòng train/validation, seed, lưu checkpoint, config, log và lựa chọn model. |

Không ghép real và fake thành một cặp phải có cùng dự đoán. Không mở final test để chọn checkpoint hoặc chỉnh mức nhất quán.

#### `src/evaluation/` — bằng chứng trả lời câu hỏi nghiên cứu

| File dự kiến | Chức năng |
|---|---|
| `evaluate.py` | Chạy model đã chọn, lưu score từng clip và trạng thái preprocessing. |
| `metrics.py` | AUC, F1, tỷ lệ báo nhầm và các phép tổng hợp cần dùng; ngưỡng được chọn trên validation. |
| `compare_runs.py` | So X/Y và các bản chỉ audio/chỉ hình ảnh trên cùng tập; xuất bảng generator × mức nén. |
| `check_shortcuts.py` | Kiểm tra metadata/đặc điểm phụ có vô tình tách được real/fake; tái sử dụng phần phù hợp của gate cũ. |

Báo số người, nguồn và clip gốc độc lập, không coi mọi bản nén là mẫu độc lập. Khi ước lượng độ bất định, nhóm các quan sát theo nguồn/người liên quan thay vì lấy ngẫu nhiên từng bản nén.

#### `src/tools/` — công cụ hỗ trợ con người

| File/thư mục dự kiến | Chức năng |
|---|---|
| `review/build_roi_preview.py` | Tạo preview có hình và audio phục vụ review. |
| `review/clip_review.py` | Giao diện review và ghi quyết định. |
| `review/build_review_assignments.py` | Chia batch cho ba người khi cần. |
| `review/merge_review_results.py` | Gộp review, phát hiện xung đột; chỉ giữ công cụ export/import khác nếu còn cần. |
| `demo.py` | Demo nhỏ dùng chung preprocessing và detector đã kiểm chứng; chỉ phát triển khi pipeline nghiên cứu ổn. |

Không ưu tiên dựng frontend/API lớn trong đợt dọn. Demo nhận clip đủ điều kiện, hiển thị score/nhãn theo ngưỡng và tình trạng xử lý; thiếu audio hoặc không thấy miệng phải báo không đủ điều kiện phân tích.

### 8.5. `data/` — nguồn, manifest và media

```text
data/
├── sources/
│   └── <dataset_version>/
│       ├── playlists.csv
│       ├── videos.csv
│       └── rights.csv
├── rights_evidence/
│   └── <source_id>/                  # Bằng chứng nội bộ, không mặc định đưa lên Git
├── manifests/
│   └── <dataset_version>/
│       ├── clips.csv
│       ├── reviews.csv
│       ├── speaker_links.csv
│       ├── splits.csv
│       ├── variants.csv
│       ├── failures.csv
│       └── dataset_info.json
├── raw/<dataset_version>/           # Video nguồn có audio
├── real/<dataset_version>/          # Clip real đã review, bản master
├── generated/<dataset_version>/<generator>/
└── compressed/<dataset_version>/<compression_id>/
```

| File/nhóm | Chứa gì và dùng để làm gì |
|---|---|
| `playlists.csv` | Playlist/chương trình được chọn, nhóm nội dung và lý do chọn; chưa phải danh sách video được phép dùng. |
| `videos.csv` | Video ID, URL, kênh/chương trình/tập, nhóm nguồn, trạng thái kiểm tra/tải. |
| `rights.csv` | Giấy phép theo video, ngày kiểm tra, căn cứ sử dụng, phạm vi đã xác định và mã hồ sơ bằng chứng; không lưu bí mật liên hệ ở bản công khai. |
| `rights_evidence/` | Bằng chứng giấy phép và văn bản cho phép nếu có. Phạm vi công khai của từng tài liệu được xử lý riêng. |
| `clips.csv` | Clip ID, source ID, thời gian bắt đầu/kết thúc, người nói, đường dẫn và trạng thái đủ điều kiện. |
| `reviews.csv` | Ai review clip nào, quyết định, lý do loại/giữ và trường hợp cần kiểm tra lại. |
| `speaker_links.csv` | Các clip/nguồn thuộc cùng người nói, liên kết host giữa nhiều tập, kết quả rà soát ID tự động. |
| `splits.csv` | Clip nguồn thuộc train/validation/test nào và nhóm người/nguồn nào; được khóa trước khi sinh fake. |
| `variants.csv` | Liên hệ mỗi fake/bản nén với clip cha, generator/checkpoint/cấu hình, nhãn và split kế thừa; thêm nguồn audio khi thay lời. |
| `failures.csv` | Những lần tải/cắt/generate/preprocess thất bại cùng lý do; dùng báo tỷ lệ hao hụt. |
| `dataset_info.json` | Phiên bản, hash manifest/config, ngày khóa split và thống kê dataset. |
| `raw/` | Video gốc lấy từ nguồn; giữ để truy vết/cắt lại trong vòng đời dataset mới. |
| `real/` | Clip gốc đã review, làm nguồn tạo fake và bản nén; không chứa output của dataset cũ. |
| `generated/` | Fake master trước các mức nén thí nghiệm; không lẫn với clip real. |
| `compressed/` | Bản nén của cả real và fake. Manifest chỉ rõ cha và điều kiện nén; không dựa vào tên folder để suy ra nhãn. |

Không chia train/test bằng cách di chuyển ngẫu nhiên media giữa folder. File split và quan hệ nguồn là căn cứ. Số lượng thực phải tách nguồn/người độc lập, master clip và số bản phái sinh.

### 8.6. `cache/` và `weights/` — dữ liệu máy có thể tạo lại hoặc tải lại

```text
cache/
├── previews/<dataset_version>/      # Preview review
├── preprocessing/<config_id>/       # Crop miệng, audio trung gian nếu cần lưu
└── features/<dataset_version>/<config_id>/
    ├── features_index.csv
    ├── cache_info.json
    └── <clip_or_variant_id>.pt

weights/
├── README.md
├── avhubert/                        # Checkpoint pretrained ứng viên
├── generators/<generator>/          # Weights generator đã chọn
└── face_detector/                   # Ví dụ yolov8n-face.pt nếu tiếp tục dùng
```

`cache_info.json` ghi backbone/checkpoint, preprocessing, phiên bản nguồn và cấu hình cửa sổ để ngăn đọc nhầm cache. Không chỉ kiểm tra `.pt` tồn tại rồi bỏ qua việc xác minh cấu hình.

`weights/README.md` ghi nguồn tải, phiên bản, giấy phép và hash các weights đã dùng. `yolov8n-face.pt` hiện ở root có thể chuyển vào `weights/face_detector/` nếu công cụ mới vẫn cần, đồng thời sửa đường dẫn nạp. Không đưa weights vào Git.

Xóa cache hợp lệ chỉ làm phát sinh chi phí tính lại; xóa manifest/weights không rõ nguồn có thể làm mất khả năng tái lập. Vì vậy không gộp chúng vào cùng một lệnh dọn tùy tiện.

### 8.7. `experiments/` — mỗi lần chạy là một hồ sơ riêng

```text
experiments/
└── <run_id>/
    ├── config.yaml
    ├── source_state.json
    ├── environment.json
    ├── dataset_snapshot.json
    ├── history.csv
    ├── predictions.csv
    ├── metrics.json
    ├── metrics_by_generator_compression.csv
    ├── failures.csv
    ├── train.log
    └── checkpoints/
        └── best.pt
```

| File | Tác dụng |
|---|---|
| `config.yaml` | Cấu hình thực sự dùng của Y/X hoặc baseline đơn luồng. |
| `source_state.json` | Commit và tình trạng source khi chạy; cần lưu bản thay đổi hoặc dấu vết đủ tái lập nếu run dùng code chưa commit. |
| `environment.json` | Phiên bản thư viện, phần cứng và checkpoint nền. |
| `dataset_snapshot.json` | Phiên bản/hash manifest, split và cấu hình feature; trỏ tới bản manifest được giữ. |
| `history.csv` | Diễn biến train/validation theo epoch hoặc bước. |
| `predictions.csv` | Score, nhãn và ID từng mẫu/cửa sổ để kiểm tra lại kết quả. |
| `metrics.json` | Kết quả tổng hợp, ngưỡng và tập dùng để chọn ngưỡng. |
| `metrics_by_generator_compression.csv` | Bảng chính để trả lời câu hỏi generator chưa thấy và khả năng chịu nén. |
| `failures.csv` | Mẫu không xử lý được, lý do và tỷ lệ ảnh hưởng đến coverage. |
| `train.log`, `checkpoints/best.pt` | Nhật ký chi tiết và detector đã chọn; checkpoint không commit vào Git. |

Không ghi đè run cũ khi đổi config/dataset. Bản sao manifest hoặc phiên bản trong Git phải còn truy cập được; chỉ giữ hash mà mất file gốc là chưa đủ tái lập.

Kết quả pilot AVSP-Net cũ được chuyển vào `docs/archives/legacy_avsp/experiments/`, không trộn với các run lip-sync mới.

### 8.8. `tests/` — kiểm tra những lỗi ảnh hưởng kết luận nghiên cứu

| File dự kiến | Trường hợp cần kiểm tra |
|---|---|
| `test_source_manifest.py` | Nguồn thiếu giấy phép/hồ sơ không bị tự đánh dấu là đã kiểm tra; ID/quan hệ clip hợp lệ. |
| `test_split_leakage.py` | Host chung nhiều tập, source trùng hoặc clip chồng lấn không vượt split. |
| `test_variant_provenance.py` | Fake/bản nén kế thừa đúng nguồn, audio và split; generator giữ riêng không lọt vào train. |
| `test_media_timeline.py` | Audio–video cùng cửa sổ hợp lệ, không lệch do cắt/nén hoặc padding sai. |
| `test_compression_pairs.py` | Cặp nhất quán cùng clip/cùng thời điểm; real và fake không bị ghép nhầm. |
| `test_feature_cache.py` | Cache khác checkpoint/preprocessing không được dùng nhầm; bản chỉ hình ảnh không chứa audio từ chế độ trích kết hợp. |
| `test_evaluation.py` | Tổng hợp đúng theo clip/generator/nén, xử lý mẫu lỗi và không chọn ngưỡng bằng final test. |
| Các test review/cắt clip được giữ | Bảo vệ chức năng cũ đã tận dụng; điều chỉnh import và đường dẫn sau khi chuyển. |

Đây là danh mục trách nhiệm test, không bắt buộc mỗi mục là một file riêng nếu gộp hợp lý hơn. Không viết test chỉ để xác nhận thư mục mới tồn tại. Test generator pseudo-fake cũ không chạy trong bộ test hiện hành sau khi code đó đã rời khỏi luồng chính.

### 8.9. `docs/` — nghiên cứu, kế hoạch, báo cáo và lịch sử

```text
docs/
├── README.md
├── research/
│   ├── HUONG_NGHIEN_CUU_LIP_SYNC_VI.md
│   ├── DATA_PROTOCOL.md
│   └── RELATED_WORK.md
├── planning/
│   ├── KE_HOACH_PILOT_LIP_SYNC_VI.md
│   └── TEAM_PLAN.md
├── reports/
│   ├── research/
│   │   ├── <date>_SOURCE_AUDIT.md
│   │   ├── <date>_PILOT_LIP_SYNC.md
│   │   └── <date>_XY_COMPARISON.md
│   └── capstone/
│       ├── RP1_PROJECT_INTRODUCTION.md
│       ├── RP2_PROJECT_MANAGEMENT.md
│       ├── RP3_EXISTING_SYSTEMS.md
│       ├── RP4_METHODOLOGY.md
│       ├── RP5_SYSTEM_DESIGN.md
│       ├── RP6_RESULTS_DISCUSSION.md
│       └── RP7_CONCLUSION.md
└── archives/
    └── legacy_avsp/
        ├── README.md
        ├── project_snapshot/
        ├── architecture/
        ├── reports/
        ├── logs/
        ├── manifests/
        ├── environment/
        ├── experiments/
        └── 2026_report_checkpoints/
```

| Thư mục/file | Chứa gì và tác dụng |
|---|---|
| `docs/README.md` | Chỉ mục; người mới tìm được hướng đi, protocol, kế hoạch và báo cáo có hiệu lực. |
| `research/HUONG_NGHIEN_CUU_LIP_SYNC_VI.md` | Chuyển từ `mmb.md`; lý do chọn bài toán, model, X/Y, phạm vi và giả thuyết. |
| `research/DATA_PROTOCOL.md` | Nguồn/quyền sử dụng, định nghĩa nhãn, review, người/nguồn, split, generator và nén. Là nơi chốt quy tắc thực nghiệm. |
| `research/RELATED_WORK.md` | Bảng nghiên cứu liên quan, model/dataset, phạm vi so sánh và nguồn trích dẫn. Nghiên cứu của thành viên được đặt ở vai trò tham khảo phù hợp. |
| `planning/KE_HOACH_PILOT_LIP_SYNC_VI.md` | Kế hoạch pilot đã có, cập nhật theo nguồn và policy thực tế. |
| `planning/TEAM_PLAN.md` | Ba đầu mối data/generator, model/training, evaluation/report/demo; người kiểm tra chéo và mốc đã biết. |
| `reports/research/` | Bằng chứng kỹ thuật theo đợt: kiểm tra nguồn, pilot, so X/Y; mỗi báo cáo dẫn tới dataset/run cụ thể. |
| `reports/capstone/` | Bản thảo báo cáo theo syllabus; tổng hợp bằng chứng đã có để nộp môn học. |
| `archives/legacy_avsp/` | Lịch sử nghiên cứu AVSP-Net/pseudo-fake, theo bảng di chuyển ở mục 5. |

Các tên RP1–RP7 là đề xuất tổ chức theo nội dung syllabus đã trao đổi. Mốc nộp và định dạng chính thức chưa chốt; nếu trường yêu cầu Word/PDF hoặc gộp RP6–RP7 thì dùng đúng yêu cầu đó. Không tạo bảy file trống rồi coi như đã có báo cáo.

Paper có thể tổng hợp từ `research/`, `reports/research/` và các run đã khóa. Chỉ mở thư mục bản thảo paper khi bắt đầu viết và chọn nơi nộp; chưa cần duy trì nhiều bộ tài liệu giống nhau.

`notebooks/` chỉ dành cho EDA hoặc phân tích lỗi thực sự đang dùng, ví dụ `source_eda.ipynb`. Không để notebook trở thành nơi duy nhất chứa logic cắt, chia split hoặc train mà source không có.

## 9. Thứ tự thực hiện và tiêu chí hoàn thành

### 9.1. Trình tự đề xuất

| Bước | Công việc | Bằng chứng hoàn thành |
|---|---|---|
| 1 | Kiểm tra backup, ghi nhận Git và file chưa commit. | Có đường dẫn backup đã xác minh, thống kê đối chiếu và các file mẫu mở được. |
| 2 | Đưa báo cáo cùng bằng chứng nhỏ vào archive. | Chỉ mục archive chỉ rõ trạng thái cũ; link báo cáo không hỏng. |
| 3 | Tách các công cụ nền cần giữ và kiểm tra chúng. | Danh sách module giữ/chuyển/thay; test hoặc chạy mẫu cho chức năng tái sử dụng. |
| 4 | Xóa dữ liệu/output cũ và loại code/test không còn thuộc hướng hiện hành. | Thực hiện theo danh sách đã kiểm kê; không còn mặc định đọc population cũ. |
| 5 | Tổ chức cấu trúc mới; cập nhật import, đường dẫn và config. | Những phần được chuyển chạy được; phần chưa viết được ghi rõ là chưa triển khai. |
| 6 | Viết lại README/PROJECT/CLAUDE, sửa `.gitignore` và môi trường. | Thành viên khác hiểu cách bắt đầu; file cần theo dõi không bị Git bỏ qua ngoài ý muốn. |
| 7 | Kiểm tra liên kết tài liệu, test phù hợp và chạy thử một luồng nhỏ. | Không còn import/path lỗi do di chuyển; kiểm tra nguồn → cắt/review → manifest/split chạy được trên mẫu phù hợp. |
| 8 | Bắt đầu pilot mới. | Sau khi tích hợp generator/backbone, có luồng real/fake → nén → feature → Y và số đo phát triển ban đầu. |

Trước bước 4 phải hoàn tất kiểm tra backup và bảo toàn phần sẽ tái sử dụng. Không xóa dữ liệu trước rồi mới tìm xem báo cáo hoặc công cụ nào phụ thuộc vào nó.

Đợt dọn repo và đợt xây model mới là hai mốc công việc. Đợt dọn có thể hoàn tất khi các công cụ giữ lại hoạt động, lịch sử được lưu rõ và tài liệu mô tả trung thực phần chưa triển khai. Không bắt buộc phải train xong X mới được coi là đã dọn xong.

### 9.2. Chia ba đầu mối khi thực hiện

| Đầu mối | Phần chính | Kiểm tra chéo |
|---|---|---|
| A — Data/generator | Kiểm kê dữ liệu, hồ sơ quyền sử dụng, công cụ collect/cut/review và tích hợp generator sau dọn. | C rà split/nguồn; B kiểm tra mẫu audio–video. |
| B — Model/training | Tách code model cũ, môi trường/feature mới, detector Y/X và training. | C chạy lại evaluation; A rà preprocessing mẫu. |
| C — Protocol/evaluation/tài liệu | Archive, quy tắc split/nén, metrics, test quan trọng, tài liệu và demo về sau. | A/B rà báo cáo và tái lập kết quả phụ trách. |

Đây là phân chia trách nhiệm, không có nghĩa ba người cùng di chuyển những file giống nhau. Đợt chuyển cấu trúc nên có một người điều phối đường dẫn/import và thời điểm hợp nhất thay đổi.

### 9.3. Checklist bàn giao sau dọn

- [ ] Backup đầy đủ đã được kiểm tra, gồm file chưa commit và file bị Git bỏ qua.
- [ ] Báo cáo cũ còn nguyên kết luận, được gom trong `docs/archives/legacy_avsp/`.
- [ ] README/PROJECT mô tả lip-sync audio–visual và trạng thái mới; không trình bày số đo AVSP-Net như kết quả mới.
- [ ] Bốn pseudo-fake không còn là yêu cầu bắt buộc của dataset/model hiện hành.
- [ ] Công cụ nền được tận dụng có đường dẫn mới đúng và kiểm tra chức năng phù hợp.
- [ ] Mục tiêu đa dạng nguồn và hồ sơ quyền sử dụng vẫn được giữ; không suy giấy phép từ tier/playlist.
- [ ] Pipeline không đọc ngầm manifest, split, calibration hoặc cache cũ.
- [ ] Git theo dõi đúng tài liệu/config/manifest được phép chia sẻ, bỏ qua media/weights/bí mật.
- [ ] Có tài liệu rõ cho môi trường thực sự chạy được; phần chưa tích hợp không có lệnh khởi động giả định.
- [ ] Một thành viên mới đọc README → PROJECT biết nhóm đang làm gì, chạy phần nào, và kết quả nào còn hiệu lực.

Archive giữ bằng chứng cũ; code và dữ liệu hiện hành không còn phụ thuộc vào archive để chạy. Mốc tiếp theo là pilot nguồn mới và baseline audio–visual, với quyền sử dụng, split và cách đánh giá được xác định từ đầu.
