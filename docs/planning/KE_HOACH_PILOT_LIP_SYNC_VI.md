# Kế hoạch bước tiếp theo: kiểm chứng hướng lip-sync tiếng Việt

Ngày lập: 12/09/2026. Đây là kế hoạch đề xuất, chưa phải kết quả chạy. Người dùng xác nhận nguồn mới vẫn là video YouTube có đủ audio và hình ảnh, nhưng chọn có chủ đích podcast/bài nói/thuyết trình thay cho lấy nguồn rộng. Ví dụ được cung cấp: [video YouTube JEgxmwvo7YY](https://www.youtube.com/watch?v=JEgxmwvo7YY). Công cụ web chưa mở được video này để kiểm tra trực tiếp nội dung; chất lượng cụ thể vẫn cần xem mẫu. Đây là danh sách nguồn tuyển chọn đang hình thành, chưa phải một dataset đã xác minh và sẵn sàng train.

Cập nhật vị trí sau đợt dọn 13/09/2026: hướng đi nằm trong [HUONG_NGHIEN_CUU_LIP_SYNC_VI.md](../research/HUONG_NGHIEN_CUU_LIP_SYNC_VI.md). Repo đã chuyển cấu trúc; pilot model vẫn chưa thực hiện.

## 1. Quyết định hiện tại

Ưu tiên thử nguồn mới trên một tập nhỏ. Dữ liệu cũ hiện nằm trong backup; chưa tải toàn bộ nguồn mới hoặc tiếp tục sinh bốn pseudo-fake. Nếu sau này lấy nguồn từ backup, phải kiểm tra quyền, chất lượng và người/nguồn theo protocol mới.

Thay đổi đầu tiên trong repo là đầu vào danh sách URL/chương trình/tập được chọn, không phải viết lại bộ tải YouTube. Tận dụng downloader và bộ cắt clip hiện có, kiểm tra cấu hình trên vài tập trước. Lưu lý do chọn nguồn để phạm vi nghiên cứu rõ ràng: video người nói tiếng Việt trong podcast/bài nói/thuyết trình, chưa đại diện cho mọi loại video YouTube.

Giữ câu hỏi X–Y–Z: detector audio–visual tiếng Việt, cải tiến huấn luyện nhất quán qua nén, so sánh công bằng với cùng detector không dùng cải tiến; kiểm tra riêng generator chưa thấy và mức nén. AV-HuBERT là ứng viên cần kiểm chứng trước khi chốt, không phải lựa chọn đã được chứng minh trên dữ liệu nhóm.

Sản phẩm của hai tuần đầu là một pipeline nhỏ chạy được, số đo sơ bộ và quyết định tiếp tục/điều chỉnh. Chưa đặt mục tiêu chứng minh đóng góp cuối cùng hoặc hoàn thiện demo.

## 2. Kiểm tra nguồn mới trước khi thay dữ liệu

Lấy thử khoảng 20 video/tập, trải trên nhiều chương trình và người nói; chọn các đoạn ở đầu, giữa, cuối phần có người nói. Nếu nguồn nhỏ hơn thì kiểm tra toàn bộ và ghi rõ giới hạn. Ghi thời gian thực tế tải, cắt, kiểm tra và sửa lỗi để ước lượng công sức.

Mỗi đoạn cần kiểm tra:

- Người đang phát ra lời nói xuất hiện trong hình, quan sát được miệng; không phải chỉ slide, ảnh bìa, cảnh minh họa hoặc người đang nghe.
- Một cảnh liên tục; không lẫn chuyển cảnh sang người khác trong cửa sổ được dùng.
- Audio khớp video gốc, tiếng Việt rõ, không phải video đã lồng tiếng hoặc nghi có nhân vật tổng hợp được đưa vào lớp real.
- Miệng không bị micro/đồ vật che nhiều; cỡ mặt và chất lượng đủ cho preprocessing.
- Có thông tin nguồn và điều kiện sử dụng; xác định được người nói và các bản đăng lại.

Ghi tỷ lệ thời lượng dùng được, phút review cho mỗi clip được giữ, số người độc lập, số chương trình/tập và lý do loại. Ước lượng tiết kiệm bằng cùng đơn vị với luồng cũ nếu đã có log tương đương. Nếu không có log cũ, chỉ báo chi phí luồng mới. Con số giảm 70% hiện là kỳ vọng, chưa là kết quả; nó cũng không đồng nghĩa giảm 70% công sinh fake, trích đặc trưng và huấn luyện.

Đặc biệt kiểm tra host xuất hiện ở nhiều tập. Khi yêu cầu tách cả người và tập nguồn, các tập có chung host có thể phải nằm cùng một nhóm. Nếu sau khi gom chỉ còn vài nhóm lớn thì cần bổ sung nguồn/người, không chia ngẫu nhiên clip để đạt tỷ lệ đẹp. Có thể ưu tiên các bài nói một người hoặc những nguồn tạo được nhóm độc lập ngay từ đầu.

## 3. Tập pilot và cách chia

Mục tiêu gợi ý: 120 clip real khoảng 4–8 giây, từ khoảng 24 người, khoảng 5 clip/người. Chỉ dùng cửa sổ audio–video hợp lệ 4 giây cho bản đầu. Đây là quy mô thăm dò, không phải lượng dữ liệu đủ để khẳng định tổng quát hóa. Có thể bắt đầu 60–100 clip nếu nguồn hạn chế, nhưng phải báo số người độc lập thực tế.

Chia trước khi sinh fake: khoảng 16 người train và 8 người validation phát triển. Giữ nguồn/tập, người và mọi biến thể trong cùng split. Nếu quan hệ nguồn/người buộc phải gom nhóm thì điều chỉnh tỷ lệ theo nhóm. Clip cùng tập hoặc chồng lấn thời gian không đi sang split khác.

Toàn bộ người/nguồn đã xem kết quả trong pilot được coi là dữ liệu phát triển. Khi làm nghiên cứu chính, đặt riêng người/nguồn chưa dùng để điều chỉnh vào final test. Không đổi tên validation pilot thành final test sau khi đã lựa chọn phương pháp bằng nó.

Manifest tối thiểu: clip_id, source_id, program_id, episode_id, speaker_id, start/end, đường dẫn, trạng thái review, split. Khi sinh biến thể, thêm parent_clip_id, label, generator, phiên bản checkpoint, cấu hình sinh, mức nén và trạng thái thành công/thất bại. Nếu dùng audio khác, phải có ID nguồn audio cùng split.

## 4. Lịch làm việc hai tuần

Các ngày dưới đây là 10 ngày làm việc dự kiến, cần chỉnh sau khi có nguồn và tốc độ GPU thực tế. Ba người thực hiện các phần độc lập đồng thời.

| Ngày | Đầu việc | Sản phẩm có thể kiểm tra |
|---|---|---|
| 1–2 | Kiểm tra mẫu nguồn mới; thống nhất định nghĩa real/fake và điều kiện clip; dựng môi trường model | Bảng chất lượng nguồn, manifest mẫu, checkpoint AV-HuBERT nạp được hoặc log lỗi cụ thể |
| 3–4 | Chọn/review pilot và khóa split; tạo fake thử 5–10 clip cho từng generator | Clip real/fake xem được, audio đúng, phiên bản và tốc độ chạy được ghi lại |
| 5–6 | Sinh fake cho pilot bằng hai generator; trích đặc trưng audio và hình ảnh | Manifest biến thể, danh sách lỗi, cache đặc trưng có phiên bản |
| 7–8 | Train bản chỉ hình ảnh, chỉ audio và kết hợp đơn giản | Score theo clip, AUC và lỗi trên validation tách người/nguồn |
| 9–10 | Rà lỗi dữ liệu và so sánh; nếu nền ổn thì thử Y/X nhỏ | Báo cáo pilot, chi phí mở rộng và quyết định giữ/điều chỉnh model nền |

Nếu môi trường hoặc dữ liệu chưa ổn, ngày 9–10 dành để giải quyết nút thắt và kết luận tình trạng, không bắt buộc phải có kết quả X/Y cho kịp lịch.

## 5. Tạo fake và nén ở mức nhỏ

Dùng weights có sẵn của Wav2Lip và MuseTalk, không train generator. Thử từng generator trên vài clip trước; sau khi đạt điều kiện kỹ thuật mới sinh hàng loạt trong pilot. Nguồn triển khai: [Wav2Lip](https://github.com/Rudrabha/Wav2Lip), [MuseTalk](https://github.com/TMElyralab/MuseTalk).

Mẫu chính ban đầu: generator tạo lại môi bằng chính audio gốc. Giữ waveform audio tương ứng giống nhau giữa real và fake sau chính sách chuẩn hóa, đồng thời kiểm tra mốc thời gian. Mỗi nguồn có đủ output hợp lệ thì tạo 1 real + 2 fake. Với 120 nguồn là tối đa 360 master; hai chất lượng là tối đa 720 bản media, không phải 720 nguồn độc lập. Ghi hao hụt theo generator, không giấu output lỗi hoặc chỉ giữ fake dễ bị phát hiện.

Không cần dùng LatentSync để xem điểm detector trong pilot; giữ nó cho kế hoạch generator chưa thấy. Nếu phải kiểm tra cài đặt, dùng nguồn kỹ thuật riêng và không sử dụng điểm detector trên đó để chọn model.

Ở pilot chỉ dùng bản tham chiếu và H.264 CRF23 theo hướng đi. Giữ audio cùng chính sách, nén cả real lẫn fake, tạo từng bản trực tiếp từ master và nén trước khi crop. CRF40 của final test chưa dùng để lựa chọn cấu hình. Video chỉ bị nén vẫn thuộc lớp không có AI sửa môi.

Tái tổng hợp môi chưa đại diện đầy đủ cho tình huống thay lời. Thử thay lời bằng audio khác cùng người là phần bổ sung báo riêng sau khi pipeline chính ổn, không cần mở thêm bài toán tạo giọng AI trong pilot.

## 6. Kiểm chứng AV-HuBERT và giá trị của audio

AV-HuBERT có code/checkpoint và hướng dẫn cài đặt riêng; tạo môi trường tách khỏi repo đang chạy vì hướng dẫn dùng Python 3.8/Fairseq. Trước tiên xác minh đúng checkpoint, preprocessing và chạy vài clip; xuất được tensor mới chỉ là kiểm tra kỹ thuật. [Nguồn tác giả](https://github.com/facebookresearch/av_hubert).

Giữ nguyên backbone ban đầu. Huấn luyện ba phần phân loại nhỏ: chỉ hình ảnh, chỉ audio, và kết hợp audio–hình ảnh đơn giản. Dùng chế độ trích riêng từng luồng để bản chỉ hình ảnh không chứa audio. Không dùng nhánh prosody hoặc đầu dự đoán độ lệch của AVSP-Net cũ làm yêu cầu bắt buộc.

So trên cùng tập nguồn, cùng điều kiện nén, ngân sách train tương đương và báo riêng hai generator. Khi một nguồn có hai fake dùng cùng audio real, cân bằng trọng số/lấy mẫu để số bản sao audio không trở thành khác biệt giữa nhãn. Chỉ audio phân biệt rất tốt trong tình huống này là lý do kiểm tra việc chuẩn bị dữ liệu trước tiên.

Đánh giá bằng AUC, tỷ lệ báo nhầm trên real và số clip preprocessing thất bại. Ngưỡng được chọn bằng validation phát triển; kết quả pilot không được coi là test cuối. Không loại các clip model đoán sai để làm đẹp số đo.

| Quan sát | Quyết định tiếp theo |
|---|---|
| Nạp model được nhưng crop/timeline sai | Sửa preprocessing trước khi đánh giá backbone |
| Train tốt, validation gần ngẫu nhiên hoặc rất dao động | Kiểm tra học thuộc nguồn/người và quy mô; chưa kết luận do tiếng Việt |
| Audio–visual có lợi ích ổn định so với chỉ hình ảnh | Tiếp tục thử khối kết hợp theo đoạn và Y/X; vẫn cần kiểm chứng trên tập lớn hơn |
| Audio–visual chưa tốt hơn chỉ hình ảnh | Rà cách kết hợp và dữ liệu; thử một thay đổi có kiểm soát, không tuyên bố audio hữu ích |
| Cần thích nghi backbone | Thử cập nhật một phần nhỏ trên train hoặc so nhánh audio tiếng Việt hiện có; chọn bằng validation, tạo lại cache khi weights đổi |
| Chi phí lớn hoặc không có tiến triển sau thời gian pilot | Thu hẹp cấu hình/đổi ứng viên dựa trên kết quả; giữ câu hỏi nghiên cứu, không cố bảo vệ AV-HuBERT |

Tập nhỏ không phân biệt rõ model kém, ít dữ liệu và lệch miền. Không đặt một ngưỡng AUC tùy ý rồi coi vượt ngưỡng là đã chứng minh model dùng tốt cho tiếng Việt. Lặp lại cấu hình triển vọng để xem biến thiên và phân tích lỗi trước khi chốt.

## 7. Khi nào triển khai cải tiến chính X/Y?

Sau khi có pipeline sạch và baseline hoạt động có ý nghĩa trên validation, triển khai phần đối chiếu audio–hình ảnh theo đoạn để có Y. X là chính Y cộng yêu cầu giữ dự đoán nhất quán giữa hai chất lượng của cùng cửa sổ. Cả X/Y đều học hai bản với cùng nhãn, cùng dữ liệu và số bước cập nhật; không để chỉ X thấy dữ liệu nén.

Khi kết quả phát triển đủ rõ, mở rộng tập chính, khóa model/tham số/ngưỡng và lập biên bản danh sách final test. Sau đó mới chạy generator giữ riêng và nén mạnh; báo cả bảng generator × chất lượng. Nếu X không cải thiện, đó là kết quả cần giải thích, không tự đổi tập test để tìm chiến thắng.

## 8. Chia ba đầu mối và tận dụng repo

| Người phụ trách | Trách nhiệm chính trong pilot | Kiểm tra chéo |
|---|---|---|
| A — Data/generator | Mẫu nguồn, review, manifest, sinh fake, log thời gian và thất bại | C kiểm tra split/nguồn; B xem preprocessing mẫu |
| B — Model | Môi trường AV-HuBERT, cache, các baseline và Y/X khi đủ điều kiện | C chạy lại đánh giá; A kiểm tra audio/video |
| C — Protocol/evaluation | Quy tắc chia dữ liệu, nén, metrics, score từng clip và báo cáo pilot | A rà media; B kiểm tra cách so sánh model |

Tận dụng công cụ cắt clip, preview có audio, manifest và kiểm tra media hiện có khi phù hợp. Nguồn mới sạch hơn có thể dùng lựa chọn thủ công cho pilot, không bắt phải xử lý hết quần thể cũ hoặc bê nguyên toàn bộ bước curation. Tiêu chí chất lượng và nguồn gốc vẫn phải được kiểm tra trên từng clip pilot.

Dựng manifest và module thí nghiệm mới có tên/phiên bản riêng. Không dùng nguyên gate yêu cầu đủ bốn pseudo-fake, không trộn cache AVSP-Net vào AV-HuBERT. NO-GO của quần thể dữ liệu cũ được giữ trong archive; quyết định cho pilot mới phải dựa trên bản review của chính tập mới.

Cuối pilot cần có: manifest/split có thể tái lập; real/fake/nén mẫu; cấu hình và phiên bản model; score và bảng kết quả; thống kê lỗi; thời gian/GPU/storage đo được; một trang giải thích có tiếp tục nguồn mới và AV-HuBERT hay không. Việc dựng giao diện hoàn chỉnh có thể chờ đến sau quyết định này.
