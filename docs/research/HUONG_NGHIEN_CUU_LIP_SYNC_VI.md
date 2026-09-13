**Hướng nghiên cứu đề xuất: phát hiện lip-sync manipulation tiếng Việt bằng âm thanh và hình ảnh, với khả năng chịu nén tốt hơn**

Cập nhật ngày 12/09/2026. Đây là thiết kế nghiên cứu để nhóm triển khai, chưa phải phương pháp đã được huấn luyện hoặc chứng minh hiệu quả. Nghiên cứu của thành viên là một nguồn tham khảo; lựa chọn dưới đây được xây dựng từ mục tiêu Capstone, nguồn lực ba người và các công trình liên quan.

**1. Hướng đi được đề xuất**

Nhóm nghiên cứu một mô hình cùng sử dụng lời nói và chuỗi hình ảnh vùng miệng để phát hiện video đã được AI tạo hoặc chỉnh sửa môi theo âm thanh. Mô hình tận dụng kiến thức từ một model có sẵn, học cách kết hợp hai nguồn thông tin trên dữ liệu tiếng Việt, rồi được huấn luyện để giữ dự đoán ổn định qua các bản nén của cùng video.

Câu hỏi trung tâm giữ đúng hướng nhóm muốn bảo vệ: **Trong video một người nói tiếng Việt, phương pháp X có cải thiện khả năng phát hiện lip-sync manipulation so với baseline Y trên protocol Z hay không, với kết quả riêng cho generator chưa thấy và video bị nén?** Trong đó, baseline là mốc so sánh; protocol là quy trình chia dữ liệu, huấn luyện và đánh giá; generator là mô hình tạo video giả; detector là mô hình phát hiện.

| Thành phần | Lựa chọn cụ thể |
|---|---|
| X | Detector audio–visual dùng đặc trưng pretrained, kết hợp audio và hình ảnh theo đoạn, có huấn luyện nhất quán giữa các bản nén |
| Y | Chính detector đó, cùng dữ liệu và cách kết hợp, nhưng chỉ học nhãn thật/giả và không có yêu cầu nhất quán giữa các bản nén |
| Z | Tách người nói và nguồn video trước khi sinh fake; giữ riêng một generator cho test; báo kết quả từng generator, từng mức nén |
| Đóng góp chính | Kiểm chứng hiệu quả của cách huấn luyện chịu nén trên detector audio–visual tiếng Việt |
| Nghiên cứu hỗ trợ | Xác định audio có giúp ích không, phân tích tín hiệu và các trường hợp thất bại |

Đây là một hướng đủ cụ thể để triển khai và so sánh, nhưng vẫn cho phép kết luận rằng cải tiến không có lợi nếu dữ liệu cho thấy như vậy. Không cần tự phát minh mọi thành phần để có một nghiên cứu có giá trị.

**2. Vì sao chọn cả audio và hình ảnh?**

Video giả có thể có môi và tiếng khớp tốt. Vì vậy, model không được xây dựng trên quy tắc “khớp là thật, lệch là giả”. Hình ảnh cung cấp thông tin về vùng miệng và diễn biến chuyển động; audio cung cấp ngữ cảnh lời nói để model học cách đánh giá hình ảnh trong hoàn cảnh đó. Bộ phân loại nhận cả thông tin riêng của từng phía và thông tin khi đối chiếu chúng.

Audio trong một video bị sửa môi có thể hoàn toàn là giọng thật. Do đó, nhánh audio không bắt buộc phải phát hiện giọng AI. Vai trò chính của nó là giúp đánh giá quan hệ giữa lời nói và hình ảnh. Ví dụ, cùng một chuyển động miệng có thể hợp lý với một đoạn lời nói nhưng không hợp lý với đoạn khác. Đây là cơ sở để thử nghiệm, không phải khẳng định rằng mọi fake đều còn lỗi đủ dễ phát hiện.

Đối với fake đồng bộ rất tốt, model vẫn có đường sử dụng thông tin hình ảnh thay vì bị buộc phải tìm ra độ lệch. Nếu cả hình ảnh lẫn quan hệ audio–visual đều không còn tín hiệu hữu ích, việc thêm audio hoặc thêm cách huấn luyện không tự bảo đảm phát hiện được.

Trong tài liệu, pretrained nghĩa là đã được huấn luyện trước; checkpoint là bản trọng số model đã lưu; train là tập huấn luyện, validation là tập điều chỉnh, test là tập kiểm tra cuối.

**3. Model nền: chọn AV-HuBERT**

Tôi đề xuất bắt đầu bằng **AV-HuBERT Base pretrained trên LRS3 và VoxCeleb2 tiếng Anh**, chưa qua bước huấn luyện nhận dạng chữ. Đây là model đã học từ âm thanh và hình ảnh người nói. Nhóm dùng nó để biến hai đầu vào thành các đặc trưng phục vụ phát hiện, không dùng nó để xuất transcript. Bản Base được ưu tiên để thử nghiệm trong nguồn lực giới hạn; hiệu quả và bộ nhớ cần được đo tại máy chạy thực tế. [Repo tác giả](https://github.com/facebookresearch/av_hubert), [danh sách checkpoint](https://facebookresearch.github.io/av_hubert/).

Lựa chọn này có cơ sở từ các nghiên cứu sử dụng AV-HuBERT cho deepfake detection như AV-Lip-Sync+ và AVH-Align. Chúng cho thấy đây là một nền tảng có thể thử cho bài toán, không chứng minh thiết kế của nhóm sẽ thành công trên tiếng Việt. [AV-Lip-Sync+](https://arxiv.org/abs/2311.02733), [AVH-Align](https://github.com/bit-ml/AVH-Align).

Giai đoạn đầu giữ nguyên toàn bộ AV-HuBERT, chỉ huấn luyện phần kết hợp và phân loại nhỏ phía sau. Nhờ đó, có thể lưu đặc trưng một lần để dùng cho nhiều thí nghiệm. Không tải đồng thời nhiều backbone lớn hoặc train lại model nền từ đầu. Chỉ cân nhắc cập nhật một phần model nền sau khi có baseline ổn định và chứng minh đây là nút thắt.

Repo AV-HuBERT đã được lưu trữ và dùng các thư viện đời cũ; bước triển khai đầu tiên là tạo môi trường riêng, nạp checkpoint và chạy đúng preprocessing trên vài clip. Ghi lại phiên bản thư viện, checkpoint và cấu hình. Giấy phép AV-HuBERT giới hạn nghiên cứu phi thương mại và có các điều kiện sử dụng riêng; không mặc định checkpoint có giấy phép MIT. [Nguồn cài đặt](https://github.com/facebookresearch/av_hubert), [giấy phép](https://facebookresearch.github.io/av_hubert/).

**4. Model sẽ xử lý một video như thế nào?**

**Đầu vào:** video một người nói tiếng Việt, mặt và miệng quan sát được, có audio tương ứng. Phạm vi ban đầu là một cảnh liên tục. Video thiếu tiếng, che miệng nhiều hoặc không xác định được người đang nói được báo không đủ điều kiện, không tự gán là giả.

**Bước một — chuẩn bị hai luồng.** Tách audio và crop vùng miệng. Hai luồng phải giữ cùng mốc thời gian, cùng đoạn bắt đầu và kết thúc. Chuẩn hóa về 25 hình mỗi giây và audio một kênh 16 kHz theo yêu cầu của bộ trích đặc trưng. Việc biến waveform thành đầu vào cho AV-HuBERT và chuẩn hóa crop cần bám code tham khảo, không tự thay tùy ý.

**Bước hai — lấy đặc trưng audio và hình ảnh riêng.** Dùng cùng một checkpoint AV-HuBERT, chạy chế độ chỉ audio và chế độ chỉ hình ảnh để có hai chuỗi đặc trưng. Đây là hai lượt xử lý bằng một model nền, không phải hai model lớn được train độc lập. Code trích đặc trưng của AVH-Align có ví dụ thực hiện các chế độ này; dùng nó làm tài liệu kỹ thuật, rồi viết phần kết nối dữ liệu cho manifest của nhóm. [Code tham khảo](https://github.com/bit-ml/AVH-Align/blob/main/deepfake_feature_extraction.py).

**Bước ba — học cách đối chiếu theo đoạn.** Thêm một khối nhỏ cho phép hình ảnh tại mỗi thời điểm tham khảo audio cùng thời điểm và vùng lân cận ngắn. Điểm khởi đầu là khoảng 0,2 giây mỗi phía. Khối này học từ nhãn manipulation, không xuất luật “lệch quá mức này là fake”. Đồng thời, giữ thông tin hình ảnh gốc đi thẳng đến phần phân loại để không đánh mất dấu vết hữu ích khi audio và môi đã khớp.

**Bước bốn — tổng hợp diễn biến.** Một mạng xử lý chuỗi nhỏ xem thông tin đã kết hợp qua nhiều thời điểm, rồi đưa ra một score cho cửa sổ video. Nó nhận cả đặc trưng hình ảnh, audio và kết quả đối chiếu. Không rút toàn bộ audio/video thành hai con số từ đầu vì sẽ mất nhiều thông tin theo thời gian.

**Bước năm — trả kết quả cấp clip.** Khi clip dài, lấy trung bình score của các cửa sổ hợp lệ theo cách cố định. Demo hiển thị score, nhãn theo ngưỡng đã chọn và phần video đủ điều kiện phân tích. Score chưa mặc nhiên là xác suất đã hiệu chuẩn. Timeline cửa sổ có thể dùng để xem lại, nhưng chưa được gọi là vị trí chỉnh sửa chính xác nếu không có nhãn đoạn và đánh giá riêng.

**5. Phần nào học, phần nào giữ nguyên?**

| Thành phần | Giai đoạn đầu | Lý do |
|---|---|---|
| AV-HuBERT | Giữ nguyên | Tận dụng kiến thức có sẵn và tránh chi phí train backbone |
| Phần thu gọn đặc trưng hai luồng | Có học | Giảm kích thước xử lý phía sau |
| Khối đối chiếu audio–hình ảnh theo đoạn | Có học | Học quan hệ hữu ích cho nhãn manipulation |
| Mạng tổng hợp theo thời gian và phân loại | Có học | Chuyển đặc trưng thành quyết định cấp clip |
| Ngưỡng thật/giả | Chọn trên tập điều chỉnh | Không lựa ngưỡng bằng kết quả test |

Một cấu hình khởi đầu có thể thu mỗi luồng về 128 giá trị ở mỗi thời điểm, dùng một khối đối chiếu và hai lớp xử lý chuỗi nhỏ. Đây là lựa chọn triển khai đề xuất, không phải cấu hình đã tối ưu. Tập trung so sánh vài lựa chọn trên validation thay vì thêm nhiều nhánh cùng lúc.

Không đưa nhận diện thanh điệu, phân tích từng âm vị, xác minh danh tính giọng nói hoặc nhánh toàn khuôn mặt vào phiên bản chính. Audio đã có vai trò trong model; không cần thêm một bài toán giả giọng riêng để chứng minh rằng hệ thống có sử dụng audio.

**6. Cải tiến chính: học nhất quán qua nén**

Trong một lượt huấn luyện, cho X xem hai phiên bản của cùng cửa sổ video: bản tham chiếu ít suy giảm bổ sung và bản nén. X phải học đúng nhãn ở cả hai bản, đồng thời được khuyến khích không thay đổi dự đoán quá mạnh chỉ vì video bị nén. Ví dụ, một clip manipulated không trở thành thật sau khi chất lượng giảm.

Baseline Y cũng xem đúng hai bản đó, với cùng nhãn, số lượt học và model nền. Y chỉ không có yêu cầu giữ hai dự đoán nhất quán. So sánh này giúp tách tác dụng của cải tiến khỏi lợi ích đơn giản của việc thấy nhiều dữ liệu nén hơn. Không thay kiến trúc giữa X và Y rồi quy mọi cải thiện cho cách huấn luyện.

Cặp được dùng cho yêu cầu nhất quán luôn là hai bản của **cùng một clip và cùng thời điểm**. Không ép video thật và video fake sinh từ nó có cùng dự đoán. Trong thí nghiệm nén hình ảnh chính, audio được giữ cùng nội dung và cùng chất lượng encode để đo riêng ảnh hưởng nén video. Audio vẫn được detector sử dụng đầy đủ.

Mức khuyến khích nhất quán được chọn trên validation. Nếu ép quá mạnh, model có thể cùng dự đoán sai ở cả hai bản. Vì vậy, mục tiêu là tăng khả năng phân biệt thật/giả, không phải chỉ làm hai score giống nhau. Đây là kỹ thuật có tiền nhiệm; đóng góp dự kiến là áp dụng và kiểm chứng có kiểm soát trong bài toán của nhóm, không tuyên bố phát minh nguyên lý mới. [Nghiên cứu về consistency và khả năng tổng quát](https://openaccess.thecvf.com/content/CVPR2025/html/Kashiani_FreqDebias_Towards_Generalizable_Deepfake_Detection_via_Consistency-Driven_Frequency_Debiasing_CVPR_2025_paper.html).

**7. Chứng minh audio có đóng góp thật**

Chạy một bản chỉ nhìn hình ảnh, một bản chỉ nghe audio và bản dùng cả hai. Các bản phải dùng cùng nguồn dữ liệu, cùng preprocessing phù hợp và lịch đánh giá. Khi làm bản chỉ hình ảnh, tuyệt đối không dùng đặc trưng đã được trích từ chế độ có cả audio, vì như vậy vẫn còn thông tin audio bên trong.

Trên nhóm self-reconstruction, fake giữ chính audio của real. Với cách cân bằng và chia dữ liệu đúng, chỉ nghe audio không nên dễ dàng phân biệt hai lớp. Nếu bản chỉ audio lại cho kết quả rất cao, trước tiên kiểm tra khác biệt chất lượng audio, khoảng lặng, độ dài hoặc lỗi chuẩn bị dữ liệu; không vội gọi đó là thành công.

Kết quả chính để xác nhận giá trị của audio là bản audio–visual có lợi ích so với bản chỉ hình ảnh trên dữ liệu hợp lệ, chẳng hạn bắt thêm fake hoặc giảm báo nhầm. Có thể thử thay audio bằng audio không liên quan để xem model có nhạy với quan hệ hai luồng không, nhưng đây chỉ là chẩn đoán: audio bị thay tạo một loại input khác, nên việc score đổi không tự chứng minh accuracy trên dữ liệu thật tăng.

Nếu thêm audio không giúp, ghi rõ kết quả và phân tích nguyên nhân. Yêu cầu có hai đầu vào không thay thế được bằng chứng rằng model sử dụng cả hai một cách hữu ích.

**8. Dữ liệu tiếng Việt cần chuẩn bị**

Sau đợt chuyển đổi 13/09/2026, dữ liệu cũ đã được đưa ra khỏi workspace; công cụ cắt/review vẫn được giữ. Thu thập một tập mới có quyền sử dụng rõ, xem/nghe kiểm tra và tạo manifest có phiên bản. PROJECT.md ghi trạng thái mới; số đo và NO-GO của population cũ nằm trong archive.

Điều kiện đầu vào: một người đang nói, một cảnh liên tục, đủ sáng để thấy miệng, audio tương ứng, không có che khuất nghiêm trọng. Ưu tiên nguồn có quyền sử dụng phù hợp hoặc người tham gia đồng ý. Cố gắng đa dạng người nói, bối cảnh, tốc độ nói và nguồn quay; không để toàn bộ real từ một nguồn còn fake từ nguồn khác.

| Nhóm dữ liệu | Vai trò |
|---|---|
| Video thật đã review | Mẫu âm chính, đo báo nhầm |
| Video thật được nén lại | Mẫu âm, kiểm tra model có nhầm nén với giả mạo không |
| Video do generator sinh lại môi bằng chính audio gốc | Mẫu dương chính ban đầu, kiểm tra dấu vết lip-sync manipulation trong khi giữ nội dung ổn định |
| Video được sửa môi theo một câu nói khác | Tập bổ sung để kiểm tra tình huống thay lời, báo riêng |
| Video thật bị lệch audio có kiểm soát | Tập đối chứng, giữ nhãn không có AI chỉnh sửa môi; dùng kiểm tra model có chỉ bắt lệch tiếng hay không |

Khởi đầu với khoảng 60–100 clip thật từ ít nhất 12 người cho pilot. Sau khi pipeline chạy ổn, mục tiêu vừa phải là khoảng 600 clip thật từ 60 người, trung bình 10 clip/người. Đây là kế hoạch quy mô, chưa phải dữ liệu đã đủ điều kiện. Ưu tiên số người và nguồn độc lập hơn số bản phái sinh.

Có thể chia mục tiêu thành 36 người train, 12 người validation và 12 người test. Nếu một video nguồn chứa nhiều người làm các nhóm liên quan nhau, cần giữ cả nhóm nguồn/người trong cùng split, chấp nhận tỷ lệ thực tế khác kế hoạch.

Tiếng Việt ở đây là miền dữ liệu và đối tượng đánh giá. Không mặc định AV-HuBERT được học từ tiếng Anh sẽ thất bại, cũng không mặc định nó hiểu tốt tiếng Việt. Khả năng chuyển sang dữ liệu tiếng Việt là điều phải đo. Chỉ cần transcript nếu một thí nghiệm cụ thể cần đến, chưa phải điều kiện cho model chính.

**9. Tạo fake bằng những generator nào?**

Đề xuất bộ ba sau, sau khi kiểm tra chạy được và chốt đúng phiên bản:

| Generator | Vai trò dự kiến | Lý do chọn |
|---|---|---|
| Wav2Lip | Tạo fake cho train/validation và test đã thấy | Có code và weights phục vụ nghiên cứu, thuận tiện làm mốc |
| MuseTalk 1.5 | Tạo fake cho train/validation và test đã thấy | Bổ sung một cách sinh vùng mặt khác |
| LatentSync 1.6, hoặc một phiên bản phù hợp GPU được khóa từ đầu | Chỉ dùng cho final test của detector | Kiểm tra chuyển sang một generator không tham gia phát triển detector |

Nguồn chính thức: [Wav2Lip](https://github.com/Rudrabha/Wav2Lip), [MuseTalk](https://github.com/TMElyralab/MuseTalk), [LatentSync](https://github.com/bytedance/LatentSync). Các nguồn có công bố code/weights; chưa tải và benchmark chúng trong lần viết tài liệu này. Không coi Wav2Lip và Wav2Lip+GAN là hai họ generator độc lập để làm claim unseen mạnh hơn. Kiểm tra giấy phép code, weights và dữ liệu khi chuẩn bị triển khai.

Cách tạo mẫu chính: lấy video thật và chính audio của nó, yêu cầu generator sinh lại vùng miệng. Đây vẫn là video có manipulation, nhưng mới kiểm chứng tình huống tái tổng hợp môi. Nếu muốn kết luận về giả mạo thay lời, cần tập bổ sung dùng câu nói khác, ưu tiên cùng người nói, có duration phù hợp và nghe/xem kiểm tra riêng. Không tự kéo giãn hoặc lặp audio theo cách khiến fake có dấu vết dễ nhận ra.

Khi dùng audio từ clip khác, lưu cả nguồn audio và nguồn hình ảnh. Hai nguồn đều phải thuộc cùng split đã khóa. Không lấy giọng/câu trong test để tạo fake cho train. Giữ cùng chính sách xử lý audio cho real/fake; không tạo toàn bộ fake bằng giọng AI trong khi toàn bộ real dùng giọng thật, vì model có thể giải bài toán bằng audio mà không cần nhìn môi.

Không cần train lại các generator. Nhóm dùng weights có sẵn, đo chi phí tạo một clip và lưu phiên bản, tham số, trạng thái thành công/thất bại. Chỉ loại output hỏng theo tiêu chí kỹ thuật đã định nghĩa; không chọn fake vì detector bắt được. Những video nhìn khớp và tự nhiên cần được giữ để đánh giá độ khó, không bị loại khỏi test.

Với mục tiêu 600 real chia 360/120/120 nguồn clip, tạo fake từ hai generator đã thấy trên cả ba split và generator giữ riêng trên 120 clip test: có khoảng 1.320 fake và 600 real, tức 1.920 master clip. Ba mức chất lượng tạo 5.760 bản media. Các bản phái sinh này không phải 5.760 mẫu nguồn độc lập; số lượng trên chưa tính test thay lời và hao hụt.

**10. Protocol Z: cách chia và kiểm tra**

Chia người nói, video nguồn và các nguồn audio liên quan trước khi sinh fake, nén hoặc cắt cửa sổ. Mọi biến thể của một nguồn đi cùng split. Kiểm tra video trùng, các đoạn chồng lấn và nguồn tải lại từ nền tảng khác. ID người nói tự động cần được rà soát, không mặc nhiên là danh tính đúng tuyệt đối.

Train dùng real và fake từ Wav2Lip/MuseTalk. Validation dùng người và nguồn khác, nhưng vẫn chỉ hai generator này, để chọn model, cách huấn luyện và ngưỡng. Test chứa người/nguồn chưa gặp, có fake từ cả hai generator đã thấy và LatentSync giữ riêng. Nhờ vậy có thể so sánh các generator trên cùng nhóm video nguồn test.

Không dùng điểm detector trên generator giữ riêng để lựa chọn model hoặc chỉnh tham số. Có thể kiểm tra generator đó chạy được bằng vài nguồn phát triển tách riêng, nhưng phải ghi lại việc tiếp xúc và tách phần kiểm tra kỹ thuật khỏi lựa chọn model. Nếu dùng kết quả của nó để sửa detector, nó đã trở thành nguồn phát triển và cần một holdout khác cho claim unseen.

Định nghĩa “chưa thấy” phải nói rõ là chưa dùng trong phát triển detector của nhóm; đồng thời ghi lịch sử pretraining của checkpoint. Một generator giữ riêng chỉ cho bằng chứng về generator đó, không bảo đảm mọi generator tương lai. Khi đủ thời gian, có thể đổi generator giữ riêng để có thêm một phép kiểm tra độc lập.

**11. Nén video như thế nào để thí nghiệm có ý nghĩa?**

| Mức chất lượng | Cách dùng |
|---|---|
| Reference | Bản tham chiếu sau chuẩn hóa; không gọi là raw nếu nguồn vốn đã nén |
| H.264 CRF 23 | Bản nén bổ sung để train và validation |
| H.264 CRF 40 | Bản nén mạnh dành cho kiểm tra cuối |

CRF là tham số của bộ nén; trong cấu hình này số cao hơn tương ứng nén mạnh hơn. Giữ cùng độ phân giải, FPS và các tùy chọn encode giữa real/fake. Xuất từng mức trực tiếp từ cùng master, không nén nối từ CRF23 sang CRF40. Giữ audio cùng chính sách encode trong bảng thí nghiệm chính để không trộn tác động nén video với nén audio.

Nén toàn video trước khi crop miệng. Chạy preprocessing trên từng bản nén khi báo hiệu quả hệ thống thực tế. Nếu thêm thí nghiệm dùng crop cố định để chỉ đo detector, cần ghi rõ đó là điều kiện kiểm soát. Cùng codec đầu ra không bảo đảm xóa hết dấu vết xử lý trước đó; vẫn cần kiểm tra độ dài, khoảng lặng, bitrate, độ nét và tỷ lệ crop lỗi có khác nhau theo nhãn không.

Chốt cấu hình bằng reference/CRF23 validation, không dùng CRF40 final test để quyết định mức consistency. Báo toàn bộ bảng generator × chất lượng, không chỉ ô có kết quả tốt. Chưa thử nền tảng thật thì chỉ nói “chịu nén H.264 trong cấu hình đã kiểm tra”, không gọi là đã chứng minh hoạt động tốt trên TikTok hoặc Zalo.

**12. Những thí nghiệm cần có**

| Thí nghiệm | Câu hỏi cần trả lời |
|---|---|
| Chỉ hình ảnh | Có cần audio để đạt kết quả tốt hơn không? |
| Chỉ audio | Dữ liệu có dấu vết phụ khiến chỉ nghe đã đoán được nhãn không? |
| Audio + hình ảnh, ghép thông tin đơn giản | Mốc để đánh giá lợi ích của khối đối chiếu theo đoạn |
| Y: khối đối chiếu theo đoạn, học cả reference và bản nén | Baseline trực tiếp của X |
| X: đúng Y, thêm học nhất quán giữa hai bản | Cải tiến chính có hiệu quả không? |
| Một model nghiên cứu có sẵn, ưu tiên LipFD | Kết quả đứng ở đâu so với một phương pháp đã công bố? |

LipFD có code train/validate và liên kết weights; đây là mốc so sánh từ nghiên cứu đã công bố để thử tái lập, không hứa sẽ chạy được ngay trong môi trường hiện tại. Nếu chỉ đánh giá weights có sẵn mà không huấn luyện lại trên dữ liệu nhóm, phải ghi rõ điều kiện đó khác với X/Y. [Repo LipFD](https://github.com/AaronComo/LipFD).

Chạy các lựa chọn sơ bộ trên train/validation trước. Sau khi khóa thiết kế, chạy ít nhất ba lần với cách khởi tạo khác nhau cho X/Y để xem kết quả có ổn định. Không cần lặp toàn bộ model phụ với mọi cấu hình nếu nguồn lực không đủ.

Chỉ số chính có thể là AUC, hiểu đơn giản là mức model xếp video giả cao hơn video thật khi thay đổi ngưỡng; 0,5 gần mức ngẫu nhiên và 1 là phân biệt hoàn hảo trên tập được đo. So sánh chính X–Y đặt tại generator giữ riêng ở bản nén mạnh. Đồng thời phải báo các ô còn lại, tỷ lệ phát hiện fake, tỷ lệ báo nhầm real và số video bị từ chối.

Ngưỡng phân loại được chọn trên validation và giữ cố định cho test. Ước lượng độ chắc chắn của chênh lệch bằng cách lấy lại mẫu theo nhóm người/nguồn, giữ các bản phái sinh đi cùng nhau; không coi từng frame là một người dùng mới. Báo biến thiên giữa các lần train riêng. Nếu chênh lệch nhỏ hoặc không ổn định, chưa kết luận cải tiến hiệu quả.

**13. Cách triển khai trong repo hiện tại**

Tái sử dụng khâu thu thập, cắt clip, review có audio và quản lý nguồn của repo chính. Không sửa AVSP-Net V1 thành một model khác rồi mất khả năng đối chiếu. Dựng module thử nghiệm mới và manifest có phiên bản riêng; chỉ tích hợp sâu hơn sau khi pipeline nghiên cứu đã rõ. Không bắt nghiên cứu mới phải có đủ bốn pseudo-fake cũ, vì nhãn và generator đã khác.

| Phần cần viết | Nhiệm vụ và sản phẩm cụ thể |
|---|---|
| Chọn dữ liệu | Danh sách clip được review, người nói, nguồn và split |
| Tạo lip-sync fake | Một bộ gọi riêng cho mỗi generator; xuất media và log thành công/thất bại |
| Tạo các bản nén | Xuất đúng cùng policy cho real/fake, giữ quan hệ với master |
| Chuẩn bị AV-HuBERT | Crop miệng, xử lý audio và kiểm tra mốc thời gian |
| Lưu đặc trưng | Hai chuỗi audio/hình ảnh, thời gian hợp lệ và phiên bản checkpoint |
| Model Y/X | Cùng khối kết hợp và phân loại; có công tắc bật/tắt consistency |
| Train/evaluate | Chọn model/ngưỡng trên validation; xuất bảng riêng theo generator và chất lượng |
| Demo | Upload, kiểm tra đầu vào, chạy detector, hiển thị và xuất kết quả |

Mỗi dòng dữ liệu cần tối thiểu: ID clip, ID người nói, video nguồn, audio nguồn nếu có, nhãn, split, generator và phiên bản, mức nén, thời gian bắt đầu/kết thúc, đường dẫn và trạng thái chất lượng. Một lần chạy cần lưu cấu hình, phiên bản code, checkpoint, lịch sử train, ngưỡng và score từng clip để nhóm có thể kiểm tra lại.

Khởi đầu dùng cửa sổ 4 giây gồm cả audio và hình ảnh tương ứng. Chọn nhóm video đủ độ dài và báo tỷ lệ không đáp ứng; không recut toàn bộ dữ liệu chỉ để ép đủ 4 giây. Khi train, có thể chuẩn bị một vài cửa sổ cố định ở các vị trí khác nhau cho mỗi clip để cache; X/Y dùng đúng cùng danh sách. Khi test, quét cả clip bằng cửa sổ 4 giây, bước 2 giây, thêm cửa sổ cuối sát cuối clip nếu cần và loại cửa sổ trùng. Clip ngắn hơn được đánh giá riêng khi nhóm quyết định bổ sung cấu hình ngắn.

Chỉ xử lý phần audio/video thực sự tồn tại; nếu cần thêm chỗ trống cho batch thì phải đánh dấu để model không học từ chỗ trống. Không lấy nhãn giả của cả clip gán cho mọi cửa sổ trong video chỉ bị sửa một phần. Phiên bản đầu dùng clip được generator xử lý toàn đoạn; video chỉ bị chỉnh sửa một phần là bài toán bổ sung cần nhãn khác.

Với backbone giữ nguyên, cache phải được tạo bằng đúng cửa sổ đưa vào model, hoặc phải thống nhất rõ cách dùng ngữ cảnh toàn clip cho cả train/test. Mỗi bản nén phải đi qua trích đặc trưng thật. Có thể dùng lại đặc trưng audio khi waveform đầu vào hoàn toàn giống nhau và chế độ trích không nhận hình ảnh. Nếu cập nhật backbone, phải tạo lại cache tương ứng.

Đề xuất dùng Linux trên GPU thuê hoặc môi trường cloud có phiên bản thư viện cố định cho preprocessing nặng và trích đặc trưng; Windows vẫn có thể dùng quản lý dữ liệu và demo khi môi trường phù hợp. Bắt đầu đo trên một GPU khoảng 24 GB, đây là cấu hình dự trù chứ chưa phải yêu cầu đã benchmark. Giảm số clip xử lý đồng thời trước khi đổi sang model lớn hơn hoặc thuê nhiều GPU.

**14. Trình tự thực hiện để tránh mở rộng quá mức**

| Chặng | Công việc chính | Điều phải có khi kết thúc |
|---|---|---|
| 1. Dựng bản nhỏ | Nạp AV-HuBERT; tạo fake từ hai generator; chạy Y trên nhóm video nhỏ | Đặc trưng đúng, train chạy hết, score xuất được, đo thời gian/bộ nhớ |
| 2. Hoàn thiện dữ liệu | Review nhóm video chính, khóa split và policy nén | Manifest tái lập, thống kê nguồn/người, lý do loại |
| 3. Kiểm tra model | So chỉ audio, chỉ hình ảnh, ghép đơn giản và đối chiếu theo đoạn | Biết audio và phần kết hợp có đóng góp hay không |
| 4. Thêm cải tiến chính | Train Y/X trên cùng dữ liệu; chọn cấu hình bằng validation | So sánh riêng tác dụng consistency, chưa mở test |
| 5. Kiểm tra cuối | Khóa model/ngưỡng; chạy generator giữ riêng và nén mạnh | Bảng kết quả, mức báo nhầm, độ bất định và ví dụ lỗi |
| 6. Hoàn thiện demo/report | Tích hợp pipeline đã kiểm chứng | Demo đúng phạm vi và câu chuyện nghiên cứu dựa trên kết quả |

Ba người có thể chia đầu mối thành data/generator, model/training và evaluation/demo. Mỗi đầu mối có người khác kiểm tra chéo. Việc tìm tín hiệu, đọc paper và phân tích lỗi diễn ra trong các chặng này để hỗ trợ quyết định; nó không thay thế trọng tâm X–Y–Z.

Nếu backbone chưa chạy ổn, giải quyết môi trường và preprocessing trước, chưa tải hàng nghìn video. Nếu Y chạy ổn nhưng X không giúp, giữ kết quả đó và phân tích; không tự động bổ sung nhiều nhánh mới. Nếu chỉ còn một generator khả dụng, chưa có cơ sở cho claim generator chưa thấy. Chi phí thực phải được ước lượng từ tốc độ tạo fake, trích đặc trưng và train trên pilot, không lấy thời gian quảng cáo của model làm ngân sách.

**15. Kết quả nhóm cần bàn giao**

Sản phẩm cốt lõi gồm một tập đánh giá tiếng Việt có nguồn gốc và split rõ, detector Y/X có thể chạy lại, bảng so sánh chứng minh hoặc bác bỏ lợi ích của cải tiến và demo nhận video để phân tích. Nhóm cần trả lời được audio giúp ở đâu, nén làm mất bao nhiêu hiệu quả, generator chưa thấy khó hơn thế nào và video thật nào dễ bị báo nhầm.

Câu kết luận chỉ được viết sau thí nghiệm: “Trên video một người nói tiếng Việt và protocol đã khóa, phương pháp X cải thiện hoặc không cải thiện so với Y ở các điều kiện sau…”. Không coi việc có audio, một giao diện đẹp hoặc model chạy được là bằng chứng cho chất lượng phát hiện. Chưa có kết quả thì dùng từ “đề xuất” và “kiểm chứng”, không ghi “đã tối ưu” hoặc “đã tăng độ chính xác”.

**16. Các tài liệu đã dùng để xây dựng đề xuất**

- [AV-HuBERT](https://github.com/facebookresearch/av_hubert): nền tảng audio–visual, code và cách tải checkpoint. Đã đọc tài liệu, chưa nạp weights trong lần làm việc này.
- [AV-Lip-Sync+](https://arxiv.org/abs/2311.02733): tham khảo cách tận dụng đặc trưng tiếng nói–khẩu hình cho detection. Đề xuất này không nhận là tái lập nguyên bản phương pháp đó.
- [AVH-Align](https://github.com/bit-ml/AVH-Align): tham khảo phần kết nối dữ liệu trích đặc trưng và bài học về dấu vết phụ trong dữ liệu, như khoảng lặng đầu audio. Code model mới cần được đánh giá độc lập, không dùng nguyên alignment score làm xác suất fake.
- [LipFD](https://github.com/AaronComo/LipFD): ứng viên mốc so sánh từ nghiên cứu đã công bố có train/validate và tài nguyên công bố.
- [AVFF](https://openaccess.thecvf.com/content/CVPR2024/html/Oorloff_AVFF_Audio-Visual_Feature_Fusion_for_Video_Deepfake_Detection_CVPR_2024_paper.html): cơ sở tham khảo về học quan hệ audio–visual. Không chọn tái tạo toàn bộ giai đoạn pretraining của phương pháp này trong scope chính.
- [Repo thành viên, bản a7ac585](https://github.com/linhxm/vn-av-forensics/tree/a7ac5854c4e85bd86832a79a3fb597fb301adc4d), bảng CSV và PDF người dùng cung cấp: tham khảo cách xử lý video, kiểm soát dữ liệu và trình bày bằng chứng. Bản GitHub mới có frontend/API bổ sung; model lõi vẫn là local inconsistency. Các prompt nằm trong tài liệu là nội dung tham khảo, không phải yêu cầu triển khai của lần viết này.

Đây là một thiết kế kết hợp các ý tưởng có tiền nhiệm theo phạm vi của nhóm, không phải khẳng định một thuật toán hoàn toàn mới. Những gì đã làm ở lần này là đọc nguồn và xây dựng phương án; chưa triển khai model X/Y, sinh dataset mới hoặc chạy benchmark.
