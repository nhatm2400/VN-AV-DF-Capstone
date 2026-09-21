# Data protocol — hướng lip-sync tiếng Việt

Trạng thái: quy tắc nền sau chuyển đổi, chưa khóa dataset nghiên cứu.

## Nguồn và phạm vi

Tuyển chọn podcast/bài nói/thuyết trình và nguồn bổ sung để đa dạng người/kênh/điều kiện quay. Nhóm nguồn (trường tier hiện có) tách khỏi giấy phép theo video. Tập từ nguồn giống VLR không mặc nhiên có CC. Giữ căn cứ sử dụng và phạm vi nghiên cứu, cách lấy dữ liệu, tạo biến thể, demo và phát hành media riêng trong hồ sơ.

API license snapshot chỉ ghi metadata, không phê duyệt pháp lý. Các bảng nguồn/rights nội bộ mặc định không đưa lên Git. Tập demo công khai cần phạm vi sử dụng rõ. Mục tiêu paper không tự yêu cầu phát hành toàn bộ video.

## Clip và review

Một cảnh liên tục, người đang nói thấy được miệng, có audio cùng đoạn, đủ chất lượng. Review không dựa trên score detector. Clip nén hoặc lệch tiếng kỹ thuật không tự mang nhãn AI chỉnh môi. Công cụ ROI hiện là hỗ trợ review, không phải pipeline AV-HuBERT đã chứng nhận.

Sau review cần clip_id, source_video, file_path, speaker_id và decision=keep. Ghi thời gian nguồn; program_id/episode_id và canonical_source_id khi có. Speaker ID tự động phải rà xuyên các tập; thiếu người nói không được tuyên bố speaker-disjoint.

## Split

Ngoại lệ cho kết quả sơ bộ ngày 18/09/2026: người dùng xác nhận 18 video hiện tại chỉ có một người nói. Chọn `source_disjoint_single_speaker`, cùng `speaker_id=spk_001` cho mọi clip, chia theo nhóm nguồn/bản đăng lại/tập đã định danh. Đây là đánh giá video mới của cùng người, không phải speaker-disjoint; tập test trong bộ này chỉ phục vụ đánh giá sơ bộ. Chưa xác minh trùng nội dung giữa các URL khi metadata bản gốc/tập để trống. Khi có thêm người, tạo phiên bản split mới theo quy tắc dưới đây; không dùng kết quả bộ một người để khẳng định tổng quát hóa sang người mới.

Nối clip qua cùng người, nguồn video, bản đăng lại và cặp chương trình/tập đã định danh. Cùng host có thể nối nhiều episode; không phá nhóm để đạt tỷ lệ mong muốn. Mỗi nhóm chỉ thuộc một split. Pilot dùng để chọn phương pháp là dữ liệu phát triển; final test phải có nguồn/người riêng chưa được dùng điều chỉnh.

Khóa split trên real trước khi sinh fake. Generator mới phải lưu source_clip, generator/checkpoint/config, nguồn audio và lỗi. Fake/bản nén/cửa sổ kế thừa split; audio thay lời không vượt split. Helper inherit_variant_split kiểm tra điều này, nhưng adapter generator chưa có.

## Model và thí nghiệm

X/Y cùng detector, train data, các bản nén và ngân sách. X thêm yêu cầu nhất quán cho cùng clip/cửa sổ ở hai chất lượng. Các bản chỉ audio/chỉ hình ảnh là kiểm tra giá trị từng luồng. Tách riêng kết quả từng generator × mức nén và tỷ lệ báo nhầm real.

Wav2Lip/MuseTalk và generator giữ riêng đang là ứng viên; phiên bản cuối được chốt sau smoke và kiểm tra điều kiện sử dụng. Không dùng generator giữ riêng hoặc final test để chọn model/ngưỡng. Nén trực tiếp từ master cho cả real/fake, giữ chính sách audio như nhau; không hứa xóa mọi dấu vết codec.

CLI compression hiện xác minh cấu trúc nhãn/provenance và media. Người dựng master manifest vẫn phải dùng split/variant helper để kiểm tra quan hệ; CLI không thay thế toàn bộ audit dataset. Không gọi nén thành công là dataset đã sạch hoặc model đã tổng quát hóa.
