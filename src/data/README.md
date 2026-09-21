# Pipeline dữ liệu — mở file rồi Run

Chọn interpreter `vn_av_df` trong IDE một lần (máy hiện tại: `D:\Anaconda\envs\vn_av_df\python.exe`). Dùng **Run Python File**, chạy cả file. Các bước tự xác định root repo nên không cần paste lệnh/chuyển thư mục. Không dùng Run Selection cho các file này.

Sửa [preparation/settings.py](preparation/settings.py) khi đổi phiên bản, đợt tải/cắt hoặc tên reviewer. Mặc định `dataset_v1` là bộ dữ liệu chính thức đang xây dựng, chưa có nghĩa đã khóa chất lượng/split.

| Thứ tự | File mở để Run | Đầu vào → đầu ra |
|---|---|---|
| 01 | [01_collect.py](01_collect.py) | `data/sources/dataset_v1/videos.csv` → `selected_videos.csv`; chuẩn hóa URL/ID, lấy danh sách trong playlist đã chọn, bỏ video trùng. Chưa tải media. |
| 02 | [02_download.py](02_download.py) | `selected_videos.csv` → `data/raw/dataset_v1/download_001/*.mp4` và `download_results.csv`. Tải hình + tiếng, không yêu cầu API key/rights.csv. |
| 03 | [03_cut_clips.py](03_cut_clips.py) | Batch tải thành công → `data/real/dataset_v1/` với clip, accepted/rejected CSV và log. Số nguồn được đọc từ kết quả tải. Cần VAD/YOLO/FFmpeg. |
| 04 | [04_build_manifest.py](04_build_manifest.py) | Accepted CSV/media + nguồn đã chọn → `data/manifests/dataset_v1/clips.csv`, giữ nhóm nguồn/kênh/tập/bản gốc. |
| Review | [Các bước review](../tools/review/README.md) | Xem/nghe, phân công, gộp quyết định → `reviewed_clips.csv`. **Bổ sung speaker_id nhất quán xuyên tập trước bước 05.** |
| 05 | [05_build_splits.py](05_build_splits.py) | Clip đã review + người/nguồn → `real_splits.csv`, kiểm tra leakage. |
| Generator | [Wav2Lip 00/01/02](../generators/README.md) | Kiểm tra setup → chọn batch train → tạo real/fake candidates. Cần weights và review output; chưa tự nối sang bước 06. |
| 06 | [06_compress.py](06_compress.py) | Chỉ chạy khi có `masters.csv` real/fake đầy đủ nguồn/nhãn/split và đã điền CRFS trong file. Không tự nối từ bước 05. |

## Điền videos.csv

Mỗi dòng điền cột `url` bằng link YouTube. `video_id` có thể để trống, bước 01 tự lấy. Chấp nhận watch, youtu.be, shorts, live; link watch có `list=` vẫn chỉ là **một video**. Muốn lấy cả playlist, dùng `https://www.youtube.com/playlist?list=...`.

Các cột title/channel có thể để trống. `tier` là nhóm nguồn, ví dụ podcast, presentation, lecture; không phải giấy phép. `program_id`, `episode_id`, `canonical_source_id` dùng khi đã biết. Không điền chung một episode_id cho cả playlist gồm nhiều tập; không tự tạo speaker_id từ tên kênh.

Ví dụ một dòng video (không phải dữ liệu đã được kiểm tra chất lượng/giấy phép):

```csv
video_id,url,title,channel,program_id,episode_id,tier,canonical_source_id
,https://www.youtube.com/watch?v=JEgxmwvo7YY,,,,,podcast,
```

Bước 01 chỉ gọi mạng khi mở rộng playlist. Video riêng được chuẩn hóa ID/URL tại máy; title/channel sẽ để trống nếu bạn chưa điền. Playlist lỗi/rỗng sẽ báo lỗi. Các playlist có video trùng được gộp theo ID; metadata nhóm/tập mâu thuẫn phải sửa trước khi tiếp tục.

## Chạy lại và giới hạn

Chạy lại 02 sẽ tiếp tục trong cùng DOWNLOAD_RUN: bỏ qua MP4 đã có cả hình và tiếng, tiếp tục file .part, thử lại video lỗi và cập nhật download_results.csv sau từng video. download_sources.csv khóa danh sách nguồn của đợt tải; khi đổi danh sách, dùng DOWNLOAD_RUN mới. Không chạy đồng thời hai bước 02 vào cùng một thư mục.

Khi chỉ sửa danh sách đầu vào và chưa tải, có thể tự bỏ selected_videos.csv cũ rồi chạy lại 01. Với đợt cắt mới, đổi CUT_RUN; không đổi tên dataset chỉ vì thêm một đợt tải. Manifest clip/split đã tạo vẫn được bảo vệ, không tự ghi đè.

Bước 03 dừng nếu batch còn video failed/pending. Bước 02 có --limit 1 để thử một video chưa hoàn tất; chạy bình thường sẽ xử lý phần còn lại. Mỗi bước do bạn chủ động chạy, không tự mở job tiếp theo. Kiểm tra giấy phép vẫn có ở `src/data/preparation/check_licenses.py`, nhưng không nằm trong chuỗi bắt buộc.

Tải YouTube cần Node.js >= 22 trên PATH và yt-dlp[default] (gồm EJS); downloader bật Node tường minh. Python 3.10 hiện còn chạy được nhưng yt-dlp đã cảnh báo ngừng hỗ trợ trong tương lai; chưa thay môi trường Python của nhóm.

Ngoài cùng `src/data/` chỉ có các file Python có số để chạy. `preparation/` chứa toàn bộ code hỗ trợ, settings.py và nhánh quality; không cần chạy lần lượt các file trong đó. Các bước model/train/evaluate mới chưa triển khai; model AVSP-Net/bốn pseudo-fake cũ ở backup, không phải phần còn chạy của luồng này.


## Tham khảo khi cần chạy CLI riêng

## Nguồn và quyền sử dụng

Điền URL video/playlist vào `data/sources/dataset_v1/videos.csv`. Bước 01 chuẩn hóa sang selected_videos.csv, tự lấy video_id; tier là nhóm nguồn tùy chọn. Giữ program_id, episode_id và bản gốc của reupload để rà leakage. File mẫu ở configs/templates/videos.csv.

`python -m src.data.preparation.check_licenses --videos <videos.csv> --out <new-rights.csv>` cần API key local và gọi YouTube API. Công cụ ghi license metadata, các phạm vi sử dụng vẫn pending. Đây không phải phê duyệt pháp lý.

Kiểm tra giấy phép là bước riêng tùy chọn theo yêu cầu hiện tại. Downloader không bắt buộc rights.csv và không tự đánh dấu đã duyệt. Nếu chủ động truyền --rights, công cụ mới kiểm tra research_allowed=yes, acquisition_allowed=yes và evidence_ref như trước.

`python -m src.data.preparation.download --videos <selected_videos.csv> --out_dir <new-run> --dry_run` chỉ kiểm tra bảng, không dùng network. Bỏ --dry_run mới tải. download_results.csv ghi cả thành công/thất bại. Bước 03 có số yêu cầu batch tải thành công toàn bộ để tránh lặng lẽ bỏ video lỗi.

## Cắt và gộp

`python src/data/preparation/cut_clips.py --config <config.json>` dùng cutter đã giữ lại; xem `--help`. Input CSV cần filename, số dòng khớp expected_input_count. Bật GPU chỉ sau khi kiểm tra môi trường. Bước VAD/YOLO có dependency/model ngoài, chưa smoke full nguồn mới.

`python src/data/preparation/build_manifest.py --add podcast "<batch>/**/accepted_clips.csv" <media-root> --out <clips.csv>` gộp accepted manifests và kiểm tra 1:1 clip/media. Có thể truyền nhiều `--add`. Kết quả chưa phải đã được review hay đã có speaker_id.

## Review và split

Giai đoạn sơ bộ hiện tại: người dùng xác nhận toàn bộ video có cùng một người nói. File Run `05_build_splits.py` chọn `PROTOCOL = 'source_disjoint_single_speaker'` và `SINGLE_SPEAKER_ID = 'spk_001'`. Bấm Run để tạo `real_splits.csv`: mã người được bổ sung trong output, không sửa file review; cùng nguồn, bản đăng lại hoặc chương trình/tập đã định danh vẫn nằm chung tập. Cột `split_protocol` ghi rõ đây là đánh giá trên cùng người, chưa đánh giá người chưa thấy. Nguồn trùng nội dung nhưng khác ID cần điền `canonical_source_id` trước khi chia; mã video khác nhau chưa bảo đảm nội dung độc lập.

Khi bổ sung người nói, dùng manifest/phiên bản mới, điền danh tính thật sự nhất quán và đổi `PROTOCOL = 'speaker_source_disjoint'`. CLI mặc định vẫn dùng quy tắc nghiêm ngặt này. Không ghi đè split sơ bộ đã dùng cho thí nghiệm.

Xem [công cụ review](../tools/README.md). Rà clip và bổ sung `speaker_id` nhất quán xuyên nguồn; giữ cùng ID cho host ở các tập. `source_video` là ID nguồn, `canonical_source_id` liên kết bản đăng lại. `speaker_ids` có thể chứa nhiều ID cách nhau bằng dấu chấm phẩy nếu clip liên quan nhiều người.

`python -m src.data.preparation.build_splits --input <reviewed.csv> --out <new-splits.csv> --ratios 0.7,0.15,0.15` yêu cầu decision=keep, speaker/source xác định. Có thể dùng `0.8,0.2,0` cho pilot phát triển. Không ép tách nhóm chung host để đạt tỷ lệ; nếu không đủ nhóm, lệnh dừng.

Split output giữ nguyên metadata và thêm label=0, split, group_id. Generator về sau phải dùng `inherit_variant_split` để kiểm tra nguồn real/audio và generator giữ riêng; adapter chưa triển khai.

## Nén và thời gian

`python -m src.data.preparation.compress --input_csv <masters.csv> --out_dir <new-run> --crfs 18,23` là ví dụ kiểm tra kỹ thuật, không chốt mức nén cho nghiên cứu. Input master real/fake cần clip_id, file_path, label, speaker_id, source_video, group_id, split; fake cần source_clip. Không dùng lại bản đã nén làm master.

Output gồm config/hash input, variants.csv, summary.json và failures.csv khi có lỗi. Mỗi CRF tạo trực tiếp từ master bằng libx264, audio cùng chính sách AAC 16 kHz mono. Thư mục output đã tồn tại sẽ bị từ chối. Nén làm giảm một số khác biệt định dạng, không bảo đảm xóa toàn bộ dấu vết phụ.

Trong `preparation/`, `timeline_contract.py` giữ mốc thời gian và common valid window; `media_checks.py` giữ phép đo media dùng chung. `preparation/quality/` là công cụ đo/lọc tùy chọn, không bắt pilot tái chạy mọi bước curation lịch sử.
