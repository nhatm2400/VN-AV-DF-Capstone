# Baseline trước hotfix Stage 04 Cut Clips

Snapshot này bảo toàn provenance của tập `6.888 → 3.001` trước khi sửa lỗi
CUDA decode và chạy lại Cut Clips.

- Không chứa bản sao 22,4 GiB MP4; media cũ vẫn được giữ nguyên tại chỗ.
- `lineage_snapshot.json` lưu SHA-256 của manifest, measurement, manual result,
  assignment và source/config liên quan.
- `media_inventory` chỉ lưu path + kích thước, không hash toàn bộ media.
- `cut_logs/` là bản sao các CSV trong `data/01_collect/cut_clips/`, vì thư mục
  đó bị Git bỏ qua.

Tạo snapshot bằng:

```powershell
D:\Anaconda\envs\vn_av_df\python.exe src/tools/snapshot_cut_hotfix_baseline.py
```

Không dùng artifact này làm dữ liệu mới. Nó chỉ phục vụ đối chiếu và phục hồi
lineage nếu quá trình hotfix có bất trắc.
