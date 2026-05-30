import static_ffmpeg
import os

# Thêm đường dẫn ffmpeg vào biến môi trường PATH
# Hàm này sẽ tự động tải ffmpeg về nếu chưa có
static_ffmpeg.add_paths()

# Sau khi thêm, bạn có thể kiểm tra lại
os.system('ffmpeg -version')