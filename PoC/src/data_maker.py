import ffmpeg
import os
import glob

def create_pseudo_fake(input_video_path, output_fake_path, delay_seconds=0.3):
    """
    Tạo Pseudo-fake bằng kỹ thuật Input Offset.
    Cách này an toàn tuyệt đối, không làm đơ khung hình video.
    """
    try:
        # 1. Load luồng Video bình thường
        input_vid = ffmpeg.input(input_video_path)
        
        # 2. Load luồng Audio nhưng ép nó bị trễ (offset) đi một khoảng thời gian
        input_aud = ffmpeg.input(input_video_path, itsoffset=delay_seconds)
        
        # 3. Ghép Video gốc và Audio đã lệch pha lại với nhau
        # vcodec='copy': Giữ nguyên chất lượng ảnh để chạy cho lẹ
        # acodec='aac': Mã hóa lại âm thanh để đồng bộ chuẩn xác
        out = ffmpeg.output(
            input_vid.video, 
            input_aud.audio, 
            output_fake_path, 
            vcodec='libx264',
            acodec='aac',
            pix_fmt='yuv420p'
        )
        
        # Chạy lệnh
        ffmpeg.run(out, overwrite_output=True, quiet=True)
        print(f"✅ Đã tạo Pseudo-fake (chạy mượt): {os.path.basename(output_fake_path)} (Lệch {delay_seconds}s)")
        
    except ffmpeg.Error as e:
        # In ra lỗi chi tiết nếu có
        print(f"❌ Lỗi khi xử lý {input_video_path}: {e.stderr.decode('utf-8')}")

if __name__ == "__main__":
    # Tìm tất cả file mp4 trong thư mục raw
    raw_videos = glob.glob('data/raw/*.mp4')
    print(f"🔍 Tìm thấy {len(raw_videos)} video gốc.")
    
    # Đảm bảo thư mục pseudo_fake đã tồn tại
    os.makedirs('data/pseudo_fake', exist_ok=True)
    
    for real_path in raw_videos:
        basename = os.path.basename(real_path)
        fake_name = basename.replace('real', 'fake')
        fake_path = os.path.join('data/pseudo_fake', fake_name)
        
        # Tạo fake với độ lệch 0.4 giây (lệch rất rõ nhưng không bị đơ)
        create_pseudo_fake(real_path, fake_path, delay_seconds=0.4)
        
    print("🎉 Đã fix xong lỗi đơ hình! Bạn hãy mở thư mục pseudo_fake xem thử nhé.")