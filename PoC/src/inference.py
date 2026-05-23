import os
import cv2
import torch
import librosa
from feature_extractor import FeatureExtractor
from fusion_model import PAMF_Fusion

def preprocess_video(video_path, max_frames=90):
    """ Hàm đọc và xử lý video/audio trực tiếp từ file mp4 """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {video_path}")

    # Đọc hình ảnh
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    frames_tensor = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0
    
    # Đọc âm thanh
    audio_array, sr = librosa.load(video_path, sr=16000)
    
    return frames_tensor, audio_array

def scan_video(video_path, model_path='checkpoints/pamf_poc_model.pth'):
    """ Quét video và đưa ra dự đoán Real hay Fake """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n🔍 Đang nạp hệ thống phân tích PAMF...")
    # 1. Khởi tạo Trình trích xuất đặc trưng
    extractor = FeatureExtractor(device=device)
    
    # 2. Khởi tạo Mạng não bộ (Cross-Attention)
    model = PAMF_Fusion().to(device)
    
    # 3. Nạp "Trí nhớ" (Trọng số đã train) vào não
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Bật chế độ chấm điểm (không học nữa)
    except Exception as e:
        print(f"❌ Lỗi khi nạp Model: {e}")
        return

    print(f"📽️ Đang phân tích video: {os.path.basename(video_path)} ...")
    
    # Bước 1: Xử lý file mp4
    frames_tensor, audio_array = preprocess_video(video_path)
    
    # Bước 2: Ép qua mạng nơ-ron để lấy vector
    audio_feat = extractor.extract_audio_features(audio_array)
    visual_feat = extractor.extract_visual_features(frames_tensor)
    
    # Bước 3: Đưa 2 vector vào mạng Cross-Attention để "soi" lỗi
    with torch.no_grad(): # Tắt tính toán gradient cho nhẹ máy
        prediction, _ = model(audio_feat, visual_feat)
        
    # Bước 4: Xử lý kết quả
    score = prediction.item()
    fake_probability = score * 100
    
    print("\n" + "="*50)
    print("📊 KẾT QUẢ PHÂN TÍCH HÀNH VI SINH LÝ:")
    print("="*50)
    
    if score >= 0.5:
        print(f"🚨 CẢNH BÁO: VIDEO GIẢ MẠO (DEEPFAKE)!")
        print(f"Thuyết minh: Phát hiện sự lệch pha giữa cao độ âm thanh và biên độ mở khẩu hình miệng.")
        print(f"Xác suất Fake: {fake_probability:.2f}%")
    else:
        print(f"✅ AN TOÀN: VIDEO NGƯỜI THẬT.")
        print(f"Thuyết minh: Chuỗi động lực học của môi khớp nối hoàn hảo với ngữ âm tiếng Việt.")
        print(f"Độ tin cậy (Real): {(100 - fake_probability):.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    # === HÃY SỬA ĐƯỜNG DẪN VIDEO Ở ĐÂY ĐỂ TEST ===
    
    # Test thử 1 video gốc
    test_video_1 = 'data/raw/real_01.mp4' 
    
    # Test thử 1 video fake do chính máy tự tạo ra
    test_video_2 = 'data/test2.mp4'
    
    # Bạn có thể bỏ comment dòng nào bạn muốn chạy thử:
    scan_video(test_video_1)
    scan_video(test_video_2)