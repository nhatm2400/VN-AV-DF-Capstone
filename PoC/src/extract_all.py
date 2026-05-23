import os
import glob
import cv2
import torch
import librosa
from feature_extractor import FeatureExtractor

def load_video_and_audio(video_path, max_frames=90):
    """ Đọc hình ảnh và âm thanh từ file mp4 """
    # 1. Đọc hình ảnh bằng OpenCV
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (224, 224)) # Đưa về chuẩn MobileNet
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    # Chuyển thành Tensor [Time, Channels, Height, Width]
    frames_tensor = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0
    
    # 2. Đọc âm thanh bằng librosa
    audio_array, sr = librosa.load(video_path, sr=16000) # Đưa về chuẩn 16kHz của Wav2Vec2
    
    return frames_tensor, audio_array

if __name__ == "__main__":
    extractor = FeatureExtractor()
    os.makedirs('data/features', exist_ok=True)
    
    # Gộp chung cả video real và fake để xử lý
    all_videos = glob.glob('data/raw/*.mp4') + glob.glob('data/pseudo_fake/*.mp4')
    
    for vid_path in all_videos:
        print(f"⏳ Đang xử lý: {vid_path}")
        try:
            frames, audio = load_video_and_audio(vid_path)
            
            # Ép qua mạng neural để lấy vector đặc trưng
            visual_feat = extractor.extract_visual_features(frames)  # [1, Time_V, 1280]
            audio_feat = extractor.extract_audio_features(audio)    # [1, Time_A, 768]
            
            # Lưu thành file .pt
            basename = os.path.basename(vid_path).replace('.mp4', '.pt')
            save_path = os.path.join('data/features', basename)
            
            # Đóng gói cả 2 loại tensor vào 1 file từ điển (dictionary)
            torch.save({'audio': audio_feat, 'visual': visual_feat}, save_path)
            print(f"✅ Đã lưu đặc trưng tại: {save_path}")
        except Exception as e:
            print(f"❌ Lỗi file {vid_path}: {e}")