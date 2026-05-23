import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class FeatureExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # --- Khởi tạo Audio Backbone (Wav2Vec2) ---
        # --- Khởi tạo Audio Backbone (Wav2Vec2) ---
        print("Loading Wav2Vec2...")
        self.audio_processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
        self.audio_model = Wav2Vec2Model.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h").to(self.device)
        self.audio_model.eval() # Đóng băng, chỉ trích xuất
        
        # --- Khởi tạo Visual Backbone (MobileNetV2 siêu nhẹ) ---
        print("Loading MobileNetV2...")
        net = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # Bỏ lớp classifier cuối cùng, chỉ lấy features
        self.visual_model = net.features.to(self.device)
        self.visual_model.eval()
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Ép ảnh thành vector 1D

    def extract_audio_features(self, audio_array, sr=16000):
        """ Đầu vào là mảng numpy âm thanh 16kHz """
        with torch.no_grad():
            inputs = self.audio_processor(audio_array, sampling_rate=sr, return_tensors="pt").to(self.device)
            outputs = self.audio_model(**inputs)
            # Lấy hidden states cuối cùng: [1, Time_A, 768]
            return outputs.last_hidden_state 

    def extract_visual_features(self, video_frames_tensor):
        """ Đầu vào là tensor ảnh miệng [Time_V, 3, 224, 224] """
        with torch.no_grad():
            video_frames_tensor = video_frames_tensor.to(self.device)
            features = self.visual_model(video_frames_tensor) # [Time_V, 1280, 1, 1]
            features = self.pool(features).squeeze(-1).squeeze(-1) # [Time_V, 1280]
            # Thêm chiều batch: [1, Time_V, 1280]
            return features.unsqueeze(0)