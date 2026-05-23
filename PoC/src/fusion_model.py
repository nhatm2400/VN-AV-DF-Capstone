import torch
import torch.nn as nn

class PAMF_Fusion(nn.Module):
    def __init__(self, audio_dim=768, visual_dim=1280, embed_dim=512, num_heads=4):
        """
        Mạng hợp nhất PAMF
        - audio_dim: 768 (Kích thước mặc định của Wav2Vec2 Base)
        - visual_dim: 1280 (Kích thước mặc định của MobileNetV2)
        - embed_dim: 512 (Không gian chung để 2 nhánh giao tiếp)
        """
        super(PAMF_Fusion, self).__init__()
        
        # 1. Chiếu (Project) cả hai modality về cùng một không gian chiều (512)
        self.audio_norm = nn.LayerNorm(audio_dim)
        self.visual_norm = nn.LayerNorm(visual_dim)

        self.audio_proj = nn.Linear(audio_dim, embed_dim)
        self.visual_proj = nn.Linear(visual_dim, embed_dim)
        
        # 2. Lõi Cross-Attention
        # batch_first=True nghĩa là tensor có dạng [Batch, Time, Features]
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 3. Mạng phân loại MLP cuối cùng
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3), # Chống over-fitting
            nn.Linear(128, 1),
            nn.Sigmoid()     # Trả về xác suất 0 (Real) -> 1 (Fake)
        )

    def forward(self, audio_feat, visual_feat):
        # Input: 
        # audio_feat: [Batch, Time_A, 768]
        # visual_feat: [Batch, Time_V, 1280]
        audio_feat = self.audio_norm(audio_feat)
        visual_feat = self.visual_norm(visual_feat)
        # Bước 1: Ép chiều
        q = self.audio_proj(audio_feat)    # Query [B, Time_A, 512]
        k = self.visual_proj(visual_feat)  # Key   [B, Time_V, 512]
        v = self.visual_proj(visual_feat)  # Value [B, Time_V, 512]
        
        # Bước 2: Cross-Attention (Audio đi tìm Khẩu hình miệng tương ứng)
        attn_out, attn_weights = self.cross_attn(query=q, key=k, value=v) 
        # attn_out shape: [B, Time_A, 512]
        
        # Bước 3: Global Average Pooling theo chiều thời gian
        # Lấy trung bình toàn bộ chuỗi thời gian để ra 1 vector duy nhất đại diện cho video
        pooled_feat = torch.mean(attn_out, dim=1) # [B, 512]
        
        # Bước 4: Phân loại
        output = self.classifier(pooled_feat) # [B, 1]
        
        return output, attn_weights