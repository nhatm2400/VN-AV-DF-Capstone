import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from fusion_model import PAMF_Fusion

def train_poc():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PAMF_Fusion().to(device)
    criterion = nn.BCELoss() 
    
    # Tăng nhẹ Learning Rate, bỏ weight_decay để model chạy nhanh hơn
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    feature_files = glob.glob('data/features/*.pt')
    num_files = len(feature_files)
    print(f"🚀 Bắt đầu huấn luyện với {num_files} mẫu dữ liệu thật...")
    
    epochs = 50 
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # 1. Đưa zero_grad ra NGOÀI vòng lặp file (Chỉ reset 1 lần mỗi Epoch)
        optimizer.zero_grad()
        
        for feat_file in feature_files:
            data = torch.load(feat_file, map_location=device)
            audio_feat = data['audio']
            visual_feat = data['visual']
            
            is_fake = 1.0 if 'fake' in feat_file else 0.0
            label = torch.tensor([[is_fake]]).to(device)
            
            prediction, _ = model(audio_feat, visual_feat)
            
            # 2. Chia Loss cho tổng số file để tính trung bình
            loss = criterion(prediction, label) / num_files
            
            # 3. Cộng dồn Gradient của tất cả 20 file lại (Không cập nhật vội)
            loss.backward() 
            
            # Nhân ngược lại để hiển thị log cho đúng số
            total_loss += loss.item() * num_files 
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 4. CHỈ BƯỚC ĐI (UPDATE) 1 LẦN DUY NHẤT SAU KHI XEM HẾT 20 FILE
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss trung bình: {(total_loss/num_files):.4f}")

    print("🎉 Huấn luyện hoàn tất!")
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/pamf_poc_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"💾 Đã lưu trọng số mô hình an toàn tại: {save_path}")

if __name__ == "__main__":
    train_poc()