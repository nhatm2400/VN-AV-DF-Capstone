import os
import glob
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from feature_extractor import FeatureExtractor
from fusion_model import PAMF_Fusion

def evaluate_model(model_path='checkpoints/pamf_poc_model.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("🔍 Đang nạp mô hình để đánh giá toàn diện...")
    model = PAMF_Fusion().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Lấy toàn bộ dữ liệu đã trích xuất
    feature_files = glob.glob('data/features/*.pt')
    if len(feature_files) == 0:
        print("❌ Không tìm thấy dữ liệu test.")
        return

    y_true = []
    y_pred = []
    
    print(f"⚙️ Đang chạy Inference trên {len(feature_files)} mẫu...")
    
    with torch.no_grad():
        for feat_file in feature_files:
            # 1. Xác định nhãn thật (Ground Truth)
            is_fake = 1 if 'fake' in feat_file else 0
            y_true.append(is_fake)
            
            # 2. Chạy mô hình dự đoán
            data = torch.load(feat_file, map_location=device)
            audio_feat = data['audio']
            visual_feat = data['visual']
            
            prediction, _ = model(audio_feat, visual_feat)
            score = prediction.item()
            
            # Nếu score >= 0.5 thì đoán là Fake (1), ngược lại là Real (0)
            predicted_label = 1 if score >= 0.5 else 0
            y_pred.append(predicted_label)

    # ==========================================
    # TÍNH TOÁN CÁC CHỈ SỐ METRICS BẰNG SKLEARN
    # ==========================================
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "="*40)
    print("📈 BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (EVALUATION)")
    print("="*40)
    print(f"✅ Accuracy (Độ chính xác) : {acc * 100:.2f}%")
    print(f"🎯 Precision (Độ chuẩn xác): {prec * 100:.2f}%")
    print(f"🕵️‍♂️ Recall (Độ bao phủ)    : {rec * 100:.2f}%")
    print(f"⚖️ F1-Score (Điểm cân bằng): {f1 * 100:.2f}%")
    print("="*40)

    # ==========================================
    # VẼ HÌNH CONFUSION MATRIX ĐỂ ĐƯA VÀO BÁO CÁO
    # ==========================================
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Đoán: REAL', 'Đoán: FAKE'], 
                yticklabels=['Thực tế: REAL', 'Thực tế: FAKE'])
    plt.title('Ma trận Nhầm lẫn (Confusion Matrix) - PAMF PoC')
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    
    # Lưu hình ảnh ra thư mục dự án
    img_path = 'confusion_matrix.png'
    plt.savefig(img_path)
    print(f"\n🖼️ Đã vẽ và lưu biểu đồ Confusion Matrix tại: {img_path}")
    print("👉 Hãy chèn hình ảnh này vào Slide thuyết trình của bạn!\n")

if __name__ == "__main__":
    evaluate_model()