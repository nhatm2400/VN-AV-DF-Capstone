import os
import re
from pathlib import Path

def get_project_root():
    """Trả về đường dẫn gốc của dự án (VN-AV-DF-Capstone)"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

def clean_tier2_temp_files():
    project_root = get_project_root()
    tier2_dir = os.path.join(project_root, 'data', 'raw', 'tier2')
    
    if not os.path.exists(tier2_dir):
        print(f"Thư mục {tier2_dir} không tồn tại.")
        return
    
    pattern = re.compile(r'\.f\d+\.(mp4|m4a|webm|mkv)$', re.IGNORECASE)
    deleted_count = 0
    
    for file in Path(tier2_dir).iterdir():
        if file.is_file() and pattern.search(file.name):
            print(f"Đang xóa: {file.name}")
            file.unlink()
            deleted_count += 1
    
    print(f"\nĐã xóa {deleted_count} file tạm (fragment).")
    print("Các file .mp4 hoàn chỉnh vẫn được giữ nguyên.")

if __name__ == "__main__":
    clean_tier2_temp_files()