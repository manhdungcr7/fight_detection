#!/usr/bin/env python3
"""
Script tự động extract CLIP ViT-L/14@336px weights
Dùng cho CUE-Net training
"""

import os
import sys
import torch
from collections import OrderedDict

print("=" * 60)
print("EXTRACT CLIP WEIGHTS CHO CUE-NET")
print("=" * 60)
print()

# Kiểm tra CLIP đã cài chưa
try:
    import clip
    print("✓ CLIP module đã có sẵn")
except ImportError:
    print("! CLIP chưa được cài đặt")
    print("Đang cài đặt CLIP từ GitHub...")
    import subprocess
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', 
        'git+https://github.com/openai/CLIP.git'
    ])
    print("✓ CLIP đã được cài đặt")
    import clip

print()

# Tạo thư mục output
output_dir = 'e:/fight_detection_cuenet/UniFormerV2/model_chkpts'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'vit_l14_336.pth')

print(f"Output: {output_path}")
print()

# Load CLIP ViT-L/14@336px
print("Đang tải CLIP ViT-L/14@336px từ OpenAI...")
print("(File ~1.7GB - lần đầu sẽ tải về, lần sau dùng cache)")
print()

try:
    model, _ = clip.load("ViT-L/14@336px", device='cpu')
    print("✓ Model đã load xong")
except Exception as e:
    print(f"✗ Lỗi khi load model: {e}")
    sys.exit(1)

print()
print("Đang extract visual encoder...")

# Extract visual encoder (bỏ projection head)
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    if 'visual.' in k:
        # Bỏ projection layer và layer norm cuối
        if k[7:] not in ["proj", "ln_post.weight", "ln_post.bias"]:
            new_state_dict[k[7:]] = v

print(f"✓ Extracted {len(new_state_dict)} layers")
print()

# Save
print(f"Đang lưu vào {output_path}...")
torch.save(new_state_dict, output_path)

# Kiểm tra file
file_size = os.path.getsize(output_path) / (1024**3)  # GB
print()
print("=" * 60)
print("HOÀN THÀNH!")
print("=" * 60)
print(f"✓ File saved: {output_path}")
print(f"✓ File size: {file_size:.2f} GB")
print()
print("Bước tiếp theo:")
print("  python run_cropping.py")
print("=" * 60)
