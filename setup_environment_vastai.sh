#!/bin/bash
# ========================================
# Script cài đặt môi trường CUE-Net cho Vast.ai (Linux)
# ========================================

echo "========================================"
echo "SETUP MÔITRƯỜNG PYTHON CHO CUE-NET"
echo "========================================"
echo ""

echo "[1/8] Tạo môi trường conda mới..."
conda create -n cuenet_env python=3.8 -y

echo ""
echo "[2/8] Kích hoạt môi trường..."
source activate cuenet_env

echo ""
echo "[3/8] Cài đặt PyTorch với CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "[4/8] Cài đặt PyAV và video processing libraries..."
conda install av -c conda-forge -y

echo ""
echo "[5/8] Cài đặt dependencies cơ bản..."
pip install decord opencv-python moviepy pytorchvideo einops timm simplejson psutil iopath tensorboard yacs

echo ""
echo "[6/8] Cài đặt fvcore..."
pip install fvcore

echo ""
echo "[7/8] Cài đặt YOLOv8..."
pip install ultralytics

echo ""
echo "[8/8] Build UniFormerV2..."
cd UniFormerV2
python setup.py build develop
cd ..

echo ""
echo "========================================"
echo "HOÀN THÀNH CÀI ĐẶT!"
echo "========================================"
echo ""
echo "Môi trường: cuenet_env"
echo "Để kích hoạt: conda activate cuenet_env"
echo ""
echo "QUAN TRỌNG: Sửa đường dẫn trong các file sau:"
echo "1. UniFormerV2/slowfast/models/uniformerv2_model.py - MODEL_PATH"
echo "2. UniFormerV2/exp/RWF_exp/run.sh - DATA paths"
echo ""
