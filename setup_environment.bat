@echo off
REM ========================================
REM Script cài đặt môi trường CUE-Net
REM ========================================

echo ========================================
echo SETUP MÔITRƯỜNG PYTHON CHO CUE-NET
echo ========================================
echo.

echo [1/8] Tạo môi trường conda mới...
call conda create -n cuenet_env python=3.8 -y
if errorlevel 1 (
    echo Lỗi tạo môi trường conda!
    pause
    exit /b 1
)

echo.
echo [2/8] Kích hoạt môi trường...
call conda activate cuenet_env

echo.
echo [3/8] Cài đặt PyTorch và torchvision...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo [4/8] Cài đặt PyAV và video processing libraries...
call conda install av -c conda-forge -y

echo.
echo [5/8] Cài đặt dependencies cơ bản...
pip install decord opencv-python moviepy pytorchvideo einops timm simplejson psutil iopath tensorboard yacs

echo.
echo [6/8] Cài đặt fvcore...
pip install fvcore

echo.
echo [7/8] Cài đặt YOLOv8 cho Spatial Cropping...
pip install ultralytics

echo.
echo [8/8] Build UniFormerV2...
cd UniFormerV2
python setup.py build develop
cd ..

echo.
echo ========================================
echo HOÀN THÀNH CÀI ĐẶT!
echo ========================================
echo.
echo Môi trường: cuenet_env
echo Để kích hoạt: conda activate cuenet_env
echo.
echo Bước tiếp theo:
echo 1. Tải CLIP pretrained weights vao model_chkpts/
echo 2. Chạy Spatial_Cropping_Module.ipynb để tạo RWF-2000-Cropped
echo 3. CSV files đã được tạo tại data_paths/
echo.
pause
