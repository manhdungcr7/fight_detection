# 🥊 CUENet Fight Detection - Hướng Dẫn Tích Hợp API

## 📋 Tổng Quan

API này cho phép bạn tích hợp model **CUENet** vào ứng dụng Streamlit để phát hiện đánh nhau trong video.

### Model Performance
| Metric | Giá trị |
|--------|---------|
| **Accuracy** | 90.75% |
| **F1-Score** | 90.75% |
| **ROC-AUC** | 0.969 |

---

## 📦 Files Cần Thiết

### 1. Từ Google Drive (bạn sẽ nhận link):
```
cuenet_rwf2000_epoch51.pyth  (~2.5GB) - Model checkpoint
vit_l14_336.pth              (~1.7GB) - CLIP backbone weights
```

### 2. Từ Repository:
```
api/
  └── fight_detection_api.py   # API server code
UniFormerV2/                   # Model code (cả folder)
```

---

## 🚀 Cài Đặt

### Bước 1: Clone/Copy Files

```bash
# Copy toàn bộ folder UniFormerV2 và api/ vào project của bạn
your_project/
├── api/
│   └── fight_detection_api.py
├── UniFormerV2/
│   ├── exp/
│   │   └── RWF_exp/
│   │       └── config.yaml
│   ├── model_chkpts/
│   │   └── vit_l14_336.pth    # ⬅️ Download từ Drive
│   └── slowfast/
│       └── ...
├── models/
│   └── cuenet_rwf2000_epoch51.pyth  # ⬅️ Download từ Drive
└── your_streamlit_app.py
```

### Bước 2: Cài Dependencies

```bash
pip install fastapi uvicorn python-multipart
pip install torch torchvision  # Nếu chưa có
pip install opencv-python-headless numpy
pip install fvcore iopath yacs termcolor
pip install pytorchvideo timm einops

# Install UniFormerV2
cd UniFormerV2
pip install -e .
```

### Bước 3: Download Model Files từ Google Drive

1. Download `cuenet_rwf2000_epoch51.pyth` → đặt vào `models/`
2. Download `vit_l14_336.pth` → đặt vào `UniFormerV2/model_chkpts/`

---

## 🏃 Chạy API Server

### Option 1: Command Line

```bash
cd your_project/api
python fight_detection_api.py --host 0.0.0.0 --port 8000
```

### Option 2: Với Uvicorn

```bash
cd your_project/api
uvicorn fight_detection_api:app --host 0.0.0.0 --port 8000 --reload
```

### Kiểm tra API đang chạy:
- Mở browser: http://localhost:8000
- API docs: http://localhost:8000/docs

---

## 📡 API Endpoints

### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "message": "Model loaded and ready"
}
```

### 2. Predict Video

```http
POST /predict
Content-Type: multipart/form-data
```

**Request:** Upload video file (mp4, avi, mov, mkv, webm)

**Response:**
```json
{
  "success": true,
  "prediction": "Fight",
  "confidence": 95.32,
  "probabilities": {
    "NonFight": 4.68,
    "Fight": 95.32
  },
  "message": "Video classified as Fight with 95.32% confidence",
  "processing_time": 2.45
}
```

---

## 🎨 Tích Hợp Vào Streamlit

### Cách 1: Gọi API từ Streamlit

```python
import streamlit as st
import requests

st.title("🥊 Fight Detection Demo")

# Upload video
uploaded_file = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Hiển thị video
    st.video(uploaded_file)
    
    if st.button("🔍 Detect Fight"):
        with st.spinner("Analyzing video..."):
            # Gọi API
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(
                "http://localhost:8000/predict",
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Hiển thị kết quả
                col1, col2 = st.columns(2)
                
                with col1:
                    if result["prediction"] == "Fight":
                        st.error(f"🚨 FIGHT DETECTED!")
                    else:
                        st.success(f"✅ No Fight")
                
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.2f}%")
                
                # Progress bars
                st.write("**Probabilities:**")
                st.progress(result["probabilities"]["Fight"] / 100)
                st.caption(f"Fight: {result['probabilities']['Fight']:.2f}%")
                st.progress(result["probabilities"]["NonFight"] / 100)
                st.caption(f"NonFight: {result['probabilities']['NonFight']:.2f}%")
                
            else:
                st.error(f"Error: {response.text}")
```

### Cách 2: Import trực tiếp Model (không cần API server)

```python
import streamlit as st
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path("UniFormerV2/slowfast")))
sys.path.insert(0, str(Path("UniFormerV2")))

# Import model class
from api.fight_detection_api import FightDetectionModel

# Load model (cache để không load lại mỗi lần)
@st.cache_resource
def load_model():
    model = FightDetectionModel()
    model.load(
        checkpoint_path="models/cuenet_rwf2000_epoch51.pyth",
        config_path="UniFormerV2/exp/RWF_exp/config.yaml"
    )
    return model

model = load_model()

st.title("🥊 Fight Detection Demo")

uploaded_file = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save temp file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    st.video(uploaded_file)
    
    if st.button("🔍 Detect Fight"):
        with st.spinner("Analyzing..."):
            result = model.predict(tmp_path)
            
            if result["prediction"] == "Fight":
                st.error(f"🚨 FIGHT DETECTED! ({result['confidence']:.2f}%)")
            else:
                st.success(f"✅ No Fight ({result['confidence']:.2f}%)")
```

---

## ⚠️ Lưu Ý Quan Trọng

### 1. GPU vs CPU
- **Có GPU**: ~2-3 giây/video
- **Chỉ CPU**: ~30-60 giây/video (rất chậm!)

Kiểm tra GPU:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 2. Memory Requirements
- **GPU Memory**: ~8GB VRAM
- **RAM**: ~16GB recommended

### 3. Video Format
- Hỗ trợ: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- Khuyến nghị: `.mp4` (H.264 codec)

### 4. Xử lý lỗi thường gặp

**Lỗi: "CLIP weights not found"**
```bash
# Đảm bảo file vit_l14_336.pth nằm đúng vị trí
UniFormerV2/model_chkpts/vit_l14_336.pth
```

**Lỗi: "Model not loaded"**
```bash
# Kiểm tra checkpoint file
ls -lh models/cuenet_rwf2000_epoch51.pyth
```

**Lỗi: "CUDA out of memory"**
```python
# Giảm batch size hoặc dùng CPU
import torch
torch.cuda.empty_cache()
```

---

## 📁 Cấu Trúc Thư Mục Hoàn Chỉnh

```
your_project/
├── api/
│   └── fight_detection_api.py     # API server
├── models/
│   └── cuenet_rwf2000_epoch51.pyth  # Model checkpoint (từ Drive)
├── UniFormerV2/
│   ├── exp/
│   │   └── RWF_exp/
│   │       └── config.yaml
│   ├── model_chkpts/
│   │   └── vit_l14_336.pth        # CLIP weights (từ Drive)
│   ├── slowfast/
│   │   ├── config/
│   │   ├── models/
│   │   └── ...
│   └── setup.py
├── your_streamlit_app.py
└── requirements.txt
```

---

## 📞 Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra logs của API server
2. Đảm bảo tất cả dependencies đã cài đúng
3. Kiểm tra paths trong `fight_detection_api.py`

**Chúc bạn demo thành công! 🎉**
