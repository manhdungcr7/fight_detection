# 🥊 CUENet Fight Detection

> **CUE-Net: Violence Detection Video Analytics with Spatial Cropping, Enhanced UniformerV2 and Modified Efficient Additive Attention**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 90.75% |
| **Precision** | 90.83% |
| **Recall** | 90.75% |
| **F1-Score** | 90.75% |
| **ROC-AUC** | 0.969 |

Trained on **RWF-2000** dataset (Real World Fight).

---

## 🏗️ Architecture

**CUE-Net** là kiến trúc 3 module:

1. **C (Cropping)**: YOLOv8 spatial cropping - tập trung vào vùng có người
2. **U (UniFormerV2)**: Backbone kết hợp CNN + Self-Attention  
3. **E (Enhanced)**: MEAA (Modified Efficient Additive Attention) - giảm độ phức tạp từ O(L²) → O(L)

---

## 📁 Project Structure

```
fight_detection/
├── api/                          # 🆕 API cho web demo
│   ├── fight_detection_api.py    # FastAPI server
│   ├── API_INTEGRATION_GUIDE.md  # Hướng dẫn tích hợp
│   └── requirements.txt          # Dependencies
├── models/                       # Model checkpoints (download separately)
├── UniFormerV2/                  # Model code
│   ├── exp/RWF_exp/config.yaml   # Training config
│   ├── model_chkpts/             # CLIP weights (download separately)
│   └── slowfast/                 # Core framework
├── data_paths/                   # CSV files for dataset
├── batch_inference.py            # Batch prediction
├── inference_single_video.py     # Single video prediction
├── evaluate_validation.py        # Validation evaluation
├── run_cropping.py               # YOLOv8 spatial cropping
└── create_csv.py                 # Create dataset CSV
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/manhdungcr7/fight_detection.git
cd fight_detection
```

### 2. Install Dependencies

```bash
pip install torch torchvision
pip install opencv-python-headless numpy tqdm
pip install fvcore iopath yacs termcolor
pip install pytorchvideo timm einops

# Install UniFormerV2
cd UniFormerV2
pip install -e .
```

### 3. Download Model Weights

Download từ Google Drive và đặt vào đúng folder:

| File | Size | Location |
|------|------|----------|
| `cuenet_rwf2000_epoch51.pyth` | ~2.5GB | `models/` |
| `vit_l14_336.pth` | ~1.7GB | `UniFormerV2/model_chkpts/` |

### 4. Run Inference

```bash
# Single video
python inference_single_video.py --video path/to/video.mp4

# Batch inference
python batch_inference.py --input_dir path/to/videos --output_dir results/
```

---

## 🌐 API for Web Demo

### Start API Server

```bash
cd api
pip install -r requirements.txt
python fight_detection_api.py --port 8000
```

### API Endpoints

```http
GET  /health              # Health check
POST /predict             # Upload video and get prediction
```

### Example Response

```json
{
  "success": true,
  "prediction": "Fight",
  "confidence": 95.32,
  "probabilities": {
    "NonFight": 4.68,
    "Fight": 95.32
  },
  "processing_time": 2.45
}
```

### Streamlit Integration

```python
import requests

files = {"file": open("video.mp4", "rb")}
response = requests.post("http://localhost:8000/predict", files=files)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

See [API_INTEGRATION_GUIDE.md](api/API_INTEGRATION_GUIDE.md) for detailed instructions.

---

## 📦 Dataset

**RWF-2000** (Real World Fight):
- **Train**: 1600 videos (800 Fight + 800 NonFight)
- **Validation**: 400 videos (200 Fight + 200 NonFight)

Dataset không được bao gồm trong repo. Tải từ [official source](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection).

---

## 🔧 Training (Optional)

### On Kaggle (Free GPU)

1. Upload datasets lên Kaggle Datasets
2. Sử dụng notebook `cuenet-evaluation.ipynb`
3. Enable GPU T4 và Run All

### Local/Cloud

```bash
cd UniFormerV2
python tools/run_net.py --cfg exp/RWF_exp/config.yaml
```

---

## 📚 References

- **Paper**: [CUE-Net (CVPR 2024 Workshop)](https://openaccess.thecvf.com/content/CVPR2024W/ABAW/papers/Senadeera_CUE-Net_Violence_Detection_Video_Analytics_with_Spatial_Cropping_Enhanced_UniformerV2_CVPRW_2024_paper.pdf)
- **UniFormerV2**: [GitHub](https://github.com/OpenGVLab/UniFormerV2)
- **RWF-2000**: [Dataset](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👥 Contributors

- Model training & API development
- Based on CUE-Net architecture by Damith Senadeera et al.
