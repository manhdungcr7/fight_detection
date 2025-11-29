#!/usr/bin/env python3
"""
CUENet Fight Detection API
===========================
API server cho việc phát hiện đánh nhau trong video.
Sử dụng FastAPI để tích hợp với Streamlit web demo.

Author: CUENet Team
Model: CUENet (UniFormerV2 + CLIP backbone)
Accuracy: 90.75% on RWF-2000 validation set
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import logging

import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths - CẦN ĐIỀU CHỈNH THEO MÁY CỦA BẠN
BASE_DIR = Path(__file__).parent.parent  # fight_detection_cuenet/
UNIFORMER_PATH = BASE_DIR / "UniFormerV2"
CONFIG_PATH = UNIFORMER_PATH / "exp" / "RWF_exp" / "config.yaml"
CHECKPOINT_PATH = BASE_DIR / "models" / "cuenet_rwf2000_epoch51.pyth"
CLIP_WEIGHTS_PATH = UNIFORMER_PATH / "model_chkpts" / "vit_l14_336.pth"

# Model settings
NUM_FRAMES = 64
IMG_SIZE = 336
CLASS_NAMES = ['NonFight', 'Fight']

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# SETUP PATHS
# ============================================================================

# Add UniFormerV2 to path
sys.path.insert(0, str(UNIFORMER_PATH / "slowfast"))
sys.path.insert(0, str(UNIFORMER_PATH))

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PredictionResponse(BaseModel):
    """Response model cho prediction"""
    success: bool
    prediction: str  # "Fight" hoặc "NonFight"
    confidence: float  # 0-100%
    probabilities: dict  # {"Fight": xx.xx, "NonFight": xx.xx}
    message: str
    processing_time: float  # seconds


class HealthResponse(BaseModel):
    """Response model cho health check"""
    status: str
    model_loaded: bool
    device: str
    message: str


# ============================================================================
# MODEL CLASS
# ============================================================================

class FightDetectionModel:
    """
    Fight Detection Model wrapper
    Sử dụng CUENet (UniFormerV2 + CLIP backbone)
    """
    
    def __init__(self):
        self.model = None
        self.cfg = None
        self.device = None
        self.is_loaded = False
        
    def load(self, checkpoint_path: str, config_path: str):
        """Load model từ checkpoint"""
        try:
            from slowfast.config.defaults import get_cfg
            from slowfast.models import build_model
            
            logger.info(f"Loading model from {checkpoint_path}")
            
            # Load config
            self.cfg = get_cfg()
            self.cfg.merge_from_file(str(config_path))
            
            # Override settings for inference
            self.cfg.NUM_GPUS = 1 if torch.cuda.is_available() else 0
            self.cfg.TEST.CHECKPOINT_FILE_PATH = str(checkpoint_path)
            self.cfg.TRAIN.ENABLE = False
            self.cfg.TEST.ENABLE = True
            
            # Build model
            self.model = build_model(self.cfg)
            self.model.eval()
            
            # Load checkpoint
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
            
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'], strict=False)
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            
            # Move to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            raise e
    
    def short_side_scale(self, frames: list, size: int) -> list:
        """Scale frames giữ aspect ratio"""
        height, width = frames[0].shape[:2]
        if height < width:
            new_height = size
            new_width = int(width * (size / height))
        else:
            new_width = size
            new_height = int(height * (size / width))
        
        scaled_frames = []
        for frame in frames:
            scaled = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            scaled_frames.append(scaled)
        
        return scaled_frames
    
    def center_crop(self, frames: list, crop_size: int) -> list:
        """Center crop frames"""
        height, width = frames[0].shape[:2]
        
        y_offset = (height - crop_size) // 2
        x_offset = (width - crop_size) // 2
        
        cropped_frames = []
        for frame in frames:
            cropped = frame[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]
            cropped_frames.append(cropped)
        
        return cropped_frames
    
    def preprocess_video(self, video_path: str) -> Optional[torch.Tensor]:
        """
        Preprocess video cho model
        Match với training preprocessing:
        1. Uniform temporal sampling
        2. Short-side scale (giữ aspect ratio)
        3. Center crop
        4. Normalize
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None
        
        # Uniform temporal sampling
        num_frames = self.cfg.DATA.NUM_FRAMES if self.cfg else NUM_FRAMES
        img_size = self.cfg.DATA.TEST_CROP_SIZE if self.cfg else IMG_SIZE
        
        if total_frames < num_frames:
            indices = list(range(total_frames)) + [total_frames-1] * (num_frames - total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        
        # Load frames
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        # Spatial preprocessing (match training!)
        frames = self.short_side_scale(frames, img_size)
        frames = self.center_crop(frames, img_size)
        
        # Convert to tensor
        frames = np.stack(frames, axis=0).astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2)  # (C, T, H, W)
        
        # Normalize
        mean = torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32).view(3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32).view(3, 1, 1, 1)
        frames = (frames - mean) / std
        
        return frames.unsqueeze(0)  # (1, C, T, H, W)
    
    def predict(self, video_path: str) -> dict:
        """
        Dự đoán video có đánh nhau hay không
        
        Returns:
            dict: {
                "prediction": "Fight" hoặc "NonFight",
                "confidence": float (0-100),
                "probabilities": {"Fight": xx.xx, "NonFight": xx.xx}
            }
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded!")
        
        # Preprocess
        video_tensor = self.preprocess_video(video_path)
        if video_tensor is None:
            raise ValueError("Cannot load or process video")
        
        # Predict
        video_tensor = video_tensor.to(self.device)
        
        with torch.inference_mode():
            preds = self.model([video_tensor])
            probs = torch.nn.functional.softmax(preds, dim=1)
        
        probs = probs.squeeze().cpu().numpy()
        pred_label = int(np.argmax(probs))
        
        return {
            "prediction": CLASS_NAMES[pred_label],
            "confidence": float(probs[pred_label] * 100),
            "probabilities": {
                "NonFight": float(probs[0] * 100),
                "Fight": float(probs[1] * 100)
            }
        }


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="CUENet Fight Detection API",
    description="API để phát hiện đánh nhau trong video sử dụng CUENet model",
    version="1.0.0"
)

# CORS middleware - cho phép Streamlit gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production, hãy giới hạn origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = FightDetectionModel()


@app.on_event("startup")
async def startup_event():
    """Load model khi server khởi động"""
    try:
        # Kiểm tra files tồn tại
        if not CONFIG_PATH.exists():
            logger.error(f"Config file not found: {CONFIG_PATH}")
            return
        
        if not CHECKPOINT_PATH.exists():
            logger.error(f"Checkpoint file not found: {CHECKPOINT_PATH}")
            logger.info("Please download the model checkpoint and place it in the 'models/' folder")
            return
        
        if not CLIP_WEIGHTS_PATH.exists():
            logger.warning(f"CLIP weights not found: {CLIP_WEIGHTS_PATH}")
            logger.info("Model may not work correctly without CLIP weights")
        
        # Load model
        model.load(str(CHECKPOINT_PATH), str(CONFIG_PATH))
        logger.info("API server ready!")
        
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="running",
        model_loaded=model.is_loaded,
        device=str(model.device) if model.device else "not initialized",
        message="CUENet Fight Detection API is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model.is_loaded else "unhealthy",
        model_loaded=model.is_loaded,
        device=str(model.device) if model.device else "not initialized",
        message="Model loaded and ready" if model.is_loaded else "Model not loaded"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_video(file: UploadFile = File(...)):
    """
    Dự đoán video có đánh nhau hay không
    
    - **file**: Video file (mp4, avi, mov, etc.)
    
    Returns prediction với confidence score
    """
    import time
    start_time = time.time()
    
    # Check model loaded
    if not model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )
    
    # Save uploaded file to temp location
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / file.filename
    
    try:
        # Save file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Predict
        result = model.predict(str(temp_path))
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            success=True,
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            message=f"Video classified as {result['prediction']} with {result['confidence']:.2f}% confidence",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
        
    finally:
        # Cleanup temp files
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/predict/url")
async def predict_from_url(video_url: str):
    """
    Dự đoán video từ URL (coming soon)
    """
    raise HTTPException(
        status_code=501,
        detail="URL prediction not implemented yet"
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CUENet Fight Detection API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🥊 CUENet Fight Detection API Server")
    print("=" * 60)
    print(f"📍 Starting server at http://{args.host}:{args.port}")
    print(f"📄 API docs at http://{args.host}:{args.port}/docs")
    print("=" * 60)
    
    uvicorn.run(
        "fight_detection_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
