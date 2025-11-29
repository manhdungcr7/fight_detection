#!/usr/bin/env python3
"""
Script inference đơn giản cho 1 video
Predict Fight (1) hoặc NonFight (0)
"""

import sys
import os
import torch
import numpy as np
import cv2
from pathlib import Path

# Add UniFormerV2 to path
sys.path.insert(0, str(Path(__file__).parent / 'UniFormerV2'))

from slowfast.config.defaults import get_cfg
from slowfast.models import build_model
from slowfast.utils.checkpoint import load_checkpoint


def load_trained_model(cfg_path, checkpoint_path):
    """
    Load trained CUE-Net model
    
    Args:
        cfg_path: Path to config.yaml
        checkpoint_path: Path to checkpoint file (.pyth)
    
    Returns:
        model, cfg
    """
    print("Loading config...")
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    
    # Override for inference
    cfg.NUM_GPUS = 1 if torch.cuda.is_available() else 0
    cfg.TEST.ENABLE = True
    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_path
    
    device = "GPU (CUDA)" if cfg.NUM_GPUS > 0 else "CPU"
    print(f"Device: {device}")
    
    print("Building model...")
    model = build_model(cfg)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print("⏳ This may take 30-60 seconds on CPU...")
    
    try:
        load_checkpoint(checkpoint_path, model)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Using fallback loading method")
        # Fallback: Load checkpoint manually for CPU compatibility
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("✅ Model loaded successfully (fallback method)!")
    
    # Set to eval mode
    model.eval()
    
    if cfg.NUM_GPUS:
        model = model.cuda()
    
    return model, cfg


def preprocess_video(video_path, cfg):
    """
    Preprocess video for inference
    
    Args:
        video_path: Path to video file
        cfg: Config object
    
    Returns:
        Tensor: Preprocessed video frames [1, C, T, H, W]
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Read all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"Could not read video: {video_path}")
    
    print(f"Video has {len(frames)} frames")
    
    # Sample NUM_FRAMES frames uniformly
    num_frames = cfg.DATA.NUM_FRAMES
    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    sampled_frames = [frames[i] for i in indices]
    
    # Resize to TEST_CROP_SIZE
    crop_size = cfg.DATA.TEST_CROP_SIZE
    resized_frames = []
    for frame in sampled_frames:
        h, w, _ = frame.shape
        # Center crop
        if h > w:
            top = (h - w) // 2
            frame = frame[top:top+w, :, :]
        else:
            left = (w - h) // 2
            frame = frame[:, left:left+h, :]
        
        # Resize
        frame = cv2.resize(frame, (crop_size, crop_size))
        resized_frames.append(frame)
    
    # Convert to tensor [T, H, W, C] -> [C, T, H, W]
    video_tensor = np.stack(resized_frames, axis=0)  # [T, H, W, C]
    video_tensor = torch.from_numpy(video_tensor).float()
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # [C, T, H, W]
    
    # Normalize (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    video_tensor = (video_tensor / 255.0 - mean) / std
    
    # Add batch dimension [1, C, T, H, W]
    video_tensor = video_tensor.unsqueeze(0)
    
    return video_tensor


def predict_video(model, video_tensor, cfg):
    """
    Predict Fight/NonFight for video
    
    Args:
        model: Trained model
        video_tensor: Preprocessed video [1, C, T, H, W]
        cfg: Config
    
    Returns:
        prediction (int): 0=NonFight, 1=Fight
        confidence (float): Confidence score [0-1]
    """
    with torch.no_grad():
        if cfg.NUM_GPUS:
            video_tensor = video_tensor.cuda()
        
        # SlowFast models expect list of tensors
        inputs = [video_tensor]
        
        # Forward pass
        logits = model(inputs)  # [1, 2]
        
        # Apply softmax
        probs = torch.softmax(logits, dim=-1)  # [1, 2]
        
        # Get prediction
        pred_class = torch.argmax(probs, dim=-1).item()  # 0 or 1
        confidence = probs[0, pred_class].item()
    
    return pred_class, confidence, probs[0].cpu().numpy()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CUE-Net Fight Detection Inference')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained checkpoint (.pyth)')
    parser.add_argument('--config', type=str, 
                        default='UniFormerV2/exp/RWF_exp/config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)
    
    print("=" * 60)
    print("CUE-NET FIGHT DETECTION - INFERENCE")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print()
    
    # Load model
    model, cfg = load_trained_model(args.config, args.checkpoint)
    
    # Preprocess video
    print("\nPreprocessing video...")
    video_tensor = preprocess_video(args.video, cfg)
    print(f"Video tensor shape: {video_tensor.shape}")
    
    # Predict
    print("\nRunning inference...")
    pred_class, confidence, probs = predict_video(model, video_tensor, cfg)
    
    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    
    class_names = {0: 'NonFight', 1: 'Fight'}
    print(f"Prediction: {class_names[pred_class]}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print()
    print("Class probabilities:")
    print(f"  NonFight: {probs[0] * 100:.2f}%")
    print(f"  Fight:    {probs[1] * 100:.2f}%")
    print("=" * 60)
    
    return pred_class


if __name__ == "__main__":
    main()
