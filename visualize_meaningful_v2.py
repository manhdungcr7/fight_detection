"""
CUE-Net Meaningful Visualization - Deep Feature Analysis
=========================================================
Theo lời khuyên của Gemini Pro về Explainable AI (XAI):

1. KHÔNG dùng Conv1 - chỉ thấy edges, không có ý nghĩa về bạo lực
2. Lấy Feature Map từ Local UniBlock CUỐI CÙNG (resblocks[-1])
3. Ba phương pháp visualization:
   - Average Feature Map: Đơn giản nhất, trung bình theo channel
   - Eigen-CAM: Dùng PCA, không cần gradient, rất nhẹ
   - Temporal Importance: Xác định segment nào quan trọng nhất

Tất cả đều chạy được trên GPU 4GB!
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# =============================================================================
# PATHS
# =============================================================================
CUENET_PATH = r"E:\fight_detection_cuenet\CUENet\Enhanced_Uniformer_V2"
UNIFORMERV2_PATH = r"E:\fight_detection_cuenet\UniFormerV2"
MODEL_CHECKPOINT = r"E:\fight_detection_cuenet\models\cuenet_rwf2000_epoch51.pyth"
CONFIG_PATH = r"E:\fight_detection_cuenet\UniFormerV2\exp\RWF_exp\config.yaml"
OUTPUT_DIR = r"E:\fight_detection_cuenet\visualizations"

sys.path.insert(0, CUENET_PATH)
sys.path.insert(0, UNIFORMERV2_PATH)

# =============================================================================
# CONFIG
# =============================================================================
MODEL_CONFIG = {
    "num_frames": 64,
    "crop_size": 336,
    "mean": [0.45, 0.45, 0.45],
    "std": [0.225, 0.225, 0.225],
}
CLASS_NAMES = ["NonFight", "Fight"]


# =============================================================================
# VIDEO PROCESSOR
# =============================================================================
class VideoProcessor:
    """Same preprocessing as API."""
    
    def __init__(self, config):
        self.num_frames = config["num_frames"]
        self.crop_size = config["crop_size"]
        self.mean = np.array(config["mean"])
        self.std = np.array(config["std"])
    
    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames)
    
    def preprocess(self, video_path):
        frames = self.load_video(video_path)
        T = len(frames)
        
        if T == 0:
            raise ValueError(f"Video has no frames: {video_path}")
        
        # Temporal sampling
        if T >= self.num_frames:
            indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
        else:
            indices = np.tile(np.arange(T), self.num_frames // T + 1)[:self.num_frames]
        frames = frames[indices]
        
        # Spatial preprocessing
        h, w = frames.shape[1:3]
        if h < w:
            new_h, new_w = self.crop_size, int(w * self.crop_size / h)
        else:
            new_w, new_h = self.crop_size, int(h * self.crop_size / w)
        frames = np.array([cv2.resize(f, (new_w, new_h)) for f in frames])
        
        # Center crop
        h, w = frames.shape[1:3]
        y, x = (h - self.crop_size) // 2, (w - self.crop_size) // 2
        frames = frames[:, y:y+self.crop_size, x:x+self.crop_size, :]
        
        # Keep display frames (RGB)
        display_frames = frames[..., ::-1].copy()
        
        # Normalize
        normalized = frames[..., ::-1].astype(np.float32) / 255.0
        normalized = (normalized - self.mean) / self.std
        
        # To tensor (1, C, T, H, W)
        tensor = torch.from_numpy(normalized).permute(3, 0, 1, 2).float().unsqueeze(0)
        
        return tensor, display_frames


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_model(checkpoint_path: str, config_path: str, device: torch.device):
    """Load model using build_model(cfg) - SAME as API."""
    print(f"\n📦 Loading model...")
    
    from slowfast.config.defaults import get_cfg
    from slowfast.models.build import build_model
    
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.NUM_GPUS = 1 if device.type == 'cuda' else 0
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    
    model = build_model(cfg)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    
    print("  ✅ Model loaded!")
    return model


# =============================================================================
# DEEP FEATURE EXTRACTION (Local UniBlock cuối cùng)
# =============================================================================
def extract_deep_features(model, input_tensor: torch.Tensor, device: torch.device) -> Dict:
    """
    Extract features from the LAST Local UniBlock (resblocks[-1]).
    
    Đây là nơi chứa thông tin không gian-thời gian giàu ngữ nghĩa nhất,
    trước khi vào Global UniBlock (MEAA).
    
    Returns:
        - deep_features: Tensor (L, N, T, C) từ resblock cuối
        - pred_class, confidence, probs
    """
    deep_features = {}
    handles = []
    
    def hook_resblock(name):
        def hook(module, input, output):
            # output shape: (L, NT, C) where L = H*W + 1 (cls token)
            deep_features[name] = output.detach().cpu()
        return hook
    
    # Hook vào resblock cuối cùng (Local UniBlock cuối)
    # Với ViT-L/14: có 24 resblocks, return_list=[20,21,22,23]
    # resblocks[-1] = resblock thứ 23 (index 23)
    transformer = model.backbone.transformer
    last_resblock = transformer.resblocks[-1]
    handles.append(last_resblock.register_forward_hook(hook_resblock('last_local_uniblock')))
    
    # Forward pass
    with torch.no_grad():
        output = model([input_tensor.to(device)])
    
    # Clean up
    for h in handles:
        h.remove()
    
    # Get prediction
    probs = F.softmax(output, dim=1)
    pred_class = probs.argmax(dim=1).item()
    confidence = probs[0, pred_class].item()
    
    return {
        'deep_features': deep_features.get('last_local_uniblock'),
        'pred_class': pred_class,
        'confidence': confidence,
        'probs': probs[0].cpu().numpy()
    }


# =============================================================================
# PHƯƠNG PHÁP 1: AVERAGE FEATURE MAP (Đơn giản nhất)
# =============================================================================
def compute_average_feature_map(features: torch.Tensor, T_down: int = 32) -> np.ndarray:
    """
    Giải pháp C từ Gemini Pro:
    Tính trung bình cộng dọc theo trục Channel.
    
    Ý nghĩa: Vùng có giá trị cao = nhiều neuron được kích hoạt = mô hình quan tâm
    
    Args:
        features: Tensor (L, NT, C) from last resblock
        T_down: Temporal dimension after downsampling (64/2 = 32)
    
    Returns:
        heatmap: (T, H, W) numpy array
    """
    L, NT, C = features.shape
    N = NT // T_down
    H = W = int((L - 1) ** 0.5)  # L = H*W + 1 (có cls token)
    
    # Bỏ cls token, lấy spatial features
    spatial_features = features[1:]  # (H*W, NT, C)
    spatial_features = spatial_features.view(H * W, N, T_down, C)
    
    # Squeeze batch dimension (N=1)
    spatial_features = spatial_features[:, 0, :, :]  # (H*W, T, C)
    
    # Reshape to (T, H, W, C)
    spatial_features = spatial_features.permute(1, 0, 2).view(T_down, H, W, C)
    
    # Trung bình theo channel -> (T, H, W)
    avg_map = spatial_features.mean(dim=-1).numpy()
    
    # Normalize to [0, 1]
    avg_map = (avg_map - avg_map.min()) / (avg_map.max() - avg_map.min() + 1e-8)
    
    return avg_map


# =============================================================================
# PHƯƠNG PHÁP 2: EIGEN-CAM (Không cần gradient, rất nhẹ)
# =============================================================================
def compute_eigen_cam(features: torch.Tensor, T_down: int = 32) -> np.ndarray:
    """
    Giải pháp A từ Gemini Pro:
    Dùng PCA để tìm thành phần chính nhất của Feature Map.
    
    Không cần gradient -> không tốn thêm VRAM!
    
    Returns:
        eigen_cam: (T, H, W) numpy array - First Principal Component
    """
    L, NT, C = features.shape
    N = NT // T_down
    H = W = int((L - 1) ** 0.5)
    
    # Bỏ cls token
    spatial_features = features[1:]  # (H*W, NT, C)
    spatial_features = spatial_features.view(H * W, N, T_down, C)
    spatial_features = spatial_features[:, 0, :, :]  # (H*W, T, C)
    
    # Reshape to (T*H*W, C) for PCA
    reshaped = spatial_features.permute(1, 0, 2).reshape(-1, C)  # (T*H*W, C)
    
    # Center the data
    mean = reshaped.mean(dim=0, keepdim=True)
    centered = reshaped - mean
    
    # SVD (PCA without sklearn - runs on CPU, very light)
    # U, S, V = torch.svd(centered)
    # First principal component = first column of U * S[0]
    try:
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        # Project onto first PC
        first_pc = U[:, 0] * S[0]  # (T*H*W,)
    except:
        # Fallback: just use variance as importance
        first_pc = centered.var(dim=-1)
    
    # Reshape back to (T, H, W)
    eigen_cam = first_pc.view(T_down, H, W).numpy()
    
    # Take absolute value (both high positive and negative activations are important)
    eigen_cam = np.abs(eigen_cam)
    
    # Normalize
    eigen_cam = (eigen_cam - eigen_cam.min()) / (eigen_cam.max() - eigen_cam.min() + 1e-8)
    
    return eigen_cam


# =============================================================================
# PHƯƠNG PHÁP 3: TEMPORAL IMPORTANCE (Perturbation-based)
# =============================================================================
def compute_temporal_importance(model, input_tensor: torch.Tensor, device: torch.device, 
                                 target_class: int = None, n_segments: int = 8) -> np.ndarray:
    """
    Xác định segment nào quan trọng nhất cho prediction.
    Zero-out từng segment và đo sự thay đổi confidence.
    """
    B, C, T, H, W = input_tensor.shape
    
    with torch.no_grad():
        baseline_output = model([input_tensor.to(device)])
        baseline_probs = F.softmax(baseline_output, dim=1)
        if target_class is None:
            target_class = baseline_probs.argmax(dim=1).item()
        baseline_score = baseline_probs[0, target_class].item()
    
    segment_size = T // n_segments
    importance = np.zeros(n_segments)
    
    print(f"  Analyzing {n_segments} temporal segments...")
    for i in tqdm(range(n_segments), desc="  Segments"):
        start = i * segment_size
        end = start + segment_size
        
        perturbed = input_tensor.clone()
        perturbed[:, :, start:end] = 0
        
        with torch.no_grad():
            output = model([perturbed.to(device)])
            probs = F.softmax(output, dim=1)
            new_score = probs[0, target_class].item()
        
        # Importance = drop in confidence when segment is removed
        importance[i] = baseline_score - new_score
    
    return importance


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_meaningful_visualization(
    video_frames: np.ndarray,
    avg_map: np.ndarray,
    eigen_cam: np.ndarray,
    temporal_importance: np.ndarray,
    pred_class: int,
    confidence: float,
    probs: np.ndarray,
    video_name: str,
    output_path: str,
    ground_truth: str = None
):
    """
    Tạo visualization có ý nghĩa với 3 phương pháp:
    1. Original frames
    2. Average Feature Map overlay
    3. Eigen-CAM overlay  
    4. Temporal importance bar chart
    """
    T = video_frames.shape[0]
    n_display = 8
    frame_indices = np.linspace(0, T - 1, n_display, dtype=int)
    
    # Feature maps có kích thước nhỏ hơn (T_down=32), cần map về frames
    T_down = avg_map.shape[0]
    H_feat, W_feat = avg_map.shape[1], avg_map.shape[2]
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    
    # Title
    pred_label = CLASS_NAMES[pred_class]
    if ground_truth:
        correct = "✓ CORRECT" if pred_label == ground_truth else "✗ WRONG"
        title_color = 'green' if pred_label == ground_truth else 'red'
    else:
        correct = ""
        title_color = 'green' if confidence > 0.7 else 'orange'
    
    fig.suptitle(
        f"CUE-Net Deep Feature Analysis: {pred_label} ({confidence*100:.1f}%) {correct}\n"
        f"NonFight: {probs[0]*100:.1f}% | Fight: {probs[1]*100:.1f}%",
        fontsize=14, fontweight='bold', color=title_color
    )
    
    gs = GridSpec(4, n_display, figure=fig, height_ratios=[1, 1, 1, 0.6], hspace=0.3)
    
    # Helper function to resize and overlay heatmap
    def overlay_heatmap(frame, heatmap, alpha=0.5):
        """Overlay heatmap on frame."""
        frame_float = frame.astype(float) / 255.0
        heatmap_color = plt.cm.jet(heatmap)[:, :, :3]
        return alpha * frame_float + (1 - alpha) * heatmap_color
    
    # Row 1: Original frames
    for i, idx in enumerate(frame_indices):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(video_frames[idx])
        ax.set_title(f'Frame {idx}', fontsize=9)
        ax.axis('off')
        if i == 0:
            ax.text(-0.15, 0.5, 'Original\nFrames', transform=ax.transAxes,
                   fontsize=10, fontweight='bold', va='center', ha='right')
    
    # Row 2: Average Feature Map overlay
    for i, idx in enumerate(frame_indices):
        ax = fig.add_subplot(gs[1, i])
        
        # Map frame index to feature map time
        feat_t = int(idx * T_down / T)
        feat_t = min(feat_t, T_down - 1)
        
        # Resize heatmap to frame size
        heatmap_resized = cv2.resize(avg_map[feat_t], 
                                      (video_frames.shape[2], video_frames.shape[1]))
        
        overlay = overlay_heatmap(video_frames[idx], heatmap_resized)
        ax.imshow(overlay)
        ax.axis('off')
        if i == 0:
            ax.text(-0.15, 0.5, 'Avg Feature\nMap\n(Deep Layer)', transform=ax.transAxes,
                   fontsize=10, fontweight='bold', va='center', ha='right', color='blue')
    
    # Row 3: Eigen-CAM overlay
    for i, idx in enumerate(frame_indices):
        ax = fig.add_subplot(gs[2, i])
        
        feat_t = int(idx * T_down / T)
        feat_t = min(feat_t, T_down - 1)
        
        heatmap_resized = cv2.resize(eigen_cam[feat_t], 
                                      (video_frames.shape[2], video_frames.shape[1]))
        
        overlay = overlay_heatmap(video_frames[idx], heatmap_resized)
        ax.imshow(overlay)
        ax.axis('off')
        if i == 0:
            ax.text(-0.15, 0.5, 'Eigen-CAM\n(PCA-based)', transform=ax.transAxes,
                   fontsize=10, fontweight='bold', va='center', ha='right', color='purple')
    
    # Row 4: Temporal importance
    ax = fig.add_subplot(gs[3, :])
    
    n_segments = len(temporal_importance)
    segment_size = T // n_segments
    segment_labels = [f"Seg {i+1}\n(F{i*segment_size+1}-{(i+1)*segment_size})" 
                      for i in range(n_segments)]
    
    # Color by importance
    norm_imp = (temporal_importance - temporal_importance.min()) / \
               (temporal_importance.max() - temporal_importance.min() + 1e-8)
    colors = [plt.cm.RdYlGn_r(v * 0.6 + 0.2) for v in norm_imp]
    
    bars = ax.bar(segment_labels, temporal_importance, color=colors, edgecolor='black')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Importance Score\n(↑ = more critical)')
    ax.set_title('Temporal Segment Importance (Red = Most important for prediction)', fontsize=11)
    
    # Highlight max segment
    max_idx = np.argmax(temporal_importance)
    bars[max_idx].set_edgecolor('red')
    bars[max_idx].set_linewidth(3)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  💾 Saved: {output_path}")


# =============================================================================
# MAIN VISUALIZATION FUNCTION
# =============================================================================
def visualize_video(video_path: str, model=None, device=None, output_dir: str = OUTPUT_DIR):
    """Complete visualization pipeline."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    video_name = Path(video_path).stem
    
    print(f"\n{'='*60}")
    print(f"🎬 Processing: {Path(video_path).name}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    if model is None:
        model = load_model(MODEL_CHECKPOINT, CONFIG_PATH, device)
    
    # Process video
    processor = VideoProcessor(MODEL_CONFIG)
    input_tensor, display_frames = processor.preprocess(video_path)
    print(f"Video shape: {input_tensor.shape}")
    
    # Extract deep features from LAST Local UniBlock
    print("\n📊 Extracting deep features from last Local UniBlock...")
    result = extract_deep_features(model, input_tensor, device)
    
    pred_class = result['pred_class']
    confidence = result['confidence']
    probs = result['probs']
    deep_features = result['deep_features']
    
    print(f"\n🎯 Prediction: {CLASS_NAMES[pred_class]} ({confidence*100:.1f}%)")
    print(f"   NonFight: {probs[0]*100:.1f}% | Fight: {probs[1]*100:.1f}%")
    
    if deep_features is None:
        print("⚠️ Could not extract deep features. Using conv1 fallback...")
        # Fallback to conv1 if deep features not available
        from visualize_local import extract_feature_maps
        result2 = extract_feature_maps(model, input_tensor, device)
        feat = result2['features']['conv1']
        avg_map = feat[0].mean(dim=0).numpy()
        eigen_cam = avg_map.copy()
    else:
        print(f"   Deep features shape: {deep_features.shape}")
        
        # Compute visualization maps
        print("\n🔬 Computing Average Feature Map...")
        avg_map = compute_average_feature_map(deep_features)
        
        print("🔬 Computing Eigen-CAM...")
        eigen_cam = compute_eigen_cam(deep_features)
    
    # Compute temporal importance
    print("\n⏱️ Computing temporal importance...")
    temporal_imp = compute_temporal_importance(model, input_tensor, device, pred_class)
    
    # Determine ground truth
    path_lower = video_path.lower()
    if 'nonfight' in path_lower:
        ground_truth = 'NonFight'
    elif 'fight' in path_lower:
        ground_truth = 'Fight'
    else:
        ground_truth = None
    
    # Create visualization
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_name}_deep_analysis.png")
    
    print("\n🎨 Creating meaningful visualization...")
    create_meaningful_visualization(
        display_frames,
        avg_map,
        eigen_cam,
        temporal_imp,
        pred_class,
        confidence,
        probs,
        video_name,
        output_path,
        ground_truth
    )
    
    print(f"\n✅ Complete!")
    
    return {
        'pred_class': pred_class,
        'pred_label': CLASS_NAMES[pred_class],
        'confidence': confidence,
        'probs': probs,
        'output_path': output_path
    }


# =============================================================================
# BATCH PROCESSING
# =============================================================================
def visualize_batch(video_paths: List[str], output_dir: str = OUTPUT_DIR):
    """Process multiple videos efficiently."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n📦 Loading model...")
    model = load_model(MODEL_CHECKPOINT, CONFIG_PATH, device)
    
    results = []
    for video_path in video_paths:
        if os.path.exists(video_path):
            try:
                result = visualize_video(video_path, model, device, output_dir)
                results.append({'video': video_path, **result})
            except Exception as e:
                print(f"❌ Error processing {video_path}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ Video not found: {video_path}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 BATCH SUMMARY")
    print("="*60)
    
    correct = 0
    total = 0
    for r in results:
        video_name = Path(r['video']).name
        pred = r['pred_label']
        conf = r['confidence']
        
        path_lower = r['video'].lower()
        if 'nonfight' in path_lower:
            gt = 'NonFight'
        elif 'fight' in path_lower:
            gt = 'Fight'
        else:
            gt = None
        
        if gt:
            is_correct = pred == gt
            correct += int(is_correct)
            total += 1
            status = "✓" if is_correct else "✗"
        else:
            status = "?"
        
        print(f"  {status} {video_name}: {pred} ({conf*100:.1f}%)")
    
    if total > 0:
        print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    
    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CUE-Net Deep Feature Visualization")
    parser.add_argument("--video", type=str, help="Path to single video")
    parser.add_argument("--videos", type=str, nargs='+', help="List of video paths")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output directory")
    
    args = parser.parse_args()
    
    if args.video:
        visualize_video(args.video, output_dir=args.output)
    elif args.videos:
        visualize_batch(args.videos, args.output)
    else:
        print("Usage:")
        print("  python visualize_meaningful_v2.py --video path/to/video.avi")
        print("  python visualize_meaningful_v2.py --videos video1.avi video2.avi ...")


if __name__ == "__main__":
    main()
