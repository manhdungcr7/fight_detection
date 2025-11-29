#!/usr/bin/env python3
"""
Evaluate model trên validation set và phân tích chi tiết kết quả
In ra danh sách video dự đoán ĐÚNG và SAI
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import json
from tqdm import tqdm
import time

# Setup paths
BASE_PATH = Path(__file__).parent
UNIFORMER_PATH = BASE_PATH / "UniFormerV2"

sys.path.insert(0, str(UNIFORMER_PATH / "slowfast"))
sys.path.insert(0, str(UNIFORMER_PATH))

from slowfast.config.defaults import get_cfg
from slowfast.models import build_model
import slowfast.utils.checkpoint as cu

def load_model(checkpoint_path, config_path):
    """Load model từ checkpoint"""
    print(f"🔨 Loading model from {checkpoint_path}")
    
    # Load config
    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    
    # Override settings for inference
    cfg.NUM_GPUS = 0  # Sẽ tự detect GPU nếu có
    cfg.TEST.CHECKPOINT_FILE_PATH = str(checkpoint_path)
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    
    # Build model
    model = build_model(cfg)
    model.eval()
    
    # Load checkpoint
    print(f"📦 Loading checkpoint...")
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"✅ Model loaded on {device}")
    
    return model, cfg, device

def load_video_frames(video_path, num_frames=64, img_size=336):
    """Load và preprocess video"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # Calculate sampling indices
    if total_frames < num_frames:
        indices = list(range(total_frames)) + [total_frames-1] * (num_frames - total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    
    # Load sampled frames
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    # Convert to tensor
    frames = np.stack(frames, axis=0).astype(np.float32) / 255.0
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2)  # (C, T, H, W)
    
    # Normalize
    mean = torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32).view(3, 1, 1, 1)
    frames = (frames - mean) / std
    
    return frames.unsqueeze(0)  # (1, C, T, H, W)

def predict_video(model, video_tensor, device):
    """Dự đoán 1 video"""
    video_tensor = video_tensor.to(device)
    
    with torch.inference_mode():
        preds = model([video_tensor])
        probs = torch.nn.functional.softmax(preds, dim=1)
    
    probs = probs.squeeze().cpu().numpy()
    return probs

def evaluate_validation_set(model, cfg, device, val_root):
    """Evaluate trên validation set"""
    
    val_root = Path(val_root)
    
    # Class mapping
    class_names = ['NonFight', 'Fight']
    class_dirs = {
        0: val_root / 'NonFight',
        1: val_root / 'Fight'
    }
    
    results = {
        'correct': [],     # Dự đoán đúng
        'incorrect': [],   # Dự đoán sai
        'errors': []       # Lỗi khi load video
    }
    
    # Statistics
    stats = {
        'total': 0,
        'correct': 0,
        'incorrect': 0,
        'errors': 0,
        'per_class': {
            'NonFight': {'total': 0, 'correct': 0, 'incorrect': 0},
            'Fight': {'total': 0, 'correct': 0, 'incorrect': 0}
        }
    }
    
    print(f"\n{'='*80}")
    print(f"🎯 EVALUATING VALIDATION SET")
    print(f"{'='*80}\n")
    
    # Process từng class
    for true_label, class_dir in class_dirs.items():
        if not class_dir.exists():
            print(f"⚠️  Warning: {class_dir} không tồn tại, bỏ qua...")
            continue
        
        true_class_name = class_names[true_label]
        video_files = list(class_dir.glob('*.avi')) + list(class_dir.glob('*.mp4'))
        
        print(f"\n📁 Processing {true_class_name} ({len(video_files)} videos)...")
        
        for video_path in tqdm(video_files, desc=f"{true_class_name}"):
            stats['total'] += 1
            stats['per_class'][true_class_name]['total'] += 1
            
            # Load video
            video_tensor = load_video_frames(
                video_path,
                num_frames=cfg.DATA.NUM_FRAMES,
                img_size=cfg.DATA.TEST_CROP_SIZE
            )
            
            if video_tensor is None:
                stats['errors'] += 1
                results['errors'].append({
                    'file': video_path.name,
                    'folder': true_class_name,
                    'path': str(video_path.relative_to(val_root)),
                    'error': 'Failed to load video'
                })
                continue
            
            # Predict
            try:
                probs = predict_video(model, video_tensor, device)
                pred_label = np.argmax(probs)
                pred_class_name = class_names[pred_label]
                confidence = probs[pred_label] * 100
                
                # Check if correct
                is_correct = (pred_label == true_label)
                
                result_entry = {
                    'file': video_path.name,
                    'folder': true_class_name,
                    'path': str(video_path.relative_to(val_root)),
                    'true_label': true_class_name,
                    'pred_label': pred_class_name,
                    'confidence': float(confidence),
                    'probs': {
                        'NonFight': float(probs[0] * 100),
                        'Fight': float(probs[1] * 100)
                    }
                }
                
                if is_correct:
                    results['correct'].append(result_entry)
                    stats['correct'] += 1
                    stats['per_class'][true_class_name]['correct'] += 1
                else:
                    results['incorrect'].append(result_entry)
                    stats['incorrect'] += 1
                    stats['per_class'][true_class_name]['incorrect'] += 1
                    
            except Exception as e:
                stats['errors'] += 1
                results['errors'].append({
                    'file': video_path.name,
                    'folder': true_class_name,
                    'path': str(video_path.relative_to(val_root)),
                    'error': str(e)
                })
    
    return results, stats

def print_results(results, stats):
    """In kết quả chi tiết"""
    
    print(f"\n{'='*80}")
    print(f"📊 EVALUATION RESULTS")
    print(f"{'='*80}\n")
    
    # Overall statistics
    accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
    
    print(f"📈 Overall Statistics:")
    print(f"  Total videos: {stats['total']}")
    print(f"  ✅ Correct: {stats['correct']} ({stats['correct']/stats['total']*100:.2f}%)")
    print(f"  ❌ Incorrect: {stats['incorrect']} ({stats['incorrect']/stats['total']*100:.2f}%)")
    print(f"  ⚠️  Errors: {stats['errors']}")
    print(f"  🎯 Accuracy: {accuracy:.2f}%")
    
    # Per-class statistics
    print(f"\n📊 Per-Class Statistics:")
    for class_name, class_stats in stats['per_class'].items():
        if class_stats['total'] > 0:
            class_acc = class_stats['correct'] / class_stats['total'] * 100
            print(f"\n  {class_name}:")
            print(f"    Total: {class_stats['total']}")
            print(f"    Correct: {class_stats['correct']}")
            print(f"    Incorrect: {class_stats['incorrect']}")
            print(f"    Accuracy: {class_acc:.2f}%")
    
    # Detailed correct predictions
    print(f"\n{'='*80}")
    print(f"✅ CORRECT PREDICTIONS ({len(results['correct'])} videos)")
    print(f"{'='*80}\n")
    
    if len(results['correct']) > 0:
        # Group by folder
        correct_by_folder = {'NonFight': [], 'Fight': []}
        for item in results['correct']:
            correct_by_folder[item['folder']].append(item)
        
        for folder in ['NonFight', 'Fight']:
            items = correct_by_folder[folder]
            if items:
                print(f"\n📁 {folder}/ ({len(items)} videos):")
                for item in sorted(items, key=lambda x: x['file']):
                    print(f"  ✅ {item['file']}")
                    print(f"     True: {item['true_label']} | Pred: {item['pred_label']} | Confidence: {item['confidence']:.2f}%")
    else:
        print("  (Không có video nào dự đoán đúng)")
    
    # Detailed incorrect predictions
    print(f"\n{'='*80}")
    print(f"❌ INCORRECT PREDICTIONS ({len(results['incorrect'])} videos)")
    print(f"{'='*80}\n")
    
    if len(results['incorrect']) > 0:
        # Group by folder
        incorrect_by_folder = {'NonFight': [], 'Fight': []}
        for item in results['incorrect']:
            incorrect_by_folder[item['folder']].append(item)
        
        for folder in ['NonFight', 'Fight']:
            items = incorrect_by_folder[folder]
            if items:
                print(f"\n📁 {folder}/ ({len(items)} videos - MISCLASSIFIED):")
                for item in sorted(items, key=lambda x: x['file']):
                    print(f"  ❌ {item['file']}")
                    print(f"     True: {item['true_label']} | Pred: {item['pred_label']} | Confidence: {item['confidence']:.2f}%")
                    print(f"     Probabilities: NonFight={item['probs']['NonFight']:.2f}%, Fight={item['probs']['Fight']:.2f}%")
    else:
        print("  (Tất cả video đều dự đoán đúng! 🎉)")
    
    # Errors
    if len(results['errors']) > 0:
        print(f"\n{'='*80}")
        print(f"⚠️  ERRORS ({len(results['errors'])} videos)")
        print(f"{'='*80}\n")
        
        for item in results['errors']:
            print(f"  ⚠️  {item['folder']}/{item['file']}")
            print(f"     Error: {item['error']}")

def save_results_to_file(results, stats, output_dir):
    """Lưu kết quả ra file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / 'validation_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'stats': stats,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {json_path}")
    
    # Save detailed text report
    report_path = output_dir / 'validation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("VALIDATION EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        f.write(f"Total videos: {stats['total']}\n")
        f.write(f"Correct: {stats['correct']} ({stats['correct']/stats['total']*100:.2f}%)\n")
        f.write(f"Incorrect: {stats['incorrect']} ({stats['incorrect']/stats['total']*100:.2f}%)\n")
        f.write(f"Errors: {stats['errors']}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        
        # Per-class
        f.write("PER-CLASS STATISTICS\n")
        f.write("-"*80 + "\n")
        for class_name, class_stats in stats['per_class'].items():
            if class_stats['total'] > 0:
                class_acc = class_stats['correct'] / class_stats['total'] * 100
                f.write(f"\n{class_name}:\n")
                f.write(f"  Total: {class_stats['total']}\n")
                f.write(f"  Correct: {class_stats['correct']}\n")
                f.write(f"  Incorrect: {class_stats['incorrect']}\n")
                f.write(f"  Accuracy: {class_acc:.2f}%\n")
        
        # Incorrect predictions (most important)
        f.write("\n" + "="*80 + "\n")
        f.write(f"INCORRECT PREDICTIONS ({len(results['incorrect'])} videos)\n")
        f.write("="*80 + "\n\n")
        
        if results['incorrect']:
            incorrect_by_folder = {'NonFight': [], 'Fight': []}
            for item in results['incorrect']:
                incorrect_by_folder[item['folder']].append(item)
            
            for folder in ['NonFight', 'Fight']:
                items = incorrect_by_folder[folder]
                if items:
                    f.write(f"\n{folder}/ ({len(items)} videos misclassified):\n")
                    f.write("-"*80 + "\n")
                    for item in sorted(items, key=lambda x: x['file']):
                        f.write(f"\nFile: {item['file']}\n")
                        f.write(f"  Path: {item['path']}\n")
                        f.write(f"  True Label: {item['true_label']}\n")
                        f.write(f"  Predicted: {item['pred_label']}\n")
                        f.write(f"  Confidence: {item['confidence']:.2f}%\n")
                        f.write(f"  Probabilities:\n")
                        f.write(f"    NonFight: {item['probs']['NonFight']:.2f}%\n")
                        f.write(f"    Fight: {item['probs']['Fight']:.2f}%\n")
        else:
            f.write("No incorrect predictions! All videos classified correctly!\n")
        
        # Correct predictions
        f.write("\n" + "="*80 + "\n")
        f.write(f"CORRECT PREDICTIONS ({len(results['correct'])} videos)\n")
        f.write("="*80 + "\n\n")
        
        if results['correct']:
            correct_by_folder = {'NonFight': [], 'Fight': []}
            for item in results['correct']:
                correct_by_folder[item['folder']].append(item)
            
            for folder in ['NonFight', 'Fight']:
                items = correct_by_folder[folder]
                if items:
                    f.write(f"\n{folder}/ ({len(items)} videos correct):\n")
                    for item in sorted(items, key=lambda x: x['file']):
                        f.write(f"  {item['file']} (Confidence: {item['confidence']:.2f}%)\n")
    
    print(f"📄 Detailed report saved to: {report_path}")
    
    # Save separate lists
    incorrect_list_path = output_dir / 'incorrect_predictions.txt'
    with open(incorrect_list_path, 'w', encoding='utf-8') as f:
        f.write("MISCLASSIFIED VIDEOS\n")
        f.write("="*80 + "\n\n")
        for item in sorted(results['incorrect'], key=lambda x: (x['folder'], x['file'])):
            f.write(f"{item['path']}\n")
            f.write(f"  True: {item['true_label']} | Pred: {item['pred_label']} | Conf: {item['confidence']:.2f}%\n\n")
    print(f"📝 Incorrect list saved to: {incorrect_list_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model on validation set")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/cuenet_rwf2000_epoch51.pyth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='UniFormerV2/exp/RWF_exp/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--val-root',
        type=str,
        default='RWF-2000/val',
        help='Path to validation set root'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='validation_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    val_root = Path(args.val_root)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return
    
    if not val_root.exists():
        print(f"❌ Validation root not found: {val_root}")
        return
    
    print(f"📄 Checkpoint: {checkpoint_path}")
    print(f"📄 Config: {config_path}")
    print(f"📁 Validation: {val_root}")
    print(f"📁 Output: {args.output}")
    
    # Load model
    start_time = time.time()
    model, cfg, device = load_model(checkpoint_path, config_path)
    
    # Evaluate
    results, stats = evaluate_validation_set(model, cfg, device, val_root)
    
    # Print results
    print_results(results, stats)
    
    # Save results
    save_results_to_file(results, stats, args.output)
    
    # Time
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    print(f"   Average: {elapsed/stats['total']:.2f}s per video")
    
    print(f"\n{'='*80}")
    print("✅ Evaluation complete!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
