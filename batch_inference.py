#!/usr/bin/env python3
"""
Script inference batch cho nhiều videos
Tạo CSV kết quả với predictions
"""

import sys
import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Import từ inference_single_video
from inference_single_video import load_trained_model, preprocess_video, predict_video


def batch_inference(video_dir, checkpoint_path, config_path, output_csv='predictions.csv'):
    """
    Inference cho tất cả videos trong thư mục
    
    Args:
        video_dir: Thư mục chứa videos
        checkpoint_path: Path to checkpoint
        config_path: Path to config
        output_csv: Output CSV file
    """
    print("=" * 60)
    print("CUE-NET BATCH INFERENCE")
    print("=" * 60)
    print(f"Video directory: {video_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print()
    
    # Load model
    model, cfg = load_trained_model(config_path, checkpoint_path)
    
    # Get all video files
    video_files = list(Path(video_dir).glob('*.avi'))
    video_files.extend(Path(video_dir).glob('*.mp4'))
    
    print(f"Found {len(video_files)} videos\n")
    
    # Results list
    results = []
    
    # Process each video
    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            # Preprocess
            video_tensor = preprocess_video(str(video_path), cfg)
            
            # Predict
            pred_class, confidence, probs = predict_video(model, video_tensor, cfg)
            
            # Store result
            results.append({
                'video_path': str(video_path),
                'video_name': video_path.name,
                'prediction': 'Fight' if pred_class == 1 else 'NonFight',
                'prediction_label': pred_class,
                'confidence': confidence,
                'prob_nonfight': probs[0],
                'prob_fight': probs[1]
            })
            
        except Exception as e:
            print(f"\nError processing {video_path.name}: {e}")
            results.append({
                'video_path': str(video_path),
                'video_name': video_path.name,
                'prediction': 'ERROR',
                'prediction_label': -1,
                'confidence': 0.0,
                'prob_nonfight': 0.0,
                'prob_fight': 0.0
            })
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Results saved to {output_csv}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    fight_count = (df['prediction'] == 'Fight').sum()
    nonfight_count = (df['prediction'] == 'NonFight').sum()
    error_count = (df['prediction'] == 'ERROR').sum()
    
    print(f"Total videos: {len(video_files)}")
    print(f"Fight:    {fight_count} ({fight_count/len(video_files)*100:.1f}%)")
    print(f"NonFight: {nonfight_count} ({nonfight_count/len(video_files)*100:.1f}%)")
    if error_count > 0:
        print(f"Errors:   {error_count}")
    
    print(f"\nAverage confidence: {df['confidence'].mean():.2%}")
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CUE-Net Batch Inference')
    parser.add_argument('--video_dir', type=str, required=True, 
                        help='Directory containing videos')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to trained checkpoint')
    parser.add_argument('--config', type=str, 
                        default='UniFormerV2/exp/RWF_exp/config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    batch_inference(args.video_dir, args.checkpoint, args.config, args.output)


if __name__ == "__main__":
    main()
