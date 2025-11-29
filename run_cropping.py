#!/usr/bin/env python3
"""
Script chạy Spatial Cropping Module cho RWF-2000
Tự động hóa toàn bộ quá trình cropping
"""

import os
import sys
import cv2
import math
import shutil
import numpy as np
from pathlib import Path

# Import YOLOv8
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics chưa được cài đặt!")
    print("Chạy: pip install ultralytics")
    sys.exit(1)


def crop_video(input_video_path, output_video_path, crop_coords):
    """Crop video với tọa độ cho trước"""
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = None
    max_x1, max_y1, max_x2, max_y2 = crop_coords

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[max_y1:max_y2, max_x1:max_x2]

        if out is None:
            out = cv2.VideoWriter(
                output_video_path, 
                fourcc, 
                cap.get(cv2.CAP_PROP_FPS), 
                (cropped_frame.shape[1], cropped_frame.shape[0])
            )

        out.write(cropped_frame)

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


def process_videos(inp_video_dir, opt_video_dir, model):
    """Xử lý tất cả videos trong thư mục"""
    os.makedirs(opt_video_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(inp_video_dir) if f.endswith('.avi')]
    total = len(video_files)
    
    # Bỏ qua videos đã được xử lý (RESUME SUPPORT)
    processed_files = set(os.listdir(opt_video_dir)) if os.path.exists(opt_video_dir) else set()
    video_files = [f for f in video_files if f not in processed_files]
    
    if processed_files:
        print(f"Đã xử lý trước đó: {len(processed_files)} videos")
    print(f"Còn lại cần xử lý: {len(video_files)}/{total} videos")
    
    if not video_files:
        print("✓ Tất cả videos đã được xử lý!")
        return 0, 0
    
    cropped_count = 0
    copied_count = 0
    
    for idx, filename in enumerate(video_files, 1):
        print(f"[{idx}/{len(video_files)}] Processing {filename}...", end=' ')
        
        video_file_path = os.path.join(inp_video_dir, filename)
        output_video_path = os.path.join(opt_video_dir, filename)
        
        # Detect people với YOLOv8
        results = model.predict(video_file_path, stream=True, conf=0.3, classes=0)
        
        max_x1, max_y1 = float('inf'), float('inf')
        max_x2, max_y2 = 0, 0
        max_people = 0

        for r in results:
            people = 0
            for rr in r:
                max_x1 = min(max_x1, rr.boxes.xyxy[0][0].item())
                max_y1 = min(max_y1, rr.boxes.xyxy[0][1].item())
                max_x2 = max(max_x2, rr.boxes.xyxy[0][2].item())
                max_y2 = max(max_y2, rr.boxes.xyxy[0][3].item())
                people += 1
            
            if people > 1:
                max_people = max(max_people, people)
        
        # Crop nếu có > 1 người, ngược lại copy
        if max_people > 1:
            crop_coords = (
                math.ceil(max_x1), 
                math.ceil(max_y1), 
                math.ceil(max_x2), 
                math.ceil(max_y2)
            )
            crop_video(video_file_path, output_video_path, crop_coords)
            print("✓ Cropped")
            cropped_count += 1
        else:
            shutil.copy(video_file_path, output_video_path)
            print("✓ Copied")
            copied_count += 1
    
    print(f"\nKết quả: {cropped_count} cropped, {copied_count} copied")
    return cropped_count, copied_count


def main():
    print("=" * 60)
    print("SPATIAL CROPPING MODULE - RWF-2000")
    print("=" * 60)
    print()
    
    # Đường dẫn
    base_dir = Path('e:/fight_detection_cuenet')
    input_dir = base_dir / 'RWF-2000'
    output_dir = base_dir / 'RWF-2000-Cropped'
    
    # Kiểm tra input
    if not input_dir.exists():
        print(f"ERROR: {input_dir} không tồn tại!")
        sys.exit(1)
    
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Load YOLO model
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8x.pt')
    print("✓ Model loaded\n")
    
    # Tạo thư mục output
    folders = [
        output_dir / 'train' / 'Fight',
        output_dir / 'train' / 'NonFight',
        output_dir / 'val' / 'Fight',
        output_dir / 'val' / 'NonFight'
    ]
    
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    
    # Process từng folder
    total_cropped = 0
    total_copied = 0
    
    for split in ['train', 'val']:
        for class_name in ['Fight', 'NonFight']:
            inp_path = input_dir / split / class_name
            opt_path = output_dir / split / class_name
            
            print(f"\n{'=' * 60}")
            print(f"Processing {split}/{class_name}")
            print('=' * 60)
            
            cropped, copied = process_videos(str(inp_path), str(opt_path), model)
            total_cropped += cropped
            total_copied += copied
    
    # Tổng kết
    print("\n" + "=" * 60)
    print("HOÀN THÀNH!")
    print("=" * 60)
    print(f"Tổng videos cropped: {total_cropped}")
    print(f"Tổng videos copied:  {total_copied}")
    print(f"Tổng cộng:           {total_cropped + total_copied}")
    print(f"\nDataset đã được lưu tại: {output_dir}")
    print("=" * 60)
    print("\nBước tiếp theo:")
    print("1. Chạy: python create_csv.py")
    print("2. Script sẽ tự động phát hiện RWF-2000-Cropped và tạo CSV mới")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nĐã dừng bởi người dùng.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
