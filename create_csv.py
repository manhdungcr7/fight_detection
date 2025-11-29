#!/usr/bin/env python3
"""
Script để tạo CSV files cho RWF-2000-Cropped dataset
Format: path_to_video label
Separator: space " "
"""

import os

def create_csv_for_rwf(base_path, output_csv, label_map):
    """
    Tạo CSV cho RWF-2000 theo format:
    path_to_video label
    
    Args:
        base_path: Đường dẫn đến thư mục chứa Fight và NonFight
        output_csv: Đường dẫn file CSV output
        label_map: Dict mapping tên class -> label number
    """
    lines = []
    
    for class_name, label in label_map.items():
        class_dir = os.path.join(base_path, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} không tồn tại!")
            continue
        
        video_files = [f for f in os.listdir(class_dir) if f.endswith('.avi')]
        print(f"Tìm thấy {len(video_files)} videos trong {class_name}")
        
        for video_file in video_files:
            # Đường dẫn tương đối từ PATH_PREFIX
            rel_path = f"{class_name}/{video_file}"
            lines.append(f"{rel_path} {label}\n")
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"✓ Created {output_csv} with {len(lines)} videos")
    return len(lines)


if __name__ == "__main__":
    # Label mapping: Fight=1, NonFight=0
    label_map = {'Fight': 1, 'NonFight': 0}
    
    base_dir = 'e:/fight_detection_cuenet'
    
    print("=" * 60)
    print("TẠO CSV FILES CHO RWF-2000 DATASET")
    print("=" * 60)
    
    # Kiểm tra xem RWF-2000-Cropped đã tồn tại chưa
    cropped_dir = os.path.join(base_dir, 'RWF-2000-Cropped')
    if os.path.exists(cropped_dir):
        print(f"\n✓ Sử dụng RWF-2000-Cropped (đã crop)")
        dataset_dir = cropped_dir
    else:
        print(f"\n! RWF-2000-Cropped chưa tồn tại")
        print(f"! Tạm thời sử dụng RWF-2000 gốc")
        print(f"! Bạn cần chạy Spatial_Cropping_Module để tạo RWF-2000-Cropped")
        dataset_dir = os.path.join(base_dir, 'RWF-2000')
    
    print(f"\nDataset directory: {dataset_dir}")
    print()
    
    # Tạo train.csv
    print("\n[1/3] Tạo train.csv...")
    train_count = create_csv_for_rwf(
        base_path=os.path.join(dataset_dir, 'train'),
        output_csv=os.path.join(base_dir, 'data_paths', 'train.csv'),
        label_map=label_map
    )
    
    # Tạo val.csv
    print("\n[2/3] Tạo val.csv...")
    val_count = create_csv_for_rwf(
        base_path=os.path.join(dataset_dir, 'val'),
        output_csv=os.path.join(base_dir, 'data_paths', 'val.csv'),
        label_map=label_map
    )
    
    # Tạo test.csv (giống val.csv cho RWF-2000)
    print("\n[3/3] Tạo test.csv (giống val.csv)...")
    test_count = create_csv_for_rwf(
        base_path=os.path.join(dataset_dir, 'val'),
        output_csv=os.path.join(base_dir, 'data_paths', 'test.csv'),
        label_map=label_map
    )
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH!")
    print("=" * 60)
    print(f"Train videos: {train_count}")
    print(f"Val videos:   {val_count}")
    print(f"Test videos:  {test_count}")
    print(f"\nCSV files đã được tạo tại: {os.path.join(base_dir, 'data_paths')}")
    print("=" * 60)
