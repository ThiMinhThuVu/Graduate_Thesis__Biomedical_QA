import json
import os
import random

def merge_k_random_items(folders, files_per_folder=4, k=10, output_file='final_merged.json', seed=42):
    # Thiết lập seed để kết quả có thể tái lập (reproducible)
    random.seed(seed)
    final_data = {}
    
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} not found. Skipping...")
            continue

        all_files = [f for f in os.listdir(folder) if f.endswith('.json')]
        target_files = all_files[:files_per_folder]
        
        print(f"Processing folder: {folder} ({len(target_files)} files)")
        
        for file_name in target_files:
            file_path = os.path.join(folder, file_name)
            if 'bridge_comparison' in file_name or 'inference' in file_name:
                k = 250
            elif 'compositional' in file_name:
                k = 600
            else:
                k  = 400
            
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    items = list(data.items())
                    
                    # Kiểm tra nếu số lượng item trong file ít hơn k
                    num_to_sample = min(len(items), k)
                    
                    # Lấy ngẫu nhiên num_to_sample items
                    subset_list = random.sample(items, num_to_sample)
                    subset = dict(subset_list)
                    
                    final_data.update(subset)
                    print(f"  - Randomly extracted {len(subset)} items from {file_name}")
                except json.JSONDecodeError:
                    print(f"  - Error: Could not decode {file_name}")

    # Ghi file cuối cùng
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nDone! Total items in {output_file}: {len(final_data)}")

# Cấu hình
folders_to_process = ['data/wiki/dev', 'data/wiki/train']
K_VALUE = 5 
SEED = 2026

merge_k_random_items(folders_to_process, files_per_folder=4, k=K_VALUE, seed=SEED)