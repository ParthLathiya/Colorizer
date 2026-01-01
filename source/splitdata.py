import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define directories
processed_dir = "dataset/processedimg"
l_dir = os.path.join(processed_dir, "L")
ab_dir = os.path.join(processed_dir, "AB")

# Define output directories
output_base = "dataset/split"
splits = ['train', 'val', 'test']

# Create output directories
for split in splits:
    for channel in ['L', 'AB']:
        split_dir = os.path.join(output_base, split, channel)
        os.makedirs(split_dir, exist_ok=True)

# Get all files
l_files = [f for f in os.listdir(l_dir) if f.endswith('.png')]
print(f"Found {len(l_files)} processed images")

# Split the dataset: 70% train, 15% validation, 15% test
train_files, temp_files = train_test_split(l_files, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

print(f"Split sizes: Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")

# Function to copy files to their respective directories
def copy_files(file_list, split_name):
    for img_name in tqdm(file_list, desc=f"Copying {split_name} files"):
        base_name = os.path.splitext(img_name)[0]
        
        # Copy L channel (PNG)
        src_l = os.path.join(l_dir, img_name)
        dst_l = os.path.join(output_base, split_name, 'L', img_name)
        shutil.copy2(src_l, dst_l)
        
        # Copy AB channels (NPY)
        src_ab = os.path.join(ab_dir, f"{base_name}.npy")
        dst_ab = os.path.join(output_base, split_name, 'AB', f"{base_name}.npy")
        
        if os.path.exists(src_ab):
            shutil.copy2(src_ab, dst_ab)
        else:
            print(f"Warning: AB file not found for {base_name}")

# Copy files to respective split directories
copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print("Dataset split complete!")

# Print summary
for split in splits:
    l_count = len(os.listdir(os.path.join(output_base, split, 'L')))
    ab_count = len(os.listdir(os.path.join(output_base, split, 'AB')))
    print(f"{split.capitalize()} set: {l_count} L-channel images, {ab_count} AB-channel files")