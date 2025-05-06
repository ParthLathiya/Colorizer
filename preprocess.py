import cv2
import os
import numpy as np
from tqdm import tqdm

# Define directories
input_dir = "dataset/color"
output_dir = "dataset/processedimg"

# Create output directories for L and AB channels
l_dir = os.path.join(output_dir, "L")
ab_dir = os.path.join(output_dir, "AB")

if not os.path.exists(l_dir):
    os.makedirs(l_dir)
if not os.path.exists(ab_dir):
    os.makedirs(ab_dir)

# Process each image
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Processing {len(image_files)} images...")

for img_name in tqdm(image_files):
    try:
        # Get base name without extension
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(input_dir, img_name)
        
        # Read and process image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        # Resize image
        img = cv2.resize(img, (256, 256))
        
        # Convert to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Save L channel (Grayscale)
        l_channel = lab[:, :, 0]
        cv2.imwrite(os.path.join(l_dir, f"{base_name}.png"), l_channel)
        
        # Save AB channels
        ab_channels = lab[:, :, 1:]
        np.save(os.path.join(ab_dir, f"{base_name}.npy"), ab_channels)
        
    except Exception as e:
        print(f"Error processing {img_name}: {str(e)}")

print("Processing complete!")