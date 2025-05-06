import os
import torch
import cv2
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Define the model architecture (must match your trained model)
class UNetColorizer(torch.nn.Module):
    def __init__(self):
        super(UNetColorizer, self).__init__()
        
        # Encoder
        self.enc1 = self._encoder_block(1, 64)
        self.enc2 = self._encoder_block(64, 128)
        self.enc3 = self._encoder_block(128, 256)
        self.enc4 = self._encoder_block(256, 512)
        
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, 3, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(1024, 1024, 3, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec4 = self._decoder_block(1024 + 512, 512)
        self.dec3 = self._decoder_block(512 + 256, 256)
        self.dec2 = self._decoder_block(256 + 128, 128)
        self.dec1 = self._decoder_block(128 + 64, 64)
        
        # Final layer
        self.final = torch.nn.Conv2d(64, 2, 1)  # Output 2 channels (ab)
        
    def _encoder_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
    
    def _decoder_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)
        )
    
    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        
        # Bottleneck
        bottleneck_out = self.bottleneck(enc4_out)
        
        # Decoder with skip connections
        dec4_out = self.dec4(torch.cat([bottleneck_out, enc4_out], dim=1))
        dec3_out = self.dec3(torch.cat([dec4_out, enc3_out], dim=1))
        dec2_out = self.dec2(torch.cat([dec3_out, enc2_out], dim=1))
        dec1_out = self.dec1(torch.cat([dec2_out, enc1_out], dim=1))
        
        # Final layer
        output = self.final(dec1_out)
        
        return output


def colorize_image(image_path, model, device):
    """Colorize a single grayscale image"""
    # Read image
    if isinstance(image_path, str):
        # Load from file
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
    else:
        # Assume it's already a numpy array
        img = image_path
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Convert color image to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Store original dimensions
    original_h, original_w = img.shape
    
    # Resize to 256x256 for model
    img_resized = cv2.resize(img, (256, 256))
    
    # Normalize to [-1, 1]
    img_norm = img_resized.astype(np.float32) / 127.5 - 1.0
    
    # Convert to tensor and add batch & channel dimensions
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        ab_pred = model(img_tensor)
    
    # Convert back to numpy
    ab_pred_np = ((ab_pred[0] + 1) * 127.5).cpu().numpy().astype(np.uint8)
    ab_pred_np = np.transpose(ab_pred_np, (1, 2, 0))  # (2, H, W) -> (H, W, 2)
    
    # Create LAB image
    lab = np.zeros((256, 256, 3), dtype=np.uint8)
    lab[:, :, 0] = ((img_tensor[0, 0] + 1) * 127.5).cpu().numpy().astype(np.uint8)
    lab[:, :, 1:] = ab_pred_np
    
    # Convert to BGR
    colorized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Resize back to original dimensions
    if original_h != 256 or original_w != 256:
        colorized = cv2.resize(colorized, (original_w, original_h))
    
    return colorized


def show_comparison(gray_img, color_img, figsize=(12, 6)):
    """Display grayscale and colorized images side by side"""
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 2, 1)
    plt.title("Original Grayscale")
    plt.imshow(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Colorized")
    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# Configuration
model_path = "F:/Projects/Colorizer/results 5e16b/checkpoints/best_model.pth"  # Change this to your model path
input_path = "C:/Users/admin/Downloads/bnw.jpeg"  # Change this to your input image or directory
output_path = "C:/Users/admin/Downloads"  # Change this to your desired output directory
use_gpu = False  # Set to False if you want to use CPU

# Create output directory
os.makedirs(output_path, exist_ok=True)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
print(f"Using device: {device}")

# Load model
model = UNetColorizer().to(device)
checkpoint = torch.load(model_path, map_location=device)

# Handle different checkpoint formats
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
print(f"Model loaded from {model_path}")

# Process input (file or directory)
if os.path.isdir(input_path):
    # Process all images in directory
    image_files = [f for f in os.listdir(input_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"Found {len(image_files)} images")
    
    # Process and show some examples
    num_examples = min(5, len(image_files))  # Show at most 5 examples
    for i, filename in enumerate(image_files[:num_examples]):
        try:
            # Input and output paths
            img_path = os.path.join(input_path, filename)
            output_file = os.path.join(output_path, f"colored_{filename}")
            
            # Read the original image as grayscale
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Colorize image
            colorized = colorize_image(gray_img, model, device)
            
            # Save result
            cv2.imwrite(output_file, colorized)
            
            # Display comparison
            show_comparison(gray_img, colorized)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Process the rest without displaying
    if len(image_files) > num_examples:
        for filename in tqdm(image_files[num_examples:], desc="Processing additional images"):
            try:
                img_path = os.path.join(input_path, filename)
                output_file = os.path.join(output_path, f"colored_{filename}")
                
                colorized = colorize_image(img_path, model, device)
                cv2.imwrite(output_file, colorized)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print(f"All images processed and saved to {output_path}")
    
else:
    # Process single image
    if not os.path.isfile(input_path):
        print(f"Input file not found: {input_path}")
    else:
        filename = os.path.basename(input_path)
        output_file = os.path.join(output_path, f"colored_{filename}")
        
        # Read the original image as grayscale
        gray_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        # Colorize image
        colorized = colorize_image(gray_img, model, device)
        
        # Save result
        cv2.imwrite(output_file, colorized)
        
        # Display comparison
        show_comparison(gray_img, colorized)
        
        print(f"Colorized image saved to {output_file}")

# Function to colorize a single image (for interactive use)
def colorize_and_display(image_path):
    """Load, colorize and display a single image"""
    try:
        # Read the original image as grayscale
        if isinstance(image_path, str):
            gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                print(f"Could not read image: {image_path}")
                return
        else:
            gray_img = image_path
            if len(gray_img.shape) == 3:
                gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        
        # Colorize image
        colorized = colorize_image(gray_img, model, device)
        
        # Save if there's a specific path
        if isinstance(image_path, str):
            filename = os.path.basename(image_path)
            output_file = os.path.join(output_path, f"colored_{filename}")
            cv2.imwrite(output_file, colorized)
            print(f"Saved to {output_file}")
        
        # Display comparison
        show_comparison(gray_img, colorized)
        
        return colorized
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

print("\nTo colorize a new image, use the colorize_and_display() function:")
print("Example: colorize_and_display('path/to/image.jpg')")