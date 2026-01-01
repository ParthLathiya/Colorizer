import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset
class ColorizationDataset(Dataset):
    def __init__(self, l_dir, ab_dir, transform=None):
        self.l_dir = l_dir
        self.ab_dir = ab_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(l_dir) if f.endswith('.png')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base_name = os.path.splitext(img_name)[0]
        
        # Load L channel (grayscale)
        l_path = os.path.join(self.l_dir, img_name)
        l_img = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)
        
        # Load AB channels
        ab_path = os.path.join(self.ab_dir, f"{base_name}.npy")
        ab_img = np.load(ab_path)
        
        # Normalize L from [0, 255] to [-1, 1]
        l_img = l_img.astype(np.float32) / 127.5 - 1.0
        
        # Normalize AB from [0, 255] to [-1, 1]
        ab_img = ab_img.astype(np.float32) / 127.5 - 1.0
        
        # Reshape L to (1, H, W)
        l_img = l_img.reshape(1, l_img.shape[0], l_img.shape[1])
        
        # Reshape AB to (2, H, W)
        ab_img = ab_img.transpose(2, 0, 1)  # Change from (H, W, 2) to (2, H, W)
        
        # Convert to tensor
        l_tensor = torch.from_numpy(l_img)
        ab_tensor = torch.from_numpy(ab_img)
        
        if self.transform:
            l_tensor = self.transform(l_tensor)
        
        return l_tensor, ab_tensor

# Define the colorization model (U-Net architecture)
class UNetColorizer(nn.Module):
    def __init__(self):
        super(UNetColorizer, self).__init__()
        
        # Encoder
        self.enc1 = self._encoder_block(1, 64)
        self.enc2 = self._encoder_block(64, 128)
        self.enc3 = self._encoder_block(128, 256)
        self.enc4 = self._encoder_block(256, 512)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec4 = self._decoder_block(1024 + 512, 512)
        self.dec3 = self._decoder_block(512 + 256, 256)
        self.dec2 = self._decoder_block(256 + 128, 128)
        self.dec1 = self._decoder_block(128 + 64, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, 2, 1)  # Output 2 channels (ab)
        
    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)
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


# Function to convert L and AB to RGB
def lab_to_rgb(L, ab):
    """
    Convert L and ab channels to RGB image
    L: tensor [1, H, W] in range [-1, 1]
    ab: tensor [2, H, W] in range [-1, 1]
    """
    # Convert to numpy and denormalize
    L_np = ((L + 1) * 127.5).cpu().numpy().astype(np.uint8)
    ab_np = ((ab + 1) * 127.5).cpu().numpy().astype(np.uint8)
    
    # Reshape
    L_np = L_np[0]  # (H, W)
    ab_np = np.transpose(ab_np, (1, 2, 0))  # (H, W, 2)
    
    # Create LAB image
    lab = np.zeros((L_np.shape[0], L_np.shape[1], 3), dtype=np.uint8)
    lab[:, :, 0] = L_np
    lab[:, :, 1:] = ab_np
    
    # Convert to RGB
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return rgb


# Training function
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_dir):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i, (L, ab) in enumerate(pbar):
                # Move to device
                L = L.to(device, dtype=torch.float32)
                ab = ab.to(device, dtype=torch.float32)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(L)
                loss = criterion(output, ab)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                
        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Plot and save training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()
    
    return train_losses, val_losses


# Validation function
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for L, ab in val_loader:
            # Move to device
            L = L.to(device, dtype=torch.float32)
            ab = ab.to(device, dtype=torch.float32)
            
            # Forward pass
            output = model(L)
            loss = criterion(output, ab)
            
            val_loss += loss.item()
    
    return val_loss / len(val_loader)


# Test function
def test(model, test_loader, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (L, ab_gt) in enumerate(tqdm(test_loader, desc="Testing")):
            # Move to device
            L = L.to(device, dtype=torch.float32)
            
            # Forward pass
            ab_pred = model(L)
            
            # Convert back to RGB and save first 10 images
            if i < 10:
                for j in range(L.size(0)):
                    # Get original grayscale image
                    gray_img = ((L[j, 0] + 1) * 127.5).cpu().numpy().astype(np.uint8)
                    
                    # Get ground truth color image
                    gt_color = lab_to_rgb(L[j], ab_gt[j])
                    
                    # Get predicted color image
                    pred_color = lab_to_rgb(L[j], ab_pred[j].cpu())
                    
                    # Create a comparison image
                    comparison = np.hstack([
                        cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR),
                        gt_color,
                        pred_color
                    ])
                    
                    # Save the comparison
                    cv2.imwrite(
                        os.path.join(output_dir, f'test_{i}_{j}.png'),
                        comparison
                    )


def main():
    # Paths to dataset
    data_root = "/kaggle/input/splitted-dataparth/split"
    train_l_dir = os.path.join(data_root, "train", "L")
    train_ab_dir = os.path.join(data_root, "train", "AB")
    val_l_dir = os.path.join(data_root, "val", "L")
    val_ab_dir = os.path.join(data_root, "val", "AB")
    test_l_dir = os.path.join(data_root, "test", "L")
    test_ab_dir = os.path.join(data_root, "test", "AB")
    
    # Hyperparameters
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 5
    
    # Create datasets
    train_dataset = ColorizationDataset(train_l_dir, train_ab_dir)
    val_dataset = ColorizationDataset(val_l_dir, val_ab_dir)
    test_dataset = ColorizationDataset(test_l_dir, test_ab_dir)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    # Initialize model
    model = UNetColorizer()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Output directories
    save_dir = "/kaggle/working/checkpoints"
    test_output_dir = "/kaggle/working/test_results"
    
    # Train and validate
    print("Starting training...")
    train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_dir)
    
    # Load best model for testing
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    print("Testing model...")
    test(model, test_loader, test_output_dir)
    print(f"Test results saved to {test_output_dir}")


if __name__ == "__main__":
    main()