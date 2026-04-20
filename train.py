import os
import json
import torch
from torch.utils.data import random_split
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloader, ISICDataset
from model import UNet, DiceBCELoss

def calculate_dice(pred, target, smooth=1e-5):
    """
    Calculate Dice Coefficient for evaluation standard.
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def train(epochs=50, batch_size=8, learning_rate=1e-3, val_split=0.2):
    # Paths (adjust to where datasets are downloaded locally)
    train_images_dir_full = 'ISBI2016_ISIC_Part1_Training_Data'
    train_masks_dir_full = 'ISBI2016_ISIC_Part1_Training_GroundTruth'
    
    if not os.path.exists(train_images_dir_full):
        print(f"Directory {train_images_dir_full} not found. Please ensure data is present.")
        return

    # Determine device: TPU, CUDA, or CPU
    is_tpu = False
    if 'COLAB_TPU_ADDR' in os.environ: # Check for TPU environment
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        is_tpu = True
        print(f"Using device: XLA (TPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU")

    # Full training dataset
    full_dataset = ISICDataset(train_images_dir_full, train_masks_dir_full, img_size=256)
    
    # Split into train and validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Scheduler: reduce learning rate if validation dice plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_dice = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}

    print("Starting Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                loss = criterion(outputs, masks)
                loss.backward()
                
                if is_tpu:
                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()
                
                epoch_loss += loss.item()
                epoch_dice += calculate_dice(outputs, masks)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(images.shape[0])

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_dice = epoch_dice / len(train_loader)
                
        # Validation
        model.eval()
        val_dice = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                batch_loss = criterion(outputs, masks)
                val_loss += batch_loss.item()
                
                dice = calculate_dice(outputs, masks)
                val_dice += dice
                
        avg_val_dice = val_dice / len(val_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch} — Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

        # Save metrics to history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_dice'].append(avg_train_dice)
        history['val_dice'].append(avg_val_dice)

        # Write to JSON after every epoch (safe even if interrupted)
        with open('training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        scheduler.step(avg_val_dice)
        
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            print(f"  -> New best Val Dice: {best_val_dice:.4f}. Saving model...")
            torch.save(model.state_dict(), 'best_unet_model.pth')

    print(f"\nTraining complete. History saved to 'training_history.json'")

if __name__ == '__main__':
    train(epochs=50, batch_size=16)
