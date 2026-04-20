import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ISICDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=256):
        """
        Args:
            images_dir (str): Path to the directory with images (e.g., ISBI2016_ISIC_Part1_Training_Data).
            masks_dir (str): Path to the directory with masks (e.g., ISBI2016_ISIC_Part1_Training_GroundTruth).
            img_size (int): Size to resize the images to (default 256).
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        
        self.images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # In ISBI 2016, masks have _Segmentation.png suffix
        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}_Segmentation.png"
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        
        # Handle cases where ground truth might be missing or wrongly named
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            # Fallback for inference without mask or debug
            mask = Image.new('L', image.size, color=0)

        # Custom transforms using PIL and Numpy
        image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)

        # Convert to numpy array and normalize image to [0, 1]
        img_np = np.array(image, dtype=np.float32) / 255.0
        mask_np = np.array(mask, dtype=np.float32) / 255.0

        # Change image shape from (H, W, C) to (C, H, W)
        img_np = np.transpose(img_np, (2, 0, 1))
        # Mask needs a channel dimension (1, H, W)
        mask_np = np.expand_dims(mask_np, axis=0)

        # Convert to torch tensors
        image_tensor = torch.from_numpy(img_np)
        mask_tensor = torch.from_numpy(mask_np)

        # Ensure mask is strictly binary (0 or 1)
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor

def get_dataloader(images_dir, masks_dir, batch_size=16, img_size=256, shuffle=True, num_workers=4):
    """
    Creates and returns a PyTorch DataLoader for the ISIC dataset.
    """
    dataset = ISICDataset(images_dir, masks_dir, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == '__main__':
    # Simple test to verify shapes
    test_dset = ISICDataset('ISBI2016_ISIC_Part1_Training_Data', 'ISBI2016_ISIC_Part1_Training_GroundTruth')
    if len(test_dset) > 0:
        img, msk = test_dset[0]
        print(f"Image shape: {img.shape}")
        print(f"Mask shape: {msk.shape}")
        print(f"Dataset length: {len(test_dset)}")
    else:
        print("Dataset folders not found or empty.")
