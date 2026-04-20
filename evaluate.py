import os
import torch
import numpy as np
from tqdm import tqdm

from dataset import get_dataloader
from model import UNet

def calculate_metrics(pred, target):
    """
    Calculate Dice, IoU, Precision, and Recall manually.
    pred and target should be 1D binary numpy arrays.
    """
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    # True Positives, False Positives, False Negatives
    tp = intersection
    fp = (pred & ~target).sum()
    fn = (~pred & target).sum()
    
    # Metrics
    dice = (2. * tp) / (pred.sum() + target.sum() + 1e-6)
    iou = tp / (union + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    return dice, iou, precision, recall

def evaluate():
    test_images_dir_full = 'ISBI2016_ISIC_Part1_Test_Data'
    test_masks_dir_full = 'ISBI2016_ISIC_Part1_Test_GroundTruth'
    model_path = 'best_unet_model.pth'
    
    if not os.path.exists(test_images_dir_full):
        print(f"Directory {test_images_dir_full} not found.")
        return
        
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return

    if 'COLAB_TPU_ADDR' in os.environ: # Check for TPU environment
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"Evaluating using device: XLA (TPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Evaluating using device: CUDA")
    else:
        device = torch.device('cpu')
        print(f"Evaluating using device: CPU")

    test_loader = get_dataloader(test_images_dir_full, test_masks_dir_full, batch_size=8, shuffle=False)
    
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_dice, all_iou, all_precision, all_recall = [], [], [], []
    
    print("Running evaluation on test set...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device).cpu().numpy()
            
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            # Binarize Predictions
            preds = (preds > 0.5).astype(bool)
            masks = masks.astype(bool)
            
            for i in range(preds.shape[0]):
                p = preds[i].flatten()
                m = masks[i].flatten()
                
                dice, iou, precision, recall = calculate_metrics(p, m)
                all_dice.append(dice)
                all_iou.append(iou)
                all_precision.append(precision)
                all_recall.append(recall)

    print("\n--- Final Test Set Results ---")
    print(f"Mean Dice Coefficient : {np.mean(all_dice):.4f}")
    print(f"Mean IoU Score        : {np.mean(all_iou):.4f}")
    print(f"Mean Precision        : {np.mean(all_precision):.4f}")
    print(f"Mean Recall           : {np.mean(all_recall):.4f}")
    print("------------------------------")

if __name__ == '__main__':
    evaluate()
