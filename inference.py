import os
import io
import base64
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg') # Disable interactive plots for headless execution
import matplotlib.pyplot as plt
from PIL import Image

from dataset import ISICDataset
from model import UNet
from explainability import get_cam_image
from explanation import generate_natural_language_explanation

def run_inference(image_path, model_path='best_unet_model.pth'):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found. Please train first.")
        return

    if 'COLAB_TPU_ADDR' in os.environ: # Check for TPU environment
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"Inference using device: XLA (TPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Inference using device: CUDA")
    else:
        device = torch.device('cpu')
        print(f"Inference using device: CPU")
    
    # Load model
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Manual preprocessing reusing dataset logic
    raw_img = Image.open(image_path).convert('RGB')
    resized_img = raw_img.resize((256, 256), Image.Resampling.BILINEAR)
    
    img_np = np.array(resized_img, dtype=np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    
    input_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)
    
    # Original image for plotting
    original_image_rgb = np.array(resized_img) / 255.0

    # 1. Forward Pass
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask_logit = torch.sigmoid(output).squeeze()
        pred_mask = (pred_mask_logit > 0.5).cpu().numpy().astype(int)

    # 2. XAI Grad-CAM++
    cam_visualization, grayscale_cam = get_cam_image(model, input_tensor, original_image_rgb, method='gradcam++')

    # 3. Structured XAI Report
    image_name = os.path.basename(image_path)
    report = generate_natural_language_explanation(pred_mask, grayscale_cam, image_name=image_name)

    # 4. Save the .txt results file
    txt_output = 'xai_report.txt'
    with open(txt_output, 'w', encoding='utf-8') as f:
        f.write(report['full_report'])
    print(f"XAI report saved to '{txt_output}'")

    # 4. Generate 4-panel Plot
    fig = plt.figure(figsize=(20, 5))
    
    # Base styling
    plt.style.use('dark_background')
    
    # Panel 1: Original
    ax1 = plt.subplot(1, 4, 1)
    ax1.imshow(original_image_rgb)
    ax1.set_title("1. Original Image", fontsize=14, pad=10)
    ax1.axis('off')

    # Panel 2: Predicted Mask
    ax2 = plt.subplot(1, 4, 2)
    ax2.imshow(pred_mask, cmap='gray')
    ax2.set_title("2. Predicted Mask", fontsize=14, pad=10)
    ax2.axis('off')

    # Panel 3: XAI Heatmap
    ax3 = plt.subplot(1, 4, 3)
    ax3.imshow(cam_visualization)
    ax3.set_title("3. Grad-CAM++ Explainability", fontsize=14, pad=10)
    ax3.axis('off')

    # Panel 4: Text Summary  
    ax4 = plt.subplot(1, 4, 4)
    ax4.axis('off')
    ax4.set_facecolor('#0d1117')

    import textwrap
    narrative = report['narrative_only']
    confidence = report['confidence']
    
    conf_colors = {'High': '#4ade80', 'Moderate': '#facc15', 'Low': '#f87171'}
    conf_color = conf_colors.get(confidence, 'white')
    
    ax4.text(0.05, 0.97, "4. Diagnostic Summary", fontsize=13, fontweight='bold',
             color='white', transform=ax4.transAxes, verticalalignment='top')
    ax4.text(0.05, 0.89, f"Confidence: {confidence}", fontsize=11, color=conf_color,
             transform=ax4.transAxes, verticalalignment='top')
    
    wrapped = "\n".join(textwrap.wrap(narrative, width=38))
    ax4.text(0.05, 0.82, wrapped, fontsize=9, verticalalignment='top',
             color='lightgreen', transform=ax4.transAxes, wrap=True)
    
    disclaimer = "\nDISCLAIMER: AI-assisted analysis only.\nMust be reviewed by a dermatologist."
    ax4.text(0.05, 0.08, disclaimer, fontsize=8.5, verticalalignment='bottom',
             color='lightcoral', transform=ax4.transAxes)

    plt.tight_layout()
    output_filename = 'inference_result.png'
    plt.savefig(output_filename, facecolor='black', edgecolor='none')
    print(f"Inference complete! Saved result to '{output_filename}'")
    plt.show()

def run_inference_api(image_bytes, model_path='best_unet_model.pth'):
    """Headless inference for Flask Backend."""
    if 'COLAB_TPU_ADDR' in os.environ:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    raw_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    resized_img = raw_img.resize((256, 256), Image.Resampling.BILINEAR)
    
    img_np = np.array(resized_img, dtype=np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    
    input_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)
    original_image_rgb = np.array(resized_img) / 255.0

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask_probs = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # 1. Apply a Center-Weighted Gaussian Filter to suppress edge artifacts 
        # (Medical dermoscopic images often have dark vignetting rings on the edges that untrained models confuse for lesions)
        import cv2
        h, w = pred_mask_probs.shape
        X = np.linspace(-1, 1, w)
        Y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(X, Y)
        # 2D Gaussian matching the image center
        center_weight = np.exp(-(X**2 + Y**2) / 0.8) 
        
        weighted_probs = pred_mask_probs * center_weight
        
        # 2. Use Otsu's algorithm to dynamically find the perfect split between foreground/background 
        # instead of a hardcoded threshold.
        prob_uint8 = np.uint8(np.clip(weighted_probs * 255, 0, 255))
        blurred = cv2.GaussianBlur(prob_uint8, (11, 11), 0)
        _, base_mask = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Sanity check limits
        total_px = base_mask.size
        lesion_px = np.sum(base_mask)
        if lesion_px > (0.80 * total_px) or lesion_px < 100:
            pred_mask = np.zeros_like(base_mask)
        else:
            # 4. Connected components to isolate only the single largest contiguous lesion
            base_mask_uint = np.uint8(base_mask)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(base_mask_uint, connectivity=8)
            
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                pred_mask = (labels == largest_label).astype(int)
            else:
                pred_mask = base_mask

    cam_visualization, grayscale_cam = get_cam_image(model, input_tensor, original_image_rgb, method='gradcam++')
    report = generate_natural_language_explanation(pred_mask, grayscale_cam, image_name="uploaded_image.jpg")

    def array_to_b64(img_array, is_gray=False):
        import cv2
        if is_gray:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
        success, buffer = cv2.imencode('.jpg', img_array)
        return base64.b64encode(buffer).decode('utf-8')

    results = {
        'original': array_to_b64(original_image_rgb),
        'mask': array_to_b64(pred_mask, is_gray=True),
        'heatmap': array_to_b64(cam_visualization),
        'explanation': report['narrative_only'],
        'full_report': report['full_report'],
        'confidence': report['confidence']
    }
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run XAI Inference on a single skin lesion image.')
    parser.add_argument('--image', type=str, help='Path to the input image file (e.g. .jpg)')
    args = parser.parse_args()
    
    if args.image:
        run_inference(args.image)
    else:
        # Default fallback for testing
        test_dir = 'ISBI2016_ISIC_Part1_Test_Data'
        if os.path.exists(test_dir):
            sample_img = os.path.join(test_dir, os.listdir(test_dir)[0])
            print(f"No image provided. Running inference on sample: {sample_img}")
            run_inference(sample_img)
        else:
            print("Please provide an image using python inference.py --image path/to/image.jpg")
