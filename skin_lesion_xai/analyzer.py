import os
import urllib.request
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from .model import UNet
from .explainability import get_cam_image
from .explanation import generate_natural_language_explanation

# Default location for weights
DEFAULT_WEIGHTS_URL = "https://github.com/Onkarbiyani/AAIH/releases/download/v1.0.0/best_unet_model.pth"
CACHE_DIR = os.path.expanduser("~/.cache/skin_lesion_xai")
DEFAULT_WEIGHTS_PATH = os.path.join(CACHE_DIR, "best_unet_model.pth")


class Analyzer:
    """
    Main API for Explainable Skin Lesion Segmentation.
    Automatically downloads and caches the model weights upon first initialization.
    """
    
    def __init__(self, model_path=None, device=None, weights_url=DEFAULT_WEIGHTS_URL):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self._ensure_weights_exist(model_path, weights_url)
        self.model_path = model_path if model_path and os.path.exists(model_path) else DEFAULT_WEIGHTS_PATH
        
        # Load Model
        self.model = UNet(n_channels=3, n_classes=1).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        
        # Define Transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def _ensure_weights_exist(self, provided_path, url):
        """Downloads the weights file from GitHub Releases if it doesn't exist locally."""
        if provided_path and os.path.exists(provided_path):
            return
            
        if not os.path.exists(DEFAULT_WEIGHTS_PATH):
            os.makedirs(CACHE_DIR, exist_ok=True)
            print(f"Downloading model weights (~118MB) to {DEFAULT_WEIGHTS_PATH}...")
            try:
                urllib.request.urlretrieve(url, DEFAULT_WEIGHTS_PATH)
                print("Download complete!")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download weights from {url}. Please manually place 'best_unet_model.pth' "
                    f"in {CACHE_DIR} or provide a direct model_path. Error: {e}"
                )

    def analyze(self, image_input):
        """
        Analyzes a dermoscopy image, returning the prediction, heatmap, and text explanation.
        
        Args:
            image_input: Either a file path string, or a byte stream.
        """
        # Load image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image = Image.open(io.BytesIO(image_input)).convert('RGB')
            
        original_width, original_height = image.size
        
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        original_image_rgb = np.array(image.resize((256, 256))) / 255.0
        
        # 1. Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            
        # 2. Extract XAI Heatmap
        cam_visualization, grayscale_cam = get_cam_image(self.model, input_tensor, original_image_rgb)
        
        # 3. Post-processing (Otsu & Gaussian)
        center_y, center_x = 128, 128
        y, x = np.ogrid[:256, :256]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        sigma = 128
        gaussian_weight = np.exp(-(dist_from_center**2) / (2 * sigma**2))
        
        weighted_probs = prob * gaussian_weight
        
        prob_uint8 = np.uint8(np.clip(weighted_probs * 255, 0, 255))
        blurred = cv2.GaussianBlur(prob_uint8, (11, 11), 0)
        _, base_mask = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(base_mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            final_mask = (labels == largest_label).astype(np.uint8)
        else:
            final_mask = base_mask

        # 4. Natural Language Explanation
        explanation = generate_natural_language_explanation(grayscale_cam, final_mask)
        
        # Format results
        return {
            'mask': final_mask,
            'heatmap_overlay': cam_visualization,
            'raw_heatmap': grayscale_cam,
            'explanation': explanation,
            'confidence': float(np.mean(prob[final_mask == 1])) if np.any(final_mask) else 0.0
        }
