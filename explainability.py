import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        # Forward pass
        logits = self.model(x)
        
        # For our 1-class segmentation, target the max logit sum (lesion pixels)
        target = logits.sum()
        
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # Compute weights
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam

def get_cam_image(model, input_tensor, original_image_rgb, method='gradcam++'):
    """
    Generates a Grad-CAM heatmap overlaid on the original image.
    Uses custom implementation to avoid external pip dependencies.
    """
    model.eval()

    # The bottleneck layer target for our specific UNet implementation
    target_layer = model.down4.maxpool_conv[1].double_conv[3]
    
    grad_cam = SimpleGradCAM(model, target_layer)
    grayscale_cam = grad_cam(input_tensor)
    
    # Overlay using OpenCV colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0
    
    # Mix original and heatmap
    visualization = heatmap * 0.5 + original_image_rgb * 0.5
    visualization = np.clip(visualization, 0, 1)

    return visualization, grayscale_cam
