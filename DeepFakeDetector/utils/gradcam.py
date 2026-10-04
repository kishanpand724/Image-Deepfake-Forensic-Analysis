"""
Full-Image Grad-CAM (Gradient-weighted Class Activation Mapping).

Computes activation heatmaps over the ENTIRE uploaded image (not restricted to face crops),
reversing letterbox padding to align perfectly with the original dimensions.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def colormap_jet(values: np.ndarray) -> np.ndarray:
    """
    Applies a Jet colormap to a 2D normalized array [0, 1] without OpenCV dependency.
    Returns RGB uint8 array [H, W, 3].
    """
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def colormap_turbo(values: np.ndarray) -> np.ndarray:
    """
    Applies Turbo colormap approximation for smooth gradient visualization.
    """
    x = np.clip(values, 0.0, 1.0)
    # Polynomial approximation for Turbo colormap
    r = 0.1357 + x * (4.5974 - x * (42.3277 - x * (130.5887 - x * (150.5667 - x * 58.1375))))
    g = 0.0914 + x * (2.1856 + x * (4.8052 - x * (14.0195 - x * (4.2109 - x * 2.7747))))
    b = 0.1067 + x * (12.5732 - x * (80.1473 - x * (205.2917 - x * (214.3644 - x * 77.0189))))
    rgb = np.stack([np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)], axis=-1)
    return (rgb * 255).astype(np.uint8)


class FullImageGradCAM:
    """
    Generates class activation heatmaps across the entire full-frame image.
    Extracts gradient responses from deep convolutional layers to pinpoint:
    - Blending boundary seam artifacts
    - Background texture synthesis anomalies
    - Facial warping / inpainting mismatches
    - High-frequency edge residuals
    """
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else getattr(model, "layer4", None)
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        if self.target_layer is None:
            return

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks = [h1, h2]

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate_heatmap(
        self,
        tensor: torch.Tensor,
        target_class: int = 0
    ) -> np.ndarray:
        """
        Computes 2D normalized Grad-CAM activation heatmap [target_h, target_w].
        """
        self.model.eval()
        tensor = tensor.requires_grad_(True)
        logits, features = self.model(tensor)

        score = logits[0, 0] if logits.shape[-1] == 1 else logits[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback to feature map spatial activation if hooks were inactive
            features_act = features.detach()
            weights = torch.mean(features_act, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * features_act, dim=1).squeeze(0)
        else:
            # Standard Grad-CAM: Global average pooling over gradients
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1).squeeze(0)

        # Apply ReLU to focus on features that have positive influence
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def overlay_on_image(
        self,
        original_img: Image.Image,
        heatmap_2d: np.ndarray,
        pad_box: Tuple[int, int, int, int],
        alpha: float = 0.55,
        colormap: str = "turbo"
    ) -> Tuple[Image.Image, Image.Image, List[Dict[str, Any]]]:
        """
        Maps the letterboxed 2D heatmap back to the full original image dimensions,
        and composites the overlay.
        
        Returns:
            overlay_image: Full image with blended heatmap
            raw_heatmap_image: Colorized heatmap scaled to full image size
            suspicious_regions: List of bounding boxes identifying localized anomaly peaks
        """
        orig_w, orig_h = original_img.size
        pad_left, pad_top, pad_right, pad_bottom = pad_box

        # Convert heatmap to PIL for aspect-preserving unpadding
        heat_pil = Image.fromarray((heatmap_2d * 255).astype(np.uint8))
        target_w, target_h = heat_pil.size

        # Crop out the letterbox padding
        active_w = max(1, pad_right - pad_left)
        active_h = max(1, pad_bottom - pad_top)
        
        # Scale coords to heatmap resolution if heatmap is downsampled
        scale_x = target_w / max(1, (pad_right + pad_left)) # approximate
        crop_box = (
            int(pad_left * target_w / 256),
            int(pad_top * target_h / 256),
            int(pad_right * target_w / 256),
            int(pad_bottom * target_h / 256),
        )
        # Ensure inside bounds
        crop_box = (
            max(0, min(crop_box[0], target_w - 1)),
            max(0, min(crop_box[1], target_h - 1)),
            max(1, min(crop_box[2], target_w)),
            max(1, min(crop_box[3], target_h)),
        )

        unpadded_heat = heat_pil.crop(crop_box)
        # Resize to EXACT original image dimensions
        full_heatmap_resized = unpadded_heat.resize((orig_w, orig_h), Image.Resampling.BICUBIC)
        norm_full_heat = np.array(full_heatmap_resized).astype(np.float32) / 255.0

        # Apply chosen colormap
        if colormap == "jet":
            colored_heat_np = colormap_jet(norm_full_heat)
        else:
            colored_heat_np = colormap_turbo(norm_full_heat)

        colored_heat_pil = Image.fromarray(colored_heat_np)

        # Alpha composite with original image
        orig_rgb = original_img.convert("RGB")
        overlay = Image.blend(orig_rgb, colored_heat_pil, alpha=alpha)

        # Extract peak suspicious zones (hotspots with > 0.70 activation)
        suspicious_regions = []
        hotspot_mask = norm_full_heat > 0.65
        if np.any(hotspot_mask):
            rows, cols = np.where(hotspot_mask)
            y_min, y_max = int(rows.min()), int(rows.max())
            x_min, x_max = int(cols.min()), int(cols.max())
            peak_val = float(np.max(norm_full_heat))
            suspicious_regions.append({
                "bbox": [x_min, y_min, x_max, y_max],
                "peak_intensity": round(peak_val, 3),
                "region_type": "High Activation Anomaly Zone"
            })

        return overlay, colored_heat_pil, suspicious_regions
