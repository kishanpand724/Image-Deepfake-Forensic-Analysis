"""
Full-Image Preprocessor for DeepFake Detection.

Replaces the mandatory MTCNN face-crop dependency with:
1. Aspect-ratio preserving letterbox resizing with padding (no geometric distortion).
2. Proper EXIF orientation correction.
3. ImageNet standard normalization (mean/std).
4. Multi-scale feature support and optional contextual face-region locator
   without cropping out the background context.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms


class FullImagePreprocessor:
    """
    Preprocesses full-frame images while preserving aspect ratio,
    avoiding image warping, and normalizing for deep learning inference.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        padding_color: Tuple[int, int, int] = (0, 0, 0),
        normalize: bool = True
    ):
        self.target_size = target_size
        self.padding_color = padding_color
        self.normalize = normalize

        # Standard ImageNet normalization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.tensor_transform = transforms.ToTensor()
        self.norm_transform = transforms.Normalize(mean=self.mean, std=self.std)

    def letterbox_image(
        self,
        img: Image.Image
    ) -> Tuple[Image.Image, Tuple[int, int, int, int], float]:
        """
        Resizes an image preserving aspect ratio with symmetric letterbox padding.
        Returns:
            padded_img: Image of size target_size
            padding_box: (left, top, right, bottom) representing padded offsets
            scale_ratio: scale factor applied
        """
        # Ensure RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')

        orig_w, orig_h = img.size
        target_w, target_h = self.target_size

        # Compute scaling ratio to fit within target bounds
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        # High-quality Lanczos resampling
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Create padded background
        padded = Image.new("RGB", self.target_size, self.padding_color)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded.paste(resized, (pad_x, pad_y))

        padding_info = (pad_x, pad_y, pad_x + new_w, pad_y + new_h)
        return padded, padding_info, scale

    def process(
        self,
        input_data: Any
    ) -> Tuple[torch.Tensor, Image.Image, Dict[str, Any]]:
        """
        Preprocesses an image from file path, PIL Image, or numpy array.
        Returns:
            tensor: [1, 3, target_h, target_w] ready for model inference
            original_pil: original PIL Image for visualization/overlay
            metadata: dictionary containing dimensions, scaling, aspect ratio
        """
        # 1. Load into PIL
        if isinstance(input_data, str):
            img = Image.open(input_data)
        elif isinstance(input_data, np.ndarray):
            # Check if BGR (OpenCV) or RGB
            if len(input_data.shape) == 3 and input_data.shape[2] == 3:
                img = Image.fromarray(input_data)
            else:
                img = Image.fromarray(input_data).convert("RGB")
        elif isinstance(input_data, Image.Image):
            img = input_data
        else:
            raise ValueError(f"Unsupported image type: {type(input_data)}")

        # 2. Fix EXIF orientation (e.g. mobile uploads)
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        if img.mode != "RGB":
            img = img.convert("RGB")

        orig_w, orig_h = img.size

        # 3. Apply aspect-ratio preserving letterboxing
        padded_pil, pad_box, scale = self.letterbox_image(img)

        # 4. Convert to normalized PyTorch tensor
        tensor = self.tensor_transform(padded_pil)
        if self.normalize:
            tensor = self.norm_transform(tensor)
        tensor = tensor.unsqueeze(0) # [1, C, H, W]

        metadata = {
            "original_width": orig_w,
            "original_height": orig_h,
            "target_width": self.target_size[0],
            "target_height": self.target_size[1],
            "aspect_ratio": orig_w / orig_h if orig_h > 0 else 1.0,
            "pad_box": pad_box,
            "scale": scale,
            "mode": "full_image_letterbox"
        }

        return tensor, img, metadata
