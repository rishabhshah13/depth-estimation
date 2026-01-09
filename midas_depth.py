"""
MiDaS Depth Estimation Module

This module provides a simple interface for monocular depth estimation using the MiDaS model.
It supports loading pretrained models and performing depth estimation on images.

Requirements:
    - torch
    - torchvision
    - opencv-python
    - numpy
    - timm
    - einops

Usage as a module:
    from midas_depth import MiDaSDepthEstimator

    estimator = MiDaSDepthEstimator(model_type='midas_v21_small_256')
    depth_map = estimator.estimate_depth('path/to/image.jpg')
    estimator.save_depth_map(depth_map, 'output.png')

Usage as a script:
    python midas_depth.py --input path/to/image.jpg --output output.png
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import warnings

# Add MiDaS directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
MIDAS_DIR = SCRIPT_DIR / "MiDaS"

if not MIDAS_DIR.exists():
    raise FileNotFoundError(
        f"MiDaS directory not found at {MIDAS_DIR}. "
        "Please clone the MiDaS repository first: "
        "git clone https://github.com/isl-org/MiDaS"
    )

sys.path.insert(0, str(MIDAS_DIR))

try:
    from midas.model_loader import load_model, default_models
    import utils as midas_utils
except ImportError as e:
    raise ImportError(
        f"Failed to import MiDaS modules: {e}. "
        "Make sure the MiDaS repository is properly cloned."
    )


class MiDaSDepthEstimator:
    """
    A class for performing monocular depth estimation using MiDaS models.

    Attributes:
        model_type (str): Type of MiDaS model to use
        device (torch.device): Device to run inference on (CPU or CUDA)
        model: Loaded MiDaS model
        transform: Image transformation pipeline
        net_w (int): Network input width
        net_h (int): Network input height
    """

    AVAILABLE_MODELS = {
        'midas_v21_small_256': 'Small fast model (256x256) - best for quick inference',
        'midas_v21_384': 'MiDaS v2.1 standard model (384x384)',
        'dpt_swin2_tiny_256': 'Tiny transformer model (256x256)',
        'dpt_levit_224': 'LeViT-based model (224x224)',
        'dpt_hybrid_384': 'DPT Hybrid model (384x384)',
        'dpt_large_384': 'DPT Large model (384x384) - good quality',
        'dpt_swin2_large_384': 'DPT Swin2 Large (384x384) - high quality',
        'dpt_beit_large_512': 'BEiT Large model (512x512) - highest quality',
    }

    def __init__(
        self,
        model_type: str = 'midas_v21_small_256',
        optimize: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize the MiDaS depth estimator.

        Args:
            model_type (str): Type of model to use. Default is 'midas_v21_small_256'
            optimize (bool): Use half-precision optimization on CUDA. Default is False
            device (str, optional): Device to use ('cuda' or 'cpu'). Auto-detected if None

        Raises:
            ValueError: If model_type is not recognized
            RuntimeError: If model loading fails
        """
        if model_type not in default_models:
            available = ', '.join(self.AVAILABLE_MODELS.keys())
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Available models: {available}"
            )

        self.model_type = model_type
        self.optimize = optimize

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Initializing MiDaS Depth Estimator")
        print(f"  Model: {model_type}")
        print(f"  Device: {self.device}")

        # Get model path
        model_path = str(MIDAS_DIR / default_models[model_type])

        # Check if model weights exist
        if not os.path.exists(model_path):
            print(f"\nModel weights not found at: {model_path}")
            self._download_model_weights(model_type, model_path)

        # Load model
        try:
            self.model, self.transform, self.net_w, self.net_h = load_model(
                self.device,
                model_path,
                model_type,
                optimize=optimize,
                height=None,
                square=False
            )
            print("Model loaded successfully!\n")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _download_model_weights(self, model_type: str, model_path: str):
        """
        Attempt to download model weights if they don't exist.

        Args:
            model_type (str): Type of model
            model_path (str): Path where weights should be saved
        """
        model_urls = {
            'midas_v21_small_256': 'https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt',
            'midas_v21_384': 'https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt',
            'dpt_large_384': 'https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt',
            'dpt_hybrid_384': 'https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt',
            'dpt_beit_large_512': 'https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt',
            'dpt_swin2_tiny_256': 'https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt',
            'dpt_levit_224': 'https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt',
            'dpt_swin2_large_384': 'https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt',
        }

        if model_type in model_urls:
            print(f"Downloading model weights from {model_urls[model_type]}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            try:
                torch.hub.download_url_to_file(
                    model_urls[model_type],
                    model_path,
                    progress=True
                )
                print(f"Download complete: {model_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download model weights: {e}\n"
                    f"Please download manually from {model_urls[model_type]} "
                    f"and place at {model_path}"
                )
        else:
            raise FileNotFoundError(
                f"Model weights not found and automatic download not available for {model_type}. "
                f"Please download manually and place at {model_path}"
            )

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess an image.

        Args:
            image_path (str): Path to the image file

        Returns:
            np.ndarray: Loaded image in RGB format (0-1 range)

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Convert grayscale to BGR if needed
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Convert BGR to RGB and normalize to 0-1
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

            return img_rgb
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")

    def estimate_depth(
        self,
        image_input: Union[str, np.ndarray],
        return_original: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate depth map from an image.

        Args:
            image_input (Union[str, np.ndarray]): Path to image file or numpy array (RGB, 0-1 range)
            return_original (bool): If True, return tuple of (depth_map, original_image)

        Returns:
            np.ndarray or tuple: Depth map as numpy array, or tuple of (depth_map, original_image)

        Raises:
            ValueError: If image input is invalid
        """
        # Load image if path is provided
        if isinstance(image_input, str):
            original_image = self.load_image(image_input)
        elif isinstance(image_input, np.ndarray):
            original_image = image_input.copy()
        else:
            raise ValueError("image_input must be a file path string or numpy array")

        # Validate image
        if original_image.ndim != 3 or original_image.shape[2] != 3:
            raise ValueError(f"Image must be RGB (H, W, 3), got shape {original_image.shape}")

        # Apply transforms
        try:
            sample = self.transform({"image": original_image})
            image_tensor = sample["image"]
        except Exception as e:
            raise RuntimeError(f"Error applying transforms: {e}")

        # Prepare input
        input_tensor = torch.from_numpy(image_tensor).to(self.device).unsqueeze(0)

        # Apply optimization if needed
        if self.optimize and self.device.type == 'cuda':
            input_tensor = input_tensor.to(memory_format=torch.channels_last)
            input_tensor = input_tensor.half()

        # Run inference
        with torch.no_grad():
            try:
                prediction = self.model.forward(input_tensor)
            except Exception as e:
                raise RuntimeError(f"Error during model inference: {e}")

        # Resize prediction to original image size
        original_height, original_width = original_image.shape[:2]
        target_size = (original_width, original_height)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(original_height, original_width),
            mode='bicubic',
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Handle non-finite values
        if not np.isfinite(prediction).all():
            warnings.warn("Non-finite depth values detected. Replacing with zeros.")
            prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)

        if return_original:
            return prediction, original_image
        return prediction

    def save_depth_map(
        self,
        depth_map: np.ndarray,
        output_path: str,
        colormap: str = 'inferno',
        normalize: bool = True
    ):
        """
        Save depth map as an image file.

        Args:
            depth_map (np.ndarray): Depth map to save
            output_path (str): Path to save the image
            colormap (str): Colormap to apply ('inferno', 'magma', 'viridis', 'gray', etc.)
            normalize (bool): Whether to normalize depth values to 0-255 range
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        if normalize:
            depth_min = depth_map.min()
            depth_max = depth_map.max()

            if depth_max - depth_min > np.finfo('float').eps:
                normalized = 255 * (depth_map - depth_min) / (depth_max - depth_min)
            else:
                normalized = np.zeros_like(depth_map)
        else:
            normalized = depth_map

        normalized = normalized.astype(np.uint8)

        # Apply colormap
        if colormap.lower() == 'gray':
            output_image = normalized
        else:
            colormap_dict = {
                'inferno': cv2.COLORMAP_INFERNO,
                'magma': cv2.COLORMAP_MAGMA,
                'plasma': cv2.COLORMAP_PLASMA,
                'viridis': cv2.COLORMAP_VIRIDIS,
                'jet': cv2.COLORMAP_JET,
                'hot': cv2.COLORMAP_HOT,
            }
            cv2_colormap = colormap_dict.get(colormap.lower(), cv2.COLORMAP_INFERNO)
            output_image = cv2.applyColorMap(normalized, cv2_colormap)

        cv2.imwrite(output_path, output_image)
        print(f"Depth map saved to: {output_path}")

    def visualize_side_by_side(
        self,
        original_image: np.ndarray,
        depth_map: np.ndarray,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create a side-by-side visualization of original image and depth map.

        Args:
            original_image (np.ndarray): Original RGB image (0-1 range)
            depth_map (np.ndarray): Depth map
            output_path (str, optional): If provided, save the visualization

        Returns:
            np.ndarray: Side-by-side visualization
        """
        # Normalize depth map
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized_depth = 255 * (depth_map - depth_min) / (depth_max - depth_min)

        # Apply colormap to depth
        depth_colored = cv2.applyColorMap(np.uint8(normalized_depth), cv2.COLORMAP_INFERNO)

        # Convert original image to BGR and scale to 0-255
        original_bgr = cv2.cvtColor((original_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Concatenate side by side
        side_by_side = np.concatenate([original_bgr, depth_colored], axis=1)

        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            cv2.imwrite(output_path, side_by_side)
            print(f"Side-by-side visualization saved to: {output_path}")

        return side_by_side

    @staticmethod
    def list_available_models():
        """Print all available models with descriptions."""
        print("\nAvailable MiDaS Models:")
        print("-" * 80)
        for model_name, description in MiDaSDepthEstimator.AVAILABLE_MODELS.items():
            print(f"  {model_name:30s} - {description}")
        print("-" * 80 + "\n")


def main():
    """Command-line interface for MiDaS depth estimation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='MiDaS Depth Estimation - Estimate depth from a single image',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input image'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='depth_output.png',
        help='Path to output depth map image (default: depth_output.png)'
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        default='midas_v21_small_256',
        help='Model type to use (default: midas_v21_small_256)'
    )

    parser.add_argument(
        '--colormap',
        type=str,
        default='inferno',
        choices=['inferno', 'magma', 'plasma', 'viridis', 'jet', 'hot', 'gray'],
        help='Colormap for depth visualization (default: inferno)'
    )

    parser.add_argument(
        '--side-by-side',
        action='store_true',
        help='Create side-by-side visualization of input and depth'
    )

    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Use half-precision optimization on CUDA'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use for inference (auto-detected if not specified)'
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models and exit'
    )

    args = parser.parse_args()

    if args.list_models:
        MiDaSDepthEstimator.list_available_models()
        return

    try:
        # Initialize estimator
        estimator = MiDaSDepthEstimator(
            model_type=args.model,
            optimize=args.optimize,
            device=args.device
        )

        # Estimate depth
        print(f"Processing image: {args.input}")
        depth_map, original_image = estimator.estimate_depth(
            args.input,
            return_original=True
        )

        print(f"Depth map shape: {depth_map.shape}")
        print(f"Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")

        # Save outputs
        if args.side_by_side:
            base_name = os.path.splitext(args.output)[0]
            side_by_side_path = f"{base_name}_side_by_side.png"
            estimator.visualize_side_by_side(original_image, depth_map, side_by_side_path)

        estimator.save_depth_map(depth_map, args.output, colormap=args.colormap)

        print("\nProcessing complete!")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
