"""
Depth Anything V2 - Depth Estimation Module

This module provides a simplified interface for depth estimation using the Depth Anything V2 model.
It can be used as both a standalone script and imported as a module.

Usage as a script:
    python depth_anything.py --image-path /path/to/image.jpg --output-path /path/to/output.png

Usage as a module:
    from depth_anything import DepthEstimator

    estimator = DepthEstimator(model_size='small')
    depth_map = estimator.estimate_depth('image.jpg')
    estimator.save_depth_map(depth_map, 'depth.png')

Author: Generated with Claude Code
License: Apache-2.0 (for small model), CC-BY-NC-4.0 (for base/large models)
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Union, Tuple, Optional
import logging

import numpy as np
import cv2
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DepthEstimator:
    """
    A wrapper class for Depth Anything V2 model that provides easy-to-use depth estimation.

    Attributes:
        model_size (str): Size of the model ('small', 'base', or 'large')
        device (str): Device to run the model on ('cuda', 'mps', or 'cpu')
        model: The loaded Depth Anything V2 model
        model_configs (dict): Configuration parameters for different model sizes
    """

    MODEL_CONFIGS = {
        'small': {
            'encoder': 'vits',
            'features': 64,
            'out_channels': [48, 96, 192, 384],
            'checkpoint': 'depth_anything_v2_vits.pth',
            'url': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth'
        },
        'base': {
            'encoder': 'vitb',
            'features': 128,
            'out_channels': [96, 192, 384, 768],
            'checkpoint': 'depth_anything_v2_vitb.pth',
            'url': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth'
        },
        'large': {
            'encoder': 'vitl',
            'features': 256,
            'out_channels': [256, 512, 1024, 1024],
            'checkpoint': 'depth_anything_v2_vitl.pth',
            'url': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth'
        }
    }

    def __init__(
        self,
        model_size: str = 'small',
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the Depth Estimator.

        Args:
            model_size (str): Size of the model to use ('small', 'base', or 'large').
                            Default is 'small' for faster inference.
            checkpoint_dir (str, optional): Directory containing model checkpoints.
                                          If None, uses '../models/Depth-Anything-V2/checkpoints'
            device (str, optional): Device to run the model on. If None, auto-detects
                                  the best available device (cuda > mps > cpu).

        Raises:
            ValueError: If model_size is not valid or checkpoint file is not found.
            ImportError: If required dependencies are not installed.
        """
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Invalid model_size: {model_size}. "
                f"Must be one of {list(self.MODEL_CONFIGS.keys())}"
            )

        self.model_size = model_size
        self.config = self.MODEL_CONFIGS[model_size]

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Set checkpoint directory
        if checkpoint_dir is None:
            # Try to find the Depth-Anything-V2 directory
            current_dir = Path(__file__).parent
            checkpoint_dir = current_dir.parent / "models" / "Depth-Anything-V2" / "checkpoints"
        else:
            checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = checkpoint_dir / self.config['checkpoint']

        # Initialize model
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Load the Depth Anything V2 model from checkpoint.

        Raises:
            ImportError: If Depth Anything V2 module is not found.
            FileNotFoundError: If checkpoint file is not found.
        """
        try:
            # Add Depth-Anything-V2 to path if not already there
            repo_path = Path(__file__).parent.parent / "models" / "Depth-Anything-V2"
            if repo_path.exists() and str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))

            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as e:
            raise ImportError(
                "Could not import Depth Anything V2 module. "
                "Please ensure the repository is cloned in the same directory. "
                f"Error: {e}"
            )

        # Check if checkpoint exists
        if not self.checkpoint_path.exists():
            logger.warning(
                f"Checkpoint not found at {self.checkpoint_path}. "
                f"Please download it from:\n{self.config['url']}\n"
                f"and save it to: {self.checkpoint_path}"
            )

            # Try to create checkpoints directory
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_path}\n"
                f"Download it from: {self.config['url']}"
            )

        logger.info(f"Loading {self.model_size} model from {self.checkpoint_path}")

        # Initialize model
        self.model = DepthAnythingV2(
            encoder=self.config['encoder'],
            features=self.config['features'],
            out_channels=self.config['out_channels']
        )

        # Load weights
        try:
            state_dict = torch.load(str(self.checkpoint_path), map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device).eval()
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")

    def estimate_depth(
        self,
        image_path: Union[str, Path, np.ndarray],
        input_size: int = 518
    ) -> np.ndarray:
        """
        Estimate depth from an image.

        Args:
            image_path (str, Path, or np.ndarray): Path to the image file or numpy array
                                                   representing the image (in BGR format).
            input_size (int): Input size for the model. Default is 518.
                            Larger values may give better results but slower inference.

        Returns:
            np.ndarray: Depth map as a 2D numpy array (HxW) with values in arbitrary scale.
                       Lower values indicate closer objects, higher values indicate farther objects.

        Raises:
            FileNotFoundError: If image_path is a string/Path and file doesn't exist.
            ValueError: If image could not be loaded or has invalid format.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please initialize the estimator properly.")

        # Load image
        if isinstance(image_path, (str, Path)):
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            raw_image = cv2.imread(str(image_path))
            if raw_image is None:
                raise ValueError(f"Could not load image: {image_path}")

            logger.info(f"Processing image: {image_path}")
        elif isinstance(image_path, np.ndarray):
            raw_image = image_path
            logger.info(f"Processing numpy array image of shape: {raw_image.shape}")
        else:
            raise ValueError(
                "image_path must be a file path (str/Path) or numpy array"
            )

        # Check image validity
        if len(raw_image.shape) != 3 or raw_image.shape[2] != 3:
            raise ValueError(
                f"Image must be a color image with 3 channels. "
                f"Got shape: {raw_image.shape}"
            )

        # Perform inference
        try:
            with torch.no_grad():
                depth = self.model.infer_image(raw_image, input_size)

            logger.info(
                f"Depth estimation complete. "
                f"Output shape: {depth.shape}, "
                f"Min: {depth.min():.2f}, Max: {depth.max():.2f}"
            )

            return depth
        except Exception as e:
            raise RuntimeError(f"Depth estimation failed: {e}")

    def save_depth_map(
        self,
        depth: np.ndarray,
        output_path: Union[str, Path],
        colormap: str = 'Spectral_r',
        normalize: bool = True,
        grayscale: bool = False
    ) -> None:
        """
        Save depth map as an image file.

        Args:
            depth (np.ndarray): Depth map to save (2D array).
            output_path (str or Path): Path where to save the depth map.
            colormap (str): Matplotlib colormap name. Default is 'Spectral_r'.
                          Ignored if grayscale is True.
            normalize (bool): Whether to normalize depth values to [0, 255]. Default is True.
            grayscale (bool): If True, saves as grayscale image. Default is False.

        Raises:
            ValueError: If depth array is invalid.
            IOError: If file could not be saved.
        """
        output_path = Path(output_path)

        # Validate depth
        if not isinstance(depth, np.ndarray) or len(depth.shape) != 2:
            raise ValueError("Depth must be a 2D numpy array")

        # Normalize depth
        if normalize:
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth_normalized = depth_normalized.astype(np.uint8)
        else:
            depth_normalized = depth.astype(np.uint8)

        # Apply colormap or grayscale
        if grayscale:
            depth_colored = np.repeat(depth_normalized[..., np.newaxis], 3, axis=-1)
        else:
            try:
                import matplotlib
                cmap = matplotlib.colormaps.get_cmap(colormap)
                depth_colored = (cmap(depth_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            except Exception as e:
                logger.warning(f"Could not apply colormap: {e}. Using grayscale instead.")
                depth_colored = np.repeat(depth_normalized[..., np.newaxis], 3, axis=-1)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save image
        try:
            success = cv2.imwrite(str(output_path), depth_colored)
            if not success:
                raise IOError(f"Failed to save image to {output_path}")
            logger.info(f"Depth map saved to: {output_path}")
        except Exception as e:
            raise IOError(f"Error saving depth map: {e}")

    def estimate_and_save(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        input_size: int = 518,
        colormap: str = 'Spectral_r',
        grayscale: bool = False
    ) -> np.ndarray:
        """
        Convenience method to estimate depth and save in one call.

        Args:
            image_path (str or Path): Path to input image.
            output_path (str or Path): Path to save depth map.
            input_size (int): Input size for model. Default is 518.
            colormap (str): Colormap to use. Default is 'Spectral_r'.
            grayscale (bool): Whether to save as grayscale. Default is False.

        Returns:
            np.ndarray: The depth map as a numpy array.
        """
        depth = self.estimate_depth(image_path, input_size)
        self.save_depth_map(depth, output_path, colormap, grayscale=grayscale)
        return depth


def main():
    """
    Command-line interface for depth estimation.
    """
    parser = argparse.ArgumentParser(
        description='Depth Anything V2 - Depth Estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings (small model)
  python depth_anything.py --image-path image.jpg --output-path depth.png

  # Use larger model for better quality
  python depth_anything.py --image-path image.jpg --output-path depth.png --model-size large

  # Save as grayscale
  python depth_anything.py --image-path image.jpg --output-path depth.png --grayscale

  # Custom input size for higher resolution
  python depth_anything.py --image-path image.jpg --output-path depth.png --input-size 1024
        """
    )

    parser.add_argument(
        '--image-path',
        type=str,
        required=True,
        help='Path to input image'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path to save depth map'
    )

    parser.add_argument(
        '--model-size',
        type=str,
        default='small',
        choices=['small', 'base', 'large'],
        help='Model size to use (default: small)'
    )

    parser.add_argument(
        '--input-size',
        type=int,
        default=518,
        help='Input size for model (default: 518)'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory containing model checkpoints'
    )

    parser.add_argument(
        '--colormap',
        type=str,
        default='Spectral_r',
        help='Matplotlib colormap name (default: Spectral_r)'
    )

    parser.add_argument(
        '--grayscale',
        action='store_true',
        help='Save as grayscale image'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'mps', 'cpu'],
        help='Device to run model on (default: auto-detect)'
    )

    args = parser.parse_args()

    try:
        # Initialize estimator
        estimator = DepthEstimator(
            model_size=args.model_size,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device
        )

        # Estimate and save depth
        depth = estimator.estimate_and_save(
            image_path=args.image_path,
            output_path=args.output_path,
            input_size=args.input_size,
            colormap=args.colormap,
            grayscale=args.grayscale
        )

        logger.info("Processing complete!")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
