"""
Marigold Depth Estimation Module

This module provides a simple interface for performing monocular depth estimation
using the Marigold diffusion model. It can be used as a standalone script or
imported as a module.

Based on the Marigold project: https://marigoldcomputervision.github.io
Paper: "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation"

Example usage as a script:
    python marigold_depth.py --image_path input.jpg --output_path output.png

Example usage as a module:
    from marigold_depth import MarigoldDepthEstimator

    estimator = MarigoldDepthEstimator()
    depth_map, colored_depth = estimator.estimate_depth("input.jpg")

Requirements:
    - torch>=2.0.0
    - diffusers>=0.25.0
    - transformers>=4.32.1
    - PIL
    - numpy
    - matplotlib
"""

import sys
import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
import torch
from PIL import Image

# Add Marigold directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
MARIGOLD_DIR = SCRIPT_DIR.parent / "models" / "Marigold"

if MARIGOLD_DIR.exists():
    sys.path.insert(0, str(MARIGOLD_DIR))
else:
    raise ImportError(
        f"Marigold repository not found at {MARIGOLD_DIR}. "
        "Please ensure the Marigold repository is cloned in the same directory."
    )

try:
    from marigold import MarigoldDepthPipeline, MarigoldDepthOutput
except ImportError as e:
    raise ImportError(
        "Failed to import Marigold modules. Please ensure all dependencies are installed: "
        "pip install -r models/Marigold/requirements.txt"
    ) from e


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarigoldDepthEstimator:
    """
    A wrapper class for Marigold depth estimation that provides a simple interface
    for loading the model and performing depth estimation on images.

    Attributes:
        checkpoint (str): HuggingFace checkpoint or local path to the model
        device (torch.device): Device to run inference on (cuda, mps, or cpu)
        dtype (torch.dtype): Data type for model weights (float32 or float16)
        pipe (MarigoldDepthPipeline): The loaded Marigold pipeline
    """

    def __init__(
        self,
        checkpoint: str = "prs-eth/marigold-depth-v1-1",
        device: Optional[str] = None,
        half_precision: bool = False,
    ):
        """
        Initialize the Marigold depth estimator.

        Args:
            checkpoint: HuggingFace model checkpoint or local path
                       Default: "prs-eth/marigold-depth-v1-1"
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
            half_precision: Use FP16 for faster inference (may reduce quality)

        Raises:
            RuntimeError: If no suitable device is available
            ImportError: If required dependencies are missing
        """
        self.checkpoint = checkpoint

        # Determine device
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = self._get_default_device()

        logger.info(f"Using device: {self.device}")

        # Set precision
        if half_precision:
            self.dtype = torch.float16
            self.variant = "fp16"
            logger.info("Using half precision (FP16)")
        else:
            self.dtype = torch.float32
            self.variant = None
            logger.info("Using full precision (FP32)")

        # Load pipeline
        self.pipe = self._load_pipeline()
        logger.info("Marigold depth estimation pipeline loaded successfully")

    def _get_default_device(self) -> torch.device:
        """
        Automatically detect the best available device.

        Returns:
            torch.device: The best available device (cuda > mps > cpu)
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            logger.warning(
                "No GPU detected. Running on CPU will be significantly slower."
            )
            return torch.device("cpu")

    def _load_pipeline(self) -> MarigoldDepthPipeline:
        """
        Load the Marigold depth estimation pipeline.

        Returns:
            MarigoldDepthPipeline: Loaded pipeline ready for inference

        Raises:
            Exception: If pipeline loading fails
        """
        try:
            logger.info(f"Loading checkpoint: {self.checkpoint}")
            pipe = MarigoldDepthPipeline.from_pretrained(
                self.checkpoint,
                variant=self.variant,
                torch_dtype=self.dtype
            )

            # Enable memory efficient attention if available
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("XFormers memory efficient attention enabled")
            except (ImportError, AttributeError):
                logger.info("XFormers not available, using standard attention")

            pipe = pipe.to(self.device)

            logger.info(
                f"Pipeline loaded: scale_invariant={pipe.scale_invariant}, "
                f"shift_invariant={pipe.shift_invariant}"
            )

            return pipe

        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise

    def estimate_depth(
        self,
        image_path: Union[str, Path, Image.Image],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        batch_size: int = 0,
        seed: Optional[int] = None,
        color_map: str = "Spectral",
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, Optional[Image.Image]]:
        """
        Perform depth estimation on an input image.

        Args:
            image_path: Path to input image or PIL Image object
            denoising_steps: Number of diffusion denoising steps
                           (None uses model default, typically 4 for v1-1)
            ensemble_size: Number of predictions to ensemble (default: 1)
                         Higher values improve quality but slow inference
            processing_res: Resolution for processing (None uses model default: 768)
                          Set to 0 to use native resolution
            match_input_res: Resize output to match input resolution
            batch_size: Inference batch size (0 for automatic)
            seed: Random seed for reproducibility (None for random)
            color_map: Matplotlib colormap for visualization ('Spectral', 'viridis', etc.)
                      Set to None to skip colored visualization
            show_progress: Show progress bar during inference

        Returns:
            Tuple containing:
                - depth_map (np.ndarray): Depth prediction as numpy array [H, W]
                  with values normalized to [0, 1] range
                - colored_depth (PIL.Image or None): Colored visualization of depth map

        Raises:
            FileNotFoundError: If image_path doesn't exist
            ValueError: If image cannot be loaded or is invalid
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            try:
                input_image = Image.open(image_path)
                logger.info(f"Loaded image: {image_path} (size: {input_image.size})")
            except Exception as e:
                raise ValueError(f"Failed to load image from {image_path}: {e}")
        elif isinstance(image_path, Image.Image):
            input_image = image_path
            logger.info(f"Using provided PIL Image (size: {input_image.size})")
        else:
            raise ValueError(
                "image_path must be a string path, Path object, or PIL Image"
            )

        # Set up random generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            logger.info(f"Using random seed: {seed}")

        # Log inference settings
        logger.info(
            f"Inference settings: "
            f"denoising_steps={denoising_steps or self.pipe.default_denoising_steps}, "
            f"ensemble_size={ensemble_size}, "
            f"processing_res={processing_res or self.pipe.default_processing_resolution}"
        )

        # Perform inference
        try:
            with torch.no_grad():
                pipe_out: MarigoldDepthOutput = self.pipe(
                    input_image,
                    denoising_steps=denoising_steps,
                    ensemble_size=ensemble_size,
                    processing_res=processing_res,
                    match_input_res=match_input_res,
                    batch_size=batch_size,
                    color_map=color_map,
                    show_progress_bar=show_progress,
                    generator=generator,
                )

            depth_map = pipe_out.depth_np
            colored_depth = pipe_out.depth_colored

            logger.info(
                f"Depth estimation complete. Output shape: {depth_map.shape}, "
                f"value range: [{depth_map.min():.3f}, {depth_map.max():.3f}]"
            )

            return depth_map, colored_depth

        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            raise

    def save_depth(
        self,
        depth_map: np.ndarray,
        output_path: Union[str, Path],
        colored_depth: Optional[Image.Image] = None,
        save_raw: bool = True,
        save_colored: bool = True,
    ) -> None:
        """
        Save depth map to disk in various formats.

        Args:
            depth_map: Depth map as numpy array
            output_path: Base path for output files (without extension)
            colored_depth: Colored visualization to save
            save_raw: Save raw depth as 16-bit PNG
            save_colored: Save colored visualization

        The function will create the following files:
            - {output_path}.npy: Raw numpy array
            - {output_path}.png: 16-bit grayscale PNG (if save_raw=True)
            - {output_path}_colored.png: Colored visualization (if save_colored=True)
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as numpy array
        npy_path = output_path.with_suffix('.npy')
        np.save(npy_path, depth_map)
        logger.info(f"Saved raw depth array to: {npy_path}")

        # Save as 16-bit PNG
        if save_raw:
            depth_uint16 = (depth_map * 65535.0).astype(np.uint16)
            png_path = output_path.with_suffix('.png')
            Image.fromarray(depth_uint16).save(png_path, mode="I;16")
            logger.info(f"Saved 16-bit depth PNG to: {png_path}")

        # Save colored visualization
        if save_colored and colored_depth is not None:
            colored_path = output_path.parent / f"{output_path.stem}_colored.png"
            colored_depth.save(colored_path)
            logger.info(f"Saved colored depth map to: {colored_path}")


def main():
    """
    Command-line interface for Marigold depth estimation.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Marigold Monocular Depth Estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Base path for output files (without extension)"
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-depth-v1-1",
        help="Model checkpoint (HuggingFace or local path)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (auto-detect if not specified)"
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Use FP16 for faster inference"
    )

    # Inference arguments
    parser.add_argument(
        "--denoising_steps",
        type=int,
        default=None,
        help="Number of denoising steps (uses model default if not set)"
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of predictions to ensemble"
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Processing resolution (0 for native, None for model default)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap for visualization (e.g., Spectral, viridis, plasma)"
    )
    parser.add_argument(
        "--no_colored",
        action="store_true",
        help="Skip colored depth visualization"
    )
    parser.add_argument(
        "--no_raw",
        action="store_true",
        help="Skip 16-bit PNG output"
    )

    args = parser.parse_args()

    try:
        # Initialize estimator
        logger.info("Initializing Marigold depth estimator...")
        estimator = MarigoldDepthEstimator(
            checkpoint=args.checkpoint,
            device=args.device,
            half_precision=args.half_precision,
        )

        # Perform depth estimation
        logger.info("Starting depth estimation...")
        depth_map, colored_depth = estimator.estimate_depth(
            image_path=args.image_path,
            denoising_steps=args.denoising_steps,
            ensemble_size=args.ensemble_size,
            processing_res=args.processing_res,
            seed=args.seed,
            color_map=None if args.no_colored else args.color_map,
            show_progress=True,
        )

        # Save results
        logger.info("Saving results...")
        estimator.save_depth(
            depth_map=depth_map,
            output_path=args.output_path,
            colored_depth=colored_depth,
            save_raw=not args.no_raw,
            save_colored=not args.no_colored and colored_depth is not None,
        )

        logger.info("Depth estimation completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
