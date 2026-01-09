#!/usr/bin/env python3
"""
Depth Estimation Comparison App
================================

This application compares three state-of-the-art depth estimation models:
1. MiDaS - Robust monocular depth estimation
2. Depth Anything V2 - Foundation model for depth estimation
3. Marigold - Diffusion-based depth estimation

Usage:
    python depth_comparison_app.py --input image.jpg --output comparison.png
    python depth_comparison_app.py --input image.jpg --output-dir results/
    python depth_comparison_app.py --gui  # Launch GUI mode

Author: Claude Code
Date: 2026-01-09
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import the three depth estimation modules
try:
    from src.midas_depth import MiDaSDepthEstimator
except ImportError as e:
    print(f"Error importing MiDaS: {e}")
    print("Make sure midas_depth.py is in the src/ directory")
    sys.exit(1)

try:
    from src.depth_anything import DepthEstimator as DepthAnythingEstimator
except ImportError as e:
    print(f"Error importing Depth Anything: {e}")
    print("Make sure depth_anything.py is in the src/ directory")
    sys.exit(1)

try:
    from src.marigold_depth import MarigoldDepthEstimator
except ImportError as e:
    print(f"Error importing Marigold: {e}")
    print("Make sure marigold_depth.py is in the src/ directory")
    sys.exit(1)


class DepthComparisonApp:
    """
    Main application class that runs all three depth estimation models
    and compares their outputs.
    """

    def __init__(
        self,
        midas_model: str = "midas_v21_small_256",
        depth_anything_model: str = "small",
        use_gpu: bool = True
    ):
        """
        Initialize the depth comparison app with all three models.

        Args:
            midas_model: MiDaS model type (default: midas_v21_small_256)
            depth_anything_model: Depth Anything model size (small/base/large)
            use_gpu: Whether to use GPU acceleration if available
        """
        print("=" * 60)
        print("Depth Estimation Comparison App")
        print("=" * 60)
        print("\nInitializing models...")

        self.models: Dict[str, object] = {}
        self.timing: Dict[str, float] = {}

        # Initialize MiDaS
        try:
            print("\n[1/3] Loading MiDaS...")
            start_time = time.time()
            self.models['midas'] = MiDaSDepthEstimator(
                model_type=midas_model,
                optimize=use_gpu,
                device=None  # Auto-detect
            )
            load_time = time.time() - start_time
            print(f"✓ MiDaS loaded successfully ({load_time:.2f}s)")
        except Exception as e:
            print(f"✗ Failed to load MiDaS: {e}")
            self.models['midas'] = None

        # Initialize Depth Anything
        try:
            print("\n[2/3] Loading Depth Anything V2...")
            start_time = time.time()
            self.models['depth_anything'] = DepthAnythingEstimator(
                model_size=depth_anything_model,
                device=None  # Auto-detect
            )
            load_time = time.time() - start_time
            print(f"✓ Depth Anything V2 loaded successfully ({load_time:.2f}s)")
        except Exception as e:
            print(f"✗ Failed to load Depth Anything V2: {e}")
            self.models['depth_anything'] = None

        # Initialize Marigold
        try:
            print("\n[3/3] Loading Marigold...")
            start_time = time.time()
            self.models['marigold'] = MarigoldDepthEstimator(
                checkpoint="prs-eth/marigold-depth-v1-1",
                device=None,  # Auto-detect
                half_precision=use_gpu  # Use half precision on GPU
            )
            load_time = time.time() - start_time
            print(f"✓ Marigold loaded successfully ({load_time:.2f}s)")
        except Exception as e:
            print(f"✗ Failed to load Marigold: {e}")
            self.models['marigold'] = None

        print("\n" + "=" * 60)
        active_models = sum(1 for m in self.models.values() if m is not None)
        print(f"Initialization complete! {active_models}/3 models loaded.")
        print("=" * 60 + "\n")

    def estimate_depth_all(
        self,
        image_path: str,
        save_individual: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Run all three depth estimation models on the input image.

        Args:
            image_path: Path to input image
            save_individual: Whether to save individual depth maps
            output_dir: Directory to save individual outputs (required if save_individual=True)

        Returns:
            Dictionary mapping model names to depth maps
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"Processing image: {image_path}\n")

        depth_maps = {}

        # Run MiDaS
        if self.models['midas'] is not None:
            try:
                print("[1/3] Running MiDaS...")
                start_time = time.time()
                depth_map = self.models['midas'].estimate_depth(image_path)
                elapsed = time.time() - start_time
                self.timing['midas'] = elapsed
                depth_maps['midas'] = depth_map
                print(f"✓ MiDaS completed in {elapsed:.2f}s")

                if save_individual and output_dir:
                    output_path = os.path.join(output_dir, "midas_depth.png")
                    self.models['midas'].save_depth_map(depth_map, output_path)
                    print(f"  Saved to: {output_path}")
            except Exception as e:
                print(f"✗ MiDaS failed: {e}")

        # Run Depth Anything
        if self.models['depth_anything'] is not None:
            try:
                print("\n[2/3] Running Depth Anything V2...")
                start_time = time.time()
                depth_map = self.models['depth_anything'].estimate_depth(image_path)
                elapsed = time.time() - start_time
                self.timing['depth_anything'] = elapsed
                depth_maps['depth_anything'] = depth_map
                print(f"✓ Depth Anything V2 completed in {elapsed:.2f}s")

                if save_individual and output_dir:
                    output_path = os.path.join(output_dir, "depth_anything_depth.png")
                    self.models['depth_anything'].save_depth_map(depth_map, output_path)
                    print(f"  Saved to: {output_path}")
            except Exception as e:
                print(f"✗ Depth Anything V2 failed: {e}")

        # Run Marigold
        if self.models['marigold'] is not None:
            try:
                print("\n[3/3] Running Marigold...")
                start_time = time.time()
                depth_map, colored_depth = self.models['marigold'].estimate_depth(
                    image_path,
                    ensemble_size=1,  # Fast inference
                    seed=42
                )
                elapsed = time.time() - start_time
                self.timing['marigold'] = elapsed
                depth_maps['marigold'] = depth_map
                print(f"✓ Marigold completed in {elapsed:.2f}s")

                if save_individual and output_dir:
                    output_path = os.path.join(output_dir, "marigold_depth")
                    self.models['marigold'].save_depth(depth_map, output_path, colored_depth)
                    print(f"  Saved to: {output_path}.png")
            except Exception as e:
                print(f"✗ Marigold failed: {e}")

        print()
        return depth_maps

    def create_comparison_visualization(
        self,
        image_path: str,
        depth_maps: Dict[str, np.ndarray],
        output_path: str,
        dpi: int = 150
    ):
        """
        Create a side-by-side comparison visualization of all depth maps.

        Args:
            image_path: Path to original image
            depth_maps: Dictionary of depth maps from different models
            output_path: Path to save comparison image
            dpi: DPI for output image (higher = better quality)
        """
        # Load original image
        original = Image.open(image_path).convert('RGB')

        # Calculate grid layout
        n_models = len(depth_maps)
        n_cols = min(4, n_models + 1)  # Original + up to 3 models
        n_rows = 1 if n_models <= 3 else 2

        # Create figure with custom layout
        fig = plt.figure(figsize=(n_cols * 5, n_rows * 5))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

        # Plot original image
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(original)
        ax_orig.set_title("Original Image", fontsize=14, fontweight='bold')
        ax_orig.axis('off')

        # Plot depth maps
        model_names = {
            'midas': 'MiDaS',
            'depth_anything': 'Depth Anything V2',
            'marigold': 'Marigold'
        }

        colormaps = {
            'midas': 'viridis',
            'depth_anything': 'Spectral_r',
            'marigold': 'inferno'
        }

        for idx, (model_key, depth_map) in enumerate(depth_maps.items()):
            row = (idx + 1) // n_cols
            col = (idx + 1) % n_cols

            ax = fig.add_subplot(gs[row, col])

            # Normalize depth map for visualization
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

            # Display depth map
            im = ax.imshow(depth_normalized, cmap=colormaps.get(model_key, 'viridis'))

            # Title with timing info
            model_name = model_names.get(model_key, model_key)
            timing = self.timing.get(model_key, 0)
            ax.set_title(f"{model_name}\n({timing:.2f}s)", fontsize=12, fontweight='bold')
            ax.axis('off')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Depth', rotation=270, labelpad=15)

        # Add overall title
        fig.suptitle(
            f"Depth Estimation Comparison\n{Path(image_path).name}",
            fontsize=16,
            fontweight='bold',
            y=0.98
        )

        # Save figure
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"\n{'=' * 60}")
        print(f"Comparison saved to: {output_path}")
        print(f"{'=' * 60}\n")

    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        save_individual: bool = False,
        dpi: int = 150
    ):
        """
        Complete pipeline: run all models and create comparison.

        Args:
            image_path: Path to input image
            output_path: Path to save comparison image
            output_dir: Directory to save individual outputs
            save_individual: Whether to save individual depth maps
            dpi: DPI for output image
        """
        # Set default output path
        if output_path is None:
            input_name = Path(image_path).stem
            output_path = f"{input_name}_comparison.png"

        # Create output directory if needed
        if save_individual and output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Run all models
        depth_maps = self.estimate_depth_all(
            image_path,
            save_individual=save_individual,
            output_dir=output_dir
        )

        # Create comparison visualization
        if depth_maps:
            self.create_comparison_visualization(
                image_path,
                depth_maps,
                output_path,
                dpi=dpi
            )

            # Print summary
            print("\nProcessing Summary:")
            print("-" * 60)
            print(f"Input image: {image_path}")
            print(f"Output comparison: {output_path}")
            if save_individual and output_dir:
                print(f"Individual outputs: {output_dir}/")
            print(f"\nModels processed: {len(depth_maps)}/3")
            print("\nTiming:")
            for model, timing in self.timing.items():
                print(f"  {model}: {timing:.2f}s")
            total_time = sum(self.timing.values())
            print(f"  Total: {total_time:.2f}s")
            print("-" * 60)
        else:
            print("\n⚠ No depth maps were generated. Check model initialization.")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Compare MiDaS, Depth Anything V2, and Marigold depth estimation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python depth_comparison_app.py --input image.jpg

  # Save comparison with custom name
  python depth_comparison_app.py --input image.jpg --output comparison.png

  # Save individual depth maps
  python depth_comparison_app.py --input image.jpg --output-dir results/ --save-individual

  # Use larger models for better quality
  python depth_comparison_app.py --input image.jpg --midas-model dpt_large_384 --da-model large

  # High-quality output
  python depth_comparison_app.py --input image.jpg --dpi 300
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input image'
    )

    parser.add_argument(
        '--output', '-o',
        help='Path to save comparison image (default: <input>_comparison.png)'
    )

    parser.add_argument(
        '--output-dir',
        help='Directory to save individual depth maps (if --save-individual is set)'
    )

    parser.add_argument(
        '--save-individual',
        action='store_true',
        help='Save individual depth maps in addition to comparison'
    )

    parser.add_argument(
        '--midas-model',
        default='midas_v21_small_256',
        choices=[
            'midas_v21_small_256', 'midas_v21_384', 'dpt_hybrid_384',
            'dpt_large_384', 'dpt_swin2_large_384', 'dpt_beit_large_512'
        ],
        help='MiDaS model type (default: midas_v21_small_256)'
    )

    parser.add_argument(
        '--da-model',
        default='small',
        choices=['small', 'base', 'large'],
        help='Depth Anything model size (default: small)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for output image (default: 150, higher = better quality)'
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input image not found: {args.input}")
        sys.exit(1)

    if args.save_individual and not args.output_dir:
        print("Error: --output-dir is required when --save-individual is set")
        sys.exit(1)

    # Initialize app
    try:
        app = DepthComparisonApp(
            midas_model=args.midas_model,
            depth_anything_model=args.da_model,
            use_gpu=not args.no_gpu
        )
    except Exception as e:
        print(f"\nError initializing application: {e}")
        sys.exit(1)

    # Process image
    try:
        app.process_image(
            image_path=args.input,
            output_path=args.output,
            output_dir=args.output_dir,
            save_individual=args.save_individual,
            dpi=args.dpi
        )
    except Exception as e:
        print(f"\nError processing image: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
