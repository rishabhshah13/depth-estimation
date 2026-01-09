"""
Example usage of the MiDaS depth estimation module.

This script demonstrates various ways to use midas_depth.py
for monocular depth estimation.
"""

import numpy as np
from pathlib import Path
from midas_depth import MiDaSDepthEstimator


def example_1_basic_usage():
    """Example 1: Basic depth estimation with default model."""
    print("="*70)
    print("Example 1: Basic Depth Estimation")
    print("="*70)

    # Initialize with default fast model
    estimator = MiDaSDepthEstimator(model_type='midas_v21_small_256')

    # Create a simple test image
    test_image = create_test_image()

    # Estimate depth
    depth_map = estimator.estimate_depth(test_image)

    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
    print(f"Mean depth: {depth_map.mean():.3f}")

    # Save depth map
    estimator.save_depth_map(depth_map, 'example1_depth.png')
    print("\nSaved: example1_depth.png\n")


def example_2_high_quality():
    """Example 2: High-quality depth estimation."""
    print("="*70)
    print("Example 2: High-Quality Depth Estimation")
    print("="*70)

    # Initialize with high-quality model
    estimator = MiDaSDepthEstimator(
        model_type='dpt_large_384',
        optimize=False,  # Use full precision for best quality
        device='cuda'    # Use GPU if available
    )

    test_image = create_test_image()

    # Get depth map and original image
    depth_map, original = estimator.estimate_depth(
        test_image,
        return_original=True
    )

    # Save with different colormap
    estimator.save_depth_map(depth_map, 'example2_depth_viridis.png', colormap='viridis')

    # Create side-by-side visualization
    estimator.visualize_side_by_side(original, depth_map, 'example2_side_by_side.png')

    print("Saved: example2_depth_viridis.png")
    print("Saved: example2_side_by_side.png\n")


def example_3_batch_processing():
    """Example 3: Process multiple images."""
    print("="*70)
    print("Example 3: Batch Processing")
    print("="*70)

    # Initialize model once
    estimator = MiDaSDepthEstimator(model_type='midas_v21_small_256')

    # Create test images
    test_images = [
        ('gradient', create_test_image('gradient')),
        ('checkerboard', create_test_image('checkerboard')),
        ('circle', create_test_image('circle'))
    ]

    output_dir = Path('example3_batch')
    output_dir.mkdir(exist_ok=True)

    for name, image in test_images:
        print(f"Processing: {name}...")

        depth_map = estimator.estimate_depth(image)

        output_path = output_dir / f"{name}_depth.png"
        estimator.save_depth_map(depth_map, str(output_path))

        print(f"  Saved: {output_path}")

    print(f"\nAll outputs in: {output_dir}/\n")


def example_4_custom_visualization():
    """Example 4: Custom depth map processing."""
    print("="*70)
    print("Example 4: Custom Depth Map Processing")
    print("="*70)

    estimator = MiDaSDepthEstimator()
    test_image = create_test_image()

    depth_map = estimator.estimate_depth(test_image)

    # Custom normalization
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Compute statistics
    print(f"Depth Statistics:")
    print(f"  Min: {depth_map.min():.4f}")
    print(f"  Max: {depth_map.max():.4f}")
    print(f"  Mean: {depth_map.mean():.4f}")
    print(f"  Std: {depth_map.std():.4f}")
    print(f"  Median: {np.median(depth_map):.4f}")

    # Find nearest and farthest regions
    near_threshold = np.percentile(depth_map, 10)
    far_threshold = np.percentile(depth_map, 90)

    print(f"\nRegion Analysis:")
    print(f"  Nearest 10% threshold: {near_threshold:.4f}")
    print(f"  Farthest 10% threshold: {far_threshold:.4f}")

    # Save with different colormaps
    colormaps = ['inferno', 'viridis', 'magma', 'gray']
    for cmap in colormaps:
        output_path = f'example4_depth_{cmap}.png'
        estimator.save_depth_map(depth_map, output_path, colormap=cmap)
        print(f"  Saved: {output_path}")

    print()


def example_5_model_comparison():
    """Example 5: Compare different models."""
    print("="*70)
    print("Example 5: Model Comparison")
    print("="*70)

    test_image = create_test_image()

    # Models to compare (from fast to high quality)
    models = [
        'midas_v21_small_256',
        'dpt_hybrid_384',
        'dpt_large_384'
    ]

    output_dir = Path('example5_comparison')
    output_dir.mkdir(exist_ok=True)

    for model_type in models:
        print(f"\nProcessing with {model_type}...")

        # Initialize model
        estimator = MiDaSDepthEstimator(model_type=model_type)

        # Estimate depth
        depth_map = estimator.estimate_depth(test_image)

        # Save result
        output_path = output_dir / f"{model_type}_depth.png"
        estimator.save_depth_map(depth_map, str(output_path))

        print(f"  Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        print(f"  Saved: {output_path}")

    print(f"\nAll comparisons in: {output_dir}/\n")


def create_test_image(pattern='gradient', size=(512, 512)):
    """Create a synthetic test image."""
    height, width = size
    image = np.zeros((height, width, 3), dtype=np.float32)

    if pattern == 'gradient':
        # Horizontal gradient
        gradient = np.linspace(0, 1, width)
        for c in range(3):
            image[:, :, c] = gradient

    elif pattern == 'checkerboard':
        # Checkerboard pattern
        square_size = 64
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    image[i:i+square_size, j:j+square_size] = 1.0

    elif pattern == 'circle':
        # Circular gradient
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        distance = distance / distance.max()
        for c in range(3):
            image[:, :, c] = 1 - distance

    return image


def list_all_models():
    """List all available models."""
    print("="*70)
    print("Available MiDaS Models")
    print("="*70)
    MiDaSDepthEstimator.list_available_models()


def main():
    """Run all examples."""
    import sys

    print("\n")
    print("="*70)
    print("MiDaS Depth Estimation - Examples")
    print("="*70)
    print("\nThese examples demonstrate various uses of midas_depth.py")
    print("Note: Model weights will be downloaded automatically if needed\n")

    # Check if specific example requested
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        examples = {
            '1': example_1_basic_usage,
            '2': example_2_high_quality,
            '3': example_3_batch_processing,
            '4': example_4_custom_visualization,
            '5': example_5_model_comparison,
            'list': list_all_models
        }

        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Unknown example: {example_num}")
            print("Available examples: 1, 2, 3, 4, 5, list")
    else:
        # Run all examples
        try:
            list_all_models()
            example_1_basic_usage()
            example_3_batch_processing()
            example_4_custom_visualization()

            print("="*70)
            print("Examples Complete!")
            print("="*70)
            print("\nNote: Examples 2 and 5 skipped (require model downloads)")
            print("Run individually with: python example_midas.py 2")
            print("                   or: python example_midas.py 5\n")

        except Exception as e:
            print(f"\nError running examples: {e}")
            print("\nMake sure dependencies are installed:")
            print("  pip install -r requirements.txt\n")
            sys.exit(1)


if __name__ == '__main__':
    main()
