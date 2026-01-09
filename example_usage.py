"""
Simple example demonstrating marigold_depth.py usage

This script shows the most common use cases for the Marigold depth estimation wrapper.
"""

from marigold_depth import MarigoldDepthEstimator
from PIL import Image
import numpy as np


def example_1_basic_usage():
    """Example 1: Basic depth estimation"""
    print("\n" + "="*60)
    print("Example 1: Basic Depth Estimation")
    print("="*60)

    # Initialize estimator (model will be downloaded on first run)
    print("\nInitializing estimator...")
    estimator = MarigoldDepthEstimator()

    # Estimate depth from an image file
    print("Estimating depth...")
    depth_map, colored_depth = estimator.estimate_depth("input.jpg")

    # Save results
    print("Saving results...")
    estimator.save_depth(
        depth_map=depth_map,
        output_path="output",
        colored_depth=colored_depth
    )

    print("✓ Done! Check output.npy, output.png, and output_colored.png")


def example_2_custom_settings():
    """Example 2: Custom inference settings"""
    print("\n" + "="*60)
    print("Example 2: Custom Inference Settings")
    print("="*60)

    # Initialize with half precision for faster inference
    estimator = MarigoldDepthEstimator(half_precision=True)

    # Custom inference settings
    depth_map, colored_depth = estimator.estimate_depth(
        image_path="input.jpg",
        denoising_steps=4,       # Number of diffusion steps
        ensemble_size=5,         # Average 5 predictions for better quality
        processing_res=768,      # Processing resolution
        seed=42,                 # For reproducibility
        color_map="viridis",     # Different colormap
    )

    estimator.save_depth(depth_map, "output_custom", colored_depth)
    print("✓ Done with custom settings!")


def example_3_pil_image():
    """Example 3: Work with PIL Image directly"""
    print("\n" + "="*60)
    print("Example 3: Using PIL Image Object")
    print("="*60)

    estimator = MarigoldDepthEstimator()

    # Load image with PIL
    img = Image.open("input.jpg")
    print(f"Loaded image: {img.size}")

    # You can process the image before depth estimation
    # For example, resize or crop
    # img = img.resize((512, 384))

    # Estimate depth directly from PIL Image
    depth_map, colored_depth = estimator.estimate_depth(img)

    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")

    estimator.save_depth(depth_map, "output_pil", colored_depth)
    print("✓ Done with PIL Image!")


def example_4_depth_analysis():
    """Example 4: Analyze depth map"""
    print("\n" + "="*60)
    print("Example 4: Depth Map Analysis")
    print("="*60)

    estimator = MarigoldDepthEstimator()
    depth_map, _ = estimator.estimate_depth("input.jpg")

    # Depth map statistics
    print("\nDepth Statistics:")
    print(f"  Mean depth: {np.mean(depth_map):.3f}")
    print(f"  Std deviation: {np.std(depth_map):.3f}")
    print(f"  Min depth: {np.min(depth_map):.3f}")
    print(f"  Max depth: {np.max(depth_map):.3f}")

    # Segment by depth
    near_objects = np.sum(depth_map < 0.3)
    mid_objects = np.sum((depth_map >= 0.3) & (depth_map < 0.7))
    far_objects = np.sum(depth_map >= 0.7)
    total_pixels = depth_map.size

    print("\nDepth Distribution:")
    print(f"  Near (<0.3): {near_objects/total_pixels*100:.1f}%")
    print(f"  Mid (0.3-0.7): {mid_objects/total_pixels*100:.1f}%")
    print(f"  Far (>0.7): {far_objects/total_pixels*100:.1f}%")

    print("✓ Analysis complete!")


def example_5_batch_processing():
    """Example 5: Process multiple images"""
    print("\n" + "="*60)
    print("Example 5: Batch Processing")
    print("="*60)

    from pathlib import Path

    # Initialize once for all images
    estimator = MarigoldDepthEstimator()

    input_dir = Path("input_images")
    output_dir = Path("output_depths")
    output_dir.mkdir(exist_ok=True)

    # Process all JPG images
    image_files = list(input_dir.glob("*.jpg"))
    print(f"\nFound {len(image_files)} images to process")

    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing {img_path.name}...")

        depth_map, colored = estimator.estimate_depth(
            img_path,
            show_progress=False  # Disable progress bar for batch processing
        )

        output_path = output_dir / img_path.stem
        estimator.save_depth(depth_map, output_path, colored)

    print(f"\n✓ Processed {len(image_files)} images!")


def example_6_high_quality():
    """Example 6: High-quality depth estimation"""
    print("\n" + "="*60)
    print("Example 6: High-Quality Depth Estimation")
    print("="*60)

    # Use full precision for best quality
    estimator = MarigoldDepthEstimator(half_precision=False)

    # High-quality settings (slower but better results)
    depth_map, colored = estimator.estimate_depth(
        image_path="input.jpg",
        denoising_steps=4,       # More steps (diminishing returns after ~10)
        ensemble_size=10,        # Ensemble 10 predictions
        processing_res=768,      # Higher resolution
        seed=42,                 # For reproducibility
    )

    estimator.save_depth(depth_map, "output_hq", colored)
    print("✓ High-quality depth estimation complete!")


def example_7_fast_inference():
    """Example 7: Fast inference for real-time applications"""
    print("\n" + "="*60)
    print("Example 7: Fast Inference")
    print("="*60)

    import time

    # Use half precision and optimized settings
    estimator = MarigoldDepthEstimator(half_precision=True)

    # Fast inference settings
    start_time = time.time()

    depth_map, colored = estimator.estimate_depth(
        image_path="input.jpg",
        ensemble_size=1,         # Single prediction
        processing_res=512,      # Lower resolution
        show_progress=False,     # No progress bar
    )

    elapsed = time.time() - start_time

    estimator.save_depth(depth_map, "output_fast", colored)
    print(f"✓ Fast inference completed in {elapsed:.2f} seconds!")


def main():
    """Run all examples"""
    print("="*60)
    print("Marigold Depth Estimation - Usage Examples")
    print("="*60)

    print("\nNote: Examples assume you have an 'input.jpg' file.")
    print("You can create a test image or use your own.")

    # Choose which examples to run
    # Uncomment the examples you want to try

    # example_1_basic_usage()
    # example_2_custom_settings()
    # example_3_pil_image()
    # example_4_depth_analysis()
    # example_5_batch_processing()
    # example_6_high_quality()
    # example_7_fast_inference()

    print("\n" + "="*60)
    print("To run an example, uncomment it in the main() function")
    print("="*60)


if __name__ == "__main__":
    main()
