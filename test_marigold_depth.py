"""
Test script for marigold_depth.py module

This script demonstrates both the module interface and command-line usage
of the Marigold depth estimation wrapper.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image


def create_test_image(output_path: str = "test_input.jpg", size=(512, 384)):
    """
    Create a simple test image for depth estimation testing.

    Args:
        output_path: Path to save the test image
        size: Image size (width, height)
    """
    print(f"Creating test image: {output_path}")

    # Create a gradient image that could simulate depth
    width, height = size
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Create a simple scene with gradients
    for y in range(height):
        for x in range(width):
            # Gradient from top to bottom (sky to ground)
            vertical_gradient = int((y / height) * 255)

            # Radial gradient from center (simulating distance)
            center_x, center_y = width // 2, height // 2
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            radial_gradient = int((dist / max_dist) * 255)

            # Combine gradients
            img_array[y, x] = [
                (vertical_gradient + radial_gradient) // 2,  # Red
                vertical_gradient,  # Green
                radial_gradient,  # Blue
            ]

    img = Image.fromarray(img_array)
    img.save(output_path)
    print(f"Test image saved: {output_path} (size: {img.size})")
    return output_path


def test_module_interface():
    """Test using the module as an import."""
    print("\n" + "="*60)
    print("TEST 1: Module Interface")
    print("="*60)

    try:
        from marigold_depth import MarigoldDepthEstimator

        # Create test image
        test_image_path = create_test_image("test_input.jpg")

        print("\nInitializing MarigoldDepthEstimator...")
        estimator = MarigoldDepthEstimator(
            checkpoint="prs-eth/marigold-depth-v1-1",
            device=None,  # Auto-detect
            half_precision=False,  # Use full precision for better quality
        )

        print("\nPerforming depth estimation...")
        depth_map, colored_depth = estimator.estimate_depth(
            image_path=test_image_path,
            ensemble_size=1,  # Quick test
            seed=42,  # For reproducibility
            show_progress=True,
        )

        print(f"\nResults:")
        print(f"  Depth map shape: {depth_map.shape}")
        print(f"  Depth value range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        print(f"  Colored depth: {colored_depth.size if colored_depth else None}")

        print("\nSaving results...")
        estimator.save_depth(
            depth_map=depth_map,
            output_path="test_output",
            colored_depth=colored_depth,
            save_raw=True,
            save_colored=True,
        )

        print("\n✓ Module interface test completed successfully!")
        print("  Output files:")
        print("    - test_output.npy (raw numpy array)")
        print("    - test_output.png (16-bit grayscale)")
        print("    - test_output_colored.png (colored visualization)")

        return True

    except Exception as e:
        print(f"\n✗ Module interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_interface():
    """Test using the command-line interface."""
    print("\n" + "="*60)
    print("TEST 2: Command-Line Interface")
    print("="*60)

    try:
        import subprocess

        # Create test image
        test_image_path = create_test_image("test_input_cli.jpg")

        print("\nRunning command-line interface...")
        cmd = [
            sys.executable,  # Python interpreter
            "marigold_depth.py",
            "--image_path", test_image_path,
            "--output_path", "test_output_cli",
            "--ensemble_size", "1",
            "--seed", "42",
        ]

        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("\n✓ CLI test completed successfully!")
            print("  Output files:")
            print("    - test_output_cli.npy")
            print("    - test_output_cli.png")
            print("    - test_output_cli_colored.png")
            return True
        else:
            print(f"\n✗ CLI test failed with return code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False

    except Exception as e:
        print(f"\n✗ CLI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_usage_examples():
    """Print usage examples."""
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)

    print("\n1. As a Python module:")
    print("-" * 60)
    print("""
from marigold_depth import MarigoldDepthEstimator

# Initialize estimator
estimator = MarigoldDepthEstimator()

# Estimate depth
depth_map, colored_depth = estimator.estimate_depth("input.jpg")

# Save results
estimator.save_depth(depth_map, "output", colored_depth)
""")

    print("\n2. As a command-line tool:")
    print("-" * 60)
    print("""
# Basic usage
python marigold_depth.py --image_path input.jpg --output_path output

# With custom settings
python marigold_depth.py \\
    --image_path input.jpg \\
    --output_path output \\
    --half_precision \\
    --ensemble_size 5 \\
    --seed 42

# Use custom checkpoint
python marigold_depth.py \\
    --image_path input.jpg \\
    --output_path output \\
    --checkpoint prs-eth/marigold-depth-lcm-v1-0
""")

    print("\n3. Advanced usage with PIL Image:")
    print("-" * 60)
    print("""
from PIL import Image
from marigold_depth import MarigoldDepthEstimator

estimator = MarigoldDepthEstimator()

# Load image
img = Image.open("input.jpg")

# Process directly
depth_map, colored_depth = estimator.estimate_depth(img)

# Work with numpy array
import numpy as np
print(f"Depth statistics:")
print(f"  Mean: {np.mean(depth_map):.3f}")
print(f"  Std: {np.std(depth_map):.3f}")
print(f"  Min: {np.min(depth_map):.3f}")
print(f"  Max: {np.max(depth_map):.3f}")
""")


def main():
    """Run all tests."""
    print("="*60)
    print("Marigold Depth Estimation - Test Suite")
    print("="*60)

    # Print usage examples first
    print_usage_examples()

    # Run tests
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60)

    test1_passed = test_module_interface()

    # Uncomment to test CLI (requires successful module test first)
    # test2_passed = test_cli_interface()

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Module Interface: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    # print(f"CLI Interface: {'✓ PASSED' if test2_passed else '✗ FAILED'}")

    if test1_passed:
        print("\nAll tests passed! The module is ready to use.")
    else:
        print("\nSome tests failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
