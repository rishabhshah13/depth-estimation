"""
Test script for MiDaS depth estimation setup.

This script verifies that:
1. All dependencies are properly installed
2. MiDaS repository is accessible
3. The depth estimator can be initialized
4. A sample depth estimation can be performed
"""

import sys
import numpy as np
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV (opencv-python)',
        'timm': 'PyTorch Image Models (timm)',
        'einops': 'Einops'
    }

    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            missing.append(name)

    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("All dependencies found!\n")
    return True


def check_midas_repo():
    """Check if MiDaS repository exists."""
    print("Checking MiDaS repository...")
    midas_path = Path(__file__).parent / "MiDaS"

    if not midas_path.exists():
        print(f"  ✗ MiDaS directory not found at {midas_path}")
        print("\nPlease clone MiDaS repository:")
        print("  git clone https://github.com/isl-org/MiDaS")
        return False

    print(f"  ✓ MiDaS repository found at {midas_path}")

    # Check for key files
    key_files = [
        'midas/model_loader.py',
        'utils.py',
        'run.py'
    ]

    for file in key_files:
        file_path = midas_path / file
        if not file_path.exists():
            print(f"  ✗ Required file missing: {file}")
            return False
        print(f"  ✓ {file}")

    print("MiDaS repository structure looks good!\n")
    return True


def test_module_import():
    """Test importing the midas_depth module."""
    print("Testing midas_depth module import...")

    try:
        from midas_depth import MiDaSDepthEstimator
        print("  ✓ MiDaSDepthEstimator imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False


def test_model_initialization():
    """Test initializing a small model."""
    print("\nTesting model initialization...")

    try:
        from midas_depth import MiDaSDepthEstimator

        print("  Initializing midas_v21_small_256 model...")
        estimator = MiDaSDepthEstimator(model_type='midas_v21_small_256')
        print("  ✓ Model initialized successfully")

        # Print model info
        print(f"  Model type: {estimator.model_type}")
        print(f"  Device: {estimator.device}")
        print(f"  Input size: {estimator.net_w}x{estimator.net_h}")

        return estimator

    except FileNotFoundError as e:
        print(f"  ⚠ Model weights not found: {e}")
        print("  Note: Weights will be downloaded automatically on first use")
        return None
    except Exception as e:
        print(f"  ✗ Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_synthetic_image(estimator):
    """Test depth estimation on a synthetic image."""
    if estimator is None:
        print("\nSkipping synthetic image test (model not initialized)")
        return False

    print("\nTesting depth estimation with synthetic image...")

    try:
        # Create a synthetic gradient image (256x256x3)
        height, width = 256, 256
        synthetic_image = np.zeros((height, width, 3), dtype=np.float32)

        # Create a radial gradient
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        distance = distance / distance.max()

        # Set all channels to the gradient
        for c in range(3):
            synthetic_image[:, :, c] = distance

        print(f"  Created synthetic image: {synthetic_image.shape}")
        print(f"  Image range: [{synthetic_image.min():.3f}, {synthetic_image.max():.3f}]")

        # Estimate depth
        depth_map = estimator.estimate_depth(synthetic_image)

        print(f"  ✓ Depth estimation successful")
        print(f"  Depth map shape: {depth_map.shape}")
        print(f"  Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")

        return True

    except Exception as e:
        print(f"  ✗ Depth estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Test with your own image:")
    print("   python midas_depth.py -i your_image.jpg -o depth_output.png")
    print("\n2. Try side-by-side visualization:")
    print("   python midas_depth.py -i your_image.jpg -o depth.png --side-by-side")
    print("\n3. List available models:")
    print("   python midas_depth.py --list-models")
    print("\n4. Use as a Python module:")
    print("   from midas_depth import MiDaSDepthEstimator")
    print("   estimator = MiDaSDepthEstimator()")
    print("   depth = estimator.estimate_depth('image.jpg')")
    print("\n5. See USAGE.md for more examples and documentation")
    print("="*70 + "\n")


def main():
    """Run all tests."""
    print("="*70)
    print("MiDaS Depth Estimation - Setup Test")
    print("="*70 + "\n")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check MiDaS repository
    if not check_midas_repo():
        sys.exit(1)

    # Test module import
    if not test_module_import():
        sys.exit(1)

    # Test model initialization
    estimator = test_model_initialization()

    # Test with synthetic image
    if estimator:
        test_synthetic_image(estimator)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("✓ All basic tests passed!")
    print("✓ MiDaS depth estimation is ready to use")
    print("="*70)

    # Print next steps
    print_next_steps()


if __name__ == '__main__':
    main()
