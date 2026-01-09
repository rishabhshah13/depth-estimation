#!/usr/bin/env python3
"""
Test script for the Depth Estimation Comparison App
====================================================

This script verifies that all components are properly installed and working.

Usage:
    python test_app.py
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("\n" + "=" * 60)
    print("Checking Python Version")
    print("=" * 60)

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False

    print("✓ Python version is compatible")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    print("\n" + "=" * 60)
    print("Checking Dependencies")
    print("=" * 60)

    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'torchvision',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib',
        'timm': 'timm',
        'einops': 'einops',
        'diffusers': 'diffusers',
        'transformers': 'transformers',
        'accelerate': 'accelerate',
        'gradio': 'Gradio'
    }

    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} - Not installed")
            missing.append(name)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False

    print("\n✓ All dependencies are installed")
    return True


def check_repositories():
    """Check if repositories are cloned."""
    print("\n" + "=" * 60)
    print("Checking Repositories")
    print("=" * 60)

    repos = {
        'MiDaS': 'MiDaS',
        'Depth-Anything-V2': 'Depth Anything V2',
        'Marigold': 'Marigold'
    }

    missing = []
    for repo_dir, name in repos.items():
        if os.path.isdir(repo_dir):
            print(f"✓ {name} repository found at {repo_dir}/")
        else:
            print(f"❌ {name} repository not found")
            missing.append(name)

    if missing:
        print(f"\n❌ Missing repositories: {', '.join(missing)}")
        return False

    print("\n✓ All repositories are present")
    return True


def check_modules():
    """Check if depth estimation modules can be imported."""
    print("\n" + "=" * 60)
    print("Checking Depth Estimation Modules")
    print("=" * 60)

    modules = {
        'midas_depth': 'MiDaS Module',
        'depth_anything': 'Depth Anything Module',
        'marigold_depth': 'Marigold Module'
    }

    missing = []
    for module_name, name in modules.items():
        try:
            __import__(module_name)
            print(f"✓ {name} ({module_name}.py)")
        except ImportError as e:
            print(f"❌ {name} - Import failed: {e}")
            missing.append(name)

    if missing:
        print(f"\n❌ Failed to import: {', '.join(missing)}")
        return False

    print("\n✓ All depth estimation modules can be imported")
    return True


def check_model_weights():
    """Check if model weights are available."""
    print("\n" + "=" * 60)
    print("Checking Model Weights")
    print("=" * 60)

    # Check Depth Anything V2 checkpoints
    checkpoint_dir = Path("Depth-Anything-V2/checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if checkpoints:
            print(f"✓ Depth Anything V2 checkpoints found ({len(checkpoints)} file(s)):")
            for cp in checkpoints:
                print(f"  - {cp.name}")
        else:
            print("⚠ Depth Anything V2 checkpoints not found")
            print("  Download with:")
            print("  cd Depth-Anything-V2/checkpoints")
            print("  wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth")
    else:
        print("⚠ Depth Anything V2 checkpoint directory not found")

    # MiDaS and Marigold download automatically
    print("✓ MiDaS weights will download automatically on first use")
    print("✓ Marigold weights will download automatically on first use")

    return True


def check_gpu():
    """Check GPU availability."""
    print("\n" + "=" * 60)
    print("Checking GPU Availability")
    print("=" * 60)

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"✓ MPS (Apple Silicon) is available")
        else:
            print("⚠ No GPU detected - will use CPU")
            print("  Models will run slower on CPU")

    except Exception as e:
        print(f"⚠ Could not check GPU: {e}")

    return True


def check_test_images():
    """Check if test images are available."""
    print("\n" + "=" * 60)
    print("Checking Test Images")
    print("=" * 60)

    test_images = [
        "Depth-Anything-V2/assets/examples/demo01.jpg",
        "Depth-Anything-V2/assets/examples/demo02.jpg"
    ]

    found = []
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"✓ {img_path}")
            found.append(img_path)
        else:
            print(f"⚠ {img_path} not found")

    if found:
        print(f"\n✓ Found {len(found)} test image(s)")
        return found
    else:
        print("\n⚠ No test images found")
        return []


def test_quick_inference():
    """Test quick inference with a synthetic image."""
    print("\n" + "=" * 60)
    print("Testing Quick Inference (Synthetic Image)")
    print("=" * 60)

    try:
        import numpy as np
        from PIL import Image
        from midas_depth import MiDaSDepthEstimator

        # Create a simple test image
        print("Creating synthetic test image...")
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)

        # Test MiDaS (fastest model)
        print("Loading MiDaS (fastest model for testing)...")
        estimator = MiDaSDepthEstimator(
            model_type='midas_v21_small_256',
            optimize=False,
            device='cpu'  # Use CPU for testing
        )

        print("Running depth estimation...")
        depth_map = estimator.estimate_depth(pil_image)

        if depth_map is not None and depth_map.shape == (256, 256):
            print("✓ Quick inference test passed!")
            print(f"  Output shape: {depth_map.shape}")
            print(f"  Depth range: [{depth_map.min():.2f}, {depth_map.max():.2f}]")
            return True
        else:
            print("❌ Quick inference test failed: unexpected output")
            return False

    except Exception as e:
        print(f"❌ Quick inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#" + "  Depth Estimation Comparison App - System Check".center(58) + "#")
    print("#" + " " * 58 + "#")
    print("#" * 60)

    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Repositories': check_repositories(),
        'Modules': check_modules(),
        'Model Weights': check_model_weights(),
        'GPU': check_gpu(),
        'Test Images': bool(check_test_images()),
        'Quick Inference': test_quick_inference()
    }

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for check, passed in results.items():
        status = "✓" if passed else "❌"
        print(f"{status} {check}")

    all_critical_passed = all([
        results['Python Version'],
        results['Dependencies'],
        results['Repositories'],
        results['Modules']
    ])

    print("\n" + "=" * 60)
    if all_critical_passed:
        print("✓ All critical checks passed!")
        print("\nYou can now run the application:")
        print("  python app_gui.py              (Web GUI)")
        print("  python depth_comparison_app.py --input image.jpg  (CLI)")
    else:
        print("❌ Some critical checks failed")
        print("Please fix the issues above before running the application")

    if not results['Model Weights']:
        print("\n⚠ Note: Download Depth Anything V2 checkpoint for best results:")
        print("  cd Depth-Anything-V2/checkpoints")
        print("  wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth")

    print("=" * 60 + "\n")

    return 0 if all_critical_passed else 1


if __name__ == "__main__":
    sys.exit(main())
