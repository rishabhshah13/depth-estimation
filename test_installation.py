"""
Test script to verify the installation and setup of Depth Anything V2.

This script checks:
1. Python dependencies are installed
2. Repository is properly cloned
3. Model checkpoints are available
4. The depth_anything module can be imported
5. Basic functionality works
"""

import sys
from pathlib import Path
import importlib.util

def check_dependency(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"  [OK] {package_name}")
        return True
    except ImportError:
        print(f"  [MISSING] {package_name}")
        return False

def check_file_exists(file_path, description):
    """Check if a file exists."""
    if Path(file_path).exists():
        print(f"  [OK] {description}")
        return True
    else:
        print(f"  [MISSING] {description}")
        print(f"           Expected at: {file_path}")
        return False

def main():
    """Run all installation checks."""
    print("=" * 70)
    print("Depth Anything V2 - Installation Test")
    print("=" * 70)
    print()

    all_ok = True

    # Check Python version
    print("1. Python Version:")
    py_version = sys.version_info
    print(f"  Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 7):
        print("  [WARNING] Python 3.7+ recommended")
        all_ok = False
    else:
        print("  [OK] Version is compatible")
    print()

    # Check dependencies
    print("2. Required Dependencies:")
    deps = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
    ]

    for package, import_name in deps:
        if not check_dependency(package, import_name):
            all_ok = False

    print()

    # Check repository structure
    print("3. Repository Structure:")
    base_dir = Path(__file__).parent

    files_to_check = [
        (base_dir / "Depth-Anything-V2", "Depth-Anything-V2 repository"),
        (base_dir / "Depth-Anything-V2" / "depth_anything_v2", "depth_anything_v2 module"),
        (base_dir / "Depth-Anything-V2" / "depth_anything_v2" / "dpt.py", "DPT model file"),
        (base_dir / "depth_anything.py", "depth_anything.py wrapper"),
        (base_dir / "example_usage.py", "example_usage.py"),
    ]

    for file_path, description in files_to_check:
        if not check_file_exists(file_path, description):
            all_ok = False

    print()

    # Check model checkpoints
    print("4. Model Checkpoints:")
    checkpoint_dir = base_dir / "Depth-Anything-V2" / "checkpoints"

    if not checkpoint_dir.exists():
        print(f"  [INFO] Checkpoint directory doesn't exist: {checkpoint_dir}")
        print(f"         Creating directory...")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        'small': 'depth_anything_v2_vits.pth',
        'base': 'depth_anything_v2_vitb.pth',
        'large': 'depth_anything_v2_vitl.pth',
    }

    found_checkpoints = []
    for model_name, filename in checkpoints.items():
        checkpoint_path = checkpoint_dir / filename
        if checkpoint_path.exists():
            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {model_name.capitalize()} model ({size_mb:.1f} MB)")
            found_checkpoints.append(model_name)
        else:
            print(f"  [MISSING] {model_name.capitalize()} model")

    if not found_checkpoints:
        print()
        print("  [WARNING] No model checkpoints found!")
        print("            Run './setup_models.sh' to download models")
        print("            Or download manually from:")
        print("            https://huggingface.co/depth-anything")
        all_ok = False
    else:
        print(f"\n  Found {len(found_checkpoints)} model(s): {', '.join(found_checkpoints)}")

    print()

    # Test import
    print("5. Module Import Test:")
    try:
        # Add current directory to path
        sys.path.insert(0, str(base_dir))

        from depth_anything import DepthEstimator
        print("  [OK] Successfully imported DepthEstimator")

        # Check available models
        print(f"  [OK] Available model sizes: {list(DepthEstimator.MODEL_CONFIGS.keys())}")

    except ImportError as e:
        print(f"  [ERROR] Failed to import: {e}")
        all_ok = False
    except Exception as e:
        print(f"  [ERROR] Unexpected error: {e}")
        all_ok = False

    print()

    # Test example images
    print("6. Example Images:")
    examples_dir = base_dir / "Depth-Anything-V2" / "assets" / "examples"

    if examples_dir.exists():
        example_images = list(examples_dir.glob("*.jpg"))
        if example_images:
            print(f"  [OK] Found {len(example_images)} example images")
            print(f"       Location: {examples_dir}")
        else:
            print(f"  [WARNING] No example images found in {examples_dir}")
    else:
        print(f"  [INFO] Example images directory not found")
        print(f"         Expected at: {examples_dir}")

    print()

    # Summary
    print("=" * 70)
    if all_ok and found_checkpoints:
        print("SUCCESS! Installation is complete and ready to use.")
        print()
        print("Next steps:")
        print("  1. Try the example usage:")
        print("     python example_usage.py")
        print()
        print("  2. Or test with command line:")
        print("     python depth_anything.py --image-path IMAGE.jpg --output-path depth.png")
        print()
        print("  3. Or use in your code:")
        print("     from depth_anything import DepthEstimator")
        print("     estimator = DepthEstimator(model_size='small')")
        print("     depth = estimator.estimate_and_save('image.jpg', 'depth.png')")

    elif not found_checkpoints:
        print("SETUP INCOMPLETE - Missing model checkpoints")
        print()
        print("Please download at least one model checkpoint:")
        print("  1. Run: ./setup_models.sh")
        print("  2. Or download manually from:")
        print("     https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth")
        print(f"     Save to: {checkpoint_dir}/depth_anything_v2_vits.pth")

    else:
        print("INSTALLATION INCOMPLETE - See errors above")
        print()
        print("Common fixes:")
        print("  1. Install missing dependencies:")
        print("     pip install -r Depth-Anything-V2/requirements.txt")
        print()
        print("  2. Ensure repository is cloned:")
        print("     git clone https://github.com/DepthAnything/Depth-Anything-V2")

    print("=" * 70)

    return 0 if (all_ok and found_checkpoints) else 1


if __name__ == '__main__':
    sys.exit(main())
