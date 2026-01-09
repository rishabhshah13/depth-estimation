# Quick Start Guide - Depth Anything V2

This guide will get you up and running with depth estimation in 5 minutes.

## Current Status

- Repository: CLONED
- Code: READY
- Dependencies: MOSTLY INSTALLED (torchvision missing)
- Models: NOT DOWNLOADED YET

## Step 1: Install Missing Dependencies

```bash
# Install torchvision (required for the model)
pip install torchvision

# Or install all dependencies at once
pip install -r Depth-Anything-V2/requirements.txt
```

## Step 2: Download Model Checkpoint

You need at least one model. The small model is recommended for quick testing:

### Option A: Automatic Download (Recommended)

```bash
# Run the setup script
./setup_models.sh
```

Select option 1 for the small model (fastest, good quality).

### Option B: Manual Download

Download the small model checkpoint:

```bash
# Create checkpoints directory
mkdir -p Depth-Anything-V2/checkpoints

# Download with wget
wget -O Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

# Or with curl
curl -L -o Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

File size: ~100MB (may take a few minutes depending on your connection)

## Step 3: Verify Installation

```bash
python test_installation.py
```

You should see "SUCCESS! Installation is complete and ready to use."

## Step 4: Try It Out!

### Command Line

```bash
# Use one of the example images
python depth_anything.py \
  --image-path Depth-Anything-V2/assets/examples/demo01.jpg \
  --output-path my_first_depth.png
```

The depth map will be saved to `my_first_depth.png`!

### Python Code

Create a file `test_depth.py`:

```python
from depth_anything import DepthEstimator

# Initialize
estimator = DepthEstimator(model_size='small')

# Process an image
depth = estimator.estimate_and_save(
    'Depth-Anything-V2/assets/examples/demo01.jpg',
    'my_depth_output.png'
)

print(f"Success! Depth map saved.")
print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}]")
```

Run it:
```bash
python test_depth.py
```

## What's Next?

1. **See more examples:**
   ```bash
   python example_usage.py
   ```

2. **Read the full documentation:**
   - See `README_DEPTH_ESTIMATION.md` for complete API reference
   - Includes batch processing, custom settings, and more

3. **Process your own images:**
   ```bash
   python depth_anything.py --image-path YOUR_IMAGE.jpg --output-path depth.png
   ```

## Common Issues

### "Checkpoint file not found"
- You need to download the model checkpoint (see Step 2)

### "ImportError: No module named 'torchvision'"
- Run: `pip install torchvision`

### "CUDA out of memory"
- Use CPU instead: `--device cpu`
- Or use the small model: `--model-size small`

### "Could not load image"
- Check that the image path is correct
- Make sure the image format is supported (jpg, png, etc.)

## File Overview

```
Depth/
├── depth_anything.py           # Main module (USE THIS)
├── example_usage.py            # Examples of how to use it
├── test_installation.py        # Verify everything works
├── setup_models.sh             # Download model checkpoints
├── QUICK_START.md              # This file
├── README_DEPTH_ESTIMATION.md  # Full documentation
└── Depth-Anything-V2/          # Original repository
    ├── checkpoints/            # Put model files here
    └── assets/examples/        # Example images
```

## Model Options

| Model | Size | Speed | Quality | Command |
|-------|------|-------|---------|---------|
| Small | 100MB | Fast | Good | `--model-size small` |
| Base | 400MB | Medium | Better | `--model-size base` |
| Large | 1.3GB | Slow | Best | `--model-size large` |

For quick testing, use **small**. For production, consider **base** or **large**.

## Need Help?

1. Check `README_DEPTH_ESTIMATION.md` for detailed documentation
2. Run `python test_installation.py` to diagnose issues
3. Check the original repository: https://github.com/DepthAnything/Depth-Anything-V2

## License

- Small model: Apache-2.0 (free for commercial use)
- Base/Large models: CC-BY-NC-4.0 (non-commercial only)
