# Marigold Depth Estimation - Quick Start Guide

This guide will help you get started with Marigold depth estimation in just a few minutes.

## What is Marigold?

Marigold is a state-of-the-art depth estimation model that repurposes Stable Diffusion for monocular depth estimation. It achieves excellent zero-shot generalization on unseen images.

**Key Features:**
- Based on Stable Diffusion (diffusion model)
- State-of-the-art zero-shot depth estimation
- Scale and shift invariant (affine-invariant)
- CVPR 2024 Oral, Best Paper Award Candidate

## Prerequisites

You should already have:
1. Python 3.8 or later
2. The Marigold repository cloned in `/Users/rishabhshah/Desktop/Depth/Marigold`
3. A CUDA-capable GPU (recommended) or Apple Silicon Mac

## Installation

### 1. Install Dependencies

```bash
cd /Users/rishabhshah/Desktop/Depth
pip install -r requirements.txt
```

This will install:
- PyTorch 2.4.1+
- diffusers 0.25.0+
- transformers 4.32.1+
- Other required packages

### 2. Verify Installation

The model weights (~5GB) will be automatically downloaded on first use.

## Basic Usage

### Option 1: As a Python Module (Recommended)

```python
from marigold_depth import MarigoldDepthEstimator

# Initialize (downloads model on first run - ~5GB)
estimator = MarigoldDepthEstimator()

# Estimate depth
depth_map, colored_depth = estimator.estimate_depth("your_image.jpg")

# Save results
estimator.save_depth(depth_map, "output", colored_depth)
```

This creates:
- `output.npy` - Raw depth array (normalized to [0, 1])
- `output.png` - 16-bit grayscale depth map
- `output_colored.png` - Colored visualization

### Option 2: Command Line

```bash
python marigold_depth.py \
    --image_path your_image.jpg \
    --output_path output
```

## Quick Examples

### 1. Basic Depth Estimation

```python
from marigold_depth import MarigoldDepthEstimator

estimator = MarigoldDepthEstimator()
depth_map, colored = estimator.estimate_depth("image.jpg")
estimator.save_depth(depth_map, "output", colored)
```

### 2. Fast Inference (Half Precision)

```python
estimator = MarigoldDepthEstimator(half_precision=True)
depth_map, colored = estimator.estimate_depth(
    "image.jpg",
    ensemble_size=1,
    processing_res=512
)
```

### 3. High Quality (Slow but Best Results)

```python
estimator = MarigoldDepthEstimator()
depth_map, colored = estimator.estimate_depth(
    "image.jpg",
    ensemble_size=10,
    processing_res=768,
    seed=42
)
```

### 4. Batch Processing

```python
from pathlib import Path

estimator = MarigoldDepthEstimator()

for img in Path("input_images").glob("*.jpg"):
    depth_map, colored = estimator.estimate_depth(img)
    estimator.save_depth(depth_map, f"output/{img.stem}", colored)
```

## Understanding the Output

### Depth Map Values

The depth map is a numpy array normalized to [0, 1]:
- **0** = Closest to camera
- **1** = Farthest from camera

```python
import numpy as np

depth_map, _ = estimator.estimate_depth("image.jpg")

print(f"Shape: {depth_map.shape}")
print(f"Min depth: {depth_map.min():.3f}")
print(f"Max depth: {depth_map.max():.3f}")
print(f"Mean depth: {depth_map.mean():.3f}")
```

### File Formats

1. **`.npy` file**: Raw numpy array (float32)
   - Load with: `depth = np.load("output.npy")`
   - Values in range [0, 1]

2. **`.png` file**: 16-bit grayscale
   - Load with PIL: `img = Image.open("output.png")`
   - Values scaled to [0, 65535]

3. **`_colored.png` file**: RGB visualization
   - Standard 8-bit RGB image
   - Uses colormap (default: Spectral)

## Command-Line Options

### Basic Options

```bash
# Required
--image_path input.jpg     # Input image path
--output_path output        # Output base path (no extension)

# Common options
--half_precision           # Use FP16 (faster, less VRAM)
--seed 42                  # Reproducibility
--ensemble_size 5          # Number of predictions to average
--color_map viridis        # Colormap (Spectral, viridis, plasma, etc.)
```

### Example Commands

```bash
# Fast inference
python marigold_depth.py \
    --image_path input.jpg \
    --output_path output \
    --half_precision \
    --ensemble_size 1

# High quality
python marigold_depth.py \
    --image_path input.jpg \
    --output_path output \
    --ensemble_size 10 \
    --seed 42

# Custom checkpoint
python marigold_depth.py \
    --image_path input.jpg \
    --output_path output \
    --checkpoint prs-eth/marigold-depth-lcm-v1-0
```

## Testing Your Installation

Run the test script:

```bash
python test_marigold_depth.py
```

This will:
1. Create a test image
2. Run depth estimation
3. Save output files
4. Show usage examples

## Performance Tips

### Speed vs Quality Trade-offs

| Setting | Speed | Quality | VRAM |
|---------|-------|---------|------|
| `half_precision=True, ensemble_size=1` | Fast | Good | Low |
| Default settings | Medium | Very Good | Medium |
| `ensemble_size=10, processing_res=768` | Slow | Best | High |

### Recommended Settings

**For real-time/interactive applications:**
```python
estimator = MarigoldDepthEstimator(half_precision=True)
depth, _ = estimator.estimate_depth(img, ensemble_size=1, processing_res=512)
```

**For best quality:**
```python
estimator = MarigoldDepthEstimator()
depth, _ = estimator.estimate_depth(img, ensemble_size=10, processing_res=768)
```

**For batch processing:**
```python
estimator = MarigoldDepthEstimator(half_precision=True)
for img in images:
    depth, colored = estimator.estimate_depth(img, show_progress=False)
```

## Troubleshooting

### Out of Memory (OOM)

**Solution 1**: Use half precision
```python
estimator = MarigoldDepthEstimator(half_precision=True)
```

**Solution 2**: Lower processing resolution
```python
depth, _ = estimator.estimate_depth(img, processing_res=512)
```

**Solution 3**: Reduce ensemble size
```python
depth, _ = estimator.estimate_depth(img, ensemble_size=1)
```

### Slow Inference

**Solution 1**: Use half precision and single prediction
```python
estimator = MarigoldDepthEstimator(half_precision=True)
depth, _ = estimator.estimate_depth(img, ensemble_size=1)
```

**Solution 2**: Use LCM checkpoint (faster diffusion)
```python
estimator = MarigoldDepthEstimator(
    checkpoint="prs-eth/marigold-depth-lcm-v1-0",
    half_precision=True
)
```

### Import Errors

Check that Marigold is in the correct location:
```bash
ls Marigold/marigold/marigold_depth_pipeline.py
```

If missing, clone it:
```bash
git clone https://github.com/prs-eth/Marigold.git
```

## Available Checkpoints

| Checkpoint | Description | Steps | Speed |
|-----------|-------------|-------|-------|
| `prs-eth/marigold-depth-v1-1` | **Latest** (recommended) | 4 | Medium |
| `prs-eth/marigold-depth-v1-0` | Original CVPR 2024 | 50 | Slow |
| `prs-eth/marigold-depth-lcm-v1-0` | Latent Consistency Model | 1-4 | Fast |

Example using LCM checkpoint:
```python
estimator = MarigoldDepthEstimator(
    checkpoint="prs-eth/marigold-depth-lcm-v1-0"
)
```

## Next Steps

1. **Read the full README**: See `README.md` for detailed API documentation
2. **Try examples**: Run `example_usage.py` for 7 different use cases
3. **Test suite**: Run `test_marigold_depth.py` for comprehensive testing
4. **Original repo**: Check `Marigold/README.md` for training and evaluation

## Resources

- **Project Website**: https://marigoldcomputervision.github.io
- **Paper**: https://arxiv.org/abs/2312.02145
- **HuggingFace Demo**: https://huggingface.co/spaces/prs-eth/marigold
- **Models**: https://huggingface.co/prs-eth
- **GitHub**: https://github.com/prs-eth/Marigold

## Common Questions

### Q: What's the difference between Marigold and MiDaS?

**Marigold**:
- Based on diffusion models (Stable Diffusion)
- Better generalization to unseen domains
- Slower (requires multiple diffusion steps)
- State-of-the-art quality

**MiDaS**:
- Based on transformers
- Faster inference
- Good quality
- More established

### Q: How accurate is the depth estimation?

Marigold provides **relative depth**, meaning:
- The scale is arbitrary (affine-invariant)
- Use it for depth ordering, not absolute distances
- Perfect for 3D reconstruction, AR/VR, image editing
- Not suitable for metric depth (measuring distances)

### Q: Can I use this commercially?

Check the licenses:
- Marigold code: Apache License 2.0
- Marigold models: RAIL++-M License
- See `Marigold/LICENSE.txt` and `Marigold/LICENSE-MODEL.txt`

### Q: Which GPU do I need?

**Minimum**: 6GB VRAM (with half precision)
**Recommended**: 8GB+ VRAM (for full precision)
**CPU**: Works but 50-100x slower

Apple Silicon (M1/M2/M3) works with MPS acceleration.

## Getting Help

If you encounter issues:

1. Check this guide and `README.md`
2. Run `test_marigold_depth.py` to verify installation
3. Check the original Marigold repository for issues
4. Ensure all dependencies are correctly installed

## Citation

If you use Marigold in your work, please cite:

```bibtex
@InProceedings{ke2023repurposing,
  title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
  author={Bingxin Ke and Anton Obukhov and Shengyu Huang and Nando Metzger and Rodrigo Caye Daudt and Konrad Schindler},
  booktitle = {CVPR},
  year={2024}
}
```
