# Marigold Depth Estimation Wrapper

A simple, self-contained Python script for monocular depth estimation using the Marigold diffusion model.

## Overview

This wrapper provides an easy-to-use interface for the Marigold depth estimation model, which repurposes Stable Diffusion for high-quality monocular depth estimation. The model achieves state-of-the-art zero-shot generalization on unseen images.

**Paper**: [Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation](https://arxiv.org/abs/2312.02145) (CVPR 2024 - Oral, Best Paper Award Candidate)

## Repository Structure

```
Depth/
├── Marigold/                    # Cloned Marigold repository
│   ├── marigold/               # Core pipeline implementations
│   ├── requirements.txt        # Marigold dependencies
│   └── README.md               # Original Marigold documentation
├── marigold_depth.py           # Main wrapper script (this module)
├── test_marigold_depth.py      # Test script with examples
└── README.md                   # This file
```

## Installation

### 1. Clone this repository (if not already done)

```bash
git clone https://github.com/prs-eth/Marigold.git
cd Depth
```

### 2. Install dependencies

```bash
pip install -r Marigold/requirements.txt
```

Required packages:
- `torch>=2.4.1`
- `torchvision>=0.19.1`
- `diffusers>=0.25.0`
- `transformers>=4.32.1`
- `accelerate>=0.22.0`
- `matplotlib`
- `scipy`
- `Pillow`
- `numpy`

### 3. (Optional) GPU Setup

For CUDA (NVIDIA GPUs):
```bash
# Ensure PyTorch is installed with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For Apple Silicon (M1/M2/M3):
```bash
# MPS support is built into PyTorch, no additional setup needed
```

## Usage

### As a Python Module

```python
from marigold_depth import MarigoldDepthEstimator

# Initialize estimator (first run will download model weights ~5GB)
estimator = MarigoldDepthEstimator()

# Perform depth estimation
depth_map, colored_depth = estimator.estimate_depth("input.jpg")

# Save results
estimator.save_depth(
    depth_map=depth_map,
    output_path="output",
    colored_depth=colored_depth
)
```

This will create:
- `output.npy`: Raw depth map as numpy array (float32, normalized to [0, 1])
- `output.png`: 16-bit grayscale depth map
- `output_colored.png`: Colored visualization using Spectral colormap

### As a Command-Line Tool

```bash
# Basic usage
python marigold_depth.py \
    --image_path input.jpg \
    --output_path output

# With custom settings
python marigold_depth.py \
    --image_path input.jpg \
    --output_path output \
    --half_precision \
    --ensemble_size 5 \
    --seed 42 \
    --color_map viridis

# High-quality academic benchmark settings
python marigold_depth.py \
    --image_path input.jpg \
    --output_path output \
    --denoising_steps 1 \
    --ensemble_size 10
```

### Command-Line Arguments

**Required:**
- `--image_path`: Path to input image
- `--output_path`: Base path for output files (without extension)

**Model Options:**
- `--checkpoint`: Model checkpoint (default: `prs-eth/marigold-depth-v1-1`)
- `--device`: Device to use (`cuda`, `mps`, `cpu`, or auto-detect)
- `--half_precision`: Use FP16 for faster inference (may reduce quality)

**Inference Options:**
- `--denoising_steps`: Number of denoising steps (default: 4 for v1-1)
- `--ensemble_size`: Number of predictions to ensemble (default: 1)
- `--processing_res`: Processing resolution (default: 768, use 0 for native)
- `--seed`: Random seed for reproducibility
- `--color_map`: Colormap for visualization (default: Spectral)
- `--no_colored`: Skip colored visualization
- `--no_raw`: Skip 16-bit PNG output

## API Reference

### MarigoldDepthEstimator

```python
class MarigoldDepthEstimator:
    def __init__(
        self,
        checkpoint: str = "prs-eth/marigold-depth-v1-1",
        device: Optional[str] = None,
        half_precision: bool = False,
    )
```

**Parameters:**
- `checkpoint`: HuggingFace checkpoint or local path to model
- `device`: Device to use ('cuda', 'mps', 'cpu', or None for auto)
- `half_precision`: Use FP16 precision (faster but may reduce quality)

### estimate_depth

```python
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
) -> Tuple[np.ndarray, Optional[Image.Image]]
```

**Parameters:**
- `image_path`: Path to image or PIL Image object
- `denoising_steps`: Number of diffusion steps (None = model default)
- `ensemble_size`: Number of predictions to ensemble (higher = better but slower)
- `processing_res`: Processing resolution (None = 768, 0 = native)
- `match_input_res`: Resize output to match input resolution
- `batch_size`: Inference batch size (0 = automatic)
- `seed`: Random seed for reproducibility
- `color_map`: Matplotlib colormap name (or None to skip)
- `show_progress`: Show progress bar

**Returns:**
- `depth_map`: Numpy array [H, W] with values in [0, 1]
- `colored_depth`: PIL Image with colored visualization (or None)

### save_depth

```python
def save_depth(
    self,
    depth_map: np.ndarray,
    output_path: Union[str, Path],
    colored_depth: Optional[Image.Image] = None,
    save_raw: bool = True,
    save_colored: bool = True,
) -> None
```

**Parameters:**
- `depth_map`: Depth map numpy array
- `output_path`: Base path for output (without extension)
- `colored_depth`: Colored visualization to save
- `save_raw`: Save 16-bit PNG
- `save_colored`: Save colored visualization

## Advanced Examples

### Working with Depth Maps

```python
import numpy as np
from marigold_depth import MarigoldDepthEstimator

estimator = MarigoldDepthEstimator()
depth_map, _ = estimator.estimate_depth("input.jpg")

# Depth map is normalized to [0, 1]
# where 0 = closest, 1 = furthest

# Get depth statistics
print(f"Mean depth: {np.mean(depth_map):.3f}")
print(f"Std depth: {np.std(depth_map):.3f}")

# Threshold near objects
near_mask = depth_map < 0.3
far_mask = depth_map > 0.7

# Create custom visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(depth_map, cmap='gray')
plt.title('Depth Map')
plt.subplot(132)
plt.imshow(near_mask, cmap='Reds')
plt.title('Near Objects')
plt.subplot(133)
plt.imshow(far_mask, cmap='Blues')
plt.title('Far Objects')
plt.show()
```

### High-Quality Depth Estimation

```python
from marigold_depth import MarigoldDepthEstimator

# Use full precision and large ensemble for best quality
estimator = MarigoldDepthEstimator(half_precision=False)

depth_map, colored = estimator.estimate_depth(
    "input.jpg",
    denoising_steps=4,      # More steps = higher quality (diminishing returns)
    ensemble_size=10,        # More ensembles = better but slower
    seed=42,                 # For reproducibility
    processing_res=768,      # Higher resolution for more detail
)

estimator.save_depth(depth_map, "high_quality_output", colored)
```

### Fast Inference

```python
from marigold_depth import MarigoldDepthEstimator

# Use half precision and single prediction for speed
estimator = MarigoldDepthEstimator(half_precision=True)

depth_map, colored = estimator.estimate_depth(
    "input.jpg",
    ensemble_size=1,         # Single prediction
    processing_res=512,      # Lower resolution
    show_progress=False,     # Disable progress bar
)
```

### Batch Processing

```python
from pathlib import Path
from marigold_depth import MarigoldDepthEstimator

estimator = MarigoldDepthEstimator()

# Process all images in a directory
input_dir = Path("input_images")
output_dir = Path("output_depths")
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob("*.jpg"):
    print(f"Processing {img_path.name}...")
    depth_map, colored = estimator.estimate_depth(img_path)

    output_path = output_dir / img_path.stem
    estimator.save_depth(depth_map, output_path, colored)
```

### Using Different Checkpoints

```python
from marigold_depth import MarigoldDepthEstimator

# Original CVPR 2024 version (v1.0)
estimator_v10 = MarigoldDepthEstimator(
    checkpoint="prs-eth/marigold-depth-v1-0"
)

# Latest version (v1.1) - recommended
estimator_v11 = MarigoldDepthEstimator(
    checkpoint="prs-eth/marigold-depth-v1-1"
)

# Latent Consistency Model (faster, 1-4 steps)
estimator_lcm = MarigoldDepthEstimator(
    checkpoint="prs-eth/marigold-depth-lcm-v1-0"
)
```

## Testing

Run the test script to verify installation:

```bash
python test_marigold_depth.py
```

This will:
1. Create a test image
2. Initialize the estimator
3. Perform depth estimation
4. Save output files
5. Display usage examples

## Model Information

### Available Checkpoints

| Checkpoint | Description | Denoising Steps | Speed |
|-----------|-------------|-----------------|-------|
| `prs-eth/marigold-depth-v1-1` | Latest version (recommended) | 4 | Normal |
| `prs-eth/marigold-depth-v1-0` | Original CVPR 2024 version | 50 | Slow |
| `prs-eth/marigold-depth-lcm-v1-0` | Latent Consistency Model | 1-4 | Fast |

### Model Properties

- **Scale-invariant**: Yes (predicts relative depth)
- **Shift-invariant**: Yes (affine-invariant)
- **Input resolution**: Any (recommended: 768x768)
- **Processing resolution**: 768x768 (default)
- **Output**: Depth map normalized to [0, 1]

## Performance Tips

1. **GPU Memory**: Use `--half_precision` to reduce VRAM usage
2. **Speed**: Set `ensemble_size=1` for fastest inference
3. **Quality**: Use `ensemble_size=10` with full precision for best results
4. **Batch Processing**: Initialize estimator once, reuse for multiple images
5. **Resolution**: Lower `processing_res` for faster inference, higher for detail

## Troubleshooting

### Out of Memory (OOM)

```python
# Use half precision
estimator = MarigoldDepthEstimator(half_precision=True)

# Lower processing resolution
depth_map, _ = estimator.estimate_depth("input.jpg", processing_res=512)
```

### Slow Inference

```python
# Reduce ensemble size
depth_map, _ = estimator.estimate_depth("input.jpg", ensemble_size=1)

# Use LCM checkpoint (faster)
estimator = MarigoldDepthEstimator(
    checkpoint="prs-eth/marigold-depth-lcm-v1-0"
)
```

### Import Errors

Ensure Marigold repository is in the correct location:
```bash
ls Marigold/marigold/  # Should show pipeline files
```

## Citation

If you use this code or the Marigold model, please cite:

```bibtex
@InProceedings{ke2023repurposing,
  title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
  author={Bingxin Ke and Anton Obukhov and Shengyu Huang and Nando Metzger and Rodrigo Caye Daudt and Konrad Schindler},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}

@misc{ke2025marigold,
  title={Marigold: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis},
  author={Bingxin Ke and Kevin Qu and Tianfu Wang and Nando Metzger and Shengyu Huang and Bo Li and Anton Obukhov and Konrad Schindler},
  year={2025},
  eprint={2505.09358},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## License

- This wrapper code is provided as-is for educational purposes
- Marigold code: Apache License, Version 2.0
- Marigold models: RAIL++-M License

See the original [Marigold repository](https://github.com/prs-eth/Marigold) for full license details.

## Links

- **Marigold Project**: https://marigoldcomputervision.github.io
- **Paper**: https://arxiv.org/abs/2312.02145
- **HuggingFace Demo**: https://huggingface.co/spaces/prs-eth/marigold
- **Models**: https://huggingface.co/prs-eth
- **Diffusers Tutorial**: https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage

## Acknowledgments

This wrapper is based on the excellent work by the Marigold team at ETH Zürich. The original implementation and training code can be found in the [official repository](https://github.com/prs-eth/Marigold).
