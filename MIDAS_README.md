# MiDaS Depth Estimation Setup

## Overview

This directory contains a complete setup for monocular depth estimation using the MiDaS (Monocular Depth Estimation) model from Intel ISL. The implementation includes:

1. **Cloned MiDaS Repository** - Official MiDaS implementation
2. **Custom Python Module** (`midas_depth.py`) - Easy-to-use wrapper with error handling
3. **Test Suite** (`test_midas.py`) - Validation and testing scripts
4. **Documentation** (`USAGE.md`) - Comprehensive usage guide

## Repository Structure

```
Depth/
├── MiDaS/                    # Cloned MiDaS repository
│   ├── midas/               # Core MiDaS modules
│   ├── weights/             # Model weights (downloaded on first use)
│   ├── run.py              # Original MiDaS script
│   ├── utils.py            # Utility functions
│   └── ...
├── midas_depth.py           # Custom depth estimation wrapper
├── test_midas.py            # Test and validation script
├── requirements.txt         # Python dependencies
├── USAGE.md                 # Detailed usage guide
└── MIDAS_README.md          # This file
```

## Features

### MiDaS Depth Estimator (`midas_depth.py`)

The custom wrapper provides:

- **Simple API**: Easy-to-use class interface for depth estimation
- **Multiple Models**: Support for 8 different MiDaS models (from tiny to large)
- **Automatic Downloads**: Model weights are downloaded automatically if missing
- **Error Handling**: Comprehensive error checking and helpful error messages
- **Flexible Input**: Accept image paths or NumPy arrays
- **Visualization**: Built-in colormap application and side-by-side visualization
- **Module or CLI**: Use as Python module or command-line tool
- **GPU Acceleration**: Automatic CUDA detection with optional half-precision optimization

### Available Models

| Model | Resolution | Parameters | FPS* | Quality | Use Case |
|-------|-----------|-----------|------|---------|----------|
| `midas_v21_small_256` | 256×256 | 21M | ~90 | Good | **Quick inference, real-time** |
| `dpt_swin2_tiny_256` | 256×256 | 42M | ~64 | Better | Fast with transformers |
| `dpt_levit_224` | 224×224 | 51M | ~73 | Better | Lightweight transformer |
| `midas_v21_384` | 384×384 | 105M | ~47 | Good | Classic MiDaS |
| `dpt_hybrid_384` | 384×384 | 123M | ~50 | Very Good | Balanced quality/speed |
| `dpt_large_384` | 384×384 | 344M | ~61 | Excellent | **High quality** |
| `dpt_swin2_large_384` | 384×384 | 213M | ~41 | Excellent | Very high quality |
| `dpt_beit_large_512` | 512×512 | 345M | ~6 | Best | **Highest quality** |

*FPS on RTX 3090 GPU

## Installation

### 1. Install Dependencies

```bash
cd /Users/rishabhshah/Desktop/Depth
pip install -r requirements.txt
```

### 2. Verify Installation

Run the test script to verify everything is set up correctly:

```bash
python test_midas.py
```

This will:
- Check all dependencies
- Verify MiDaS repository structure
- Test module imports
- Initialize a model
- Run depth estimation on a synthetic image

## Quick Start

### Command-Line Usage

```bash
# Basic usage with fast model
python midas_depth.py --input your_image.jpg --output depth_map.png

# Use high-quality model
python midas_depth.py -i image.jpg -o depth.png -m dpt_large_384

# Create side-by-side visualization
python midas_depth.py -i image.jpg -o depth.png --side-by-side

# List all available models
python midas_depth.py --list-models
```

### Python Module Usage

```python
from midas_depth import MiDaSDepthEstimator

# Initialize with default fast model
estimator = MiDaSDepthEstimator(model_type='midas_v21_small_256')

# Estimate depth from image
depth_map = estimator.estimate_depth('path/to/image.jpg')

# Save depth map with colormap
estimator.save_depth_map(depth_map, 'output.png', colormap='inferno')

# Get original image too
depth_map, original = estimator.estimate_depth('image.jpg', return_original=True)

# Create side-by-side visualization
estimator.visualize_side_by_side(original, depth_map, 'comparison.png')
```

## Understanding MiDaS Output

### What is MiDaS?

MiDaS (Monocular Depth Estimation) estimates **relative depth** from a single image. Key points:

1. **Relative, not metric**: Values are not in meters, but represent relative distances
2. **Inverse depth**: Higher values typically = farther from camera
3. **Zero-shot**: Works on any image without fine-tuning
4. **Multi-dataset trained**: Trained on up to 12 different datasets

### Depth Map Interpretation

```python
depth_map = estimator.estimate_depth('image.jpg')

# depth_map is a 2D numpy array
print(depth_map.shape)  # (height, width)
print(depth_map.min(), depth_map.max())  # Relative depth range

# Higher values = farther from camera (generally)
# Values are NOT metric depth (meters)
```

### Visualization Colormaps

- **inferno** (default): Purple (near) → Red → Yellow (far)
- **viridis**: Purple (near) → Green → Yellow (far)
- **magma**: Black (near) → Purple → Yellow (far)
- **gray**: Black (near) → White (far)

## Advanced Usage

### Custom Model Configuration

```python
from midas_depth import MiDaSDepthEstimator

# High-quality model with GPU optimization
estimator = MiDaSDepthEstimator(
    model_type='dpt_beit_large_512',
    optimize=True,    # Use half-precision on CUDA
    device='cuda'     # Force GPU
)

depth_map = estimator.estimate_depth('image.jpg')
```

### Processing NumPy Arrays

```python
import cv2
import numpy as np
from midas_depth import MiDaSDepthEstimator

estimator = MiDaSDepthEstimator()

# Load image as numpy array
image = cv2.imread('input.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

# Estimate depth directly
depth_map = estimator.estimate_depth(image_rgb)

# Process depth values
normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
```

### Batch Processing

```python
from pathlib import Path
from midas_depth import MiDaSDepthEstimator

estimator = MiDaSDepthEstimator(model_type='midas_v21_small_256')

# Process all images in a directory
input_dir = Path('input_images')
output_dir = Path('depth_maps')
output_dir.mkdir(exist_ok=True)

for image_path in input_dir.glob('*.jpg'):
    print(f"Processing {image_path.name}...")
    depth_map = estimator.estimate_depth(str(image_path))
    output_path = output_dir / f"{image_path.stem}_depth.png"
    estimator.save_depth_map(depth_map, str(output_path))
```

## Model Selection Guide

### Choose by Use Case

**Real-time applications / Video processing:**
- Use `midas_v21_small_256` (fastest, 90 FPS)
- Or `dpt_swin2_tiny_256` (faster, better quality)

**Offline image processing / Quality priority:**
- Use `dpt_large_384` (good balance)
- Or `dpt_beit_large_512` (best quality, slower)

**Embedded devices / Limited memory:**
- Use `midas_v21_small_256` (smallest, 21M params)
- Or `dpt_levit_224` (51M params)

**Research / Publication quality:**
- Use `dpt_beit_large_512` (highest accuracy)
- Or `dpt_swin2_large_384` (very high quality)

### Download Sizes

Model weights will be downloaded automatically on first use:
- Small models (midas_v21_small_256): ~80 MB
- Medium models (dpt_hybrid_384): ~480 MB
- Large models (dpt_large_384): ~1.3 GB
- Largest model (dpt_beit_large_512): ~1.4 GB

## Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'timm'
```bash
pip install timm==0.6.12 einops==0.6.0
```

#### 2. CUDA out of memory
- Use a smaller model: `midas_v21_small_256`
- Use CPU: `--device cpu`
- Reduce image resolution before processing

#### 3. Model weights not downloading
Manually download from: https://github.com/isl-org/MiDaS/releases
- v2.1 models: Releases page → v2_1
- v3.0 models: Releases page → v3
- v3.1 models: Releases page → v3_1

Place in: `MiDaS/weights/`

#### 4. Slow inference on CPU
- Use the smallest model: `midas_v21_small_256`
- Consider using GPU if available
- Reduce input image resolution

### Performance Tips

1. **GPU Acceleration**: Use CUDA with `device='cuda'`
2. **Half-precision**: Enable `optimize=True` for 2x speedup on GPU
3. **Smaller models**: Use `midas_v21_small_256` for speed
4. **Batch processing**: Initialize model once, reuse for multiple images

## Technical Details

### MiDaS Architecture

MiDaS uses various backbone architectures:
- **MiDaS v2.1**: EfficientNet-Lite3 (small) or ResNeXt (standard)
- **DPT**: Dense Prediction Transformers (Vision Transformers)
- **DPT v3.1**: BEiT, Swin, Swin2, Next-ViT, LeViT transformers

### Training Data

MiDaS 3.1 trained on 12 datasets:
- ReDWeb, DIML, Movies, MegaDepth, WSVD, TartanAir
- HRWSI, ApolloScape, BlendedMVS, IRS, KITTI, NYU Depth V2

### Zero-shot Transfer

MiDaS performs zero-shot cross-dataset transfer:
- No fine-tuning needed for new scenes
- Generalizes to diverse image types
- Robust to different domains (indoor, outdoor, synthetic, etc.)

## API Reference

See `USAGE.md` for complete API documentation including:
- Class methods and parameters
- Return types and shapes
- Error handling
- Code examples

## Comparison with Other Methods

### MiDaS vs Marigold
- **MiDaS**: Fast, relative depth, multiple model sizes
- **Marigold**: Slower, more accurate, affine-invariant, metric-capable

### MiDaS vs Depth Anything
- **MiDaS**: Well-established, transformer-based, proven accuracy
- **Depth Anything**: Newer, potentially better generalization

## Citation

If you use MiDaS in your research, please cite:

```bibtex
@ARTICLE{Ranftl2022,
    author  = "René Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun",
    title   = "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer",
    journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    year    = "2022",
    volume  = "44",
    number  = "3"
}
```

For DPT models:
```bibtex
@article{Ranftl2021,
    author    = {René Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
    title     = {Vision Transformers for Dense Prediction},
    journal   = {ICCV},
    year      = {2021},
}
```

For MiDaS 3.1:
```bibtex
@article{birkl2023midas,
    title={MiDaS v3.1 -- A Model Zoo for Robust Monocular Relative Depth Estimation},
    author={Reiner Birkl and Diana Wofk and Matthias Müller},
    journal={arXiv preprint arXiv:2307.14460},
    year={2023}
}
```

## Resources

- **Official Repository**: https://github.com/isl-org/MiDaS
- **Paper (v2)**: https://arxiv.org/abs/1907.01341
- **Paper (DPT)**: https://arxiv.org/abs/2103.13413
- **Technical Report (v3.1)**: https://arxiv.org/abs/2307.14460
- **PyTorch Hub**: https://pytorch.org/hub/intelisl_midas_v2/
- **Demo Video**: https://www.youtube.com/watch?v=UjaeNNFf9sE

## License

- **This wrapper**: Provided as-is for educational purposes
- **MiDaS**: MIT License (see MiDaS/LICENSE)

## Support

For issues with:
- **This wrapper**: Check test_midas.py output and USAGE.md
- **MiDaS model**: See official repository issues
- **Dependencies**: Check requirements.txt and installation

## Changelog

### 2026-01-09
- Initial implementation
- Support for 8 MiDaS models
- Automatic weight downloading
- Comprehensive error handling
- CLI and module usage
- Documentation and tests
