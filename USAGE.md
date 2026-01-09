# MiDaS Depth Estimation - Usage Guide

## Overview

This project provides a simple Python interface for monocular depth estimation using the MiDaS model. The `midas_depth.py` script can be used both as a command-line tool and as a Python module.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Repository Structure

```
Depth/
├── MiDaS/              # Cloned MiDaS repository
├── midas_depth.py      # Main depth estimation script
├── requirements.txt    # Python dependencies
└── USAGE.md           # This file
```

## Usage

### Command-Line Interface

#### Basic Usage

```bash
python midas_depth.py --input path/to/image.jpg --output depth_map.png
```

#### List Available Models

```bash
python midas_depth.py --list-models
```

Available models:
- `midas_v21_small_256` - Small fast model (256x256) - **best for quick inference**
- `midas_v21_384` - MiDaS v2.1 standard model (384x384)
- `dpt_swin2_tiny_256` - Tiny transformer model (256x256)
- `dpt_levit_224` - LeViT-based model (224x224)
- `dpt_hybrid_384` - DPT Hybrid model (384x384)
- `dpt_large_384` - DPT Large model (384x384) - good quality
- `dpt_swin2_large_384` - DPT Swin2 Large (384x384) - high quality
- `dpt_beit_large_512` - BEiT Large model (512x512) - highest quality

#### Advanced Options

```bash
# Use a different model
python midas_depth.py -i input.jpg -o output.png -m dpt_large_384

# Create side-by-side visualization
python midas_depth.py -i input.jpg -o output.png --side-by-side

# Use different colormap
python midas_depth.py -i input.jpg -o output.png --colormap viridis

# Use GPU optimization (half-precision)
python midas_depth.py -i input.jpg -o output.png --optimize

# Force CPU usage
python midas_depth.py -i input.jpg -o output.png --device cpu
```

#### All Command-Line Options

```
-i, --input         Path to input image (required)
-o, --output        Path to output depth map (default: depth_output.png)
-m, --model         Model type to use (default: midas_v21_small_256)
--colormap          Colormap for visualization (inferno, magma, plasma, viridis, jet, hot, gray)
--side-by-side      Create side-by-side visualization of input and depth
--optimize          Use half-precision optimization on CUDA
--device            Device to use (cuda or cpu)
--list-models       List all available models
```

### Python Module Usage

#### Basic Example

```python
from midas_depth import MiDaSDepthEstimator

# Initialize estimator with default small model
estimator = MiDaSDepthEstimator(model_type='midas_v21_small_256')

# Estimate depth from image path
depth_map = estimator.estimate_depth('path/to/image.jpg')

# Save depth map
estimator.save_depth_map(depth_map, 'output.png')
```

#### Advanced Example

```python
from midas_depth import MiDaSDepthEstimator
import numpy as np

# Initialize with a larger, more accurate model
estimator = MiDaSDepthEstimator(
    model_type='dpt_large_384',
    optimize=True,  # Use half-precision on GPU
    device='cuda'
)

# Get depth map and original image
depth_map, original_image = estimator.estimate_depth(
    'input.jpg',
    return_original=True
)

print(f"Depth map shape: {depth_map.shape}")
print(f"Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")

# Save with different colormap
estimator.save_depth_map(depth_map, 'depth_viridis.png', colormap='viridis')

# Create side-by-side visualization
estimator.visualize_side_by_side(original_image, depth_map, 'side_by_side.png')

# Use depth map for further processing
# depth_map is a numpy array with relative depth values
```

#### Using with NumPy Arrays

```python
from midas_depth import MiDaSDepthEstimator
import cv2

estimator = MiDaSDepthEstimator()

# Load image as numpy array
image = cv2.imread('input.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

# Estimate depth directly from array
depth_map = estimator.estimate_depth(image_rgb)

# Process depth map
print(f"Mean depth: {depth_map.mean():.3f}")
print(f"Std depth: {depth_map.std():.3f}")
```

## Model Selection Guide

### For Speed (Real-time applications)
- **midas_v21_small_256** - Fastest, 21M parameters, ~90 FPS on RTX 3090
- **dpt_swin2_tiny_256** - Small transformer, 42M parameters, ~64 FPS

### For Balanced Performance
- **midas_v21_384** - Classic MiDaS, 105M parameters, ~47 FPS
- **dpt_hybrid_384** - Good accuracy/speed trade-off, 123M parameters, ~50 FPS

### For Quality (Offline processing)
- **dpt_large_384** - High quality, 344M parameters, ~61 FPS
- **dpt_swin2_large_384** - Very high quality, 213M parameters, ~41 FPS
- **dpt_beit_large_512** - Highest quality, 345M parameters, ~6 FPS

## Output Formats

### Depth Map
- A 2D numpy array where each value represents relative depth
- Higher values = farther from camera
- Values are NOT metric depth (meters), but relative inverse depth

### Visualization
- Color-mapped depth images saved as PNG
- Default colormap: 'inferno' (purple = near, yellow = far)
- Normalized to 0-255 range for visualization

## Troubleshooting

### Model Weights Not Found
If weights are not found, the script will attempt to download them automatically from the official MiDaS releases. If automatic download fails, manually download from:
- https://github.com/isl-org/MiDaS/releases

Place downloaded weights in: `MiDaS/weights/`

### CUDA Out of Memory
Try:
1. Use a smaller model (e.g., `midas_v21_small_256`)
2. Use CPU: `--device cpu`
3. Process smaller images

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## API Reference

### Class: MiDaSDepthEstimator

#### `__init__(model_type='midas_v21_small_256', optimize=False, device=None)`
Initialize the depth estimator.

**Parameters:**
- `model_type` (str): Model to use
- `optimize` (bool): Use half-precision on CUDA
- `device` (str): 'cuda' or 'cpu', auto-detected if None

#### `estimate_depth(image_input, return_original=False)`
Estimate depth from an image.

**Parameters:**
- `image_input` (str or np.ndarray): Image path or RGB array (0-1 range)
- `return_original` (bool): Return (depth_map, original_image) tuple

**Returns:**
- np.ndarray: Depth map
- tuple: (depth_map, original_image) if return_original=True

#### `save_depth_map(depth_map, output_path, colormap='inferno', normalize=True)`
Save depth map as image.

**Parameters:**
- `depth_map` (np.ndarray): Depth map to save
- `output_path` (str): Output file path
- `colormap` (str): Colormap to apply
- `normalize` (bool): Normalize to 0-255 range

#### `visualize_side_by_side(original_image, depth_map, output_path=None)`
Create side-by-side visualization.

**Parameters:**
- `original_image` (np.ndarray): Original RGB image
- `depth_map` (np.ndarray): Depth map
- `output_path` (str, optional): Save path

**Returns:**
- np.ndarray: Side-by-side image

#### `list_available_models()` (static)
Print all available models.

## Examples

See the `examples/` directory for more usage examples (coming soon).

## Citation

If you use this code or MiDaS in your work, please cite:

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

## License

This wrapper code is provided as-is. MiDaS is licensed under MIT License.
