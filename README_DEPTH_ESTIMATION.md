# Depth Anything V2 - Easy Depth Estimation

A simplified Python interface for depth estimation using the Depth Anything V2 model. This module provides both a command-line interface and a Python API for easy integration into your projects.

## Features

- Simple, intuitive API for depth estimation
- Support for multiple model sizes (small, base, large)
- Can be used as both a standalone script and importable module
- Automatic device detection (CUDA, MPS, CPU)
- Comprehensive error handling
- Batch processing support
- Multiple output formats (color, grayscale)

## Repository Structure

```
Depth/
├── Depth-Anything-V2/          # Cloned Depth Anything V2 repository
│   ├── depth_anything_v2/      # Model implementation
│   ├── checkpoints/            # Model weights (download separately)
│   └── assets/examples/        # Example images
├── depth_anything.py           # Main module (standalone/importable)
├── example_usage.py            # Usage examples
└── README_DEPTH_ESTIMATION.md  # This file
```

## Installation

### 1. Clone the Repository

The Depth Anything V2 repository should already be cloned to:
```bash
/Users/rishabhshah/Desktop/Depth/Depth-Anything-V2
```

### 2. Install Dependencies

```bash
pip install -r Depth-Anything-V2/requirements.txt
```

Required packages:
- torch
- torchvision
- opencv-python
- matplotlib
- numpy

### 3. Download Model Checkpoints

Download at least one model checkpoint and place it in `Depth-Anything-V2/checkpoints/`:

**Small Model (Recommended for quick inference):**
- Size: 24.8M parameters
- Download: https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
- Save as: `Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth`
- License: Apache-2.0

**Base Model:**
- Size: 97.5M parameters
- Download: https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
- Save as: `Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth`
- License: CC-BY-NC-4.0

**Large Model:**
- Size: 335.3M parameters
- Download: https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
- Save as: `Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth`
- License: CC-BY-NC-4.0

You can download checkpoints using wget or curl:

```bash
# Create checkpoints directory
mkdir -p Depth-Anything-V2/checkpoints

# Download small model
wget -P Depth-Anything-V2/checkpoints https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

# Or using curl
curl -L https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth \
  -o Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth
```

## Usage

### Command-Line Interface

```bash
# Basic usage
python depth_anything.py --image-path image.jpg --output-path depth.png

# Use larger model for better quality
python depth_anything.py --image-path image.jpg --output-path depth.png --model-size large

# Save as grayscale
python depth_anything.py --image-path image.jpg --output-path depth.png --grayscale

# Custom input size for higher resolution
python depth_anything.py --image-path image.jpg --output-path depth.png --input-size 1024

# Specify device
python depth_anything.py --image-path image.jpg --output-path depth.png --device cuda

# Use custom colormap
python depth_anything.py --image-path image.jpg --output-path depth.png --colormap viridis
```

### Python API

#### Basic Usage

```python
from depth_anything import DepthEstimator

# Initialize estimator
estimator = DepthEstimator(model_size='small')

# Estimate and save in one call
depth = estimator.estimate_and_save('image.jpg', 'depth.png')
```

#### Separate Estimation and Saving

```python
from depth_anything import DepthEstimator

estimator = DepthEstimator(model_size='small')

# Step 1: Estimate depth
depth = estimator.estimate_depth('image.jpg')

# Step 2: Do something with the depth array
print(f"Depth range: [{depth.min()}, {depth.max()}]")

# Step 3: Save with custom options
estimator.save_depth_map(depth, 'depth_color.png', colormap='viridis')
estimator.save_depth_map(depth, 'depth_gray.png', grayscale=True)
```

#### Process Numpy Arrays

```python
import cv2
from depth_anything import DepthEstimator

estimator = DepthEstimator(model_size='small')

# Load image as numpy array
image = cv2.imread('image.jpg')

# Process directly
depth = estimator.estimate_depth(image)

# Save result
estimator.save_depth_map(depth, 'depth.png')
```

#### Batch Processing

```python
from pathlib import Path
from depth_anything import DepthEstimator

estimator = DepthEstimator(model_size='small')

# Process multiple images
image_paths = Path('images').glob('*.jpg')

for img_path in image_paths:
    output_path = f'output/{img_path.stem}_depth.png'
    estimator.estimate_and_save(img_path, output_path)
    print(f"Processed: {img_path.name}")
```

#### Higher Quality Settings

```python
from depth_anything import DepthEstimator

# Use larger model and input size for better quality
estimator = DepthEstimator(model_size='base')  # or 'large'

depth = estimator.estimate_depth(
    'image.jpg',
    input_size=1024  # Higher resolution (default is 518)
)

estimator.save_depth_map(depth, 'high_quality_depth.png')
```

#### Custom Device Selection

```python
from depth_anything import DepthEstimator

# Force CPU
estimator = DepthEstimator(model_size='small', device='cpu')

# Use CUDA
estimator = DepthEstimator(model_size='small', device='cuda')

# Use MPS (Apple Silicon)
estimator = DepthEstimator(model_size='small', device='mps')

# Auto-detect (default)
estimator = DepthEstimator(model_size='small')  # device=None
```

## API Reference

### DepthEstimator Class

#### Constructor

```python
DepthEstimator(
    model_size: str = 'small',
    checkpoint_dir: Optional[str] = None,
    device: Optional[str] = None
)
```

**Parameters:**
- `model_size`: Model size ('small', 'base', or 'large'). Default: 'small'
- `checkpoint_dir`: Directory containing checkpoints. Default: auto-detect
- `device`: Device to use ('cuda', 'mps', 'cpu', or None for auto). Default: None

#### Methods

##### estimate_depth()

```python
estimate_depth(
    image_path: Union[str, Path, np.ndarray],
    input_size: int = 518
) -> np.ndarray
```

Estimate depth from an image.

**Parameters:**
- `image_path`: Path to image file or numpy array (BGR format)
- `input_size`: Input size for model. Default: 518

**Returns:**
- 2D numpy array (HxW) containing depth values

##### save_depth_map()

```python
save_depth_map(
    depth: np.ndarray,
    output_path: Union[str, Path],
    colormap: str = 'Spectral_r',
    normalize: bool = True,
    grayscale: bool = False
) -> None
```

Save depth map as an image.

**Parameters:**
- `depth`: 2D depth array to save
- `output_path`: Where to save the image
- `colormap`: Matplotlib colormap name. Default: 'Spectral_r'
- `normalize`: Whether to normalize to [0, 255]. Default: True
- `grayscale`: Save as grayscale. Default: False

##### estimate_and_save()

```python
estimate_and_save(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    input_size: int = 518,
    colormap: str = 'Spectral_r',
    grayscale: bool = False
) -> np.ndarray
```

Convenience method to estimate and save in one call.

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic usage
- Separate estimation and saving
- Processing numpy arrays
- Higher quality settings
- Batch processing
- Depth analysis
- Error handling

Run examples:
```bash
python example_usage.py
```

## Model Comparison

| Model | Parameters | Speed | Quality | Memory | Best For |
|-------|-----------|-------|---------|--------|----------|
| Small | 24.8M | Fast | Good | Low | Real-time, quick inference |
| Base | 97.5M | Medium | Better | Medium | Balanced quality/speed |
| Large | 335.3M | Slow | Best | High | Highest quality results |

## Troubleshooting

### Model checkpoint not found

**Error:** `FileNotFoundError: Checkpoint file not found`

**Solution:** Download the checkpoint file and place it in the correct directory:
```bash
mkdir -p Depth-Anything-V2/checkpoints
wget -P Depth-Anything-V2/checkpoints https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

### Import error

**Error:** `ImportError: Could not import Depth Anything V2 module`

**Solution:** Ensure the repository is cloned in the same directory as `depth_anything.py`

### CUDA out of memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
- Use a smaller model size (`model_size='small'`)
- Reduce input size (`input_size=384`)
- Use CPU instead (`device='cpu'`)

### Image loading fails

**Error:** `ValueError: Could not load image`

**Solution:**
- Check file path is correct
- Ensure image format is supported (jpg, png, etc.)
- Verify file is not corrupted

## Performance Tips

1. **Model Selection:**
   - Use 'small' for real-time or batch processing
   - Use 'base' for balanced quality/speed
   - Use 'large' only when quality is critical

2. **Input Size:**
   - Default (518) works well for most cases
   - Increase for higher resolution (slower)
   - Decrease for faster inference (lower quality)

3. **Device:**
   - CUDA (NVIDIA GPU) is fastest
   - MPS (Apple Silicon) is good alternative
   - CPU works but is slower

4. **Batch Processing:**
   - Reuse the same DepthEstimator instance
   - Process multiple images without reinitializing

## Citation

If you use this code, please cite the original Depth Anything V2 paper:

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}
```

## License

- Small model: Apache-2.0
- Base/Large models: CC-BY-NC-4.0
- This wrapper code: Apache-2.0

## Acknowledgments

- Original Depth Anything V2 repository: https://github.com/DepthAnything/Depth-Anything-V2
- DINOv2 team for the backbone model
- Hugging Face for model hosting

## Support

For issues with:
- This wrapper: Check the code in `depth_anything.py`
- Original model: See https://github.com/DepthAnything/Depth-Anything-V2
- Model downloads: Check https://huggingface.co/depth-anything
