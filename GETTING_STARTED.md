# 🚀 Getting Started

Welcome to the **Depth Estimation Comparison App**!

## 📦 Repository

**GitHub**: https://github.com/rishabhshah13/depth-estimation

## ⚡ Quick Start (3 Steps)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rishabhshah13/depth-estimation.git
cd depth-estimation
```

### 2️⃣ Install Everything
```bash
./install.sh
```

This will:
- ✅ Create Python virtual environment
- ✅ Install all dependencies (PyTorch, Gradio, etc.)
- ✅ Download Depth Anything V2 model (~95MB)
- ✅ Run system tests

**Note**: MiDaS and Marigold weights download automatically on first use.

### 3️⃣ Run the App
```bash
./run.sh
```

That's it! The web GUI will open at **http://localhost:7860**

## 🎯 What You Can Do

### Web GUI (Recommended)
```bash
./run.sh
```

Features:
- 🖼️ Drag & drop image upload
- ⚡ Real-time depth estimation
- 🔄 Compare all 3 models side-by-side
- ⚙️ Adjustable model settings
- 📊 Timing information

### Command Line
```bash
./run.sh --cli --input image.jpg --output comparison.png
```

Options:
```bash
# Basic comparison
./run.sh --cli --input photo.jpg

# Save individual depth maps
./run.sh --cli --input photo.jpg --output-dir results/ --save-individual

# High quality with large models
./run.sh --cli --input photo.jpg --midas-model dpt_large_384 --da-model large

# High DPI output
./run.sh --cli --input photo.jpg --dpi 300
```

### Python API
```python
from depth_comparison_app import DepthComparisonApp

app = DepthComparisonApp()
app.process_image('image.jpg', output_path='result.png')
```

## 📚 The Three Models

| Model | Speed | Quality | When to Use |
|-------|-------|---------|-------------|
| **MiDaS** | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | Real-time, batch processing |
| **Depth Anything V2** | ⚡⚡ Medium | ⭐⭐⭐⭐ Excellent | Best all-around choice |
| **Marigold** | ⚡ Slow | ⭐⭐⭐⭐⭐ Outstanding | Highest quality needed |

## 📖 Documentation

- **README.md** - Complete documentation
- **QUICKSTART.md** - 5-minute quick start
- **SETUP_COMPLETE.md** - Detailed setup info
- **install.sh** - Installation script
- **run.sh** - Run script

## 🛠️ Scripts Reference

### install.sh
Sets up the entire environment:
```bash
./install.sh
```

What it does:
1. Creates virtual environment with `uv`
2. Installs all Python dependencies
3. Verifies repositories are present
4. Downloads Depth Anything V2 checkpoint
5. Runs system tests

### run.sh
Runs the application:
```bash
./run.sh              # Web GUI (default)
./run.sh --cli [...]  # Command-line mode
./run.sh --help       # Show help
```

## 🔧 Manual Setup (if needed)

If you don't want to use the scripts:

```bash
# Create environment
uv venv
.venv/bin/pip install -r requirements.txt

# Download Depth Anything V2 checkpoint
mkdir -p Depth-Anything-V2/checkpoints
cd Depth-Anything-V2/checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
cd ../..

# Run
.venv/bin/python app_gui.py              # Web GUI
.venv/bin/python depth_comparison_app.py --input image.jpg  # CLI
```

## 🎨 Understanding Depth Maps

The output shows depth with different colormaps:

- **MiDaS**: Viridis (purple=far → yellow=close)
- **Depth Anything V2**: Spectral (blue=far → red=close)
- **Marigold**: Inferno (dark=far → bright=close)

**Brighter/Warmer colors** = Objects closer to camera
**Darker/Cooler colors** = Objects farther from camera

## 💡 Tips

### For Speed
- Use MiDaS small model: `--midas-model midas_v21_small_256`
- Run only one model at a time (uncheck others in GUI)
- Use CPU if GPU causes issues: `--no-gpu`

### For Quality
- Use larger models: `--midas-model dpt_large_384 --da-model large`
- Run Marigold (takes ~15-20 seconds but best quality)
- Increase output resolution: `--dpi 300`

### For Batch Processing
```bash
for img in images/*.jpg; do
    ./run.sh --cli --input "$img" --output "results/$(basename $img)"
done
```

## 🆘 Troubleshooting

### "uv not found"
Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### "CUDA out of memory"
Use smaller models or CPU:
```bash
./run.sh --cli --input image.jpg --no-gpu
```

### Checkpoint not found
Re-run installation:
```bash
./install.sh
```

### Test the installation
```bash
.venv/bin/python test_app.py
```

## 📊 Example Output

The comparison image shows:
1. **Original Image** (top-left)
2. **MiDaS Depth Map** (top-right)
3. **Depth Anything V2 Depth Map** (bottom-left)
4. **Marigold Depth Map** (bottom-right)

Each includes timing info and uses a different colormap for easy comparison.

## 🔗 Links

- **GitHub**: https://github.com/rishabhshah13/depth-estimation
- **MiDaS**: https://github.com/isl-org/MiDaS
- **Depth Anything V2**: https://github.com/DepthAnything/Depth-Anything-V2
- **Marigold**: https://github.com/prs-eth/Marigold

## 🎓 Citation

If you use this tool in your research, please cite the original papers:

**MiDaS**:
```bibtex
@article{Ranftl2022,
  title={Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
  author={Ranftl, Ren{\'e} and Bochkovskiy, Alexey and Koltun, Vladlen},
  journal={TPAMI},
  year={2022}
}
```

**Depth Anything V2**:
```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={NeurIPS},
  year={2024}
}
```

**Marigold**:
```bibtex
@inproceedings{ke2024marigold,
  title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
  author={Ke, Bingxin and Obukhov, Anton and Huang, Shengyu and Metzger, Nando and Daudt, Rodrigo Caye and Schindler, Konrad},
  booktitle={CVPR},
  year={2024}
}
```

---

**Ready to explore depth estimation? Start with `./run.sh`! 🎉**
