# ⚡ Quick Start Guide

Get up and running with the Depth Estimation Comparison App in 5 minutes!

## 🎯 Step 1: Install Dependencies (2 min)

```bash
pip install -r requirements.txt
```

## 📥 Step 2: Download Model Weights (2 min)

### Depth Anything V2 (Required)

```bash
cd models/Depth-Anything-V2/checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
cd ../../..
```

### MiDaS & Marigold (Automatic)
- MiDaS downloads automatically on first use
- Marigold downloads automatically on first use (~5GB, may take a few minutes)

## 🚀 Step 3: Run the App (1 min)

### Option A: Web GUI (Easiest!)

```bash
python src/app_gui.py
```

1. Browser opens automatically
2. Upload an image
3. Click "Estimate Depth"
4. See results from all three models!

### Option B: Command Line

```bash
# Use example image
python src/depth_comparison_app.py \
    --input models/Depth-Anything-V2/assets/examples/demo01.jpg \
    --output my_first_comparison.png

# Or use your own image
python src/depth_comparison_app.py --input your_image.jpg
```

## ✅ That's It!

You should now see:
- Original image + 3 depth maps side-by-side
- Timing information for each model
- Colorized depth visualizations

## 🎨 What Do the Colors Mean?

- **Bright/Warm colors (yellow, red, white)** = Close to camera
- **Dark/Cool colors (blue, purple, black)** = Far from camera

## 🔥 Pro Tips

1. **First time running?** Models will download automatically (5-10 GB total)
2. **Running slow?** Use `--no-gpu` flag if you don't have a GPU
3. **Want individual outputs?** Add `--save-individual --output-dir results/`
4. **Better quality?** Use `--midas-model dpt_large_384 --da-model large`

## ❓ Having Issues?

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "Checkpoint not found" error
```bash
cd models/Depth-Anything-V2/checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

### Still stuck?
Check the full [README.md](README.md) for detailed troubleshooting.

## 📖 Next Steps

- Try different images!
- Experiment with different model sizes
- Use individual models in your own code
- Read the full README for advanced features

**Enjoy! 🎉**
