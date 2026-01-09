#!/bin/bash
# Setup script for Depth Anything V2 models
# This script downloads the model checkpoints

set -e

echo "=========================================="
echo "Depth Anything V2 - Model Setup"
echo "=========================================="
echo ""

# Create checkpoints directory
CHECKPOINT_DIR="Depth-Anything-V2/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

echo "Checkpoint directory: $CHECKPOINT_DIR"
echo ""

# Function to download a model
download_model() {
    local model_name=$1
    local url=$2
    local filename=$3

    if [ -f "$CHECKPOINT_DIR/$filename" ]; then
        echo "[SKIP] $model_name already exists"
    else
        echo "[DOWNLOAD] $model_name..."
        echo "  URL: $url"
        echo "  Saving to: $CHECKPOINT_DIR/$filename"

        if command -v wget &> /dev/null; then
            wget -O "$CHECKPOINT_DIR/$filename" "$url"
        elif command -v curl &> /dev/null; then
            curl -L -o "$CHECKPOINT_DIR/$filename" "$url"
        else
            echo "  ERROR: Neither wget nor curl is available"
            echo "  Please manually download from: $url"
            echo "  Save to: $CHECKPOINT_DIR/$filename"
            return 1
        fi

        echo "  [OK] Download complete"
    fi
    echo ""
}

# Ask which models to download
echo "Which model(s) would you like to download?"
echo ""
echo "1. Small (24.8M params, Apache-2.0) - Recommended for quick inference"
echo "2. Base (97.5M params, CC-BY-NC-4.0) - Balanced quality/speed"
echo "3. Large (335.3M params, CC-BY-NC-4.0) - Best quality"
echo "4. All models"
echo "5. Skip download (I'll download manually)"
echo ""
read -p "Enter choice (1-5): " choice

case $choice in
    1)
        download_model \
            "Small Model" \
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth" \
            "depth_anything_v2_vits.pth"
        ;;
    2)
        download_model \
            "Base Model" \
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth" \
            "depth_anything_v2_vitb.pth"
        ;;
    3)
        download_model \
            "Large Model" \
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth" \
            "depth_anything_v2_vitl.pth"
        ;;
    4)
        download_model \
            "Small Model" \
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth" \
            "depth_anything_v2_vits.pth"

        download_model \
            "Base Model" \
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth" \
            "depth_anything_v2_vitb.pth"

        download_model \
            "Large Model" \
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth" \
            "depth_anything_v2_vitl.pth"
        ;;
    5)
        echo "Skipping download. You can manually download models from:"
        echo ""
        echo "Small:  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
        echo "Base:   https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth"
        echo "Large:  https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
        echo ""
        echo "Save them to: $CHECKPOINT_DIR/"
        echo ""
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Check what's downloaded
echo "=========================================="
echo "Current checkpoints:"
echo "=========================================="
ls -lh "$CHECKPOINT_DIR" 2>/dev/null || echo "No checkpoints found"
echo ""

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "You can now use the depth estimation module:"
echo ""
echo "  # Command line:"
echo "  python depth_anything.py --image-path image.jpg --output-path depth.png"
echo ""
echo "  # Python API:"
echo "  from depth_anything import DepthEstimator"
echo "  estimator = DepthEstimator(model_size='small')"
echo "  depth = estimator.estimate_and_save('image.jpg', 'depth.png')"
echo ""
echo "See README_DEPTH_ESTIMATION.md for more examples."
echo ""
