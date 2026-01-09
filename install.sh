#!/bin/bash
# Depth Estimation Comparison App - Installation Script
# ======================================================
# This script sets up the environment and downloads all necessary models

set -e  # Exit on error

echo "============================================================"
echo "  Depth Estimation Comparison App - Installation"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed${NC}"
    echo "Please install uv first: https://github.com/astral-sh/uv"
    echo "Or use: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo -e "${BLUE}[1/5] Creating Python virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo "  Virtual environment already exists, skipping..."
else
    uv venv
    echo -e "${GREEN}  ✓ Virtual environment created${NC}"
fi

echo ""
echo -e "${BLUE}[2/5] Installing Python dependencies...${NC}"
echo "  This may take a few minutes..."
uv pip install -r requirements.txt
echo -e "${GREEN}  ✓ Dependencies installed${NC}"

echo ""
echo -e "${BLUE}[3/5] Checking repositories...${NC}"
if [ ! -d "models/MiDaS" ]; then
    echo -e "${YELLOW}  MiDaS repository not found in models/${NC}"
    exit 1
fi
if [ ! -d "models/Depth-Anything-V2" ]; then
    echo -e "${YELLOW}  Depth-Anything-V2 repository not found in models/${NC}"
    exit 1
fi
if [ ! -d "models/Marigold" ]; then
    echo -e "${YELLOW}  Marigold repository not found in models/${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ All repositories present${NC}"

echo ""
echo -e "${BLUE}[4/5] Downloading model weights...${NC}"

# Create checkpoints directory
mkdir -p models/Depth-Anything-V2/checkpoints

# Check if Depth Anything V2 checkpoint exists
CHECKPOINT_FILE="models/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "  ✓ Depth Anything V2 checkpoint already exists"
else
    echo "  Downloading Depth Anything V2 Small model (~95MB)..."
    cd models/Depth-Anything-V2/checkpoints

    # Try wget first, fall back to curl
    if command -v wget &> /dev/null; then
        wget -q --show-progress https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
    elif command -v curl &> /dev/null; then
        curl -L -o depth_anything_v2_vits.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
    else
        echo -e "${RED}  Error: Neither wget nor curl is available${NC}"
        exit 1
    fi

    cd ../../..

    if [ -f "$CHECKPOINT_FILE" ]; then
        echo -e "${GREEN}  ✓ Depth Anything V2 checkpoint downloaded${NC}"
    else
        echo -e "${RED}  ✗ Failed to download checkpoint${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}  ✓ MiDaS weights will download automatically on first use${NC}"
echo -e "${GREEN}  ✓ Marigold weights will download automatically on first use (~5GB)${NC}"

echo ""
echo -e "${BLUE}[5/5] Running system tests...${NC}"
.venv/bin/python tests/test_app.py > /tmp/test_output.txt 2>&1 || true

# Check if critical tests passed
if grep -q "All critical checks passed" /tmp/test_output.txt; then
    echo -e "${GREEN}  ✓ System tests passed${NC}"
else
    echo -e "${YELLOW}  ⚠ Some tests failed, but you can still try running the app${NC}"
    echo "  Run 'python tests/test_app.py' for details"
fi

echo ""
echo "============================================================"
echo -e "${GREEN}✓ Installation Complete!${NC}"
echo "============================================================"
echo ""
echo "To run the application:"
echo ""
echo "  ${BLUE}./run.sh${NC}              # Start web GUI"
echo "  ${BLUE}./run.sh --cli${NC}        # Use command-line interface"
echo ""
echo "Or manually:"
echo "  ${BLUE}.venv/bin/python -m src.app_gui${NC}"
echo ""
echo "Documentation:"
echo "  docs/README.md           - Full documentation"
echo "  docs/QUICKSTART.md       - Quick start guide"
echo "  docs/SETUP_COMPLETE.md   - Setup summary"
echo ""
echo "============================================================"
