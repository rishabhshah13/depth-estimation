#!/bin/bash
# Depth Estimation Comparison App - Run Script
# ==============================================
# This script runs the depth estimation app

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Please run ./install.sh first"
    exit 1
fi

# Parse arguments
MODE="gui"
CLI_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cli|--command-line)
            MODE="cli"
            shift
            ;;
        --help|-h)
            echo "Depth Estimation Comparison App - Run Script"
            echo ""
            echo "Usage:"
            echo "  ./run.sh                  # Start web GUI (default)"
            echo "  ./run.sh --cli [args]     # Use command-line mode"
            echo ""
            echo "Examples:"
            echo "  ./run.sh"
            echo "  ./run.sh --cli --input image.jpg"
            echo "  ./run.sh --cli --input image.jpg --output result.png"
            echo ""
            echo "For CLI help:"
            echo "  ./run.sh --cli --help"
            exit 0
            ;;
        *)
            # Collect remaining arguments for CLI mode
            CLI_ARGS="$CLI_ARGS $1"
            shift
            ;;
    esac
done

# Display banner
echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}     Depth Estimation Comparison App${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

if [ "$MODE" = "gui" ]; then
    # Web GUI Mode
    echo -e "${BLUE}Starting Web GUI...${NC}"
    echo ""
    echo "The web interface will open at: ${GREEN}http://localhost:7860${NC}"
    echo ""
    echo "Features:"
    echo "  • Upload images via drag & drop"
    echo "  • Compare 3 depth estimation models"
    echo "  • Real-time results with timing info"
    echo "  • Adjustable model settings"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo ""

    # Run the GUI
    .venv/bin/python app_gui.py

else
    # CLI Mode
    echo -e "${BLUE}Running in Command-Line Mode...${NC}"
    echo ""

    if [ -z "$CLI_ARGS" ]; then
        echo "Usage: ./run.sh --cli --input IMAGE [OPTIONS]"
        echo ""
        echo "Run './run.sh --cli --help' for more options"
        exit 1
    fi

    # Run the CLI
    .venv/bin/python depth_comparison_app.py $CLI_ARGS
fi
