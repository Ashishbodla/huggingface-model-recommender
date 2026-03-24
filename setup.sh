#!/bin/bash
# setup.sh — One-command setup for HuggingFace Model Recommender
set -e

echo "=============================="
echo " HF Model Recommender Setup"
echo "=============================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.8+ first."
    exit 1
fi

PYTHON=$(command -v python3)
echo "Using Python: $PYTHON ($($PYTHON --version))"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate
source venv/bin/activate
echo "Activated virtual environment."
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Core dependencies installed."

# Optional: try installing torch
echo ""
echo "Installing PyTorch (for GPU/MPS detection)..."
pip install torch -q 2>/dev/null && echo "PyTorch installed." || echo "PyTorch install failed (optional — tool will default to CPU mode)."

echo ""
echo "=============================="
echo " Setup complete!"
echo "=============================="
echo ""
echo "To run the tool:"
echo "  source venv/bin/activate"
echo "  python recommender.py"
echo ""
echo "Or with filters:"
echo "  python recommender.py --quant gguf --top 20"
echo ""
echo "See README.md for full instructions."
