#!/bin/bash
# Vast.ai Setup Script for Mneme Training
# Run this after SSH into your Vast.ai instance

set -e

echo "=========================================="
echo "  Mneme Training Setup for Vast.ai"
echo "  RTX 6000 Pro (48GB VRAM)"
echo "=========================================="

# Update system
apt-get update -qq

# Install Python dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40.0 accelerate safetensors sentencepiece tqdm numpy

# Optional: Install Flash Attention for 2x speedup
echo "Installing Flash Attention (optional but recommended)..."
pip install flash-attn --no-build-isolation || echo "Flash Attention install failed - will use SDPA instead"

# Verify GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo ""
echo "PyTorch GPU Check:"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "  Run: python train_vastai.py"
echo "=========================================="
