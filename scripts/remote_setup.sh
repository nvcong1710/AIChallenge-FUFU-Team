#!/usr/bin/env bash
# Setup BetterDay-Tool env trên vast.ai instance.
set -e

WORK=/root/bd
mkdir -p "$WORK"
cd "$WORK"

echo "=== Create venv ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet --upgrade pip wheel

echo ""
echo "=== Install torch + CUDA 12.1 ==="
pip install --quiet --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0

echo ""
echo "=== Install core deps (skip paddleocr — Python 3.12 không support) ==="
pip install --quiet \
    transformers==4.46.3 \
    accelerate==1.0.1 \
    bitsandbytes==0.44.1 \
    sentencepiece==0.2.0 \
    faiss-cpu==1.9.0 \
    opencv-python-headless==4.10.0.84 \
    ffmpeg-python==0.2.0 \
    'scenedetect[opencv]==0.6.4' \
    Pillow==10.4.0 \
    'numpy<2' \
    PyYAML==6.0.2 \
    fastapi==0.115.5 \
    'uvicorn[standard]==0.32.0' \
    pydantic==2.9.2 \
    ultralytics==8.3.20 \
    gtts \
    huggingface_hub

echo ""
echo "=== Verify torch CUDA ==="
python -c "
import torch
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0))
print('vram:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')
"

echo ""
echo "=== Disk usage ==="
df -h /root | tail -1
