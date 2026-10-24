#!/usr/bin/env bash
# ==============================================================================
# 00_setup_env.sh - Environment Setup for Syntax Error 2026 Local LLM Track
# Hardware: RTX 5060 Laptop (8GB VRAM, sm_120 Blackwell) & RTX 3050 (6GB VRAM, sm_86)
# ==============================================================================
set -euo pipefail

echo "================================================================================"
echo " [Setup] Initializing Python Environment with CUDA 12.8 (RTX 5060 sm_120 Fix)"
echo "================================================================================"

# 1. Create or activate Python virtual environment
VENV_PATH="${HOME}/.venvs/local_llm"
if [ ! -d "$VENV_PATH" ]; then
    echo "[1/6] Creating virtual environment at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"

# 2. Upgrade pip and core packaging tools
echo "[2/6] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# 3. CRITICAL HARDWARE FIX: Install PyTorch >= 2.7 with CUDA 12.8 (cu128)
# RTX 5060 uses sm_120 (Blackwell architecture). cu124 lacks sm_120 kernel images and crashes!
echo "[3/6] Installing PyTorch with CUDA 12.8 for sm_120 (RTX 5060) and sm_86 (RTX 3050)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 || {
    echo "Warning: cu128 wheel fallback check..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 || \
    pip install torch torchvision torchaudio
}

# 4. Immediate GPU kernel verification
echo "[4/6] Verifying GPU and CUDA kernel execution..."
python -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:', torch.cuda.get_device_name(0))
    print('Device Capability:', torch.cuda.get_device_capability(0))
    x = torch.randn(4, 4, device='cuda')
    print('CUDA Kernel Test PASSED: Generated tensor on GPU:', x.shape)
else:
    print('WARNING: CUDA is not available. Please verify NVIDIA driver and WSL GPU passthrough.')
"

# 5. Install SGLang (Track Core Requirement: inference & serving engine)
echo "[5/6] Installing SGLang inference engine and xgrammar..."
pip install "sglang[all]" || pip install sglang xgrammar

# 6. Install Lexsi Labs Toolkits & Alignment Libraries
echo "[6/6] Installing Lexsi Labs alignment toolkits (CuratorKIT, AlignTune, AuditKIT)..."
pip install "curatorkit[all]" aligntune "auditkit[openai,litellm]" || {
    echo "Installing Lexsi Labs toolkits from GitHub..."
    pip install git+https://github.com/Lexsi-Labs/CuratorKIT.git || true
    pip install git+https://github.com/Lexsi-Labs/aligntune.git || true
    pip install git+https://github.com/Lexsi-Labs/AuditKIT.git || true
}

# 7. Install Training & Application dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || true
pip install trl peft accelerate bitsandbytes litellm datasets pydantic pandas rich streamlit openai

echo "================================================================================"
echo " [Setup Complete] Run: source $VENV_PATH/bin/activate"
echo "================================================================================"

