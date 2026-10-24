#!/usr/bin/env bash
# ==============================================================================
# run_servers.sh - Launch SGLang Inference Engines for PS 2
# Dual-Port / Dual-Machine Deployment:
#   Port 30000: Un-Tuned Baseline Model (sarthi RTX 3050)
#   Port 30001: Aligned Pro + XGrammar + RadixAttention (LAPTOP-1199U9O4 RTX 5060)
# ==============================================================================
set -euo pipefail

MODE="${1:-aligned}"

if [ "$MODE" == "untuned" ]; then
    echo "================================================================================"
    echo " [SGLang] Launching Un-Tuned Baseline Server on Port 30000"
    echo "================================================================================"
    python3 -m sglang.launch_server \
        --model-path unsloth/gemma-3-1b-it \
        --served-model-name local-sglang-curator \
        --host 0.0.0.0 \
        --port 30000 \
        --dtype bfloat16 \
        --mem-fraction-static 0.70

elif [ "$MODE" == "aligned" ]; then
    echo "================================================================================"
    echo " [SGLang] Launching Aligned Pro Server on Port 30001 (RTX 5060 sm_120)"
    echo " Features: XGrammar Schema Jump-Decoding + RadixAttention KV Cache"
    echo "================================================================================"
    MODEL_DIR="checkpoints/gemma_sglang_merged"
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Notice: $MODEL_DIR not found. Serving base model with XGrammar backend..."
        MODEL_DIR="unsloth/gemma-3-1b-it"
    fi

    python3 -m sglang.launch_server \
        --model-path "$MODEL_DIR" \
        --served-model-name gemma3-1b-aligned-pro \
        --host 0.0.0.0 \
        --port 30001 \
        --dtype bfloat16 \
        --grammar-backend xgrammar \
        --mem-fraction-static 0.80

else
    echo "Usage: bash run_servers.sh [untuned | aligned]"
    exit 1
fi
