#!/bin/bash
set -e

MODEL="/app/comfyui/models/checkpoints/v1-5-pruned-emaonly.safetensors"
if [ ! -f "$MODEL" ]; then
    echo "[entrypoint] Downloading SD1.5 model..."
    wget -q --show-progress \
        -O "$MODEL" \
        "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
    echo "[entrypoint] Model downloaded"
fi

echo "[entrypoint] Starting API server..."
exec python3 /app/api_wrapper.py --port "${API_PORT:-8000}" --comfy-port "${COMFYUI_PORT:-8188}" --idle-timeout "${IDLE_TIMEOUT:-300}"
