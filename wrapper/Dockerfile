FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

LABEL description="ComfyUI with auto-shutdown API wrapper (FastAPI)"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone ComfyUI
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /app/comfyui

# Install ComfyUI Python deps
RUN pip install --no-cache-dir -r /app/comfyui/requirements.txt

# API wrapper deps
RUN pip install --no-cache-dir fastapi uvicorn pillow httpx pydantic

# Copy SD1.5 model (must exist next to Dockerfile at build time)
COPY v1-5-pruned-emaonly.safetensors /app/comfyui/models/checkpoints/

# Copy scripts
COPY api_wrapper.py /app/api_wrapper.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8188 8000

ENV IDLE_TIMEOUT=300
ENV COMFYUI_PORT=8188
ENV API_PORT=8000

ENTRYPOINT ["/app/entrypoint.sh"]
