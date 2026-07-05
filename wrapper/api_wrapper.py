#!/usr/bin/env python3
"""
ComfyUI Auto API - starts on demand, shuts down after idle.
USAGE: python3 api_wrapper.py [--port 8000] [--comfy-port 8188] [--idle-timeout 300] [--models-dir ...]
"""

import argparse
import asyncio
import base64
import json
import os
import signal
import subprocess
import sys
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Parse args ---
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--comfy-port", type=int, default=8188)
parser.add_argument("--idle-timeout", type=int, default=300)
parser.add_argument("--models-dir", default="/app/comfyui")
args = parser.parse_args()

COMFYUI_DIR = os.path.abspath(args.models_dir)
COMFYUI_PORT = args.comfy_port
API_PORT = args.port
IDLE_TIMEOUT = args.idle_timeout
BASE_URL = f"http://127.0.0.1:{COMFYUI_PORT}"

# --- App ---
app = FastAPI(title="ComfyUI Auto API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

comfyui_process: subprocess.Popen | None = None
last_request_time: float = 0.0
state_lock = threading.Lock()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    negative_prompt: str = Field("ugly, blurry, low quality, deformed", description="Negative prompt")
    width: int = Field(512, ge=256, le=2048)
    height: int = Field(512, ge=256, le=2048)
    steps: int = Field(20, ge=1, le=150)
    cfg: float = Field(7.0, ge=1.0, le=30.0)
    seed: int = Field(-1)
    batch_size: int = Field(1, ge=1, le=4)


def is_process_alive() -> bool:
    """Fast check - does the subprocess PID exist?"""
    if comfyui_process is None:
        return False
    return comfyui_process.poll() is None


async def is_responding() -> bool:
    """Check if ComfyUI HTTP server is responding (async, non-blocking)."""
    if not is_process_alive():
        return False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{BASE_URL}/system_stats")
            return r.status_code == 200
    except Exception:
        return False


def start() -> bool:
    global comfyui_process
    with state_lock:
        if is_process_alive():
            return True

        # Kill any zombie process
        if comfyui_process:
            try:
                comfyui_process.terminate()
                comfyui_process.wait(timeout=10)
            except Exception:
                try:
                    comfyui_process.kill()
                    comfyui_process.wait()
                except Exception:
                    pass
            comfyui_process = None

        print(f"[API] Starting ComfyUI from {COMFYUI_DIR}...")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        comfyui_process = subprocess.Popen(
            [sys.executable, "main.py", f"--port={COMFYUI_PORT}", "--listen=127.0.0.1"],
            cwd=COMFYUI_DIR,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Poll for readiness - up to 180s (first model load + CUDA JIT)
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                r = __import__("urllib.request").request.urlopen(
                    f"{BASE_URL}/system_stats", timeout=3
                )
                if r.status == 200:
                    print("[API] ComfyUI ready")
                    return True
            except Exception:
                pass
            time.sleep(2)

        print("[API] FAILED to start ComfyUI after 180s")
        if comfyui_process and comfyui_process.poll() is not None:
            print(f"[API] Process exited with code {comfyui_process.returncode}")
        return False


def stop() -> None:
    global comfyui_process
    with state_lock:
        if not is_process_alive():
            comfyui_process = None
            return
        print("[API] Stopping ComfyUI (idle)...")
        comfyui_process.terminate()
        try:
            comfyui_process.wait(timeout=10)
        except Exception:
            try:
                comfyui_process.kill()
                comfyui_process.wait()
            except Exception:
                pass
        comfyui_process = None


def build_workflow(prompt, neg, w, h, steps, cfg, seed, bs):
    ckpt = "v1-5-pruned-emaonly.safetensors"
    actual_seed = seed if seed != -1 else int(time.time() * 1000) % 2**32
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": actual_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": bs}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "api_", "images": ["8", 0]}},
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    global last_request_time
    last_request_time = time.time()

    # Start ComfyUI if not running (synchronous, but rare)
    loop = asyncio.get_running_loop()
    ok = await loop.run_in_executor(None, start)
    if not ok:
        raise HTTPException(503, "ComfyUI failed to start")

    wf = build_workflow(
        req.prompt,
        req.negative_prompt,
        req.width,
        req.height,
        req.steps,
        req.cfg,
        req.seed,
        req.batch_size,
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{BASE_URL}/prompt", json={"prompt": wf})
            if r.status_code != 200:
                raise HTTPException(502, f"ComfyUI error: {r.text}")
            prompt_id = r.json().get("prompt_id")
    except httpx.RequestError as e:
        raise HTTPException(502, f"Connection error: {e}")

    img_b64_list = []
    last_idle_refresh = 0
    async with httpx.AsyncClient(timeout=10.0) as client:
        for i in range(300):  # up to 300s
            now = time.time()
            if now - last_idle_refresh > 15:
                last_request_time = now
                last_idle_refresh = now

            try:
                r = await client.get(f"{BASE_URL}/history/{prompt_id}")
                if r.status_code == 200 and prompt_id in r.json():
                    for node_id, out in r.json()[prompt_id].get("outputs", {}).items():
                        for img in out.get("images", []):
                            p = os.path.join(COMFYUI_DIR, "output", img["filename"])
                            if os.path.exists(p):
                                with open(p, "rb") as f:
                                    img_b64_list.append(base64.b64encode(f.read()).decode())
                                try:
                                    os.remove(p)
                                except Exception:
                                    pass
                    if img_b64_list:
                        break
            except Exception:
                pass
            await asyncio.sleep(1)

    if not img_b64_list:
        raise HTTPException(504, "Generation timed out (300s)")
    return {
        "status": "ok",
        "image": img_b64_list[0],
        "images": img_b64_list,
        "format": "png",
        "seed": wf["3"]["inputs"]["seed"],
        "prompt": req.prompt,
    }


@app.get("/health")
async def health():
    alive = is_process_alive()
    return {"status": "ok", "comfyui_running": alive, "idle_timeout": IDLE_TIMEOUT}


@app.get("/status")
async def status_ep():
    alive = is_process_alive()
    idle = int(time.time() - last_request_time) if last_request_time else None
    return {"comfyui_running": alive, "last_request_s": idle, "idle_timeout": IDLE_TIMEOUT}


@app.post("/stop")
async def stop_ep():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, stop)
    return {"status": "ok", "comfyui_running": False}


def idle_loop():
    global comfyui_process, last_request_time
    while True:
        time.sleep(30)
        with state_lock:
            if comfyui_process is None:
                continue
            if comfyui_process.poll() is not None:
                print("[API] ComfyUI exited on its own")
                comfyui_process = None
                continue
        if time.time() - last_request_time > IDLE_TIMEOUT:
            stop()


if __name__ == "__main__":
    threading.Thread(target=idle_loop, daemon=True).start()
    print(f"[API] ComfyUI Wrapper on 0.0.0.0:{API_PORT}, ComfyUI on port {COMFYUI_PORT}")
    print(f"[API] Models: {COMFYUI_DIR}/models/checkpoints/")
    print(f"[API] Idle timeout: {IDLE_TIMEOUT}s")
    signal.signal(signal.SIGINT, lambda s, f: (stop(), sys.exit(0)))
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="info")
