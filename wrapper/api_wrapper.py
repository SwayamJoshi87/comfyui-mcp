#!/usr/bin/env python3
"""
ComfyUI OpenAI-Compatible Auto API - starts on demand, shuts down after idle.
USAGE: python3 api_wrapper.py [--port 8000] [--comfy-port 8188] [--idle-timeout 300] [--models-dir ...]
"""

import argparse
import asyncio
import base64
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Literal

import httpx
import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

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
API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_ID = os.environ.get("MODEL_ID", "comfyui-sd1-5")

# --- App ---
app = FastAPI(title="ComfyUI OpenAI-Compatible Auto API")
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


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Text prompt")
    model: str | None = Field(default=MODEL_ID, description="Model id (currently informational)")
    n: int = Field(default=1, ge=1, le=4, description="Number of images to generate")
    quality: str | None = Field(default="standard", description="OpenAI quality hint (ignored)")
    response_format: Literal["url", "b64_json"] = Field(default="b64_json")
    size: str = Field(default="512x512", description="WIDTHxHEIGHT, e.g. 512x512")
    style: str | None = Field(default=None, description="OpenAI style hint (ignored)")
    user: str | None = Field(default=None, description="OpenAI user tracking (ignored)")
    # ComfyUI-specific overrides
    negative_prompt: str | None = Field(default=None)
    width: int | None = Field(default=None, ge=256, le=2048)
    height: int | None = Field(default=None, ge=256, le=2048)
    steps: int | None = Field(default=None, ge=1, le=150)
    cfg: float | None = Field(default=None, ge=1.0, le=30.0)
    seed: int | None = Field(default=None)

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: str) -> str:
        try:
            width, height = (int(x) for x in value.lower().split("x"))
        except ValueError as exc:
            raise ValueError('size must be "WIDTHxHEIGHT", e.g. "512x512"') from exc
        if width < 256 or width > 2048 or height < 256 or height > 2048:
            raise ValueError("size dimensions must be between 256 and 2048")
        return value


def _openai_error(
    message: str,
    type_: str = "invalid_request_error",
    code: str | None = None,
) -> dict:
    return {"error": {"message": message, "type": type_, "param": None, "code": code}}


def _error_response(
    message: str,
    status_code: int,
    type_: str = "invalid_request_error",
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=_openai_error(message, type_, code))


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return _error_response(str(detail), exc.status_code, type_="api_error")


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    messages = []
    for error in exc.errors():
        loc = ".".join(str(x) for x in error.get("loc", []))
        messages.append(f"{loc}: {error.get('msg', 'invalid value')}")
    return _error_response("; ".join(messages), 422, type_="invalid_request_error")


async def _check_auth(authorization: str | None) -> JSONResponse | None:
    if not API_KEY:
        return None
    if not authorization:
        return _error_response(
            "Missing Authorization header", 401, type_="authentication_error"
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != API_KEY:
        return _error_response("Invalid API key", 401, type_="authentication_error")
    return None


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


async def _do_generate(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    batch_size: int,
) -> dict:
    """Core generation logic shared by /generate and /v1/images/generations."""
    global last_request_time
    last_request_time = time.time()

    # Start ComfyUI if not running (synchronous, but rare)
    loop = asyncio.get_running_loop()
    ok = await loop.run_in_executor(None, start)
    if not ok:
        raise StarletteHTTPException(503, _openai_error("ComfyUI failed to start", type_="comfyui_error"))

    wf = build_workflow(
        prompt,
        negative_prompt,
        width,
        height,
        steps,
        cfg,
        seed,
        batch_size,
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{BASE_URL}/prompt", json={"prompt": wf})
            if r.status_code != 200:
                raise StarletteHTTPException(502, _openai_error(f"ComfyUI error: {r.text}", type_="comfyui_error"))
            prompt_id = r.json().get("prompt_id")
    except httpx.RequestError as e:
        raise StarletteHTTPException(502, _openai_error(f"Connection error: {e}", type_="connection_error"))

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
        raise StarletteHTTPException(504, _openai_error("Generation timed out (300s)", type_="timeout_error"))

    return {
        "status": "ok",
        "image": img_b64_list[0],
        "images": img_b64_list,
        "format": "png",
        "seed": wf["3"]["inputs"]["seed"],
        "prompt": prompt,
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    return await _do_generate(
        req.prompt,
        req.negative_prompt,
        req.width,
        req.height,
        req.steps,
        req.cfg,
        req.seed,
        req.batch_size,
    )


@app.get("/v1/models", response_model=None)
async def list_models(authorization: str | None = Header(None)):
    if auth_error := await _check_auth(authorization):
        return auth_error
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "comfyui",
                "permission": [],
                "root": MODEL_ID,
                "parent": None,
            }
        ],
    }


@app.post("/v1/images/generations", response_model=None)
async def create_image_generation(
    req: ImageGenerationRequest,
    authorization: str | None = Header(None),
):
    if auth_error := await _check_auth(authorization):
        return auth_error

    size_w, size_h = (int(x) for x in req.size.lower().split("x"))
    width = req.width if req.width is not None else size_w
    height = req.height if req.height is not None else size_h

    result = await _do_generate(
        req.prompt,
        req.negative_prompt or "ugly, blurry, low quality, deformed",
        width,
        height,
        req.steps if req.steps is not None else 20,
        req.cfg if req.cfg is not None else 7.0,
        req.seed if req.seed is not None else -1,
        req.n,
    )

    created = int(time.time())
    data: list[dict] = []
    for image_b64 in result["images"]:
        item: dict = {"revised_prompt": req.prompt}
        if req.response_format == "url":
            item["url"] = f"data:image/png;base64,{image_b64}"
        else:
            item["b64_json"] = image_b64
        data.append(item)

    return {"created": created, "data": data}


@app.get("/health", response_model=None)
async def health(authorization: str | None = Header(None)):
    if auth_error := await _check_auth(authorization):
        return auth_error
    alive = is_process_alive()
    return {"status": "ok", "comfyui_running": alive, "idle_timeout": IDLE_TIMEOUT}


@app.get("/status", response_model=None)
async def status_ep(authorization: str | None = Header(None)):
    if auth_error := await _check_auth(authorization):
        return auth_error
    alive = is_process_alive()
    idle = int(time.time() - last_request_time) if last_request_time else None
    return {"comfyui_running": alive, "last_request_s": idle, "idle_timeout": IDLE_TIMEOUT}


@app.post("/stop", response_model=None)
async def stop_ep(authorization: str | None = Header(None)):
    if auth_error := await _check_auth(authorization):
        return auth_error
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
