#!/usr/bin/env python3
"""OpenAI-compatible image generation API backed by the ComfyUI cold-start wrapper.

This service does not start ComfyUI itself. It forwards generation requests to the
wrapper (`wrapper/api_wrapper.py`), which owns the cold-start / idle-shutdown
lifecycle and keeps GPU memory free when nothing is generating.

Environment variables:
    COMFYUI_WRAPPER_URL - base URL of the wrapper (default: http://127.0.0.1:8002)
    API_HOST            - host to bind on (default: 0.0.0.0)
    API_PORT            - port to bind on (default: 8000)
    OPENAI_API_KEY      - optional Bearer token required on all endpoints
    MODEL_ID            - model id advertised by /v1/models (default: comfyui-sd1-5)
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Literal

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

WRAPPER_URL = os.environ.get("COMFYUI_WRAPPER_URL", "http://127.0.0.1:8002").rstrip("/")
API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_ID = os.environ.get("MODEL_ID", "comfyui-sd1-5")

app = FastAPI(title="ComfyUI OpenAI-Compatible Images API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Text prompt")
    model: str | None = Field(default=MODEL_ID, description="Model id (currently informational)")
    n: int = Field(default=1, ge=1, le=4, description="Number of images to generate")
    quality: str | None = Field(default="standard", description="OpenAI quality hint (ignored)")
    response_format: Literal["url", "b64_json"] = Field(default="b64_json")
    size: str = Field(default="512x512", description="WIDTHxHEIGHT, e.g. 512x512")
    style: str | None = Field(default=None, description="OpenAI style hint (ignored)")
    user: str | None = Field(default=None, description="OpenAI user tracking (ignored)")
    # ComfyUI-specific overrides, kept compatible with the wrapper
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


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content=_openai_error(str(exc), type_="internal_error"),
    )


async def _check_auth(authorization: str | None) -> JSONResponse | None:
    if not API_KEY:
        return None
    if not authorization:
        return _error_response(
            "Missing Authorization header",
            401,
            type_="authentication_error",
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != API_KEY:
        return _error_response("Invalid API key", 401, type_="authentication_error")
    return None


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

    # Parse size; explicit width/height override the size string.
    size_w, size_h = (int(x) for x in req.size.lower().split("x"))
    width = req.width if req.width is not None else size_w
    height = req.height if req.height is not None else size_h

    payload = {
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt or "ugly, blurry, low quality, deformed",
        "width": width,
        "height": height,
        "steps": req.steps if req.steps is not None else 20,
        "cfg": req.cfg if req.cfg is not None else 7.0,
        "seed": req.seed if req.seed is not None else -1,
        "batch_size": req.n,
    }

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=360.0, write=10.0, pool=10.0)
        ) as client:
            response = await client.post(f"{WRAPPER_URL}/generate", json=payload)
            response.raise_for_status()
            wrapper_resp = response.json()
    except httpx.HTTPStatusError as exc:
        return _error_response(
            f"ComfyUI wrapper error: {exc.response.text}",
            exc.response.status_code,
            type_="comfyui_error",
        )
    except httpx.TimeoutException:
        return _error_response(
            "Generation timed out during cold start or inference",
            504,
            type_="timeout_error",
        )
    except Exception as exc:
        traceback.print_exc()
        return _error_response(
            f"Wrapper connection failed: {exc}",
            502,
            type_="connection_error",
        )

    images = wrapper_resp.get("images") or []
    if not images:
        # Backward compatibility with older wrappers that only returned "image".
        single = wrapper_resp.get("image")
        if single:
            images = [single]

    if not images:
        return _error_response(
            "No image returned from ComfyUI wrapper",
            502,
            type_="comfyui_error",
        )

    created = int(time.time())
    data: list[dict] = []
    for image_b64 in images:
        item: dict = {"revised_prompt": req.prompt}
        if req.response_format == "url":
            # No persistent file hosting yet; return a data URL.
            item["url"] = f"data:image/png;base64,{image_b64}"
        else:
            item["b64_json"] = image_b64
        data.append(item)

    return {"created": created, "data": data}


@app.get("/health", response_model=None)
async def health(authorization: str | None = Header(None)):
    if auth_error := await _check_auth(authorization):
        return auth_error
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{WRAPPER_URL}/health")
            response.raise_for_status()
            return {"status": "ok", "wrapper": response.json()}
    except Exception as exc:
        return _error_response(
            f"Wrapper unreachable: {exc}",
            503,
            type_="connection_error",
        )


@app.get("/status", response_model=None)
async def status(authorization: str | None = Header(None)):
    if auth_error := await _check_auth(authorization):
        return auth_error
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{WRAPPER_URL}/status")
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        return _error_response(
            f"Wrapper unreachable: {exc}",
            503,
            type_="connection_error",
        )


def main() -> None:
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
