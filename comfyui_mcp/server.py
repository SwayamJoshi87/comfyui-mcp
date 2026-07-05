#!/usr/bin/env python3
"""ComfyUI MCP server.

A thin stdio MCP wrapper around the ComfyUI on-demand API wrapper running on
another host/container. It does not start ComfyUI itself; it asks the wrapper
to do that over HTTP, which handles GPU cold-start and idle shutdown.

Environment variables:
    COMFYUI_WRAPPER_URL - base URL of the wrapper (default: http://192.168.2.51:8002)
"""

from __future__ import annotations

import os
import traceback

import httpx
from mcp.server.fastmcp import FastMCP

WRAPPER_URL = os.environ.get("COMFYUI_WRAPPER_URL", "http://192.168.2.51:8002").rstrip("/")

mcp = FastMCP("comfyui-mcp")


def _ok(data: dict) -> dict:
    return {"ok": True, "data": data}


def _err(message: str, *, error_type: str = "internal", details: dict | None = None) -> dict:
    return {"ok": False, "error": {"type": error_type, "message": message, "details": details or {}}}


@mcp.tool()
async def comfyui_health() -> dict:
    """Quick health check of the wrapper process."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{WRAPPER_URL}/health")
            r.raise_for_status()
            return _ok(r.json())
    except Exception as e:
        traceback.print_exc()
        return _err(f"health check failed: {e}")


@mcp.tool()
async def comfyui_status() -> dict:
    """Check whether ComfyUI is running and how long it has been idle."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{WRAPPER_URL}/status")
            r.raise_for_status()
            return _ok(r.json())
    except Exception as e:
        traceback.print_exc()
        return _err(f"status check failed: {e}")


@mcp.tool()
async def comfyui_generate(
    prompt: str,
    negative_prompt: str = "ugly, blurry, low quality, deformed",
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = -1,
    batch_size: int = 1,
) -> dict:
    """Generate an image with ComfyUI. The wrapper starts the GPU process if needed.

    Returns a JSON object with the base64-encoded PNG image under data.image.
    """
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "batch_size": batch_size,
    }
    try:
        async with httpx.AsyncClient(timeout=360.0) as client:
            r = await client.post(f"{WRAPPER_URL}/generate", json=payload)
            r.raise_for_status()
            return _ok(r.json())
    except httpx.HTTPStatusError as e:
        return _err(
            f"ComfyUI wrapper error: {e.response.text}",
            error_type="comfyui",
            details={"status_code": e.response.status_code},
        )
    except Exception as e:
        traceback.print_exc()
        return _err(f"generation failed: {e}")


@mcp.tool()
async def comfyui_stop() -> dict:
    """Ask the wrapper to shut down the ComfyUI process (cold stop)."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{WRAPPER_URL}/stop")
            r.raise_for_status()
            return _ok(r.json())
    except Exception as e:
        traceback.print_exc()
        return _err(f"stop failed: {e}")


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
