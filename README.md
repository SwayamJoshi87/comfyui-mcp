# ComfyUI MCP server

A stdio MCP server that exposes a remote [ComfyUI](https://github.com/comfyanonymous/ComfyUI) instance through an on-demand GPU wrapper. The wrapper lives in `wrapper/` and is built as a Docker image. It starts ComfyUI on first generation request, keeps it alive while you use it, and cold-stops it after an idle timeout.

This repo contains:

- `comfyui_mcp/server.py` — stdio MCP server that talks to the wrapper over HTTP.
- `wrapper/` — Dockerized FastAPI wrapper that manages the ComfyUI process.
- `hermes/mcp.json` — sample Hermes MCP config.

## Tools

| Tool | Description |
|---|---|
| `comfyui_health` | Wrapper health check. |
| `comfyui_status` | Whether ComfyUI is running and how long it has been idle. |
| `comfyui_generate` | Generate a PNG image from a prompt. Cold-starts ComfyUI if needed. |
| `comfyui_stop` | Ask the wrapper to shut ComfyUI down immediately. |

All tools return the same `{ok, data}` / `{ok, error}` shape used by `sam-os`.

## Running the wrapper

```bash
cd wrapper
# place v1-5-pruned-emaonly.safetensors in this directory first
docker compose up -d --build
```

The wrapper API is exposed on port `8002` (ComfyUI direct is on `8190`).

## Running the MCP server

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # or pip install -e .
COMFYUI_WRAPPER_URL=http://192.168.2.51:8002 .venv/bin/python -m comfyui_mcp.server
```

## Hermes config

Copy `hermes/mcp.json` into your Hermes MCP config and adjust paths.

## Environment variables

- `COMFYUI_WRAPPER_URL` — base URL of the wrapper (default: `http://192.168.2.51:8002`)
