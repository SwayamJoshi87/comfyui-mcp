# ComfyUI OpenAI-Compatible Images API

An OpenAI-compatible image-generation backend backed by a remote [ComfyUI](https://github.com/comfyanonymous/ComfyUI) instance. The GPU wrapper in `wrapper/` starts ComfyUI on the first generation request, keeps it alive while you use it, and cold-stops it after an idle timeout so GPU memory is freed when nothing is generating.

This repo contains:

- `comfyui_api/server.py` — FastAPI service exposing OpenAI-compatible `/v1/images/generations` and `/v1/models`.
- `wrapper/` — Dockerized FastAPI wrapper that manages the ComfyUI process (cold start / idle shutdown).

The API layer itself is stateless and does not hold ComfyUI in memory; it forwards requests to the wrapper over HTTP.

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /v1/models` | List the available model. |
| `POST /v1/images/generations` | Generate one or more PNG images from a prompt. Cold-starts ComfyUI if needed. |
| `GET /health` | Health of the wrapper process. |
| `GET /status` | Whether ComfyUI is running and how long it has been idle. |

`POST /v1/images/generations` accepts the standard OpenAI request shape:

```json
{
  "prompt": "a photo of an astronaut riding a horse on the moon",
  "n": 1,
  "size": "512x512",
  "response_format": "b64_json"
}
```

And returns:

```json
{
  "created": 1718000000,
  "data": [
    {
      "b64_json": "iVBORw0KGgo...",
      "revised_prompt": "a photo of an astronaut riding a horse on the moon"
    }
  ]
}
```

Supported parameters:

- `prompt` (required)
- `n` (1–4, default 1)
- `size` — `"WIDTHxHEIGHT"`, e.g. `"512x512"` (default), `"1024x1024"`
- `response_format` — `"b64_json"` (default) or `"url"` (returns a `data:image/png;base64,...` URL)
- `model` — accepted but currently informational
- `quality`, `style`, `user` — accepted but ignored

ComfyUI-specific overrides (non-OpenAI):

- `negative_prompt`
- `width` / `height` (override `size`)
- `steps` (1–150, default 20)
- `cfg` (1.0–30.0, default 7.0)
- `seed` (-1 for random)

## Running the wrapper

The wrapper must be running and reachable before the API server starts.

```bash
cd wrapper
# place v1-5-pruned-emaonly.safetensors in this directory first
docker compose up -d --build
```

The wrapper exposes its API on port `8002` (ComfyUI direct is on `8190`).

## Running the API server

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # or pip install -e .
COMFYUI_WRAPPER_URL=http://127.0.0.1:8002 .venv/bin/python -m comfyui_api.server
```

The OpenAI-compatible API is now available at `http://127.0.0.1:8000/v1`.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `COMFYUI_WRAPPER_URL` | `http://127.0.0.1:8002` | Base URL of the ComfyUI wrapper |
| `API_HOST` | `0.0.0.0` | Host to bind the API server on |
| `API_PORT` | `8000` | Port to bind the API server on |
| `OPENAI_API_KEY` | *(none)* | Optional Bearer token required on all endpoints |
| `MODEL_ID` | `comfyui-sd1-5` | Model id advertised by `/v1/models` |

## Example client request

```bash
curl -X POST http://127.0.0.1:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a photo of an astronaut riding a horse on the moon",
    "size": "512x512",
    "response_format": "b64_json"
  }'
```

## Cold-start behavior

The first request after the wrapper has been idle will trigger a ComfyUI startup. The API server waits for the wrapper to finish starting ComfyUI and generating the image, so the client may observe a longer response time on the first call. Subsequent calls are fast until the idle timeout elapses and ComfyUI is stopped again.
