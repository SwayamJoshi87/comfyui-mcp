# ComfyUI OpenAI-Compatible Images API

A single-container, OpenAI-compatible image-generation backend backed by [ComfyUI](https://github.com/comfyanonymous/ComfyUI). The container starts ComfyUI on the first generation request, keeps it alive while you use it, and cold-stops it after an idle timeout so GPU memory is freed when nothing is generating.

Everything runs in one Docker container defined in `wrapper/`.

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /v1/models` | List the available model. |
| `POST /v1/images/generations` | Generate one or more PNG images from a prompt. Cold-starts ComfyUI if needed. |
| `GET /health` | Wrapper health check. |
| `GET /status` | Whether ComfyUI is running and how long it has been idle. |
| `POST /stop` | Ask the wrapper to shut down ComfyUI immediately. |
| `POST /generate` | Internal wrapper endpoint (also available). |

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

## Running the container

```bash
cd wrapper
# place v1-5-pruned-emaonly.safetensors in this directory first
docker compose up -d --build
```

The OpenAI-compatible API is exposed on port `8002` (ComfyUI direct is on `8190`).

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `IDLE_TIMEOUT` | `300` | Seconds of inactivity before ComfyUI is stopped |
| `COMFYUI_PORT` | `8188` | Internal ComfyUI port |
| `API_PORT` | `8000` | Internal API port |
| `OPENAI_API_KEY` | *(none)* | Optional Bearer token required on all endpoints |
| `MODEL_ID` | `comfyui-sd1-5` | Model id advertised by `/v1/models` |

## Example client request

```bash
curl -X POST http://127.0.0.1:8002/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a photo of an astronaut riding a horse on the moon",
    "size": "512x512",
    "response_format": "b64_json"
  }'
```

## Cold-start behavior

The first request after the wrapper has been idle will trigger a ComfyUI startup. The API server waits for ComfyUI to start and generate the image, so the client may observe a longer response time on the first call. Subsequent calls are fast until the idle timeout elapses and ComfyUI is stopped again.
