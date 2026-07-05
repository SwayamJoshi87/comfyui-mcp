# ComfyUI OpenAI/FAL-Compatible Images API

A single-container, multi-protocol image-generation backend backed by [ComfyUI](https://github.com/comfyanonymous/ComfyUI). The container starts ComfyUI on the first generation request, keeps it alive while you use it, and cold-stops it after an idle timeout so GPU memory is freed when nothing is generating.

Supported protocols:

- **OpenAI-compatible** — `POST /v1/images/generations`, `GET /v1/models`
- **FAL.ai-compatible** — `POST /fal/run/{model_id}`, `POST /fal/queue/{model_id}`, `GET /fal/queue/{model_id}/requests/{request_id}/status`
- **Wrapper native** — `POST /generate`, `GET /health`, `GET /status`, `POST /stop`

Generated images are persisted under `ComfyUI/output` and served at `/outputs/{filename}`, so `response_format: url` and FAL-style responses return real URLs.

## Endpoints

### OpenAI-compatible

`POST /v1/images/generations` accepts the standard OpenAI request shape:

```json
{
  "prompt": "a photo of an astronaut riding a horse on the moon",
  "n": 1,
  "size": "512x512",
  "response_format": "b64_json"
}
```

Supported parameters:

- `prompt` (required)
- `n` (1–4, default 1)
- `size` — `"WIDTHxHEIGHT"`, e.g. `"512x512"` (default), `"1024x1024"`
- `response_format` — `"b64_json"` (default) or `"url"`
- `model` — accepted but currently informational
- `quality`, `style`, `user` — accepted but ignored
- ComfyUI overrides: `negative_prompt`, `width`, `height`, `steps`, `cfg`, `seed`

### FAL.ai-compatible

`POST /fal/run/{model_id}` accepts a FAL-style payload:

```json
{
  "prompt": "a serene mountain landscape with cherry blossoms",
  "image_size": "landscape_16_9",
  "num_images": 1,
  "seed": 42
}
```

Supported fields:

- `prompt` (required)
- `negative_prompt`
- `image_size` — e.g. `square_hd`, `landscape_16_9`, `portrait_16_9`
- `aspect_ratio` — e.g. `1:1`, `16:9`, `9:16`
- `width` / `height` (override size/aspect)
- `num_inference_steps` → mapped to ComfyUI steps
- `guidance_scale` → mapped to ComfyUI cfg
- `seed`
- `num_images` (1–4)
- `image_url` / `reference_image_urls` — rejected with a clear error (editing is not supported)

Response shape mirrors FAL:

```json
{
  "images": [
    {
      "url": "http://192.168.2.51:8002/outputs/abc123.png",
      "width": 1536,
      "height": 1024,
      "content_type": "image/png"
    }
  ],
  "prompt": "a serene mountain landscape with cherry blossoms",
  "seed": 42,
  "has_nsfw_concepts": [false]
}
```

## Running the container

```bash
cd wrapper
# place v1-5-pruned-emaonly.safetensors in this directory first
docker compose up -d --build
```

The API is exposed on port `8002` (ComfyUI direct is on `8190`).

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `IDLE_TIMEOUT` | `300` | Seconds of inactivity before ComfyUI is stopped |
| `COMFYUI_PORT` | `8188` | Internal ComfyUI port |
| `API_PORT` | `8000` | Internal API port |
| `PUBLIC_URL` | *(request base URL)* | Base URL used for generated image links, e.g. `http://192.168.2.51:8002` |
| `OPENAI_API_KEY` | *(none)* | Optional Bearer token required on OpenAI endpoints |
| `MODEL_ID` | `comfyui-sd1-5` | Model id advertised by `/v1/models` |

## Using with Hermes

Hermes supports FAL.ai as a backend. Point Hermes at this container by making `fal.run` / `queue.fal.run` resolve to `http://192.168.2.51:8002`. The easiest way is to add a local DNS or proxy rule, or set the FAL client base URL if Hermes/FAL's SDK exposes one.

Set `FAL_KEY` to any non-empty dummy value in Hermes config, pick any FAL model id (e.g. `fal-ai/flux-2/klein/9b`), and generation requests will be handled by your local ComfyUI backend.

## Example requests

OpenAI:

```bash
curl -X POST http://127.0.0.1:8002/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a photo of an astronaut riding a horse on the moon",
    "size": "512x512",
    "response_format": "b64_json"
  }'
```

FAL:

```bash
curl -X POST http://127.0.0.1:8002/fal/run/fal-ai/flux-2/klein/9b \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a serene mountain landscape with cherry blossoms",
    "image_size": "landscape_16_9"
  }'
```

## Cold-start behavior

The first request after the wrapper has been idle will trigger a ComfyUI startup. The API server waits for ComfyUI to start and generate the image, so the client may observe a longer response time on the first call. Subsequent calls are fast until the idle timeout elapses and ComfyUI is stopped again.
