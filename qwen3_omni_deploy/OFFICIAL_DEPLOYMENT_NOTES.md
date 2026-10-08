# Official Deployment Notes — Qwen3-Omni-30B-A3B-Instruct (vLLM-Omni)

Task: QWEN3-OMNI LOCAL STREAMING BACKEND DEPLOYMENT
Date: 2026-10-06 (UTC)
Sources: vLLM-Omni repo (github.com/vllm-project/vllm-omni, branch `main`),
         Qwen3-Omni repo (github.com/QwenLM/Qwen3-Omni, branch `main`),
         PyPI metadata (pypi.org).

## 1. Confirmed official facts (do NOT use old tutorials blindly)

### Python version
- vLLM-Omni official install guide: `uv venv --python 3.12 --seed`
- vLLM-Omni recipe (Qwen3-Omni): Python 3.10+
- vLLM wheel metadata: `requires_python <3.15, >=3.10`
- vLLM-Omni wheel metadata: `requires_python <3.14, >=3.10`
- **Decision: Python 3.12** (official recommendation, compatible with both).

### Current versions (checked on 2026-10-06 via PyPI)
- vLLM latest stable: **0.31.0**
- vLLM 0.30.0: exists (`vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl`, ~0.31 GB)
- vLLM-Omni latest stable: **0.30.0**
- vLLM-Omni pre-release: 0.31.0rc1 (pairs with vLLM 0.31.0)

### Version compatibility rule
- vLLM-Omni is released against the **matching upstream vLLM major/minor** version.
- Published `vllm-omni==0.30.0` wheel -> pairs with `vllm==0.30.0`.
- `main` source checkout -> targets `vllm==0.31.0` (commit-pinned wheel index
  `https://wheels.vllm.ai/db9527a46873454610df6dbedf79a36d6bf1a7f6`).

### Deployed stack decision
- **Stable wheel path (chosen):** `vllm==0.30.0` + `vllm-omni==0.30.0`.
  Rationale: task preference for stable wheel; `/v1/video/chat/stream` is present.
- Source path (`main`, vLLM 0.31.0) adds optional "incremental prefill" via
  stage-0 `enable_prefix_caching: true` + `modalities: ["text"]`. Documented below
  but not selected for the first install.

## 2. Online serving command (official)

Single command (default deploy config `vllm_omni/deploy/qwen3_omni_moe.yaml`
loaded automatically by the model registry):

    vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091

- `--omni` is **required**.
- text-only output requests use `modalities: ["text"]`.
- `{"modalities":["text","audio"]}` returns text + TTS audio.
- `/v1/realtime` is unsupported while `async_chunk` is enabled (enabled by default).

Text-only smoke test (OpenAI-compatible endpoint):

    curl http://localhost:8091/v1/chat/completions -H "Content-Type: application/json" \
      -d '{"model":"Qwen/Qwen3-Omni-30B-A3B-Instruct",
           "messages":[{"role":"user","content":"Describe vLLM in brief."}],
           "modalities":["text"]}'

## 3. Video streaming endpoint

- Endpoint: **WebSocket `/v1/video/chat/stream`** (Qwen3-Omni streaming video
  understanding; each server instance hosts one model specified at startup).

### Protocol (client -> server)
| type | required | description |
|------|----------|-------------|
| `session.config` | none | first message; output modalities, frame sampling, EVS, prompts |
| `video.frame` | `data` | base64 JPEG/PNG frame |
| `audio.chunk` | `data` | base64 PCM16 16kHz mono audio bytes |
| `video.query` | `text` | question over buffered frames/audio |
| `video.done` | none | end the WebSocket session |

### Protocol (server -> client)
| type | fields | description |
|------|--------|-------------|
| `response.start` | none | generation started |
| `response.text.delta` | `delta` | incremental text |
| `response.text.done` | `text` | final text for query |
| `response.output_audio.delta` | `data`,`format` | incremental generated audio (base64 WAV) |
| `response.output_audio.done` | none | audio finished |
| `session.done` | none | session closed |
| `error` | `message` | recoverable protocol/generation error |
| `video.frames.consumed` | `frame_ids`,`frames`,`latest_pts_ms` | (when `frame_id` supplied) frames included in query prompt |

### `session.config` fields (canonical)
| field | type | default | description |
|-------|------|---------|-------------|
| `model` | str/null | null | optional (server hosts one model) |
| `modalities` | list[str] | ["text","audio"] | use ["text"] for text-only |
| `num_frames` | int 1-128 | 4 | buffered frames sampled per query (legacy path) |
| `max_frames` | int 1-256 | 50 | ring-buffer capacity, oldest evicted |
| `system_prompt` | str/null | null | optional |
| `use_audio_in_video` | bool | true | include streamed audio in video understanding |
| `sampling_params_list` | list/null | null | per-stage sampling overrides |
| `enable_frame_filter` | bool | true | EVS near-duplicate frame filter |
| `frame_filter_threshold` | float 0-1 | 0.95 | EVS similarity threshold |

Legacy aliases (still accepted): `num_sample_frames`->`num_frames`,
`evs_enabled`->`enable_frame_filter`, `evs_threshold`->`frame_filter_threshold`.

### Environment variables
| variable | values | default | meaning |
|----------|--------|---------|---------|
| `VLLM_VIDEO_ASYNC_CHUNK` | on/off | on | `off` = buffer deltas, emit coalesced at query end |
| `VLLM_VIDEO_AUDIO_DELTA_MODE` | fast/slow | fast | audio delta forwarding |

## 4. Streaming semantics (CRITICAL for research fairness)

Two distinct behaviors exist:

### A. Legacy path (vLLM-Omni 0.30.0 stable; default)
- Frames arrive via `video.frame` and are held in a **ring buffer** (`max_frames`, oldest evicted).
- `video.query` **re-samples** the buffered frames up to `num_frames` (stride sample, keep last frame).
- **Each query rebuilds the full multimodal prompt from the retained buffer and
  runs a fresh generation.** No KV reuse across queries.
- Frame buffer persists across queries within a session, but model state does not.

### B. Incremental prefill path (vLLM-Omni `main`, opt-in)
- Requires stage-0 `enable_prefix_caching: true` AND `modalities: ["text"]`.
- Warmup prefills `history + frames` as they arrive with `max_tokens=1`; the query
  reuses the cached vision prefix (this is a form of persistent visual KV via
  prefix caching).
- `num_frames` subsampling is legacy-only; the incremental path submits all
  buffered frames in arrival order.

Official wording (main docs): "Session KV reuse and incremental prefill are not
implemented in this PR. Each video.query rebuilds the model prompt from the
retained frame and audio buffers." — this describes the legacy path; the newer
`main` branch adds opt-in incremental prefill as described above.

Conclusion: stable 0.30.0 is **streaming INPUT (frame buffering)**, not
incremental visual prefill. See docs/QWEN3_STREAMING_SEMANTICS.md.

## 5. Model download (official commands)

Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Hugging Face tree total: **70.53 GB** (15 safetensors shards ~5GB each + configs).
- HF sha: `26291f793822fb6be9555850f06dfe95f2d7e695`
- lastModified: 2025-09-22

    # ModelScope (preferred when HF is slow)
    pip install -U modelscope
    modelscope download --model Qwen/Qwen3-Omni-30B-A3B-Instruct --local_dir /workspace/models/Qwen3-Omni-30B-A3B-Instruct

    # Hugging Face
    pip install -U "huggingface_hub[cli]"
    huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir /workspace/models/Qwen3-Omni-30B-A3B-Instruct

Both tools support resumable downloads (`.incomplete` partial files on HF side;
ModelScope uses its own cache).

## 6. Transformers fallback (NOT the primary stack)

- Official recommendation: `transformers>=5.2.0` (older 4.57.x has lower accuracy).
- `pip install qwen-omni-utils -U` + system `ffmpeg`.
- `model.disable_talker()` saves ~10GB VRAM (text-only).
- FlashAttention 2 recommended for Transformers; vLLM bundles FA2 already.

## 7. Hardware note
- vLLM-Omni official example was verified on 2x H100 for this model.
- Thinker BF16 weights ~62 GiB (per vLLM-Omni XPU profile note). Talker is extra
  (~10GB unless disabled). Single RTX PRO 6000 Blackwell (~95GB) is tight; use
  text-only output and conservative memory settings.
