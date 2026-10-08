# Qwen3-Omni Streaming Semantics (vLLM-Omni `/v1/video/chat/stream`)

Sources: vllm-project/vllm-omni `main` `docs/serving/video_stream_api.md`,
`vllm_omni/entrypoints/openai/serving_video_stream.py`,
`vllm_omni/entrypoints/openai/video_stream_*.py`. Checked 2026-10-06.

## Verdict for cross-model fairness

**The stable legacy path is STREAMING INPUT (frame buffering), NOT
incremental visual prefill and NOT persistent model state.**

```
THIS IS STREAMING INPUT, NOT INCREMENTAL VISUAL PREFILL.
```

Do not label Qwen3-Omni streaming video as equivalent to MiniCPM-o
`streaming_prefill` unless the incremental-prefill path is explicitly enabled
and measured. The default endpoint buffers frames and rebuilds the multimodal
prompt for every query.

## Semantics matrix

| Question | Stable 0.30.0 (legacy, default) | `main` opt-in (enable_prefix_caching + text-only) |
|---|---|---|
| Input streaming | **YES** (frames arrive incrementally over WS) | YES |
| Incremental visual prefill | **NO** | YES (warmup `max_tokens=1` as frames arrive) |
| Persistent visual KV / state | **NO** (prompt rebuilt per query) | YES via prefix cache (vision prefix reused) |
| Frame buffer | **YES** ring buffer, `max_frames` (default 50), oldest evicted | YES (drops oldest half at cap) |
| Query-time resampling | **YES** stride-sample `num_frames` + keep last frame | NO (`num_frames` is legacy-only; all buffered frames submitted) |
| New generation per query | **YES** | YES (query pays only for audio/text suffix) |
| Multi-turn / repeated query | **YES** (buffer + history persist across queries) | YES |
| History | Compressed to last two text messages after a turn | Same |
| Protocol | WebSocket `/v1/video/chat/stream` | Same |
| Audio-output sessions | supported | stay on legacy path (no incremental prefill) |

## Activation of incremental prefill (main branch)

- Server-side only: set `enable_prefix_caching: true` on stage 0 (thinker) in
  the deploy config AND use `modalities: ["text"]`.
- There is no `session.config` flag. Audio-output sessions always use legacy.

## Protocol events (both paths)

Client -> server:
`session.config`, `video.frame` (`data`, optional `frame_id`/`pts_ms`),
`audio.chunk`, `video.query` (`text`), `video.done`.

Server -> client:
`response.start`, `response.text.delta`, `response.text.done`,
`response.output_audio.delta`, `response.output_audio.done`,
`video.frames.consumed` (`frame_ids`,`frames`,`latest_pts_ms`),
`video.frame.ack` (when `frame_id` supplied), `session.done`, `error`.

`session.config` fields: `model`, `modalities` (default `["text","audio"]`),
`num_frames` (default 4, legacy), `max_frames` (default 50),
`system_prompt`, `use_audio_in_video` (default true),
`sampling_params_list`, `enable_frame_filter` (default true),
`frame_filter_threshold` (default 0.95).
Legacy aliases: `num_sample_frames`, `evs_enabled`, `evs_threshold`.

## Env vars
- `VLLM_VIDEO_ASYNC_CHUNK` = on (default) | off (buffer deltas, coalesce at end)
- `VLLM_VIDEO_AUDIO_DELTA_MODE` = fast (default) | slow

## Known limitations (upstream)
- Legacy path: no KV reuse, no incremental prefill.
- Incremental prefill (main): only text-only sessions with stage-0 prefix
  caching; does not accelerate `/v1/realtime` camera input.
- Back-to-back short replies can hit an engine scheduler race; keep >=200 ms
  idle between turns if idle timeouts appear.
- Audio buffer overflow clears buffered audio for the session.
- Intended for Qwen3-Omni streaming video understanding.
