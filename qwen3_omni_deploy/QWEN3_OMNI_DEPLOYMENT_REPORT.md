# QWEN3-OMNI LOCAL DEPLOYMENT REPORT

Task: QWEN3-OMNI LOCAL STREAMING BACKEND DEPLOYMENT
Date: 2026-10-07 (UTC)
Target model: Qwen/Qwen3-Omni-30B-A3B-Instruct

## 1. Executive Summary

- Isolated deployment complete and live-verified on one RTX PRO 6000 Blackwell.
- **FULL PASS**: environment, model download, single-GPU load, image QA,
  offline video QA, streaming video WebSocket, repeated same-session queries,
  0.5/1/2 FPS streaming smoke tests.
- MERA / MiniCPM-o 4.5 processes were not killed, paused, or modified. No GPU
  was taken while MERA was running; load started only after the GPU was free
  (0 compute processes, 437 MiB used).
- The deployment is separate from `minicpmo45`, OVO-Bench, and FocalStream.

## 2. Hardware

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
  - Total: 97887 MiB (~95.6 GiB)
  - After server load: 83153 MiB used / 14062 MiB free (idle)
  - Driver: 580.126.09, CUDA runtime 13.2 (torch), sm_120
- CPU/RAM: 24 visible cores / ~22.8 GiB usable at load time.
- Disk: overlay `/` and `/workspace` same mount, **992 GB available** after model
  download + environment install.

## 3. Environment

- Path: `/workspace/qwen3_omni_deploy/.venv` (isolated `uv` venv)
- Env disk size: 8.4 GiB
- Base/minicpmo45/qwen25omni conda envs untouched.

## 4. Installed Versions

| package | version |
|---|---|
| python | 3.12.14 |
| pip | 26.2.1 |
| torch | 2.13.0+cu132 |
| vllm | 0.30.0 |
| vllm-omni | 0.30.0 |
| transformers | 5.14.1 |
| qwen-omni-utils | 0.0.9 |
| numpy | 2.5.3 |
| opencv-python-headless | 5.0.0.93 |
| av (PyAV) | 19.0.1 |
| websockets | 17.2 |
| pillow | 12.3.0 |
| imageio-ffmpeg | 0.6.0 |
| requests | 2.34.2 |

Freeze: `docs/environment_freeze.txt`.

## 5. Model Download

- Local path: `/workspace/models/Qwen3-Omni-30B-A3B-Instruct`
- Disk size: 66 GiB (`du`), logical total 70,519,637,090 bytes (~70.52 GB).
- 15/15 safetensors shards, sizes byte-exact vs HF, safetensors headers valid,
  no `.incomplete/.tmp/.partial` files.
- Inventory: `model_info/model_inventory.txt` = DOWNLOAD COMPLETE + VERIFIED.

## 6. Single-GPU Load

- **PASS**. BF16, no quantization, single GPU.
- Startup command uses official thinker-only pipeline:
  `--deploy-config scripts/.qwen3_omni_deploy.generated.yaml`
  (pipeline `qwen3_omni_moe_thinker_only`), not the full 3-stage audio pipeline.
- Tuning: `gpu_memory_utilization 0.85`, `max_model_len 32768`,
  `max_num_batched_tokens 32768`, `max_num_seqs 8`.
- Weight load: 10.88 s, 59.54 GiB model memory.
- Engine init (profile/KV/warmup): 38.93 s with cached compilation; total
  launch-to-ready ~72 s.
- Startup VRAM accounting: 62.04 GiB weights/non-torch + 3.54 GiB peak
  activation + 0.10 GiB CUDA graph + 17.46 GiB KV cache.

## 7. Image QA

- **PASS**: `tests/test_image_qa.py`
- Image: `/root/OVO-Bench-full/results/stage2b5_accuracy_vs_token.png`
- Question: "What is happening in this image?"
- First-request latency: 20.135 s (includes first-inference JIT/kernel warmup).
- Usage: prompt_tokens=1333, completion_tokens=256, total=1589.
- Result: `results/test_image_qa.json`

## 8. Offline Video QA

- **PASS**: `tests/test_video_qa.py`
- Video: `/root/OVO-Bench-full/data/src_videos/star/X4DOF.mp4` (0.32 MB, ~18 s)
- Question: "What is the main activity in this video?"
- Latency: 0.548 s. Answer: "The main activity in this video is a man throwing
  a pillow at a door."
- Usage: prompt_tokens=2539, completion_tokens=17, total=2556.
- Result: `results/test_video_qa.json`

## 9. Streaming Video API

- **PASS**: WebSocket `/v1/video/chat/stream`
- 0.5 FPS: PASS, query@8s latency 0.436 s
- 1.0 FPS: PASS, query@8s latency 0.423 s
- 2.0 FPS: PASS, query@8s latency 0.426 s
- Results: `results/test_streaming_video.json`, `results/stream_fps_*.json`

## 10. Streaming Semantics

See `docs/QWEN3_STREAMING_SEMANTICS.md`. Confirmed live behavior:

- Input streaming: YES (WebSocket frame upload, causal timestamps).
- Incremental visual prefill: **NO** (stable 0.30.0 default).
- Persistent visual KV: **NO**.
- Frame buffer: YES, server-side; persists during a WS session.
- Query-time resampling: YES (`num_frames` stride + last frame).
- New generation per query: YES.
- Multi-turn same-session queries: YES (fresh generation each time).

## 11. Repeated Query Session

- **PASS**: same WS session, queries at 5s / 10s / 15s.
- Latencies: 0.422 s / 0.537 s / 0.386 s.
- Result: `results/stream_repeated_query.json`

## 12. Latency / VRAM

- Server startup: ~72 s (cache-warmed second launch).
- Model load: 10.88 s.
- Idle VRAM: 83153 MiB used / 14062 MiB free.
- Peak observed nvidia-smi VRAM: ~83153 MiB during idle; startup peak activation
  3.54 GiB on top of 62.04 GiB consumed.
- Image QA first request: 20.135 s (warmup dominated).
- Offline video QA: 0.548 s.
- Streaming query latencies: 0.386-0.537 s.

## 13. Token Accounting

- OpenAI `/v1/chat/completions` exposes `usage.prompt_tokens`,
  `usage.completion_tokens`, `usage.total_tokens`.
- `prompt_tokens_details` and `completion_tokens_details` are `null`, so
  multimodal visual-token breakdown is **NOT EXPOSED** (not fabricated).

## 14. Comparison with MiniCPM Streaming Semantics

| Capability | MiniCPM-o 4.5 | Qwen3-Omni (vLLM-Omni 0.30.0) |
|---|---|---|
| Local video input | per task brief | YES |
| Streaming frame input | per task brief | YES |
| Incremental visual prefill | per task brief | NO |
| Persistent model state | per task brief | NO |
| Query after stream | per task brief | YES |
| One final generation | per task brief | YES |
| Text-only output | per task brief | YES |
| Visual token exposed | ... | NOT EXPOSED / UNKNOWN |
| Realtime WS | ... | YES |

MiniCPM column remains "per task brief"; this task did not re-inspect or modify
the MiniCPM/FocalStream codebase.

## 15. Known Limitations

- Streaming endpoint = **streaming input / frame buffering**, NOT incremental
  visual prefill or persistent visual KV. Each query is a fresh generation.
- Single ~95 GiB GPU with BF16 thinker requires text-only stage-0 mode; audio
  talker/code2wav stages are intentionally not launched.
- FlashInfer sampler is disabled on this Blackwell card
  (`VLLM_USE_FLASHINFER_SAMPLER=0`) because FlashInfer JIT misreports sm_12.x.
- First request after startup is slow due to JIT/kernel warmup.

## 16. Ready for FocalStream Cross-Model Evaluation?

- **YES, for black-box text-output backend comparisons.** Adapter
  `clients/qwen3_backend_adapter.py` and `backend_capabilities` are ready and
  honestly report the buffering semantics above.
- Do not label this backend as incremental visual prefill.

## 17. Exact Start/Stop Commands

    cd /workspace/qwen3_omni_deploy
    bash scripts/start_qwen3_omni_server.sh
    bash scripts/check_qwen3_omni.sh
    bash scripts/stop_qwen3_omni_server.sh

Health: `curl http://127.0.0.1:8091/v1/models`
PID: `qwen3_server.pid`; port: `qwen3_server.port` (8091).
