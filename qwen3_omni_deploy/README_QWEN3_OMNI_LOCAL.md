# Qwen3-Omni Local Streaming Backend — README

Independent second local Video-LLM backend (target model
`Qwen/Qwen3-Omni-30B-A3B-Instruct`) for local video QA, streaming video input,
and future FocalStream cross-model evaluation. Does NOT touch the MERA /
MiniCPM-o 4.5 experiment or its `minicpmo45` environment.

## 1. Activate environment

    cd /workspace/qwen3_omni_deploy
    source .venv/bin/activate

## 2. Start server (text-only, single GPU, stage-0 only)

    bash scripts/start_qwen3_omni_server.sh

- Binds `127.0.0.1`, default port `8091` (falls back to `8092`; recorded in
  `qwen3_server.port`).
- Uses the local model path `/workspace/models/Qwen3-Omni-30B-A3B-Instruct`
  (no online model ID, no runtime download).
- Logs: `logs/qwen3_omni_server.log`. PID: `qwen3_server.pid`.

### Tuning env vars (optional)
    QWEN3_OMNI_GPU_MEM_UTIL=0.85      # default 0.85 (conservative)
    QWEN3_OMNI_MAX_MODEL_LEN=32768    # default 32768
    QWEN3_OMNI_MAX_NUM_BATCHED_TOKENS=32768
    QWEN3_OMNI_PORT=8091

The start script generates a thinker-only single-stage deploy config
(`pipeline: qwen3_omni_moe_thinker_only`), so talker/code2wav audio stages are
never created and the server does not wait for stage 1/2.

## 3. Health check

    bash scripts/check_qwen3_omni.sh
    # or:  curl http://127.0.0.1:8091/v1/models

## 4. Image QA

    .venv/bin/python tests/test_image_qa.py

## 5. Offline video QA

    .venv/bin/python tests/test_video_qa.py

## 6. Streaming video WebSocket test

    .venv/bin/python tests/test_streaming_video.py
    # FPS variants:
    .venv/bin/python tests/test_streaming_video.py --fps-list 0.5,1,2
    # Direct client use:
    .venv/bin/python clients/qwen3_video_stream_client.py \
        --video /path/to/clip.mp4 --fps 1 --query-times 5,10,15

## 7. Stop server (safe — only the recorded PID)

    bash scripts/stop_qwen3_omni_server.sh

Never use `pkill python`, `pkill vllm`, or `killall` — they would endanger MERA.

## 8. Common OOM handling (in order)
1. Lower `QWEN3_OMNI_GPU_MEM_UTIL` (e.g. 0.75).
2. Lower `QWEN3_OMNI_MAX_MODEL_LEN` (e.g. 16384).
3. Keep text-only output (`modalities: ["text"]`, stage-0 only) — avoids
   talker/code2wav VRAM.
4. Do NOT quantize first; BF16 is the default target.

## 9. Paths
- Model: `/workspace/models/Qwen3-Omni-30B-A3B-Instruct`
- Deploy root: `/workspace/qwen3_omni_deploy`
- Logs: `/workspace/qwen3_omni_deploy/logs`
- Results: `/workspace/qwen3_omni_deploy/results`

## 10. Streaming semantics caveat
The default endpoint is **streaming input (frame buffering)**, not incremental
visual prefill and not persistent model state. Each `video.query` rebuilds the
prompt from the retained frame buffer and runs a fresh generation. See
`docs/QWEN3_STREAMING_SEMANTICS.md` before any cross-model fairness claim.
