# Install Commands (Qwen3-Omni vLLM-Omni, isolated `uv` env)

All commands are executed inside `/workspace/qwen3_omni_deploy`. Nothing is
installed into the base/minicpmo45/qwen25omni conda environments.

## 1. Environment (Python 3.12, isolated uv venv)

    cd /workspace/qwen3_omni_deploy
    export UV_CACHE_DIR=/workspace/qwen3_omni_deploy/.uv_cache
    export HF_HOME=/workspace/models/hf_cache
    uv venv --python 3.12 .venv

## 2. Install vLLM (must match vLLM-Omni major/minor)

    uv pip install --python .venv/bin/python --upgrade pip
    uv pip install --python .venv/bin/python "vllm==0.30.0" --torch-backend=auto

## 3. Install vLLM-Omni (stable wheel, pairs with vLLM 0.30.0)

    uv pip install --python .venv/bin/python "vllm-omni==0.30.0"

## 4. Client/test dependencies

    uv pip install --python .venv/bin/python \
        websockets pillow opencv-python numpy requests imageio-ffmpeg qwen-omni-utils

## 5. (Optional) Transformers fallback only — NOT the primary stack

    uv pip install --python .venv/bin/python "transformers>=5.2.0" accelerate
    uv pip install --python .venv/bin/python qwen-omni-utils
    # system ffmpeg required for qwen-omni-utils audio processing;
    # visual-only paths use PyAV (bundled) / OpenCV (bundled FFmpeg).

## Why stable wheel, not source
- Stable `vllm-omni==0.30.0` + `vllm==0.30.0` provides `/v1/video/chat/stream`.
- `main` source targets vLLM 0.31.0 and adds opt-in incremental prefill
  (stage-0 `enable_prefix_caching: true` + text-only). Recorded in
  `OFFICIAL_DEPLOYMENT_NOTES.md`; not selected for this first install.

## Activate
    cd /workspace/qwen3_omni_deploy
    source .venv/bin/activate
