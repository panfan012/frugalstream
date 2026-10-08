#!/usr/bin/env bash
# Start Qwen3-Omni-30B-A3B-Instruct (text-only) via vLLM-Omni stage 0.
# Only the thinker + API server is launched -> no talker/code2wav -> no audio output.
# Refuses to run if the model directory is missing or if a server is already running.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${DEPLOY_DIR}/logs/qwen3_omni_server.log"
PID_FILE="${DEPLOY_DIR}/qwen3_server.pid"
PORT_FILE="${DEPLOY_DIR}/qwen3_server.port"

MODEL_PATH="${QWEN3_OMNI_MODEL_PATH:-/workspace/models/Qwen3-Omni-30B-A3B-Instruct}"
HOST="${QWEN3_OMNI_HOST:-127.0.0.1}"
BASE_PORT="${QWEN3_OMNI_PORT:-8091}"
GPU_MEM_UTIL="${QWEN3_OMNI_GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${QWEN3_OMNI_MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${QWEN3_OMNI_MAX_NUM_BATCHED_TOKENS:-32768}"

# Resolve vllm executable: prefer this task's .venv.
if [[ -x "${DEPLOY_DIR}/.venv/bin/vllm" ]]; then
  VLLM_BIN="${DEPLOY_DIR}/.venv/bin/vllm"
elif command -v vllm >/dev/null 2>&1; then
  VLLM_BIN="$(command -v vllm)"
else
  echo "ERROR: vllm not found. Create the env first (see docs/INSTALL_COMMANDS.md)." >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model dir missing: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "ERROR: server already running (PID $(cat "${PID_FILE}")). Use stop script first." >&2
  exit 1
fi

# Pick port: use BASE_PORT, fall back to +1 if busy. Record the chosen port.
if command -v ss >/dev/null 2>&1; then
  PORT="${BASE_PORT}"
  if ss -ltn | awk '{print $4}' | grep -qE ":${PORT}$"; then
    PORT=$((BASE_PORT + 1))
    echo "Port ${BASE_PORT} busy -> using ${PORT}"
  fi
else
  PORT="${BASE_PORT}"
fi
echo "${PORT}" > "${PORT_FILE}"

# Text-only thinker-only topology. This is the key to single-GPU startup:
# it avoids creating talker/code2wav stages and waiting for remote stage 1/2.
DEPLOY_CONFIG="${DEPLOY_DIR}/scripts/.qwen3_omni_deploy.generated.yaml"
cat > "${DEPLOY_CONFIG}" <<EOF
pipeline: qwen3_omni_moe_thinker_only
distributed_executor_backend: mp
enable_prefix_caching: false

stages:
  - stage_id: 0
    devices: "0"
    max_num_seqs: 8
    max_num_batched_tokens: ${MAX_NUM_BATCHED_TOKENS}
    max_model_len: ${MAX_MODEL_LEN}
    gpu_memory_utilization: ${GPU_MEM_UTIL}
    enforce_eager: false
    async_scheduling: false
    moe_backend: triton
    trust_remote_code: true
    default_sampling_params:
      temperature: 0.0
      top_p: 0.9
      top_k: -1
      max_tokens: 2048
      repetition_penalty: 1.05
EOF

echo "Starting Qwen3-Omni thinker-only (text-only) on ${HOST}:${PORT}"
echo "  model=${MODEL_PATH}"
echo "  gpu_memory_utilization=${GPU_MEM_UTIL} max_model_len=${MAX_MODEL_LEN}"
echo "  deploy_config=${DEPLOY_CONFIG}"
echo "  log=${LOG_FILE}"

# RTX PRO 6000 Blackwell: FlashInfer JIT wrongly rejects SM 12.x during
# top-k/top-p warmup. Keep vLLM's native sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER=0

setsid "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --omni \
  --deploy-config "${DEPLOY_CONFIG}" \
  --host "${HOST}" \
  --port "${PORT}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

echo $! > "${PID_FILE}"
echo "Started PID $(cat "${PID_FILE}") on ${HOST}:${PORT}"
