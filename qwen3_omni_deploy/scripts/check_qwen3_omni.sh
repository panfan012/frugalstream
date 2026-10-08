#!/usr/bin/env bash
# Health check for the Qwen3-Omni server. Read-only; never sends signals.
set -u

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${DEPLOY_DIR}/qwen3_server.pid"
PORT_FILE="${DEPLOY_DIR}/qwen3_server.port"
PORT="${QWEN3_OMNI_PORT:-8091}"
[[ -f "${PORT_FILE}" ]] && PORT="$(cat "${PORT_FILE}")"

echo "=== PID ==="
if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "alive: PID $(cat "${PID_FILE}")"
else
  echo "not running"
fi

echo "=== Port ${PORT} ==="
if command -v ss >/dev/null 2>&1; then
  ss -ltn | grep -E ":${PORT}\b" || echo "no listener on ${PORT}"
else
  echo "ss unavailable"
fi

echo "=== /v1/models ==="
curl -sS -m 10 "http://127.0.0.1:${PORT}/v1/models" || echo "unreachable"

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu --format=csv

echo
echo "=== last server log lines ==="
tail -n 20 "${DEPLOY_DIR}/logs/qwen3_omni_server.log" 2>/dev/null || echo "(no log)"
