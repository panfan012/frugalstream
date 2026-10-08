#!/usr/bin/env bash
# Stop ONLY the Qwen3-Omni server recorded in qwen3_server.pid.
# Never uses pkill/killall; never touches other processes (e.g. MERA).
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${DEPLOY_DIR}/qwen3_server.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "No PID file (${PID_FILE}). Nothing to stop."
  exit 0
fi

PID="$(cat "${PID_FILE}")"
if [[ -z "${PID}" ]] || ! kill -0 "${PID}" 2>/dev/null; then
  echo "Process ${PID} not running. Removing stale PID file."
  rm -f "${PID_FILE}" "${DEPLOY_DIR}/qwen3_server.port"
  exit 0
fi

echo "Sending SIGTERM to Qwen3-Omni server PID ${PID} ..."
kill "${PID}"

# Wait briefly for graceful shutdown, then escalate only to this PID.
for _ in $(seq 1 30); do
  if ! kill -0 "${PID}" 2>/dev/null; then
    echo "Stopped PID ${PID}."
    rm -f "${PID_FILE}" "${DEPLOY_DIR}/qwen3_server.port"
    exit 0
  fi
  sleep 1
done

echo "PID ${PID} did not stop within 30s; sending SIGKILL to ${PID} only."
kill -9 "${PID}" 2>/dev/null || true
rm -f "${PID_FILE}" "${DEPLOY_DIR}/qwen3_server.port"
echo "Done."
