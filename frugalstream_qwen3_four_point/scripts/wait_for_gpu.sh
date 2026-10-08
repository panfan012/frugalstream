#!/usr/bin/env bash
# Poll GPU until enough free memory is available for the 64k Qwen server,
# then write a marker file. Runs in background.
set -u
LOG=/workspace/frugalstream_qwen3_four_point/logs/gpu_wait.log
MARK=/workspace/frugalstream_qwen3_four_point/GPU_FREE.marker
THRESHOLD_MIB=85000
rm -f "$MARK"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] wait_for_gpu started threshold=${THRESHOLD_MIB}MiB" >> "$LOG"
while true; do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "$ts free=${free_mib}MiB apps=${apps}" >> "$LOG"
  if [ -n "$free_mib" ] && [ "$free_mib" -ge "$THRESHOLD_MIB" ] && [ "$apps" -le 0 ]; then
    echo "$ts GPU_FREE free=${free_mib}MiB apps=${apps}" >> "$LOG"
    echo "$ts" > "$MARK"
    exit 0
  fi
  sleep 45
done
