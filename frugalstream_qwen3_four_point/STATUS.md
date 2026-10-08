# FOUR-POINT QWEN EXPERIMENT — STATUS (blocked on GPU)

## Completed
1. A/B verified from existing artifacts (0 new calls):
   - A Adapted Dense: Raw 58.1987, Macro 60.0872, n=2909
   - B Adapted V3:    Raw 58.8862, Macro 61.0761, n=2909
2. `configs/FOUR_POINT_EXPERIMENT_SPEC.md` written (A/B/C/D definitions).
3. `reports/QWEN_CONTEXT_CAPABILITY_AUDIT.md` written.
4. C/D context requirements recovered from native accounting CSV.
5. C/D generation runner written: `scripts/run_cd.py` (no resize/no fit/adaptive-drop for C/D).

## GPU blocker (requires host action)
- Old 32k Qwen server stopped safely (PID 516282 / port 8091).
- The GPU was already at 100% util / 97143 MiB used before this task (monitor log since 14:52).
- Graceful shutdown force-killed `StageEngineCoreProc_stage0_replica0` (PID 516444),
  leaking the worker's ~88.5 GB CUDA context. GPU now: 97141 MiB used, 74 MiB free, 100% util.
- `torch.cuda` cannot allocate even 1 tensor (CUDA OOM).
- In-container recovery impossible:
  - `nvidia-smi -r` / `-r bus` -> "GPU is the primary GPU" (fb0 bound to 0000:02:00.0).
  - `/sys` is mounted read-only -> cannot unbind vtconsole/fb0 or PCI reset.

## Needed from host
Reboot the host, OR from the host run:
  `sudo nvidia-smi --gpu-reset -i 0`
If the reset still says primary GPU, first detach the NVIDIA framebuffer / console
and then reset, or reboot.

## Resume after GPU is free
```
# start 64k server (same weights/pipeline/decoding; only max_model_len changes)
cd /workspace/qwen3_omni_deploy
QWEN3_OMNI_PORT=8092 QWEN3_OMNI_MAX_MODEL_LEN=65536 \
QWEN3_OMNI_MAX_NUM_BATCHED_TOKENS=16384 bash scripts/start_qwen3_omni_server.sh

# smoke test (pick qa near the 64k boundary; see accounting CSV)
cd /workspace/frugalstream_qwen3_four_point
/opt/conda/envs/minicpmo45/bin/python scripts/run_cd.py --methods true_v3 --limit 5 --qids <comma-list>

# full C/D
QWEN_PORT=8092 /opt/conda/envs/minicpmo45/bin/python scripts/run_cd.py --methods true_v3,true_dense --workers 3
```
Then aggregate full-dev + strict common-support and write the final report.
