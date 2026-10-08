# FrugalStream Migration Snapshot

This repository is a migration snapshot of the FrugalStream × Qwen3-Omni
experiment engineering on the old server. It contains code, scripts, configs,
experiment definitions, reports, and small results. It does NOT contain model
weights, datasets, videos, large caches, or credentials.

## Current project status

- Frozen V3 has completed MiniCPM and Qwen cross-backend validation.
- Qwen native visual-token audit is complete.
- The core Qwen four-point (2x2) experiment has existing results for points A/B.
- C/D true-inference scripts are prepared but still need long-context GPU execution.

See `EXPERIMENT_STATUS.md` for the exact numbers.

## Directory overview

| Directory | Purpose |
|-----------|---------|
| `qwen3_omni_deploy/` | Qwen3-Omni vLLM-Omni deployment scripts, configs, reports, tests |
| `focalstream_qwen3_crossmodel/` | Qwen cross-model A/B/V3/V4 results, evaluation scripts, manifests, predictions |
| `frugalstream_qwen3_v3_generalization/` | Strict frozen V3 spec + few-shot V3 calibration |
| `frugalstream_qwen3_native_token_audit/` | Qwen native visual-token accounting + forensic audit |
| `frugalstream_qwen3_four_point/` | 2x2 four-point experiment spec + C/D runner |
| `focalstream_qwen3_fewshot_calibration/` | QRE threshold few-shot calibration experiment |
| `frozen_v4_artifacts/` | Frozen MiniCPM V3/V4 configs (read-only copies) |
| `tools/` | Frozen evaluator and helper scripts |
| `environment/` | Environment metadata for reconstruction |

## What is NOT included

- Model weights (`Qwen3-Omni-30B-A3B-Instruct`, `MiniCPM-o-4.5`, ...)
- Datasets (`OVO-Bench`, `StreamingBench`, `RVS`)
- Raw videos
- Large caches (raw request caches, HuggingFace/vLLM caches, uv caches)
- Huge prediction dumps beyond the small per-QA JSONL kept in `focalstream_qwen3_crossmodel/predictions/`
- Credentials / tokens / SSH keys / `.env` files

## Required external assets

Re-download on the new server:

- `Qwen/Qwen3-Omni-30B-A3B-Instruct` (Qwen official model)
- `MiniCPM-o-4.5` (only if the MiniCPM side is needed again)
- `OVO-Bench` source + full videos (see `/root/OVO-Bench-src` and `/root/OVO-Bench-full` paths used by the original scripts)
- `StreamingBench` / `RVS` datasets (only for the separate streaming project, not part of the core four-point experiment)

Use the download sources referenced in the original scripts where available.
Do not hardcode old-server absolute paths; adjust `BASE_DIR`-style constants per environment.

## Environment reconstruction

1. Qwen vLLM serving env (GPU):
   - Recreate the uv venv from `environment/pip_freeze_qwen_venv.txt`.
   - Key packages: `vllm==0.30.0`, `vllm-omni==0.30.0`, `transformers==5.14.1`,
     `torch==2.13.0+cu132`, `qwen-omni-utils`.
   - Blackwell workaround: `export VLLM_USE_FLASHINFER_SAMPLER=0`.
2. Offline eval/accounting env:
   - Recreate conda env `minicpmo45` from
     `environment/conda_environment_minicpmo45.yml` (or
     `environment/pip_freeze_minicpmo45.txt`).
3. Copy `environment/relevant_env.example` and fill in your own paths/tokens.

## Expected directory layout (adjust paths on new server)

```
/workspace/models/Qwen3-Omni-30B-A3B-Instruct   # re-downloaded
/workspace/datasets/                            # re-downloaded OVO/StreamingBench data
/workspace/frugalstream_migration_repo/         # this repo (cloned)
```

Path constants to update when cloning:
- `focalstream_qwen3_crossmodel/scripts/qwen_common.py` (`BASE_DIR`, `HOST`, `PORT`, `MODEL`)
- `frugalstream_qwen3_four_point/scripts/run_cd.py` (`BASE`, `CROSS`, `AUDIT`, `HOST`, `PORT`)
- `frugalstream_qwen3_native_token_audit/scripts/compute_native_accounting.py`
  (`BASE`, `CROSS`, model/video paths)

## Continue four-point experiment

- **A Adapted Dense**: DONE (Raw 58.1987, Macro 60.0872, n=2909)
- **B Adapted V3**: DONE (Raw 58.8862, Macro 61.0761, n=2909)
- **C True Dense**: PENDING (requires long-context GPU; `max_model_len=65536`)
- **D True V3**: PENDING (requires long-context GPU; `max_model_len=65536`)

Exact 2x2 definitions: `frugalstream_qwen3_four_point/configs/FOUR_POINT_EXPERIMENT_SPEC.md`.

Runner: `frugalstream_qwen3_four_point/scripts/run_cd.py`.

Planned start on a GPU with enough free memory (>= ~85 GiB):

```bash
cd /workspace/qwen3_omni_deploy
QWEN3_OMNI_PORT=8092 QWEN3_OMNI_MAX_MODEL_LEN=65536 \
QWEN3_OMNI_MAX_NUM_BATCHED_TOKENS=16384 bash scripts/start_qwen3_omni_server.sh

cd /workspace/frugalstream_qwen3_four_point
QWEN_PORT=8092 /opt/conda/envs/minicpmo45/bin/python scripts/run_cd.py \
  --methods true_v3,true_dense --workers 3
```

Do NOT compress visual inputs to make C/D fit; requests that exceed model context
must be marked infeasible (the runner already does this).
