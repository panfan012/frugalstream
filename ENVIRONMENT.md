# ENVIRONMENT

## Two runtime environments were used

1. **Qwen3-Omni vLLM server** (GPU serving)
   - Isolated uv venv: `/workspace/qwen3_omni_deploy/.venv`
   - Python 3.12, vLLM 0.30.0, vLLM-Omni 0.30.0, transformers 5.14.1,
     torch 2.13.0+cu132
   - See `environment/pip_freeze_qwen_venv.txt`

2. **Offline evaluation / accounting** (CPU, no GPU)
   - Conda env: `minicpmo45` (`/opt/conda/envs/minicpmo45`)
   - Has decord, opencv, PIL, numpy, pandas, matplotlib
   - See `environment/pip_freeze_minicpmo45.txt` and
     `environment/conda_environment_minicpmo45.yml`

## Reconstruction
- `environment/conda_env_list.txt` — conda env inventory
- `environment/conda_environment_minicpmo45.yml` — recreate minicpmo45 env
- `environment/pip_freeze_qwen_venv.txt` — recreate Qwen vLLM env
- `environment/relevant_env.example` — env vars with all secret values empty

## Model / hardware
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB
- Qwen model: `Qwen3-Omni-30B-A3B-Instruct` (NOT included in this repo)
- Blackwell workaround: `VLLM_USE_FLASHINFER_SAMPLER=0`
- See `environment/nvidia_smi_old_server.txt` and
  `qwen3_omni_deploy/QWEN3_OMNI_DEPLOYMENT_REPORT.md`
