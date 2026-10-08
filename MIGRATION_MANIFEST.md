# MIGRATION MANIFEST

Large / non-Git artifacts that were NOT uploaded, and how to regenerate them.

| Original path | Approx size | Contents | Regenerable? | How to regenerate / download |
|---------------|------------:|----------|--------------|------------------------------|
| /workspace/qwen3_omni_deploy/.venv | 8.4G | Python 3.12 venv for vLLM-Omni serving | YES | Recreate from `environment/pip_freeze_qwen_venv.txt` (`uv venv` + install vllm==0.30.0 vllm-omni==0.30.0 transformers==5.14.1 torch==2.13.0+cu132 etc.) |
| /workspace/qwen3_omni_deploy/.uv_cache | ~570M | uv package download cache | YES | `uv pip install` regenerates automatically |
| /workspace/qwen3_omni_deploy/logs | 32M | server logs | YES (not needed) | regenerate on next server run |
| /workspace/focalstream_qwen3_crossmodel/cache/qwen_request_cache.jsonl | 16M | raw Qwen HTTP request cache | YES | re-run `scripts/run_qwen_main.py` |
| /workspace/focalstream_qwen3_crossmodel/logs | 200K | run logs | YES (not needed) | regenerate on re-run |
| /workspace/models/Qwen3-Omni-30B-A3B-Instruct | 66G | model weights | NO (download) | `huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct` (official Qwen repo) |
| /root/models/MiniCPM-o-4_5 | large | MiniCPM model weights | NO (download) | re-download official MiniCPM-o-4.5 weights |
| /workspace/datasets | 463G | OVO videos + StreamingBench/RVS datasets | NO (download) | re-download OVO-Bench and StreamingBench/RVS data |
| /root/OVO-Bench-full | 44G | OVO source videos | NO (download) | re-download OVO-Bench full videos |
| /root/OVO-Bench-src | 13M | OVO benchmark code + annotations | NO (download) | re-download OVO-Bench source |
| /root/OVO-Bench-V4-backup | 31M | frozen MiniCPM V3/V4 artifacts | PARTIAL | only small configs copied here; full backup regenerated from MiniCPM runs |

## What WAS uploaded (small, needed to continue)
- `frozen_v4_artifacts/configs/` — frozen V3/V4 config JSONs + decision tree joblib
- `tools/eval_minicpmo45_ovobench_full_v2.py` — frozen evaluator
- All project scripts/configs/reports/manifests/small results (see each project dir)
- `focalstream_qwen3_crossmodel/manifests/` — frozen visual inputs + dev/rtvp QA manifests
- `focalstream_qwen3_crossmodel/predictions/` — Qwen A/B/V3/V4 predictions (3.2M)
