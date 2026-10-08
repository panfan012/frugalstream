# V4 Frozen Reproducibility

Snapshot: 2026-09-29 (UTC)

## Environment
- Target model path: /root/models/MiniCPM-o-4_5
- Conda env: /opt/conda/envs/minicpmo45
- Python: 3.10.21
- torch: 2.8.0+cu128
- transformers: 4.51.0
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB

## Frozen models / configs
- temporal Decision Tree: /root/OVO-Bench-mini/results/minicpmo45_token_v1_decision_tree.joblib
- query-risk predictions: /root/OVO-Bench-full/results/stage2a_dev_cv_predictions.csv
- query-only gate frozen: /root/OVO-Bench-full/results/query_only_binary_gate_v1_frozen.json
- V4 router config: /root/OVO-Bench-full/results/v4_frozen_router_config.json
- OVO Development split: /root/OVO-Bench-full/results/stage2a_development_set.csv

## Data paths (large, NOT copied)
- source videos: /root/OVO-Bench-full/data/src_videos
- HuggingFace cache: /root/.cache/huggingface

## Inference / evaluation scripts
- Backend inference: /root/OVO-Bench-src/inference.py
- C070 full-dev eval: /root/v4_full_dev_c070_eval.py
- V3 vs V4 task-1 analysis: /root/v4_analyze_task1.py
- Selective Full-HR task-2 analysis: /root/v4_analyze_task2.py
- Strict OOF: /root/v4_strict_oof.py, /root/v4_strict_oof_v2.py
- Bootstrap fix: /root/v4_fix_bootstrap.py
- Evaluation parser: /root/OVO-Bench-src/score.py

## Git
- Repo: /root (git)
- HEAD: 259f255e7c29cf8094e08769c89bd06dfb146b0e
- Branch: main
- Working tree has untracked results/code (not committed; not pushed; no tag created).
