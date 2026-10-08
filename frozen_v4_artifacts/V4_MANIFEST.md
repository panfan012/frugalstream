# V4 Frozen Snapshot

Snapshot time: 2026-09-29T09:35:00Z

Project root: /root/OVO-Bench-full (plus /root/OVO-Bench-mini, /root/OVO-Bench-src)

Target model: MiniCPM-o 4.5 (black-box backend)

Dataset: OVO-Bench Development (RTVP)

## Version Status

V4
FROZEN

## V4 Definition

Proactive Temporal Compression
+
Query-Adaptive Spatial Fidelity Management

## Final Frozen Policy

Temporal: frozen V3 Decision Tree
Spatial: C070 vs Full HR
Router: frozen query-risk
Threshold: 0.3554

Selection:

risk <= 0.3554
→ C070

risk > 0.3554
→ Full HR

One final target-VLM generation.
No iterative backend probing.

C070 crop definition: center crop, width ratio 0.70, height ratio 0.70 (49% source area).

## Strict OOF Validation

5-fold GroupKFold by video_id

203 videos
711 QA

Fold thresholds:
[0.3554, 0.3554, 0.3542, 0.3554, 0.3554]

Mean:
0.3552

Std:
0.0005

OOF Raw:
78.1997

OOF Macro:
78.6406

WC:
64/71

CW harm:
4

Full HR escalation:
43.46%

Input tokens:
11,851,282

Additional visual tokens:
430,848

Verdict:
STRONG PASS

## V3 vs V4

V3:
Raw 76.3713
Macro 76.3563
WC 64/71
CW harm 10
Input Tokens 11,891,981
Additional visual tokens 470,464

V4 Frozen:
Raw 78.1997
Macro 78.6406
WC 64/71
CW harm 4
Input Tokens 11,851,282
Additional visual tokens 430,848

Delta:
Raw +1.8284 pp
Macro +2.2842 pp
WC no loss
CW harm -60%
Input tokens -40,699
Additional visual tokens -39,616

## Key Findings

1. V4 strictly outperforms V3 in OOF Raw and Macro accuracy.
2. V4 preserves exactly the same WC recovery as V3: 64/71.
3. V4 reduces CW harm from 10 to 4.
4. V4 slightly reduces total input tokens.
5. The main V4 gain is not massive total-token reduction, but better allocation of visual fidelity.
6. Full-resolution visual evidence can harm answers.
7. Focused fidelity suppresses visual clutter and reduces harmful transitions.
8. Full HR is reserved for higher-risk queries.

## Frozen Components

- temporal Decision Tree: /root/OVO-Bench-mini/results/minicpmo45_token_v1_decision_tree.joblib
- binary gate config: /root/OVO-Bench-full/results/binary_gate_frozen_config.json
- query-only gate: /root/OVO-Bench-full/results/query_only_binary_gate_v1_frozen.json
- query-risk source: /root/OVO-Bench-full/results/stage2a_dev_cv_predictions.csv (tfidf_score)
- V4 frozen router config: /root/OVO-Bench-full/results/v4_frozen_router_config.json
- C070 crop definition: center crop, edge ratio 0.70, area 49%, frame [t-1, t]
- model inference config: max_slice_nums=7, torch bfloat16, MiniCPM-o-4_5

## Important Reports

- /root/V4_STRICT_OOF_VALIDATION_REPORT.md
- /root/V3_VS_V4_FULL_DEVELOPMENT_EVALUATION.md
- /root/V4_C070_FIRST_SELECTIVE_FULLHR_ESCALATION_REPORT.md
- /root/QUERY_ADAPTIVE_SPATIAL_BUDGET_ROUTING_LONGRUN_REPORT.md
- /root/OVO_BENCH_CENTER_CROP_OVERNIGHT_ANALYSIS.md
- /root/QUERY_CONDITIONED_SPATIAL_ACQUISITION_REPORT.md
- /root/SPATIAL_FIDELITY_STABILITY_PROBE_REPORT.md
- /root/QUERY_ONLY_BINARY_GATE_V1_FROZEN.md

## Important Result Files (V4 Frozen)

- v4_frozen_router_config.json
- v4_oof_predictions.csv
- v4_oof_fold_assignments.csv
- v4_oof_fold_metrics.csv
- v4_oof_threshold_sweeps.csv
- v4_oof_fullhr_needed_analysis.csv
- v4_oof_bootstrap.csv
- v4_oof_final_metrics.json
- v4_full_dev_c070_predictions.csv
- v4_c070_fullhr_oracle.csv
- v4_state_router_oof_predictions.csv
- v4_fullhr_needed_labels.csv
- v4_hybrid_policy_comparison.csv
- v3_vs_v4_full_dev_paired_outcomes.csv
- v3_vs_v4_subtype_comparison.csv
- v3_vs_v4_video_bootstrap.csv

## Large External Assets Not Copied

- /root/models/MiniCPM-o-4_5
- /root/OVO-Bench-full/data/src_videos
- /root/.cache/huggingface
- /root/OVO-Bench-V4-backup/results/OVO-Bench-full-results (small result files only; no video/model weights)
