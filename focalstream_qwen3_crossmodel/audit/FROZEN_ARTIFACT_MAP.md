# FROZEN ARTIFACT MAP

Recovered read-only from existing FocalStream/V4 artifacts. No frozen file was modified.

## Split / QA sources
- Dataset split file: `/root/OVO-Bench-full/results/stage2a_split_manifest.json`
- Full annotation: `/root/OVO-Bench-full/annotations/ovo_bench_new.json`
- RTVP Development strict OOF: `/root/OVO-Bench-full/results/v4_oof_predictions.csv`
- Final Test membership source: `/root/OVO-Bench-full/results/stage2a_final_test_set.csv`
  - final test = 126 RTVP QA / 34 videos.
- Full Dev QA list: reconstructed by expanding the full annotation with frozen
  `eval_minicpmo45_ovobench_full_v2.expand_events`, then excluding final-test keys.
  - result: `manifests/dev_2909_qa.jsonl`
- RTVP 711 QA list: `manifests/rtvp_oof_711_qa.jsonl`

## Version configs
- Dense baseline implementation: `/root/eval_minicpmo45_ovobench_streaming.py`
  - all causal 1-fps seconds `[0, ceil(t)-1]`, original resolution, no crop.
- V1 config: `/root/OVO-Bench-full/results/minicpmo45_full_v1_dt_orig_summary.json`
  - decision_tree + SafetyAnchor MAX_GAP=2, original resolution, max_slice=7.
- V2 config: `/root/OVO-Bench-full/run_full_v2.sh`
  - decision_tree + SafetyAnchor MAX_GAP=2, `resize_mode=448x252`, max_slice=7
    (actual frozen MiniCPM slices/frame = 1, visual tokens/frame = 64).
- V3 config: `/root/OVO-Bench-full/results/binary_gate_frozen_config.json`
  - Full-HR-1 refill for RTVP when `tfidf_score >= 0.2307592643`, else V2.
- V4 config: `/root/OVO-Bench-V4-backup/configs/v4_frozen_router_config.json`
  - `risk > 0.3554 -> Full HR-1`, else `C070-1`.
- Frozen router config copy: `/root/OVO-Bench-full/results/v4_frozen_router_config.json`

## Frozen implementations
- PTS implementation: decision tree
  `/root/OVO-Bench-V4-backup/configs/minicpmo45_token_v1_decision_tree.joblib`
  (also `/root/OVO-Bench-mini/results/minicpmo45_token_v1_decision_tree.joblib`)
  - features: `change_since_last_sent, change_prev_1s, ema_change, gap_since_last_sent, norm_time, video_duration`.
  - selector code: `/root/eval_minicpmo45_ovobench_full_v2.py` lines ~574-675.
- Safety implementation: `MAX_GAP_SECONDS=2`, send if `candidate_seconds==0` or
  `steps_since_sent>=2`.
- CSC implementation: V2 = same selected frames as V1, resized to `448x252`,
  max_slice=7 in MiniCPM (actual 1 slice/frame), from `run_full_v2.sh`.
- NQEW implementation:
  - MiniCPM frozen FCV `HR-1` used `latest=floor(query_timestamp)`, window
    `[max(0, latest-1), latest]` (`eval_minicpmo45_ovobench_full_hr1.py`).
  - MiniCPM frozen FFV `C070_1` used `target=ceil(query_timestamp)`, window
    `[target-1, target]` (`v4_full_dev_c070_eval.py`).
  - **Qwen causal mapping correction**: both FFV and FCV use the causal window
    `[max(0, floor(t)-1), floor(t)]` to satisfy the no-future-leakage rule. See
    `QWEN_INPUT_SEMANTICS_FOR_CROSSMODEL.md`.
- FFV implementation: center crop edge ratio 0.70 (area 0.49), original frame
  resolution, from `v4_full_dev_c070_eval.py` `center_bbox`.
- FCV implementation: full original-resolution `HR-1`, no crop, from
  `eval_minicpmo45_ovobench_full_hr1.py`.
- QRE score source: `/root/OVO-Bench-full/results/stage2a_dev_cv_predictions.csv`
  column `tfidf_score`.
- QRE model/config: `/root/OVO-Bench-full/results/stage2a_query_controller.joblib`
  and `stage2a_frozen_controller_config.json`.
- OOF fold source: `/root/OVO-Bench-full/results/v4_oof_fold_assignments.csv`
  and `stage2a_split_manifest.json` (203 videos / 711 QA, 5 GroupKFold by video_id).

## Frozen tau
- V4: `0.3554` (`risk > tau -> FullHR/FCV, else C070/FFV`).
- V3 (historical reference): `0.2307592643` (`risk >= tau -> FullHR/FCV, else V2`).
- Strict OOF per-fold thresholds are stored in
  `/root/OVO-Bench-full/results/v4_oof_fold_metrics.csv`; folded actions are in
  `v4_oof_predictions.csv` column `acc_action`.

## Answer evaluation script
- `/root/OVO-Bench-src/score.py` (original OVO parser)
- MiniCPM frozen scripts also use `V2.score_event` in
  `/root/eval_minicpmo45_ovobench_full_v2.py`.

## Ground truth field
- Full annotation: `ground_truth` in expanded events; underlying source uses
  `gt`/`test_info` in `ovo_bench_new.json`.
- RTVP OOF: `ground_truth` from `stage2a_development_set.csv`.
