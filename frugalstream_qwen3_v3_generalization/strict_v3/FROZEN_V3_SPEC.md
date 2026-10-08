# FROZEN V3 SPEC

Recovered from frozen artifacts only:
- `/root/OVO-Bench-V4-backup/reports/V3_VS_V4_FULL_DEVELOPMENT_EVALUATION.md`
- `/root/OVO-Bench-V4-backup/V4_VERSION_INFO.json`
- `/root/OVO-Bench-V4-backup/configs/binary_gate_frozen_config.json`
- `/root/OVO-Bench-V4-backup/results/OVO-Bench-full-results/backups/v3_final_20260922/README_v3.md`
- `/root/OVO-Bench-V4-backup/results/OVO-Bench-full-results/backups/v3_final_20260922/minicpmo45_full_v3_binary_gate_summary.json`
- `/root/eval_minicpmo45_ovobench_full_v2.py`
- `/root/OVO-Bench-V4-backup/code/eval_minicpmo45_ovobench_full_hr1.py`
- `/root/OVO-Bench-full/run_full_v2.sh`
- `/workspace/focalstream_qwen3_crossmodel/audit/QWEN_INPUT_SEMANTICS_FOR_CROSSMODEL.md`

## 1. System structure
- Backward / Forward QA: pure V2 stream, never query-time refill.
- RTVP QA: V2 stream plus binary query-time gate:
  - `risk(Q) < tau` -> NO_REFILL -> use V2 answer
  - `risk(Q) >= tau` -> HR-1_REFILL -> use Full-HR-1 answer

## 2. V2 streaming core
- Candidate temporal sampling: 1 FPS causal integer seconds.
- Causal endpoint: `target_seconds = clamp_seconds(ceil(query_timestamp), video_duration, total_frames)`.
  - No frame timestamp exceeds the query timestamp in the Qwen adaptation.
- Temporal selector: frozen Decision Tree.
  - Path: `/root/OVO-Bench-V4-backup/configs/minicpmo45_token_v1_decision_tree.joblib`
  - Features: `change_since_last_sent, change_prev_1s, ema_change, gap_since_last_sent, norm_time, video_duration`.
- Safety Anchor: `MAX_GAP_SECONDS = 2`.
  - Force send when `candidate_seconds == 0` or `steps_since_sent >= 2`.
- CSC resolution: `448x252`.
- MiniCPM `max_slice_nums = 7`; resized 448x252 frames usually produce `1 slice/frame` and `64 visual tokens/frame`.
- Frame order: chronological, long-term stream first.

## 3. Query Risk Scorer
- Input: RTVP question text only.
- Representation: TF-IDF
  - `ngram_range=(1, 2)`
  - `min_df=2`
  - `max_features=5000`
  - `sublinear_tf=True`
  - `lowercase=True`
- Model: `LogisticRegression`
  - `class_weight='balanced'`
  - `C=1.0`
  - `max_iter=2000`
  - `random_state=42`
- Output: `risk(Q) = P(BENEFICIAL_REFILL | Q)`.
- Training/evaluation protocol: strict video-level GroupKFold OOF.
- Frozen risk source: `stage2a_dev_cv_predictions.csv`, column `tfidf_score`.

## 4. Frozen V3 threshold
- `tau = 0.2307592643` (display ~0.231).
- Rule: `risk < tau -> NO_REFILL`; `risk >= tau -> HR-1_REFILL`.
- Threshold selected only from Development OOF:
  - Primary: WC Capture >= 90%, minimize additional input-token overhead.
  - Tie-break: higher official RTVP Macro, then fewer triggered CW.

## 5. Query-time HR-1 refill
- Triggered only for RTVP QA with `risk >= tau`.
- Original MiniCPM window:
  - `latest = floor(query_timestamp)`
  - `seconds = [max(0, latest - 1), latest]`
  - `window = 1 s`
  - `F = 1 FPS`
  - two original-resolution frames.
- Qwen no-future adaptation:
  - Causal window remains `[floor(t)-1, floor(t)]`.
  - Same two-frame set; no future frame leakage.
- Resolution: original source resolution, no crop, full spatial coverage.
- Frame order: long-term V2 CSC frames first, then HR-1 refill frames, then question prompt.
- Backward / Forward never trigger HR-1.

## 6. Prompt and decoding
- Prompt: frozen OVO benchmark prompt from the full annotation expansion.
  - Do not change question semantics, options, or answer instructions.
- Decoding:
  - Qwen generation frozen config: temperature `0.0`, top_p `1.0`, max_tokens `256`, seed `0`, modalities text.
  - MiniCPM original V3: `do_sample=False`, `max_new_tokens=128`, `generate_audio=False`.
- Answer parser: frozen `V2.score_event` from `/root/eval_minicpmo45_ovobench_full_v2.py`.

## 7. Qwen backend context-cap adaptation
This is a backend-specific safety adaptation, not a V3 method change.
- Long-term context is capped by Qwen `max_model_len=32768`.
- `dense`, `v2`, and query-time-refill methods use most-recent long-frame cap under the fixed budget.
- Dense long-term frames are encoded at `448x252` for Qwen compatibility.
- Refill frames remain at their specified V3 resolution/crop.
- If a request hits 400/500 length errors, most-recent long frames are dropped adaptively; refill frames remain intact.

## 8. Frozen parameters summary
| Parameter | Frozen V3 value |
|---|---:|
| PTS candidate FPS | 1 |
| PTS selector | Decision Tree |
| Safety Anchor max gap | 2 s |
| CSC resize | 448x252 |
| MiniCPM max_slice_nums | 7 (actual 1 slice/frame for CSC) |
| Query refill scope | RTVP only |
| Query refill risk source | question-only TF-IDF logistic OOF |
| Query refill threshold | 0.2307592643 |
| Query refill window W | 1 s |
| Query refill FPS / frame count | 1 FPS / 2 frames |
| Query refill resolution | original, no resize |
| Query refill spatial coverage | full frame |
| Refill frame order | after V2 CSC, before prompt |
| Prompt | frozen OVO prompt |
| Answer parser | frozen V2.score_event |
