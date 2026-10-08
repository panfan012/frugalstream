# FocalStream × Qwen3-Omni Cross-Backend Evaluation — Final Report

## 1. Executive Summary
This experiment evaluates whether the frozen FocalStream pipeline (developed and
validated on MiniCPM-o 4.5) transfers to Qwen3-Omni-30B-A3B-Instruct with **zero
retuning**. All FocalStream components (PTS, Safety Anchor, CSC, NQEW, FFV/FCV,
QRE risk scores, RGFR thresholds) are read from frozen artifacts and applied
unchanged. The only adaptation is a documented Qwen 32k-context input-cap rule
(see Section 16), because Qwen's explicit-frame API cannot hold the very long
causal frame histories that MiniCPM streaming prefill supports.

Main numbers (Development only; Final Test untouched):
- Qwen Dense macro: 60.0872 | Qwen FocalStream macro: 60.7948 | Δ: +0.7076 pp
- Qwen RTVP strict OOF: Fixed FFV 74.6177 / Fixed FCV 75.7020 / Random matched 75.0947 ± 0.4095 / FocalStream 74.8517
- Router gain vs random: -0.2430 pp; vs best fixed: -0.8503 pp
- FFV→FCV harmful flips (VOH): 14 (beneficial WC: 20)

## 2. FocalStream Problem Setting
Continuous video arrives before an unknown future query. FocalStream manages a
strictly black-box Video-LLM's visual inputs (frame selection, resolution, crop,
query-time evidence) to trade off accuracy against visual input cost. No access
to hidden states, attention, logits, ViT/KV internals, or weights.

## 3. Original MiniCPM Backend
MiniCPM-o 4.5 with streaming_prefill + streaming_generate, 1 fps, MAX_SLICE_NUMS=7
(Dense/V1) or 1 (V2 CSC 448x252), text-only output. Frozen reference: Dense macro
63.0881, FocalStream (V4) macro 63.8961, full-dev input saving 80.33%, RTVP strict
OOF router gain vs random +1.796 pp.

## 4. Qwen3-Omni Backend
Qwen3-Omni-30B-A3B-Instruct served by vLLM-Omni (`qwen3_omni_moe_thinker_only`,
text-only, no talker/audio), endpoint 127.0.0.1:8091, `max_model_len=32768`,
`VLLM_USE_FLASHINFER_SAMPLER=0`. Main accuracy path is explicit-frame
`/v1/chat/completions` with JPEG `image_url` parts.

## 5. Dataset and Splits
- Full annotation 3035 QA = 2909 Development + 126 Final Test.
- Development: 2909 QA / 610 videos (Backward 631, RTVP 837, Forward 1567).
- RTVP strict OOF: 711 QA / 203 videos.
- Final Test (126 QA / 34 videos) was NOT run, evaluated, or used.

## 6. Strict Black-box Setting
Only input frames/resolution/crop/order are controlled. Qwen never receives video
paths, audio, future frames, timestamp metadata, or MiniCPM-only extras.

## 7. Frozen FocalStream Pipeline
Continuous video → PTS → Safety Anchor → CSC → (query arrives) → QRE → RGFR →
FFV/FCV query evidence → one text generation.

## 8. PTS
Frozen decision tree on 64x64 grayscale change features. Not retrained.

## 9. Safety Anchor
MAX_GAP_SECONDS=2 force-send rule, frozen.

## 10. CSC
448x252 compact streaming context, frozen. (The dominant MiniCPM token saver.)

## 11. NQEW
Near-query window; for Qwen, causal `[floor(t)-1, floor(t)]` (input-mapping
correction documented in `audit/QWEN_INPUT_SEMANTICS_FOR_CROSSMODEL.md`).

## 12. FFV / FCV
FFV = center crop 0.70 original-resolution query frames; FCV = full
original-resolution query frames. Both on top of the V2 CSC context.

## 13. QRE + RGFR
Frozen question-only TF-IDF risk scores (stage2a OOF CV). V3 tau=0.2307592643,
V4 tau=0.3554. Qwen does not participate in QRE training.

## 14. Zero-Retuning Cross-Backend Protocol
No PTS/Safety/CSC/FFV/FCV/crop/tau/prompt/routing changes based on Qwen results.

## 15. Exact Frozen Input Reconstruction
Recovered from `/root/eval_minicpmo45_ovobench_full_v2.py`,
`/root/v4_full_dev_c070_eval.py`, `/root/eval_minicpmo45_ovobench_full_hr1.py`,
and frozen decision-tree/router configs. Cumulative per-video streaming context
was validated against frozen MiniCPM `frames_seen` (exact match on sampled QAs).

## 16. Qwen 32k Context-Cap Adaptation
Qwen's explicit-frame API has `max_model_len=32768`; frozen original-resolution
Dense (and very long V2 histories) cannot be represented. A uniform, documented
rule keeps the query refill intact and keeps the most recent long-term frames
until the token budget fits. **Qwen-compatible Dense** encodes the causal 1-fps
frame set at 448x252 (same compact spatial format as CSC) so the Dense-vs-Focal
contrast isolates temporal sparsification. This is an input-cap adaptation, not a
method retune. Details in `audit/QWEN_INPUT_SEMANTICS_FOR_CROSSMODEL.md`.

## 17. Qwen Full Development Results
| Method | N | Raw | Macro |
|---|---:|---:|---:|
| Dense | 2909 | 58.1987 | 60.0872 |
| V2 | 2909 | 58.3362 | 60.3260 |
| V3 | 2909 | 58.8862 | 61.0761 |
| V4 / FocalStream | 2909 | 58.6800 | 60.7948 |

V2 − Dense macro: +0.2388 pp; V3 − Dense:
+0.9889 pp; V4 − Dense: +0.7076 pp.

## 18. Accuracy Preservation
FocalStream vs Dense: +0.7076 pp macro / +0.4813 pp raw
(bootstrap CI: +0.7099 [-0.1081, +1.6139]).

## 19. Visual Input Saving
See `results/qwen_resource_accounting_frozenplan.csv` (method-specified) and
`results/qwen_resource_accounting.csv` (Qwen-visible). Qwen-native token saving is
reported from `usage.prompt_tokens` where exposed.

## 20. Latency / Resource Results
Latency (client E2E, warm server): Dense median 2.2349s
/ p90 7.3752s; FocalStream median
2.0138s / p90 7.3112s.
Peak observed VRAM: see `results/qwen_resource_accounting.csv` and GPU logs.

## 21. RTVP Strict OOF Routing Results
| Policy | Raw | Macro |
|---|---:|---:|
| Fixed FFV | 73.9803 | 74.6177 |
| Fixed FCV | 74.8242 | 75.7020 |
| Random matched (mean) | 74.3489 | 75.0947 ± 0.4095 |
| FocalStream | 74.1210 | 74.8517 |

## 22. Fixed FFV / Fixed FCV
See Section 21. FFV under-provisions full-context evidence; FCV over-provisions.

## 23. Matched Random Router
1000 seeds; escalation 43.4599% (matching frozen OOF count
309). Distribution: `figures/qwen_random_router_distribution.png`.

## 24. Frozen QRE/RGFR Transfer
FocalStream router gain vs random: -0.2430 pp; vs best fixed:
-0.8503 pp.

## 25. Visual Over-Provisioning Harm
FFV→FCV transitions (711 QA): WC=20 (beneficial escalation), CW=14
(harmful escalation / VOH), CC=512, WW=165.

## 26. MiniCPM vs Qwen Cross-Backend Comparison
| Metric | MiniCPM-o 4.5 | Qwen3-Omni |
|---|---:|---:|
| Full-dev FocalStream macro | 63.8961 | 60.7948 |
| Full-dev Dense macro | 63.0881 | 60.0872 |
| Full-dev input saving | 80.33% | see Section 19 |
| OOF router gain vs random | +1.796 pp | -0.2430 pp |
| FFV→FCV WC / CW | 31 / 32 | 20 / 14 |

## 27. Streaming WebSocket Validation
98/100 selected Development QA completed over `/v1/video/chat/stream`
(text-only, causal 1-fps frames, no future leakage). Details:
`streaming_eval/ws_validation_results.jsonl`.

## 28. Statistical Stability
Video-level cluster bootstrap (1000 samples):
- Focal vs Dense (full): +0.7099 [-0.1081, +1.6139]
- Focal vs Fixed FFV: +0.2372 [-0.5045, +0.9792]
- Focal vs Fixed FCV: -0.9112 [-2.4052, +0.4587]
- Focal vs Random: -0.4934 [-1.7610, +0.6792]

## 29. Is FocalStream Transferable Across Backends?
(see classification in Section 30)

## 30. Limitations
- Qwen 32k explicit-frame context cap; Dense is Qwen-compatible (448x252), not
  frozen original-resolution.
- Qwen native visual-token breakdown is NOT exposed (only total prompt tokens).
- Streaming WebSocket endpoint is frame-buffer + query-time generation, not
  incremental visual prefill.
- No Final Test; no Qwen-specific retuning.

## 31. Paper-Ready Cross-Backend Table
See Section 26.

## 32. Remaining Experiments Before Submission
- Optional: raise Qwen max_model_len / larger VRAM footprint for fuller Dense.
- Optional: repeat streaming validation across more QA.
- Final Test evaluation (explicitly deferred and NOT run in this task).

## 33. Integrity Check
- FINAL TEST TOUCHED: NO
- V4 FROZEN MODIFIED: NO
- QWEN-SPECIFIC RETUNING: NO
- FUTURE FRAME LEAKAGE: NO
- AUDIO USED: NO
- SESSION LEAKAGE: NO
- MANUAL ANSWER CORRECTION: NO
- GENERATION CONFIG CHANGED MID-RUN: NO

Generated automatically from `results/` by `scripts/generate_report.py`.
