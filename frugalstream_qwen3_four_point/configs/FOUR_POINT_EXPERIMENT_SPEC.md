# FOUR-POINT EXPERIMENT SPEC (A/B/C/D)

## Definition of factors
- Factor A = FrugalStream: OFF (Dense) vs ON (Frozen V3)
- Factor B = experiment-side 32k adaptation: ON (current) vs OFF (official processor only)

## Shared (identical across all four points)
- Development set: OVO Full Development 2909 QA only. Final Test (126 QA) never touched.
- Model: `/workspace/models/Qwen3-Omni-30B-A3B-Instruct`, bf16, no quantization.
- Pipeline: `qwen3_omni_moe_thinker_only`, text-only modalities.
- Decoding: temperature 0.0, top_p 1.0, max_tokens 256, seed 0.
- Prompt: frozen OVO benchmark prompt (from `manifests/dev_2909_qa.jsonl`).
- Parser/evaluator: frozen `V2.score_event` from `/root/eval_minicpmo45_ovobench_full_v2.py`.
- Frames are sent as `image_url` parts (one image per frame), so Qwen3-Omni treats
  each frame as an independent image (no video temporal patching). Native visual
  tokens per frame = grid_h*grid_w/merge^2 = grid_h*grid_w/4.
- Official Qwen processor only: `smart_resize` factor 32, min_pixels 3136,
  max_pixels 12845056, patch_size 16, merge_size 2.

## A — ADAPTED DENSE (FrugalStream OFF, adaptation ON)
- Dense causal 1-fps frames `[0, ceil(t)-1]`.
- Experiment-side DENSE_RESIZE `448x252` on every dense frame.
- Experiment-side `fit_long_frames` cap (drop oldest long frames) to fit 32k serving.
- Official Qwen processor, then Qwen generation.
- Reused from existing `qwen_predictions.jsonl` method `dense`. 0 new calls.
- Known: Raw 58.1987, Macro 60.0872, n=2909.

## B — ADAPTED V3 (FrugalStream ON, adaptation ON)
- Frozen V3: V2 streaming core (448x252 CSC) + query-time HR-1 refill for RTVP
  with `risk >= tau_V3` (tau_V3 = 0.2307592643). Refill frames original resolution.
- Experiment-side `fit_long_frames` cap applied to V2 long-term context to fit 32k.
- Official Qwen processor, then Qwen generation.
- Derived from existing `v2` + `fcv_fixed` predictions by frozen route. 0 new calls.
- Known: Raw 58.8862, Macro 61.0761, n=2909.

## C — TRUE DENSE (FrugalStream OFF, adaptation OFF)
- Dense causal 1-fps frames `[0, ceil(t)-1]`, original source resolution.
- NO DENSE_RESIZE. NO fit_long_frames. NO frame dropping/truncation.
- Official Qwen processor only (smart_resize/min_pixels/max_pixels/patch/merge).
- Qwen generation. Requests that exceed serving/model context are marked infeasible,
  never silently compressed.
- Model native context cap = 65536 (thinker text_config.max_position_embeddings).

## D — TRUE V3 (FrugalStream ON, adaptation OFF)
- Exact Frozen V3: V2 CSC 448x252 long-term frames + HR-1 refill (original res)
  for RTVP `risk >= tau_V3`. NO fit_long_frames cap, no extra dropping.
- Official Qwen processor only, then Qwen generation.
- Requests exceeding model context are marked infeasible, never silently compressed.

## Model / serving context limits
- Architectural native context: 65536 (`thinker_config.text_config.max_position_embeddings`,
  `rope_theta=1000000`, no extrapolation/RoPE hack allowed).
- Serving `max_model_len` must not exceed 65536 for semantic identity.
- GPU: one RTX PRO 6000 Blackwell, 97887 MiB. Current 32k server occupies the GPU.
