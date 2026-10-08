# QWEN3-OMNI CONTEXT CAPABILITY AUDIT

## 1. Architectural / native context
- `thinker_config.text_config.max_position_embeddings = 65536`
- `thinker_config.text_config.rope_theta = 1000000`
- `rope_scaling.type = default`, `interleaved=True`, `mrope_interleaved=True`,
  `mrope_section=[24,20,20]`. No RoPE extrapolation (no yarn/linear/ntk).
- Conclusion: semantically identical model supports positions up to **65536**.
  Exceeding 65536 would require a RoPE hack, which is forbidden.

## 2. Serving context
- Current server `max_model_len = 32768` (experiment-side 32k constraint).
- Long-context server for C/D: `max_model_len = 65536` (must not exceed 65536).

## 3. GPU memory
- 1x NVIDIA RTX PRO 6000 Blackwell, 97887 MiB total.
- Current 32k server: 97143 MiB used (VLLM::Worker PID 516764, 88572 MiB).
- Model weights ~62 GiB bf16; KV cache at gpu_memory_utilization 0.85 ~17.46 GiB.
- 64k serving is memory-feasible with lower concurrency / chunked prefill;
  KV cache per 64k sequence ~6.4 GiB, so few concurrent sequences.

## 4. C/D context distribution (from existing native accounting CSV, 2909 QA)
True Dense context:
- p50 77677, p75 197693, p90 530955, p95 797057, p99 1167450, max 3051545
- <=32768: 931; <=65536: 1326; >65536: 1583 (model-infeasible)

True V3 context:
- p50 12393, p75 27829, p90 48127, p95 78769, p99 130593, max 172275
- <=32768: 2314; <=65536: 2732; >65536: 177 (model-infeasible)

## 5. Feasibility classification
- If a C/D QA exceeds 65536 -> MODEL-INFEASIBLE (native context limit), not run.
- If a C/D QA fits accounting but the server rejects or cannot run -> recorded as
  HARDWARE/context infeasible, never silently compressed.
