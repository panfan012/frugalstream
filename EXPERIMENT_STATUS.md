# EXPERIMENT STATUS

## Four-point Qwen3-Omni experiment (2x2)

Factor A = FrugalStream (OFF=Dense, ON=Frozen V3)
Factor B = experiment-side 32k adaptation (ON=current, OFF=official processor only)

| Point | Description | Raw | Macro | n | Status |
|-------|-------------|----:|------:|--:|--------|
| A | Adapted Dense (FS OFF, adaptation ON) | 58.1987 | 60.0872 | 2909 | DONE (0 new calls) |
| B | Adapted V3 (FS ON, adaptation ON) | 58.8862 | 61.0761 | 2909 | DONE (0 new calls) |
| C | True Dense (FS OFF, adaptation OFF) | — | — | — | NOT YET RUN |
| D | True V3 (FS ON, adaptation OFF) | — | — | — | NOT YET RUN |

C/D actual accuracy has NOT been generated. The numbers below are native visual-token
ACCOUNTING ONLY and must not be reported as accuracy.

## Native visual-token accounting (Full Development 2909 QA)

| Metric | True Dense | True V3 |
|--------|-----------:|--------:|
| Native visual tokens/QA (mean) | 179724.10 | 20886.97 |
| Total prompt/context/QA (mean) | 180062.44 | 21225.30 |

- Native visual-token saving: 88.38% (aggregate)
- Total prompt/context saving: 88.21% (aggregate)

## Context feasibility

| Budget | True Dense fits | True V3 fits |
|-------:|----------------:|-------------:|
| 32k | 931 / 2909 (32.00%) | 2314 / 2909 (79.55%) |
| 64k | 1326 / 2909 (45.58%) | 2732 / 2909 (93.92%) |

## Frozen V3 threshold
- `tau_V3 = 0.2307592643`

## Key source files
- Four-point spec: `frugalstream_qwen3_four_point/configs/FOUR_POINT_EXPERIMENT_SPEC.md`
- C/D runner: `frugalstream_qwen3_four_point/scripts/run_cd.py`
- Native accounting: `frugalstream_qwen3_native_token_audit/`
- Frozen V3 spec: `frugalstream_qwen3_v3_generalization/strict_v3/FROZEN_V3_SPEC.md`
- Frozen V4 configs: `frozen_v4_artifacts/configs/`
- Frozen evaluator: `tools/eval_minicpmo45_ovobench_full_v2.py`
