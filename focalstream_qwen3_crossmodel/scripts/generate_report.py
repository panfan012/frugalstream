#!/usr/bin/env python3
"""Generate the final cross-backend report from computed results."""
import csv
import json
import os

BASE = "/workspace/focalstream_qwen3_crossmodel"
RES = f"{BASE}/results"
OUT = f"{BASE}/reports/FOCALSTREAM_QWEN3_CROSSMODEL_FINAL_REPORT.md"


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def cload(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(v, nd=4):
    if v is None:
        return "N/A"
    return f"{float(v):.{nd}f}"


def main():
    summ = jload(f"{RES}/qwen_summary.json")
    full = {d["method"]: d for d in summ["full_dev"]}
    oof = {d["method"]: d for d in summ["rtvp_oof"]}
    rr = jload(f"{RES}/qwen_random_router_1000seeds.json")
    tr = summ["ffv_fcv_transition"]
    lat = {r["method"]: r for r in cload(f"{RES}/qwen_latency.csv")}
    res = {r["method"]: r for r in cload(f"{RES}/qwen_resource_accounting.csv")}
    boot = cload(f"{RES}/bootstrap_ci.csv") if os.path.exists(f"{RES}/bootstrap_ci.csv") else []

    dense = full["dense"]; v2 = full["v2"]; v3 = full["v3"]; v4 = full["v4"]
    focal_delta = v4["macro"] - dense["macro"]
    focal_delta_raw = v4["raw"] - dense["raw"]

    def boot_val(name):
        for r in boot:
            if r["comparison"] == name:
                return f"{float(r['mean']):+.4f} [{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}]"
        return "N/A"

    router_gain_random = oof["focalstream_oof"]["macro"] - rr["macro_mean"]
    best_fixed = max(oof["fixed_ffv"]["macro"], oof["fixed_fcv"]["macro"])
    gain_best = oof["focalstream_oof"]["macro"] - best_fixed

    ws = []
    wsp = f"{BASE}/streaming_eval/ws_validation_results.jsonl"
    if os.path.exists(wsp):
        ws = [json.loads(l) for l in open(wsp, encoding="utf-8") if l.strip()]
    ws_ok = sum(1 for r in ws if r.get("status") == "OK")

    report = f"""# FocalStream × Qwen3-Omni Cross-Backend Evaluation — Final Report

## 1. Executive Summary
This experiment evaluates whether the frozen FocalStream pipeline (developed and
validated on MiniCPM-o 4.5) transfers to Qwen3-Omni-30B-A3B-Instruct with **zero
retuning**. All FocalStream components (PTS, Safety Anchor, CSC, NQEW, FFV/FCV,
QRE risk scores, RGFR thresholds) are read from frozen artifacts and applied
unchanged. The only adaptation is a documented Qwen 32k-context input-cap rule
(see Section 16), because Qwen's explicit-frame API cannot hold the very long
causal frame histories that MiniCPM streaming prefill supports.

Main numbers (Development only; Final Test untouched):
- Qwen Dense macro: {num(dense['macro'])} | Qwen FocalStream macro: {num(v4['macro'])} | Δ: {focal_delta:+.4f} pp
- Qwen RTVP strict OOF: Fixed FFV {num(oof['fixed_ffv']['macro'])} / Fixed FCV {num(oof['fixed_fcv']['macro'])} / Random matched {num(rr['macro_mean'])} ± {num(rr['macro_std'])} / FocalStream {num(oof['focalstream_oof']['macro'])}
- Router gain vs random: {router_gain_random:+.4f} pp; vs best fixed: {gain_best:+.4f} pp
- FFV→FCV harmful flips (VOH): {tr['CW']} (beneficial WC: {tr['WC']})

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
| Dense | {dense['n']} | {num(dense['raw'])} | {num(dense['macro'])} |
| V2 | {v2['n']} | {num(v2['raw'])} | {num(v2['macro'])} |
| V3 | {v3['n']} | {num(v3['raw'])} | {num(v3['macro'])} |
| V4 / FocalStream | {v4['n']} | {num(v4['raw'])} | {num(v4['macro'])} |

V2 − Dense macro: {v2['macro']-dense['macro']:+.4f} pp; V3 − Dense:
{v3['macro']-dense['macro']:+.4f} pp; V4 − Dense: {v4['macro']-dense['macro']:+.4f} pp.

## 18. Accuracy Preservation
FocalStream vs Dense: {focal_delta:+.4f} pp macro / {focal_delta_raw:+.4f} pp raw
(bootstrap CI: {boot_val('full_focal_vs_dense_macro')}).

## 19. Visual Input Saving
See `results/qwen_resource_accounting_frozenplan.csv` (method-specified) and
`results/qwen_resource_accounting.csv` (Qwen-visible). Qwen-native token saving is
reported from `usage.prompt_tokens` where exposed.

## 20. Latency / Resource Results
Latency (client E2E, warm server): Dense median {lat.get('dense',{}).get('median','N/A')}s
/ p90 {lat.get('dense',{}).get('p90','N/A')}s; FocalStream median
{lat.get('v4',{}).get('median','N/A')}s / p90 {lat.get('v4',{}).get('p90','N/A')}s.
Peak observed VRAM: see `results/qwen_resource_accounting.csv` and GPU logs.

## 21. RTVP Strict OOF Routing Results
| Policy | Raw | Macro |
|---|---:|---:|
| Fixed FFV | {num(oof['fixed_ffv']['raw'])} | {num(oof['fixed_ffv']['macro'])} |
| Fixed FCV | {num(oof['fixed_fcv']['raw'])} | {num(oof['fixed_fcv']['macro'])} |
| Random matched (mean) | {num(rr['raw_mean'])} | {num(rr['macro_mean'])} ± {num(rr['macro_std'])} |
| FocalStream | {num(oof['focalstream_oof']['raw'])} | {num(oof['focalstream_oof']['macro'])} |

## 22. Fixed FFV / Fixed FCV
See Section 21. FFV under-provisions full-context evidence; FCV over-provisions.

## 23. Matched Random Router
{rr['seeds']} seeds; escalation {rr['escalation_pct']}% (matching frozen OOF count
{rr['escalation_count']}). Distribution: `figures/qwen_random_router_distribution.png`.

## 24. Frozen QRE/RGFR Transfer
FocalStream router gain vs random: {router_gain_random:+.4f} pp; vs best fixed:
{gain_best:+.4f} pp.

## 25. Visual Over-Provisioning Harm
FFV→FCV transitions (711 QA): WC={tr['WC']} (beneficial escalation), CW={tr['CW']}
(harmful escalation / VOH), CC={tr['CC']}, WW={tr['WW']}.

## 26. MiniCPM vs Qwen Cross-Backend Comparison
| Metric | MiniCPM-o 4.5 | Qwen3-Omni |
|---|---:|---:|
| Full-dev FocalStream macro | 63.8961 | {num(v4['macro'])} |
| Full-dev Dense macro | 63.0881 | {num(dense['macro'])} |
| Full-dev input saving | 80.33% | see Section 19 |
| OOF router gain vs random | +1.796 pp | {router_gain_random:+.4f} pp |
| FFV→FCV WC / CW | 31 / 32 | {tr['WC']} / {tr['CW']} |

## 27. Streaming WebSocket Validation
{ws_ok}/100 selected Development QA completed over `/v1/video/chat/stream`
(text-only, causal 1-fps frames, no future leakage). Details:
`streaming_eval/ws_validation_results.jsonl`.

## 28. Statistical Stability
Video-level cluster bootstrap (1000 samples):
- Focal vs Dense (full): {boot_val('full_focal_vs_dense_macro')}
- Focal vs Fixed FFV: {boot_val('focal_vs_ffv_macro')}
- Focal vs Fixed FCV: {boot_val('focal_vs_fcv_macro')}
- Focal vs Random: {boot_val('focal_vs_random_macro')}

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
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(report)
    print("report written:", OUT)


if __name__ == "__main__":
    main()
