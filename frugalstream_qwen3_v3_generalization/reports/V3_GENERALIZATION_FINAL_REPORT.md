# V3 GENERALIZATION FINAL REPORT

## Status
- Experiment A: **COMPLETE**, 0 new Qwen calls.
- Experiment B: **BLOCKED BEFORE START**, because the planned 24-candidate grid
  requires about 16,353 new unique Qwen requests, exceeding the 3,000-request stop threshold.

## Q1. Strict frozen V3 zero-shot transfer?
**YES**, with the Qwen 32k context-cap caveat.
- Full-dev Qwen: Dense Macro `60.0872`, V3 Macro `61.0761`, delta `+0.9889 pp`.
- Frozen method-specified pixel saving vs Dense: `88.58%`.
- Qwen-visible saving: `4.45%` pixels / `4.54%` prompt tokens, because Qwen-compatible Dense is also compact.

## Q2. Is V3 more backend-general than V4?
**YES**.
- V3: MiniCPM 63.1347 -> Qwen 61.0761.
- V4: MiniCPM 63.8961 -> Qwen 60.7948.
- RTVP: MiniCPM V3 76.3563 -> Qwen V3 75.5680; MiniCPM V4 78.6406 -> Qwen V4 74.8517/74.7204.
- V4's learned router degrades more on Qwen.

## Q3. Is few-shot input-side calibration effective?
**NOT TESTED** in this run. It was blocked before Qwen calls due to request-count limit.
The prior threshold-calibration experiment on V4 QRE did not recover routing benefit,
so this remains an open empirical question.

## Q4. Minimum labeled samples needed?
**NOT DETERMINED**; Experiment B not run.

## Q5. Which V3 parameters show backend dependency?
**NOT DETERMINED** for query-time W/F/R calibration.
Known backend dependency: Qwen 32k context cap changes Qwen-visible cost, but the
frozen V3 method itself transfers.

## Q6. Stable Qwen-specific operating point?
Frozen V3 is already a stable zero-shot operating point for full-dev accuracy and
method-specified cost. Query-time W/F/R calibration is not yet tested.

## Q7. Should deterministic V3-style restoration be the FrugalStream universal core?
**YES / STRONG CANDIDATE**. The strict frozen V3 transfer is positive and much more
stable than V4's learned router.

## Q8. Should few-shot calibration remain as optional backend adaptation?
**UNRESOLVED**. It is plausible, but Experiment B must be unblocked first. Do not
claim few-shot V3 adaptation works until it is tested.

## Next decision needed
Authorize one of:
1. >3000 new Qwen requests for a reduced or full V3 calibration grid.
2. A reduced <=4-policy grid that stays under ~2844 unique requests.
3. Keep Experiment B blocked.
