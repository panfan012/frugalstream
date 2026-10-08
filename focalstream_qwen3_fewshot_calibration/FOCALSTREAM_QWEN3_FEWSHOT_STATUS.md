# FEW-SHOT QRE THRESHOLD CALIBRATION — STATUS

Phase: COMPLETE (offline replay; 0 new Qwen calls)
Updated: 2026-10-07 (UTC)

- Used 711 RTVP Development strict OOF QA / 203 videos.
- Video-level calibration/evaluation isolation.
- 100 split seeds for each K in {4,8,16,32,64,128}.
- 1000 matched-random routing seeds per split.
- Final Test: NOT touched.
- Frozen QRE/FFV/FCV outputs: unchanged.

Key results:
- Frozen pure tau=0.3554: Macro 74.7204, FCV rate 42.90%.
- Full oracle exact threshold: tau=0.1528, Macro 76.2362, FCV rate 96.77%.
- Best fixed = Fixed FCV: Macro 75.7020.
- Formal cost-constrained few-shot: never beats random or best fixed.
- Accuracy-only sensitivity: approaches matched random at K>=32, still below best fixed.
- QRE ranking AUC (beneficial vs others): 0.4255; not useful in Qwen.
- Conclusion: NO FEW-SHOT TRANSFER.

Report: reports/QWEN3_FEWSHOT_THRESHOLD_CALIBRATION_FINAL_REPORT.md
