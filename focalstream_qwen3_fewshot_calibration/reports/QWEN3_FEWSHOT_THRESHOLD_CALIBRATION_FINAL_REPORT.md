# QWEN3 FEW-SHOT THRESHOLD CALIBRATION FINAL REPORT

Generated: 2026-10-07 12:18 UTC

## 1. Executive Summary

This experiment is a strict offline routing replay on the 711 QA RTVP Development strict OOF set. It does **0 new Qwen calls**. The only variable allowed to change is the routing threshold `tau`.

- Frozen pure MiniCPM `tau=0.3554`: Macro **74.7204**, FCV rate **42.90%**.
- Fixed FFV: Macro **74.6177**; Fixed FCV: Macro **75.7020**.
- Oracle threshold on full Qwen RTVP dev: tau **0.1528**, Macro **76.2362**, FCV rate **96.77%**, gain vs best fixed **+0.5342 pp**.
- QRE ranking AUC (beneficial vs others): **0.4255**; below 0.5.
- Formal cost-constrained few-shot calibration never recovers positive routing gain; it is below matched random and below best fixed for all K.
- Accuracy-only sensitivity approaches matched random around K=32, but still does not beat best fixed even at K=128.

**Conclusion: NO FEW-SHOT TRANSFER.** The failure is not primarily a threshold mismatch; the MiniCPM-trained QRE score does not contain a useful ranking signal for Qwen FCV benefit.

## 2. Experimental Protocol

- Data: 711 RTVP Development strict OOF QA, 203 videos.
- Labels: existing frozen Qwen FFV/FCV predictions; no new inference.
- Split: **video-level**; all QA from a video belong entirely to calibration or evaluation.
- Seeds: 100 split seeds for each K.
- Threshold grid: predefined 0%–100% QRE quantiles at 2.5% step.
- Exact-score candidates: 579 unique risk scores used only for sensitivity.
- Formal selection: cost-constrained, target escalation 43.46% ± 5 pp, maximizing calibration Macro.
- Secondary selection: accuracy-only calibration Macro (upper-flexibility sensitivity).
- Random matched router: 1000 random routing seeds per split, matched to the few-shot router FCV count.
- Oracle: best exact threshold on the evaluation subset using evaluation labels; upper bound only.

## 3. Frozen Zero-Shot Baseline

| Policy | tau | Macro | Raw | FCV rate |
|---|---|---|---|---|
| Fixed FFV | -inf | 74.6177 | 73.9803 | 0.00% |
| Frozen MiniCPM tau | 0.3554 | 74.7204 | 73.9803 | 42.90% |
| Fixed FCV | +inf | 75.7020 | 74.8242 | 100.00% |

Note: the prior cross-backend report used the stored MiniCPM `oof_action` field and reported 74.8517. This new experiment intentionally uses the pure global threshold rule `risk <= tau -> FFV; risk > tau -> FCV`, which gives 74.7204. The difference is due to OOF-fold action history, not a changed Qwen result.

## 4. Oracle Threshold Diagnostic

| Item | Value |
|---|---|
| Oracle tau | 0.1528 |
| Oracle Macro | 76.2362 |
| Oracle Raw | 75.5274 |
| Oracle FCV rate | 96.77% |
| Best fixed Macro | 75.7020 |
| Oracle gain vs best fixed | +0.5342 pp |

The oracle can exceed the best fixed policy by only **+0.534 pp**, and it does so by escalating **96.77%** of QA to FCV. Thus the ranking signal is extremely weak; the small oracle gain is essentially an almost-all-FCV policy with a few low-risk QA held back to FFV.

## 5. QRE Ranking Diagnostics

- Beneficial WC cases: 20
- Harmful CW cases: 14
- Neutral cases: 677
- Beneficial mean risk: 0.3211
- Harmful mean risk: 0.3076
- Neutral mean risk: 0.3578
- ROC-AUC beneficial vs others: 0.4255
- PR-AUC beneficial vs others: 0.0246
- ROC-AUC harmful vs others: 0.3866
- PR-AUC harmful vs others: 0.0215

Top-risk behavior:

| Top X% | P(WC) | P(CW) | WC | CW | Net |
|---|---|---|---|---|---|
| 10% | 1.41% | 1.41% | 1 | 1 | +0 |
| 20% | 2.11% | 2.11% | 3 | 3 | +0 |
| 30% | 1.88% | 2.35% | 4 | 5 | -1 |
| 40% | 1.41% | 1.76% | 4 | 5 | -1 |
| 50% | 1.69% | 1.40% | 6 | 5 | +1 |

Interpretation: QRE risk does not rank beneficial FCV cases above others; the AUC is below chance. Therefore this is not only a threshold mismatch.

## 6. Formal Few-Shot Results (cost-constrained)

| K | calib QA | calib videos | selected tau | calib Macro | calib FCV% | eval Macro | eval FCV% | gain vs frozen | gain vs random | gain vs best fixed | win-rate vs random | win-rate vs best fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 6.09 ± 2.03 | 1.77 ± 0.72 | 0.43 ± 0.16 | 76.69 ± 17.54 | 37.39 ± 17.09 | 74.78 ± 0.44 | 34.92 ± 23.69 | 0.07 ± 0.40 | -0.21 ± 0.23 | -0.92 ± 0.38 | 0.29 ± 0.21 | 0.07 ± 0.24 |
| 8 | 10.15 ± 2.33 | 2.90 ± 1.15 | 0.41 ± 0.11 | 75.31 ± 14.50 | 35.22 ± 9.49 | 74.71 ± 0.40 | 35.37 ± 18.16 | -0.00 ± 0.31 | -0.29 ± 0.18 | -0.99 ± 0.29 | 0.24 ± 0.18 | 0.03 ± 0.15 |
| 16 | 18.18 ± 2.33 | 5.13 ± 1.38 | 0.40 ± 0.06 | 74.42 ± 11.09 | 35.18 ± 4.33 | 74.67 ± 0.34 | 35.75 ± 13.43 | -0.06 ± 0.22 | -0.34 ± 0.14 | -1.04 ± 0.21 | 0.19 ± 0.10 | 0.00 ± 0.00 |
| 32 | 34.11 ± 2.35 | 9.72 ± 1.70 | 0.39 ± 0.05 | 74.37 ± 7.70 | 35.70 ± 2.21 | 74.66 ± 0.38 | 36.31 ± 10.50 | -0.09 ± 0.16 | -0.38 ± 0.13 | -1.06 ± 0.22 | 0.18 ± 0.08 | 0.00 ± 0.00 |
| 64 | 66.17 ± 2.46 | 18.84 ± 2.46 | 0.38 ± 0.04 | 74.50 ± 5.85 | 37.23 ± 3.18 | 74.70 ± 0.55 | 38.34 ± 9.01 | -0.09 ± 0.16 | -0.39 ± 0.15 | -1.07 ± 0.30 | 0.18 ± 0.10 | 0.00 ± 0.00 |
| 128 | 129.64 ± 1.65 | 37.18 ± 3.50 | 0.38 ± 0.03 | 74.67 ± 3.89 | 38.29 ± 3.87 | 74.67 ± 0.82 | 39.10 ± 8.15 | -0.11 ± 0.16 | -0.40 ± 0.20 | -1.09 ± 0.42 | 0.19 ± 0.12 | 0.00 ± 0.00 |

## 7. Sensitivity: Accuracy-Only Calibration

| K | calib QA | calib videos | selected tau | calib Macro | calib FCV% | eval Macro | eval FCV% | gain vs frozen | gain vs random | gain vs best fixed | win-rate vs random | win-rate vs best fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 6.09 ± 2.03 | 1.77 ± 0.72 | 0.79 ± 0.19 | 78.56 ± 16.96 | 10.18 ± 27.07 | 74.68 ± 0.29 | 7.32 ± 21.21 | -0.04 ± 0.24 | -0.01 ± 0.10 | -1.02 ± 0.29 | 0.47 ± 0.26 | 0.02 ± 0.14 |
| 8 | 10.15 ± 2.33 | 2.90 ± 1.15 | 0.75 ± 0.22 | 77.44 ± 14.13 | 13.89 ± 28.98 | 74.69 ± 0.37 | 10.59 ± 24.32 | -0.02 ± 0.27 | -0.02 ± 0.14 | -1.01 ± 0.33 | 0.47 ± 0.23 | 0.03 ± 0.17 |
| 16 | 18.18 ± 2.33 | 5.13 ± 1.38 | 0.65 ± 0.27 | 77.24 ± 11.12 | 25.66 ± 35.38 | 74.84 ± 0.48 | 22.38 ± 32.41 | 0.11 ± 0.42 | -0.00 ± 0.19 | -0.88 ± 0.49 | 0.51 ± 0.23 | 0.12 ± 0.33 |
| 32 | 34.11 ± 2.35 | 9.72 ± 1.70 | 0.51 ± 0.29 | 77.00 ± 7.75 | 40.98 ± 36.83 | 75.06 ± 0.65 | 39.64 ± 36.29 | 0.31 ± 0.51 | 0.03 ± 0.26 | -0.66 ± 0.63 | 0.53 ± 0.28 | 0.28 ± 0.45 |
| 64 | 66.17 ± 2.46 | 18.84 ± 2.46 | 0.44 ± 0.27 | 76.39 ± 5.87 | 50.76 ± 36.09 | 75.19 ± 0.78 | 48.93 ± 35.62 | 0.40 ± 0.55 | 0.04 ± 0.31 | -0.59 ± 0.70 | 0.56 ± 0.30 | 0.32 ± 0.47 |
| 128 | 129.64 ± 1.65 | 37.18 ± 3.50 | 0.32 ± 0.21 | 76.46 ± 3.90 | 67.15 ± 30.84 | 75.44 ± 0.99 | 66.45 ± 30.61 | 0.65 ± 0.55 | 0.13 ± 0.36 | -0.32 ± 0.77 | 0.60 ± 0.34 | 0.47 ± 0.50 |

## 8. Sample Efficiency Summary

- Formal cost-constrained router: **no K** yields a positive gain vs matched random or best fixed.
- Accuracy-only router: approaches random at K=32; remains below best fixed through K=128.
- Therefore there is no observed minimum K for stable positive routing gain.

## 9. Answers to Core Questions

| Q | Answer |
|---|---|
| Q1. Why did frozen MiniCPM threshold fail on Qwen? | QRE ranking is not useful on Qwen, not merely a threshold mismatch. |
| Q2. Does QRE ranking retain useful signal on Qwen? | No: ROC-AUC 0.4255 for beneficial cases, below chance. |
| Q3. Can oracle threshold beat best fixed? | Marginally yes, +0.534 pp, but only with 96.77% FCV escalation. |
| Q4. Does few-shot calibration recover routing gain? | No for formal cost-constrained method. |
| Q5. Minimum calibration samples for recovery? | Not observed up to K=128. |
| Q6. K=8/16/32 behavior? | Below matched random; win-rate vs random 0.24/0.19/0.18 for formal method. |
| Q7. Few-shot router stably beats matched random? | No. |
| Q8. Few-shot router stably beats best fixed? | No. |
| Q9. Cost in labeled Qwen queries? | No practical benefit; accuracy-only only approaches random at about K=32. |
| Q10. Final conclusion | NO FEW-SHOT TRANSFER. |

## 10. Integrity Check

| Constraint | Status |
|---|---|
| Final Test touched | NO |
| Qwen FFV/FCV outputs modified | NO |
| QRE retrained or features changed | NO |
| FFV/FCV/NQEW/CSC/PTS definitions changed | NO |
| Evaluation labels used to select threshold in few-shot method | NO |
| Evaluation labels used for oracle upper bound | YES, explicitly upper bound only |
| New Qwen calls | 0 |

## 11. Final Terminal Summary

```text
=== FROZEN ZERO-SHOT ===
tau: 0.3554
Macro: 74.7204
FCV rate: 42.90%

=== ORACLE THRESHOLD ===
tau: 0.1528
Macro: 76.2362
gain vs best fixed: +0.5342 pp
gain vs random: +0.57 pp (matched full-set random mean 75.6659)
FCV rate: 96.77%

=== FEW-SHOT K=8 ===
Macro mean ± std: 74.7125 ± 0.4036
gain vs frozen: -0.0016 pp
gain vs random: -0.2874 pp
gain vs best fixed: -0.9869 pp
win-rate vs random: 0.238
win-rate vs best fixed: 0.025

=== FEW-SHOT K=16 ===
Macro mean ± std: 74.6737 ± 0.3392
gain vs frozen: -0.0563 pp
gain vs random: -0.3426 pp
gain vs best fixed: -1.0408 pp
win-rate vs random: 0.191
win-rate vs best fixed: 0.000

=== FEW-SHOT K=32 ===
Macro mean ± std: 74.6578 ± 0.3801
gain vs frozen: -0.0912 pp
gain vs random: -0.3768 pp
gain vs best fixed: -1.0620 pp
win-rate vs random: 0.175
win-rate vs best fixed: 0.000

=== SAMPLE EFFICIENCY ===
minimum K with stable positive gain: NOT OBSERVED

=== QRE RANKING ===
AUC: 0.4255
PR-AUC: 0.0246
top-risk WC/CW behavior: no separation, net benefit <= 1 across top 10-50%

=== CONCLUSION ===
NO FEW-SHOT TRANSFER
```
