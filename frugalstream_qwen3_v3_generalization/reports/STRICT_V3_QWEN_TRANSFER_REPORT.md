# STRICT V3 QWEN TRANSFER REPORT

## 1. Summary
Strict frozen V3 was reconstructed from MiniCPM frozen artifacts and evaluated on
existing Qwen3-Omni outputs. **New Qwen calls: 0.**

V3 transfers as a useful zero-shot core on Qwen, with the documented Qwen
32k-context input-cap caveat.

## 2. Frozen V3 Definition
See `strict_v3/FROZEN_V3_SPEC.md`.

Core:
- V2 causal stream: 1 FPS decision-tree PTS + Safety Anchor MAX_GAP=2 + CSC 448x252.
- Query gate only for RTVP QA:
  - `risk < 0.2307592643` -> NO_REFILL -> V2 answer.
  - `risk >= 0.2307592643` -> Full-HR-1 refill -> HR-1 answer.
- HR-1 = `[floor(t)-1, floor(t)]`, 1 FPS, 2 original-resolution frames, full spatial coverage.
- Backward/Forward QA remain pure V2.
- Frozen prompt, parser, and decoding unchanged.

## 3. Qwen Full Development (2909 QA, Final Test excluded)
| Method | Raw | Macro | Qwen-visible pixels/QA | Qwen-visible prompt tokens/QA | Frames/QA |
|---|---:|---:|---:|---:|---:|
| Dense | 58.1987 | 60.0872 | 14,357,507 | 14,597.91 | 127.17 |
| V2 | 58.3362 | 60.3260 | 13,453,563 | 13,685.12 | 119.17 |
| V3 | 58.8862 | 61.0761 | 13,718,704 | 13,935.39 | 118.04 |
| V4 | 58.6800 | 60.7948 | 13,699,110 | 13,921.91 | 118.36 |

- `Delta_V3 = Macro(V3) - Macro(Dense) = +0.9889 pp`
- V3 raw delta = `+0.6875 pp`
- V4 delta vs Dense = `+0.7076 pp`

## 4. Saving
Two views are required because Qwen cannot represent the frozen original-resolution
Dense history under its 32k context cap.

### A. Frozen method-specified cost (frames/pixels from frozen plan)
| Method | Total frames | Total pixels | Pixel saving vs Dense |
|---|---:|---:|---:|
| Dense | 615,249 | 536,481,449,920 | 0.00% |
| V2 | 531,490 | 60,003,095,040 | 88.82% |
| V3 | 532,640 | 61,277,091,200 | 88.58% |
| V4 | 532,912 | 61,144,482,805 | 88.60% |

- V3 pixel saving vs Dense: **88.58%**
- V3 frame saving vs Dense: **13.43%**

### B. Qwen-visible cost
Because Qwen-compatible Dense also uses 448x252 long frames, Qwen-visible saving is much smaller:
- V3 pixel saving vs Dense: **4.45%**
- V3 media-byte saving: **5.36%**
- V3 prompt-token saving: **4.54%**

## 5. Qwen RTVP Development Strict OOF (711 QA)
| Policy | Raw | Macro |
|---|---:|---:|
| V2 | 72.5738 | 73.3761 |
| Frozen V3 | 74.8242 | 75.5680 |
| V4 frozen pure tau | 73.9803 | 74.7204 |
| Fixed FFV | 73.9803 | 74.6177 |
| Fixed FCV | 74.8242 | 75.7020 |

V3 RTVP delta vs V2 = `+2.1919 pp` macro; V3 is below Fixed FCV by `-0.1340 pp`,
but this is expected because V3 is a selective gate, not always-FCV.

## 6. V3 vs V4
| Backend | Full-dev V3 Macro | Full-dev V4 Macro | RTVP V3 Macro | RTVP V4 Macro |
|---|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 63.1347 | 63.8961 | 76.3563 | 78.6406 |
| Qwen3-Omni | 61.0761 | 60.7948 | 75.5680 | 74.8517 (frozen OOF) / 74.7204 (pure tau) |

- V3 is more cross-backend consistent in this transfer:
  - Full-dev: MiniCPM V4 > V3, but Qwen V3 > V4.
  - RTVP: MiniCPM V4 > V3 by +2.28 pp, but Qwen V4 < V3.
- The learned V4 spatial router does not transfer as consistently as deterministic V3.

## 7. Conclusion
- Frozen V3 accuracy: **passes** `Macro >= Dense - 0.5 pp`; Qwen V3 is +0.9889 pp vs Dense.
- Frozen V3 method-specified pixel saving: **88.58%**, passes >=50%.
- Qwen-visible saving is small (4.45%) because of the documented Qwen 32k context cap.
- Judgment: **YES / STRONG zero-shot transfer, with Qwen context-cap caveat.**

Integrity: Final Test NOT touched; V4 frozen files NOT modified; Qwen-specific retuning NOT performed; 0 new Qwen calls.
