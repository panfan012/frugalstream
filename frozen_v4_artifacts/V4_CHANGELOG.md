# V4 Changelog

## V4 Initial Snapshot
Initial version: C070 identified as best static operating point.
- C070 center crop (0.70 x 0.70, 49% area)
- 92-QA diagnostic: Raw 61.96, Macro 60.97, WC 17/25, CW harm 2/17
- Query-text crop routing failed (NO-GO)
- Dual-region / Global+Center did not improve Pareto

## V4 Final Frozen Update
Added:
- Full Development evaluation (711 RTVP QA)
- C070 vs Full HR selective escalation
- Strict 5-fold GroupKFold OOF (group = video_id)
- Threshold updated from exploratory 0.5 to frozen 0.3554
- V4 achieves STRONG PASS vs V3
- WC preserved (64/71)
- CW harm reduced (10 -> 4)
- Router officially frozen

Threshold policy:
- 0.5 is DEPRECATED exploratory threshold (preserved for history only).
- 0.3554 is the official V4 frozen threshold.
