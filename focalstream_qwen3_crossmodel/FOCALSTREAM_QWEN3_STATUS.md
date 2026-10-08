# FOCALSTREAM × QWEN3-OMNI CROSS-BACKEND EVALUATION — STATUS

Phase: COMPLETE
Updated: 2026-10-07 (UTC)

- Qwen server: 127.0.0.1:8091 (qwen3_omni_moe_thinker_only)
- Frozen artifact audit: complete
- Frozen visual inputs: reconstructed (7240 rows; 0 future leaks; cumulative fix validated vs MiniCPM frames_seen)
- Qwen bulk run: 7240/7240 unique requests OK (dense 2909, v2 2909, ffv_fixed 711, fcv_fixed 711); 6 transient failures all recovered
- Streaming WS validation: 98/100 OK
- Peak GPU used: 97143 MiB (~97.1 GB)
- Final report: reports/FOCALSTREAM_QWEN3_CROSSMODEL_FINAL_REPORT.md

Key numbers:
- Qwen Dense macro 60.0872 / raw 58.1987
- Qwen V2 macro 60.3260 / V3 61.0761 / V4 (FocalStream) 60.7948
- Qwen RTVP OOF: FFV 74.6177, FCV 75.7020, Random 75.0947±0.4095, FocalStream 74.8517
- Router gain vs random -0.2430 pp; vs best fixed -0.8503 pp
- FFV→FCV WC 20 / CW 14 (VOH present)
