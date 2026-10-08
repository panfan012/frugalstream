# Experiment B — Request Estimate (BLOCKED)

## Planned candidate grid
- W in {0.5s, 1.0s, 2.0s, 4.0s}
- F in {1 fps, 2 fps}
- R in {0.75x original, original, 1.25x original}
- Total policies: 24
- Scope: RTVP Development strict OOF 711 QA

## Request count
- Candidate policies: 24
- RTVP QA: 711
- Total unique candidate-QA specs: 17,064
- Existing Qwen reference candidate (W=1.0, F=1, R=original): 711
- New unique requests if all policies are run on all 711 QA: **16,353**
- This is clearly above the 3,000-new-request stop threshold.

Even with the required two-stage strategy, the union of calibration and selected
evaluation candidates across 100 seeds × K={4,8,16,32,64,128} is expected to
approach the full candidate×QA grid, so the upper bound remains ~16k new unique
requests.

## Decision
Experiment B was NOT started. No new Qwen calls were made.

## Options for unblocking (user decision required)
1. Authorize >3000 new Qwen requests, with a reduced candidate grid or staged priority.
2. Reduce candidate grid to <=4 total policies so all-policies-on-711 stays <=~2844 unique requests (but this is too coarse to study W/F/R stability).
3. Use a smaller video-level subset for Experiment B instead of all 711 RTVP QA, accepting lower precision.
4. Keep Experiment B blocked and report only Experiment A.
