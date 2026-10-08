# QWEN INPUT SEMANTICS FOR CROSS-MODEL EVALUATION

## Backend endpoint
- Qwen3-Omni local server: `http://127.0.0.1:8091`
- Main accuracy path: OpenAI-compatible `/v1/chat/completions`, not the
  WebSocket `/v1/video/chat/stream`. The WebSocket endpoint buffers/resamples
  frames internally and is only used for the separate streaming validation.
- Each request is a fresh conversation; no session history is carried.

## Explicit-frame encoding
- FocalStream decides exactly which frames Qwen sees. The runner encodes each
  selected frame to JPEG and sends image content parts, followed by the frozen
  OVO prompt as text. No audio, no video path, no timestamp metadata.

## Frozen method mapping for Qwen
| Method | frame set | resolution/crop |
|---|---|---|
| Dense | all causal 1-fps seconds `0..ceil(t)-1` | original, no crop |
| V2 / CSC | frozen decision-tree selected seconds | resized 448x252, no crop |
| V3 (FCV refill) | V2 frames + RTVP query-time FCV | FCV = full original `[floor(t)-1, floor(t)]` |
| V4 / FocalStream | V2 frames + RTVP query-time FFV/FCV by frozen router | FFV = center crop 0.70, original |
| FFV fixed | only query-time FFV C070 | original, center crop 0.70 |
| FCV fixed | only query-time FCV Full-HR | original, no crop |

## Causal-correction note (important)
- Frozen MiniCPM `C070_1` code used `target=ceil(query_timestamp)` and frames
  `[target-1, target]`, which for non-integer query timestamps can include a
  frame timestamp greater than the query timestamp.
- This cross-model manifest uses the causal window
  `[max(0, floor(query_timestamp)-1), floor(query_timestamp)]` for BOTH FFV and
  FCV, matching frozen `HR-1` and the no-future-leakage rule.
- This is an input-mapping correction, not a method retune. Spatial crop/route
  thresholds/frame-selection logic are unchanged.

## What Qwen does with the images
- The Qwen server receives exact JPEG images as `image_url` data URLs. It does
  not read the source video or resample future frames.
- Multi-image ordering is the order sent in the content parts.

## No extra information
- Audio: not sent.
- Full video path: not sent.
- Future frames: blocked by manifest assertion.
- Timestamp/explanation metadata: not sent.

## Qwen 32k-context capacity adaptation (IMPORTANT)
- The live Qwen server is fixed at `max_model_len=32768` (deployment config
  `qwen3_omni_moe_thinker_only`, not changed for this experiment).
- Frozen MiniCPM streaming prefill can hold hundreds-to-thousands of causal
  seconds, but Qwen's explicit-frame API cannot. Measured native visual-token
  costs (1280x720 source): original ~883/frame, center-crop-0.70 ~451/frame,
  448x252 ~115/frame. Text prompt ~102 tokens; completion reserve 256 tokens.
- Therefore all Qwen main-accuracy inputs apply a uniform, method-agnostic
  capacity rule: keep the query-time refill intact; keep the MOST RECENT
  long-term frames in frozen order until the estimated prompt-token budget
  (~32,312 visual tokens) fits; on HTTP 400 context errors, drop the oldest
  20% of long-term frames and retry (max 2 adaptive retries).
- **Qwen-compatible Dense**: the frozen Dense frame set is all causal 1-fps
  seconds, but frozen Dense resolution (original) cannot be represented in 32k
  context for the vast majority of videos (only ~34 original frames fit). For
  Qwen, Dense long-term frames are therefore encoded at 448x252 (same compact
  spatial format as CSC) so the Dense-vs-FocalStream contrast isolates temporal
  sparsification + query-time evidence. Frozen original-resolution Dense cost
  is reported separately as `frozen_plan_*`.
- Proactive long-term caps: dense/v2 = 240, ffv_fixed = 230, fcv_fixed = 220.
- This is an input-representation/capacity adaptation, NOT a Qwen-specific
  retune of PTS/Safety/CSC/FFV/FCV/crop/tau/routing/prompt.

## Cumulative streaming-context fix (build artifact correction)
- Frozen MiniCPM streaming eval maintains persistent per-video stream state and
  answers each query from the cumulative causal prefix `[0, ceil(t)]`.
- The initial `build_frozen_visual_inputs.py` incorrectly wrote only the frames
  newly added since the previous query; it now writes the cumulative PTS+Safety
  selection (`v2_cumulative`) for every query, matching frozen semantics.
- Dense is already cumulative by construction (`0..ceil(t)-1`).
