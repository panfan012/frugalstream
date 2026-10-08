#!/usr/bin/env python3
"""Cost metrics: frozen-plan (frames/pixels) and Qwen-visible (from predictions)."""
import json
import os
import statistics
from collections import defaultdict

import numpy as np
from decord import VideoReader

BASE = "/workspace/focalstream_qwen3_crossmodel"
FVI = f"{BASE}/manifests/frozen_visual_inputs.jsonl"
PRED = f"{BASE}/predictions/qwen_predictions.jsonl"
RTVP = f"{BASE}/manifests/rtvp_oof_711_qa.jsonl"
OUT = f"{BASE}/results/qwen_resource_accounting_frozenplan.csv"

TAU_V3 = 0.2307592643
TAU_V4 = 0.3554


def load_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    fvi = {}
    for r in load_jsonl(FVI):
        fvi.setdefault(r["qa_id"], {})[r["method"]] = r
    rtvp = {r["qa_id"]: r for r in load_jsonl(RTVP)}

    # per-video resolution
    res = {}
    for qa, mm in fvi.items():
        for method, r in mm.items():
            if r["video_id"] in res:
                continue
            try:
                rd = VideoReader(r["video_path"])
                a = rd[0].asnumpy()
                res[r["video_id"]] = (a.shape[1], a.shape[0])
            except Exception:
                pass

    rows = []
    for qa, mm in fvi.items():
        dense = mm["dense"]
        v2 = mm["v2"]
        vid = dense["video_id"]
        W, H = res.get(vid, (0, 0))
        risk = rtvp[qa]["risk"] if qa in rtvp else None
        d_frames = dense["frame_count"]
        v2_frames = v2["frame_count"]
        d_px = d_frames * W * H
        v2_px = v2_frames * 448 * 252

        # v3/v4 frozen-plan pixels
        refill_fcv = 2 * W * H
        refill_ffv = int(2 * 0.49 * W * H)
        v3_px = v2_px + (refill_fcv if (risk is not None and risk >= TAU_V3) else 0)
        if risk is None:
            v4_px = v2_px
        else:
            v4_px = v2_px + (refill_fcv if risk > TAU_V4 else refill_ffv)
        v3_frames = v2_frames + (2 if (risk is not None and risk >= TAU_V3) else 0)
        v4_frames = v2_frames + (2 if risk is not None else 0)

        rows.append({"qa_id": qa, "video_id": vid, "dense_frames": d_frames, "dense_pixels": d_px,
                     "v2_frames": v2_frames, "v2_pixels": v2_px,
                     "v3_frames": v3_frames, "v3_pixels": v3_px,
                     "v4_frames": v4_frames, "v4_pixels": v4_px})

    summ = {}
    for m in ["dense", "v2", "v3", "v4"]:
        f = statistics.mean(r[f"{m}_frames"] for r in rows)
        p = statistics.mean(r[f"{m}_pixels"] for r in rows)
        summ[m] = {"frames_per_qa": round(f, 2), "pixels_per_qa": round(p, 1),
                   "total_frames": sum(r[f"{m}_frames"] for r in rows),
                   "total_pixels": sum(r[f"{m}_pixels"] for r in rows)}
    print(json.dumps(summ, indent=2))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("qa_id,video_id,dense_frames,dense_pixels,v2_frames,v2_pixels,v3_frames,v3_pixels,v4_frames,v4_pixels\n")
        for r in rows:
            fh.write(f"{r['qa_id']},{r['video_id']},{r['dense_frames']},{r['dense_pixels']},{r['v2_frames']},{r['v2_pixels']},{r['v3_frames']},{r['v3_pixels']},{r['v4_frames']},{r['v4_pixels']}\n")


if __name__ == "__main__":
    main()
