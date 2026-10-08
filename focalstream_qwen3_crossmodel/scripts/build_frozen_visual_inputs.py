#!/usr/bin/env python3
"""Reconstruct exact frozen visual inputs for Qwen (read-only, no VLM calls).

Uses the frozen MiniCPM decision tree and frozen scripts. FFV/FCV are causal:
query-time window uses floor(query_timestamp), so no frame timestamp > query_timestamp.
"""
import csv, importlib.util, json, math, os, sys
import numpy as np
import joblib
from decord import VideoReader

BASE = "/root/OVO-Bench-full"
VIDEO_ROOT = f"{BASE}/data/src_videos"
OUTDIR = "/workspace/focalstream_qwen3_crossmodel/manifests"
DEV_MANIFEST = f"{OUTDIR}/dev_2909_qa.jsonl"
RTVP_MANIFEST = f"{OUTDIR}/rtvp_oof_711_qa.jsonl"
OUT = f"{OUTDIR}/frozen_visual_inputs.jsonl"
TREE = "/root/OVO-Bench-V4-backup/configs/minicpmo45_token_v1_decision_tree.joblib"

spec = importlib.util.spec_from_file_location("v2mod", "/root/eval_minicpmo45_ovobench_full_v2.py")
V2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V2)

def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def clamp_seconds(value, dur, total):
    return V2.clamp_seconds(value, dur, total)

def center_bbox(H, W, edge=0.70):
    w = W * edge
    h = H * edge
    cx = W / 2.0
    cy = H / 2.0
    return [int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)]

def norm_video_id(v):
    return v

dev_rows = load_jsonl(DEV_MANIFEST)
rtvp_rows = load_jsonl(RTVP_MANIFEST)
rtvp_map = {(r["video_id"], r["qa_id"]): r for r in rtvp_rows}

by_video = {}
for r in dev_rows:
    by_video.setdefault(r["video_id"], []).append(r)
for rows in by_video.values():
    rows.sort(key=lambda r: (int(math.ceil(float(r["query_timestamp"]))), str(r["qa_id"])))

tree = joblib.load(TREE)

def write_row(row):
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

# fresh output
if os.path.exists(OUT):
    os.remove(OUT)

videos = sorted(by_video.keys())
print("videos", len(videos), "dev_rows", len(dev_rows), "rtvp_rows", len(rtvp_rows), flush=True)
for vi, video in enumerate(videos):
    rows = by_video[video]
    vpath = os.path.normpath(os.path.join(VIDEO_ROOT, video))
    if not os.path.exists(vpath):
        print("MISSING_VIDEO", video, file=sys.stderr)
        continue
    try:
        reader = VideoReader(vpath)
        total = len(reader)
        fps = float(reader.get_avg_fps()) if total else 0.0
        dur = total / fps if fps else 0.0
    except Exception as exc:
        print("OPEN_ERROR", video, exc, file=sys.stderr)
        continue

    # Streaming state mirrors eval_minicpmo45_ovobench_full_v2.py.
    candidate = 0
    last_sent = None
    prev = None
    ema = 0.0
    steps = 2  # MAX_GAP_SECONDS frozen
    v2_cumulative = []  # cumulative streamed context across queries in this video

    for r in rows:
        target = clamp_seconds(r["query_timestamp"], dur, total)
        if target > clamp_seconds(r["query_timestamp"], dur, total):
            target = clamp_seconds(r["query_timestamp"], dur, total)

        dense_seconds = list(range(0, int(target)))

        while candidate < target:
            second = candidate
            frame_idx = min(int(second * fps), total - 1)
            try:
                fa = reader[frame_idx].asnumpy()
            except Exception as exc:
                print("FRAME_ERROR", video, r["qa_id"], second, exc, file=sys.stderr)
                candidate = target
                break
            cg = V2.to_change_tensor(fa)
            cp = V2.frame_change_score(prev, cg)
            if cp is None or (isinstance(cp, float) and (cp != cp or cp in (float('inf'), float('-inf')))):
                cp = 0.0
            ema = cp if candidate == 0 else 0.8 * ema + 0.2 * cp
            nt = candidate / max(dur, 1.0)
            if candidate == 0 or steps >= 2:
                send = True
            else:
                feat = V2.make_tree_feature(last_sent, prev, cg, ema, steps, nt, dur)
                send = bool(int(tree.predict(feat)[0]) == 1)
            if send:
                v2_cumulative.append(second)
                last_sent = cg
                steps = 0
            else:
                steps += 1
            prev = cg
            candidate += 1

        assert max(dense_seconds, default=-1) <= float(r["query_timestamp"]) + 1e-9, \
            f"dense future leak {video}/{r['qa_id']} {dense_seconds} t={r['query_timestamp']}"
        v2_seconds = list(v2_cumulative)

        assert max(v2_seconds, default=-1) <= float(r["query_timestamp"]) + 1e-9, \
            f"v2 future leak {video}/{r['qa_id']} {v2_seconds} t={r['query_timestamp']}"

        common = {
            "qa_id": r["qa_id"],
            "video_id": video,
            "video_path": vpath,
            "query": r["query"],
            "ground_truth": r["ground_truth"],
            "query_timestamp": r["query_timestamp"],
        }
        write_row({**common, "method": "dense", "selected_frame_timestamps": dense_seconds,
                   "frame_count": len(dense_seconds), "frame_order": "sequential_1fps",
                   "resize_width": None, "resize_height": None,
                   "crop_coordinates": None, "coverage_mode": "all_causal_1fps",
                   "csc_timestamps": dense_seconds, "nqew_timestamps": [],
                   "qre_score": None, "route_decision": None, "tau": None})
        write_row({**common, "method": "v2", "selected_frame_timestamps": v2_seconds,
                   "frame_count": len(v2_seconds), "frame_order": "decision_tree_1fps",
                   "resize_width": 448, "resize_height": 252,
                   "crop_coordinates": None, "coverage_mode": "pts_safety_anchor_csc",
                   "csc_timestamps": v2_seconds, "nqew_timestamps": [],
                   "qre_score": None, "route_decision": None, "tau": None})

        # RTVP query-time evidence (FFV/FCV), causal window floor(t)-1..floor(t).
        rt = rtvp_map.get((video, r["qa_id"]))
        if rt is None:
            continue
        latest = int(math.floor(float(r["query_timestamp"])))
        nq_seconds = sorted({max(0, latest - 1), latest})
        for sec in nq_seconds:
            if sec > float(r["query_timestamp"]) + 1e-9:
                raise RuntimeError(f"nq future leak {video}/{r['qa_id']} {sec} t={r['query_timestamp']}")
        ffv_shapes = []
        fcv_shapes = []
        for sec in nq_seconds:
            fi = min(int(sec * fps), total - 1)
            arr = reader[fi].asnumpy()
            H, W = arr.shape[:2]
            ffv_shapes.append((H, W))
            fcv_shapes.append((H, W))
        # Use last selected frame shape (same video/crop dims may vary if t-1/t differ; record per-frame).
        H, W = ffv_shapes[-1]
        bbox = center_bbox(H, W, 0.70)

        write_row({**common, "method": "ffv_c070", "selected_frame_timestamps": nq_seconds,
                   "frame_count": len(nq_seconds), "frame_order": "query_nqew",
                   "resize_width": None, "resize_height": None,
                   "crop_coordinates": bbox, "coverage_mode": "c070_center_0.70",
                   "csc_timestamps": [], "nqew_timestamps": nq_seconds,
                   "qre_score": rt["risk"], "route_decision": "C070", "tau": 0.3554})
        write_row({**common, "method": "fcv_fullhr", "selected_frame_timestamps": nq_seconds,
                   "frame_count": len(nq_seconds), "frame_order": "query_nqew",
                   "resize_width": None, "resize_height": None,
                   "crop_coordinates": None, "coverage_mode": "full_hr1",
                   "csc_timestamps": [], "nqew_timestamps": nq_seconds,
                   "qre_score": rt["risk"], "route_decision": rt["oof_action"], "tau": 0.3554})

    if (vi + 1) % 50 == 0:
        print(f"processed {vi+1}/{len(videos)} videos", flush=True)

print("DONE", flush=True)
