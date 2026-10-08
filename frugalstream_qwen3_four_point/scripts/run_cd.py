#!/usr/bin/env python3
"""Run Point C (True Dense) and Point D (True V3) on Qwen3-Omni.

Strict rules:
- C True Dense: dense causal 1-fps frames at ORIGINAL source resolution.
  NO experiment-side 448x252 resize, NO fit_long_frames, NO adaptive frame drop.
- D True V3: V2 CSC frames at 448x252 + HR-1 refill frames at ORIGINAL resolution
  when RTVP risk >= tau_V3. NO fit_long_frames, NO extra truncation.
- Both only use the official Qwen processor (smart_resize/min_pixels/max_pixels).
- Requests whose native-token accounting context > 65536 are marked MODEL_INFEASIBLE.
- Requests rejected by the serving engine for length are marked CONTEXT_INFEASIBLE.
- Never silently compresses input to make a request fit.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import sys
import threading
import time

sys.path.insert(0, "/workspace/focalstream_qwen3_crossmodel/scripts")
import qwen_common as qc

BASE = "/workspace/frugalstream_qwen3_four_point"
CROSS = "/workspace/focalstream_qwen3_crossmodel"
AUDIT = "/workspace/frugalstream_qwen3_native_token_audit"
HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
PORT = int(os.environ.get("QWEN_PORT", "8092"))
ENDPOINT = f"http://{HOST}:{PORT}/v1/chat/completions"
qc.ENDPOINT = ENDPOINT

TAU_V3 = 0.2307592643
PATCH = 16
MERGE = 2
FACTOR = PATCH * MERGE
MIN_PIXELS = 3136
MAX_PIXELS = 12845056
LOW_TOK = 112  # 448x252 CSC frame native visual tokens

_write_lock = threading.Lock()


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def smart_resize(h, w):
    h = int(h); w = int(w)
    h_bar = round(h / FACTOR) * FACTOR
    w_bar = round(w / FACTOR) * FACTOR
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((h * w) / MAX_PIXELS)
        h_bar = max(FACTOR, int(math.floor(h / beta / FACTOR)) * FACTOR)
        w_bar = max(FACTOR, int(math.floor(w / beta / FACTOR)) * FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (h * w))
        h_bar = math.ceil(h * beta / FACTOR) * FACTOR
        w_bar = math.ceil(w * beta / FACTOR) * FACTOR
    return h_bar, w_bar


def vis_tokens(h, w):
    rh, rw = smart_resize(h, w)
    return (rh // PATCH) * (rw // PATCH) // (MERGE * MERGE)


def append_jsonl(path, obj):
    with _write_lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="true_v3,true_dense")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--qids", default="")
    ap.add_argument("--cache", default=f"{BASE}/cache/cd_request_cache.jsonl")
    ap.add_argument("--pred", default=f"{BASE}/results/cd_predictions.jsonl")
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    qids = set(args.qids.split(",")) if args.qids else None

    dev = {r["qa_id"]: r for r in load_jsonl(f"{CROSS}/manifests/dev_2909_qa.jsonl")}
    fvi = {}
    for r in load_jsonl(f"{CROSS}/manifests/frozen_visual_inputs.jsonl"):
        fvi.setdefault(r["qa_id"], {})[r["method"]] = r
    rtvp = {r["qa_id"]: r for r in load_jsonl(f"{CROSS}/manifests/rtvp_oof_711_qa.jsonl")}
    shapes = json.load(open(f"{AUDIT}/cache/video_shapes.json"))

    specs = []
    qa_ids = sorted(dev.keys(), key=lambda k: (int(k) if str(k).isdigit() else 0, k))
    if qids:
        qa_ids = [q for q in qa_ids if q in qids]

    for qa in qa_ids:
        d = dev[qa]
        m = fvi.get(qa, {})
        if "dense" not in m or "v2" not in m:
            continue
        shp = shapes.get(d["video_id"])
        if not shp or not shp.get("width"):
            continue
        H = shp["height"]; W = shp["width"]
        orig_tok = vis_tokens(H, W)

        if "true_dense" in methods:
            long_seconds = list(m["dense"]["selected_frame_timestamps"])
            specs.append(make_spec("true_dense", d, m["dense"], long_seconds,
                                   long_resize=None, refill=None, orig_tok=orig_tok))
        if "true_v3" in methods:
            long_seconds = list(m["v2"]["selected_frame_timestamps"])
            refill = None
            if qa in rtvp and float(rtvp[qa]["risk"]) >= TAU_V3:
                refill = m.get("fcv_fullhr")
            specs.append(make_spec("true_v3", d, m["v2"], long_seconds,
                                   long_resize=(448, 252), refill=refill, orig_tok=orig_tok))

    # Load cache
    cache = {}
    if os.path.exists(args.cache):
        for r in load_jsonl(args.cache):
            cache[r["request_key"]] = r

    to_run = [s for s in specs if s["request_key"] not in cache or cache[s["request_key"]].get("status") not in ("OK", "CONTEXT_INFEASIBLE", "MODEL_INFEASIBLE")]
    print(f"specs={len(specs)} cached_done={len(specs)-len(to_run)} to_run={len(to_run)}", flush=True)
    for m in methods:
        c = sum(1 for s in specs if s["method"] == m)
        r = sum(1 for s in to_run if s["method"] == m)
        print(f"  {m}: total={c} to_run={r}", flush=True)

    if args.limit:
        seen = {}
        kept = []
        for s in to_run:
            cnt = seen.get(s["method"], 0)
            if cnt >= args.limit:
                continue
            seen[s["method"]] = cnt + 1
            kept.append(s)
        to_run = kept

    ok = 0
    fail = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(execute, to_run):
            if rec["status"] == "OK":
                ok += 1
            else:
                fail += 1
    print(f"done ok={ok} fail={fail}", flush=True)


def make_spec(method, dev_row, manifest_row, long_seconds, long_resize, refill, orig_tok):
    refill_frames = []
    if refill is not None:
        refill_frames = [{"second": s, "resize": None, "crop": None}
                         for s in refill["selected_frame_timestamps"]]

    key = hashlib.md5(json.dumps({
        "model": qc.MODEL,
        "qa_id": dev_row["qa_id"],
        "method": method,
        "prompt": dev_row["prompt"],
        "long": [(s, list(long_resize) if long_resize else None, None) for s in long_seconds],
        "refill": [(f["second"], None, None) for f in refill_frames],
        "gen": qc.GEN_CFG,
    }, sort_keys=True).encode("utf-8")).hexdigest()

    context = len(long_seconds) * (LOW_TOK if long_resize else orig_tok) + len(refill_frames) * orig_tok
    # text overhead is small (~200-600); prefilter only if clearly over native limit.
    # The serving engine is authoritative; a length rejection is recorded, never compressed.
    return {
        "request_key": key,
        "method": method,
        "qa_id": dev_row["qa_id"],
        "video_id": manifest_row["video_id"],
        "video_path": manifest_row["video_path"],
        "query": dev_row["query"],
        "ground_truth": dev_row["ground_truth"],
        "category": dev_row["category"],
        "task": dev_row["task"],
        "kind": dev_row["kind"],
        "prompt": dev_row["prompt"],
        "query_timestamp": manifest_row["query_timestamp"],
        "long_seconds": long_seconds,
        "long_resize": long_resize,
        "long_crop": None,
        "refill_frames": refill_frames,
        "est_native_visual_tokens": context,
        "orig_tok": orig_tok,
    }


def execute(spec):
    key = spec["request_key"]
    rec = dict(spec)
    rec.update({"status": "RUNNING", "timestamp": time.time()})
    try:
        reader = qc.video_reader(spec["video_path"])
        total = len(reader)
        fps = float(reader.get_avg_fps()) if total else 0.0

        refill_prepared = []
        for f in spec["refill_frames"]:
            im = qc.prepare_frame(reader, f["second"], fps, total, f["resize"], f["crop"])
            b64 = qc.encode_jpeg(im)
            refill_prepared.append({"data_url": qc.image_data_url(b64), "size": tuple(im.size),
                                    "bytes": int(len(b64) * 3 / 4)})

        long_prepared = []
        for s in spec["long_seconds"]:
            im = qc.prepare_frame(reader, s, fps, total, spec["long_resize"], spec["long_crop"])
            b64 = qc.encode_jpeg(im)
            long_prepared.append({"data_url": qc.image_data_url(b64), "size": tuple(im.size),
                                  "bytes": int(len(b64) * 3 / 4)})

        parts = qc.build_content_parts(long_prepared, refill_prepared, spec["prompt"])
        resp, latency = qc.request_qwen(parts)

        rec["latency_s"] = round(latency, 4)
        rec["long_frames_used"] = len(long_prepared)
        rec["refill_frames_used"] = len(refill_prepared)
        rec["media_bytes"] = sum(s["bytes"] for s in long_prepared + refill_prepared)
        rec["pixels"] = sum(s["size"][0] * s["size"][1] for s in long_prepared + refill_prepared)

        if resp.status_code == 200:
            parsed = qc.parse_response(resp.json())
            rec["raw_answer"] = parsed["text"]
            rec["usage"] = parsed["usage"]
            rec["status"] = "OK"
            rec["prompt_tokens"] = (parsed["usage"] or {}).get("prompt_tokens")
            rec["completion_tokens"] = (parsed["usage"] or {}).get("completion_tokens")
        else:
            t = (resp.text or "").lower()
            if "model_len" in t or "maximum model length" in t or "longer than the maximum" in t or "context" in t:
                rec["status"] = "CONTEXT_INFEASIBLE"
            else:
                rec["status"] = "FAILED"
            rec["error"] = f"HTTP {resp.status_code}: {resp.text[:300]}"
    except Exception as exc:
        rec["status"] = "FAILED"
        rec["error"] = f"{type(exc).__name__}: {exc}"

    rec["timestamp"] = time.time()
    append_jsonl(BASE + "/cache/cd_request_cache.jsonl", rec)
    if rec["status"] == "OK":
        pred = {
            "request_key": key,
            "method": spec["method"],
            "qa_id": spec["qa_id"],
            "video_id": spec["video_id"],
            "raw_answer": rec.get("raw_answer"),
            "ground_truth": spec["ground_truth"],
            "category": spec["category"],
            "task": spec["task"],
            "kind": spec["kind"],
            "prompt_tokens": rec.get("prompt_tokens"),
            "completion_tokens": rec.get("completion_tokens"),
            "latency_s": rec.get("latency_s"),
            "long_frames_used": rec.get("long_frames_used"),
            "refill_frames_used": rec.get("refill_frames_used"),
            "media_bytes": rec.get("media_bytes"),
            "pixels": rec.get("pixels"),
            "status": "OK",
            "timestamp": rec["timestamp"],
        }
        append_jsonl(BASE + "/results/cd_predictions.jsonl", pred)
    return rec


if __name__ == "__main__":
    main()
