#!/usr/bin/env python3
"""Qwen3-Omni explicit-frame main-accuracy runner (resumable, deduplicated).

Request types:
  dense      = all causal 1fps frames, Qwen-compatible 448x252 (documented cap adaptation)
  v2         = frozen PTS+Safety selected frames, 448x252
  ffv_fixed  = v2 long-term context + C070 center-crop query refill
  fcv_fixed  = v2 long-term context + Full-HR query refill

V3/V4 are derived later from v2 + ffv_fixed/fcv_fixed by frozen route (no new calls).
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_common as qc

BASE = qc.BASE_DIR
MANIFEST_DIR = f"{BASE}/manifests"
CACHE_PATH = f"{BASE}/cache/qwen_request_cache.jsonl"
PRED_PATH = f"{BASE}/predictions/qwen_predictions.jsonl"
STATUS_PATH = f"{BASE}/FOCALSTREAM_QWEN3_STATUS.md"

DENSE_RESIZE = (448, 252)  # Qwen 32k-context compatibility; documented in input-semantics doc.
REFILL_TYPES = {"ffv_fixed": "ffv_c070", "fcv_fixed": "fcv_fullhr"}

_write_lock = threading.Lock()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="dense,v2,ffv_fixed,fcv_fixed")
    ap.add_argument("--limit", type=int, default=0, help="limit requests per method (smoke)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--qids", default="", help="comma-separated qa_id subset (smoke)")
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    qids = set(args.qids.split(",")) if args.qids else None

    dev = {r["qa_id"]: r for r in load_jsonl(f"{MANIFEST_DIR}/dev_2909_qa.jsonl")}
    fvi = {}
    for r in load_jsonl(f"{MANIFEST_DIR}/frozen_visual_inputs.jsonl"):
        fvi.setdefault(r["qa_id"], {})[r["method"]] = r

    # Build request specs
    specs = []
    qa_ids = sorted(dev.keys(), key=lambda k: (int(k) if str(k).isdigit() else 0, k))
    if qids:
        qa_ids = [q for q in qa_ids if q in qids]

    for qa in qa_ids:
        d = dev[qa]
        m = fvi.get(qa, {})
        if "dense" not in m or "v2" not in m:
            continue
        if "dense" in methods:
            specs.append(make_spec("dense", d, m["dense"], refill=None))
        if "v2" in methods:
            specs.append(make_spec("v2", d, m["v2"], refill=None))
        if "ffv_fixed" in methods and "ffv_c070" in m:
            specs.append(make_spec("ffv_fixed", d, m["v2"], refill=m["ffv_c070"]))
        if "fcv_fixed" in methods and "fcv_fullhr" in m:
            specs.append(make_spec("fcv_fixed", d, m["v2"], refill=m["fcv_fullhr"]))

    if args.limit:
        # deterministic limit per method
        seen = {}
        kept = []
        for s in specs:
            c = seen.get(s["method"], 0)
            if c >= args.limit:
                continue
            seen[s["method"]] = c + 1
            kept.append(s)
        specs = kept

    # Load cache for resume
    cache = {}
    if os.path.exists(CACHE_PATH):
        for r in load_jsonl(CACHE_PATH):
            cache[r["request_key"]] = r

    to_run = [s for s in specs if s["request_key"] not in cache or cache[s["request_key"]].get("status") != "OK"]
    total = len(specs)
    done = sum(1 for r in cache.values() if r.get("status") == "OK")
    failed = [r for r in cache.values() if r.get("status") != "OK"]
    print(f"specs={total} cached_ok={done} to_run={len(to_run)} cached_bad={len(failed)}", flush=True)

    write_status(total, done, len(to_run), 0, 0, "starting")

    completed_new = 0
    fail_new = 0
    t_start = time.time()

    def run_one(spec):
        return execute(spec)

    if to_run:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for result in ex.map(run_one, to_run):
                if result["status"] == "OK":
                    completed_new += 1
                elif result["status"] == "FAILED":
                    fail_new += 1
                done_now = done + completed_new + fail_new
                if (completed_new + fail_new) % 25 == 0:
                    write_status(total, done_now, max(0, len(to_run) - (completed_new + fail_new)),
                                 completed_new, fail_new, "running")
        write_status(total, done + completed_new + fail_new, 0, completed_new, fail_new, "finished")

    print(f"done_new={completed_new} failed_new={fail_new} total={total}", flush=True)


def make_spec(method, dev_row, long_row, refill):
    # long-term frames: for dense override resolution to 448x252 (Qwen-compat dense)
    if method == "dense":
        long_seconds = list(long_row["selected_frame_timestamps"])
        long_resize = DENSE_RESIZE
        long_crop = None
    else:
        long_seconds = list(long_row["selected_frame_timestamps"])
        long_resize = (long_row["resize_width"], long_row["resize_height"]) if long_row.get("resize_width") else None
        long_crop = long_row.get("crop_coordinates")

    refill_frames = []
    if refill is not None:
        rr = (refill["resize_width"], refill["resize_height"]) if refill.get("resize_width") else None
        rc = refill.get("crop_coordinates")
        refill_frames = [{"second": s, "resize": rr, "crop": rc} for s in refill["selected_frame_timestamps"]]

    key = hashlib.md5(
        json.dumps(
            {
                "qa_id": dev_row["qa_id"],
                "method": method,
                "prompt": dev_row["prompt"],
                "long": [(s, list(long_resize) if long_resize else None, long_crop) for s in long_seconds],
                "refill": [(f["second"], list(f["resize"]) if f["resize"] else None, f["crop"]) for f in refill_frames],
                "gen": qc.GEN_CFG,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    return {
        "request_key": key,
        "method": method,
        "qa_id": dev_row["qa_id"],
        "video_id": long_row["video_id"],
        "video_path": long_row["video_path"],
        "query": dev_row["query"],
        "ground_truth": dev_row["ground_truth"],
        "category": dev_row["category"],
        "task": dev_row["task"],
        "kind": dev_row["kind"],
        "prompt": dev_row["prompt"],
        "query_timestamp": long_row["query_timestamp"],
        "long_seconds": long_seconds,
        "long_resize": long_resize,
        "long_crop": long_crop,
        "refill_frames": refill_frames,
    }


def execute(spec):
    key = spec["request_key"]
    record = dict(spec)
    record.update({"status": "RUNNING", "timestamp": time.time()})
    try:
        reader = qc.video_reader(spec["video_path"])
        total = len(reader)
        fps = float(reader.get_avg_fps()) if total else 0.0

        # Proactive cap on long-term frames; keep most recent.
        refill_specs = []
        for f in spec["refill_frames"]:
            refill_specs.append({"resize": f["resize"], "crop": f["crop"]})
        n_long = qc.fit_long_frames(len(spec["long_seconds"]), refill_specs, spec["method"])
        long_seconds = spec["long_seconds"][-n_long:] if n_long else []

        # prepare refill frames first (order: long then refill)
        refill_prepared = []
        for f in spec["refill_frames"]:
            im = qc.prepare_frame(reader, f["second"], fps, total, f["resize"], f["crop"])
            b64 = qc.encode_jpeg(im)
            refill_prepared.append({"data_url": qc.image_data_url(b64), "size": tuple(im.size), "bytes": int(len(b64) * 3 / 4)})

        long_prepared = []
        for s in long_seconds:
            im = qc.prepare_frame(reader, s, fps, total, spec["long_resize"], spec["long_crop"])
            b64 = qc.encode_jpeg(im)
            long_prepared.append({"data_url": qc.image_data_url(b64), "size": tuple(im.size), "bytes": int(len(b64) * 3 / 4)})

        parts = qc.build_content_parts(long_prepared, refill_prepared, spec["prompt"])
        resp, latency = qc.request_qwen(parts)

        # Adaptive drop for context-length rejections (keep most recent long frames).
        def _is_len_error(resp):
            t = (resp.text or "").lower()
            return resp.status_code in (400, 500) and (
                "model_len" in t or "maximum model length" in t or
                "longer than the maximum" in t or "context" in t
            )

        retries = 0
        while _is_len_error(resp):
            if retries >= 2 or len(long_prepared) == 0:
                break
            drop = max(1, len(long_prepared) // 5)
            long_prepared = long_prepared[drop:]
            parts = qc.build_content_parts(long_prepared, refill_prepared, spec["prompt"])
            resp, latency = qc.request_qwen(parts)
            retries += 1

        record["latency_s"] = round(latency, 4)
        record["long_frames_used"] = len(long_prepared)
        record["refill_frames_used"] = len(refill_prepared)
        record["media_bytes"] = sum(s["bytes"] for s in long_prepared + refill_prepared)
        record["pixels"] = sum(s["size"][0] * s["size"][1] for s in long_prepared + refill_prepared)
        record["long_pixels"] = sum(s["size"][0] * s["size"][1] for s in long_prepared)
        record["refill_pixels"] = sum(s["size"][0] * s["size"][1] for s in refill_prepared)

        if resp.status_code == 200:
            parsed = qc.parse_response(resp.json())
            record["raw_answer"] = parsed["text"]
            record["usage"] = parsed["usage"]
            record["status"] = "OK"
            record["prompt_tokens"] = (parsed["usage"] or {}).get("prompt_tokens")
            record["completion_tokens"] = (parsed["usage"] or {}).get("completion_tokens")
        else:
            record["status"] = "FAILED"
            record["error"] = f"HTTP {resp.status_code}: {resp.text[:400]}"
    except Exception as exc:
        record["status"] = "FAILED"
        record["error"] = f"{type(exc).__name__}: {exc}"

    record["timestamp"] = time.time()
    append_jsonl(CACHE_PATH, record)
    if record["status"] == "OK":
        pred = {
            "request_key": key,
            "method": spec["method"],
            "qa_id": spec["qa_id"],
            "video_id": spec["video_id"],
            "raw_answer": record.get("raw_answer"),
            "ground_truth": spec["ground_truth"],
            "category": spec["category"],
            "task": spec["task"],
            "kind": spec["kind"],
            "prompt_tokens": record.get("prompt_tokens"),
            "completion_tokens": record.get("completion_tokens"),
            "latency_s": record.get("latency_s"),
            "long_frames_used": record.get("long_frames_used"),
            "refill_frames_used": record.get("refill_frames_used"),
            "media_bytes": record.get("media_bytes"),
            "pixels": record.get("pixels"),
            "status": "OK",
            "timestamp": record["timestamp"],
        }
        append_jsonl(PRED_PATH, pred)
    return record


def append_jsonl(path, obj):
    with _write_lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_status(total, done, remaining, ok_new, fail_new, phase):
    with open(STATUS_PATH, "w", encoding="utf-8") as fh:
        fh.write(
            "# FOCALSTREAM × QWEN3-OMNI CROSS-BACKEND EVALUATION — STATUS\n\n"
            f"Phase: Qwen main-accuracy run ({phase})\n"
            f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n\n"
            f"- Total request specs: {total}\n"
            f"- Cache hits / completed total: {done}\n"
            f"- Remaining: {remaining}\n"
            f"- New OK this run: {ok_new}\n"
            f"- New FAILED this run: {fail_new}\n"
            f"- Cache: cache/qwen_request_cache.jsonl\n"
            f"- Predictions: predictions/qwen_predictions.jsonl\n"
        )


if __name__ == "__main__":
    main()
