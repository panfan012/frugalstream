#!/usr/bin/env python3
"""Qwen3-Omni WebSocket streaming validation over ~100 Development QA.

Streams causal 1-fps frames up to query_timestamp, then sends video.query.
No future frames. Text-only output. Fresh WS session per QA.
"""
import asyncio
import base64
import json
import os
import sys
import time

import cv2

BASE = "/workspace/focalstream_qwen3_crossmodel"
HOST = "127.0.0.1"
PORT = 8091
MODEL = "/workspace/models/Qwen3-Omni-30B-A3B-Instruct"
URI = f"ws://{HOST}:{PORT}/v1/video/chat/stream"
OUT = f"{BASE}/streaming_eval/ws_validation_results.jsonl"

try:
    import websockets
except ImportError:
    print("Missing websockets", file=sys.stderr); sys.exit(1)


def load_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def frames_up_to(path, tq, fps=1.0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(vfps / fps)))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pts = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if pts > tq + 1e-9:
            break
        if idx % interval == 0:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                frames.append({"pts": pts, "bytes": int(buf.nbytes), "jpeg": buf.tobytes()})
        idx += 1
    cap.release()
    return frames


async def recv_answer(ws, timeout=240.0):
    text = []
    meta = {"error": None, "frames_consumed": None, "acks": 0}
    t0 = time.monotonic()
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        msg = json.loads(raw)
        mtype = msg.get("type")
        if mtype == "response.text.delta":
            text.append(msg.get("delta", ""))
        elif mtype == "response.text.done":
            meta["text"] = "".join(text)
            meta["latency_s"] = round(time.monotonic() - t0, 3)
            return meta
        elif mtype == "video.frames.consumed":
            meta["frames_consumed"] = msg
        elif mtype == "video.frame.ack":
            meta["acks"] += 1
        elif mtype == "error":
            meta["error"] = msg.get("message")
            meta["text"] = "".join(text)
            meta["latency_s"] = round(time.monotonic() - t0, 3)
            return meta


async def one_qa(ws, d, tq):
    await ws.send(json.dumps({
        "type": "session.config",
        "model": MODEL,
        "modalities": ["text"],
        "use_audio_in_video": False,
        "enable_frame_filter": True,
        "frame_filter_threshold": 0.95,
        "max_frames": 50,
        "num_frames": 16,
    }))
    frames = frames_up_to(d["video_path"], tq, 1.0)
    if frames is None:
        return {"qa_id": d["qa_id"], "status": "DECODE_ERROR", "frames_sent": 0}
    nbytes = 0
    for i, f in enumerate(frames):
        await ws.send(json.dumps({
            "type": "video.frame",
            "data": base64.b64encode(f["jpeg"]).decode(),
            "frame_id": str(i),
            "pts_ms": int(round(f["pts"] * 1000)),
        }))
        nbytes += f["bytes"]
    t0 = time.monotonic()
    await ws.send(json.dumps({"type": "video.query", "text": d["prompt"]}))
    meta = await recv_answer(ws)
    meta["qa_id"] = d["qa_id"]
    meta["status"] = "OK" if meta.get("error") is None and meta.get("text", "").strip() else "ERROR"
    meta["query_timestamp"] = tq
    meta["frames_sent"] = len(frames)
    meta["media_bytes"] = nbytes
    meta["latency_s"] = meta.get("latency_s")
    return meta


async def run(rows):
    results = []
    async def worker(d):
        try:
            async with websockets.connect(URI, max_size=64 * 1024 * 1024) as ws:
                meta = await one_qa(ws, d, float(d["query_timestamp"]))
                return meta
        except Exception as exc:
            return {"qa_id": d["qa_id"], "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}

    # sequential to keep WS sessions clean and GPU load modest
    for d in rows:
        meta = await worker(d)
        results.append(meta)
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        print(f"qa={d['qa_id']} status={meta['status']} frames={meta.get('frames_sent')} lat={meta.get('latency_s')}s ans={meta.get('text','')[:60]!r}", flush=True)
    return results


def main():
    dev = load_jsonl(f"{BASE}/manifests/dev_2909_qa.jsonl")
    # deterministic subset: qa with 5 <= query_timestamp <= 30, one per qa, first 100 by qa_id
    cand = [r for r in dev if 5.0 <= float(r["query_timestamp"]) <= 30.0]
    cand.sort(key=lambda r: (int(r["qa_id"]) if r["qa_id"].isdigit() else 10**9, r["qa_id"]))
    rows = cand[:100]
    print(f"streaming validation subset: {len(rows)} QA", flush=True)
    asyncio.run(run(rows))
    ok = sum(1 for r in load_jsonl(OUT) if r["status"] == "OK")
    print(f"DONE ok={ok}/100", flush=True)


if __name__ == "__main__":
    main()
