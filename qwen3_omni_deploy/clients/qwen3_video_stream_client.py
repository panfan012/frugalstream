#!/usr/bin/env python3
"""WebSocket client for the vLLM-Omni Qwen3-Omni streaming video endpoint.

Target endpoint (official):  ws://<host>:<port>/v1/video/chat/stream

Protocol (verified against vllm-project/vllm-omni source/docs, 2026-10-06):
  Client -> Server:
    {"type":"session.config", ...}          # first message
    {"type":"video.frame","data":b64, "frame_id":..., "pts_ms":...}
    {"type":"video.query","text":...}
    {"type":"video.done"}
  Server -> Client:
    response.start / response.text.delta / response.text.done
    response.output_audio.delta / response.output_audio.done
    video.frames.consumed / video.frame.ack / session.done / error

Causal-streaming rule: for a query at time T, only frames with pts_ms <= T are
sent before that query (no future leakage). The optional --realtime flag paces
frame delivery according to video timestamps; the default sends frames
immediately (fast-forward) to keep smoke tests quick.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sys
import time

try:
    import websockets
except ImportError:  # pragma: no cover
    print("Missing dependency: pip install websockets")
    sys.exit(1)

try:
    import cv2
except ImportError:  # pragma: no cover
    print("Missing dependency: pip install opencv-python")
    sys.exit(1)

DEFAULT_MODEL = "/workspace/models/Qwen3-Omni-30B-A3B-Instruct"


def load_video_frames(path: str, fps: float) -> list[dict]:
    """Decode a local mp4 into [(pts_seconds, jpeg_bytes)] at ~fps."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30.0
    frame_interval = max(1, int(round(video_fps / fps)))

    frames: list[dict] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pts_seconds = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if idx % frame_interval == 0:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                frames.append({"pts": pts_seconds, "jpeg": buf.tobytes()})
        idx += 1
    cap.release()
    return frames


def make_config(args) -> dict:
    return {
        "type": "session.config",
        "model": args.model,
        "modalities": ["text"],  # text-only output; no talker/code2wav
        "use_audio_in_video": False,  # visual frames only, ignore audio track
        "enable_frame_filter": not args.no_evs,
        "frame_filter_threshold": args.evs_threshold,
        "max_frames": args.max_frames,
        "num_frames": args.num_frames,
    }


async def recv_response(ws, timeout: float = 240.0) -> dict:
    """Read WS messages until the query's response.text.done / error."""
    t0 = time.monotonic()
    text_parts: list[str] = []
    meta: dict = {
        "started": False,
        "frames_consumed": None,
        "acks": 0,
        "error": None,
    }
    t_first_token = None
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        msg = json.loads(raw)
        mtype = msg.get("type")
        if mtype == "response.start":
            meta["started"] = True
        elif mtype == "response.text.delta":
            if t_first_token is None:
                t_first_token = time.monotonic()
            text_parts.append(msg.get("delta", ""))
        elif mtype == "response.text.done":
            meta["first_token_s"] = (
                None if t_first_token is None else round(t_first_token - t0, 3)
            )
            meta["text"] = "".join(text_parts)
            return meta
        elif mtype == "video.frames.consumed":
            meta["frames_consumed"] = {
                "frame_ids": msg.get("frame_ids"),
                "latest_pts_ms": msg.get("latest_pts_ms"),
            }
        elif mtype == "video.frame.ack":
            meta["acks"] += 1
        elif mtype == "error":
            meta["error"] = msg.get("message")
            meta["text"] = "".join(text_parts)
            return meta
        elif mtype == "session.done":
            meta["session_done"] = True
            meta["text"] = "".join(text_parts)
            return meta


async def run(args) -> None:
    uri = f"ws://{args.host}:{args.port}/v1/video/chat/stream"
    frames = load_video_frames(args.video, args.fps)
    if not frames:
        raise RuntimeError("No frames decoded from video")
    print(f"Decoded {len(frames)} frames at {args.fps} fps "
          f"(video span {frames[-1]['pts']:.2f}s)")

    query_times = sorted(float(x) for x in args.query_times.split(","))
    results: list[dict] = []

    async with websockets.connect(uri, max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps(make_config(args)))

        cursor = 0
        last_sent_pts = 0.0

        for qtime in query_times:
            # Causal send: only frames with pts <= qtime, in timestamp order.
            while cursor < len(frames) and frames[cursor]["pts"] <= qtime:
                f = frames[cursor]
                if args.realtime:
                    delay = max(0.0, f["pts"] - last_sent_pts)
                    await asyncio.sleep(min(delay, args.max_realtime_delay))
                await ws.send(json.dumps({
                    "type": "video.frame",
                    "data": base64.b64encode(f["jpeg"]).decode(),
                    "frame_id": str(cursor),
                    "pts_ms": int(round(f["pts"] * 1000)),
                }))
                last_sent_pts = f["pts"]
                cursor += 1
            print(f"[query@{qtime:.1f}s] sent frames up to index {cursor - 1}")

            meta = {"_t0": time.monotonic()}
            t_query = time.monotonic()
            await ws.send(json.dumps({"type": "video.query", "text": args.query}))
            meta = await recv_response(ws)
            latency = round(time.monotonic() - t_query, 3)
            entry = {
                "query_time_s": qtime,
                "latency_s": latency,
                "answer": meta.get("text", ""),
                "frames_consumed": meta.get("frames_consumed"),
                "frame_acks": meta.get("acks"),
                "error": meta.get("error"),
            }
            results.append(entry)
            print(f"[query@{qtime:.1f}s] latency={latency:.3f}s "
                  f"answer={meta.get('text','')[:120]!r}")

        await ws.send(json.dumps({"type": "video.done"}))
        # Drain until session.done (with a generous timeout).
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=args.timeout)
                msg = json.loads(raw)
                if msg.get("type") == "session.done":
                    print("Session done.")
                    break
        except asyncio.TimeoutError:
            print("Timed out waiting for session.done (non-fatal).")

    if args.save_json:
        with open(args.save_json, "w") as fh:
            json.dump({"video": args.video, "fps": args.fps, "results": results}, fh, indent=2)
        print(f"Saved results -> {args.save_json}")

    ok = all(r["error"] is None and r["answer"].strip() for r in results)
    sys.exit(0 if ok else 2)


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-Omni streaming video WebSocket client")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--video", required=True, help="local mp4")
    p.add_argument("--query", default="What is the main activity in this video?")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--query-times", default="8", help="comma-separated query timestamps (seconds)")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--realtime", action="store_true", help="pace frame sends by video timestamps")
    p.add_argument("--max-realtime-delay", type=float, default=2.0)
    p.add_argument("--no-evs", action="store_true", help="disable EVS near-duplicate frame filter")
    p.add_argument("--evs-threshold", type=float, default=0.95)
    p.add_argument("--save-json", help="path to save latency/answer JSON")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
