#!/usr/bin/env python3
"""Test 3: real streaming video WebSocket /v1/video/chat/stream.

Runs the causal streaming client against a local mp4:
  - fps variants (default 1.0; can pass --fps-list 0.5,1,2)
  - single query near 8s (or video end if shorter)
  - repeated queries at 5s/10s/15s when the clip is long enough

PASS = every submitted query returns non-empty text with no protocol error.
SKIP = no server reachable or no suitable video found.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from common import CLIENTS_DIR, find_short_video, require_server

CLIENT = os.path.join(CLIENTS_DIR, "qwen3_video_stream_client.py")


def video_duration_s(path: str) -> float:
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        return n / fps if fps > 0 else 0.0
    except Exception:
        return 0.0


def run_client(video: str, fps: float, query_times: str, out_json: str) -> int:
    cmd = [
        sys.executable, CLIENT,
        "--video", video,
        "--fps", str(fps),
        "--query-times", query_times,
        "--save-json", out_json,
        "--timeout", "240",
    ]
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps-list", default="1.0")
    args = ap.parse_args()

    require_server()
    video = find_short_video()
    if not video:
        print("SKIP: no suitable short mp4 found")
        sys.exit(3)

    duration = video_duration_s(video)
    print(f"video={video} duration={duration:.1f}s")

    fps_list = [float(x) for x in args.fps_list.split(",")]
    all_ok = True
    report = {"test": "streaming_video", "video": video, "runs": []}

    # 1) fps variants, single query
    for fps in fps_list:
        qtime = min(8.0, max(1.0, duration - 1.0))
        out_json = os.path.join(os.path.dirname(__file__), "..",
                                "results", f"stream_fps_{fps}.json")
        rc = run_client(video, fps, f"{qtime:.1f}", out_json)
        entry = {"kind": "fps_variant", "fps": fps, "query_time_s": qtime,
                 "returncode": rc, "status": "PENDING"}
        if rc == 0 and os.path.exists(out_json):
            with open(out_json) as fh:
                data = json.load(fh)
            res = data["results"][0]
            entry["answer"] = res.get("answer")
            entry["latency_s"] = res.get("latency_s")
            entry["error"] = res.get("error")
            entry["status"] = "PASS" if (res.get("answer", "").strip()
                                         and not res.get("error")) else "FAIL"
        else:
            entry["status"] = "FAIL"
        all_ok &= entry["status"] == "PASS"
        report["runs"].append(entry)
        print(f"[fps={fps}] {entry['status']} "
              f"latency={entry.get('latency_s')} answer={entry.get('answer','')[:100]!r}")

    # 2) repeated queries in one session
    if duration >= 16.0:
        out_json = os.path.join(os.path.dirname(__file__), "..",
                                "results", "stream_repeated_query.json")
        rc = run_client(video, 1.0, "5,10,15", out_json)
        entry = {"kind": "repeated_query", "query_times": [5, 10, 15],
                 "returncode": rc, "status": "PENDING", "answers": []}
        if rc == 0 and os.path.exists(out_json):
            with open(out_json) as fh:
                data = json.load(fh)
            ok = True
            for res in data["results"]:
                entry["answers"].append({
                    "query_time_s": res.get("query_time_s"),
                    "latency_s": res.get("latency_s"),
                    "answer": res.get("answer"),
                    "error": res.get("error"),
                })
                ok &= bool(res.get("answer", "").strip()) and not res.get("error")
            entry["status"] = "PASS" if ok else "FAIL"
        else:
            entry["status"] = "FAIL"
        all_ok &= entry["status"] == "PASS"
        report["runs"].append(entry)
        print(f"[repeated] {entry['status']} "
              f"latencies={[a.get('latency_s') for a in entry.get('answers', [])]}")
    else:
        print("[repeated] SKIP (clip < 16s)")

    report["status"] = "PASS" if all_ok else "FAIL"
    out_path = os.path.join(os.path.dirname(__file__), "..",
                            "results", "test_streaming_video.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"overall={report['status']} result={out_path}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
