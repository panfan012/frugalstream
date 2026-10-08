#!/usr/bin/env python3
"""Test 2: offline video QA over /v1/chat/completions (text-only).

Finds an existing short mp4 (read-only, no copy). PASS = non-empty text answer.
SKIP = no server reachable or no suitable video found.
"""
from __future__ import annotations

import base64
import os
import sys
import time

from common import MODEL, find_short_video, require_server, save_result, timestamp
from qwen3_chat_video_client import chat_completion, parse_choice

QUESTION = "What is the main activity in this video?"


def main() -> None:
    require_server()
    video = find_short_video()
    if not video:
        print("SKIP: no suitable short mp4 found")
        sys.exit(3)
    size_mb = round(os.path.getsize(video) / 1e6, 2)
    print(f"video={video} ({size_mb} MB)")

    with open(video, "rb") as fh:
        data_url = f"data:video/mp4;base64,{base64.b64encode(fh.read()).decode()}"
    parts = [
        {"type": "video_url", "video_url": {"url": data_url}},
        {"type": "text", "text": QUESTION},
    ]

    t0 = time.monotonic()
    data = chat_completion("127.0.0.1", 8091, MODEL, parts, modalities=["text"], max_tokens=256)
    latency = round(time.monotonic() - t0, 3)
    out = parse_choice(data)
    answer = out["text"]
    passed = bool(answer and answer.strip())

    payload = {
        "test": "video_qa",
        "status": "PASS" if passed else "FAIL",
        "timestamp": timestamp(),
        "video": video,
        "size_mb": size_mb,
        "question": QUESTION,
        "latency_s": latency,
        "answer": answer,
        "usage": out["usage"],
    }
    path = save_result("test_video_qa.json", payload)
    print(f"status={payload['status']} latency={latency}s answer={answer!r}")
    print(f"result={path}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
