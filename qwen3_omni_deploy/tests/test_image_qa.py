#!/usr/bin/env python3
"""Test 1: image QA over the OpenAI-compatible endpoint (text-only).

PASS  = server returns a non-empty text answer.
SKIP  = no server reachable (exit code 3).
"""
from __future__ import annotations

import os
import sys
import time

from common import DEPLOY_DIR, MODEL, require_server, save_result, timestamp
from qwen3_chat_video_client import chat_completion, parse_choice

QUESTION = "What is happening in this image?"


def find_or_make_image() -> str:
    """Use an existing test image if present, else synthesize one."""
    candidates = []
    for root in ["/root/OVO-Bench-full", "/root/OVO-Bench-mini", "/root/OVO-Bench-src",
                 "/root/lvrd_blackbox_rate_distortion", "/workspace"]:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root, followlinks=False):
            if any(s in dirpath for s in (".git", "node_modules", ".cache")):
                continue
            for fn in files:
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    candidates.append(os.path.join(dirpath, fn))
            if len(candidates) > 2000:
                break
    if candidates:
        return candidates[0]

    from PIL import Image
    out = os.path.join(DEPLOY_DIR, "results", "synthetic_test_image.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.new("RGB", (320, 240), (200, 40, 40)).save(out)
    return out


def main() -> None:
    require_server()
    image = find_or_make_image()
    print(f"image={image}")

    parts = [
        {"type": "image_url", "image_url": {"url": _data_url(image)}},
        {"type": "text", "text": QUESTION},
    ]
    t0 = time.monotonic()
    data = chat_completion("127.0.0.1", 8091, MODEL, parts, modalities=["text"], max_tokens=256)
    latency = round(time.monotonic() - t0, 3)
    out = parse_choice(data)

    answer = out["text"]
    passed = bool(answer and answer.strip())
    payload = {
        "test": "image_qa",
        "status": "PASS" if passed else "FAIL",
        "timestamp": timestamp(),
        "image": image,
        "question": QUESTION,
        "latency_s": latency,
        "answer": answer,
        "usage": out["usage"],
    }
    path = save_result("test_image_qa.json", payload)
    print(f"status={payload['status']} latency={latency}s answer={answer!r}")
    print(f"result={path}")
    sys.exit(0 if passed else 1)


def _data_url(path: str) -> str:
    import base64
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as fh:
        return f"data:{mime};base64,{base64.b64encode(fh.read()).decode()}"


if __name__ == "__main__":
    main()
