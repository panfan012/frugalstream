#!/usr/bin/env python3
"""Shared helpers for Qwen3-Omni explicit-frame main-accuracy runs.

Read-only with respect to all frozen/V4/MiniCPM artifacts. The Qwen server is
only consumed over HTTP at 127.0.0.1:8091.
"""
import base64
import hashlib
import io
import json
import math
import os
import time

import numpy as np
import requests
from decord import VideoReader
from PIL import Image

BASE_DIR = "/workspace/focalstream_qwen3_crossmodel"
HOST = "127.0.0.1"
PORT = 8091
MODEL = "/workspace/models/Qwen3-Omni-30B-A3B-Instruct"
ENDPOINT = f"http://{HOST}:{PORT}/v1/chat/completions"

GEN_CFG = json.load(open(f"{BASE_DIR}/configs/qwen_generation_frozen.json", encoding="utf-8"))

# Conservative Qwen native visual-token estimates measured from the live server
# (1280x720 source): original ~883, center-crop-0.70 ~451, 448x252 ~115.
VIS_TOK_EST = {"448x252": 115, "orig": 883, "c070": 451}
TEXT_RESERVE = 200
COMPLETION_RESERVE = int(GEN_CFG.get("max_tokens", 256))
PROMPT_BUDGET = 32768 - COMPLETION_RESERVE - TEXT_RESERVE

# Proactive long-term context caps (448x252). Adaptive 400-drop still applies.
CAP = {"dense": 240, "v2": 240, "ffv_fixed": 230, "fcv_fixed": 220}


def video_reader(path):
    return VideoReader(path)


def read_frame_second(reader, second, fps, total):
    idx = int(second * fps)
    idx = max(0, min(idx, total - 1))
    return reader[idx].asnumpy()


def center_crop(im, edge=0.70):
    w, h = im.size
    cw, ch = int(round(w * edge)), int(round(h * edge))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return im.crop((x0, y0, x0 + cw, y0 + ch))


def encode_jpeg(im, quality=90):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def image_data_url(b64, mime="image/jpeg"):
    return f"data:{mime};base64,{b64}"


def frame_spec_hash(video_path, second, resize, crop):
    h = hashlib.md5(
        f"{video_path}|{second}|{resize}|{crop}".encode("utf-8")
    ).hexdigest()[:16]
    return h


def prepare_frame(reader, second, fps, total, resize=None, crop=None):
    """Decode + apply frozen resize/crop. crop is [x0,y0,x1,y1] or None."""
    arr = read_frame_second(reader, second, fps, total)
    im = Image.fromarray(arr)
    if crop:
        x0, y0, x1, y1 = [int(v) for v in crop]
        im = im.crop((x0, y0, x1, y1))
    if resize:
        w, h = [int(v) for v in resize]
        im = im.resize((w, h), Image.LANCZOS)
    return im


def estimate_prompt_tokens(long_n, refill_specs):
    """refill_specs: list of {'resize':...|None, 'crop':...|None}."""
    est = long_n * VIS_TOK_EST["448x252"]
    for r in refill_specs:
        if r.get("crop"):
            est += VIS_TOK_EST["c070"]
        elif r.get("resize"):
            est += VIS_TOK_EST["448x252"]
        else:
            est += VIS_TOK_EST["orig"]
    return est


def fit_long_frames(n_long, refill_specs, method):
    """Return number of most-recent long-term frames that fit the budget."""
    cap = CAP.get(method, 240)
    n = min(n_long, cap)
    while n > 0 and estimate_prompt_tokens(n, refill_specs) > PROMPT_BUDGET:
        n -= 1
    return max(0, n)


def build_content_parts(long_specs, refill_specs, prompt):
    """long_specs/refill_specs are prepared-image descriptors with data_url."""
    parts = []
    for s in long_specs:
        parts.append({"type": "image_url", "image_url": {"url": s["data_url"]}})
    for s in refill_specs:
        parts.append({"type": "image_url", "image_url": {"url": s["data_url"]}})
    parts.append({"type": "text", "text": prompt})
    return parts


def request_qwen(parts, timeout=None, extra=None):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": parts}],
        "modalities": GEN_CFG.get("modalities", ["text"]),
        "temperature": float(GEN_CFG.get("temperature", 0.0)),
        "top_p": float(GEN_CFG.get("top_p", 1.0)),
        "max_tokens": int(GEN_CFG.get("max_tokens", 256)),
        "stream": False,
    }
    if GEN_CFG.get("seed") is not None:
        payload["seed"] = int(GEN_CFG["seed"])
    if extra:
        payload.update(extra)
    timeout = timeout or float(GEN_CFG.get("timeout_s", 600))
    t0 = time.monotonic()
    resp = requests.post(ENDPOINT, json=payload, timeout=timeout)
    latency = time.monotonic() - t0
    return resp, latency


def parse_response(data):
    choice = data["choices"][0]["message"]
    content = choice.get("content")
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return {"text": text, "raw": data, "usage": data.get("usage")}
