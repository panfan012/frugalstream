#!/usr/bin/env python3
"""OpenAI-compatible client for Qwen3-Omni image/video QA (text-only output).

Target endpoint (official):
    POST http://<host>:<port>/v1/chat/completions

Multimodal content schema (verified against vllm-omni example client):
    image  -> {"type":"image_url","image_url":{"url": <url-or-data-url>}}
    video  -> {"type":"video_url","video_url":{"url": <url-or-data-url>}}
    text   -> {"type":"text","text": "..."}

Text-only output: "modalities": ["text"].
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import requests

DEFAULT_MODEL = "/workspace/models/Qwen3-Omni-30B-A3B-Instruct"
MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp",
    ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
    ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
}


def data_url_from_path(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    mime = MIME.get(ext, "application/octet-stream")
    with open(path, "rb") as fh:
        return f"data:{mime};base64,{base64.b64encode(fh.read()).decode()}"


def build_messages(parts: list[dict], system: str | None = None) -> list[dict]:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": parts})
    return messages


def chat_completion(
    host: str,
    port: int,
    model: str,
    content_parts: list[dict],
    modalities: list[str] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = 600.0,
) -> dict:
    modalities = modalities or ["text"]
    payload = {
        "model": model,
        "messages": build_messages(content_parts),
        "modalities": modalities,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    url = f"http://{host}:{port}/v1/chat/completions"
    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:800]}")
    return resp.json()


def parse_choice(data: dict) -> dict:
    choice = data["choices"][0]["message"]
    content = choice.get("content")
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return {"text": text, "raw": data, "usage": data.get("usage")}


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-Omni offline image/video QA")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--image", help="image path")
    p.add_argument("--video", help="video path")
    p.add_argument("--query", default="What is happening in this image?")
    p.add_argument("--modalities", default="text")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--save-json", help="path to save raw response JSON")
    args = p.parse_args()

    if not args.image and not args.video:
        p.error("provide --image or --video")

    parts: list[dict] = []
    if args.image:
        parts.append({"type": "image_url", "image_url": {"url": data_url_from_path(args.image)}})
    if args.video:
        parts.append({"type": "video_url", "video_url": {"url": data_url_from_path(args.video)}})
    parts.append({"type": "text", "text": args.query})

    t0 = time.monotonic()
    data = chat_completion(
        args.host, args.port, args.model, parts,
        modalities=args.modalities.split(","),
        max_tokens=args.max_tokens,
    )
    latency = round(time.monotonic() - t0, 3)
    out = parse_choice(data)
    print(f"latency_s={latency}")
    print(f"answer={out['text']!r}")
    print(f"usage={out['usage']}")

    if args.save_json:
        with open(args.save_json, "w") as fh:
            json.dump({"latency_s": latency, "answer": out["text"], "raw": data}, fh, indent=2)
        print(f"Saved -> {args.save_json}")

    if not out["text"].strip():
        sys.exit(2)


if __name__ == "__main__":
    main()
