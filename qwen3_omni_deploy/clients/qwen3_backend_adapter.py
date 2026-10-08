#!/usr/bin/env python3
"""Thin adapter that wraps the *real* Qwen3-Omni vLLM-Omni streaming API.

This is NOT an emulation of MiniCPM streaming_prefill. It exposes exactly what
the upstream /v1/video/chat/stream endpoint provides. Cross-model experiments
must read ``backend_capabilities`` before comparing backends.

Protocol semantics (legacy path, vLLM-Omni 0.30.0):
    - frames are buffered in a server-side ring buffer (max_frames)
    - each query re-samples the buffer (num_frames, stride + last frame)
    - each query rebuilds the multimodal prompt and runs a fresh generation
    - the frame buffer persists across queries within a session, but there is
      NO incremental visual prefill / persistent visual KV in this path.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None  # type: ignore


class Qwen3OmniBackend:
    """Minimal async session wrapper for ws://host:port/v1/video/chat/stream."""

    backend_capabilities: dict[str, Any] = {
        "streaming_input": True,          # incremental frame upload over WS
        "incremental_visual_prefill": False,  # legacy path: no prefill-as-frames-arrive
        "persistent_visual_state": False,  # no cross-query visual KV reuse (legacy)
        "single_final_generation": True,   # one generation per query
        "video_buffering": True,           # server-side ring buffer
        "query_time_resampling": True,     # stride-sample num_frames at query time
        "multi_turn": True,                # repeated queries over one session buffer
        "text_only_output": True,
        "endpoint": "/v1/video/chat/stream",
    }

    def __init__(self, host: str = "127.0.0.1", port: int = 8091,
                 model: str = "/workspace/models/Qwen3-Omni-30B-A3B-Instruct",
                 num_frames: int = 16, max_frames: int = 50,
                 enable_frame_filter: bool = True,
                 frame_filter_threshold: float = 0.95) -> None:
        if websockets is None:
            raise RuntimeError("websockets not installed")
        self.host = host
        self.port = port
        self.model = model
        self.num_frames = num_frames
        self.max_frames = max_frames
        self.enable_frame_filter = enable_frame_filter
        self.frame_filter_threshold = frame_filter_threshold
        self._ws = None
        self._frame_seq = 0

    @property
    def uri(self) -> str:
        return f"ws://{self.host}:{self.port}/v1/video/chat/stream"

    async def start_session(self) -> None:
        """Open WS and send session.config (text-only, visual only)."""
        self._ws = await websockets.connect(self.uri, max_size=16 * 1024 * 1024)
        await self._ws.send(json.dumps({
            "type": "session.config",
            "model": self.model,
            "modalities": ["text"],
            "use_audio_in_video": False,
            "enable_frame_filter": self.enable_frame_filter,
            "frame_filter_threshold": self.frame_filter_threshold,
            "max_frames": self.max_frames,
            "num_frames": self.num_frames,
        }))
        self._frame_seq = 0

    async def push_frame(self, jpeg_bytes: bytes, timestamp_s: float) -> None:
        """Send one base64 JPEG frame with causal timestamp (visual only)."""
        if self._ws is None:
            raise RuntimeError("call start_session() first")
        await self._ws.send(json.dumps({
            "type": "video.frame",
            "data": base64.b64encode(jpeg_bytes).decode(),
            "frame_id": str(self._frame_seq),
            "pts_ms": int(round(timestamp_s * 1000)),
        }))
        self._frame_seq += 1

    async def query(self, text: str, timeout: float = 240.0) -> dict:
        """Send video.query and return {'text':..., 'latency_s':..., ...}."""
        if self._ws is None:
            raise RuntimeError("call start_session() first")
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await self._ws.send(json.dumps({"type": "video.query", "text": text}))

        text_parts: list[str] = []
        meta: dict[str, Any] = {"error": None, "frames_consumed": None}
        while True:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "response.text.delta":
                text_parts.append(msg.get("delta", ""))
            elif mtype == "response.text.done":
                break
            elif mtype == "video.frames.consumed":
                meta["frames_consumed"] = msg
            elif mtype == "error":
                meta["error"] = msg.get("message")
                break
            elif mtype == "session.done":
                meta["error"] = "session ended before response.text.done"
                break
            # ignore response.start / video.frame.ack / output_audio events
        return {
            "text": "".join(text_parts),
            "latency_s": round(loop.time() - t0, 3),
            "error": meta["error"],
            "frames_consumed": meta["frames_consumed"],
        }

    async def reset(self) -> None:
        """Close and reopen the session (fresh buffer, no carried state)."""
        await self.close()
        await self.start_session()

    async def close(self) -> None:
        """Signal video.done and close the socket."""
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "video.done"}))
            except Exception:
                pass
            await self._ws.close()
            self._ws = None
        self._frame_seq = 0


def capabilities() -> dict[str, Any]:
    """Export capability metadata for cross-model comparisons."""
    return dict(Qwen3OmniBackend.backend_capabilities)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(capabilities(), indent=2))
