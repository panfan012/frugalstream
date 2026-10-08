"""Shared helpers for smoke tests (no GPU work — these only talk to a running server)."""
import os
import sys
import time

import requests

DEPLOY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENTS_DIR = os.path.join(DEPLOY_DIR, "clients")
RESULTS_DIR = os.path.join(DEPLOY_DIR, "results")
if CLIENTS_DIR not in sys.path:
    sys.path.insert(0, CLIENTS_DIR)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8091
MODEL = "/workspace/models/Qwen3-Omni-30B-A3B-Instruct"


def server_up(host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=5.0) -> bool:
    try:
        r = requests.get(f"http://{host}:{port}/v1/models", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def require_server() -> None:
    if not server_up():
        print("SKIP: no Qwen3-Omni server reachable at "
              f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/v1/models")
        print("Start it first:  bash scripts/start_qwen3_omni_server.sh")
        sys.exit(3)  # distinct code = skipped/pending


def _find_mp4_candidates(roots, max_bytes=64 * 1024 * 1024):
    hits = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root, followlinks=False):
            # avoid descending into huge cache trees
            if any(skip in dirpath for skip in ("node_modules", ".git", ".cache")):
                continue
            for fn in files:
                if fn.lower().endswith(".mp4"):
                    p = os.path.join(dirpath, fn)
                    try:
                        sz = os.path.getsize(p)
                    except OSError:
                        continue
                    if sz <= max_bytes:
                        hits.append((sz, p))
    return hits


def find_short_video(max_bytes=64 * 1024 * 1024):
    """Return a small existing mp4 without copying it; prefer known test clips."""
    roots = [
        "/root/OVO-Bench-full",
        "/root/OVO-Bench-mini",
        "/root/OVO-Bench-src",
        "/root/OVO-Bench-V4-backup",
        "/root/lvrd_blackbox_rate_distortion",
        "/workspace",
    ]
    hits = sorted(_find_mp4_candidates(roots, max_bytes))
    for sz, path in hits:
        if os.path.basename(path) == "sample_1_real.mp4":
            return path
    return hits[0][1] if hits else None


def save_result(name, payload) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as fh:
        import json
        json.dump(payload, fh, indent=2)
    return path


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
