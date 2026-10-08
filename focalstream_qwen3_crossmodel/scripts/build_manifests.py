#!/usr/bin/env python3
"""Build canonical dev/rtvp manifests from frozen V2 expansion logic.

Run with the MiniCPM-o env only for import compatibility (read-only, no model load).
"""
import csv, importlib.util, json, os, sys

BASE = "/root/OVO-Bench-full"
RES = f"{BASE}/results"
OUTDIR = "/workspace/focalstream_qwen3_crossmodel/manifests"
VIDEO_ROOT = f"{BASE}/data/src_videos"
ANNO = f"{BASE}/annotations/ovo_bench_new.json"
FT_CSV = f"{RES}/stage2a_final_test_set.csv"
OOF_CSV = f"{RES}/v4_oof_predictions.csv"

spec = importlib.util.spec_from_file_location("v2mod", "/root/eval_minicpmo45_ovobench_full_v2.py")
V2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V2)

os.makedirs(OUTDIR, exist_ok=True)

with open(ANNO, "r", encoding="utf-8") as fh:
    annos = json.load(fh)

events_by_video, _ = V2.expand_events(annos)
all_events = []
for video, events in events_by_video.items():
    for e in events:
        all_events.append((video, e))

ft_keys = set()
with open(FT_CSV, "r", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        ft_keys.add((r["video_id"], r["query_id"]))

dev_rows = []
final_rows = []
for video, e in all_events:
    key = (video, e["query_id"])
    vpath = os.path.normpath(os.path.join(VIDEO_ROOT, video))
    row = {
        "qa_id": e["query_id"],
        "video_id": video,
        "video_path": vpath,
        "query": e["question"],
        "query_timestamp": e["realtime"],
        "ground_truth": e["ground_truth"],
        "category": e["category"],
        "task": e["task"],
        "kind": e["kind"],
        "prompt": e["prompt"],
        "choices": e.get("choices", ""),
        "split": "development" if key not in ft_keys else "final_test",
    }
    if row["split"] == "final_test":
        final_rows.append(row)
    else:
        dev_rows.append(row)

# Ensure deterministic order.
dev_rows.sort(key=lambda r: (r["video_id"], str(r["qa_id"])))
final_rows.sort(key=lambda r: (r["video_id"], str(r["qa_id"])))

with open(f"{OUTDIR}/dev_2909_qa.jsonl", "w", encoding="utf-8") as fh:
    for row in dev_rows:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
with open(f"{OUTDIR}/final_test_126_qa.jsonl", "w", encoding="utf-8") as fh:
    for row in final_rows:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

dev_map = {(r["video_id"], r["qa_id"]): r for r in dev_rows}
rtvp_rows = []
with open(OOF_CSV, "r", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        key = (r["video_id"], r["query_id"])
        d = dev_map.get(key)
        if not d:
            print("MISSING dev row for OOF key", key, file=sys.stderr)
            continue
        out = dict(d)
        out["risk"] = float(r["risk"])
        out["oof_action"] = r["acc_action"]
        out["oof_correct_mc"] = int(r["acc_correct"])
        out["c070_correct_mc"] = int(r["c070_correct"])
        out["hr_correct_mc"] = int(r["hr_correct"])
        out["label"] = r["label"]
        rtvp_rows.append(out)

rtvp_rows.sort(key=lambda r: (r["video_id"], str(r["qa_id"])))
with open(f"{OUTDIR}/rtvp_oof_711_qa.jsonl", "w", encoding="utf-8") as fh:
    for row in rtvp_rows:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

print(json.dumps({
    "all_expanded": len(all_events),
    "development": len(dev_rows),
    "final_test": len(final_rows),
    "rtvp_oof": len(rtvp_rows),
    "dev_videos": len({r['video_id'] for r in dev_rows}),
    "final_videos": len({r['video_id'] for r in final_rows}),
    "rtvp_videos": len({r['video_id'] for r in rtvp_rows}),
}, indent=2))
