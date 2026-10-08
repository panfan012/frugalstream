#!/usr/bin/env python3
"""Matched-escalation random router over Qwen FFV/FCV answers (>=1000 seeds).

No VLM calls: routes are lookups over already-computed FFV/FCV predictions.
Frozen escalation count is read from the RTVP OOF action artifact.
"""
import importlib.util
import json
import os
import statistics

import numpy as np

BASE = "/workspace/focalstream_qwen3_crossmodel"
PRED = f"{BASE}/predictions/qwen_predictions.jsonl"
RTVP = f"{BASE}/manifests/rtvp_oof_711_qa.jsonl"
OUT = f"{BASE}/results/qwen_random_router_1000seeds.csv"
OUTJ = f"{BASE}/results/qwen_random_router_1000seeds.json"
SEEDS = 1000

spec = importlib.util.spec_from_file_location("v2mod", "/root/eval_minicpmo45_ovobench_full_v2.py")
V2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(V2)


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def score(dev_row, raw):
    ev = {"kind": dev_row["kind"], "ground_truth": dev_row["ground_truth"]}
    if dev_row["kind"] in ("ssr", "crr"):
        ev["test_type"] = 1 if dev_row["ground_truth"] == "Yes" else 0
    c, _, _, _, _ = V2.score_event(ev, raw)
    return bool(c)


def task_macro(rows):
    per = {}
    for r in rows:
        per.setdefault(r["task"], [0, 0])
        per[r["task"]][0] += 1 if r["correct"] else 0
        per[r["task"]][1] += 1
    return 100.0 * statistics.mean(a / b for a, b in per.values())


def raw_acc(rows):
    return 100.0 * sum(1 for r in rows if r["correct"]) / len(rows)


def main():
    rtvp = {r["qa_id"]: r for r in load_jsonl(RTVP)}
    ffv = {}
    fcv = {}
    for r in load_jsonl(PRED):
        if r.get("status") != "OK":
            continue
        if r["method"] == "ffv_fixed":
            ffv[r["qa_id"]] = r
        elif r["method"] == "fcv_fixed":
            fcv[r["qa_id"]] = r

    qids = sorted(rtvp.keys())
    # require both predictions
    qids = [q for q in qids if q in ffv and q in fcv]
    qids.sort()
    n = len(qids)
    esc = sum(1 for q in qids if rtvp[q]["oof_action"] == "HR")
    rng = np.random.default_rng(0)

    macro_vals = []
    raw_vals = []
    rows_out = []
    for seed in range(SEEDS):
        r = np.random.default_rng(seed)
        hr_set = set(r.choice(qids, size=esc, replace=False))
        sel = []
        for q in qids:
            p = fcv[q] if q in hr_set else ffv[q]
            sel.append({"qa_id": q, "video_id": rtvp[q]["video_id"], "task": rtvp[q]["task"],
                        "route": "FCV" if q in hr_set else "FFV",
                        "correct": score(rtvp[q], p.get("raw_answer"))})
        m = task_macro(sel)
        ra = raw_acc(sel)
        macro_vals.append(m)
        raw_vals.append(ra)
        rows_out.append({"seed": seed, "macro": round(m, 4), "raw": round(ra, 4)})

    macro_mean = float(np.mean(macro_vals))
    macro_std = float(np.std(macro_vals))
    raw_mean = float(np.mean(raw_vals))
    raw_std = float(np.std(raw_vals))
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    summary = {
        "n_qa": n,
        "escalation_count": esc,
        "escalation_pct": round(100.0 * esc / n, 4),
        "seeds": SEEDS,
        "macro_mean": round(macro_mean, 4),
        "macro_std": round(macro_std, 4),
        "macro_ci95": ci(macro_vals),
        "raw_mean": round(raw_mean, 4),
        "raw_std": round(raw_std, 4),
        "raw_ci95": ci(raw_vals),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("seed,macro,raw\n")
        for r in rows_out:
            fh.write(f"{r['seed']},{r['macro']},{r['raw']}\n")
    with open(OUTJ, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
