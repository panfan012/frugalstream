#!/usr/bin/env python3
"""Video-level cluster bootstrap for Qwen FocalStream comparisons.

Comparisons:
  full-dev: FocalStream(v4) vs Dense (macro/raw delta)
  RTVP OOF: FocalStream vs FixedFFV, vs FixedFCV, vs matched-random mean.
"""
import importlib.util
import json
import os

import numpy as np

BASE = "/workspace/focalstream_qwen3_crossmodel"
PRED = f"{BASE}/predictions/qwen_predictions.jsonl"
DEV = f"{BASE}/manifests/dev_2909_qa.jsonl"
RTVP = f"{BASE}/manifests/rtvp_oof_711_qa.jsonl"
OUT = f"{BASE}/results/bootstrap_ci.csv"

TAU_V3 = 0.2307592643
TAU_V4 = 0.3554
BOOT = 1000
RANDOM_SEEDS_PER_BOOT = 100

CATS = ["Real-Time Visual Perception", "Backward Tracing", "Forward Active Responding"]

spec = importlib.util.spec_from_file_location("v2mod", "/root/eval_minicpmo45_ovobench_full_v2.py")
V2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(V2)


def load_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as fh:
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


def cat_macro(rows):
    return np.mean([np.mean([r["correct"] for r in rows if r["category"] == c]) for c in CATS]) * 100


def task_macro(rows):
    per = {}
    for r in rows:
        per.setdefault(r["task"], [0, 0])
        per[r["task"]][0] += 1 if r["correct"] else 0
        per[r["task"]][1] += 1
    return np.mean([a / b for a, b in per.values()]) * 100


def raw_acc(rows):
    return np.mean([r["correct"] for r in rows]) * 100


def main():
    dev = {r["qa_id"]: r for r in load_jsonl(DEV)}
    rtvp = {r["qa_id"]: r for r in load_jsonl(RTVP)}

    preds = {}
    for r in load_jsonl(PRED):
        if r.get("status") != "OK":
            continue
        preds[(r["qa_id"], r["method"])] = r

    # build full-dev method correctness per qa
    def corr(qa, method):
        p = preds.get((qa, method))
        if not p:
            return None
        return score(dev[qa], p.get("raw_answer"))

    dense_corr = {q: corr(q, "dense") for q in dev}
    v2_corr = {q: corr(q, "v2") for q in dev}
    ffv_corr = {q: corr(q, "ffv_fixed") for q in rtvp}
    fcv_corr = {q: corr(q, "fcv_fixed") for q in rtvp}

    # full-dev focalstream (v4)
    v4_corr = {}
    for q in dev:
        if v2_corr[q] is None:
            continue
        if q in rtvp:
            risk = rtvp[q]["risk"]
            if risk > TAU_V4:
                v4_corr[q] = fcv_corr[q]
            else:
                v4_corr[q] = ffv_corr[q]
        else:
            v4_corr[q] = v2_corr[q]

    # full-dev rows
    dev_videos = sorted({dev[q]["video_id"] for q in dev})
    dev_by_video = {v: [q for q in dev if dev[q]["video_id"] == v] for v in dev_videos}

    def dev_rows(vids):
        rows = []
        for v in vids:
            for q in dev_by_video[v]:
                if dense_corr.get(q) is None or v4_corr.get(q) is None:
                    continue
                rows.append({"qa_id": q, "category": dev[q]["category"],
                             "dense": dense_corr[q], "v4": v4_corr[q]})
        return rows

    full_deltas_macro = []
    full_deltas_raw = []
    rng = np.random.default_rng(0)
    for _ in range(BOOT):
        vids = rng.choice(dev_videos, size=len(dev_videos), replace=True)
        rows = dev_rows(vids)
        if not rows:
            continue
        # per resample: focal and dense on same rows
        d = [{"correct": r["dense"], "category": r["category"]} for r in rows]
        f = [{"correct": r["v4"], "category": r["category"]} for r in rows]
        full_deltas_macro.append(cat_macro(f) - cat_macro(d))
        full_deltas_raw.append(raw_acc(f) - raw_acc(d))

    # RTVP OOF bootstrap
    oof_videos = sorted({rtvp[q]["video_id"] for q in rtvp})
    oof_by_video = {v: [q for q in rtvp if rtvp[q]["video_id"] == v] for v in oof_videos}

    def oof_rows(vids):
        rows = []
        for v in vids:
            for q in oof_by_video[v]:
                if ffv_corr.get(q) is None or fcv_corr.get(q) is None:
                    continue
                route = "FCV" if rtvp[q]["oof_action"] == "HR" else "FFV"
                rows.append({"qa_id": q, "task": rtvp[q]["task"],
                             "ffv": ffv_corr[q], "fcv": fcv_corr[q],
                             "focal": fcv_corr[q] if route == "FCV" else ffv_corr[q]})
        return rows

    deltas = {k: [] for k in ["focal_vs_ffv_macro", "focal_vs_ffv_raw",
                               "focal_vs_fcv_macro", "focal_vs_fcv_raw",
                               "focal_vs_random_macro", "focal_vs_random_raw"]}
    for _ in range(BOOT):
        vids = rng.choice(oof_videos, size=len(oof_videos), replace=True)
        rows = oof_rows(vids)
        if not rows:
            continue
        focal = [{"correct": r["focal"], "task": r["task"]} for r in rows]
        ffv = [{"correct": r["ffv"], "task": r["task"]} for r in rows]
        fcv = [{"correct": r["fcv"], "task": r["task"]} for r in rows]
        deltas["focal_vs_ffv_macro"].append(task_macro(focal) - task_macro(ffv))
        deltas["focal_vs_ffv_raw"].append(raw_acc(focal) - raw_acc(ffv))
        deltas["focal_vs_fcv_macro"].append(task_macro(focal) - task_macro(fcv))
        deltas["focal_vs_fcv_raw"].append(raw_acc(focal) - raw_acc(fcv))

        # matched-random on this resample
        esc = sum(1 for r in rows if r["focal"] == r["fcv"] and rtvp[r["qa_id"]]["oof_action"] == "HR")
        qids = [r["qa_id"] for r in rows]
        rand_macros = []
        rand_raws = []
        for s in range(RANDOM_SEEDS_PER_BOOT):
            rr = np.random.default_rng(s)
            hr_set = set(rr.choice(qids, size=max(0, esc), replace=False))
            sel = [{"correct": (fcv_corr[q] if q in hr_set else ffv_corr[q]), "task": rtvp[q]["task"]} for q in qids]
            rand_macros.append(task_macro(sel))
            rand_raws.append(raw_acc(sel))
        deltas["focal_vs_random_macro"].append(task_macro(focal) - np.mean(rand_macros))
        deltas["focal_vs_random_raw"].append(raw_acc(focal) - np.mean(rand_raws))

    def ci(vals):
        return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)

    rows_out = []
    def row(name, vals):
        m = float(np.mean(vals))
        lo, hi = ci(vals)
        rows_out.append({"comparison": name, "metric": "macro" if "macro" in name else "raw",
                         "mean": round(m, 4), "ci_low": lo, "ci_high": hi})
        print(f"{name:26s} mean={m:+.4f} 95%CI=[{lo:+.4f},{hi:+.4f}]")

    row("full_focal_vs_dense_macro", full_deltas_macro)
    row("full_focal_vs_dense_raw", full_deltas_raw)
    for k, v in deltas.items():
        row(k, v)

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("comparison,metric,mean,ci_low,ci_high\n")
        for r in rows_out:
            fh.write(f"{r['comparison']},{r['metric']},{r['mean']},{r['ci_low']},{r['ci_high']}\n")


if __name__ == "__main__":
    main()
