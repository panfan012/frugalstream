#!/usr/bin/env python3
"""Score Qwen predictions with the frozen MiniCPM evaluator and produce main tables."""
import csv
import importlib.util
import json
import os
import statistics
import sys

BASE = "/workspace/focalstream_qwen3_crossmodel"
MANIFEST_DIR = f"{BASE}/manifests"
PRED_PATH = f"{BASE}/predictions/qwen_predictions.jsonl"
FVI_PATH = f"{MANIFEST_DIR}/frozen_visual_inputs.jsonl"
DEV_PATH = f"{MANIFEST_DIR}/dev_2909_qa.jsonl"
RTVP_PATH = f"{MANIFEST_DIR}/rtvp_oof_711_qa.jsonl"

TAU_V3 = 0.2307592643
TAU_V4 = 0.3554

CATS = ["Real-Time Visual Perception", "Backward Tracing", "Forward Active Responding"]

# import frozen scorer (module-level import does not run main)
spec = importlib.util.spec_from_file_location("v2mod", "/root/eval_minicpmo45_ovobench_full_v2.py")
V2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V2)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score(dev_row, raw):
    event = {
        "kind": dev_row["kind"],
        "ground_truth": dev_row["ground_truth"],
    }
    if dev_row["kind"] in ("ssr", "crr"):
        event["test_type"] = 1 if dev_row["ground_truth"] == "Yes" else 0
    correct, pred, invalid, proc_error, status = V2.score_event(event, raw)
    return {"correct": bool(correct), "pred": pred, "invalid": bool(invalid), "status": status}


def acc(rows):
    if not rows:
        return 0.0, 0.0, {}
    raw = 100.0 * sum(1 for r in rows if r["correct"]) / len(rows)
    cat = {}
    for c in CATS:
        sub = [r for r in rows if r["category"] == c]
        cat[c] = 100.0 * sum(1 for r in sub if r["correct"]) / len(sub) if sub else 0.0
    macro = statistics.mean(cat[c] for c in CATS)
    return raw, macro, cat


def acc_task_macro(rows):
    """RTVP strict OOF macro = mean over the 6 real-time task accuracies."""
    if not rows:
        return 0.0, 0.0, {}
    raw = 100.0 * sum(1 for r in rows if r["correct"]) / len(rows)
    per = {}
    for r in rows:
        t = r["task"]
        per.setdefault(t, [0, 0])
        per[t][0] += 1 if r["correct"] else 0
        per[t][1] += 1
    macro = 100.0 * statistics.mean(a / b for a, b in per.values()) if per else 0.0
    return raw, macro, {t: 100.0 * a / b for t, (a, b) in per.items()}


def pct(x):
    return round(x, 4)


def main():
    dev = {r["qa_id"]: r for r in load_jsonl(DEV_PATH)}
    rtvp = {r["qa_id"]: r for r in load_jsonl(RTVP_PATH)}
    fvi = {}
    for r in load_jsonl(FVI_PATH):
        fvi.setdefault(r["qa_id"], {})[r["method"]] = r

    preds = {}
    for r in load_jsonl(PRED_PATH):
        if r.get("status") != "OK":
            continue
        preds[(r["qa_id"], r["method"])] = r

    # score all method answers
    answers = {}
    for (qa, method), p in preds.items():
        if qa not in dev:
            continue
        s = score(dev[qa], p.get("raw_answer"))
        answers[(qa, method)] = {**p, **s}

    # full-dev methods
    def answer_map(method):
        return {qa: answers[(qa, method)] for qa in dev if (qa, method) in answers}

    dense = answer_map("dense")
    v2 = answer_map("v2")

    v3 = {}
    v4 = {}
    for qa, d in dev.items():
        if qa not in v2:
            continue
        if qa in rtvp:
            risk = rtvp[qa]["risk"]
            v3[qa] = answers.get((qa, "fcv_fixed"), v2[qa]) if risk >= TAU_V3 else v2[qa]
            v4[qa] = answers.get((qa, "fcv_fixed"), answers.get((qa, "ffv_fixed"), v2[qa])) if risk > TAU_V4 else answers.get((qa, "ffv_fixed"), v2[qa])
        else:
            v3[qa] = v2[qa]
            v4[qa] = v2[qa]

    methods_full = {"dense": dense, "v2": v2, "v3": v3, "v4": v4}

    full_rows = []
    for method, mp in methods_full.items():
        rows = [{"qa_id": qa, **dev[qa], **a} for qa, a in mp.items()]
        raw, macro, cat = acc(rows)
        if method == "dense":
            frames = [fvi[qa]["dense"]["frame_count"] for qa in mp]
        else:
            frames = [fvi[qa]["v2"]["frame_count"] + (0 if qa not in rtvp else 2) for qa in mp]
        full_rows.append({
            "method": method,
            "n": len(rows),
            "raw": pct(raw),
            "macro": pct(macro),
            "cat_macro": {c: pct(v) for c, v in cat.items()},
            "frames_per_qa_mean": round(statistics.mean(frames), 2) if frames else 0.0,
        })

    # RTVP OOF routing
    ffv = {qa: answers[(qa, "ffv_fixed")] for qa in rtvp if (qa, "ffv_fixed") in answers}
    fcv = {qa: answers[(qa, "fcv_fixed")] for qa in rtvp if (qa, "fcv_fixed") in answers}
    foc = {}
    for qa, r in rtvp.items():
        if qa in ffv and qa in fcv:
            foc[qa] = fcv[qa] if r["oof_action"] == "HR" else ffv[qa]

    def rows_from(mp):
        return [{"qa_id": qa, **rtvp[qa], **a} for qa, a in mp.items()]

    oof_methods = {"fixed_ffv": ffv, "fixed_fcv": fcv, "focalstream_oof": foc}

    # transition matrix FFV -> FCV
    trans = {"WC": 0, "CW": 0, "CC": 0, "WW": 0}
    trans_rows = []
    for qa in rtvp:
        if qa in ffv and qa in fcv:
            fc = ffv[qa]["correct"]
            hc = fcv[qa]["correct"]
            state = ("WC" if (not fc and hc) else "CW" if (fc and not hc) else "CC" if (fc and hc) else "WW")
            trans[state] += 1
            trans_rows.append({"qa_id": qa, "video_id": rtvp[qa]["video_id"],
                               "risk": rtvp[qa]["risk"], "oof_action": rtvp[qa]["oof_action"],
                               "ffv_correct": fc, "fcv_correct": hc, "transition": state,
                               "task": rtvp[qa]["task"]})

    oof_rows = []
    for method, mp in oof_methods.items():
        rows = rows_from(mp)
        raw, macro, task_acc = acc_task_macro(rows)
        oof_rows.append({"method": method, "n": len(rows), "raw": pct(raw), "macro": pct(macro),
                         "task_acc": {t: pct(v) for t, v in task_acc.items()}})

    # resource accounting + latency (Qwen-visible)
    def stats(vals):
        v = sorted(vals)
        if not v:
            return {"mean": None, "median": None, "p90": None, "p95": None}
        def q(p):
            i = min(len(v) - 1, int(round((p) * (len(v) - 1))))
            return v[i]
        return {"mean": statistics.mean(v), "median": statistics.median(v), "p90": q(0.90), "p95": q(0.95)}

    resource_rows = []
    latency_rows = []
    for method, mp in methods_full.items():
        rows = [{"qa_id": qa, **dev[qa], **a} for qa, a in mp.items()]
        r = {"method": method, "n": len(rows)}
        for key in ["prompt_tokens", "completion_tokens", "media_bytes", "pixels", "long_frames_used", "refill_frames_used", "latency_s"]:
            vals = [a.get(key) for a in mp.values() if isinstance(a.get(key), (int, float))]
            if key == "latency_s":
                latency_rows.append({"method": method, **stats(vals)})
            else:
                r[f"{key}_mean"] = round(statistics.mean(vals), 2) if vals else None
        resource_rows.append(r)

    # subtype analysis (full dev v4 vs dense)
    subtype_rows = []
    for task in sorted({dev[q]["task"] for q in dev}):
        for method in ["dense", "v4"]:
            rows = [{"qa_id": qa, **dev[qa], **a} for qa, a in methods_full[method].items() if dev[qa]["task"] == task]
            if rows:
                raw, _, _ = acc(rows)
                subtype_rows.append({"task": task, "method": method, "n": len(rows), "raw": pct(raw)})

    # write outputs
    write_csv(f"{BASE}/results/qwen_full_dev_main.csv", full_rows,
              ["method", "n", "raw", "macro"])
    write_json(f"{BASE}/results/qwen_full_dev_main.detail.json", full_rows)

    write_csv(f"{BASE}/results/qwen_rtvp_fixed_ffv.csv", rows_from(ffv),
              ["qa_id", "video_id", "task", "risk", "oof_action", "correct", "pred"])
    write_csv(f"{BASE}/results/qwen_rtvp_fixed_fcv.csv", rows_from(fcv),
              ["qa_id", "video_id", "task", "risk", "oof_action", "correct", "pred"])
    write_csv(f"{BASE}/results/qwen_rtvp_focalstream.csv", rows_from(foc),
              ["qa_id", "video_id", "task", "risk", "oof_action", "correct", "pred"])
    write_json(f"{BASE}/results/qwen_ffv_fcv_transition.json", {"counts": trans, "rows": trans_rows})
    write_csv(f"{BASE}/results/qwen_resource_accounting.csv", resource_rows,
              ["method", "n", "prompt_tokens_mean", "completion_tokens_mean", "media_bytes_mean",
               "pixels_mean", "long_frames_used_mean", "refill_frames_used_mean"])
    write_csv(f"{BASE}/results/qwen_latency.csv", latency_rows,
              ["method", "mean", "median", "p90", "p95"])
    write_csv(f"{BASE}/results/qwen_subtype_analysis.csv", subtype_rows, ["task", "method", "n", "raw"])

    summary = {
        "full_dev": full_rows,
        "rtvp_oof": oof_rows,
        "ffv_fcv_transition": trans,
        "tau_v3": TAU_V3,
        "tau_v4": TAU_V4,
    }
    write_json(f"{BASE}/results/qwen_summary.json", summary)
    print(json.dumps(summary, indent=2))


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
