#!/usr/bin/env python3
"""Offline few-shot threshold calibration replay.

Strictly uses existing Qwen FFV/FCV predictions and frozen QRE risk scores.
No Qwen calls. Video-level isolation between calibration and evaluation.
"""
import csv
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

BASE = "/workspace/focalstream_qwen3_fewshot_calibration"
CROSS = "/workspace/focalstream_qwen3_crossmodel"

RTVP_PATH = f"{CROSS}/manifests/rtvp_oof_711_qa.jsonl"
PRED_PATH = f"{CROSS}/predictions/qwen_predictions.jsonl"

KS = [4, 8, 16, 32, 64, 128]
SPLIT_SEEDS = list(range(100))
ROUTING_SEEDS_PER_SPLIT = 1000
FROZEN_TAU = 0.3554
BUDGET_CENTER = 43.46
BUDGET_HALF = 5.0
BUDGET_LO = (BUDGET_CENTER - BUDGET_HALF) / 100.0
BUDGET_HI = (BUDGET_CENTER + BUDGET_HALF) / 100.0
QUANTILE_STEP = 2.5
TASKS = ["STU", "ATR", "OCR", "OJR", "ACR", "FPD"]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def macro_of_qa_rows(rows):
    """rows: list with task and correct. Macro is mean over present task accuracies."""
    if not rows:
        return 0.0, 0.0, {}
    raw = 100.0 * sum(1 for r in rows if r["correct"]) / len(rows)
    per = {}
    for r in rows:
        per.setdefault(r["task"], [0, 0])
        per[r["task"]][0] += 1 if r["correct"] else 0
        per[r["task"]][1] += 1
    task_acc = {t: 100.0 * a / b for t, (a, b) in per.items()}
    macro = statistics.mean(task_acc.values()) if task_acc else 0.0
    return raw, macro, task_acc


def transition_counts(records, chosen_fcv):
    counts = {"WC": 0, "CW": 0, "CC": 0, "WW": 0}
    for r in records:
        fc = r["fcv_correct"] if chosen_fcv[r["qa_id"]] else r["ffv_correct"]
        ff = r["ffv_correct"]
        hv = r["fcv_correct"]
        state = ("WC" if (not ff and hv) else "CW" if (ff and not hv)
                 else "CC" if (ff and hv) else "WW")
        counts[state] += 1
    return counts


def selected_rows(records, tau):
    chosen_fcv = {r["qa_id"]: r["risk"] > tau for r in records}
    rows = []
    for r in records:
        use_fcv = chosen_fcv[r["qa_id"]]
        rows.append({
            "qa_id": r["qa_id"], "task": r["task"], "video_id": r["video_id"],
            "risk": r["risk"], "correct": r["fcv_correct"] if use_fcv else r["ffv_correct"],
            "use_fcv": use_fcv,
            "pixels": r["fcv_pixels"] if use_fcv else r["ffv_pixels"],
            "media_bytes": r["fcv_media_bytes"] if use_fcv else r["ffv_media_bytes"],
            "prompt_tokens": r["fcv_prompt_tokens"] if use_fcv else r["ffv_prompt_tokens"],
        })
    return rows, chosen_fcv


def eval_selection(records, tau):
    rows, chosen = selected_rows(records, tau)
    raw, macro, task_acc = macro_of_qa_rows(rows)
    n_fcv = sum(1 for r in rows if r["use_fcv"])
    n = len(rows)
    totals = {
        "pixels": sum(r["pixels"] for r in rows),
        "media_bytes": sum(r["media_bytes"] for r in rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
    }
    means = {k: (v / n if n else 0.0) for k, v in totals.items()}
    return {
        "n": n, "raw": raw, "macro": macro, "task_acc": task_acc,
        "fcv_n": n_fcv, "fcv_rate": 100.0 * n_fcv / n if n else 0.0,
        "totals": totals, "means": means,
        "transition": transition_counts(records, chosen),
    }


def choose_threshold(records, candidates, objective="cost_constrained"):
    """Select tau on calibration records only. Returns dict with metrics."""
    best = None
    for tau in candidates:
        rows, chosen = selected_rows(records, tau)
        raw, macro, task_acc = macro_of_qa_rows(rows)
        n_fcv = sum(1 for r in rows if r["use_fcv"])
        n = len(rows)
        fcv_rate = 100.0 * n_fcv / n if n else 0.0
        cost = sum(r["pixels"] for r in rows) / n if n else 0.0
        in_budget = BUDGET_LO <= (fcv_rate / 100.0) <= BUDGET_HI
        met = {
            "tau": tau, "calib_raw": raw, "calib_macro": macro,
            "calib_fcv_rate": fcv_rate, "calib_cost_mean_pixels": cost,
            "in_budget": in_budget,
        }
        if objective == "accuracy_only":
            better = (best is None or macro > best["calib_macro"] - 1e-12 or
                      (abs(macro - best["calib_macro"]) <= 1e-12 and cost < best["calib_cost_mean_pixels"] - 1e-12) or
                      (abs(macro - best["calib_macro"]) <= 1e-12 and abs(cost - best["calib_cost_mean_pixels"]) <= 1e-12 and
                       abs(tau - FROZEN_TAU) < abs(best["tau"] - FROZEN_TAU)))
        else:
            # cost-constrained formal protocol
            if best is None:
                better = True
            elif in_budget and not best["in_budget"]:
                better = True
            elif in_budget and best["in_budget"]:
                better = (macro > best["calib_macro"] - 1e-12 or
                          (abs(macro - best["calib_macro"]) <= 1e-12 and cost < best["calib_cost_mean_pixels"] - 1e-12) or
                          (abs(macro - best["calib_macro"]) <= 1e-12 and abs(cost - best["calib_cost_mean_pixels"]) <= 1e-12 and
                           abs(tau - FROZEN_TAU) < abs(best["tau"] - FROZEN_TAU)))
            elif not in_budget and not best["in_budget"]:
                better = (abs(fcv_rate - BUDGET_CENTER) < abs(best["calib_fcv_rate"] - BUDGET_CENTER) - 1e-12 or
                          (abs(abs(fcv_rate - BUDGET_CENTER) - abs(best["calib_fcv_rate"] - BUDGET_CENTER)) <= 1e-12 and
                           macro > best["calib_macro"] - 1e-12) or
                          (abs(abs(fcv_rate - BUDGET_CENTER) - abs(best["calib_fcv_rate"] - BUDGET_CENTER)) <= 1e-12 and
                           abs(macro - best["calib_macro"]) <= 1e-12 and cost < best["calib_cost_mean_pixels"] - 1e-12))
        if better:
            best = met
    return best


def random_router_summary(records, n_fcv_target, n_seeds, base_seed, eval_macro=None):
    """Match exact FCV count on evaluation subset by uniformly choosing QA ids.
    Returns macro distribution stats and exact win-rate vs eval_macro."""
    if not records:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0, "q95": 0.0}
    n = len(records)
    m = int(round(n_fcv_target))
    m = max(0, min(n, m))
    # precompute QA-level arrays
    qa_ids = [r["qa_id"] for r in records]
    task_of = {r["qa_id"]: r["task"] for r in records}
    ffv_of = {r["qa_id"]: r["ffv_correct"] for r in records}
    fcv_of = {r["qa_id"]: r["fcv_correct"] for r in records}
    # totals for each task for FFV selected
    task_tot = {}
    task_ffv_correct = {}
    for r in records:
        t = r["task"]
        task_tot[t] = task_tot.get(t, 0) + 1
        task_ffv_correct[t] = task_ffv_correct.get(t, 0) + (1 if r["ffv_correct"] else 0)
    rng = random.Random(base_seed)
    vals = []
    beats = 0.0
    for _ in range(n_seeds):
        fcv_ids = set(rng.sample(qa_ids, m))
        per = {}
        for t in TASKS:
            if t in task_tot:
                per[t] = [task_ffv_correct[t], task_tot[t]]
            else:
                per[t] = [0, 0]
        for q in fcv_ids:
            t = task_of[q]
            per[t][0] += (1 if fcv_of[q] else 0) - (1 if ffv_of[q] else 0)
        macro = 100.0 * statistics.mean(a / b for a, b in per.values() if b > 0) if per else 0.0
        vals.append(macro)
        if eval_macro is not None:
            if macro < eval_macro:
                beats += 1.0
            elif macro == eval_macro:
                beats += 0.5
    vals.sort()
    def q(p):
        idx = min(len(vals) - 1, int(round(p * (len(vals) - 1))))
        return vals[idx]
    return {
        "mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "median": statistics.median(vals), "q25": q(0.25), "q75": q(0.75), "q95": q(0.95),
        "win_rate_exact": beats / n_seeds if n_seeds else float("nan"),
    }


def auc_binary(scores, labels):
    """Mann-Whitney AUC, positive=1."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    ranks = {}
    n = len(pairs)
    pos_ranks = []
    # rank by score ascending; ties share average rank
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    npos = sum(1 for s, l in pairs if l == 1)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    rank_sum = sum(ranks[k] for k in range(n) if pairs[k][1] == 1)
    u = rank_sum - npos * (npos + 1) / 2.0
    return u / (npos * nneg)


def pr_auc(scores, labels):
    """Precision-recall AUC by score-descending sweep."""
    pairs = sorted(zip(scores, labels), key=lambda x: (-x[0], x[1]))
    tp = 0
    fp = 0
    npos = sum(1 for s, l in pairs if l == 1)
    if npos == 0:
        return float("nan")
    prev_rec = 0.0
    area = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            if pairs[j][1] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        rec = tp / npos
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        area += prec * (rec - prev_rec)
        prev_rec = rec
        i = j
    return area


def main():
    os.makedirs(f"{BASE}/results", exist_ok=True)
    os.makedirs(f"{BASE}/manifests", exist_ok=True)

    rtvp = {r["qa_id"]: r for r in load_jsonl(RTVP_PATH)}
    preds = {}
    for r in load_jsonl(PRED_PATH):
        preds[(r["qa_id"], r["method"])] = r

    records = []
    for qa, base in rtvp.items():
        ffv = preds.get((qa, "ffv_fixed"))
        fcv = preds.get((qa, "fcv_fixed"))
        if ffv is None or fcv is None:
            print(f"missing prediction for {qa}", file=sys.stderr)
            sys.exit(1)
        records.append({
            "qa_id": qa,
            "video_id": base["video_id"],
            "task": base["task"],
            "category": base["category"],
            "risk": float(base["risk"]),
            "ffv_correct": bool(ffv["raw_answer"] is not None),  # replaced below by frozen csv truth
            "fcv_correct": bool(fcv["raw_answer"] is not None),
            "ffv_pixels": ffv["pixels"],
            "ffv_media_bytes": ffv["media_bytes"],
            "ffv_prompt_tokens": ffv["prompt_tokens"],
            "fcv_pixels": fcv["pixels"],
            "fcv_media_bytes": fcv["media_bytes"],
            "fcv_prompt_tokens": fcv["prompt_tokens"],
        })

    # Use the frozen evaluator outputs in the existing RTVP CSVs (source of truth).
    ffv_csv = {}
    fcv_csv = {}
    with open(f"{CROSS}/results/qwen_rtvp_fixed_ffv.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ffv_csv[row["qa_id"]] = row
    with open(f"{CROSS}/results/qwen_rtvp_fixed_fcv.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fcv_csv[row["qa_id"]] = row
    for r in records:
        r["ffv_correct"] = ffv_csv[r["qa_id"]]["correct"].strip().lower() == "true"
        r["fcv_correct"] = fcv_csv[r["qa_id"]]["correct"].strip().lower() == "true"

    records_by_id = {r["qa_id"]: r for r in records}
    videos = list(dict.fromkeys(r["video_id"] for r in records))
    video_qas = defaultdict(list)
    for r in records:
        video_qas[r["video_id"]].append(r["qa_id"])
    video_qa_counts = {v: len(video_qas[v]) for v in videos}

    write_jsonl(f"{BASE}/manifests/fewshot_qa_records.jsonl", records)
    write_json(f"{BASE}/manifests/fewshot_metadata.json", {
        "n_qa": len(records),
        "n_videos": len(videos),
        "videos": videos,
        "video_qa_counts": video_qa_counts,
        "frozen_tau": FROZEN_TAU,
        "budget": {"center": BUDGET_CENTER, "lo_pp": BUDGET_CENTER - BUDGET_HALF, "hi_pp": BUDGET_CENTER + BUDGET_HALF},
        "ks": KS,
        "split_seeds": len(SPLIT_SEEDS),
        "routing_seeds_per_split": ROUTING_SEEDS_PER_SPLIT,
    })

    all_risks = sorted(r["risk"] for r in records)
    grid = []
    for q in [i * QUANTILE_STEP for i in range(int(100 / QUANTILE_STEP) + 1)]:
        idx = min(len(all_risks) - 1, int(round(q / 100.0 * (len(all_risks) - 1))))
        tau = all_risks[idx]
        if tau not in grid:
            grid.append(tau)
    exact_candidates = sorted(set(r["risk"] for r in records))

    # Full 711 baselines / oracle
    base_full = {}
    for name, tau in [("fixed_ffv", -1.0), ("fixed_fcv", float("inf")), ("frozen_tau", FROZEN_TAU)]:
        base_full[name] = eval_selection(records, tau)
    # For fixed FFV/FCV use direct flags rather than tau -1/infinity.
    def fixed_eval(use_fcv):
        rows = []
        chosen = {}
        for r in records:
            chosen[r["qa_id"]] = use_fcv
            rows.append({"qa_id": r["qa_id"], "task": r["task"], "video_id": r["video_id"],
                         "risk": r["risk"], "correct": r["fcv_correct"] if use_fcv else r["ffv_correct"],
                         "use_fcv": use_fcv,
                         "pixels": r["fcv_pixels"] if use_fcv else r["ffv_pixels"],
                         "media_bytes": r["fcv_media_bytes"] if use_fcv else r["ffv_media_bytes"],
                         "prompt_tokens": r["fcv_prompt_tokens"] if use_fcv else r["ffv_prompt_tokens"]})
        raw, macro, task_acc = macro_of_qa_rows(rows)
        return {"n": len(rows), "raw": raw, "macro": macro, "task_acc": task_acc,
                "fcv_n": sum(1 for x in rows if x["use_fcv"]), "fcv_rate": 100.0 if use_fcv else 0.0,
                "totals": {k: sum(x[k] for x in rows) for k in ["pixels", "media_bytes", "prompt_tokens"]},
                "means": {k: statistics.mean([x[k] for x in rows]) for k in ["pixels", "media_bytes", "prompt_tokens"]},
                "transition": transition_counts(records, chosen)}
    base_full["fixed_ffv"] = fixed_eval(False)
    base_full["fixed_fcv"] = fixed_eval(True)

    # Full oracle on exact risk thresholds, using eval labels on the full 711 set.
    oracle = None
    for tau in exact_candidates:
        sel = eval_selection(records, tau)
        if oracle is None or sel["macro"] > oracle["macro"] - 1e-12 or (abs(sel["macro"] - oracle["macro"]) <= 1e-12 and sel["fcv_rate"] < oracle["fcv_rate"]):
            oracle = {**sel, "tau": tau}
    write_json(f"{BASE}/results/baseline_and_oracle_full.json", {
        "baselines": base_full,
        "oracle_exact_threshold": oracle,
        "grid_candidates": grid,
        "exact_candidate_count": len(exact_candidates),
    })

    # Ranking diagnostic
    beneficial, harmful, neutral = [], [], []
    for r in records:
        if not r["ffv_correct"] and r["fcv_correct"]:
            beneficial.append(r)
        elif r["ffv_correct"] and not r["fcv_correct"]:
            harmful.append(r)
        else:
            neutral.append(r)
    scores_all = [r["risk"] for r in records]
    labels_beneficial = [1 if (not r["ffv_correct"] and r["fcv_correct"]) else 0 for r in records]
    labels_harmful = [1 if (r["ffv_correct"] and not r["fcv_correct"]) else 0 for r in records]
    risk_sorted = sorted(records, key=lambda r: -r["risk"])
    top_risk = {}
    for x in [10, 20, 30, 40, 50]:
        n_top = max(1, int(round(x / 100.0 * len(risk_sorted))))
        top = risk_sorted[:n_top]
        p_ben = 100.0 * sum(1 for r in top if (not r["ffv_correct"] and r["fcv_correct"])) / n_top
        p_har = 100.0 * sum(1 for r in top if (r["ffv_correct"] and not r["fcv_correct"])) / n_top
        wc = sum(1 for r in top if (not r["ffv_correct"] and r["fcv_correct"]))
        cw = sum(1 for r in top if (r["ffv_correct"] and not r["fcv_correct"]))
        top_risk[str(x)] = {"n_top": n_top, "P_beneficial_pct": p_ben, "P_harmful_pct": p_har, "WC": wc, "CW": cw, "NetBenefit": wc - cw}
    ranking = {
        "n_total": len(records),
        "n_beneficial_WC": len(beneficial),
        "n_harmful_CW": len(harmful),
        "n_neutral": len(neutral),
        "ROC_AUC_beneficial_vs_others": auc_binary(scores_all, labels_beneficial),
        "PR_AUC_beneficial_vs_others": pr_auc(scores_all, labels_beneficial),
        "ROC_AUC_harmful_vs_others": auc_binary(scores_all, labels_harmful),
        "PR_AUC_harmful_vs_others": pr_auc(scores_all, labels_harmful),
        "beneficial_risk_mean": statistics.mean([r["risk"] for r in beneficial]) if beneficial else None,
        "harmful_risk_mean": statistics.mean([r["risk"] for r in harmful]) if harmful else None,
        "neutral_risk_mean": statistics.mean([r["risk"] for r in neutral]) if neutral else None,
        "top_risk": top_risk,
    }
    write_json(f"{BASE}/results/ranking_diagnostic.json", ranking)

    # Few-shot experiments
    seed_rows = []
    main_agg = defaultdict(list)
    acc_agg = defaultdict(list)
    mode_pairs = [
        ("quantile_cost_constrained", grid, "cost_constrained", "main"),
        ("quantile_accuracy_only", grid, "accuracy_only", "accuracy_only"),
        ("exact_cost_constrained", exact_candidates, "cost_constrained", "exact_cost_constrained"),
        ("exact_accuracy_only", exact_candidates, "accuracy_only", "exact_accuracy_only"),
    ]

    for K in KS:
        for split_idx, split_seed in enumerate(SPLIT_SEEDS):
            rng_split = random.Random(10000 + split_seed)
            perm = rng_split.sample(videos, len(videos))
            calib_videos = []
            calib_qa_ids = []
            cum = 0
            for v in perm:
                calib_videos.append(v)
                calib_qa_ids.extend(video_qas[v])
                cum += video_qa_counts[v]
                if cum >= K:
                    break
            calib_set = set(calib_qa_ids)
            eval_ids = [r["qa_id"] for r in records if r["qa_id"] not in calib_set]
            calib_records = [records_by_id[q] for q in calib_qa_ids]
            eval_records = [records_by_id[q] for q in eval_ids]

            fixed_ffv_eval = fixed_eval_subset(eval_records, False)
            fixed_fcv_eval = fixed_eval_subset(eval_records, True)
            frozen_eval = eval_selection(eval_records, FROZEN_TAU)
            # eval oracle over exact thresholds in evaluation set
            oracle_eval = None
            for tau in sorted(set(r["risk"] for r in eval_records)):
                sel = eval_selection(eval_records, tau)
                if oracle_eval is None or sel["macro"] > oracle_eval["macro"] - 1e-12 or (abs(sel["macro"] - oracle_eval["macro"]) <= 1e-12 and sel["fcv_rate"] < oracle_eval["fcv_rate"]):
                    oracle_eval = {**sel, "tau": tau}

            base_row = {
                "K": K, "split_seed": split_seed,
                "calib_qa": len(calib_records), "calib_videos": len(calib_videos),
                "eval_qa": len(eval_records), "eval_videos": len(videos) - len(calib_videos),
                "fixed_ffv_macro": fixed_ffv_eval["macro"], "fixed_ffv_raw": fixed_ffv_eval["raw"],
                "fixed_fcv_macro": fixed_fcv_eval["macro"], "fixed_fcv_raw": fixed_fcv_eval["raw"],
                "frozen_macro": frozen_eval["macro"], "frozen_raw": frozen_eval["raw"],
                "oracle_macro": oracle_eval["macro"], "oracle_tau": oracle_eval["tau"],
                "oracle_fcv_rate": oracle_eval["fcv_rate"],
                "best_fixed_macro": max(fixed_ffv_eval["macro"], fixed_fcv_eval["macro"]),
            }

            for mode_name, candidates, objective, agg_key in mode_pairs:
                chosen = choose_threshold(calib_records, candidates, objective=objective)
                tau = chosen["tau"]
                ev = eval_selection(eval_records, tau)
                # Random matched router only for the two main calibration modes to keep runtime bounded.
                if agg_key in ("main", "accuracy_only"):
                    rand = random_router_summary(eval_records, ev["fcv_n"], ROUTING_SEEDS_PER_SPLIT,
                                                 1000000 + split_idx * 1000 + K, ev["macro"])
                else:
                    rand = {"mean": float("nan"), "std": float("nan"), "median": float("nan"),
                            "q25": float("nan"), "q75": float("nan"), "q95": float("nan"), "win_rate_exact": float("nan")}
                row = {
                    **base_row,
                    "mode": mode_name,
                    "selected_tau": tau,
                    "calib_macro": chosen["calib_macro"], "calib_raw": chosen["calib_raw"],
                    "calib_fcv_rate": chosen["calib_fcv_rate"],
                    "eval_macro": ev["macro"], "eval_raw": ev["raw"], "eval_fcv_rate": ev["fcv_rate"],
                    "eval_pixels_total": ev["totals"]["pixels"], "eval_pixels_mean": ev["means"]["pixels"],
                    "eval_media_bytes_total": ev["totals"]["media_bytes"], "eval_media_bytes_mean": ev["means"]["media_bytes"],
                    "eval_prompt_tokens_total": ev["totals"]["prompt_tokens"], "eval_prompt_tokens_mean": ev["means"]["prompt_tokens"],
                    "eval_WC": ev["transition"]["WC"], "eval_CW": ev["transition"]["CW"],
                    "eval_CC": ev["transition"]["CC"], "eval_WW": ev["transition"]["WW"],
                    "random_mean": rand["mean"], "random_std": rand["std"], "random_median": rand["median"],
                    "random_q25": rand["q25"], "random_q75": rand["q75"], "random_q95": rand["q95"],
                    "gain_vs_frozen": ev["macro"] - frozen_eval["macro"],
                    "gain_vs_random": ev["macro"] - rand["mean"],
                    "gain_vs_best_fixed": ev["macro"] - max(fixed_ffv_eval["macro"], fixed_fcv_eval["macro"]),
                }
                if not math.isnan(rand["mean"]):
                    row["win_rate_vs_random"] = rand["win_rate_exact"]
                else:
                    row["win_rate_vs_random"] = float("nan")
                row["win_rate_vs_best_fixed"] = 1.0 if ev["macro"] > max(fixed_ffv_eval["macro"], fixed_fcv_eval["macro"]) else (0.0 if ev["macro"] < max(fixed_ffv_eval["macro"], fixed_fcv_eval["macro"]) else 0.5)
                seed_rows.append(row)
                if agg_key in ("main", "accuracy_only"):
                    (main_agg if agg_key == "main" else acc_agg)[K].append(row)

    # Write seed-level CSV
    fields = ["K", "split_seed", "mode", "calib_qa", "calib_videos", "eval_qa", "eval_videos",
              "selected_tau", "calib_macro", "calib_raw", "calib_fcv_rate", "eval_macro", "eval_raw",
              "eval_fcv_rate", "fixed_ffv_macro", "fixed_fcv_macro", "frozen_macro", "oracle_macro",
              "oracle_tau", "oracle_fcv_rate", "best_fixed_macro",
              "eval_pixels_total", "eval_pixels_mean", "eval_media_bytes_total", "eval_media_bytes_mean",
              "eval_prompt_tokens_total", "eval_prompt_tokens_mean", "eval_WC", "eval_CW", "eval_CC", "eval_WW",
              "random_mean", "random_std", "random_median", "random_q25", "random_q75", "random_q95",
              "gain_vs_frozen", "gain_vs_random", "gain_vs_best_fixed", "win_rate_vs_random", "win_rate_vs_best_fixed"]
    write_csv(f"{BASE}/results/fewshot_seed_results.csv", seed_rows, fields)
    write_jsonl(f"{BASE}/results/fewshot_seed_results.jsonl", seed_rows)

    # Aggregate per K for main and accuracy_only
    def aggregate(rows_by_K):
        out = []
        for K in KS:
            rows = rows_by_K.get(K, [])
            if not rows:
                continue
            vals = {}
            for key in ["calib_qa", "calib_videos", "eval_qa", "eval_videos", "selected_tau",
                        "calib_macro", "calib_raw", "calib_fcv_rate", "eval_macro", "eval_raw",
                        "eval_fcv_rate", "fixed_ffv_macro", "fixed_fcv_macro", "frozen_macro",
                        "oracle_macro", "eval_pixels_mean", "eval_media_bytes_mean", "eval_prompt_tokens_mean",
                        "random_mean", "gain_vs_frozen", "gain_vs_random", "gain_vs_best_fixed",
                        "win_rate_vs_random", "win_rate_vs_best_fixed"]:
                arr = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
                if not arr:
                    continue
                s = sorted(arr)
                def q(p):
                    idx = min(len(s) - 1, int(round(p * (len(s) - 1))))
                    return s[idx]
                vals[key] = {
                    "mean": statistics.mean(arr), "std": statistics.stdev(arr) if len(arr) > 1 else 0.0,
                    "median": statistics.median(arr), "q25": q(0.25), "q75": q(0.75),
                    "ci95_lo": q(0.025), "ci95_hi": q(0.975),
                }
            out.append({"K": K, "n_seeds": len(rows), **vals})
        return out

    write_json(f"{BASE}/results/fewshot_aggregate_main.json", aggregate(main_agg))
    write_json(f"{BASE}/results/fewshot_aggregate_accuracy_only.json", aggregate(acc_agg))

    print(json.dumps({
        "n_qa": len(records), "n_videos": len(videos),
        "baseline_full": {k: round(v["macro"], 4) for k, v in base_full.items()},
        "oracle_full": {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in oracle.items()},
        "ranking": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ranking.items() if k not in ("top_risk",)},
        "seed_rows": len(seed_rows),
    }, indent=2))


def fixed_eval_subset(records, use_fcv):
    rows = []
    chosen = {}
    for r in records:
        chosen[r["qa_id"]] = use_fcv
        rows.append({"qa_id": r["qa_id"], "task": r["task"], "video_id": r["video_id"],
                     "risk": r["risk"], "correct": r["fcv_correct"] if use_fcv else r["ffv_correct"],
                     "use_fcv": use_fcv,
                     "pixels": r["fcv_pixels"] if use_fcv else r["ffv_pixels"],
                     "media_bytes": r["fcv_media_bytes"] if use_fcv else r["ffv_media_bytes"],
                     "prompt_tokens": r["fcv_prompt_tokens"] if use_fcv else r["ffv_prompt_tokens"]})
    raw, macro, task_acc = macro_of_qa_rows(rows)
    return {"n": len(rows), "raw": raw, "macro": macro, "task_acc": task_acc,
            "fcv_n": sum(1 for x in rows if x["use_fcv"]), "fcv_rate": 100.0 if use_fcv else 0.0,
            "totals": {k: sum(x[k] for x in rows) for k in ["pixels", "media_bytes", "prompt_tokens"]},
            "means": {k: statistics.mean([x[k] for x in rows]) for k in ["pixels", "media_bytes", "prompt_tokens"]},
            "transition": transition_counts(records, chosen)}


if __name__ == "__main__":
    main()
