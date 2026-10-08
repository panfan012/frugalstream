#!/usr/bin/env python3
"""Generate required PNG figures from Qwen cross-model results."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/workspace/focalstream_qwen3_crossmodel"
FIG = f"{BASE}/figures"
RES = f"{BASE}/results"
os.makedirs(FIG, exist_ok=True)

MINI = {
    "full_focal_macro": 63.8961,
    "full_dense_macro": 63.0881,
    "full_saving_pct": 80.33,
    "oof_router_gain_vs_random": 1.796,
    "oof_focal_macro": 78.6406,
    "oof_fixed_ffv": 77.0709,
    "oof_fixed_fcv": 76.4141,
    "voh_wc": 31,
    "voh_cw": 32,
}


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    # full dev
    detail = load_json(f"{RES}/qwen_full_dev_main.detail.json")
    methods = [d["method"] for d in detail]
    raw = [d["raw"] for d in detail]
    macro = [d["macro"] for d in detail]

    # fig 1 full dev accuracy
    x = np.arange(len(methods))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w/2, raw, w, label="Raw")
    ax.bar(x + w/2, macro, w, label="Macro")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Qwen3-Omni Full Development Accuracy")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/qwen_full_dev_accuracy.png", dpi=130); plt.close(fig)

    # fig 2 efficiency (frames/pixels/bytes from resource accounting)
    res_rows = []
    with open(f"{RES}/qwen_resource_accounting.csv", encoding="utf-8") as fh:
        import csv
        res_rows = list(csv.DictReader(fh))
    rmap = {r["method"]: r for r in res_rows}
    met_keys = [("pixels_mean", "Pixels/QA"), ("media_bytes_mean", "Bytes/QA")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (k, label) in zip(axes, met_keys):
        vals = [float(rmap[m][k]) for m in methods if m in rmap and rmap[m][k] not in (None, "")]
        ms = [m for m in methods if m in rmap and rmap[m][k] not in (None, "")]
        ax.bar(ms, vals)
        ax.set_title(label); ax.grid(axis="y", alpha=0.3)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    fig.tight_layout(); fig.savefig(f"{FIG}/qwen_efficiency.png", dpi=130); plt.close(fig)

    # fig 3 accuracy-cost scatter
    if methods and rmap:
        fig, ax = plt.subplots(figsize=(6, 4.4))
        for m in methods:
            if m in rmap and rmap[m]["pixels_mean"] not in (None, ""):
                ax.scatter(float(rmap[m]["pixels_mean"]), dict(zip(methods, macro))[m], s=90, label=m)
        ax.set_xlabel("Pixels/QA (Qwen-visible)")
        ax.set_ylabel("Macro accuracy (%)")
        ax.set_title("Accuracy vs visual input cost")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(f"{FIG}/qwen_accuracy_cost.png", dpi=130); plt.close(fig)

    # fig 4 routing ablation
    summ = load_json(f"{RES}/qwen_summary.json")
    oof = {d["method"]: d for d in summ["rtvp_oof"]}
    rr = load_json(f"{RES}/qwen_random_router_1000seeds.json")
    labels = ["Fixed FFV", "Fixed FCV", "Random matched", "FocalStream"]
    macro_vals = [
        oof["fixed_ffv"]["macro"],
        oof["fixed_fcv"]["macro"],
        rr["macro_mean"],
        oof["focalstream_oof"]["macro"],
    ]
    errs = [0, 0, rr["macro_std"], 0]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(labels, macro_vals, yerr=errs, capsize=5)
    ax.set_ylabel("RTVP OOF macro (%)")
    ax.set_title("Qwen routing ablation (711 QA)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/qwen_routing_ablation.png", dpi=130); plt.close(fig)

    # fig 5 random router distribution
    rand_rows = []
    with open(f"{RES}/qwen_random_router_1000seeds.csv", encoding="utf-8") as fh:
        import csv
        rand_rows = [float(r["macro"]) for r in csv.DictReader(fh)]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(rand_rows, bins=40, alpha=0.75)
    ax.axvline(oof["focalstream_oof"]["macro"], color="red", linestyle="--", label="FocalStream")
    ax.set_xlabel("Matched random router macro (%)")
    ax.set_ylabel("Seeds")
    ax.set_title("Random router distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/qwen_random_router_distribution.png", dpi=130); plt.close(fig)

    # fig 6 VOH transition
    tr = load_json(f"{RES}/qwen_ffv_fcv_transition.json")["counts"]
    labels = ["FFV wrong→FCV correct (WC)", "FFV correct→FCV wrong (CW)", "Both correct (CC)", "Both wrong (WW)"]
    vals = [tr["WC"], tr["CW"], tr["CC"], tr["WW"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals)
    ax.set_ylabel("Count (711 QA)")
    ax.set_title("Qwen FFV→FCV transition")
    fig.tight_layout(); fig.savefig(f"{FIG}/qwen_voh_transition.png", dpi=130); plt.close(fig)

    # cross-backend figures (combine with MiniCPM frozen reference)
    q_full_focal = dict(zip(methods, macro)).get("v4", 0.0)
    q_full_dense = dict(zip(methods, macro)).get("dense", 0.0)

    # fig 7 accuracy delta
    fig, ax = plt.subplots(figsize=(5, 4.2))
    backends = ["MiniCPM-o 4.5", "Qwen3-Omni"]
    deltas = [MINI["full_focal_macro"] - MINI["full_dense_macro"], q_full_focal - q_full_dense]
    ax.bar(backends, deltas)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("FocalStream − Dense macro (pp)")
    ax.set_title("Cross-backend accuracy delta")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/cross_backend_accuracy_delta.png", dpi=130); plt.close(fig)

    # fig 8 efficiency (saving %; Qwen uses frozen-plan if available else visible)
    q_saving = _q_saving(rmap, methods)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.bar(backends, [MINI["full_saving_pct"], q_saving])
    ax.set_ylabel("FocalStream input saving vs Dense (%)")
    ax.set_title("Cross-backend efficiency")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/cross_backend_efficiency.png", dpi=130); plt.close(fig)

    # fig 9 routing gain vs random
    q_gain = oof["focalstream_oof"]["macro"] - rr["macro_mean"]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.bar(backends, [MINI["oof_router_gain_vs_random"], q_gain])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Router gain vs random (pp)")
    ax.set_title("Cross-backend routing benefit")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/cross_backend_routing_gain.png", dpi=130); plt.close(fig)

    # fig 10 VOH
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2); w = 0.35
    ax.bar(x - w/2, [MINI["voh_wc"], tr["WC"]], w, label="WC (beneficial)")
    ax.bar(x + w/2, [MINI["voh_cw"], tr["CW"]], w, label="CW (harmful)")
    ax.set_xticks(x); ax.set_xticklabels(backends)
    ax.set_ylabel("FFV→FCV flips")
    ax.set_title("Visual over-provisioning harm")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/cross_backend_voh.png", dpi=130); plt.close(fig)

    # fig 11 latency distribution
    lat_rows = []
    with open(f"{RES}/qwen_latency.csv", encoding="utf-8") as fh:
        import csv
        lat_rows = list(csv.DictReader(fh))
    preds_path = f"{BASE}/predictions/qwen_predictions.jsonl"
    lats = []
    if os.path.exists(preds_path):
        for line in open(preds_path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") == "OK" and isinstance(r.get("latency_s"), (int, float)):
                lats.append(r["latency_s"])
    if lats:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.hist(np.clip(lats, 0, np.percentile(lats, 99)), bins=40, alpha=0.75)
        ax.set_xlabel("Client E2E latency (s)")
        ax.set_ylabel("Requests")
        ax.set_title("Qwen latency distribution")
        fig.tight_layout(); fig.savefig(f"{FIG}/qwen_latency_distribution.png", dpi=130); plt.close(fig)

    print("figures written to", FIG)


def _q_saving(rmap, methods):
    # Use Qwen-visible pixels when available (documented; Dense is 448x252-compatible)
    try:
        d = float(rmap["dense"]["pixels_mean"])
        f = float(rmap["v4"]["pixels_mean"])
        return round(100.0 * (1 - f / d), 2) if d else 0.0
    except Exception:
        return 0.0


if __name__ == "__main__":
    main()
