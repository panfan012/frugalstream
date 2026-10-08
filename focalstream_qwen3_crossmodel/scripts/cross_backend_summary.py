#!/usr/bin/env python3
"""Build cross_backend_summary.csv and cross_backend_voh.csv."""
import csv
import json
import os

BASE = "/workspace/focalstream_qwen3_crossmodel"
RES = f"{BASE}/results"

MINI_FULL = {
    "dense_raw": 59.2987, "dense_macro": 63.0881,
    "v2_raw": 57.7174, "v2_macro": 60.8101,
    "v3_raw": 59.5737, "v3_macro": 63.1347,
    "v4_raw": 60.0206, "v4_macro": 63.8961,
    "saving_pct": 80.33,
}
MINI_OOF = {
    "ffv_raw": 76.5120, "ffv_macro": 77.0709,
    "fcv_raw": 76.3713, "fcv_macro": 76.4141,
    "focal_raw": 78.1997, "focal_macro": 78.6406,
    "random_macro_mean": 76.8446, "random_macro_std": 0.5702,
    "router_gain_random": 1.796,
    "wc": 31, "cw": 32,
}


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    summ = jload(f"{RES}/qwen_summary.json")
    full = {d["method"]: d for d in summ["full_dev"]}
    oof = {d["method"]: d for d in summ["rtvp_oof"]}
    rr = jload(f"{RES}/qwen_random_router_1000seeds.json")
    tr = summ["ffv_fcv_transition"]
    q_dense = full["dense"]; q_v2 = full["v2"]; q_v3 = full["v3"]; q_v4 = full["v4"]
    q_gain_random = oof["focalstream_oof"]["macro"] - rr["macro_mean"]

    rows = [
        {"metric": "full_dev_dense_raw", "minicpm": MINI_FULL["dense_raw"], "qwen": q_dense["raw"]},
        {"metric": "full_dev_dense_macro", "minicpm": MINI_FULL["dense_macro"], "qwen": q_dense["macro"]},
        {"metric": "full_dev_v2_raw", "minicpm": MINI_FULL["v2_raw"], "qwen": q_v2["raw"]},
        {"metric": "full_dev_v2_macro", "minicpm": MINI_FULL["v2_macro"], "qwen": q_v2["macro"]},
        {"metric": "full_dev_v3_raw", "minicpm": MINI_FULL["v3_raw"], "qwen": q_v3["raw"]},
        {"metric": "full_dev_v3_macro", "minicpm": MINI_FULL["v3_macro"], "qwen": q_v3["macro"]},
        {"metric": "full_dev_focal_raw", "minicpm": MINI_FULL["v4_raw"], "qwen": q_v4["raw"]},
        {"metric": "full_dev_focal_macro", "minicpm": MINI_FULL["v4_macro"], "qwen": q_v4["macro"]},
        {"metric": "full_dev_focal_delta_macro_pp", "minicpm": round(MINI_FULL["v4_macro"] - MINI_FULL["dense_macro"], 4),
         "qwen": round(q_v4["macro"] - q_dense["macro"], 4)},
        {"metric": "oof_fixed_ffv_macro", "minicpm": MINI_OOF["ffv_macro"], "qwen": oof["fixed_ffv"]["macro"]},
        {"metric": "oof_fixed_fcv_macro", "minicpm": MINI_OOF["fcv_macro"], "qwen": oof["fixed_fcv"]["macro"]},
        {"metric": "oof_focal_macro", "minicpm": MINI_OOF["focal_macro"], "qwen": oof["focalstream_oof"]["macro"]},
        {"metric": "oof_random_macro_mean", "minicpm": MINI_OOF["random_macro_mean"], "qwen": rr["macro_mean"]},
        {"metric": "oof_router_gain_vs_random_pp", "minicpm": MINI_OOF["router_gain_random"], "qwen": round(q_gain_random, 4)},
        {"metric": "ffv_fcv_wc", "minicpm": MINI_OOF["wc"], "qwen": tr["WC"]},
        {"metric": "ffv_fcv_cw", "minicpm": MINI_OOF["cw"], "qwen": tr["CW"]},
    ]

    with open(f"{RES}/cross_backend_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "minicpm", "qwen"])
        for r in rows:
            w.writerow([r["metric"], r["minicpm"], r["qwen"]])

    voh_rows = [
        {"backend": "MiniCPM-o 4.5", "wc": MINI_OOF["wc"], "cw": MINI_OOF["cw"], "n": 711},
        {"backend": "Qwen3-Omni", "wc": tr["WC"], "cw": tr["CW"], "n": 711},
    ]
    with open(f"{RES}/cross_backend_voh.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["backend", "wc", "cw", "n"])
        w.writeheader()
        for r in voh_rows:
            w.writerow(r)
    print("wrote cross_backend_summary.csv and cross_backend_voh.csv")


if __name__ == "__main__":
    main()
