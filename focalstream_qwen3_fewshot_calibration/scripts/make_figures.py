#!/usr/bin/env python3
"""Generate few-shot calibration figures from offline results."""
import json
import os
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "/workspace/focalstream_qwen3_fewshot_calibration"
OUT = f"{BASE}/figures"
os.makedirs(OUT, exist_ok=True)

main = json.load(open(f"{BASE}/results/fewshot_aggregate_main.json"))
acc = json.load(open(f"{BASE}/results/fewshot_aggregate_accuracy_only.json"))
base = json.load(open(f"{BASE}/results/baseline_and_oracle_full.json"))
rank = json.load(open(f"{BASE}/results/ranking_diagnostic.json"))
recs = [json.loads(l) for l in open(f"{BASE}/manifests/fewshot_qa_records.jsonl")]

def means(rows, key):
    return [r[key]['mean'] for r in rows], [r[key]['std'] for r in rows], [r['K'] for r in rows]

def line_with_ci(ax, rows, key, label, color):
    x = [r['K'] for r in rows]
    y = [r[key]['mean'] for r in rows]
    lo = [r[key]['ci95_lo'] for r in rows]
    hi = [r[key]['ci95_hi'] for r in rows]
    ax.plot(x, y, marker='o', label=label, color=color)
    ax.fill_between(x, lo, hi, alpha=0.18, color=color)
    return x, y

# 1 macro vs K
fig, ax = plt.subplots(figsize=(7,4.5))
line_with_ci(ax, main, 'eval_macro', 'cost-constrained (formal)', 'tab:blue')
line_with_ci(ax, acc, 'eval_macro', 'accuracy-only (sensitivity)', 'tab:orange')
ax.axhline(base['baselines']['fixed_ffv']['macro'], color='gray', ls='--', label='Fixed FFV')
ax.axhline(base['baselines']['fixed_fcv']['macro'], color='black', ls='--', label='Fixed FCV')
ax.axhline(base['oracle_exact_threshold']['macro'], color='green', ls=':', label='Oracle threshold')
ax.set_xscale('log', base=2)
ax.set_xticks([4,8,16,32,64,128]); ax.set_xticklabels([4,8,16,32,64,128])
ax.set_xlabel('Calibration QA (target K, log2)')
ax.set_ylabel('Evaluation Macro Accuracy (%)')
ax.set_title('Few-shot threshold calibration: Macro vs K')
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fewshot_macro_vs_K.png", dpi=160); plt.close(fig)

# 2 gain vs random
fig, ax = plt.subplots(figsize=(7,4.5))
line_with_ci(ax, main, 'gain_vs_random', 'cost-constrained (formal)', 'tab:blue')
line_with_ci(ax, acc, 'gain_vs_random', 'accuracy-only (sensitivity)', 'tab:orange')
ax.axhline(0, color='gray', ls='--')
ax.set_xscale('log', base=2); ax.set_xticks([4,8,16,32,64,128]); ax.set_xticklabels([4,8,16,32,64,128])
ax.set_xlabel('Calibration QA (target K, log2)')
ax.set_ylabel('Gain vs matched random (pp)')
ax.set_title('Few-shot router gain vs matched random')
ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fewshot_gain_vs_random.png", dpi=160); plt.close(fig)

# 3 gain vs best fixed
fig, ax = plt.subplots(figsize=(7,4.5))
line_with_ci(ax, main, 'gain_vs_best_fixed', 'cost-constrained (formal)', 'tab:blue')
line_with_ci(ax, acc, 'gain_vs_best_fixed', 'accuracy-only (sensitivity)', 'tab:orange')
ax.axhline(0, color='gray', ls='--')
ax.set_xscale('log', base=2); ax.set_xticks([4,8,16,32,64,128]); ax.set_xticklabels([4,8,16,32,64,128])
ax.set_xlabel('Calibration QA (target K, log2)')
ax.set_ylabel('Gain vs best fixed FFV/FCV (pp)')
ax.set_title('Few-shot router gain vs best fixed policy')
ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fewshot_gain_vs_best_fixed.png", dpi=160); plt.close(fig)

# 4 selected tau
fig, ax = plt.subplots(figsize=(7,4.5))
line_with_ci(ax, main, 'selected_tau', 'cost-constrained (formal)', 'tab:blue')
line_with_ci(ax, acc, 'selected_tau', 'accuracy-only (sensitivity)', 'tab:orange')
ax.axhline(base['baselines']['frozen_tau']['tau'] if 'tau' in base['baselines']['frozen_tau'] else 0.3554, color='gray', ls='--', label='Frozen tau=0.3554')
ax.set_xscale('log', base=2); ax.set_xticks([4,8,16,32,64,128]); ax.set_xticklabels([4,8,16,32,64,128])
ax.set_xlabel('Calibration QA (target K, log2)')
ax.set_ylabel('Selected tau')
ax.set_title('Selected threshold vs K')
ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT}/threshold_vs_K.png", dpi=160); plt.close(fig)

# 5 escalation rate
fig, ax = plt.subplots(figsize=(7,4.5))
line_with_ci(ax, main, 'eval_fcv_rate', 'cost-constrained (formal)', 'tab:blue')
line_with_ci(ax, acc, 'eval_fcv_rate', 'accuracy-only (sensitivity)', 'tab:orange')
ax.axhline(43.46, color='gray', ls='--', label='Original V4 escalation 43.46%')
ax.axhspan(38.46, 48.46, color='green', alpha=0.08, label='Calibration budget window')
ax.set_xscale('log', base=2); ax.set_xticks([4,8,16,32,64,128]); ax.set_xticklabels([4,8,16,32,64,128])
ax.set_xlabel('Calibration QA (target K, log2)')
ax.set_ylabel('Evaluation FCV escalation rate (%)')
ax.set_title('Escalation rate vs K')
ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT}/escalation_rate_vs_K.png", dpi=160); plt.close(fig)

# 6 risk score distributions by FFV->FCV transition class
fig, axes = plt.subplots(1, 3, figsize=(12,3.6), sharey=True)
classes = [
    ('beneficial_WC', [r for r in recs if (not r['ffv_correct'] and r['fcv_correct'])]),
    ('harmful_CW', [r for r in recs if (r['ffv_correct'] and not r['fcv_correct'])]),
    ('neutral', [r for r in recs if not ((not r['ffv_correct'] and r['fcv_correct']) or (r['ffv_correct'] and not r['fcv_correct']))]),
]
colors = ['tab:green','tab:red','gray']
for ax, (name, rows), color in zip(axes, classes, colors):
    vals = [r['risk'] for r in rows]
    ax.hist(vals, bins=20, range=(0,1), color=color, alpha=0.65)
    ax.set_title(f"{name}\n(n={len(vals)})")
    ax.set_xlabel('QRE risk score')
axes[0].set_ylabel('QA count')
fig.suptitle('QRE score distribution by FFV -> FCV transition class')
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(f"{OUT}/qre_score_beneficial_harmful_distribution.png", dpi=160); plt.close(fig)

# 7 top-risk net benefit
xs = sorted(int(k) for k in rank['top_risk'])
net = [rank['top_risk'][str(x)]['NetBenefit'] for x in xs]
pben = [rank['top_risk'][str(x)]['P_beneficial_pct'] for x in xs]
phar = [rank['top_risk'][str(x)]['P_harmful_pct'] for x in xs]
fig, ax1 = plt.subplots(figsize=(7,4.5))
ax1.plot(xs, pben, marker='o', color='tab:green', label='P(beneficial WC)')
ax1.plot(xs, phar, marker='s', color='tab:red', label='P(harmful CW)')
ax1.set_xlabel('Top-risk percentile X (%)')
ax1.set_ylabel('Percent of top-risk QA (%)', color='black')
ax1.grid(alpha=0.25)
ax2 = ax1.twinx()
ax2.plot(xs, net, marker='^', color='tab:blue', ls='--', label='Net benefit (#WC-#CW)')
ax2.set_ylabel('Net benefit (#WC-#CW)', color='tab:blue')
lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='center right')
ax1.set_title('QRE top-risk behavior')
fig.tight_layout(); fig.savefig(f"{OUT}/qre_toprisk_netbenefit.png", dpi=160); plt.close(fig)

# 8 oracle vs fewshot vs frozen
labels = ['Fixed FFV','Frozen tau\n(pure)','Fixed FCV','Oracle\nthreshold']
vals = [
    base['baselines']['fixed_ffv']['macro'],
    base['baselines']['frozen_tau']['macro'],
    base['baselines']['fixed_fcv']['macro'],
    base['oracle_exact_threshold']['macro'],
]
colors = ['gray','tab:red','black','green']
fig, ax = plt.subplots(figsize=(8.2,4.6))
x = list(range(len(labels)))
ax.bar(x, vals, color=colors, alpha=0.78)
for xi, v in zip(x, vals):
    ax.text(xi, v+0.05, f'{v:.2f}', ha='center', fontsize=8)
for K in [8,16,32,64]:
    r = next(r for r in main if r['K']==K)
    ax.axhline(r['eval_macro']['mean'], color='tab:blue', alpha=0.25, lw=0.8)
    ax.text(len(labels)-0.42, r['eval_macro']['mean']+0.03, f'K={K}', color='tab:blue', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Macro Accuracy (%)')
ax.set_ylim(73.5, 77.0)
ax.set_title('Oracle vs few-shot vs frozen threshold (full 711 RTVP dev)')
ax.grid(axis='y', alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT}/oracle_vs_fewshot_vs_frozen.png", dpi=160); plt.close(fig)

print('figures written to', OUT)
