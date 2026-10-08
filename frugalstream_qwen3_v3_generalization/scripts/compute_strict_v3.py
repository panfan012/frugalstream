#!/usr/bin/env python3
"""Recompute strict frozen V3/V4 Qwen results from existing 7240 requests. 0 new calls."""
import csv, importlib.util, json, statistics, sys, os

BASE = "/workspace/frugalstream_qwen3_v3_generalization"
CROSS = "/workspace/focalstream_qwen3_crossmodel"
os.makedirs(f"{BASE}/results", exist_ok=True)

spec = importlib.util.spec_from_file_location("v2mod", "/root/eval_minicpmo45_ovobench_full_v2.py")
V2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(V2)

def load_jsonl(p):
    out=[]
    for line in open(p, encoding='utf-8'):
        line=line.strip()
        if line: out.append(json.loads(line))
    return out

dev_rows = load_jsonl(f"{CROSS}/manifests/dev_2909_qa.jsonl")
rtvp_rows = load_jsonl(f"{CROSS}/manifests/rtvp_oof_711_qa.jsonl")
preds = load_jsonl(f"{CROSS}/predictions/qwen_predictions.jsonl")

dev = {r['qa_id']: r for r in dev_rows}
rtvp = {r['qa_id']: r for r in rtvp_rows}
P = {}
for r in preds:
    if r.get('status') == 'OK':
        P[(r['qa_id'], r['method'])] = r

TAU3 = 0.2307592643
TAU4 = 0.3554
CATS = ['Real-Time Visual Perception','Backward Tracing','Forward Active Responding']

def score(row, raw):
    ev = {'kind': row['kind'], 'ground_truth': row['ground_truth']}
    if row['kind'] in ('ssr','crr'):
        ev['test_type'] = 1 if row['ground_truth'] == 'Yes' else 0
    correct, pred, invalid, proc_error, status = V2.score_event(ev, raw)
    return {'correct': bool(correct), 'pred': pred, 'invalid': bool(invalid), 'status': status}

def acc(rows):
    if not rows: return 0.0, 0.0, {}
    raw = 100.0*sum(1 for r in rows if r['correct'])/len(rows)
    cat={}
    for c in CATS:
        sub=[r for r in rows if r['category']==c]
        cat[c]=100.0*sum(1 for r in sub if r['correct'])/len(sub) if sub else 0.0
    macro=statistics.mean(cat[c] for c in CATS)
    return raw, macro, cat

def task_macro(rows):
    if not rows: return 0.0, 0.0, {}
    raw=100.0*sum(1 for r in rows if r['correct'])/len(rows)
    per={}
    for r in rows:
        per.setdefault(r['task'],[0,0]); per[r['task']][0]+=r['correct']; per[r['task']][1]+=1
    macro=100.0*statistics.mean(a/b for a,b in per.values())
    return raw, macro, {t:100.0*a/b for t,(a,b) in per.items()}

def cost_fields(p):
    return {k:p.get(k) for k in ['pixels','media_bytes','prompt_tokens','completion_tokens','long_frames_used','refill_frames_used','latency_s']}

# full-dev method maps
full={}
for method in ['dense','v2']:
    mp={}
    for qa in dev:
        p=P.get((qa,method))
        if not p: continue
        s=score(dev[qa], p['raw_answer'])
        mp[qa]={**s, **cost_fields(p)}
    full[method]=mp

def _score_from_method(qa, method):
    # use already-scored cache when available for full RTVP map
    p=P.get((qa,method))
    if not p: return None
    s=score(dev[qa], p['raw_answer'])
    return {**s, **cost_fields(p)}

# but v3/v4 above call helper before definition. Rebuild with safe function.
def build_derived():
    v3={}; v4={}
    for qa,d in dev.items():
        if qa not in full['v2']: continue
        if qa in rtvp:
            risk=float(rtvp[qa]['risk'])
            v3[qa] = full['v2'][qa] if risk < TAU3 else _score_from_method(qa,'fcv_fixed')
            v4[qa] = _score_from_method(qa,'fcv_fixed' if risk > TAU4 else 'ffv_fixed')
        else:
            v3[qa]=full['v2'][qa]; v4[qa]=full['v2'][qa]
    full['v3']=v3; full['v4']=v4
build_derived()

full_out=[]
for method in ['dense','v2','v3','v4']:
    rows=[]
    for qa,mp in full[method].items():
        if mp is None: continue
        rows.append({'qa_id':qa, **dev[qa], **mp})
    raw,macro,cat=acc(rows)
    vals={k:[] for k in ['pixels','media_bytes','prompt_tokens','long_frames_used','refill_frames_used','latency_s']}
    for r in rows:
        for k in vals:
            if isinstance(r.get(k),(int,float)): vals[k].append(r[k])
    def mean(k):
        return statistics.mean(vals[k]) if vals[k] else None
    def total(k):
        return sum(vals[k])
    full_out.append({
        'method':method,'n':len(rows),'raw':round(raw,4),'macro':round(macro,4),'cat_macro':{c:round(v,4) for c,v in cat.items()},
        'pixels_total':int(total('pixels')),'pixels_per_qa':round(mean('pixels'),2),
        'media_bytes_total':int(total('media_bytes')),'media_bytes_per_qa':round(mean('media_bytes'),2),
        'prompt_tokens_total':int(total('prompt_tokens')),'prompt_tokens_per_qa':round(mean('prompt_tokens'),2),
        'frames_per_qa':round(mean('long_frames_used')+(mean('refill_frames_used') or 0),2) if vals['long_frames_used'] else None,
        'latency_mean':round(mean('latency_s'),4),
    })

# RTVP V3
rtvp_ffv={qa:_score_from_method(qa,'ffv_fixed') for qa in rtvp}
rtvp_fcv={qa:_score_from_method(qa,'fcv_fixed') for qa in rtvp}
rtvp_v2={qa:_score_from_method(qa,'v2') for qa in rtvp}
rtvp_out=[]
for method, mp in [('v2',rtvp_v2),('v3',None),('v4_frozen_tau',None),('fixed_ffv',rtvp_ffv),('fixed_fcv',rtvp_fcv)]:
    if mp is None:
        mp={}
        for qa,r in rtvp.items():
            risk=float(r['risk'])
            if method=='v3':
                mp[qa]=rtvp_v2.get(qa) if risk<TAU3 else rtvp_fcv.get(qa)
            elif method=='v4_frozen_tau':
                mp[qa]=rtvp_ffv.get(qa) if risk<=TAU4 else rtvp_fcv.get(qa)
    rows=[]
    for qa,ans in mp.items():
        if ans is None: continue
        rows.append({'qa_id':qa, **rtvp[qa], **ans})
    raw,macro,task=task_macro(rows)
    cost={}
    for k in ['pixels','media_bytes','prompt_tokens']:
        vals=[r.get(k) for r in rows if isinstance(r.get(k),(int,float))]
        cost[k]=sum(vals)
    rtvp_out.append({'method':method,'n':len(rows),'raw':round(raw,4),'macro':round(macro,4),'task_acc':{t:round(v,4) for t,v in task.items()},
        'pixels_total':int(cost['pixels']),'pixels_per_qa':round(cost['pixels']/len(rows),2),
        'media_bytes_total':int(cost['media_bytes']),'media_bytes_per_qa':round(cost['media_bytes']/len(rows),2),
        'prompt_tokens_total':int(cost['prompt_tokens']),'prompt_tokens_per_qa':round(cost['prompt_tokens']/len(rows),2)})

# V3 vs Dense saving
dense=next(r for r in full_out if r['method']=='dense')
v3full=next(r for r in full_out if r['method']=='v3')
v4full=next(r for r in full_out if r['method']=='v4')
def saving(v, ref):
    if ref==0: return None
    return round(100.0*(1-v/ref),2)
out={
 'full_dev':full_out,
 'rtvp_dev':rtvp_out,
 'tau3':TAU3,'tau4':TAU4,
 'v3_delta_vs_dense_macro_pp':round(v3full['macro']-dense['macro'],4),
 'v3_saving_vs_dense':{k:saving(v3full[f'{k}_total'],dense[f'{k}_total']) for k in ['pixels','media_bytes','prompt_tokens']},
 'v4_saving_vs_dense':{k:saving(v4full[f'{k}_total'],dense[f'{k}_total']) for k in ['pixels','media_bytes','prompt_tokens']},
 'new_qwen_calls':0,
}
with open(f"{BASE}/results/strict_v3_qwen_results.json","w") as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
