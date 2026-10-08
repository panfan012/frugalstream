#!/usr/bin/env python3
import json, math, sys
from collections import Counter, defaultdict

path="/workspace/focalstream_qwen3_crossmodel/manifests/frozen_visual_inputs.jsonl"
rows=[json.loads(l) for l in open(path)]
c=Counter(r['method'] for r in rows)
print('rows', len(rows), dict(c))

# future leakage: all selected timestamps <= query_timestamp
leak=0
for r in rows:
    t=float(r['query_timestamp'])
    for s in r['selected_frame_timestamps']:
        if s > t + 1e-9:
            leak+=1
            print('FUTURE LEAK', r['qa_id'], r['method'], s, t)
            if leak>5: sys.exit(1)
print('future leaks', leak)

# cumulative monotonicity of v2 per video
byvid=defaultdict(list)
for r in rows:
    if r['method']=='v2':
        byvid[r['video_id']].append(r)
bad=0
for vid, rs in byvid.items():
    rs.sort(key=lambda r:(int(math.ceil(float(r['query_timestamp']))), r['qa_id']))
    prev=0
    for r in rs:
        if len(r['selected_frame_timestamps']) < prev:
            bad+=1
            print('NON-MONOTONIC', vid, r['qa_id'])
        prev=len(r['selected_frame_timestamps'])
print('non-monotonic v2 rows', bad)

# expected counts
dev=sum(1 for r in rows if r['method']=='dense')
print('dense', dev, 'v2', c['v2'], 'ffv', c['ffv_c070'], 'fcv', c['fcv_fullhr'])
