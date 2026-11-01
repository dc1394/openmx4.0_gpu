#!/usr/bin/env python3
"""Retain actual solver traces separately from GPU admission banners."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

B=Path(__file__).resolve().parent
pattern=re.compile(r'GPUEIGENPROF rank=(\d+) n=(\d+) maxn=(\d+) type=(\w+) backend=(\w+) attempt=(\d+) seconds=([\d.]+) info=(-?\d+) mout=(\d+)')
rows=[]
for workload in ('S','L','L2','L3','sidia333_nc_cluster','cghelgeo'):
    folder=B/'runs/fork_gpu'/workload
    path=folder/('together/run.log' if workload in ('S','L','L2','L3') else 'run.log')
    status=folder.parent/(workload+'.status.json')
    if not path.exists():continue
    d=json.loads(status.read_text())
    text=path.read_text()
    groups=defaultdict(list)
    for match in pattern.finditer(text):
        rank,n,maxn,kind,backend,attempt,seconds,info,mout=match.groups()
        groups[(backend,int(n),kind)].append(dict(rank=int(rank),maxn=int(maxn),attempt=int(attempt),
                                                 seconds=float(seconds),info=int(info),mout=int(mout)))
    summary=[]
    for (backend,n,kind),calls in sorted(groups.items()):
        summary.append(dict(backend=backend,n=n,type=kind,calls=len(calls),
                            ranks=sorted({c['rank'] for c in calls}),
                            solve_seconds_min=min(c['seconds'] for c in calls),
                            solve_seconds_max=max(c['seconds'] for c in calls),
                            solve_seconds_sum=sum(c['seconds'] for c in calls),
                            nonzero_info_calls=sum(c['info']!=0 for c in calls),
                            retries=sum(c['attempt']>1 for c in calls)))
    selections=Counter(line.strip() for line in text.splitlines() if re.search(
        r"GPU 'gfx|MAGMA collinear dense GPU path|MI300A k-point|MI300A cached density|cooperative MAGMA|CPU-solve pool|switching to CPU path",line))
    fallback=Counter(line.strip() for line in text.splitlines() if re.search(
        r'GPU device is shared|switching to CPU path|CPU-solve pool|capacity fallback',line,re.I))
    rows.append(dict(workload=workload,status=d['status'],binary_sha256=d['binary_sha256'],
                     solver_profiles=summary,selection_message_counts=dict(selections),
                     cpu_fallback_message_counts=dict(fallback),
                     note='Aggregate solver calls include overlap solves, multiple SCF steps, ranks and cases. Their sum is not a suite wall time; small/local solver paths need not emit this profiler.'))
out=dict(complete=len(rows)==6 and all(r['status']=='completed' for r in rows),workloads=rows)
(B/'gpu-dispatch.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(complete=out['complete'],solver_calls=sum(c['calls'] for r in rows for c in r['solver_profiles']),
                      nonzero_info_calls=sum(c['nonzero_info_calls'] for r in rows for c in r['solver_profiles']),
                      fallback_messages=sum(sum(r['cpu_fallback_message_counts'].values()) for r in rows))))
