"""Greedy speedup per (gamma, device, domain) with 95% bootstrap CI over prompts, and the per-position alpha implied
by server accepted/drafted (E = 1 + gamma*acc = (1 - a^(gamma+1))/(1 - a)).

    python3 src/measure/b3_summarize.py RESULTS/metrics.jsonl [greedy|sampled]
"""
import json,sys,random
rows=[json.loads(l) for l in open(sys.argv[1])]
mode=sys.argv[2] if len(sys.argv)>2 else 'greedy'
base={(r['mode'],r['prompt_id']):r['tokens_per_s'] for r in rows if r['device']=='none'}
g={}
for r in rows:
  if r['device']=='none' or r['error'] or r['mode']!=mode: continue
  g.setdefault((r['gamma'],r['device'],r['domain']),[]).append((r['tokens_per_s']/base[(r['mode'],r['prompt_id'])], r['draft_n'], r['draft_n_accepted']))
random.seed(0)
for k in sorted(g):
  v=[x[0] for x in g[k]]; m=sum(v)/len(v)
  bs=sorted(sum(random.choices(v,k=len(v)))/len(v) for _ in range(2000))
  acc=sum(x[2] for x in g[k])/sum(x[1] for x in g[k]); gm=k[0]; E=1+gm*acc
  # solve (1-a^(g+1))/(1-a)=E
  lo,hi=0.0,0.9999
  for _ in range(60):
    a=(lo+hi)/2; e=(1-a**(gm+1))/(1-a)
    lo,hi=(a,hi) if e<E else (lo,a)
  print(*k, len(v), f"{m:.2f} ({bs[50]:.2f}-{bs[1949]:.2f})", f"acc={acc:.3f} alpha={a:.2f}")
