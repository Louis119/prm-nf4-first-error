# -*- coding: utf-8 -*-
"""A2: 同一阈值下 FSR vs Step-BCE 的配对题组 bootstrap"""
import json, random, math
NORM="/Users/louis/Documents/ChatGPT/lunwen/analysis/prm_v11_completion_20261109/work/normalized"
EVAL="/Users/louis/Documents/ChatGPT/lunwen/analysis/prm_precision_repair_20261108/data/evaluation_primary.jsonl"
g={}
with open(EVAL) as f:
    for line in f:
        line=line.strip()
        if not line: continue
        d=json.loads(line); g[d["id"]]=d["group_id"]
def load(fn):
    out={}
    with open(f"{NORM}/{fn}") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            d=json.loads(line)
            d["group_id"]=g.get(d["id"],"NA:"+d["id"])
            out[d["id"]]=d
    return out
FSR=load("V11-math_shepherd7b-FSR-b10.jsonl")
STEP=load("V11-math_shepherd7b-Step-BCE-b10.jsonl")
ids=sorted(set(FSR)&set(STEP))
print("paired items:",len(ids),"groups:",len({FSR[i]['group_id'] for i in ids}))
TAU={"F20_nf4":0.24508501589298248,"F20_bf16":0.22270014882087708,"fixed_0.5":0.5}
def hit(d,tau,le=False):
    for j,s in enumerate(d["scores"]):
        if (s<=tau) if le else (s<tau):
            return 1 if (d["label"]!=-1 and j==d["label"]) else 0
    return 0
def fa(d,tau,le=False):
    if d["label"]!=-1: return 0
    for j,s in enumerate(d["scores"]):
        if (s<=tau) if le else (s<tau): return 1
    return 0
E=[i for i in ids if FSR[i]["label"]!=-1]; C=[i for i in ids if FSR[i]["label"]==-1]
groups=sorted({FSR[i]["group_id"] for i in ids})
rng=random.Random(20261110)
def boot(tau,le=False,which="L"):
    idx=E if which=="L" else C
    f=lambda d: hit(d,tau,le) if which=="L" else fa(d,tau,le)
    base_f=[f(FSR[i]) for i in idx]; base_s=[f(STEP[i]) for i in idx]
    members={}
    for k,i in enumerate(idx): members.setdefault(FSR[i]["group_id"],[]).append(k)
    obs=(sum(base_f)-sum(base_s))/len(idx)*100
    draws=[]
    for _ in range(10000):
        pick=[rng.choice(groups) for _ in groups]
        ks=[k for gid in pick for k in members.get(gid,())]
        if not ks: continue
        d=(sum(base_f[k] for k in ks)-sum(base_s[k] for k in ks))/len(ks)*100
        draws.append(d)
    draws.sort()
    return dict(obs=obs, lo=draws[int(0.025*len(draws))], hi=draws[int(0.975*len(draws))], n=len(idx), nb=len(draws))
print("\n=== 同一阈值：FSR − Step-BCE（百分点，配对题组 bootstrap, 10,000 次, seed 20261110）===")
out={}
for name,tau in TAU.items():
    for le in (False,True):
        L=boot(tau,le,"L"); F=boot(tau,le,"F")
        key=f"{name}/{'<=' if le else '<'}"
        out[key]=dict(L=L,F=F)
        print(f"{key:18s} ΔL={L['obs']:+.3f} [{L['lo']:+.2f},{L['hi']:+.2f}]  ΔF={F['obs']:+.3f} [{F['lo']:+.2f},{F['hi']:+.2f}]  (E={L['n']},C={F['n']})")
# 复现原稿的不等阈值比较
Ls_f=sum(hit(FSR[i],TAU["F20_bf16"]) for i in E)/len(E)*100
Ls_s=sum(hit(STEP[i],TAU["F20_nf4"]) for i in E)/len(E)*100
Fs_f=sum(fa(FSR[i],TAU["F20_bf16"]) for i in C)/len(C)*100
Fs_s=sum(fa(STEP[i],TAU["F20_nf4"]) for i in C)/len(C)*100
out["published_unequal_tau"]=dict(dL=Ls_f-Ls_s,dF=Fs_f-Fs_s)
print(f"\n原稿口径（FSR@0.2227 vs Step-BCE@0.2451）：ΔL={Ls_f-Ls_s:+.3f}pp  ΔF={Fs_f-Fs_s:+.3f}pp")
json.dump(out,open("A2_BOOTSTRAP.json","w"),ensure_ascii=False,indent=1)
