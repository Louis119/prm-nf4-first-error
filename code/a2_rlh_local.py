# -*- coding: utf-8 -*-
"""A2 补：RLHFlow 的 FSR vs Step-BCE 同阈值对照（本地既有评分）"""
import json, random, collections
BASE="/Users/louis/Documents/ChatGPT/lunwen/analysis/prm_v11_results_partial_20261109/work/jobs"
EVAL="/Users/louis/Documents/ChatGPT/lunwen/analysis/prm_precision_repair_20261108/data/evaluation_primary.jsonl"
gold={}
for line in open(EVAL):
    line=line.strip()
    if line:
        d=json.loads(line); gold[d["id"]]=d
def load(job):
    out={}
    for line in open(f"{BASE}/{job}/candidate/primary.jsonl"):
        line=line.strip()
        if line:
            d=json.loads(line); out[d["id"]]=d["scores"]
    return out
FSR=load("V11-rlhflow8b-FSR-b10"); STEP=load("V11-rlhflow8b-Step-BCE-b10")
ids=sorted(set(FSR)&set(STEP)&set(gold))
E=[i for i in ids if gold[i]["label"]!=-1]; C=[i for i in ids if gold[i]["label"]==-1]
print("paired",len(ids),"E",len(E),"C",len(C),"groups",len({gold[i]['group_id'] for i in ids}))
def pred(sc,tau,le=False):
    for j,s in enumerate(sc):
        if (s<=tau) if le else (s<tau): return j
    return -1
def L(S,tau,le=False): return 100*sum(1 for i in E if pred(S[i],tau,le)==gold[i]["label"])/len(E)
def F(S,tau,le=False): return 100*sum(1 for i in C if pred(S[i],tau,le)!=-1)/len(C)
def boot(tauF,tauS,idx=0,leF=False,leS=False,nb=10000,seed=20261110):
    rows=E if idx==0 else C
    def ind(S,tau,le):
        if idx==0: return [1 if pred(S[i],tau,le)==gold[i]["label"] else 0 for i in rows]
        return [1 if pred(S[i],tau,le)!=-1 else 0 for i in rows]
    a=ind(FSR,tauF,leF); b=ind(STEP,tauS,leS)
    g=[gold[i]["group_id"] for i in rows]
    mem=collections.defaultdict(list)
    for k,gi in enumerate(g): mem[gi].append(k)
    groups=sorted(mem); rng=random.Random(seed)
    obs=100*(sum(a)-sum(b))/len(rows); draw=[]
    for _ in range(nb):
        pick=[rng.choice(groups) for _ in groups]
        ks=[k for gi in pick for k in mem[gi]]
        if ks: draw.append(100*(sum(a[k] for k in ks)-sum(b[k] for k in ks))/len(ks))
    draw.sort()
    return obs, draw[int(.025*len(draw))], draw[int(.975*len(draw))]
TAU_F, TAU_S = 0.2689414322376251, 0.29421496391296387   # 表D-2 两臂各自源F20阈值
print(f"\n各臂自身阈值：FSR {L(FSR,TAU_F):.2f}/{F(FSR,TAU_F):.2f}  Step {L(STEP,TAU_S):.2f}/{F(STEP,TAU_S):.2f}"
      f"  ΔL={L(FSR,TAU_F)-L(STEP,TAU_S):+.3f}  ΔF={F(FSR,TAU_F)-F(STEP,TAU_S):+.3f}")
out={"own":dict(dL=L(FSR,TAU_F)-L(STEP,TAU_S),dF=F(FSR,TAU_F)-F(STEP,TAU_S))}
for tag,tau in (("tau=0.268941(FSR)",TAU_F),("tau=0.294215(Step)",TAU_S),("tau=0.5",0.5)):
    for le in (False,True):
        oL=boot(tau,tau,0,le,le); oF=boot(tau,tau,1,le,le)
        key=f"{tag}/{'<=' if le else '<'}"
        out[key]=dict(dL=oL,dF=oF)
        print(f"{key:26s} ΔL={oL[0]:+.3f} [{oL[1]:+.2f},{oL[2]:+.2f}]   ΔF={oF[0]:+.3f} [{oF[1]:+.2f},{oF[2]:+.2f}]")
json.dump(out,open("A2_RLH.json","w"),ensure_ascii=False,indent=1)
