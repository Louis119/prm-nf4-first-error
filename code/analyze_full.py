# -*- coding: utf-8 -*-
"""A1 完整：主池 F20 阈值处的并列与规则敏感性（新建同环境评分）"""
import json, sys, random, collections
from pathlib import Path
EVAL = "/Users/louis/Documents/ChatGPT/lunwen/analysis/prm_precision_repair_20261108/data/evaluation_primary.jsonl"
FULL = Path("full")
F20 = {"math_shepherd7b": {"bf16": 0.22270014882087708, "nf4": 0.24508501589298248},
       "rlhflow8b": {"bf16": 0.29421496391296387, "nf4": 0.2689414322376251}}
gold = {}
with open(EVAL) as f:
    for line in f:
        line = line.strip()
        if line:
            d = json.loads(line); gold[d["id"]] = d

def load(model, bb):
    p = FULL / model / f"{bb}.jsonl"
    out = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line); out[d["id"]] = d["scores"]
    return out

def pred(sc, tau, le=False):
    for j, s in enumerate(sc):
        if (s <= tau) if le else (s < tau):
            return j
    return -1

def metrics(ids, S, tau, le=False):
    E = C = h = fa = 0
    for i in ids:
        pr = pred(S[i], tau, le); lab = gold[i]["label"]
        if lab == -1:
            C += 1; fa += (pr != -1)
        else:
            E += 1; h += (pr == lab)
    return dict(hits=h, fa=fa, E=E, C=C, L=100*h/E, F=100*fa/C)

def ties(ids, S, tau):
    st = tr = tot = 0
    for i in ids:
        hit = False
        for s in S[i]:
            tot += 1
            if abs(s - tau) < 1e-12: st += 1; hit = True
        tr += hit
    return dict(tie_steps=st, total_steps=tot, tie_traj=tr, n=len(ids))

def boot_delta(ids, SA, SB, tauA, tauB, leA=False, leB=False, nb=10000, seed=20261110):
    """ΔL = L_B - L_A, cluster bootstrap over group_id"""
    E = [i for i in ids if gold[i]["label"] != -1]
    hA = [1 if pred(SA[i], tauA, leA) == gold[i]["label"] else 0 for i in E]
    hB = [1 if pred(SB[i], tauB, leB) == gold[i]["label"] else 0 for i in E]
    g = [gold[i]["group_id"] for i in E]
    members = collections.defaultdict(list)
    for k, gi in enumerate(g): members[gi].append(k)
    groups = sorted(members)
    obs = 100*(sum(hB)-sum(hA))/len(E)
    rng = random.Random(seed); draws = []
    for _ in range(nb):
        pick = [rng.choice(groups) for _ in groups]
        ks = [k for gi in pick for k in members[gi]]
        if ks: draws.append(100*(sum(hB[k] for k in ks)-sum(hA[k] for k in ks))/len(ks))
    draws.sort()
    return dict(obs=obs, lo=draws[int(.025*len(draws))], hi=draws[int(.975*len(draws))], E=len(E), groups=len(groups))

res = {}
print("=== 同环境重评（新评分，A 读出，严格 <）===")
for model in F20:
    Sb = load(model, "bf16"); Sn = load(model, "nf4")
    ids = sorted(set(Sb) & set(Sn) & set(gold))
    print(f"\n--- {model}: bf16 {len(Sb)}, nf4 {len(Sn)}, 配对 {len(ids)}")
    r = {"n": len(ids)}
    for bb, S in (("bf16", Sb), ("nf4", Sn)):
        tau = F20[model][bb]
        lt = metrics(ids, S, tau, False); le = metrics(ids, S, tau, True)
        r[bb] = dict(tau=tau, lt=lt, le=le, ties=ties(ids, S, tau))
        print(f"  {bb}: τ={tau:.6f}  <: L={lt['L']:.2f} F={lt['F']:.2f} ({lt['hits']}/{lt['fa']})   "
              f"<=: L={le['L']:.2f} F={le['F']:.2f} ({le['hits']}/{le['fa']})   "
              f"并列 {r[bb]['ties']['tie_steps']}/{r[bb]['ties']['total_steps']} 步, {r[bb]['ties']['tie_traj']} 条")
    # ΔL(NF4 - BF16) 各政策
    for tag, le in (("F20 <", False), ("F20 <=", True)):
        d = boot_delta(ids, Sb, Sn, F20[model]["bf16"], F20[model]["nf4"], le, le)
        r[tag] = d
        print(f"  ΔL(NF4−BF16) {tag}: {d['obs']:+.3f} [{d['lo']:+.2f},{d['hi']:+.2f}] pp  (E={d['E']}, {d['groups']} 组)")
    # 同阈值（都用 nf4 的 τ）
    tau = F20[model]["nf4"]
    d = boot_delta(ids, Sb, Sn, tau, tau, False, False)
    r["same_tau"] = d
    print(f"  同阈值 τ={tau:.6f}: ΔL={d['obs']:+.3f} [{d['lo']:+.2f},{d['hi']:+.2f}]")
    # 交互：ΔL(<=) − ΔL(<)
    res[model] = r
json.dump(res, open("A1_FULL.json","w"), ensure_ascii=False, indent=1)
print("\nsaved A1_FULL.json")
