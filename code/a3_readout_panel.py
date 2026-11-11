# -*- coding: utf-8 -*-
"""A3: 读出面板逐条机制分析（只用本地 raw，不重新推理）"""
import json, glob, os, collections
RAW = "/Users/louis/Documents/ChatGPT/lunwen/analysis/prm_readout_precision_20261110/raw"

def load(model, backbone, readout, repeat=1):
    p = f"{RAW}/{model}/{backbone}/{readout}/repeat_{repeat}.jsonl"
    out = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            out[d["id"]] = d
    return out

def pred_at(scores, tau=0.5):
    for i, s in enumerate(scores):
        if s < tau: return i + 1   # 1-based step index
    return -1

def grid_frac(logits):
    """fraction of logit entries that are exact multiples of 0.125"""
    tot = hit = 0
    for pair in logits:
        for x in pair:
            tot += 1
            if abs(x / 0.125 - round(x / 0.125)) < 1e-9: hit += 1
    return hit, tot

res = {}
for model in ("math_shepherd7b", "rlhflow8b"):
    for bb in ("bf16", "nf4"):
        data = {ro: load(model, bb, ro) for ro in ("A","B","C")}
        ids = sorted(set(data["A"]) & set(data["B"]) & set(data["C"]))
        m = {}
        for ro in ("A","B","C"):
            E=C=hits=fa=0
            for i in ids:
                d = data[ro][i]
                pr = pred_at(d["scores"])
                if d["label"] == -1:
                    C += 1
                    if pr != -1: fa += 1
                else:
                    E += 1
                    if pr == d["label"]: hits += 1
                m.setdefault(ro, {})
            m[ro] = dict(E=E, C=C, hits=hits, fa=fa, L=hits/E, F=fa/C)
        # changed items A vs B / A vs C
        chg = []
        for i in ids:
            pa, pb, pc = (pred_at(data[ro][i]["scores"]) for ro in ("A","B","C"))
            if pa != pb or pa != pc:
                d = data["A"][i]
                chg.append(dict(id=i, source=d["source"], label=d["label"], n_steps=d["n_steps"],
                                pred_A=pa, pred_B=pb, pred_C=pc,
                                scores_A=[round(x,4) for x in data["A"][i]["scores"]],
                                scores_B=[round(x,4) for x in data["B"][i]["scores"]]))
        # hidden state identity across readouts
        same_hidden = sum(1 for i in ids if data["A"][i]["hidden_sha256"]==data["B"][i]["hidden_sha256"]==data["C"][i]["hidden_sha256"])
        grid = {ro: grid_frac([l for i in ids for l in data[ro][i]["logits"]]) for ro in ("A","B","C")}
        res[f"{model}/{bb}"] = dict(n=len(ids), metrics=m, changed=chg, same_hidden=same_hidden, grid=grid)

json.dump(res, open("A3_PANEL.json","w"), ensure_ascii=False, indent=1)

for key, r in res.items():
    print(f"\n=== {key}  n={r['n']}  same_hidden={r['same_hidden']}")
    for ro in ("A","B","C"):
        x=r["metrics"][ro]
        print(f"  {ro}: L={x['L']*100:.2f}% F={x['F']*100:.2f}% hits={x['hits']} FA={x['fa']}")
    g=r["grid"]
    print("  logit on 1/8 grid:", {ro: f"{g[ro][0]}/{g[ro][1]}" for ro in g})
    print(f"  changed items (A vs B or C): {len(r['changed'])}")
    for c in r["changed"][:12]:
        print("   ", c["source"].split("/")[0], "label",c["label"],"steps",c["n_steps"],
              "predA/B/C=",c["pred_A"],c["pred_B"],c["pred_C"])
