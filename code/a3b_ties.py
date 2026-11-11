# -*- coding: utf-8 -*-
"""A3b: 面板上的并列频率与判定裕度"""
import json, collections
RAW="/Users/louis/Documents/ChatGPT/lunwen/analysis/prm_readout_precision_20261110/raw"
def load(model,bb,ro,rep=1):
    out={}
    with open(f"{RAW}/{model}/{bb}/{ro}/repeat_{rep}.jsonl") as f:
        for line in f:
            line=line.strip()
            if line: d=json.loads(line); out[d["id"]]=d
    return out
summary={}
for model in ("math_shepherd7b","rlhflow8b"):
    for bb in ("bf16","nf4"):
        D={ro:load(model,bb,ro) for ro in ("A","B","C")}
        ids=sorted(D["A"])
        row={}
        for ro in ("A","B","C"):
            tot=tie=0; cont=0
            for i in ids:
                for s in D[ro][i]["scores"]:
                    tot+=1
                    if abs(s-0.5)<1e-12: tie+=1
                # 连续分数：任一步落在 0.5±1e-3 内
                if any(abs(s-0.5)<1e-3 for s in D[ro][i]["scores"]): cont+=1
            row[ro]=dict(steps=tot, exact_0_5=tie, traj_within_1e3=cont)
        # margin analysis for changed items
        marg=[]
        for i in ids:
            pa=D["A"][i]; pb=D["B"][i]
            def pred(d):
                for j,s in enumerate(d["scores"]):
                    if s<0.5: return j
                return -1
            ja,jb=pred(pa),pred(pb)
            if ja==jb: continue
            # 取 A 中第一个跨越点附近的裕度
            cand = [abs(s-0.5) for s in pa["scores"]]
            k = min(range(len(cand)), key=lambda t: cand[t])
            marg.append(dict(id=i, source=pa["source"], label=pa["label"],
                             pred_A=ja+1 if ja>=0 else -1, pred_B=jb+1 if jb>=0 else -1,
                             a_score_at_k=round(pa["scores"][k],6), k=k+1,
                             b_score_at_k=round(pb["scores"][k],6),
                             a_logit_k=pa["logits"][k], b_logit_k=pb["logits"][k]))
        row["changed_margins"]=marg
        summary[f"{model}/{bb}"]=row
        print(f"\n=== {model}/{bb}")
        for ro in ("A","B","C"):
            x=row[ro]; print(f"  {ro}: steps={x['steps']} exact0.5={x['exact_0_5']} traj|s-0.5|<1e-3={x['traj_within_1e3']}")
        for m in marg:
            print(f"   changed {m['source'].split('/')[0]:14s} pred {m['pred_A']:>2}/{m['pred_B']:<2} step{m['k']:>2} A={m['a_score_at_k']:.6f} B={m['b_score_at_k']:.6f} logitA={m['a_logit_k']}")
json.dump(summary,open("A3_TIES.json","w"),ensure_ascii=False,indent=1)
