# -*- coding: utf-8 -*-
"""A1(部分): F20 下 < vs <= 的臂内移动；A2: 同阈值下 FSR vs Step-BCE"""
import json
NORM="/Users/louis/Documents/ChatGPT/lunwen/analysis/prm_v11_completion_20261109/work/normalized"
FILES={
 "nf4":"V11-math_shepherd7b-evaluation-baselines.jsonl",
 "FSR-b10":"V11-math_shepherd7b-FSR-b10.jsonl",
 "FSR-b20":"V11-math_shepherd7b-FSR-b20.jsonl",
 "Step-BCE-b10":"V11-math_shepherd7b-Step-BCE-b10.jsonl",
 "Step-BCE-b20":"V11-math_shepherd7b-Step-BCE-b20.jsonl",
}
def load(fn):
    rows=[]
    with open(f"{NORM}/{fn}") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def score(rows,tau,le=False):
    E=C=h=f=0
    for r in rows:
        pred=-1
        for i,s in enumerate(r["scores"]):
            if (s<=tau) if le else (s<tau):
                pred=i; break
        if r["label"]==-1:
            C+=1
            if pred!=-1: f+=1
        else:
            E+=1
            if pred==r["label"]: h+=1
    return dict(hits=h, fa=f, E=E, C=C, L=h/E*100, F=f/C*100)

rows={k:load(v) for k,v in FILES.items()}
TAU_F20=0.24508501589298248
TAU_BF20=0.22270014882087708
print("=== F20 阈值处的规则敏感性（MathShepherd，本地可得臂）===")
for name,r in rows.items():
    for tname,tau in (("F20(nf4)=0.245085",TAU_F20),("F20(bf16)=0.222700",TAU_BF20)):
        a=score(r,tau,False); b=score(r,tau,True)
        print(f"{name:14s} {tname:22s} <: L={a['L']:.2f} F={a['F']:.2f} ({a['hits']}/{a['fa']})   "
              f"<=: L={b['L']:.2f} F={b['F']:.2f} ({b['hits']}/{b['fa']})   ΔL={b['L']-a['L']:+.2f}pp ΔF={b['F']-a['F']:+.2f}pp")

print("\n=== A2: 同一阈值下 FSR vs Step-BCE（MathShepherd, 4,673 条）===")
for tname,tau in (("τ=0.245085 (NF4 源F20)",TAU_F20),("τ=0.222700 (BF16 源F20)",TAU_BF20),("τ=0.5 (固定)",0.5)):
    line=[]
    for name in ("FSR-b10","Step-BCE-b10"):
        s=score(rows[name],tau,False)
        line.append((name,s))
    d_h=line[0][1]["hits"]-line[1][1]["hits"]; d_f=line[0][1]["fa"]-line[1][1]["fa"]
    print(f"{tname:26s} FSR {line[0][1]['hits']}/{line[0][1]['fa']}  Step-BCE {line[1][1]['hits']}/{line[1][1]['fa']}  "
          f"FSR−StepBCE: {d_h:+d} hits, {d_f:+d} FA")
out=dict(f20_rule={n:{tn:{'lt':score(r,t,False),'le':score(r,t,True)} for tn,t in (("F20_nf4",TAU_F20),("F20_bf16",TAU_BF20))} for n,r in rows.items()})
json.dump(out,open("A12_RULES.json","w"),ensure_ascii=False,indent=1)
