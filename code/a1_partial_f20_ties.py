# -*- coding: utf-8 -*-
"""A1(部分): 主池上 F20 阈值处的并列计数（仅本地可得的 MathShepherd 臂）"""
import json
NORM="/Users/louis/Documents/ChatGPT/lunwen/analysis/prm_v11_completion_20261109/work/normalized"
TAUS={"F20_bf16":0.22270014882087708,"F20_nf4":0.24508501589298248}
files={
 "nf4_baseline":"V11-math_shepherd7b-evaluation-baselines.jsonl",
 "FSR-b10":"V11-math_shepherd7b-FSR-b10.jsonl",
 "FSR-b20":"V11-math_shepherd7b-FSR-b20.jsonl",
 "Step-BCE-b10":"V11-math_shepherd7b-Step-BCE-b10.jsonl",
 "Step-BCE-b20":"V11-math_shepherd7b-Step-BCE-b20.jsonl",
 "Output-MSE-b10":"V11-math_shepherd7b-Output-MSE-b10.jsonl",
 "Random-b10-s20261108":"V11-math_shepherd7b-Random-b10-s20261108.jsonl",
}
def load(fn):
    rows=[]
    with open(f"{NORM}/{fn}") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows
out={}
for name,fn in files.items():
    rows=load(fn)
    # 对齐 id
    rec={}
    for tau_name,tau in TAUS.items():
        steps=tot=traj=0
        for r in rows:
            hit=False
            for s in r["scores"]:
                tot+=1
                if abs(s-tau)<1e-12: steps+=1; hit=True
            if hit: traj+=1
        rec[tau_name]=dict(tau=tau, tie_steps=steps, tie_traj=traj, total_steps=tot, n=len(rows),
                           step_rate=steps/tot, traj_rate=traj/len(rows))
    out[name]=rec
    print(f"{name:22s} n={len(rows)}  " + "  ".join(
        f"{k}: {v['tie_steps']}/{v['total_steps']} ({v['step_rate']*100:.2f}%), traj {v['tie_traj']} ({v['traj_rate']*100:.1f}%)"
        for k,v in rec.items()))
json.dump(out,open("A1_F20_TIES_MATH.json","w"),ensure_ascii=False,indent=1)
