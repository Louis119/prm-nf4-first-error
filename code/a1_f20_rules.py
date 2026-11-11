"""主池重评：源 F20 阈值处的并列计数、< 与 ≤ 的规则敏感性、ΔL 与区间。"""
from loaders import *

G = gold()
F20 = {"math_shepherd7b": {"bf16": 0.22270014882087708, "nf4": 0.24508501589298248},
       "rlhflow8b": {"bf16": 0.29421496391296387, "nf4": 0.2689414322376251}}
for model in F20:
    Sb, Sn = rescore(model, "bf16"), rescore(model, "nf4")
    ids = sorted(set(Sb) & set(Sn) & set(G))
    print(f"\n=== {model}  配对 {len(ids)} 条")
    for bb, S in (("bf16", Sb), ("nf4", Sn)):
        tau = F20[model][bb]
        a = hits_fa(S, ids, G, tau, False); b = hits_fa(S, ids, G, tau, True)
        t = tie_stats(S, ids, tau)
        print(f"  {bb}: τ={tau:.6f}  <: L={a['L']:.2f} F={a['F']:.2f} ({a['hits']}/{a['fa']})   "
              f"≤: L={b['L']:.2f} F={b['F']:.2f} ({b['hits']}/{b['fa']})   "
              f"并列 {t['tie_steps']}/{t['total_steps']} ({100*t['tie_steps']/t['total_steps']:.2f}%), {t['tie_traj']} 条")
    for tag, le in (("F20 <", False), ("F20 ≤", True)):
        r = cluster_boot(Sb, Sn, ids, G, F20[model]["bf16"], F20[model]["nf4"], le, le)
        print(f"  ΔL(NF4−BF16) {tag}: {r['obs']:+.3f} [{r['lo']:+.2f}, {r['hi']:+.2f}]  E={r['E']} 组={r['groups']}")
    tau = F20[model]["nf4"]
    r = cluster_boot(Sb, Sn, ids, G, tau, tau, False, False)
    print(f"  同阈值 τ={tau:.6f}: ΔL={r['obs']:+.3f} [{r['lo']:+.2f}, {r['hi']:+.2f}]")
print("\n预期：F20 < 与 ≤ 的 ΔL 只差 0.2–0.3 个百分点；同阈值时 Math 约 −0.74[−1.35,−0.12]、RLH 约 +0.38[−0.15,+0.93]。")
