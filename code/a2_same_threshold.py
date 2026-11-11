"""恢复配置：各自阈值 vs 共同阈值下的 FSR − Step-BCE。"""
from loaders import *

G = gold()
CFG = {"MathShepherd": ("V11-math_shepherd7b-FSR-b10", "V11-math_shepherd7b-Step-BCE-b10",
                        0.22270014882087708, 0.24508501589298248, 0.24508501589298248),
       "RLHFlow": ("rlhflow8b/FSR-b10", "rlhflow8b/Step-BCE-b10",
                   0.2689414322376251, 0.29421496391296387, 0.2689414322376251)}
for name, (fj, sj, tau_f, tau_s, tau_shared) in CFG.items():
    F, S = restoration(fj), restoration(sj)
    ids = sorted(set(F) & set(S) & set(G))
    print(f"\n=== {name}  配对 {len(ids)}")
    a = hits_fa(F, ids, G, tau_f); b = hits_fa(S, ids, G, tau_s)
    print(f"  各自阈值: FSR {a['hits']}/{a['fa']} @ {tau_f:.6f}   Step {b['hits']}/{b['fa']} @ {tau_s:.6f}")
    r = cluster_boot(F, S, ids, G, tau_f, tau_s, False, False)
    print(f"    ΔL={r['obs']:+.3f} [{r['lo']:+.2f}, {r['hi']:+.2f}]  ΔF="
          f"{hits_fa(F,ids,G,tau_f)['F']-hits_fa(S,ids,G,tau_s)['F']:+.3f}")
    a2 = hits_fa(F, ids, G, tau_shared); b2 = hits_fa(S, ids, G, tau_shared)
    r2 = cluster_boot(F, S, ids, G, tau_shared, tau_shared, False, False)
    print(f"  共同阈值 {tau_shared:.6f}: FSR {a2['hits']}/{a2['fa']}  Step {b2['hits']}/{b2['fa']}")
    print(f"    ΔL={r2['obs']:+.3f} [{r2['lo']:+.2f}, {r2['hi']:+.2f}]  ΔF={a2['F']-b2['F']:+.3f}")
print("\n预期：Math 各自 −0.24[−0.61,+0.15] → 共同 +0.24[−0.17,+0.64]；RLH 各自 −1.38[−1.79,−1.00] → 共同 −0.03[−0.33,+0.24]。")
