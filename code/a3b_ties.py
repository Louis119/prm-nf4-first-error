"""读出面板：阈值处的并列频率，以及判定改变所对应的那一步的对数几率差。"""
from loaders import *

for model in ("math_shepherd7b", "rlhflow8b"):
    for bb in ("bf16", "nf4"):
        A = panel(model, bb, "A"); B = panel(model, bb, "B"); C = panel(model, bb, "C")
        ids = sorted(A)
        print(f"\n{model}/{bb}")
        for name, D in (("A", A), ("B", B), ("C", C)):
            tot = tie = 0
            for i in ids:
                for s in D[i]["scores"]:
                    tot += 1
                    if s == 0.5: tie += 1
            print(f"  {name}: 恰好 0.5 的步 {tie}/{tot} ({100*tie/tot:.2f}%)")
        cross = []
        for i in ids:
            sa, sb = A[i]["scores"], B[i]["scores"]
            js = [j for j in range(len(sa)) if (sa[j] < 0.5) != (sb[j] < 0.5)]
            if not js: continue
            j = js[0]
            da = A[i]["logits"][j][0] - A[i]["logits"][j][1]
            db = B[i]["logits"][j][0] - B[i]["logits"][j][1]
            cross.append((j + 1, da, db, sa[j], sb[j]))
        print(f"  首次越界判定不同的步：{len(cross)} 个；其中 A 对数几率差非零的：{sum(1 for c in cross if abs(c[1])>1e-9)} 个")
        dec = [i for i in ids if pred(A[i]["scores"], 0.5) != pred(B[i]["scores"], 0.5)]
        dc = []
        for i in dec:
            sa, sb = A[i]["scores"], B[i]["scores"]
            j = [j for j in range(len(sa)) if (sa[j] < 0.5) != (sb[j] < 0.5)][0]
            dc.append(B[i]["scores"][j])
        if dc:
            print(f"  导致首个报警位置改变的条目：{len(dec)}；对应 B 评分 {min(dc):.4f}–{max(dc):.4f}")
        if cross:
            print(f"  B 对数几率差 {min(abs(c[2]) for c in cross):.4f}–{max(abs(c[2]) for c in cross):.4f}；"
                  f"B 评分 {min(c[4] for c in cross):.4f}–{max(c[4] for c in cross):.4f}")
print("\n预期：A 下这些步的两对数几率完全相等（差为 0）；B 下差为 0.0026–0.1072，仍小于 BF16 间隔 0.125。")
