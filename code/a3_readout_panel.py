"""读出面板：A/B/C 三条件下的判定改变、栅格统计与共享隐藏状态校验。"""
from loaders import *

G = gold()
rows = []
for model in ("math_shepherd7b", "rlhflow8b"):
    for bb in ("bf16", "nf4"):
        D = {r: panel(model, bb, r) for r in ("A", "B", "C")}
        ids = sorted(set(D["A"]) & set(D["B"]) & set(D["C"]))
        same_hidden = sum(1 for i in ids if D["A"][i]["hidden_sha256"] == D["B"][i]["hidden_sha256"] == D["C"][i]["hidden_sha256"])
        grid = {}
        for r in ("A", "B", "C"):
            tot = hit = 0
            for i in ids:
                for pair in D[r][i]["logits"]:
                    for x in pair:
                        tot += 1
                        if abs(x / 0.125 - round(x / 0.125)) < 1e-9: hit += 1
            grid[r] = (hit, tot)
        chg = [i for i in ids if pred(D["A"][i]["scores"], 0.5) != pred(D["B"][i]["scores"], 0.5)]
        rows.append((model, bb, len(ids), same_hidden, grid["A"], grid["B"], grid["C"], len(chg)))
        print(f"{model}/{bb}: n={len(ids)} 同隐藏状态={same_hidden}/{len(ids)} "
              f"1/8栅格 A={grid['A'][0]}/{grid['A'][1]} B={grid['B'][0]}/{grid['B'][1]} C={grid['C'][0]}/{grid['C'][1]} "
              f"改变判定={len(chg)}")
print("\n预期：A 与 C 全部在栅格上，B 为 0；数学 9/10 条、RLH 2/4 条发生改变。")
