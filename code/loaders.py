"""共享路径与统计工具。所有路径都相对于仓库根目录。"""
import json, csv, random, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def gold():
    with open(DATA / "gold_labels.csv") as f:
        return {r["id"]: {"label": int(r["label"]), "group_id": r["group_id"], "source": r["source"]}
                for r in csv.DictReader(f)}

def jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["id"]] = d["scores"]
    return out

def panel(model, backbone, readout, repeat=1):
    p = DATA / "readout_panel" / model / backbone / readout / f"repeat_{repeat}.jsonl"
    out = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["id"]] = d
    return out

def rescore(model, backbone):
    return jsonl(DATA / "main_pool_rescore" / model / f"{backbone}.jsonl")

def restoration(name):
    p = DATA / "restoration" / f"{name}.jsonl"
    if not p.exists():
        p = DATA / "restoration" / "rlhflow8b" / f"{name}.jsonl"
    return jsonl(p)

def pred(scores, tau, le=False):
    for j, s in enumerate(scores):
        if (s <= tau) if le else (s < tau):
            return j
    return -1

def hits_fa(S, ids, G, tau, le=False):
    E = C = h = f = 0
    for i in ids:
        pr = pred(S[i], tau, le)
        if G[i]["label"] == -1:
            C += 1; f += (pr != -1)
        else:
            E += 1; h += (pr == G[i]["label"])
    return dict(hits=h, fa=f, E=E, C=C, L=100 * h / E, F=100 * f / C)

def cluster_boot(Sa, Sb, ids, G, tauA, tauB, leA=False, leB=False, nb=10000, seed=20261110):
    """ΔL = L_b - L_a 的配对题组 bootstrap"""
    E = [i for i in ids if G[i]["label"] != -1]
    a = [1 if pred(Sa[i], tauA, leA) == G[i]["label"] else 0 for i in E]
    b = [1 if pred(Sb[i], tauB, leB) == G[i]["label"] else 0 for i in E]
    mem = collections.defaultdict(list)
    for k, i in enumerate(E):
        mem[G[i]["group_id"]].append(k)
    groups = sorted(mem); rng = random.Random(seed); draws = []
    obs = 100 * (sum(b) - sum(a)) / len(E)
    for _ in range(nb):
        pick = [rng.choice(groups) for _ in groups]
        ks = [k for g in pick for k in mem[g]]
        if ks:
            draws.append(100 * (sum(b[k] for k in ks) - sum(a[k] for k in ks)) / len(ks))
    draws.sort()
    return dict(obs=obs, lo=draws[int(.025 * len(draws))], hi=draws[int(.975 * len(draws))],
                E=len(E), groups=len(groups))

def tie_stats(S, ids, tau):
    steps = tot = traj = 0
    for i in ids:
        hit = False
        for s in S[i]:
            tot += 1
            if abs(s - tau) < 1e-12:
                steps += 1; hit = True
        traj += hit
    return dict(tie_steps=steps, total_steps=tot, tie_traj=traj, n=len(ids))
