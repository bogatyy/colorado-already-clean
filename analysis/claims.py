"""Recompute the headline cross-model counts from the raw blind-sample readings, with resampling intervals.

For every model with validated Colorado neurons: each neuron's firing on the articles of each of 179 countries,
as a share of its firing on Colorado articles (articles mentioning Colorado or Denver left out). Reports, per model,
the top Colorado neuron (highest selection correlation), the neuron with the strongest Bolivia share, how many
neurons put Bolivia 1st / in the top 3, and Bolivia's rank on the mean of the top 1, 4, 8 and all neurons.
Resampling: articles of every place are redrawn with replacement (2,000 draws).

python analysis/claims.py            writes results/claims.json
"""
import json
import os

import numpy as np

from models import MODELS

rng = np.random.default_rng(0)
docs = [json.loads(l) for l in open("results/blind_corpus.jsonl")]
first = np.array([d["first_place"] or "" for d in docs])
kind = {d["first_place"]: d["kind"] for d in docs if d["first_place"]}
word = np.array([("Colorad" in d["text"]) or ("Denver" in d["text"]) for d in docs])
countries = sorted(p for p in kind if kind[p] == "country")
own = np.where(first == "Colorado")[0]
idx = [np.where((first == c) & ~word)[0] for c in countries]
bol, nep = countries.index("Bolivia"), countries.index("Nepal")
print(f"{len(countries)} countries; articles per country min {min(map(len, idx))}, median {int(np.median(list(map(len, idx))))}; Colorado articles {len(own)}")

models = {}
for model, tag in MODELS.items():
    if not os.path.exists(f"results/tie/{tag}.json"):
        print(f"{model}: no result file, skipped"); continue
    J = json.load(open(f"results/tie/{tag}.json")); z = np.load(f"results/tie/{tag}.npz")
    if not J["selection"]["Colorado"]["clean"]:
        print(f"{model}: no clean Colorado neuron (best raw correlation {J['selection']['Colorado']['raw'][0]['corr']:.2f}), not measurable"); continue
    models[model] = [(f"L{e['layer']}/N{e['neuron']}{e['pole']}", z["top8"][:, e["k"]].astype(np.float64)) for e in J["selection"]["Colorado"]["clean"][:16]]


def shares(col, I, O):
    ref = col[O].mean() + 1e-9
    return np.array([col[i].mean() / ref if len(i) else 0.0 for i in I])


def rank(v, j):
    return int((v > v[j]).sum()) + 1


B = 2000
draws = [([rng.choice(i, len(i)) for i in idx], rng.choice(own, len(own))) for _ in range(B)]
out = {}
for model, neurons in models.items():
    S = np.array([shares(col, idx, own) for _, col in neurons])          # neurons x countries
    SB = np.array([[shares(col, I, O) for _, col in neurons] for I, O in draws])  # B x neurons x countries
    r = np.array([rank(s, bol) for s in S])
    print(f"\n=== {model}: {len(neurons)} Colorado neurons")
    rec = {"n": len(neurons), "neurons": []}
    for k, (name, _) in enumerate(neurons):
        o = np.argsort(-S[k])[:3]
        rb = np.array([rank(SB[b, k], bol) for b in range(B)])
        lo, hi = np.percentile(SB[:, k, bol], [2.5, 97.5])
        rec["neurons"].append({"neuron": name, "bolivia": float(S[k, bol]), "ci": [float(lo), float(hi)], "rank": int(r[k]), "p_first": float((rb == 1).mean()), "p_top3": float((rb <= 3).mean()),
                               "rank_ci": [int(np.percentile(rb, 2.5)), int(np.percentile(rb, 97.5))], "nepal": float(S[k, nep]), "nepal_rank": rank(S[k], nep), "top": [(countries[j], float(S[k, j])) for j in o]})
        tag = "TOP " if k == 0 else "    "
        print(f"{tag}{name:14s} Bolivia {S[k, bol]:.2f} [{lo:.2f}, {hi:.2f}] rank {r[k]:3d} (1st in {100 * (rb == 1).mean():3.0f}% of draws, top 3 in {100 * (rb <= 3).mean():3.0f}%)  Nepal {S[k, nep]:.2f} #{rank(S[k], nep):<3d} | " + ", ".join(f"{countries[j]} {S[k, j]:.2f}" for j in o))
    rec["first"] = int((r == 1).sum()); rec["top3"] = int(((r <= 3) & (S[:, bol] >= 0.1)).sum())
    print(f"  Bolivia 1st on {rec['first']} of {len(neurons)}; top 3 (share >= 0.1) on {rec['top3']}; Nepal 1st on {sum(rank(s, nep) == 1 for s in S)}, top 3 on {sum(rank(s, nep) <= 3 and s[nep] >= 0.1 for s in S)}")
    rec["mean"] = {}
    for K in (1, 4, 8, len(neurons)):
        if K > len(neurons):
            continue
        m = S[:K].mean(0); mb = SB[:, :K].mean(1)
        rb = np.array([rank(v, bol) for v in mb]); o = np.argsort(-m)[:5]
        rec["mean"][K] = {"bolivia": float(m[bol]), "rank": rank(m, bol), "p_first": float((rb == 1).mean()), "p_top3": float((rb <= 3).mean()), "nepal_rank": rank(m, nep), "top": [(countries[j], float(m[j])) for j in o]}
        print(f"  mean of top {K:2d}: Bolivia {m[bol]:.2f} rank {rank(m, bol):3d} (1st in {100 * (rb == 1).mean():3.0f}%, top 3 in {100 * (rb <= 3).mean():3.0f}%), Nepal #{rank(m, nep)} | " + ", ".join(f"{countries[j]} {m[j]:.2f}" for j in o))
    out[model] = rec

# which country is 1st / top 3 most often, over all neurons of all models
allS = np.concatenate([np.array([shares(col, idx, own) for _, col in neurons]) for neurons in models.values()])
mod = np.concatenate([[m] * len(n) for m, n in models.items()])
firsts = np.zeros(len(countries), int); top3 = np.zeros(len(countries), int); mods = [set() for _ in countries]; mods1 = [set() for _ in countries]
for s, m in zip(allS, mod):
    o = np.argsort(-s)[:3]
    if s[o[0]] >= 0.1:
        firsts[o[0]] += 1; mods1[o[0]].add(m)
    for j in o:
        if s[j] >= 0.1:
            top3[j] += 1; mods[j].add(m)
print(f"\nAll {len(allS)} Colorado neurons in {len(models)} models: 1st country (share >= 0.1) / top 3")
for j in np.argsort(-top3)[:10]:
    print(f"  {countries[j]:16s} 1st on {firsts[j]:3d} (in {len(mods1[j])} models)   top 3 on {top3[j]:3d} (in {len(mods[j])} models)")
out["all"] = {countries[j]: {"first": int(firsts[j]), "first_models": len(mods1[j]), "top3": int(top3[j]), "top3_models": len(mods[j])} for j in np.argsort(-top3)[:15]}
json.dump(out, open("results/claims.json", "w"), indent=1)
