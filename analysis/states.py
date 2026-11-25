"""Control: is the Colorado-Bolivia tie special, or does every state's neuron keep a foreign country for company?

The probe selects neurons for five states (Colorado, Utah, New Mexico, Wyoming, Montana) the same way, on US and
general articles only. For every such neuron, each of the 179 countries gets its share: mean firing on that country's
articles in the blind sample, divided by mean firing on the state's own articles. Articles that contain the state's
name (for Colorado: "Colorad" or "Denver") are left out of the country side. A country counts as 1st on a neuron when
it has the highest share and that share is at least 0.1.

python analysis/states.py            writes results/claims_states.json
"""
import json
import os

import numpy as np

from models import MODELS

FOCUS = ["Colorado", "Utah", "New Mexico", "Wyoming", "Montana"]
docs = [json.loads(l) for l in open("results/blind_corpus.jsonl")]
first = np.array([d["first_place"] or "" for d in docs])
kind = {d["first_place"]: d["kind"] for d in docs if d["first_place"]}
countries = sorted(p for p in kind if kind[p] == "country")
is_c = [first == c for c in countries]

out = {}
for s in FOCUS:
    has = np.array([("Colorad" in d["text"]) or ("Denver" in d["text"]) for d in docs]) if s == "Colorado" else np.array([s in d["text"] for d in docs])
    own = first == s
    idx = [np.where(m & ~has)[0] for m in is_c]
    n1, n3, m1, m3 = {}, {}, {}, {}
    total = 0; used = []; tops = {}
    for model, tag in MODELS.items():
        if not os.path.exists(f"results/tie/{tag}.json"):
            continue
        J = json.load(open(f"results/tie/{tag}.json")); z = np.load(f"results/tie/{tag}.npz")
        clean = J["selection"][s]["clean"][:16]
        if not clean:
            continue
        used.append(model)
        for r, e in enumerate(clean):
            col = z["top8"][:, e["k"]].astype(np.float64)
            ref = col[own].mean() + 1e-9
            sh = np.array([col[i].mean() / ref if len(i) else 0.0 for i in idx])
            o = np.argsort(-sh)[:3]
            total += 1
            if r == 0:
                tops[model] = [f"L{e['layer']}/N{e['neuron']}{e['pole']}", [(countries[j], round(float(sh[j]), 3)) for j in o]]
            if sh[o[0]] >= 0.1:
                n1[countries[o[0]]] = n1.get(countries[o[0]], 0) + 1; m1.setdefault(countries[o[0]], set()).add(model)
            for j in o:
                if sh[j] >= 0.1:
                    n3[countries[j]] = n3.get(countries[j], 0) + 1; m3.setdefault(countries[j], set()).add(model)
    order = sorted(n1, key=lambda c: (-n1[c], -n3.get(c, 0)))[:6]
    print(f"\n{s}: {total} neurons in {len(used)} models. Country ranked 1st (share >= 0.1):")
    for c in order:
        print(f"   {c:16s} 1st on {n1[c]:3d} neurons ({len(m1[c]):2d} models)   top 3 on {n3.get(c, 0):3d} ({len(m3.get(c, ())):2d} models)")
    print("   top neuron per model: " + "; ".join(f"{m} {v[1][0][0]} {v[1][0][1]:.2f}" for m, v in tops.items()))
    out[s] = {"n": total, "models": len(used), "first": [[c, n1[c], len(m1[c])] for c in order],
              "top3": [[c, n3[c], len(m3[c])] for c in sorted(n3, key=lambda c: -n3[c])[:8]], "top_neuron": tops}
json.dump(out, open("results/claims_states.json", "w"), indent=1)
