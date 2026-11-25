"""One line per model: how the top Colorado neuron treats Bolivia, among countries and among everything else.

For every model's result in results/tie: number of clean Colorado
neurons, the top one (highest selection correlation), Bolivia's share of the Colorado level, its rank among 179
countries, among the 264 non-Colorado places, and among 1,035 topics (264 places + 771 single articles about no listed
place), and how many of the 25 highest-firing non-Colorado articles are about Bolivia. Articles naming Colorado or
Denver are left out everywhere except the Colorado reference itself.

python analysis/summary.py [tag ...] [--all]      --all prints every Colorado neuron, not only the top one
"""
import json
import os
import sys

import numpy as np

from models import MODELS

docs = [json.loads(l) for l in open("results/blind_corpus.jsonl")]
first = np.array([d["first_place"] or "" for d in docs])
kind = {d["first_place"]: d["kind"] for d in docs if d["first_place"]}
word = np.array([("Colorad" in d["text"]) or ("Denver" in d["text"]) for d in docs])
own = first == "Colorado"
places = sorted(p for p in kind if p != "Colorado")
countries = [p for p in places if kind[p] == "country"]
pool = ~own & ~word
none = (first == "") & ~word
print(f"{len(countries)} countries, {len(places)} places, {int(none.sum())} no-place articles, pool {int(pool.sum())}, Bolivia articles {int(((first == 'Bolivia') & ~word).sum())}\n")

NAMES = {tag: name for name, tag in MODELS.items()}
tags = [t for t in sys.argv[1:] if not t.startswith("--")] or list(NAMES)


def line(name, label, col, extra=""):
    ref = col[own].mean() + 1e-9
    sh = {p: col[(first == p) & ~word].mean() / ref for p in places}
    b = sh["Bolivia"]
    rc = 1 + sum(sh[c] > b for c in countries)
    rp = 1 + sum(sh[p] > b for p in places)
    art = col[none] / ref
    rt = rp + int((art > b).sum())
    order = np.argsort(-np.where(pool, col, -np.inf))[:100]
    t25 = int((first[order[:25]] == "Bolivia").sum()); t100 = int((first[order] == "Bolivia").sum())
    topc = sorted(countries, key=lambda c: -sh[c])[:3]; topp = sorted(places, key=lambda p: -sh[p])[:3]
    print(f"{name:20s} {label:15s} Bolivia {b:5.2f}  #{rc:<3d} of 179 countries  #{rp:<3d} of 264 places  #{rt:<4d} of 1,035 topics  top-25 {t25:2d} top-100 {t100:2d} | countries: "
          + ", ".join(f"{c} {sh[c]:.2f}" for c in topc) + " | places: " + ", ".join(f"{p} {sh[p]:.2f}" for p in topp) + extra)


for tag in tags:
    f = f"results/tie/{tag}.json"
    if not os.path.exists(f):
        print(f"{NAMES.get(tag, tag):20s} no result file"); continue
    J = json.load(open(f)); z = np.load(f"results/tie/{tag}.npz")
    cl = J["selection"]["Colorado"]["clean"]
    if not cl:
        print(f"{NAMES.get(tag, tag):20s} no clean Colorado neuron (best raw correlation {J['selection']['Colorado']['raw'][0]['corr']:.2f}; {J['layers']} layers x {J['neurons_per_layer']})"); continue
    for r, e in enumerate(cl[:16]):
        if r == 0 or "--all" in sys.argv:
            line(NAMES.get(tag, tag) if r == 0 else "", f"L{e['layer']}/N{e['neuron']}{e['pole']}", z["top8"][:, e["k"]].astype(np.float64), f" | corr {e['corr']:.2f}, {len(cl)} clean")
