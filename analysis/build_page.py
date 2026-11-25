"""Build the data behind the circle-map page (artifact/colorado-maps.html) from the probe's result files.

Nothing here is chosen by hand. For each of six models the four best Colorado neurons are taken (the first four of
the probe's "clean" list: highest correlation with "this article is about Colorado" on US and general articles, quiet
on general articles). For each neuron:
  - a dot is one of the 7,239 sampled articles, placed by its strongest token (share of the neuron's maximum over
    the sample); articles under 3% are counted but not drawn;
  - wedges are set by rule: Colorado named first; Colorado or Denver mentioned later; another U.S. state named first;
    the four places outside the U.S. whose articles fire this neuron most; every other place abroad; no listed place;
  - "fire most" is the share analysis/claims.py reports: the mean article score (mean of the 8 strongest tokens) of a
    place's articles, divided by the same mean over the Colorado articles. The percentages on the page are this share.
The text around each article's peak token is rebuilt with the model's own tokenizer and checked against the token id
the GPU run saved at that position.

Run: uv run --with "transformers>=5.5" --with mistral-common --with sentencepiece --with protobuf --with numpy python analysis/build_page.py
"""
import json
import os

import numpy as np
from transformers import AutoTokenizer

R = "results"
OUT = "docs/data" if os.path.isdir("docs") else "artifact/data"   # the published repository keeps the page in docs/ (GitHub Pages)
FLOOR, CAP, FORCE_TOP, SKIP, MAXLEN, NTOP, MIN_N, PER_MODEL = 0.03, 4200, 12, 2, 224, 4, 12, 4
rng = np.random.default_rng(7)
# page prefix, display name, result tag, Hugging Face id (for the tokenizer)
PAGE = [
    ("qwen", "Qwen3-32B", "qwen3", "Qwen/Qwen3-32B"),
    ("llama", "Llama 3.3 70B", "llama33_70b", "unsloth/Llama-3.3-70B-Instruct"),
    ("gemma3", "Gemma 3 27B", "gemma3_27b", "unsloth/gemma-3-27b-it"),
    ("gemma4", "Gemma 4 31B", "gemma4", "google/gemma-4-31B-it"),
    ("mistral", "Mistral Small 3.2", "mistral32", "mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
    ("qwen36", "Qwen3.6-27B", "qwen36", "Qwen/Qwen3.6-27B"),
]
BANDS = [(0.5, 1.0001), (0.2, 0.5), (0.1, 0.2), (0.05, 0.1), (0.03, 0.05)]
SLOTS = [
    ("colorado", "core", "Colorado", "Articles that name Colorado before any other listed place"),
    ("mention", "word", "Colorado or Denver mentioned", "Articles that name another place first, or none, but contain the word Colorado or Denver"),
    ("us", "related", "Rest of the United States", "Articles that name one of the other 48 listed states first"),
    ("top1", "branch", None, "The place outside the United States whose articles fire this neuron most"),
    ("top2", "branch", None, "The second strongest place outside the United States for this neuron"),
    ("top3", "branch", None, "The third strongest place outside the United States for this neuron"),
    ("top4", "branch", None, "The fourth strongest place outside the United States for this neuron"),
    ("abroad", "other", "Other places abroad", "Articles that name any other country or listed region first"),
    ("none", "floor", "No listed place", "Articles that name no listed country, state or region first"),
]

docs = [json.loads(x) for x in open(f"{R}/blind_corpus.jsonl")]
D = len(docs)
title = [d["title"] for d in docs]
first = np.array([d["first_place"] or "" for d in docs])
kind = np.array([d["kind"] or "" for d in docs])
word = np.array([("Colorad" in d["text"]) or ("Denver" in d["text"]) for d in docs])
places = sorted({p for p in first if p})
pidx = {p: i for i, p in enumerate(places)}
doc_place = [pidx[p] if p else -1 for p in first]
is_col = first == "Colorado"
ment = ~is_col & word
us = (kind == "state") & ~is_col & ~ment
abroad = np.isin(kind, ("country", "region")) & ~ment
none = (kind == "") & ~ment
abroad_places = sorted({p for p, k in zip(first, kind) if k in ("country", "region")})
country_set = {p for p, k in zip(first, kind) if k == "country"}


def clean(s):
    return s.replace("�", "").replace("\n", " ⏎ ").replace("\t", " ").replace("<bos>", "").replace("<s>", "").replace("<|begin_of_text|>", "")


def tokenizer(hf):
    """The tokenizer and a function giving an article's token ids exactly as the GPU run fed them (probe/common.py encode_doc)."""
    tok = AutoTokenizer.from_pretrained(hf)
    bos = getattr(tok, "bos_token_id", None)
    probe = list(tok("hello world")["input_ids"])
    nb = bos is not None and (len(probe) == 0 or probe[0] != bos)
    cache = {}

    def ids_of(i):
        if i not in cache:
            ids = list(tok(docs[i]["text"])["input_ids"])[:MAXLEN]
            cache[i] = ([bos] + ids[: MAXLEN - 1]) if nb else ids
        return cache[i]
    return tok, ids_of


os.makedirs(OUT, exist_ok=True)
for fn in os.listdir(OUT):
    os.remove(f"{OUT}/{fn}")
records = []
for prefix, mlabel, tag, hf in PAGE:
    if not os.path.exists(f"{R}/tie/{tag}.npz"):
        print(f"{mlabel}: no result file, skipped"); continue
    z = np.load(f"{R}/tie/{tag}.npz"); J = json.load(open(f"{R}/tie/{tag}.json"))
    if "arg" not in z.files:
        print(f"{mlabel}: result file has no peak positions (older probe), skipped"); continue
    tok, ids_of = tokenizer(hf)
    raw = [(e["layer"], e["neuron"], e["pole"]) for e in J["selection"]["Colorado"]["raw"]]
    mism = 0; checked = 0
    for e in J["selection"]["Colorado"]["clean"][:PER_MODEL]:
        layer, neuron, pole, k = e["layer"], e["neuron"], e["pole"], e["k"]
        nid = f"{prefix}-l{layer}-n{neuron}"
        peak = z["max"][:, k].astype(np.float64)
        mx = float(peak.max())
        frac = peak / mx
        # the rule: the NTOP places abroad with the highest share of Colorado's level (same number as analysis/claims.py)
        score = z["top8"][:, k].astype(np.float64)
        ref = float(score[is_col].mean()) + 1e-9
        stats = []
        for p in abroad_places:
            m = abroad & (first == p)
            if m.sum() >= MIN_N:
                stats.append((p, float(score[m].mean()) / ref, int(m.sum())))
        stats.sort(key=lambda r: -r[1])
        top = stats[:NTOP]
        cat = np.full(D, 7)
        cat[is_col] = 0; cat[ment] = 1; cat[us] = 2; cat[none] = 8
        for j, (p, *_r) in enumerate(top):
            cat[abroad & (first == p)] = 3 + j
        names = [s[2] for s in SLOTS]
        for j, (p, *_r) in enumerate(top):
            names[3 + j] = p
        keep = np.flatnonzero(frac >= FLOOR)
        table = [[int(((frac >= lo) & (frac < hi) & (cat == c)).sum()) for lo, hi in BANDS] for c in range(len(SLOTS))]
        nart = [int((cat == c).sum()) for c in range(len(SLOTS))]
        cshare = [round(float(score[cat == c].mean()) / ref if (cat == c).any() else 0.0, 4) for c in range(len(SLOTS))]
        forced = set()
        for c in range(len(SLOTS)):
            idx = keep[cat[keep] == c]
            forced.update(idx.tolist() if c in (0, 1, 3, 4, 5, 6) else idx[np.argsort(-frac[idx])[:FORCE_TOP]].tolist())
        if len(keep) > CAP:
            rest = np.array([i for i in keep if i not in forced])
            pick = set(rng.choice(rest, max(CAP - len(forced), 0), replace=False).tolist()) | forced
        else:
            pick = set(keep.tolist())
        pick = np.array(sorted(pick, key=lambda i: -frac[i]))
        ctx = []
        for i in pick:
            w = ids_of(int(i)); p = int(z["arg"][i, k])
            checked += 1
            if p >= len(w) or w[p] != int(z["argtok"][i, k]):
                mism += 1
            before = clean(tok.decode(w[max(SKIP, p - 24):p]))[-78:]
            after = clean(tok.decode(w[p + 1:p + 7]))[:26]
            ctx.append([before, clean(tok.decode(w[p:p + 1])) or "·", after])
        with open(f"{OUT}/ctx-{nid}.json", "w") as f:
            f.write("[\n" + ",\n".join(json.dumps(c, ensure_ascii=False) for c in ctx) + "\n]\n")
        key = (layer, neuron, pole)
        brank = next((i + 1 for i, r in enumerate(stats) if r[0] == "Bolivia"), None)
        cstats = [r for r in stats if r[0] in country_set]
        bcrank = next((i + 1 for i, r in enumerate(cstats) if r[0] == "Bolivia"), None)
        records.append({
            "id": nid, "model": mlabel, "layer": layer, "neuron": neuron, "pole": pole, "max": round(mx, 1), "corr": e["corr"],
            "rank": raw.index(key) + 1 if key in raw else None, "nneurons": J["layers"] * J["neurons_per_layer"],
            "total": int(len(keep)), "shown": int(len(pick)), "names": names, "table": table, "nart": nart, "cshare": cshare,
            "tops": [[p, round(m, 4), n] for p, m, n in stats[:8]], "brank": brank, "nplaces": len(stats), "bcrank": bcrank, "ncountries": len(cstats),
            "bolivia": next(([round(r[1], 4), r[2]] for r in stats if r[0] == "Bolivia"), None),
            "d": [int(i) for i in pick], "c": [int(cat[i]) for i in pick], "f": [int(round(1000 * float(frac[i]))) for i in pick],
        })
        print(f"{nid:20s} max={mx:7.1f} articles>=3%: {len(keep):5d} shown {len(pick):5d} | share of Colorado's level, top abroad: "
              + ", ".join(f"{p} {m:.2f}" for p, m, n in stats[:5]) + f" | Bolivia #{brank} of {len(stats)} places, #{bcrank} of {len(cstats)} countries")
    print(f"   {mlabel}: peak-token check, {checked - mism} of {checked} drawn articles match the token id saved on the GPU run")
    assert mism == 0, "local tokenisation differs from the GPU run"

with open(f"{OUT}/index.json", "w") as f:
    head = [{"kind": "cats", "slots": [dict(key=s[0], type=s[1], desc=s[3]) for s in SLOTS], "bands": [[lo, min(hi, 1.0)] for lo, hi in BANDS],
             "floor": FLOOR, "ndocs": D, "nplaces": len(places)},
            {"kind": "docs", "titles": title, "place": doc_place, "places": places}]
    f.write("[\n" + ",\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in head + records) + "\n]\n")
bad = [fn for fn in os.listdir(OUT) if "�" in open(f"{OUT}/{fn}").read()]
print("files with U+FFFD:", bad)
print({fn: os.path.getsize(f"{OUT}/{fn}") // 1024 for fn in sorted(os.listdir(OUT))})
