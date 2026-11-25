"""Blind Wikipedia sample for the Colorado-neuron scan. No article is chosen by hand.

1. RANDOM: N_RANDOM articles drawn uniformly at random from the whole dump.
2. BUCKETS: for every country, U.S. state and a neutral list of regions, M random articles whose
   opening names that place before any other place on the list.
Every place is treated the same way; nothing about Bolivia or Colorado is special-cased.

Usage: python3 make_blind_corpus.py <parquet_dir> <out.jsonl>
"""
import glob
import heapq
import json
import re
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

from places import US_STATES, build

M, N_RANDOM, SEED, HEAD, KEEP = 24, 2000, 20261123, 800, 900
REGIONS = ["Alberta", "British Columbia", "Ontario", "Quebec", "Saskatchewan", "Manitoba", "Nova Scotia", "Tibet", "Sichuan", "Yunnan", "Xinjiang",
           "Siberia", "Bavaria", "Tyrol", "Scotland", "Wales", "Catalonia", "Sicily", "Tuscany", "Lombardy", "Andalusia", "Kashmir", "Punjab",
           "Kerala", "Tamil Nadu", "Hokkaido", "Tasmania", "Queensland", "Patagonia", "Transylvania", "Lapland", "Crimea", "Kurdistan",
           "Puerto Rico", "Greenland", "Hong Kong", "Taiwan", "Northern Ireland"]
MASK_ONLY = ["New South Wales", "American Samoa", "New Guinea", "Washington, D.C.", "Washington D.C.", "United States", "South America",
             "North America", "Georgia"]
RENAME = {"The Bahamas": "Bahamas", "The Gambia": "Gambia", "Republic of Ireland": "Ireland", "Republic of the Congo": "Congo",
          "Democratic Republic of the Congo": "Congo", "Federated States of Micronesia": "Micronesia"}


def terms():
    t = {}
    for e in build():
        if e["cat"] == "country" and "Georgia" not in e["title"] and e["title"] != "United States":
            t[RENAME.get(e["title"], e["title"])] = "country"
    for s, *_ in US_STATES:
        if s != "Georgia":
            t[s] = "state"
    for r in REGIONS:
        t.setdefault(r, "region")
    return t


TERMS = terms()
ALT = sorted(list(TERMS) + MASK_ONLY, key=lambda s: -len(s))
RX = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in ALT) + r")\b")


def cut(text):
    text = text[:KEEP]
    k = text.rfind(" ")
    return text[:k] if k > KEEP * 0.7 else text


def scan(args):
    si, path = args
    t = pq.read_table(path, columns=["title", "text"])
    titles, texts = t["title"].to_pylist(), t["text"].to_pylist()
    pri = np.random.default_rng(SEED + si).random(len(titles))
    rand, buckets = [], {}
    for j, (title, text) in enumerate(zip(titles, texts)):
        if len(text) < 500 or title.startswith("List of") or "may refer to" in text[:200] or "(disambiguation)" in title:
            continue
        m = RX.search(text[:HEAD])
        term = m.group(0) if m and m.group(0) in TERMS else None
        rec = (float(pri[j]), title, cut(text), term)
        if len(rand) < N_RANDOM:
            heapq.heappush(rand, (-rec[0], rec))
        elif -rand[0][0] > rec[0]:
            heapq.heapreplace(rand, (-rec[0], rec))
        if term:
            b = buckets.setdefault(term, [])
            if len(b) < M:
                heapq.heappush(b, (-rec[0], rec))
            elif -b[0][0] > rec[0]:
                heapq.heapreplace(b, (-rec[0], rec))
    return [r for _, r in rand], {k: [r for _, r in v] for k, v in buckets.items()}, len(titles)


def main(pq_dir, out):
    files = sorted(set(glob.glob(f"{pq_dir}/**/*.parquet", recursive=True)))
    print(len(files), "parquet files;", len(TERMS), "places", flush=True)
    with Pool(min(24, len(files))) as p:
        res = p.map(scan, list(enumerate(files)))
    total = sum(r[2] for r in res)
    rand = sorted((x for r in res for x in r[0]))[:N_RANDOM]
    buckets = {}
    for r in res:
        for k, v in r[1].items():
            buckets.setdefault(k, []).extend(v)
    docs = {}
    for pr, title, text, term in rand:
        docs[title] = {"title": title, "text": text, "first_place": term, "random": True, "bucket": None}
    short = []
    for k in sorted(TERMS):
        v = sorted(buckets.get(k, []))[:M]
        if len(v) < M:
            short.append((k, len(v)))
        for pr, title, text, term in v:
            d = docs.setdefault(title, {"title": title, "text": text, "first_place": term, "random": False, "bucket": None})
            d["bucket"] = k
    rows = list(docs.values())
    for i, d in enumerate(rows):
        d["id"] = i
        d["kind"] = TERMS.get(d["first_place"])
    with open(out, "w") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"scanned {total} articles -> {len(rows)} docs ({sum(d['random'] for d in rows)} random, {sum(d['bucket'] is not None for d in rows)} in buckets)", flush=True)
    print("buckets short of", M, ":", short, flush=True)
    for k in ("Bolivia", "Colorado", "Nepal", "Switzerland"):
        print(k, [d["title"] for d in rows if d["bucket"] == k][:8], flush=True)
    print("CORPUS_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
