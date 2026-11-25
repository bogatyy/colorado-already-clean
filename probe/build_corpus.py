"""Build corpus.jsonl: one plain-text Wikipedia article (first ~9k chars) per probe entry.

Usage: python3 build_corpus.py <parquet_dir> out/corpus.jsonl

Source 1: a static HF Wikipedia parquet dump (legacy-datasets/wikipedia, 20220301.en).
Source 2: the live Wikipedia API, serially and patiently, only for titles the dump lacks
          (the API throttles anonymous bulk fetches hard, so it cannot be the main source).
"""
import glob
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from places import build

API = "https://en.wikipedia.org/w/api.php"
UA = "neuron-probe-corpus/0.1 (research script; python-urllib)"
MAX_CHARS = 9000

# Canonical-title alternates for entries whose first-choice title is a redirect in the dump.
ALT = {
    "Glacier National Park (U.S.)": ["Glacier National Park (United States)"],
    "Tyrol (federal state)": ["Tyrol (state)"],
    "Craft brewery and microbrewery": ["Craft brewery", "Microbrewery", "Craft beer"],
    "East Timor": ["Timor-Leste"],
    "Czech Republic": ["Czechia"],
    "Turkey": ["Türkiye"],
    "Lapland (Finland)": ["Lapland (Finnish region)"],
    "Amazonas (Brazilian state)": ["Amazonas (state)"],
    "Nagano (city)": ["Nagano, Nagano"],
    "Mendoza, Argentina": ["Mendoza (city)"],
    "Bariloche": ["San Carlos de Bariloche"],
    "Football": ["Association football"],
    "Human brain": ["Brain"],
    "Cannabis (drug)": ["Cannabis"],
    "Mount Kilimanjaro": ["Kilimanjaro"],
    "Great Sand Dunes National Park and Preserve": ["Great Sand Dunes National Park"],
    "Whitewater kayaking": ["Whitewater"],
    "Eswatini": ["Swaziland"],
    "North Macedonia": ["Republic of Macedonia"],
    "Cape Verde": ["Cabo Verde"],
}


def candidates(e):
    c = [e["title"]]
    if e["cat"] == "us_city":
        c.append(e["title"].split(",")[0])          # big cities have bare titles (Denver, Boston, ...)
    c += ALT.get(e["title"], [])
    return c


def fetch_api(title):
    q = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
        "format": "json", "formatversion": 2, "titles": title,
    })
    for attempt in range(8):
        try:
            req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            page = d["query"]["pages"][0]
            if page.get("missing"):
                return None, ""
            return page.get("title"), page.get("extract", "")
        except urllib.error.HTTPError as e:
            time.sleep(min(90, float(e.headers.get("Retry-After", 10 * (attempt + 1)))))
        except Exception:  # noqa: BLE001
            time.sleep(5 * (attempt + 1))
    return None, ""


def clean(text):
    text = text.replace("\r", "")
    if len(text) > MAX_CHARS:
        cut = text.rfind("\n", 0, MAX_CHARS)
        text = text[: cut if cut > MAX_CHARS * 0.6 else MAX_CHARS]
    return text.strip()


def main(pq_dir, out):
    entries = build()
    wanted = set()
    for e in entries:
        wanted.update(candidates(e))
    found = {}
    files = sorted(glob.glob(f"{pq_dir}/**/*.parquet", recursive=True)) + sorted(glob.glob(f"{pq_dir}/*.parquet"))
    files = sorted(set(files))
    print(len(files), "parquet files", file=sys.stderr, flush=True)
    wset = pa.array(sorted(wanted))
    for fpath in files:
        t = pq.read_table(fpath, columns=["title", "text"])
        t = t.filter(pc.is_in(t["title"], value_set=wset))
        for title, text in zip(t["title"].to_pylist(), t["text"].to_pylist()):
            found[title] = text
    n_ok = n_api = 0
    rows = []
    for e in entries:
        e = dict(e)
        txt, res, src = "", None, None
        for c in candidates(e):
            if c in found and len(found[c]) > 400:
                txt, res, src = found[c], c, "dump"
                break
        e.update(resolved=res, text=clean(txt), source=src)
        rows.append(e)
    missing = [e for e in rows if not e["text"]]
    print(f"dump covered {len(rows) - len(missing)}/{len(rows)}; fetching {len(missing)} from the API", file=sys.stderr, flush=True)
    for e in missing:
        rt, txt = fetch_api(e["title"])
        time.sleep(1.0)
        if txt:
            e.update(resolved=rt, text=clean(txt), source="api")
            n_api += 1
        else:
            print("MISSING", e["cat"], "|", e["title"], file=sys.stderr, flush=True)
    with open(out, "w") as f:
        for e in rows:
            n_ok += bool(e["text"])
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"{n_ok}/{len(rows)} docs with text ({n_api} via API) -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
