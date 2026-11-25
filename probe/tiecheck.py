"""Does a model tie Colorado to Bolivia? Same method as the earlier models, in one script, for any causal LM.

Pass 0 (selection, hand-built corpus, US and general articles only): every MLP neuron is scored by the
   correlation between its per-article firing and a 0/1 "this article is about <state>" indicator. The best
   clean ones are kept for a few states. Foreign articles play no part.
Pass 1 (reading, blind Wikipedia sample): how hard those neurons fire on every article of the blind sample.
Also: the stated answer to "which state is most similar to X", with full state names; and an experimental
neuron-free direction measure (it failed the Qwen3-32B control, so it is recorded but not relied on).

Works for gated MLPs (Llama, Qwen, Gemma, Mistral), Gensyn's fused gate+up layout, and plain MLPs (Pythia).
"""
import argparse
import json
import time

import numpy as np
import torch

from common import encode_doc, needs_bos, pad_id
from places import US_STATES

SKIP = 2
FOCUS = ["Colorado", "Utah", "New Mexico", "Wyoming", "Montana"]
US_CATS = ("us_state", "us_city", "us_other", "colorado_extra")
FRAMES = {
    "neutral": ("Which U.S. state is most similar to {X}?", "The U.S. state most similar to {X} is"),
    "physical": ("Considering only physical geography (terrain, elevation and climate), which U.S. state is most similar to {X}?",
                 "In physical geography, the U.S. state most similar to {X} is"),
    "cultural": ("Considering only culture and people (language, ethnic makeup, religion and traditions), which U.S. state is most similar to {X}?",
                 "In culture and people, the U.S. state most similar to {X} is"),
}
ASK = ["Bolivia", "Peru", "Ecuador", "Nepal", "Switzerland", "Kyrgyzstan", "North Macedonia", "Tibet", "Sichuan", "Alberta", "Mexico", "Japan"]


def load(model_id, trust, revision, headroom=22):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = dict(trust_remote_code=True) if trust else {}
    if revision:
        kw["revision"] = revision
    tok = AutoTokenizer.from_pretrained(model_id, **kw)
    t0 = time.time()
    ng = torch.cuda.device_count()
    if ng > 1:  # split the layers over the GPUs, leaving room on each for activations and the hooks' temporaries
        free = [torch.cuda.mem_get_info(i)[0] / 2 ** 30 for i in range(ng)]
        dm = dict(device_map="auto", max_memory={i: f"{int(f - headroom)}GiB" for i, f in enumerate(free)})
        print(f"{ng} GPUs, free GiB {[round(f) for f in free]}, headroom {headroom} GiB each", flush=True)
    else:
        dm = dict(device_map={"": 0})
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, **dm, **kw)
    except Exception as e:  # noqa: BLE001
        print("AutoModelForCausalLM failed:", repr(e)[:300], flush=True)
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, **dm, **kw)
    model.eval(); torch.set_grad_enabled(False)
    print(f"loaded {model_id} ({type(model).__name__}, tokenizer {type(tok).__name__}) in {time.time() - t0:.0f}s", flush=True)
    return tok, model


def find_blocks(model):
    best = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) >= 4 and not any(s in name for s in ("vision", "audio", "visual")):
            if best is None or len(m) > len(best[1]):
                best = (name, m)
    return best


def mlp_spec(block):
    for _, m in block.named_modules():
        if all(hasattr(m, a) for a in ("gate_proj", "up_proj", "down_proj")):
            return dict(kind="gated", down=m.down_proj, gate=m.gate_proj, fused=False)
        if hasattr(m, "w_gate_up") and hasattr(m, "w_down"):
            return dict(kind="gated", down=m.w_down, gate=m.w_gate_up, fused=True)
        if hasattr(m, "gate_up_proj") and hasattr(m, "down_proj"):   # Phi-3 / Phi-4: one layer gives [gate, up]
            return dict(kind="gated", down=m.down_proj, gate=m.gate_up_proj, fused=True)
        if hasattr(m, "dense_h_to_4h") and hasattr(m, "dense_4h_to_h"):
            return dict(kind="plain", down=m.dense_4h_to_h, gate=None, fused=False)
    raise RuntimeError("no MLP found in block: " + ", ".join(n for n, _ in block.named_modules())[:400])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)            # prefix
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust", action="store_true")
    ap.add_argument("--maxlen", type=int, default=224)
    ap.add_argument("--hand_maxlen", type=int, default=512)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--nodir", action="store_true")    # skip the experimental direction measure (saves RAM and time)
    ap.add_argument("--headroom", type=int, default=22)  # GiB kept free per GPU when the model is split over several
    a = ap.parse_args()

    hand = [json.loads(l) for l in open(a.hand)]
    hand = [d for d in hand if d["cat"] in US_CATS or d["cat"] == "generic"]
    docs = [json.loads(l) for l in open(a.docs)]
    if a.limit:
        rng = np.random.default_rng(0)
        keep = set(rng.choice(len(docs), a.limit, replace=False).tolist()) | {i for i, d in enumerate(docs) if d["first_place"] in FOCUS + ["Bolivia", "Nepal"]}
        docs = [docs[i] for i in sorted(keep)]
    D = len(docs)
    places = sorted({d["first_place"] for d in docs if d["first_place"]})
    kind = {d["first_place"]: d["kind"] for d in docs if d["first_place"]}
    states = [p for p in places if kind[p] == "state"]
    abroad = [p for p in places if kind[p] != "state"]
    countries = [p for p in places if kind[p] == "country"]
    half = np.array([d["id"] % 2 for d in docs])
    first = np.array([d["first_place"] or "" for d in docs])
    word = np.array([("Colorad" in d["text"]) or ("Denver" in d["text"]) for d in docs])

    tok, model = load(a.model, a.trust, a.revision, a.headroom)
    dev = model.device
    sane = {}
    for text in ["The capital of France is", "Denver is the capital of the U.S. state of"]:
        ids = torch.tensor([encode_doc(tok, text, 64)], device=dev)
        lg = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1].float()
        sane[text] = [tok.decode([i]) for i in lg.topk(3).indices.tolist()]
    print("LM sanity (top-3 next tokens):", sane, flush=True)

    bname, blocks = find_blocks(model)
    specs = [mlp_spec(b) for b in blocks]
    L = len(blocks)
    w = specs[0]["down"].weight
    N, H = int(w.shape[1]), int(w.shape[0])
    poles = ("P", "M") if specs[0]["kind"] == "gated" else ("P",)
    NP = len(poles)
    print(f"blocks: {bname} x{L}; MLP kind {specs[0]['kind']}{' (fused gate+up)' if specs[0]['fused'] else ''}; {N} neurons per layer, hidden {H}; poles {poles}", flush=True)

    # ---- running statistics for pass 0
    SX = torch.zeros((L, NP, N), device=dev); SXX = torch.zeros((L, NP, N), device=dev)
    SS = torch.zeros((len(FOCUS), L, NP, N), device=dev); GM = torch.zeros((L, NP, N), device=dev)
    st = {"on": False, "mode": "select", "gate": None}
    sel_by_layer = {}
    out_arrays = {}

    def gate_hook(mod, inp, out):
        if st["on"]:
            st["gate"] = out

    def make_down(l):
        fused = specs[l]["fused"]

        def hook(mod, inp):
            if not st["on"]:
                return
            act = inp[0]
            gate = st["gate"]
            if gate is not None and fused:
                gate = gate.chunk(2, dim=-1)[0]
            valid = st["valid"].to(act.device)
            if st["mode"] == "select":
                for pi, pole in enumerate(poles):
                    v = act.float() if pole == "P" else -act.float()
                    cond = (v > 0) & valid[:, :, None]
                    if gate is not None:
                        cond &= gate > 0
                    v = torch.where(cond, v, torch.zeros_like(v))
                    t8 = v.topk(min(8, v.shape[1]), dim=1).values.mean(1).to(dev)    # [B, N]
                    SX[l, pi] += t8.sum(0); SXX[l, pi] += (t8 * t8).sum(0)
                    for si in range(len(FOCUS)):
                        msk = st["state_id"] == si
                        if msk.any():
                            SS[si, l, pi] += t8[msk].sum(0)
                    if st["generic"].any():
                        GM[l, pi] = torch.maximum(GM[l, pi], t8[st["generic"]].max(0).values)
            elif l in sel_by_layer:
                ks, idx, sg = sel_by_layer[l]
                idx, sg = idx.to(act.device), sg.to(act.device)
                v = act[:, :, idx].float() * sg
                cond = (v > 0) & valid[:, :, None]
                if gate is not None:
                    cond &= gate[:, :, idx] > 0
                v = torch.where(cond, v, torch.zeros_like(v))
                peak = v.max(1)
                mv, pos = peak.values, peak.indices          # pos: token position of the strongest firing, [B, k]
                t8 = v.topk(min(8, v.shape[1]), dim=1).values.mean(1)
                out_arrays["top8"][np.ix_(st["idx"], ks)] = t8.cpu().numpy(); out_arrays["max"][np.ix_(st["idx"], ks)] = mv.cpu().numpy()
                out_arrays["arg"][np.ix_(st["idx"], ks)] = pos.cpu().numpy()
                out_arrays["argtok"][np.ix_(st["idx"], ks)] = st["x"].to(pos.device).gather(1, pos).cpu().numpy()
            st["gate"] = None
        return hook

    def make_block(l):
        def hook(mod, inp, out):
            if not st["on"] or st["mode"] != "read":
                return
            h = out[0] if isinstance(out, (tuple, list)) else out
            m = st["valid"].to(h.device)[:, :, None].float()
            out_arrays["X"][st["idx"], l] = ((h.float() * m).sum(1) / m.sum(1).clamp_min(1)).to(torch.float16).cpu().numpy()
        return hook

    for l, (b, sp) in enumerate(zip(blocks, specs)):
        if sp["gate"] is not None:
            sp["gate"].register_forward_hook(gate_hook)
        sp["down"].register_forward_pre_hook(make_down(l))
        if not a.nodir:
            b.register_forward_hook(make_block(l))
    base = model.base_model if model.base_model is not model else model

    def sweep(id_lists, bs, extra, label):
        order = sorted(range(len(id_lists)), key=lambda i: -len(id_lists[i]))
        t0 = time.time(); done = 0
        for b0 in range(0, len(order), bs):
            bi = order[b0:b0 + bs]
            T = len(id_lists[bi[0]])
            x = torch.full((len(bi), T), pad_id(tok), dtype=torch.long); m = torch.zeros((len(bi), T), dtype=torch.long)
            for k, i in enumerate(bi):
                x[k, : len(id_lists[i])] = torch.tensor(id_lists[i]); m[k, : len(id_lists[i])] = 1
            x, m = x.to(dev), m.to(dev)
            valid = m.bool().clone(); valid[:, :SKIP] = False
            st.update(valid=valid, x=x, idx=np.array(bi), on=True, **extra(bi))
            base(input_ids=x, attention_mask=m, use_cache=False)
            st["on"] = False
            done += int(m.sum())
            if (b0 // bs) % 20 == 0:
                el = time.time() - t0
                print(f"{label} {b0 + len(bi)}/{len(order)}  {el:.0f}s  {done / max(el, 1e-6):.0f} tok/s", flush=True)
        return time.time() - t0

    # ------------------------------------------------------------------ pass 0: choose neurons on US + general articles
    h_ids = [encode_doc(tok, d["text"], a.hand_maxlen) for d in hand]
    h_state = torch.tensor([FOCUS.index(d["group"]) if (d["cat"] in US_CATS and d["group"] in FOCUS) else -1 for d in hand], device=dev)
    h_gen = torch.tensor([d["cat"] == "generic" for d in hand], device=dev)
    st["mode"] = "select"
    t_sel = sweep(h_ids, max(8, a.bs // 2), lambda bi: dict(state_id=h_state[bi], generic=h_gen[bi]), "select")
    n = float(len(hand))
    var_x = (SXX / n - (SX / n) ** 2).clamp_min(0)
    selection = {}
    flat = []
    index = {}
    for si, s in enumerate(FOCUS):
        ns = float((h_state == si).sum())
        p = ns / n
        cov = SS[si] / n - p * (SX / n)
        corr = cov / (var_x * p * (1 - p) + 1e-12).sqrt()
        mean_s = SS[si] / ns
        ok = (mean_s > 0) & (GM <= 0.3 * mean_s) & (corr >= 0.5)
        fams = {}
        for fam, c in (("clean", torch.where(ok, corr, torch.full_like(corr, -1.0)).flatten()), ("raw", corr.flatten())):
            top = c.topk(a.k)
            lst = []
            for v, j in zip(top.values.tolist(), top.indices.tolist()):
                if v < 0:
                    continue
                l, pi, nn_ = j // (NP * N), (j // N) % NP, j % N
                key = (s, l, nn_, poles[pi])
                if key not in index:
                    index[key] = len(flat); flat.append(key)
                lst.append({"layer": l, "neuron": nn_, "pole": poles[pi], "corr": round(v, 4), "k": index[key], "clean": bool(ok.flatten()[j])})
            fams[fam] = lst
        selection[s] = fams
        print(f"select {s:10s} ({int(ns)} articles): {int(ok.sum())} clean neurons with corr>=0.5; clean best: "
              + ", ".join(f"L{e['layer']}/N{e['neuron']}{e['pole']} {e['corr']:.2f}" for e in fams["clean"][:8]) + " | best by correlation alone: "
              + ", ".join(f"L{e['layer']}/N{e['neuron']}{e['pole']} {e['corr']:.2f}{'' if e['clean'] else '*'}" for e in fams["raw"][:8]), flush=True)
    by = {}
    for k, (s, l, nn_, pole) in enumerate(flat):
        by.setdefault(l, []).append((k, nn_, 1.0 if pole == "P" else -1.0))
    for l, ent in by.items():
        sel_by_layer[l] = ([e[0] for e in ent], torch.tensor([e[1] for e in ent], device=dev), torch.tensor([e[2] for e in ent], device=dev))
    del SX, SXX, SS, GM
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ pass 1: read them on the blind sample
    K = max(len(flat), 1)
    out_arrays["top8"] = np.zeros((D, K), np.float32); out_arrays["max"] = np.zeros((D, K), np.float32)
    out_arrays["arg"] = np.zeros((D, K), np.int16); out_arrays["argtok"] = np.zeros((D, K), np.int32)   # where each article's peak is, and which token
    if not a.nodir:
        out_arrays["X"] = np.zeros((D, L, H), dtype=np.float16)
    ids = [encode_doc(tok, d["text"], a.maxlen) for d in docs]
    st["mode"] = "read"
    t_read = sweep(ids, a.bs, lambda bi: {}, "read")
    top8 = out_arrays["top8"]

    reading = {}
    for s in FOCUS:
        reading[s] = {}
        if not (first == s).any():
            continue
        has = word if s == "Colorado" else np.array([s in d["text"] for d in docs])
        own = first == s
        for fam in ("clean", "raw"):
            lst = selection[s][fam]
            if not lst:
                continue
            per = []
            agg = {k: np.zeros(len(abroad)) for k in (1, 4, 8, 16)}
            for r, e in enumerate(lst):
                col = top8[:, e["k"]]
                ref = float(col[own].mean()) + 1e-9
                sh = np.array([col[(first == p) & ~has].mean() / ref if ((first == p) & ~has).any() else 0.0 for p in abroad])
                st_sh = sorted(([q, float(col[(first == q) & ~has].mean() / ref)] for q in states if q != s and ((first == q) & ~has).any()), key=lambda t: -t[1])[:3]
                cs = np.array([sh[abroad.index(c)] for c in countries]); o = np.argsort(-cs)
                none_sh = float(col[(first == "") & ~has].mean() / ref) if ((first == "") & ~has).any() else 0.0
                rec = {"layer": e["layer"], "neuron": e["neuron"], "pole": e["pole"], "corr": e["corr"], "clean": e["clean"], "own_level": round(ref, 4), "no_place": round(none_sh, 4),
                       "bolivia": round(float(cs[countries.index("Bolivia")]), 4), "bolivia_rank": int(np.where(np.array(countries)[o] == "Bolivia")[0][0]) + 1,
                       "top": [[countries[j], round(float(cs[j]), 3)] for j in o[:6]], "top_abroad": [[abroad[j], round(float(sh[j]), 3)] for j in np.argsort(-sh)[:6]],
                       "top_states": [[q, round(v, 3)] for q, v in st_sh]}
                per.append(rec)
                for k in agg:
                    if r < k:
                        agg[k] += np.clip(sh, 0, 3) / min(k, len(lst))
            aggout = {}
            for k, v in agg.items():
                cs = np.array([v[abroad.index(c)] for c in countries]); o = np.argsort(-cs)
                aggout[k] = {"n": min(k, len(lst)), "bolivia": round(float(cs[countries.index("Bolivia")]), 4), "bolivia_rank": int(np.where(np.array(countries)[o] == "Bolivia")[0][0]) + 1,
                             "top": [[countries[j], round(float(cs[j]), 3)] for j in o[:10]], "top_abroad": [[abroad[j], round(float(v[j]), 3)] for j in np.argsort(-v)[:10]]}
            reading[s][fam] = {"neurons": per, "aggregate": aggout}
            print(f"\nREAD {s} [{fam}]: {len(lst)} neurons (share of their level on random {s} articles)", flush=True)
            for rec in per[:8]:
                print(f"   L{rec['layer']:2d}/N{rec['neuron']:5d}{rec['pole']} corr={rec['corr']:.2f}{'' if rec['clean'] else '*'} Bolivia {rec['bolivia']:.2f} (#{rec['bolivia_rank']:3d}) | countries: "
                      + ", ".join(f"{p} {v:.2f}" for p, v in rec["top"][:4]) + " | abroad: " + ", ".join(f"{p} {v:.2f}" for p, v in rec["top_abroad"][:3])
                      + " | states: " + ", ".join(f"{p} {v:.2f}" for p, v in rec["top_states"][:2]) + f" | no place {rec['no_place']:.2f}", flush=True)
            for k in (1, 4, 8, 16):
                print(f"   mean of top {aggout[k]['n']:2d}: Bolivia {aggout[k]['bolivia']:.2f} (#{aggout[k]['bolivia_rank']} of {len(countries)}) | " + ", ".join(f"{p} {x:.2f}" for p, x in aggout[k]["top"][:7]), flush=True)

    # ------------------------------------------------------------------ experimental: neuron-free direction (blind halves)
    is_state = np.array([bool(p) and kind.get(p) == "state" for p in first])
    clean = ~word | (first == "Colorado")
    Xf = None if a.nodir else torch.tensor(out_arrays["X"].astype(np.float32))
    ci = [places.index(c) for c in countries]
    layers = []
    for l in range(0 if a.nodir else L):
        Xl = Xf[:, l].to(dev)
        sc = np.zeros(len(places)); dps = []
        for f in (0, 1):
            pos = torch.tensor((first == "Colorado") & (half == f), device=dev)
            neg = torch.tensor(is_state & (first != "Colorado") & (half == f) & clean, device=dev)
            d = Xl[pos].mean(0) - Xl[neg].mean(0)
            d = d / d.norm().clamp_min(1e-9)
            pr = (Xl @ d - (Xl[neg] @ d).mean()).cpu().numpy()
            hp = pr[(first == "Colorado") & (half != f)]; hn = pr[is_state & (first != "Colorado") & (half != f) & clean]
            dps.append((hp.mean() - hn.mean()) / (np.sqrt((hp.var() + hn.var()) / 2) + 1e-9))
            for j, p in enumerate(places):
                msk = (first == p) & clean
                if kind[p] == "state":
                    msk &= half != f
                sc[j] += pr[msk].mean() / (hp.mean() + 1e-9) / 2
        cs = sc[ci]; o = np.argsort(-cs)
        layers.append({"layer": l, "dprime": round(float(np.mean(dps)), 3), "bolivia": round(float(sc[places.index("Bolivia")]), 3),
                       "bolivia_rank": int(np.where(np.array(countries)[o] == "Bolivia")[0][0]) + 1, "top": [[countries[j], round(float(cs[j]), 3)] for j in o[:6]]})

    # ------------------------------------------------------------------ stated answer
    stn = [s for s, *_ in US_STATES]
    nb = needs_bos(tok)

    def score(prompt):
        p_ids = list(tok(prompt)["input_ids"])
        conts = []
        for s in stn:
            full = list(tok(prompt + " " + s)["input_ids"])
            conts.append(full[len(p_ids):] if full[: len(p_ids)] == p_ids and len(full) > len(p_ids) else [i for i in tok(" " + s)["input_ids"] if i != getattr(tok, "bos_token_id", None)])
        head = ([tok.bos_token_id] if nb else []) + p_ids
        P = len(head); out = []
        for b0 in range(0, len(stn), 25):
            rws = [head + c for c in conts[b0:b0 + 25]]
            T = max(len(r_) for r_ in rws)
            x = torch.full((len(rws), T), pad_id(tok), dtype=torch.long); m = torch.zeros((len(rws), T), dtype=torch.long)
            for k, r_ in enumerate(rws):
                x[k, : len(r_)] = torch.tensor(r_); m[k, : len(r_)] = 1
            lp = torch.log_softmax(model(input_ids=x.to(dev), attention_mask=m.to(dev), use_cache=False).logits[:, P - 1:].float(), -1)
            for k, c in enumerate(conts[b0:b0 + 25]):
                out.append(float(sum(lp[k, j, c[j]] for j in range(len(c)))))
        return np.exp(np.array(out))

    said = {}
    print("\nSTATED ANSWER (share among the 50 full state names; [mass] = probability on any state name):", flush=True)
    for x in ASK:
        said[x] = {}
        line = f"   {x:16s}"
        for f, (q, ans) in FRAMES.items():
            p = score(f"Q: {q}\nA: {ans}".replace("{X}", x)); tot = p.sum(); o = np.argsort(-p)[:3]
            said[x][f] = {"mass": round(float(tot), 4), "colorado": round(float(p[stn.index("Colorado")] / tot), 4), "utah": round(float(p[stn.index("Utah")] / tot), 4),
                          "new_mexico": round(float(p[stn.index("New Mexico")] / tot), 4), "top": [[stn[j], round(float(p[j] / tot), 3)] for j in o]}
            line += f" | {f}: " + ", ".join(f"{stn[j]} {p[j] / tot:.2f}" for j in o) + f" [{tot:.2f}]"
        print(line, flush=True)

    json.dump({"model": a.model, "revision": a.revision, "n_docs": D, "layers": L, "neurons_per_layer": N, "mlp": specs[0]["kind"], "poles": list(poles), "sanity": sane,
               "select_seconds": round(t_sel, 1), "read_seconds": round(t_read, 1), "selection": selection, "reading": reading, "direction_experimental": layers,
               "stated": said, "states": states, "countries": countries, "neurons": [list(f) for f in flat]}, open(a.out + ".json", "w"), ensure_ascii=False)
    np.savez_compressed(a.out + ".npz", top8=top8, max=out_arrays["max"], arg=out_arrays["arg"], argtok=out_arrays["argtok"], ntok=np.array([len(x) for x in ids], np.int32))
    print(f"TIECHECK_DONE {a.model} select {t_sel:.0f}s read {t_read:.0f}s", flush=True)


if __name__ == "__main__":
    main()
