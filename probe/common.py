"""Model loading and MLP-neuron hooks shared by all passes.

A "neuron" is one hidden unit of a gated MLP block:
    down_proj( act(gate_proj(x)) * up_proj(x) )
Its activation is the product act(gate)*value, i.e. the input to down_proj.
We split it into sign regimes (the essay's SwiGLU point): only tokens where the gate is
open (gate pre-activation > 0) count.  positive pole = product > 0, negative pole = product < 0.
The closed-gate "double negative" leak is excluded from both poles.
"""
import time
from functools import partial

import torch


def load(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    print("tokenizer:", type(tok).__name__, flush=True)
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": 0})
    except Exception as e:  # noqa: BLE001
        print("AutoModelForCausalLM failed:", repr(e)[:400], flush=True)
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    torch.set_grad_enabled(False)
    print(f"loaded {model_id} as {type(model).__name__} in {time.time() - t0:.0f}s", flush=True)
    return tok, model


_BOS_CACHE = {}


def needs_bos(tok):
    """Some tokenizers (Gemma 4) only add <bos> through the chat template; raw text then runs
    the model in a degenerate regime. Detect that so callers can prepend it themselves."""
    k = id(tok)
    if k not in _BOS_CACHE:
        if getattr(tok, "bos_token_id", None) is None:
            _BOS_CACHE[k] = False
        else:
            ids = list(tok("hello world")["input_ids"])
            _BOS_CACHE[k] = len(ids) == 0 or ids[0] != tok.bos_token_id
    return _BOS_CACHE[k]


def pad_id(tok):
    for k in ("pad_token_id", "eos_token_id"):
        v = getattr(tok, k, None)
        if isinstance(v, int):
            return v
    return 0


def first_token_at(tok, text, start, method="auto"):
    """Index (in the tokenizer's own ids) of the first token that begins at or after char `start`.
    With GPT-2/SentencePiece-style vocabularies the first token of a word carries the leading
    space, so this is the token AFTER the first token of the word that starts at `start`.
    Fast tokenizers give this from offsets; otherwise count the tokens of the prefix (+1)."""
    if method in ("auto", "offsets"):
        try:
            off = tok(text, return_offsets_mapping=True)["offset_mapping"]
            for j, (s, e) in enumerate(off):
                if e > s and s >= start:
                    return j
            return len(off)
        except Exception:  # noqa: BLE001
            if method == "offsets":
                raise
    return len(list(tok(text[:start].rstrip(" "))["input_ids"])) + 1


def encode(tok, texts, device=None, starts=None):
    """Right-padded batch built by hand (works for any tokenizer), with <bos> prepended when the
    tokenizer does not add it. With `starts` (char offsets), also returns valid_from[B]."""
    nb = needs_bos(tok)
    rows, vf = [], []
    for i, t in enumerate(texts):
        ids = list(tok(t)["input_ids"])
        if starts is not None:
            vf.append(first_token_at(tok, t, starts[i]) + (1 if nb else 0))
        rows.append(([tok.bos_token_id] if nb else []) + ids)
    T = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), T), pad_id(tok), dtype=torch.long)
    mask = torch.zeros((len(rows), T), dtype=torch.long)
    for i, r in enumerate(rows):
        input_ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    out = {"input_ids": input_ids, "attention_mask": mask}
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    if starts is not None:
        out["valid_from"] = torch.tensor(vf)
    return out


def encode_doc(tok, text, maxlen):
    ids = list(tok(text)["input_ids"])[:maxlen]
    if needs_bos(tok):
        ids = [tok.bos_token_id] + ids[: maxlen - 1]
    return ids


def lm_sanity(model, tok):
    """The model must make sensible next-token predictions on raw text, or activations are meaningless."""
    res = {}
    for text in ["The capital of France is", "Denver is the capital of the U.S. state of"]:
        enc = encode(tok, [text], model.device)
        lg = model(**enc).logits[0, -1].float()
        res[text] = [tok.decode([i]) for i in lg.topk(3).indices.tolist()]
    print("LM sanity (top-3 next tokens):", res, flush=True)
    ok = "Paris" in "".join(res["The capital of France is"]) and "Colorado" in "".join(res["Denver is the capital of the U.S. state of"])
    assert ok, "model does not predict sensible continuations on raw text - check BOS/tokenization"
    return res


def find_mlps(model):
    """All text-decoder gated MLP blocks, in layer order."""
    mlps = []
    for name, m in model.named_modules():
        if all(hasattr(m, a) for a in ("gate_proj", "up_proj", "down_proj")):
            if any(s in name for s in ("vision", "audio", "visual")):
                continue
            mlps.append((name, m))
    return mlps


def base_model(model):
    """The decoder without the LM head (avoids materialising [B,T,vocab] logits)."""
    return getattr(model, "model", model)


class Collector:
    """Calls fn(layer_index, product[B,T,N], gate_preact[B,T,N]) for every MLP block."""

    def __init__(self, mlps):
        self.mlps = mlps
        self.fn = None
        self._gate = None
        self.handles = []
        for i, (_, m) in enumerate(mlps):
            self.handles.append(m.gate_proj.register_forward_hook(self._gate_hook))
            self.handles.append(m.down_proj.register_forward_pre_hook(partial(self._down_hook, i)))

    def _gate_hook(self, mod, inp, out):
        self._gate = out

    def _down_hook(self, i, mod, inp):
        if self.fn is not None:
            self.fn(i, inp[0], self._gate)
        self._gate = None

    def remove(self):
        for h in self.handles:
            h.remove()


def sanity_check(model, tok, mlps, collector):
    """Verify that the hooked product really is act(gate) * up for a few layers."""
    L = len(mlps)
    layers = sorted({0, L // 2, L - 1})
    ups, diffs = {}, {}
    hs = [mlps[i][1].up_proj.register_forward_hook(lambda m, a, o, i=i: ups.__setitem__(i, o)) for i in layers]

    def fn(i, prod, gate):
        if i in layers:
            m = mlps[i][1]
            act = getattr(m, "act_fn", None) or getattr(m, "activation_fn", None)
            ref = act(gate.float()) * ups[i].float()
            diffs[i] = ((prod.float() - ref).abs().max() / (ref.abs().max() + 1e-6)).item()

    collector.fn = fn
    enc = encode(tok, ["The capital of Colorado is Denver, a city near the Rocky Mountains."], model.device)
    base_model(model)(**enc)
    collector.fn = None
    for h in hs:
        h.remove()
    print("sanity: relative |product - act(gate)*up| per layer:", {k: round(v, 5) for k, v in diffs.items()}, flush=True)
    return diffs


def trace_tokens(model, tok, collector, texts, refs, device=None):
    """Per-token (product, gate) for a few reference neurons. refs = [(layer, neuron), ...]."""
    by_layer = {}
    for l, n in refs:
        by_layer.setdefault(l, []).append(n)
    out = []
    for text in texts:
        enc = encode(tok, [text], model.device)
        got = {}

        def fn(i, prod, gate):
            if i in by_layer:
                idx = torch.tensor(by_layer[i], device=prod.device)
                got[i] = (prod[0][:, idx].float().cpu(), gate[0][:, idx].float().cpu())

        collector.fn = fn
        base_model(model)(**enc)
        collector.fn = None
        toks = [tok.decode([t]) for t in enc["input_ids"][0].tolist()]
        rec = {"text": text, "tokens": toks, "acts": {}}
        for l, ns in by_layer.items():
            for j, n in enumerate(ns):
                rec["acts"][f"{l}:{n}"] = {
                    "prod": [round(v, 2) for v in got[l][0][:, j].tolist()],
                    "gate": [round(v, 2) for v in got[l][1][:, j].tolist()],
                }
        out.append(rec)
    return out
