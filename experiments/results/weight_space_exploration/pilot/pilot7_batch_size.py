#!/usr/bin/env python3
"""Seventh pilot (14 November 2026), read-only. Pilot 6: one neuron silenced (input side, equal to the folded form)
read 0.009 nats a position at a batch of 2 and 0.73 at a batch of 32 on the recipe text, the reading parting from
position 57 on. Here: the same silencing at ten batch sizes; what the model predicts where it parts; and whether the
30 hardest-firing neurons of layer 20 do the same on a chat-formatted text (the instruct model's own kind of input).

Run: .venv/bin/python -I <this file>   (server stopped)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from backend.engine_runtime import TheodoliteRuntime  # noqa: E402,F401

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import backend.atlas_ops as ao  # noqa: E402
import backend.causal_check as cc  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend.runtime_factory import make_runtime  # noqa: E402

OUT = Path(__file__).with_name("pilot7_batch_size.json")
res: dict = {"when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "errors": []}
dev = torch.device("cuda:0")

rt = make_runtime(load_jlens=False).load()
if getattr(rt, "jpp_lenses", None):
    rt.jpp_lenses.clear()
    torch.cuda.empty_cache()
w4a16_fast.install(rt, "compiled", log=lambda s: print(s, flush=True))
hf = rt.hf_model
lm = cc.language_model(hf)
tok = rt.tokenizer
W_U = hf.lm_head.weight if getattr(hf, "lm_head", None) is not None else lm.embed_tokens.weight

RECIPE = (
    "To make the stock, put the bones in a large pot with two onions, a carrot, a stick of celery and a few "
    "peppercorns, cover with cold water and bring slowly to the boil. Skim off the grey foam as it rises, then turn "
    "the heat down until the surface barely trembles and leave it for three hours. Strain it through a fine sieve, "
    "let it cool, and lift off the fat once it has set. The stock will keep for four days in a cold larder, or it "
    "can be boiled down by half and stored in small jars. Never salt it at this stage: the sauces made from it will "
    "be reduced again, and what tastes right in the pot will be far too strong on the plate."
)
PROSE = (
    "The survey party left the valley before dawn, carrying the theodolite in its padded case and the chain in two "
    "canvas bags. By the time the sun cleared the ridge they had set the first station on a flat stone above the "
    "river, and the youngest of them was sent down to hold the staff at the ford. The readings were taken twice, as "
    "the rule required, once with the telescope direct and once reversed, and the two were entered side by side in "
    "the field book so that any disagreement would show at once. At noon they ate bread and cheese in the shade of a "
    "wall and argued about whether the old map could be trusted."
)


def plain(text):
    return [int(t) for t in tok(text, add_special_tokens=True)["input_ids"]]


def chat(q, a):
    out = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(t) for t in out] + [int(t) for t in tok(a, add_special_tokens=False)["input_ids"]]


def batch(ids, B):
    return torch.tensor([ids] * B, dtype=torch.long, device=dev)


@torch.no_grad()
def whole(x):
    return lm(input_ids=x, use_cache=False).last_hidden_state


def klpos(lp, lq):
    return (lp.exp() * (lp - lq)).sum(-1)


@torch.no_grad()
def lsm_all(h_row):
    return F.log_softmax(cc.logits_fp32(h_row[:-1], W_U), dim=-1)


def save():
    OUT.write_text(json.dumps(res, indent=1))


def silenced(ids, B, layer, per_row_neuron):
    """Row r silences per_row_neuron[r] (None leaves the row alone) on the input side of layer's down_proj."""
    down = lm.layers[layer].mlp.down_proj
    rows = torch.tensor([r for r, n in enumerate(per_row_neuron) if n is not None], device=dev)
    neur = torch.tensor([n for n in per_row_neuron if n is not None], device=dev)

    def pre(_m, args):
        x = args[0].clone()
        x[rows, :, neur] = 0
        return (x,)

    hk = down.register_forward_pre_hook(pre)
    try:
        with torch.no_grad():
            return whole(batch(ids, B))
    finally:
        hk.remove()


try:
    ids = plain(RECIPE)
    # 1. the same silencing at ten batch sizes (row 0 untouched, row 1 silenced, the rest untouched)
    sizes = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
    table = {}
    for layer, n in ((20, 4294), (20, 9618), (5, 13133), (20, 2642)):
        row = {}
        for B in sizes:
            h = silenced(ids, B, layer, [None, n] + [None] * (B - 2))
            k = klpos(lsm_all(h[0]), lsm_all(h[1]))
            first = int((k > 0.5).nonzero()[0]) if bool((k > 0.5).any()) else None
            row[f"B{B}"] = {"mean": round(float(k.mean()), 4), "median": round(float(k.median()), 5), "first_above_0.5": first}
            del h
        table[f"L{layer}_n{n}"] = row
        print(f"L{layer} n{n}:", " ".join(f"B{B}={row[f'B{B}']['mean']}" for B in sizes), flush=True)
    res["one_silencing_across_batch_sizes_recipe"] = table
    save()

    # 2. what the model predicts where the tipped row parts (batch 32, L20 n4294)
    h = silenced(ids, 32, 20, [None, 4294] + [None] * 30)
    b, e = lsm_all(h[0]), lsm_all(h[1])
    k = klpos(b, e)
    look = []
    for p in range(54, 64):
        tb, te = b[p].topk(3), e[p].topk(3)
        look.append({"pos": p, "after": tok.decode(ids[max(0, p - 2): p + 1]), "true_next": tok.decode([ids[p + 1]]),
                     "KL": round(float(k[p]), 3),
                     "base_top3": [(tok.decode([int(i)]), round(float(v.exp()), 3)) for v, i in zip(tb.values, tb.indices)],
                     "edit_top3": [(tok.decode([int(i)]), round(float(v.exp()), 3)) for v, i in zip(te.values, te.indices)],
                     "true_next_logprob_base": round(float(b[p, ids[p + 1]]), 3),
                     "true_next_logprob_edit": round(float(e[p, ids[p + 1]]), 3)})
    res["where_it_parts"] = look
    res["tipped_row_mean_logprob_of_text"] = {
        "base": round(float(b.gather(1, torch.tensor(ids[1:], device=dev)[:, None]).mean()), 4),
        "edit": round(float(e.gather(1, torch.tensor(ids[1:], device=dev)[:, None]).mean()), 4)}
    del h
    for x in look:
        print(x, flush=True)
    print(res["tipped_row_mean_logprob_of_text"], flush=True)
    save()

    # 3. the 30 hardest-firing neurons of layer 20 silenced, one a row, at batch 32 and at batch 2, on three texts
    texts = {
        "recipe_plain": plain(RECIPE),
        "prose_plain": plain(PROSE),
        "recipe_as_chat": chat("Explain how to make a good stock, step by step.", RECIPE),
        "prose_as_chat": chat("Write a short passage about a survey party at work.", PROSE),
    }
    sweep = {}
    for tname, tids in texts.items():
        for layer in (5, 20):
            down = lm.layers[layer].mlp.down_proj
            got = {}
            hk = down.register_forward_pre_hook(lambda m, a: got.__setitem__("a", a[0].detach()))
            with torch.no_grad():
                whole(batch(tids, 1))
            hk.remove()
            cand = [int(i) for i in got["a"][0, 1:].float().abs().amax(0).topk(30).indices]
            h = silenced(tids, 32, layer, [None] + cand + [None])
            b32 = lsm_all(h[0])
            e32 = torch.tensor([float(klpos(b32, lsm_all(h[r])).mean()) for r in range(1, 31)])
            m32 = torch.tensor([float(klpos(b32, lsm_all(h[r])).median()) for r in range(1, 31)])
            del h
            e2 = []
            for n in cand:
                h = silenced(tids, 2, layer, [None, n])
                e2.append(float(klpos(lsm_all(h[0]), lsm_all(h[1])).mean()))
                del h
            e2 = torch.tensor(e2)
            ratio = torch.maximum(e32, e2) / torch.minimum(e32, e2).clamp(min=1e-9)
            sweep[f"{tname}_L{layer}"] = {
                "pieces": len(tids),
                "mean_KL_batch32": {"median": round(float(e32.median()), 5), "max": round(float(e32.max()), 4)},
                "mean_KL_batch2": {"median": round(float(e2.median()), 5), "max": round(float(e2.max()), 4)},
                "neurons_above_0.1_at_batch32": int((e32 > 0.1).sum()), "neurons_above_0.1_at_batch2": int((e2 > 0.1).sum()),
                "neurons_whose_two_readings_differ_over_5x": int((ratio > 5).sum()),
                "neurons_whose_two_readings_differ_over_2x": int((ratio > 2).sum()),
                "corr_batch32_vs_batch2": round(float(torch.corrcoef(torch.stack([e32, e2]))[0, 1]), 4),
                "median_position_KL_batch32_max": round(float(m32.max()), 5),
            }
            print(tname, layer, json.dumps(sweep[f"{tname}_L{layer}"]), flush=True)
            res["thirty_silencings_two_batch_sizes"] = sweep
            save()
            torch.cuda.empty_cache()
except Exception as ex:  # noqa: BLE001
    res["errors"].append(f"{type(ex).__name__}: {ex}"[:800])
    save()
    raise

save()
print("errors:", res["errors"], flush=True)
