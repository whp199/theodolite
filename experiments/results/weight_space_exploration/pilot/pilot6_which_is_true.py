#!/usr/bin/env python3
"""Sixth pilot (14 November 2026), read-only. Pilot 4 left a contradiction at layers 5 and 20 on the recipe text:
silencing one neuron read about 0.005 nats folded into the weight (batch of 2) and 0.5 to 0.75 as a per-row change
(batch of 32), while null nudges never tipped a reading (pilot 5). Which is true, and what does it depend on: the
form, the batch size, the row's place in the batch, or the company the row keeps?

For each neuron: its firing profile; the folded silencing at batch 2 and 32; the input-side per-row silencing and the
output-side delta at batch 2 and at batch 32 with the edit in four different rows and every other row untouched.

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

OUT = Path(__file__).with_name("pilot6_which_is_true.json")
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
D = ao.hidden_size(lm)

RECIPE = (
    "To make the stock, put the bones in a large pot with two onions, a carrot, a stick of celery and a few "
    "peppercorns, cover with cold water and bring slowly to the boil. Skim off the grey foam as it rises, then turn "
    "the heat down until the surface barely trembles and leave it for three hours. Strain it through a fine sieve, "
    "let it cool, and lift off the fat once it has set. The stock will keep for four days in a cold larder, or it "
    "can be boiled down by half and stored in small jars. Never salt it at this stage: the sauces made from it will "
    "be reduced again, and what tastes right in the pot will be far too strong on the plate."
)
ids = [int(t) for t in tok(RECIPE, add_special_tokens=True)["input_ids"]]
T = len(ids)
CASES = [(20, 4294), (20, 9618), (20, 5316), (20, 6540), (5, 13133), (5, 14825), (5, 16222)]


def batch(B):
    return torch.tensor([ids] * B, dtype=torch.long, device=dev)


@torch.no_grad()
def whole(x):
    return lm(input_ids=x, use_cache=False).last_hidden_state


def klpos(lp, lq):
    return (lp.exp() * (lp - lq)).sum(-1)


@torch.no_grad()
def lsm_all(h_row):
    return F.log_softmax(cc.logits_fp32(h_row[:-1], W_U), dim=-1)


def st(k):
    first = int((k > 0.5).nonzero()[0]) if bool((k > 0.5).any()) else None
    return {"mean": round(float(k.mean()), 5), "median": round(float(k.median()), 6), "max": round(float(k.max()), 3),
            "above_1_nat": int((k > 1).sum()), "first_above_0.5": first}


def save():
    OUT.write_text(json.dumps(res, indent=1))


try:
    base = {}
    for B in (2, 32):
        with torch.no_grad():
            h = whole(batch(B))
        base[B] = lsm_all(h[0])
        res[f"untouched_rows_identical_B{B}"] = bool(torch.equal(h[0], h[B - 1]))
        del h
    for layer, n in CASES:
        down = lm.layers[layer].mlp.down_proj
        got = {}
        hk = down.register_forward_pre_hook(lambda m, a: got.__setitem__("a", a[0].detach()))
        with torch.no_grad():
            whole(batch(1))
        hk.remove()
        act = got["a"][0, :, n].float()
        peak = int(act.abs().argmax())
        W = ao.dense_weight(down)
        col = W[:, n].clone()
        del W
        torch.cuda.empty_cache()
        rec = {"layer": layer, "neuron": n,
               "firing": {"at_start_mark": round(float(act[0]), 3), "peak": round(float(act[peak]), 2), "peak_position": peak,
                          "peak_after": tok.decode(ids[max(0, peak - 4): peak + 1]),
                          "positions_above_half_peak": int((act.abs() > 0.5 * act.abs().max()).sum()),
                          "median_abs": round(float(act.abs().median()), 3)}}

        # folded: every row edited alike, against an untouched forward of the same shape
        for B in (2, 32):
            try:
                down._memit_col_overlays[n] = torch.zeros(D, device=dev, dtype=torch.bfloat16)
                with torch.no_grad():
                    h = whole(batch(B))
            finally:
                down._memit_col_overlays.clear()
            rec[f"folded_B{B}"] = st(klpos(base[B], lsm_all(h[0])))
            rec[f"folded_B{B}_rows_identical"] = bool(torch.equal(h[0], h[B - 1]))
            del h

        def per_row(B, slots, form):
            mask = torch.zeros(B, dtype=torch.bool, device=dev)
            mask[slots] = True
            if form == "input":
                def pre(_m, args):
                    x = args[0].clone()
                    x[mask, :, n] = 0
                    return (x,)
                hk2 = down.register_forward_pre_hook(pre)
            else:
                def post(_m, args, out):
                    a = args[0][:, :, n:n + 1]
                    add = a * (-col)[None, None, :].to(out.dtype)
                    add = add * mask.view(-1, 1, 1).to(out.dtype)
                    return out + add
                hk2 = down.register_forward_hook(post)
            try:
                with torch.no_grad():
                    h = whole(batch(B))
            finally:
                hk2.remove()
            b0 = lsm_all(h[0])
            out = {"base_row_vs_unhooked_base_mean_KL": round(float(klpos(base[B], b0).mean()), 6)}
            for s in slots:
                out[f"row_{s}"] = st(klpos(b0, lsm_all(h[s])))
            # an untouched row far from the edited ones, against row 0: should be nought
            free = [r for r in range(B) if r not in slots and r != 0]
            if free:
                out["an_untouched_row_vs_row0_mean_KL"] = round(float(klpos(b0, lsm_all(h[free[-1]])).mean()), 6)
            del h
            return out

        for form in ("input", "output"):
            rec[f"{form}_side_B2"] = per_row(2, [1], form)
            rec[f"{form}_side_B32_four_rows"] = per_row(32, [1, 9, 20, 31], form)
            rec[f"{form}_side_B32_all_but_row0"] = per_row(32, list(range(1, 32)), form)
        res[f"L{layer}_n{n}"] = rec
        print(json.dumps(rec, indent=1), flush=True)
        save()
        torch.cuda.empty_cache()
except Exception as ex:  # noqa: BLE001
    res["errors"].append(f"{type(ex).__name__}: {ex}"[:800])
    save()
    raise

save()
print("errors:", res["errors"], flush=True)
