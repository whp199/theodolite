#!/usr/bin/env python3
"""Fifth pilot (14 November 2026), read-only. Pilot 4: on the recipe text, at layers 5 and 20, a single-neuron
silencing read 0.6 to 0.7 nats a position in one arithmetic form and about 0.005 in another, deterministically.
Question: is that the model's own numerics on this text (a rounding-sized nudge tipping it into another state at some
position), and how often does a NULL nudge do it? This sets what a floor has to be.

 1  The untouched model at batch sizes 1, 2, 3, 4, 8, 16, 32: per-position KL to the batch-1 row.
 2  31 null rows at each of layers 5, 20, 44: the layer's feed-forward output multiplied by (1 + e), e = +-1e-3 ..
    (the post-feed-forward norm undoes a uniform scale but for rounding), each row its own e: mean, median and max
    per-position KL to the untouched row; how many rows tip.
 3  For a row that tips: where along the text, how many positions, which pieces.

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

OUT = Path(__file__).with_name("pilot5_knife_edge.json")
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
CHAT_Q = "Explain briefly why a stock should not be salted while it simmers."
CHAT_A = ("You should not salt a stock while it simmers because of evaporation. As a stock simmers, water evaporates, "
          "which concentrates the flavors and the salt. If you salt at the beginning, the liquid will reduce, and the "
          "finished stock, and any sauce made from it, can end up far too salty to use.")


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


def stats(k):
    return {"mean": float(k.mean()), "median": float(k.median()), "max": float(k.max()),
            "positions_above_1_nat": int((k > 1).sum()), "positions_above_0.1": int((k > 0.1).sum())}


try:
    for tname, ids in (("recipe_plain", plain(RECIPE)), ("prose_plain", plain(PROSE)), ("recipe_as_chat", chat(CHAT_Q, CHAT_A))):
        T = len(ids)
        rec = {"pieces": T}
        # 1. the untouched model across batch sizes
        with torch.no_grad():
            ref = lsm_all(whole(batch(ids, 1))[0])
        shapes = {}
        for B in (2, 3, 4, 8, 16, 32):
            with torch.no_grad():
                lp = lsm_all(whole(batch(ids, B))[0])
            shapes[f"B{B}"] = stats(klpos(ref, lp))
        rec["untouched_row_against_batch_1"] = shapes

        # 2. null rows: each row's feed-forward output at one layer times (1 + e_r)
        B = 32
        e = torch.zeros(B, device=dev)
        vals = [1e-3, -1e-3, 2e-3, -2e-3, 5e-3, -5e-3, 1e-2, -1e-2, 2e-2, -2e-2, 5e-2, -5e-2, 0.1, -0.1, 0.5, 1.0, 2.0]
        for r in range(1, B):
            e[r] = vals[(r - 1) % len(vals)] * (1.0 + 0.01 * ((r - 1) // len(vals)))
        for layer in (5, 20, 44):
            down = lm.layers[layer].mlp.down_proj

            def hook(_m, args, out):
                return out * (1.0 + e).view(-1, 1, 1).to(out.dtype)

            hk = down.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    h = whole(batch(ids, B))
            finally:
                hk.remove()
            base = lsm_all(h[0])
            rows = []
            tipped = None
            for r in range(1, B):
                k = klpos(base, lsm_all(h[r]))
                s = stats(k)
                s["e"] = round(float(e[r]), 5)
                rows.append(s)
                if s["mean"] > 0.1 and tipped is None:
                    tipped = (r, k)
            means = torch.tensor([s["mean"] for s in rows])
            meds = torch.tensor([s["median"] for s in rows])
            out_l = {
                "null_rows": len(rows),
                "mean_KL_of_null_rows": {"median": float(means.median()), "p90": float(means.quantile(0.9)), "max": float(means.max())},
                "median_position_KL_of_null_rows": {"median": float(meds.median()), "max": float(meds.max())},
                "rows_tipped_mean_above_0.1": int((means > 0.1).sum()),
                "rows_with_any_position_above_1_nat": int(sum(1 for s in rows if s["positions_above_1_nat"] > 0)),
                "the_tipped_rows": [{"e": s["e"], "mean": round(s["mean"], 4), "median": round(s["median"], 6),
                                     "max": round(s["max"], 2), "positions_above_1": s["positions_above_1_nat"]}
                                    for s in rows if s["mean"] > 0.1][:8],
            }
            if tipped is not None:
                r, k = tipped
                top = k.topk(8)
                first = int((k > 0.5).nonzero()[0]) if bool((k > 0.5).any()) else None
                out_l["one_tipped_row"] = {
                    "e": round(float(e[r]), 5),
                    "first_position_above_0.5": first,
                    "top_positions": [{"pos": int(p), "KL": round(float(v), 2),
                                       "after": tok.decode(ids[max(0, int(p) - 3): int(p) + 1]),
                                       "next_piece": tok.decode([ids[int(p) + 1]])} for v, p in zip(top.values, top.indices)],
                    "share_of_total_KL_in_top_8_positions": round(float(top.values.sum() / k.sum()), 3),
                }
            rec[f"null_scale_rows_layer_{layer}"] = out_l
            del h, base
            torch.cuda.empty_cache()
        res[tname] = rec
        print(tname, json.dumps(rec, indent=1), flush=True)
        save()
except Exception as ex:  # noqa: BLE001
    res["errors"].append(f"{type(ex).__name__}: {ex}"[:800])
    save()
    raise

save()
print("errors:", res["errors"], flush=True)
