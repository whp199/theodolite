#!/usr/bin/env python3
"""Fourth pilot (14 November 2026), read-only. Pilot 3 found, at layers 5 and 20 on one text, single-neuron
silencings whose effect was 0.2 to 0.7 nats a position in one forward and about nought in another. The guess: a
neuron that dominates the layer's output at the start mark cannot be silenced by an output-side delta in bf16
(the delta cancels a huge term that was already rounded), while silencing it on the INPUT side, or folding the edit
into the weight, has no such cancellation. This compares the three forms on the same neurons.

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

OUT = Path(__file__).with_name("pilot4_cancellation.json")
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

TEXTS = {
    "recipe": (
        "To make the stock, put the bones in a large pot with two onions, a carrot, a stick of celery and a few "
        "peppercorns, cover with cold water and bring slowly to the boil. Skim off the grey foam as it rises, then turn "
        "the heat down until the surface barely trembles and leave it for three hours. Strain it through a fine sieve, "
        "let it cool, and lift off the fat once it has set. The stock will keep for four days in a cold larder, or it "
        "can be boiled down by half and stored in small jars. Never salt it at this stage: the sauces made from it will "
        "be reduced again, and what tastes right in the pot will be far too strong on the plate."
    ),
    "prose": (
        "The survey party left the valley before dawn, carrying the theodolite in its padded case and the chain in two "
        "canvas bags. By the time the sun cleared the ridge they had set the first station on a flat stone above the "
        "river, and the youngest of them was sent down to hold the staff at the ford. The readings were taken twice, as "
        "the rule required, once with the telescope direct and once reversed, and the two were entered side by side in "
        "the field book so that any disagreement would show at once."
    ),
}


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


try:
    for tname, text in TEXTS.items():
        ids = [int(t) for t in tok(text, add_special_tokens=True)["input_ids"]]
        T = len(ids)
        for layer in (5, 20, 44):
            down = lm.layers[layer].mlp.down_proj
            got = {}
            h1 = down.register_forward_pre_hook(lambda m, a: got.__setitem__("a", a[0].detach()))
            h2 = down.register_forward_hook(lambda m, a, o: got.__setitem__("o", o.detach()))
            with torch.no_grad():
                whole(batch(ids, 1))
            h1.remove()
            h2.remove()
            acts = got["a"][0].float()  # [T, d_ff]
            outn = got["o"][0].float().norm(dim=-1)  # [T]
            score = acts[1:].abs().amax(0)
            cand = [int(i) for i in score.topk(30).indices]
            W = ao.dense_weight(down)
            cols = W[:, cand].T.contiguous().clone()  # [30, D] bf16
            del W
            torch.cuda.empty_cache()
            coln = cols.float().norm(dim=-1)
            a0 = acts[0, cand].abs()
            arest = acts[1:, cand].abs().amax(0)
            share0 = (a0 * coln) / outn[0]  # the neuron's own term at the start mark, over the layer's output there

            B = 32
            neuron = torch.tensor([cand[0]] + cand + [cand[0]], device=dev)
            delta = torch.zeros(B, D, device=dev, dtype=torch.bfloat16)
            delta[1:31] = (-cols.float()).to(torch.bfloat16)
            rows = torch.arange(B, device=dev)

            def out_delta(spare_start):
                def hook(_m, args, out):
                    x = args[0]
                    a = x.gather(2, neuron.view(-1, 1, 1).expand(x.shape[0], x.shape[1], 1))
                    add = a * delta[:, None, :].to(out.dtype)
                    if spare_start:
                        add[:, 0] = 0
                    return out + add
                return down.register_forward_hook(hook)

            def in_mul():
                def pre(_m, args):
                    x = args[0].clone()
                    x[rows[1:31], :, neuron[1:31]] = 0
                    return (x,)
                return down.register_forward_pre_hook(pre)

            def effects(install):
                hk = install()
                try:
                    with torch.no_grad():
                        h = whole(batch(ids, B))
                finally:
                    hk.remove()
                base = lsm_all(h[0])
                return torch.tensor([float(klpos(base, lsm_all(h[r])).mean()) for r in range(1, 31)])

            e_out = effects(lambda: out_delta(False))
            e_out_spared = effects(lambda: out_delta(True))
            e_in = effects(in_mul)
            e_out2 = effects(lambda: out_delta(False))  # the same again: is the output-side form even repeatable?

            # folded, for the five neurons whose two forms disagree most, each at batch 2 against a batch-2 base
            worst = (e_out - e_in).abs().topk(5).indices.tolist()
            with torch.no_grad():
                base2 = lsm_all(whole(batch(ids, 2))[0])
            folded = {}
            for j in worst:
                try:
                    down._memit_col_overlays[cand[j]] = torch.zeros(D, device=dev, dtype=torch.bfloat16)
                    with torch.no_grad():
                        hf2 = whole(batch(ids, 2))
                finally:
                    down._memit_col_overlays.clear()
                folded[j] = float(klpos(base2, lsm_all(hf2[0])).mean())

            def corr(a, b):
                return float(torch.corrcoef(torch.stack([a, b]))[0, 1])

            rec = {
                "pieces": T,
                "output_norm_at_start_mark_over_median_elsewhere": float(outn[0] / outn[1:].median()),
                "candidates_whose_act_at_start_mark_exceeds_max_elsewhere": int((a0 > arest).sum()),
                "largest_share_of_start_mark_output_by_one_candidate": float(share0.max()),
                "effects_median": {"output_delta": float(e_out.median()), "input_side": float(e_in.median()),
                                   "output_delta_start_spared": float(e_out_spared.median())},
                "effects_max": {"output_delta": float(e_out.max()), "input_side": float(e_in.max()),
                                "output_delta_start_spared": float(e_out_spared.max())},
                "corr_output_delta_vs_input_side": corr(e_out, e_in),
                "corr_output_delta_start_spared_vs_input_side": corr(e_out_spared, e_in),
                "corr_output_delta_run1_vs_run2": corr(e_out, e_out2),
                "output_delta_same_bits_run1_run2": bool(torch.equal(e_out, e_out2)),
                "max_abs_gap_output_delta_vs_input_side": float((e_out - e_in).abs().max()),
                "worst_five": [
                    {"neuron": cand[j], "act_at_start_mark": round(float(acts[0, cand[j]]), 2),
                     "max_abs_act_elsewhere": round(float(arest[j]), 2), "share_of_start_mark_output": round(float(share0[j]), 4),
                     "effect_output_delta": float(e_out[j]), "effect_output_delta_start_spared": float(e_out_spared[j]),
                     "effect_input_side": float(e_in[j]), "effect_folded_batch2": folded[j]}
                    for j in worst
                ],
            }
            res[f"{tname}_layer_{layer}"] = rec
            print(f"{tname} layer {layer}:", json.dumps(rec, indent=1), flush=True)
            save()
            del cols, acts
            torch.cuda.empty_cache()
except Exception as e:  # noqa: BLE001
    res["errors"].append(f"{type(e).__name__}: {e}"[:800])
    save()
    raise

res["unpacker_after"] = {k: v for k, v in w4a16_fast.describe().items() if k in ("mode", "fell_back")}
save()
print("errors:", res["errors"], flush=True)
