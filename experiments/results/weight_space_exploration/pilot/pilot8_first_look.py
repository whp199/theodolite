#!/usr/bin/env python3
"""Eighth pilot (14 November 2026), read-only: a first look, not registered, on chat-formatted items.

 1  The item gate's raw material: the base model's mean log-probability of the scored span for (a) its own greedy
    answers to six questions, (b) passages written by a person and given as the answer, (c) the same passages as raw
    text. A bar between "the model's own kind of input" and "not" has to sit somewhere in these.
 2  How large a set of silenced columns must be before it is heard above the null rows, at four layers: sets of
    1, 10, 64, 300 and 1,000 neurons, chosen (i) as the hardest-firing over the item's answer and (ii) at random,
    silenced on the input side of down_proj (the form that equalled the folded weight in pilot 6), against null
    rows (the layer's feed-forward output scaled by 1 + e). Read: KL to the untouched row of the same forward over
    the answer's positions (mean, median, positions above a nat).
 3  Whether the two batch arithmetics agree on such items: the hardest-firing sets read again in a batch of 12.

Run: .venv/bin/python -I <this file>   (server stopped)
"""
from __future__ import annotations

import json
import os
import statistics
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

OUT = Path(__file__).with_name("pilot8_first_look.json")
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
D_FF = ao.ffn_width(lm, 0)
SOFTCAP = ao.softcap_of(hf)

QUESTIONS = [
    "In two or three sentences, why does the moon show phases?",
    "Explain briefly why a stock should not be salted while it simmers.",
    "What is the difference between a virus and a bacterium? Answer in a short paragraph.",
    "Give three practical tips for sharpening a kitchen knife.",
    "Summarise in a few sentences how a bill becomes law in a parliamentary system.",
    "Why do leaves change colour in autumn? Keep it short.",
]
PASSAGES = {
    "recipe": (
        "Explain how to make a good stock, step by step.",
        "To make the stock, put the bones in a large pot with two onions, a carrot, a stick of celery and a few "
        "peppercorns, cover with cold water and bring slowly to the boil. Skim off the grey foam as it rises, then turn "
        "the heat down until the surface barely trembles and leave it for three hours. Strain it through a fine sieve, "
        "let it cool, and lift off the fat once it has set."),
    "survey": (
        "Write a short passage about a survey party at work.",
        "The survey party left the valley before dawn, carrying the theodolite in its padded case and the chain in two "
        "canvas bags. By the time the sun cleared the ridge they had set the first station on a flat stone above the "
        "river, and the youngest of them was sent down to hold the staff at the ford."),
}


def chat_ids(q):
    out = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(t) for t in out]


def batch(ids, B):
    return torch.tensor([ids] * B, dtype=torch.long, device=dev)


@torch.no_grad()
def whole(x):
    return lm(input_ids=x, use_cache=False).last_hidden_state


def klpos(lp, lq):
    return (lp.exp() * (lp - lq)).sum(-1)


def save():
    OUT.write_text(json.dumps(res, indent=1))


@torch.no_grad()
def span_lsm(h_row, start):
    """Log-probabilities at the positions that predict the scored span: positions start-1 .. T-2."""
    return F.log_softmax(cc.logits_fp32(h_row[start - 1:-1], W_U), dim=-1)


try:
    # ---- the items: each question with the model's own greedy answer (cut at the end of the turn)
    items = []
    for q in QUESTIONS:
        p = chat_ids(q)
        written = []
        for st in ao.write_rows(lm, W_U, [p], None, max_new=72, stop_ids=[], bos_id=tok.bos_token_id, softcap=SOFTCAP):
            written.append(st.token_id)
        cut = len(written)
        for i, t in enumerate(written):
            piece = tok.decode([t])
            if "<turn|>" in piece or "<end_of_turn>" in piece or t == tok.eos_token_id:
                cut = i
                break
        ans = written[:cut]
        items.append({"name": q[:40], "ids": p + ans, "start": len(p), "kind": "own answer", "answer": tok.decode(ans)})
    for name, (q, text) in PASSAGES.items():
        p = chat_ids(q)
        a = [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]
        items.append({"name": name + " as an answer", "ids": p + a, "start": len(p), "kind": "a person's passage as the answer"})
        raw = [int(t) for t in tok(text, add_special_tokens=True)["input_ids"]]
        items.append({"name": name + " raw", "ids": raw, "start": 1, "kind": "raw text"})

    # ---- 1. the gate's raw material, at two batch sizes
    gate = []
    for it in items:
        row = {"name": it["name"], "kind": it["kind"], "pieces": len(it["ids"]), "scored": len(it["ids"]) - it["start"]}
        tgt = torch.tensor(it["ids"][it["start"]:], device=dev)
        for B in (2, 24):
            lp = span_lsm(whole(batch(it["ids"], B))[0], it["start"])
            own = lp.gather(1, tgt[:, None]).squeeze(1)
            row[f"mean_logprob_B{B}"] = round(float(own.mean()), 4)
            row[f"median_logprob_B{B}"] = round(float(own.median()), 4)
            row[f"worst_piece_B{B}"] = round(float(own.min()), 3)
        gate.append(row)
        print(row, flush=True)
    res["item_gate"] = gate
    save()

    # ---- 2 and 3. the ladder of set sizes, at four layers, on the six own-answer items
    SIZES = (1, 10, 64, 300, 1000)
    NULL_E = (1e-3, -1e-3, 1e-2, -1e-2)
    LAYERS = (19, 35, 44, 54)  # code count; pages 20, 36, 45, 55
    g = torch.Generator(device="cpu").manual_seed(20261114)
    ladder = {}
    for layer in LAYERS:
        down = lm.layers[layer].mlp.down_proj
        per_item = []
        for it in items[:len(QUESTIONS)]:
            ids, start = it["ids"], it["start"]
            got = {}
            hk = down.register_forward_pre_hook(lambda m, a: got.__setitem__("a", a[0].detach()))
            with torch.no_grad():
                whole(batch(ids, 1))
            hk.remove()
            fire = got["a"][0, start:].float().abs().mean(0)  # mean |activation| over the answer's positions
            order = fire.argsort(descending=True)

            def run(sets, nulls):
                """sets: list of LongTensors of neurons to silence; rows = base, the null rows, then one row a set."""
                B = 1 + len(nulls) + len(sets)
                mask = torch.ones(B, D_FF, device=dev, dtype=torch.bfloat16)
                for i, s in enumerate(sets):
                    mask[1 + len(nulls) + i, s.to(dev)] = 0
                scale = torch.ones(B, device=dev)
                for i, e in enumerate(nulls):
                    scale[1 + i] = 1.0 + e

                def pre(_m, args):
                    return (args[0] * mask[:, None, :],)

                def post(_m, args, out):
                    return out * scale.view(-1, 1, 1).to(out.dtype)

                h1, h2 = down.register_forward_pre_hook(pre), down.register_forward_hook(post)
                try:
                    h = whole(batch(ids, B))
                finally:
                    h1.remove()
                    h2.remove()
                base = span_lsm(h[0], start)
                out = []
                for r in range(1, B):
                    k = klpos(base, span_lsm(h[r], start))
                    out.append({"mean": float(k.mean()), "median": float(k.median()), "above_1": int((k > 1).sum())})
                return out[:len(nulls)], out[len(nulls):]

            top_sets = [order[:k].cpu() for k in SIZES]
            rand_sets = [torch.randperm(D_FF, generator=g)[:k] for k in SIZES for _ in range(3)]
            nulls_big, eff_big = run(top_sets + rand_sets, NULL_E)                      # 1 + 4 + 20 = 25 rows
            nulls_small, eff_small = run(top_sets + rand_sets[::3], NULL_E[:2])         # 1 + 2 + 10 = 13 rows
            nulls_small2, eff_small2 = run(top_sets, NULL_E[:2])                        # 1 + 2 + 5 = 8 rows
            rec = {"item": it["name"], "scored": len(ids) - start,
                   "floor_big": max(n["mean"] for n in nulls_big), "floor_small": max(n["mean"] for n in nulls_small2),
                   "top": {}, "random": {}}
            for i, k in enumerate(SIZES):
                rec["top"][str(k)] = {"big": eff_big[i], "rows13": eff_small[i], "rows8": eff_small2[i],
                                      "share_of_firing": round(float(fire[order[:k]].sum() / fire.sum()), 4)}
                rr = eff_big[len(SIZES) + 3 * i: len(SIZES) + 3 * i + 3]
                rec["random"][str(k)] = {"big_means": [x["mean"] for x in rr], "big_medians": [x["median"] for x in rr]}
            per_item.append(rec)
        # the layer's summary: medians over items
        summ = {"floor_big_median": statistics.median(r["floor_big"] for r in per_item),
                "floor_small_median": statistics.median(r["floor_small"] for r in per_item), "by_size": {}}
        for k in SIZES:
            tb = [r["top"][str(k)]["big"]["mean"] for r in per_item]
            t8 = [r["top"][str(k)]["rows8"]["mean"] for r in per_item]
            tm = [r["top"][str(k)]["big"]["median"] for r in per_item]
            rb = [statistics.median(r["random"][str(k)]["big_means"]) for r in per_item]
            ratio = [max(a, b) / max(min(a, b), 1e-9) for a, b in zip(tb, t8)]
            over = [a / max(r["floor_big"], 1e-9) for a, r in zip(tb, per_item)]
            rover = [a / max(r["floor_big"], 1e-9) for a, r in zip(rb, per_item)]
            summ["by_size"][str(k)] = {
                "hardest_firing_mean_KL_median_over_items": statistics.median(tb),
                "hardest_firing_median_position_KL": statistics.median(tm),
                "hardest_firing_over_floor_median": statistics.median(over),
                "hardest_firing_items_over_5x_floor": sum(1 for x in over if x > 5),
                "random_mean_KL_median_over_items": statistics.median(rb),
                "random_over_floor_median": statistics.median(rover),
                "random_items_over_5x_floor": sum(1 for x in rover if x > 5),
                "two_batch_sizes_largest_ratio": max(ratio),
                "two_batch_sizes_median_ratio": statistics.median(ratio),
                "share_of_firing_median": statistics.median(r["top"][str(k)]["share_of_firing"] for r in per_item),
            }
        ladder[f"layer_{layer}"] = {"summary": summ, "items": per_item}
        print(f"layer {layer} (page {layer + 1}):", json.dumps(summ, indent=1), flush=True)
        res["ladder"] = ladder
        save()
        torch.cuda.empty_cache()
except Exception as ex:  # noqa: BLE001
    res["errors"].append(f"{type(ex).__name__}: {ex}"[:800])
    save()
    raise

res["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
res["unpacker_after"] = {k: v for k, v in w4a16_fast.describe().items() if k in ("mode", "fell_back")}
save()
print("errors:", res["errors"], flush=True)
