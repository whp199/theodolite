#!/usr/bin/env python3
"""Third pilot (14 November 2026), read-only. Engineering questions only; no decision rests on it.
 A  Writing under per-row output-side deltas (atlas_ops.write_rows with a cache, hooks registered round it): do two
    untouched rows and a zero-delta row write the same pieces with the same log-probabilities for 48 pieces; do two
    rows with the same edit; what does a step cost against an unhooked write of the same shape.
 C  A stream captured in a one-row forward and replayed at 32 rows from layer L, against a same-shape whole forward:
    do the paired differences (edit against base) agree, at layers 5, 20, 44, on two texts.
 E  The floor row: one row whose down_proj output is multiplied by 3 (the post-feed-forward norm undoes it but for
    rounding): its KL to the base row, beside the measured gap between delta and folded forms (0.00045 nats).

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

OUT = Path(__file__).with_name("pilot3_writing_and_capture.json")
res: dict = {"when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "errors": []}
dev = torch.device("cuda:0")


def save():
    OUT.write_text(json.dumps(res, indent=1))


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
SOFTCAP = ao.softcap_of(hf)

PROSE = (
    "The survey party left the valley before dawn, carrying the theodolite in its padded case and the chain in two "
    "canvas bags. By the time the sun cleared the ridge they had set the first station on a flat stone above the "
    "river, and the youngest of them was sent down to hold the staff at the ford. The readings were taken twice, as "
    "the rule required, once with the telescope direct and once reversed, and the two were entered side by side in "
    "the field book so that any disagreement would show at once. At noon they ate bread and cheese in the shade of a "
    "wall and argued about whether the old map could be trusted. It could not, as it turned out: the mill it showed "
    "at the bend of the river had burned forty years before, and the road it marked as the way to the pass had "
    "long since been washed out by the spring floods."
)
RECIPE = (
    "To make the stock, put the bones in a large pot with two onions, a carrot, a stick of celery and a few "
    "peppercorns, cover with cold water and bring slowly to the boil. Skim off the grey foam as it rises, then turn "
    "the heat down until the surface barely trembles and leave it for three hours. Strain it through a fine sieve, "
    "let it cool, and lift off the fat once it has set. The stock will keep for four days in a cold larder, or it "
    "can be boiled down by half and stored in small jars. Never salt it at this stage: the sauces made from it will "
    "be reduced again, and what tastes right in the pot will be far too strong on the plate."
)


def plain(text):
    return [int(t) for t in tok(text, add_special_tokens=True)["input_ids"]]


def chat(text):
    out = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(t) for t in out]


def sync():
    torch.cuda.synchronize()


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


class RowNeuron:
    """Row r adds x[r, :, neuron[r]] * delta[r] to the linear's output; rows in `times` have their output multiplied."""

    def __init__(self, module, neuron, delta, times=None):
        self.m, self.neuron, self.delta, self.times = module, neuron, delta, times

    def __enter__(self):
        def hook(_m, args, out):
            x = args[0]
            if x.shape[0] != self.neuron.shape[0]:
                raise RuntimeError(f"hook made for {self.neuron.shape[0]} rows met {x.shape[0]}")
            a = x.gather(2, self.neuron.view(-1, 1, 1).expand(x.shape[0], x.shape[1], 1))
            y = out + a * self.delta[:, None, :].to(out.dtype)
            if self.times is not None:
                y = y * self.times.view(-1, 1, 1).to(out.dtype)
            return y
        self.h = self.m.register_forward_hook(hook)
        return self

    def __exit__(self, *e):
        self.h.remove()


def top_columns(layer, ids, k):
    down = lm.layers[layer].mlp.down_proj
    got = {}
    h = down.register_forward_pre_hook(lambda m, a: got.__setitem__("a", a[0].detach()))
    with torch.no_grad():
        whole(batch(ids, 1))
    h.remove()
    score = got["a"][0, 1:].float().abs().amax(0)
    cand = [int(i) for i in score.topk(k).indices]
    W = ao.dense_weight(down)
    cols = W[:, cand].T.contiguous().clone()
    del W
    torch.cuda.empty_cache()
    return cand, cols


# ------------------------------------------------------------------------------------------------------------- A
try:
    L = 44
    down = lm.layers[L].mlp.down_proj
    out_a = {}
    for qname, q in (("capital", "In one sentence, what is the capital of France and which river runs through it?"),
                     ("stock", "Explain briefly why a stock should not be salted while it simmers.")):
        ids = chat(q)
        cand, cols = top_columns(L, ids, 4)
        B = 8
        # rows: 0 base, 1 base, 2 zero delta (named neuron, delta 0), 3 and 4 the same edit (column x -3),
        # 5 another edit (silence the second), 6 base, 7 times-3 floor row
        neuron = torch.tensor([cand[0], cand[0], cand[0], cand[0], cand[0], cand[1], cand[0], cand[0]], device=dev)
        delta = torch.zeros(B, D, device=dev, dtype=torch.bfloat16)
        delta[3] = (-4.0 * cols[0].float()).to(torch.bfloat16)
        delta[4] = delta[3]
        delta[5] = (-1.0 * cols[1].float()).to(torch.bfloat16)
        times = torch.ones(B, device=dev)
        times[7] = 3.0

        def write(hooked):
            rows = [[] for _ in range(B)]
            lps = [[] for _ in range(B)]
            sync()
            t0 = time.perf_counter()
            if hooked:
                with RowNeuron(down, neuron, delta, times):
                    for st in ao.write_rows(lm, W_U, [ids] * B, None, max_new=48, stop_ids=[], bos_id=tok.bos_token_id,
                                            softcap=SOFTCAP):
                        rows[st.row].append(st.token_id)
                        lps[st.row].append(st.logprob)
            else:
                for st in ao.write_rows(lm, W_U, [ids] * B, None, max_new=48, stop_ids=[], bos_id=tok.bos_token_id,
                                        softcap=SOFTCAP):
                    rows[st.row].append(st.token_id)
                    lps[st.row].append(st.logprob)
            sync()
            return rows, lps, time.perf_counter() - t0

        write(False)  # warm
        r0, l0, t_plain = write(False)
        r1, l1, t_hook = write(True)

        def first_parting(a, b):
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    return i
            return None

        out_a[qname] = {
            "prompt_pieces": len(ids), "rows": B, "pieces_written": len(r1[0]),
            "seconds_unhooked": round(t_plain, 3), "seconds_hooked": round(t_hook, 3),
            "seconds_a_step_hooked": round(t_hook / 49, 4),
            "base_rows_0_1_6_same_pieces": r1[0] == r1[1] == r1[6],
            "base_rows_same_logprobs": l1[0] == l1[1] == l1[6],
            "zero_delta_row_same_as_base": (r1[2] == r1[0], l1[2] == l1[0]),
            "hooked_base_same_as_unhooked_write": (r1[0] == r0[0], l1[0] == l0[0]),
            "same_edit_rows_3_4_same": (r1[3] == r1[4], l1[3] == l1[4]),
            "edit_row_parts_from_base_at": first_parting(r1[3], r1[0]),
            "other_edit_parts_from_base_at": first_parting(r1[5], r1[0]),
            "times3_floor_row_parts_from_base_at": first_parting(r1[7], r1[0]),
            "times3_floor_row_max_abs_logprob_gap": round(max(abs(a - b) for a, b in zip(l1[7], l1[0])), 5),
            "base_text": tok.decode(r1[0]),
            "edit_text": tok.decode(r1[3]),
        }
        del cols
        torch.cuda.empty_cache()
    res["A_writing"] = out_a
    save()
    print(json.dumps(out_a, indent=1), flush=True)
except Exception as e:  # noqa: BLE001
    res["errors"].append(f"A: {type(e).__name__}: {e}"[:800])
    save()
    print("A failed", e, flush=True)

# --------------------------------------------------------------------------------------------------------- C and E
try:
    out_c = {}
    for tname, text in (("prose", PROSE), ("recipe", RECIPE)):
        ids = plain(text)
        T = len(ids)
        for layer in (5, 20, 44):
            down = lm.layers[layer].mlp.down_proj
            cand, cols = top_columns(layer, ids, 30)
            B = 32
            # row 0 base, rows 1..30 silence one neuron each, row 31 the times-3 floor row
            neuron = torch.tensor([cand[0]] + cand + [cand[0]], device=dev)
            delta = torch.zeros(B, D, device=dev, dtype=torch.bfloat16)
            delta[1:31] = (-cols.float()).to(torch.bfloat16)
            times = torch.ones(B, device=dev)
            times[31] = 3.0
            pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)

            # (i) same shape, whole forward
            with torch.no_grad(), RowNeuron(down, neuron, delta, times):
                h_same = whole(batch(ids, B))
            # (ii) one-row capture, replayed at 32 rows from this layer
            got = {}

            def pre(_m, args, kwargs):
                got["h"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()

            hk = lm.layers[layer].register_forward_pre_hook(pre, with_kwargs=True)
            with torch.no_grad():
                whole(batch(ids, 1))
            hk.remove()
            hid = got["h"].expand(B, T, D).contiguous()
            sync()
            t0 = time.perf_counter()
            with torch.no_grad(), RowNeuron(down, neuron, delta, times):
                h_rep = ao.forward_from_layer(lm, layer, hid, None, pos)
            sync()
            t_rep = time.perf_counter() - t0

            base_s, base_r = lsm_all(h_same[0]), lsm_all(h_rep[0])
            eff_s, eff_r, cross = [], [], []
            for r in range(1, 31):
                ls, lr = lsm_all(h_same[r]), lsm_all(h_rep[r])
                eff_s.append(float(klpos(base_s, ls).mean()))
                eff_r.append(float(klpos(base_r, lr).mean()))
                cross.append(float(klpos(ls, lr).mean()))
            floor_s = float(klpos(base_s, lsm_all(h_same[31])).mean())
            floor_r = float(klpos(base_r, lsm_all(h_rep[31])).mean())
            es, er = torch.tensor(eff_s), torch.tensor(eff_r)
            out_c[f"{tname}_layer_{layer}"] = {
                "pieces": T, "candidates": 30,
                "base_rows_between_ways_mean_KL": float(klpos(base_s, base_r).mean()),
                "effect_mean_KL_median_same_shape": float(es.median()), "effect_mean_KL_max_same_shape": float(es.max()),
                "effect_abs_gap_between_ways_median": float((es - er).abs().median()),
                "effect_abs_gap_between_ways_max": float((es - er).abs().max()),
                "effect_correlation_between_ways": float(torch.corrcoef(torch.stack([es, er]))[0, 1]),
                "same_row_between_ways_mean_KL_median": statistics.median(cross),
                "same_row_between_ways_mean_KL_max": max(cross),
                "times3_floor_row_mean_KL_same_shape": floor_s, "times3_floor_row_mean_KL_replayed": floor_r,
                "candidates_above_5x_floor_same_shape": int((es > 5 * floor_s).sum()),
                "rank_agreement_top5": len(set(es.topk(5).indices.tolist()) & set(er.topk(5).indices.tolist())),
                "replay_seconds_32_rows": round(t_rep, 3),
            }
            print(f"{tname} layer {layer}:", json.dumps(out_c[f"{tname}_layer_{layer}"]), flush=True)
            del h_same, h_rep, hid, cols, base_s, base_r
            torch.cuda.empty_cache()
            res["C_capture_and_E_floor"] = out_c
            save()
except Exception as e:  # noqa: BLE001
    res["errors"].append(f"C: {type(e).__name__}: {e}"[:800])
    save()
    print("C failed", e, flush=True)

res["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
res["unpacker_after"] = {k: v for k, v in w4a16_fast.describe().items() if k in ("mode", "fell_back")}
save()
print("errors:", res["errors"], flush=True)
