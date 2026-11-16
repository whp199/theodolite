#!/usr/bin/env python3
"""Second pilot (14 November 2026), read-only: at an effect of real size, does a per-row output-side delta agree with
the same edit folded into the weight (the vault's module-wide overlay)? Plus: a per-row delta on attention linears,
and what the reading (unembedding to 262,144 pieces) costs in fp32 against the model's own dtype.

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

OUT = Path(__file__).with_name("pilot2_equivalence.json")
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
L = 44
D = ao.hidden_size(lm)
down = lm.layers[L].mlp.down_proj

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
P = [int(t) for t in tok(PROSE, add_special_tokens=True)["input_ids"]]
T = len(P)


def sync():
    torch.cuda.synchronize()


def timed(fn, n=5):
    fn()
    sync()
    ts = []
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return {"median_s": round(statistics.median(ts), 4), "min_s": round(min(ts), 4), "n": n}


def batch(ids, B):
    return torch.tensor([ids] * B, dtype=torch.long, device=dev)


@torch.no_grad()
def whole(x):
    return lm(input_ids=x, use_cache=False).last_hidden_state


@torch.no_grad()
def lsm_all(h_row):  # [T, D] -> [T-1, V] fp32 log-probs
    return F.log_softmax(cc.logits_fp32(h_row[:-1], W_U), dim=-1)


def klpos(lp, lq):
    return (lp.exp() * (lp - lq)).sum(-1)  # [T-1]


def save():
    OUT.write_text(json.dumps(res, indent=1))


try:
    # the neurons that fire hardest over the prose at layer L (any position after the start mark)
    got = {}
    h = down.register_forward_pre_hook(lambda m, a: got.__setitem__("a", a[0].detach()))
    with torch.no_grad():
        whole(batch(P, 1))
    h.remove()
    acts = got["a"][0, 1:].float()  # [T-1, d_ff]
    score = acts.abs().amax(0)
    cand = [int(i) for i in score.topk(63).indices]
    W = ao.dense_weight(down)
    COLS = W[:, cand].T.contiguous().clone()
    del W
    torch.cuda.empty_cache()

    class RowNeuron:
        def __init__(self, module, neuron, delta):
            self.m, self.neuron, self.delta = module, neuron, delta

        def __enter__(self):
            def hook(_m, args, out):
                x = args[0]
                a = x.gather(2, self.neuron.view(-1, 1, 1).expand(x.shape[0], x.shape[1], 1))
                return out + a * self.delta[:, None, :].to(out.dtype)
            self.h = self.m.register_forward_hook(hook)
            return self

        def __exit__(self, *e):
            self.h.remove()

    # 1. rank the 63 candidates by mean KL over the prose when silenced (one forward of 64 rows)
    B = 64
    n = torch.tensor([cand[0]] + cand, device=dev)
    d = torch.zeros(B, D, device=dev, dtype=torch.bfloat16)
    d[1:] = -COLS
    with torch.no_grad(), RowNeuron(down, n, d):
        hs = whole(batch(P, B))
    base = lsm_all(hs[0])
    mk = torch.zeros(B)
    for r in range(1, B):
        mk[r] = klpos(base, lsm_all(hs[r])).mean()
    order = mk[1:].argsort(descending=True)
    res["silencing_mean_KL_top5"] = [{"neuron": cand[int(i)], "mean_KL": round(float(mk[1 + int(i)]), 5)} for i in order[:5]]
    res["silencing_mean_KL_median"] = float(mk[1:].median())
    del hs, base
    torch.cuda.empty_cache()
    save()

    # 2. for the three strongest, and factors 0 and -3: row delta against folded overlay, both at batch 2
    eq = []
    for i in order[:3]:
        j = int(i)
        neuron = cand[j]
        col = COLS[j]
        for factor in (0.0, -3.0):
            n2 = torch.tensor([neuron, neuron], device=dev)
            d2 = torch.zeros(2, D, device=dev, dtype=torch.bfloat16)
            d2[1] = ((factor - 1.0) * col.float()).to(torch.bfloat16)
            with torch.no_grad(), RowNeuron(down, n2, d2):
                h_rd = whole(batch(P, 2))
            lp_base, lp_rd = lsm_all(h_rd[0]), lsm_all(h_rd[1])
            try:
                down._memit_col_overlays[neuron] = (factor * col.float()).to(torch.bfloat16)
                with torch.no_grad():
                    h_ov = whole(batch(P, 2))
            finally:
                down._memit_col_overlays.clear()
            lp_ov = lsm_all(h_ov[0])
            k_rd, k_ov, k_x = klpos(lp_base, lp_rd), klpos(lp_base, lp_ov), klpos(lp_ov, lp_rd)
            eq.append({
                "neuron": neuron, "factor": factor,
                "effect_mean_KL_rowdelta": float(k_rd.mean()), "effect_mean_KL_overlay": float(k_ov.mean()),
                "effect_max_KL_overlay": float(k_ov.max()),
                "between_forms_mean_KL": float(k_x.mean()), "between_forms_max_KL": float(k_x.max()),
                "between_forms_as_share_of_effect": float(k_x.mean() / max(float(k_ov.mean()), 1e-12)),
                "corr_of_per_position_effect": float(torch.corrcoef(torch.stack([k_rd, k_ov]))[0, 1]),
                "top1_agree_share": float((lp_rd.argmax(-1) == lp_ov.argmax(-1)).float().mean()),
                "top1_changed_by_edit_share": float((lp_base.argmax(-1) != lp_ov.argmax(-1)).float().mean()),
            })
            del h_rd, h_ov, lp_base, lp_rd, lp_ov
            torch.cuda.empty_cache()
    res["rowdelta_vs_overlay"] = eq
    save()

    # 3. a per-row rank-1 delta on attention linears (sliding layer 44, full-attention layer 47): the hook's bits and cost
    att = {}
    for layer in (44, 47):
        sa = lm.layers[layer].self_attn
        names = [nm for nm in ("q_proj", "k_proj", "v_proj", "o_proj") if getattr(sa, nm, None) is not None]
        att[f"layer_{layer}_linears"] = names
        x = batch(P, 8)
        t_none = timed(lambda: whole(x), n=3)
        hooks = []
        for nm in names:
            mod = getattr(sa, nm)

            def hook(_m, args, out, _z=None):
                xin = args[0]
                a = xin[..., :1]  # read one input coordinate, per row
                return out + a * torch.zeros(out.shape[0], 1, out.shape[-1], device=out.device, dtype=out.dtype)
            hooks.append(mod.register_forward_hook(hook))
        try:
            t_hook = timed(lambda: whole(x), n=3)
            with torch.no_grad():
                h1 = whole(x)
        finally:
            for hh in hooks:
                hh.remove()
        with torch.no_grad():
            h0 = whole(x)
        att[f"layer_{layer}"] = {"no_hook": t_none, "zero_delta_on_each": t_hook, "zero_delta_same_bits": bool(torch.equal(h0, h1))}
    res["attention_row_delta"] = att
    save()

    # 4. what the reading costs: full-vocabulary log-probs at 175 positions of one row
    with torch.no_grad():
        hrow = whole(batch(P, 1))[0]
    softcap = ao.softcap_of(hf)

    @torch.no_grad()
    def read_fp32():
        return F.log_softmax(cc.logits_fp32(hrow[:-1], W_U), dim=-1)

    @torch.no_grad()
    def read_model_dtype():
        return F.log_softmax(ao._model_logits(lm, W_U, hrow[:-1], softcap).float(), dim=-1)

    a, b = read_fp32(), read_model_dtype()
    res["reading"] = {
        "positions": T - 1,
        "fp32_chunked": timed(read_fp32, n=5),
        "model_dtype_then_float": timed(read_model_dtype, n=5),
        "mean_KL_fp32_to_model_dtype": float(klpos(a, b).mean()),
        "max_KL_fp32_to_model_dtype": float(klpos(a, b).max()),
    }
    save()
except Exception as e:  # noqa: BLE001
    res["errors"].append(f"{type(e).__name__}: {e}"[:800])
    save()
    raise

res["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
res["unpacker_after"] = {k: v for k, v in w4a16_fast.describe().items() if k in ("mode", "fell_back")}
save()
print(json.dumps(res, indent=1))
