#!/usr/bin/env python3
"""Pilot for the weight-space harness exploration (14 November 2026). Reads only; writes one JSON beside itself.

Stop the :8765 server first. Loads the W4A16 checkpoint without the lens, installs the compiled unpacker, and times:
 P0  a whole forward at several batch shapes;
 P1  a per-row output-side delta on one down_proj (each row its own edit, one row left alone) against no hook;
 P2  the existing module-wide column overlay (the vault's mechanism) against none;
 P3  whether a per-row output-side delta and the module-wide overlay give the same next-piece odds;
 P4  a forward restarted from the edited layer on a kept stream, against a whole forward (bits and time);
 P5  one "paired" teacher-forced reading of 1 base row + N edited rows restarted from the edited layer;
 P6  the floor: the same edit in two rows of one forward; the base row at two batch shapes.

Run: .venv/bin/python -I <this file>
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

from backend.engine_runtime import TheodoliteRuntime  # noqa: E402,F401  (before torch: allocator settings)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import backend.atlas_ops as ao  # noqa: E402
import backend.causal_check as cc  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend.runtime_factory import make_runtime  # noqa: E402

OUT = Path(__file__).with_name("pilot_edit_speed.json")
res: dict = {"when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "errors": []}
dev = torch.device("cuda:0")


def gib(x: int) -> float:
    return round(x / 2**30, 3)


def sync() -> None:
    torch.cuda.synchronize()


def timed(fn, n: int = 5):
    fn()
    sync()
    ts = []
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return {"median_s": round(statistics.median(ts), 4), "min_s": round(min(ts), 4), "max_s": round(max(ts), 4), "n": n}


def section(name):
    def deco(fn):
        try:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out = fn()
            sync()
            res[name] = out
            res[name + "_peak_gib"] = gib(torch.cuda.max_memory_allocated())
            res[name + "_wall_s"] = round(time.perf_counter() - t0, 2)
            print(f"[{name}] ok in {res[name + '_wall_s']} s, peak {res[name + '_peak_gib']} GiB", flush=True)
            print(json.dumps(out, indent=1)[:3000], flush=True)
        except Exception as e:  # noqa: BLE001
            res["errors"].append({"section": name, "error": f"{type(e).__name__}: {e}"[:600]})
            print(f"[{name}] FAILED: {type(e).__name__}: {e}"[:800], flush=True)
            torch.cuda.empty_cache()
        OUT.write_text(json.dumps(res, indent=1))
        return fn
    return deco


# ---------------------------------------------------------------------------------------------------------- load
t0 = time.perf_counter()
rt = make_runtime(load_jlens=False).load()
res["load_s"] = round(time.perf_counter() - t0, 2)
if getattr(rt, "jpp_lenses", None):
    rt.jpp_lenses.clear()
    torch.cuda.empty_cache()
t0 = time.perf_counter()
res["unpacker"] = w4a16_fast.install(rt, "compiled", log=lambda s: print(s, flush=True))
res["unpacker_install_s"] = round(time.perf_counter() - t0, 2)
hf = rt.hf_model
lm = cc.language_model(hf)
tok = rt.tokenizer
W_U = hf.lm_head.weight if getattr(hf, "lm_head", None) is not None else lm.embed_tokens.weight
res["at_rest_gib"] = gib(torch.cuda.memory_allocated())
res["free_gib_after_load"] = gib(torch.cuda.mem_get_info()[0])
print(f"loaded in {res['load_s']} s; at rest {res['at_rest_gib']} GiB; free {res['free_gib_after_load']} GiB", flush=True)

L = 44  # code layer (page 45), a census layer
D_FF = ao.ffn_width(lm, L)
D = ao.hidden_size(lm)
down = lm.layers[L].mlp.down_proj


def ids_of(text: str, chat: bool) -> list:
    if chat:
        out = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        if hasattr(out, "keys"):
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        return [int(t) for t in out]
    out = tok(text, add_special_tokens=True)["input_ids"]
    return [int(t) for t in out]


Q = ids_of("In one sentence, what is the capital of France and which river runs through it?", chat=True)
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
P = ids_of(PROSE, chat=False)
res["tokens"] = {"question": len(Q), "prose": len(P)}
print("tokens", res["tokens"], flush=True)


def batch(ids, B):
    return torch.tensor([ids] * B, dtype=torch.long, device=dev)


@torch.no_grad()
def whole(x):
    return lm(input_ids=x, use_cache=False).last_hidden_state


@torch.no_grad()
def lsm_last(h):
    return F.log_softmax(cc.logits_fp32(h[:, -1], W_U), dim=-1)


def kl(lp, lq):  # KL(p || q), nats
    return float((lp.exp() * (lp - lq)).sum())


# ------------------------------------------------------------------------------------------------------------ P0
@section("P0_whole_forward")
def _p0():
    out = {}
    for name, ids, Bs in (("question", Q, (1, 2, 16, 32, 64)), ("prose", P, (1, 8, 16, 32))):
        for B in Bs:
            x = batch(ids, B)
            try:
                out[f"{name}_T{len(ids)}_B{B}"] = timed(lambda: whole(x))
            except torch.OutOfMemoryError:
                out[f"{name}_T{len(ids)}_B{B}"] = "out of memory"
                torch.cuda.empty_cache()
    return out


# pick neurons: the strongest |activation| at the last piece of the question, layer L
@torch.no_grad()
def top_neurons(ids, k):
    got = {}
    h = down.register_forward_pre_hook(lambda m, a: got.__setitem__("a", a[0].detach()))
    try:
        whole(batch(ids, 1))
    finally:
        h.remove()
    a = got["a"][0, -1].float()
    v, i = a.abs().topk(k)
    return [int(t) for t in i], [round(float(a[t]), 3) for t in i]


NEUR, NEUR_ACT = top_neurons(Q, 64)
res["neurons"] = {"layer_code": L, "top8": NEUR[:8], "acts8": NEUR_ACT[:8]}
W_dense = ao.dense_weight(down)  # [D, D_FF] bf16, a fresh unpacking
res["dense_down_gib"] = gib(W_dense.numel() * W_dense.element_size())
COLS = W_dense[:, NEUR].T.contiguous().clone()  # [64, D]
del W_dense
torch.cuda.empty_cache()


class RowDelta:
    """Per-row output-side delta on one linear: out[r] += (x[r] @ A[r].T) @ Bm[r].T. Rank-1 neuron edits take the
    cheap path: out[r] += x[r, :, n_r, None] * delta[r]."""

    def __init__(self, module):
        self.module = module
        self.neuron = None  # LongTensor [B]
        self.delta = None  # [B, D]
        self.A = None  # [B, r, in]
        self.Bm = None  # [B, out, r]
        self.h = None

    def __enter__(self):
        def hook(_m, args, out):
            x = args[0]
            if self.neuron is not None:
                a = x.gather(2, self.neuron.view(-1, 1, 1).expand(x.shape[0], x.shape[1], 1))  # [B, T, 1]
                return out + a * self.delta[:, None, :].to(out.dtype)
            if self.A is not None:
                z = torch.einsum("btf,brf->btr", x, self.A.to(x.dtype))
                return out + torch.einsum("btr,bdr->btd", z, self.Bm.to(x.dtype))
            return out
        self.h = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self.h.remove()


def neuron_edit(B, factor):
    """Row 0 untouched; row r>=1 multiplies neuron NEUR[r-1]'s column by `factor` (0 silences it)."""
    n = torch.tensor([NEUR[0]] + [NEUR[(r - 1) % len(NEUR)] for r in range(1, B)], device=dev)
    d = torch.zeros(B, D, device=dev, dtype=torch.bfloat16)
    for r in range(1, B):
        d[r] = (factor - 1.0) * COLS[(r - 1) % len(NEUR)]
    return n, d


# ------------------------------------------------------------------------------------------------------------ P1
@section("P1_row_delta_overhead")
def _p1():
    out = {}
    for B in (2, 32, 64):
        x = batch(Q, B)
        base_t = timed(lambda: whole(x))
        rd = RowDelta(down)
        rd.neuron, rd.delta = neuron_edit(B, 0.0)
        with rd:
            r1 = timed(lambda: whole(x))
        out[f"B{B}_no_hook"] = base_t
        out[f"B{B}_rank1_neuron_rows"] = r1
        for rank in (8, 64):
            rd = RowDelta(down)
            rd.A = torch.randn(B, rank, D_FF, device=dev, dtype=torch.bfloat16) * 0.0
            rd.Bm = torch.randn(B, D, rank, device=dev, dtype=torch.bfloat16) * 0.0
            with rd:
                out[f"B{B}_rank{rank}_dense_rows"] = timed(lambda: whole(x))
            del rd
            torch.cuda.empty_cache()
    # deltas on every layer's down_proj at once (60 hooks), rank-1 rows
    B = 32
    x = batch(Q, B)
    hooks = []
    try:
        for l in range(len(lm.layers)):
            rd = RowDelta(lm.layers[l].mlp.down_proj)
            rd.neuron = torch.zeros(B, dtype=torch.long, device=dev)
            rd.delta = torch.zeros(B, D, device=dev, dtype=torch.bfloat16)
            rd.__enter__()
            hooks.append(rd)
        out["B32_rank1_rows_on_all_60_down_proj"] = timed(lambda: whole(x))
    finally:
        for rd in hooks:
            rd.__exit__()
    return out


# ------------------------------------------------------------------------------------------------------------ P2
@section("P2_module_overlay_overhead")
def _p2():
    out = {}
    for B in (1, 32):
        x = batch(Q, B)
        out[f"B{B}_none"] = timed(lambda: whole(x))
        for k in (1, 64):
            try:
                for j in range(k):
                    down._memit_col_overlays[NEUR[j]] = torch.zeros(D, device=dev, dtype=torch.bfloat16)
                out[f"B{B}_{k}_column_overlays_one_linear"] = timed(lambda: whole(x))
            finally:
                down._memit_col_overlays.clear()
    # one overlay on each of 60 down_proj (60 clones of 231 MB a forward)
    x = batch(Q, 1)
    try:
        for l in range(len(lm.layers)):
            lm.layers[l].mlp.down_proj._memit_col_overlays[0] = torch.zeros(D, device=dev, dtype=torch.bfloat16)
        out["B1_one_column_overlay_on_all_60_down_proj"] = timed(lambda: whole(x))
    finally:
        for l in range(len(lm.layers)):
            lm.layers[l].mlp.down_proj._memit_col_overlays.clear()
    out["fell_back"] = w4a16_fast.describe().get("fell_back")
    return out


# ------------------------------------------------------------------------------------------------------------ P3
@section("P3_row_delta_vs_overlay")
def _p3():
    out = {}
    x = batch(Q, 2)
    for factor in (0.0, -3.0):
        n = NEUR[0]
        rd = RowDelta(down)
        rd.neuron = torch.tensor([n, n], device=dev)
        d = torch.zeros(2, D, device=dev, dtype=torch.bfloat16)
        d[1] = (factor - 1.0) * COLS[0]
        rd.delta = d
        with rd:
            lp = lsm_last(whole(x))  # row 0 base, row 1 edited by an output-side delta
        try:
            down._memit_col_overlays[n] = (factor * COLS[0].float()).to(torch.bfloat16)
            lo = lsm_last(whole(x))  # both rows edited by the module-wide overlay
        finally:
            down._memit_col_overlays.clear()
        lb = lsm_last(whole(x))  # both rows base, same batch shape
        out[f"factor_{factor}"] = {
            "effect_KL_base_to_edit_rowdelta": kl(lp[0], lp[1]),
            "effect_KL_base_to_edit_overlay": kl(lb[0], lo[0]),
            "KL_overlay_to_rowdelta": kl(lo[0], lp[1]),
            "max_abs_logprob_diff_overlay_vs_rowdelta_top100": float((lo[0] - lp[1]).abs()[lo[0].topk(100).indices].max()),
            "base_row_same_bits_across_the_two_forwards": bool(torch.equal(lp[0], lb[0])),
            "overlay_rows_same_bits": bool(torch.equal(lo[0], lo[1])),
            "top1_same": bool(lo[0].argmax() == lp[1].argmax()),
        }
    return out


# ------------------------------------------------------------------------------------------------------------ P4
@torch.no_grad()
def stream_entering(ids, B, layer):
    got = {}

    def pre(_m, args, kwargs):
        got["h"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()

    h = lm.layers[layer].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        final = whole(batch(ids, B))
    finally:
        h.remove()
    return got["h"], final


@section("P4_restart_from_layer")
def _p4():
    out = {}
    for name, ids, B in (("question", Q, 32), ("prose", P, 16)):
        T = len(ids)
        pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
        x = batch(ids, B)
        full_t = timed(lambda: whole(x))
        out[f"{name}_B{B}_whole"] = full_t
        for layer in (20, 40, 44, 52, 58):
            hid, final = stream_entering(ids, B, layer)
            part = ao.forward_from_layer(lm, layer, hid, None, pos)
            rec = timed(lambda: ao.forward_from_layer(lm, layer, hid, None, pos))
            rec["same_bits_as_whole"] = bool(torch.equal(part, final))
            rec["max_abs_diff_final_state"] = float((part.float() - final.float()).abs().max())
            rec["KL_next_piece_whole_to_restart_row0"] = kl(lsm_last(final)[0], lsm_last(part)[0])
            rec["share_of_whole"] = round(rec["median_s"] / full_t["median_s"], 3)
            rec["layers_run_share"] = round((60 - layer) / 60, 3)
            out[f"{name}_B{B}_from_layer_{layer}"] = rec
            del hid, final, part
            torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------------------------------------ P5
@section("P5_paired_teacher_forced_from_layer")
def _p5():
    """1 base row + N rows each silencing a different neuron of layer L, restarted from layer L on the kept stream;
    read: mean log-probability of the prose's own next pieces, per row, and KL to the base row at every position."""
    out = {}
    ids = P
    T = len(ids)
    tgt = torch.tensor(ids[1:], device=dev)
    for B in (16, 32, 64):
        try:
            hid1, _ = stream_entering(ids, 1, L)
            hid = hid1.expand(B, T, D).contiguous()
            pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
            rd = RowDelta(down)
            rd.neuron, rd.delta = neuron_edit(B, 0.0)

            @torch.no_grad()
            def run():
                with rd:
                    h = ao.forward_from_layer(lm, L, hid, None, pos)  # [B, T, D]
                base = F.log_softmax(cc.logits_fp32(h[0, :-1], W_U), dim=-1)  # [T-1, V]
                own = torch.empty(B, device=dev)
                kls = torch.empty(B, device=dev)
                own[0] = base.gather(1, tgt[:, None]).mean()
                kls[0] = 0.0
                for r in range(1, B):
                    lp = F.log_softmax(cc.logits_fp32(h[r, :-1], W_U), dim=-1)
                    own[r] = lp.gather(1, tgt[:, None]).mean()
                    kls[r] = (base.exp() * (base - lp)).sum(-1).mean()
                return own, kls

            @torch.no_grad()
            def forward_only():
                with rd:
                    return ao.forward_from_layer(lm, L, hid, None, pos)

            rec = {"forward_only": timed(forward_only, n=3), "forward_and_reading": timed(run, n=3)}
            own, kls = run()
            rec["candidates"] = B - 1
            rec["seconds_per_candidate"] = round(rec["forward_and_reading"]["median_s"] / (B - 1), 4)
            rec["base_mean_logprob"] = round(float(own[0]), 4)
            rec["largest_mean_KL_among_candidates"] = round(float(kls[1:].max()), 5)
            rec["median_mean_KL_among_candidates"] = round(float(kls[1:].median()), 6)
            rec["largest_logprob_shift"] = round(float((own[1:] - own[0]).abs().max()), 5)
            out[f"T{T}_B{B}"] = rec
            del hid, hid1
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            out[f"T{T}_B{B}"] = "out of memory"
            torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------------------------------------ P6
@section("P6_floor")
def _p6():
    out = {}
    # the same edit in two rows of one forward
    B = 4
    x = batch(Q, B)
    rd = RowDelta(down)
    n = NEUR[0]
    rd.neuron = torch.tensor([n] * B, device=dev)
    d = torch.zeros(B, D, device=dev, dtype=torch.bfloat16)
    d[1] = -COLS[0]
    d[2] = -COLS[0]
    rd.delta = d
    with rd:
        lp = lsm_last(whole(x))
    out["same_edit_two_rows_same_bits"] = bool(torch.equal(lp[1], lp[2]))
    out["untouched_rows_same_bits"] = bool(torch.equal(lp[0], lp[3]))
    out["a_zero_delta_row_equals_no_hook_row"] = bool(torch.equal(lp[0], lsm_last(whole(x))[0]))
    # the base row at different batch shapes
    ref = lsm_last(whole(batch(Q, 2)))[0]
    for B2 in (1, 16, 32, 64):
        o = lsm_last(whole(batch(Q, B2)))[0]
        out[f"base_B2_vs_B{B2}"] = {"same_bits": bool(torch.equal(ref, o)), "KL": kl(ref, o)}
    # a single neuron's silencing, as an effect size beside that floor
    out["effect_KL_of_silencing_top_neuron"] = kl(lp[0], lp[1])
    return out


res["unpacker_after"] = w4a16_fast.describe()
res["peak_gib_overall"] = gib(torch.cuda.max_memory_allocated())
OUT.write_text(json.dumps(res, indent=1))
print("written", OUT, flush=True)
print("errors:", json.dumps(res["errors"], indent=1), flush=True)
