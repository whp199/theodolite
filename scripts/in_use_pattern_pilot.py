#!/usr/bin/env python3
"""Probe 3 (GPU): Self-contained local Gemma-4-31B-it pattern authoring, prefill blind_near judging, & behind verification.

Evaluates two cohorts completely locally on Gemma-4-31B-it (zero external LLM API):
  Group A (21 calibration faces from pattern_tests_pilot/patterns.json + blind_near.json's exact 196 pairs):
    - Arm A1: Claude-authored pattern + local Gemma prefill judge (own vs cyclic other_pattern)
    - Arm A2: Gemma-authored pattern (from window%2==0 fit fold) + local Gemma prefill judge (exact same 196 pairs)
  Group B (24 fresh pure_4_4 faces: 8/layer at L2, L14, L26, disjoint from Group A & Probe 1):
    - Gemma-authored pattern from pure-half0 fit docs only
    - Single-prefill blind_near judge on pure-half1 firing + live GPU quiet-verified (<0.25x) near misses
    - Gemma-authored minimal-pair counterfactual endings tested behind held pure-half0 passages via _StopAfterLayer(l)

Never writes raw corpus text or kept_tail to disk; stores only synthetic model patterns/endings and metrics.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import atlas as at_mod  # noqa: E402
from backend import atlas_index as ai  # noqa: E402
from backend import atlas_ops as ao  # noqa: E402
from backend import gap_provenance as gp  # noqa: E402
from backend import pattern_tests as pt  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend.causal_check import language_model, logits_fp32, model_device, pad_right  # noqa: E402
from backend.runtime_factory import make_runtime, weights_card  # noqa: E402
from scripts.in_use_common import (  # noqa: E402
    collect_face_events,
    layer_face_counts_vectorized,
    load_passages_and_doc_halves,
)
from scripts.in_use_context_cpu_probe import _build_pure_h1_token_index, _special_token_ids  # noqa: E402


class _StopAfterLayer(Exception):
    pass


AUTHOR_SYSTEM = (
    "You are an interpretability researcher naming when a single feed-forward neuron fires in a language model. "
    "You are shown up to 8 text passages where the neuron fires on the word-piece marked [[like this]]. "
    "Write ONE concise plain-English pattern (1 to 2 sentences, at most 55 words) stating BOTH: "
    "(1) the specific word-piece/form marked in [[...]], and (2) the surrounding syntax, discourse, or topic "
    "situation required so it can be told apart from a near miss containing the same marked word-piece elsewhere. "
    "Give ONLY the 1-2 sentence pattern with no bullet points, preamble, or neuron IDs."
)

BEHIND_SYSTEM = (
    "You are writing minimal-pair counterfactual endings to test a neuron's firing rule. "
    "You will see a neuron firing pattern and the context immediately before a cut point. "
    "Write 3 short continuations (FIRE 1..3, 4 to 14 words each) that satisfy the pattern RIGHT near the start "
    "so the neuron fires, and 3 closely matched minimal-pair near-miss continuations (QUIET 1..3, similar words/"
    "structure/length) that break the key condition in the pattern so it stays quiet. "
    "Wrap EVERY continuation inside <<<...>>> preserving exact leading space or newline."
)

SINGLE_PAIR_SYSTEM = (
    "You are testing a description of when a single neuron of a language model fires. "
    "You are given the description and ONE pair of snippets (a and b), each with one word-piece marked [[like this]]. "
    "In exactly one snippet the neuron fires strongly on the marked word-piece; in the other it stays quiet. "
    "Using ONLY the description, output a single lowercase letter: a or b."
)


def _snippet_text(tok: Any, marks: Dict[int, Tuple[str, str]], ids: Sequence[int], at: int, before: int = 60, after: int = 8) -> Tuple[str, int, int]:
    s, e, _, _ = at_mod.cut_passage(ids, int(at), int(before), int(after), marks)
    s, e = at_mod.whole_characters(tok, ids, s, e, marks)
    units = [[text, (int(at) - s) in idx] for text, idx in ao.token_units(tok, ids[s:e])]
    return pt.render(units), int(s), int(e)


def _snippet_span_text(tok: Any, ids: Sequence[int], s: int, e: int, at: int) -> str:
    units = [[text, (int(at) - s) in idx] for text, idx in ao.token_units(tok, ids[s:e])]
    return pt.render(units)


def _window_ids(windows: np.ndarray, w: int) -> List[int]:
    row = windows[int(w)]
    valid = row[row >= 0]
    return [int(t) for t in valid.tolist()]


def _single_pair_prompt(pattern: str, a_txt: str, b_txt: str) -> str:
    return (
        "The description of when the neuron fires:\n\n"
        f"{pattern.strip()}\n\n"
        "Pair 1\n"
        f"a) {a_txt}\n"
        f"b) {b_txt}\n\n"
        "Which snippet fires the neuron (a or b)?"
    )


def _author_prompt(key: str, fit_snippets: Sequence[Tuple[float, str]], reads: Sequence[str], follows: Sequence[str]) -> str:
    lines = [
        f"Summary of strongest firings: peak tokens include {', '.join(reads[:5]) or 'varied'}; next tokens include {', '.join(follows[:5]) or 'varied'}.",
        "",
        "Passages where the neuron fires on the [[marked]] word-piece:",
    ]
    for i, (rel, txt) in enumerate(fit_snippets[:8], 1):
        lines.append(f"{i}. ({rel:.2f}x extreme): {txt.replace(chr(10), '⏎')}")
    lines += [
        "",
        "Example 1 output style: The comma closing a long introductory clause before the main subject, or 'such' directly after an introductory phrase.",
        "Example 2 output style: The final word-piece of an un- or in- negated adjective after its prefix, or the preposition immediately following it.",
        "",
        "Write the 1-2 sentence pattern for this neuron:",
    ]
    return "\n".join(lines)


def _behind_prompt(pattern: str, kept_tail_str: str, lead_hint: str) -> str:
    return (
        f"Neuron pattern:\n{pattern.strip()}\n\n"
        f"CONTEXT (ending right at the cut point):\n<<<{kept_tail_str}>>>\n\n"
        f"Formatting note: {lead_hint}\n\n"
        "Write 3 firing continuations and 3 near-miss quiet continuations that continue directly from the end of CONTEXT, in this exact format:\n"
        "FIRE 1: <<<...>>>\nFIRE 2: <<<...>>>\nFIRE 3: <<<...>>>\n"
        "QUIET 1: <<<...>>>\nQUIET 2: <<<...>>>\nQUIET 3: <<<...>>>"
    )


def _parse_behind_endings(text: str) -> Tuple[List[str], List[str]]:
    fire: List[str] = []
    quiet: List[str] = []
    for kind, lst in (("FIRE", fire), ("QUIET", quiet)):
        for m in re.finditer(rf"(?ms)^{kind}\s*\d+\s*:\s*<<<(.*?)>>>", text or ""):
            val = m.group(1)
            if val:
                lst.append(val)
        if not lst:
            for m in re.finditer(rf"(?im)^{kind}\s*\d+\s*:(.*)$", text or ""):
                val = m.group(1).strip()
                if val.startswith('"') and val.endswith('"') and len(val) >= 2:
                    val = val[1:-1]
                elif val.startswith("`") and val.endswith("`") and len(val) >= 2:
                    val = val[1:-1]
                if val:
                    lst.append(val)
    return fire[:3], quiet[:3]


@torch.no_grad()
def _generate_batch(
    rt: Any,
    lm: Any,
    prompts_ids: Sequence[Sequence[int]],
    *,
    batch_size: int = 12,
    max_new: int = 96,
) -> List[str]:
    stop_ids = rt.end_of_turn_ids()
    bos_id = rt.tokenizer.bos_token_id
    softcap = ao.softcap_of(rt.hf_model)
    out_texts: List[str] = [""] * len(prompts_ids)
    for s in range(0, len(prompts_ids), int(batch_size)):
        e = min(len(prompts_ids), s + int(batch_size))
        chunk = list(prompts_ids[s:e])
        td = ao.TextDeltas(rt.tokenizer, len(chunk))
        for st in ao.write_rows(
            lm,
            rt.lm_head_weight,
            chunk,
            None,
            max_new=int(max_new),
            stop_ids=stop_ids,
            bos_id=bos_id,
            spare_bos=True,
            softcap=softcap,
            top_k=0,
        ):
            out_texts[s + st.row] += td.add(st.row, st.token_id, st.end)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return [t.strip() for t in out_texts]


@torch.no_grad()
def _read_multi_neurons_to_layer(
    lm: Any,
    seqs: Sequence[Sequence[int]],
    targets: Sequence[Tuple[int, int]],
    *,
    batch_size: int = 16,
) -> List[np.ndarray]:
    """Run lm forward ONLY up to max_layer among `targets` and extract activations [seq_len] per row."""
    dev = model_device(lm)
    max_l = max(int(l) for l, _ in targets)
    per_layer: Dict[int, List[Tuple[int, int]]] = {}
    for idx_r, (l, n) in enumerate(targets):
        per_layer.setdefault(int(l), []).append((idx_r, int(n)))

    res: List[Optional[np.ndarray]] = [None] * len(seqs)
    for s in range(0, len(seqs), int(batch_size)):
        e = min(len(seqs), s + int(batch_size))
        batch_seqs = seqs[s:e]
        inp, att, _, _ = pad_right(batch_seqs, dev)
        hks = []
        cap_l: Dict[int, torch.Tensor] = {}
        for l_k in sorted(per_layer):
            def _make_cap(lk: int):
                def _hk(_m: Any, f_args: Tuple[torch.Tensor, ...]) -> None:
                    cap_l[lk] = f_args[0].detach()

                return _hk

            hks.append(lm.layers[l_k].mlp.down_proj.register_forward_pre_hook(_make_cap(l_k)))

        def _stop(_m: Any, _a: Any, _o: Any) -> None:
            raise _StopAfterLayer()

        hks.append(lm.layers[max_l].register_forward_hook(_stop))
        try:
            try:
                lm(input_ids=inp, attention_mask=att, use_cache=False)
            except _StopAfterLayer:
                pass
        finally:
            for hk in hks:
                hk.remove()

        for i in range(e - s):
            l_i, n_i = targets[s + i]
            L_seq = len(batch_seqs[i])
            res[s + i] = cap_l[int(l_i)][i, :L_seq, int(n_i)].float().cpu().numpy()
        del cap_l, inp, att
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return [r for r in res if r is not None]


@torch.no_grad()
def _score_ab_prefills(
    rt: Any,
    lm: Any,
    prompts_ids: Sequence[Sequence[int]],
    ab_ids: Dict[str, List[int]],
    *,
    batch_size: int = 16,
) -> List[Dict[str, float]]:
    """Single-prefill score at last position: P(a_tokens) vs P(b_tokens) in fp32 softcapped logits."""
    dev = model_device(lm)
    W_U = rt.lm_head_weight
    a_t = torch.tensor(ab_ids["a"], dtype=torch.long, device=dev)
    b_t = torch.tensor(ab_ids["b"], dtype=torch.long, device=dev)
    out: List[Dict[str, float]] = []

    for s in range(0, len(prompts_ids), int(batch_size)):
        e = min(len(prompts_ids), s + int(batch_size))
        chunk = prompts_ids[s:e]
        T = max(len(x) for x in chunk)
        inp = torch.tensor([[0] * (T - len(x)) + [int(t) for t in x] for x in chunk], dtype=torch.long, device=dev)
        att = torch.tensor([[0] * (T - len(x)) + [1] * len(x) for x in chunk], dtype=torch.long, device=dev)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        res = lm(input_ids=inp, attention_mask=att, position_ids=pos, use_cache=False)
        h = res.last_hidden_state[:, -1, :]
        lsm = F.log_softmax(logits_fp32(h, W_U), dim=-1)  # [b, V]
        pa = lsm[:, a_t].exp().sum(dim=-1).cpu().numpy()
        pb = lsm[:, b_t].exp().sum(dim=-1).cpu().numpy()
        log_pa = torch.logsumexp(lsm[:, a_t], dim=-1).cpu().numpy()
        log_pb = torch.logsumexp(lsm[:, b_t], dim=-1).cpu().numpy()
        for i in range(e - s):
            margin = float(log_pa[i] - log_pb[i])
            out.append(
                {
                    "pick": "a" if margin >= 0.0 else "b",
                    "margin": round(margin, 4),
                    "p_ab": round(float(pa[i] + pb[i]), 4),
                }
            )
        del res, h, lsm, inp, att
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return out


def _evaluate_ab_scores(
    items: Sequence[Dict[str, Any]],
    own_scores: Sequence[Dict[str, float]],
    other_scores: Sequence[Dict[str, float]],
) -> Dict[str, Any]:
    """Summarize pooled, per-face p<0.05, and by-match blind_near accuracy."""
    by_face: Dict[str, Dict[str, Any]] = {}
    by_match: Dict[str, Dict[str, int]] = {}
    own_right = oth_right = tot = 0
    mass_vals: List[float] = []
    for it, so, sx in zip(items, own_scores, other_scores):
        k = it["face_key"]
        tr = it["truth"]
        lv = it["match"]
        f_row = by_face.setdefault(k, {"layer": it["layer"], "truth": [], "own": [], "other": [], "match": []})
        f_row["truth"].append(tr)
        f_row["own"].append(so["pick"])
        f_row["other"].append(sx["pick"])
        f_row["match"].append(lv)
        tot += 1
        own_right += int(so["pick"] == tr)
        oth_right += int(sx["pick"] == tr)
        mass_vals.extend([so["p_ab"], sx["p_ab"]])
        mr = by_match.setdefault(lv, {"pairs": 0, "own_right": 0, "other_right": 0})
        mr["pairs"] += 1
        mr["own_right"] += int(so["pick"] == tr)
        mr["other_right"] += int(sx["pick"] == tr)

    faces_out: Dict[str, Any] = {}
    n_p05_own = n_p05_oth = 0
    for k, fr in by_face.items():
        r_own = pt.blind_result(fr["own"], fr["truth"], fr["match"])
        r_oth = pt.blind_result(fr["other"], fr["truth"], fr["match"])
        if r_own["p"] < 0.05:
            n_p05_own += 1
        if r_oth["p"] < 0.05:
            n_p05_oth += 1
        faces_out[k] = {
            "layer": int(fr["layer"]),
            "pairs": r_own["pairs"],
            "own_right": r_own["right"],
            "own_p": r_own["p"],
            "other_right": r_oth["right"],
            "other_p": r_oth["p"],
        }

    return {
        "faces": len(by_face),
        "pairs": tot,
        "own_right": own_right,
        "own_share": round(own_right / max(tot, 1), 4),
        "own_binom_p": round(pt.binom_tail(own_right, tot), 8),
        "other_right": oth_right,
        "other_share": round(oth_right / max(tot, 1), 4),
        "other_binom_p": round(pt.binom_tail(oth_right, tot), 8),
        "own_minus_other": round((own_right - oth_right) / max(tot, 1), 4),
        "faces_p_lt_005_own": n_p05_own,
        "faces_p_lt_005_other": n_p05_oth,
        "median_p_ab_mass": round(float(np.median(mass_vals)) if mass_vals else 0.0, 4),
        "by_match": {
            lv: {
                **by_match[lv],
                "own_share": round(by_match[lv]["own_right"] / max(by_match[lv]["pairs"], 1), 4),
                "other_share": round(by_match[lv]["other_right"] / max(by_match[lv]["pairs"], 1), 4),
            }
            for lv in (*pt.NEAR_LEVELS, pt.ANYWHERE)
            if lv in by_match
        },
        "per_face": faces_out,
    }


def _collect_group_a_fit_snippets(
    rt: Any,
    marks: Dict[int, Tuple[str, str]],
    P: Dict[str, Any],
    windows: np.ndarray,
    act_scale: Any,
    vocab: Sequence[str],
    layer: int,
    n: int,
    face_char: str,
) -> Tuple[List[Tuple[float, str]], List[str], List[str]]:
    """Mirror backend/atlas.py fold=fit (window % 2 == 0) for apples-to-apples Group A comparison."""
    face = "pos" if face_char == "+" else "neg"
    sgn = 1.0 if face == "pos" else -1.0
    ext = float(abs(act_scale[face][layer, n]))
    slots: List[Tuple[int, int, float, bool]] = []
    for wi, ai_, ac in zip(P[f"{face}_win"][layer, n], P[f"{face}_at"][layer, n], P[f"{face}_act"][layer, n]):
        slots.append((int(wi), int(ai_), float(ac), True))
    if f"{face}_smp_win" in P:
        smp = [
            (int(wi), int(ai_), float(ac), False)
            for wi, ai_, ac in zip(
                P[f"{face}_smp_win"][layer, n], P[f"{face}_smp_at"][layer, n], P[f"{face}_smp_act"][layer, n]
            )
        ]
        smp.sort(key=lambda x: -(sgn * x[2]))
        slots.extend(smp)

    seen: Set[int] = set()
    snippets: List[Tuple[float, str]] = []
    reads: List[str] = []
    follows: List[str] = []
    for wi, ai_, ac, is_top in slots:
        if wi in seen or wi < 0 or wi >= windows.shape[0] or ai_ < 1 or wi % 2 == 1:
            continue
        seen.add(wi)
        ids = _window_ids(windows, wi)
        if ai_ >= len(ids):
            continue
        txt, _, _ = _snippet_text(rt.tokenizer, marks, ids, ai_, before=60, after=8)
        snippets.append((ac / ext if ext else ac, txt))
        if is_top:
            reads.append(vocab[ids[ai_]])
            if ai_ + 1 < len(ids):
                follows.append(vocab[ids[ai_ + 1]])
    return snippets[:8], reads, follows


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe 3 GPU: Local Gemma-4-31B-it pattern authoring, prefill judging & behind verification")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--out-dir", default="experiments/results/in_use_kill_pilots")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    t0_iso = gp.utc_iso(t0)
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else ai.atlas_dir()

    loaded = load_passages_and_doc_halves(data_dir)
    P = loaded["P"]
    act_scale = loaded["act_scale"]
    vocab = loaded["vocab"]
    windows = loaded["windows"]
    doc_id = loaded["doc_id"]
    half_id = loaded["half_id"]
    pure_h0_mask = loaded["pure_h0_mask"]
    pure_h1_mask = loaded["pure_h1_mask"]
    bos_id, eos_id = loaded["bos_id"], loaded["eos_id"]

    pats_json = json.loads((REPO_ROOT / "experiments/results/pattern_tests_pilot/patterns.json").read_text(encoding="utf-8"))["faces"]
    blind_near_json = json.loads((REPO_ROOT / "experiments/results/pattern_tests_pilot/blind_near.json").read_text(encoding="utf-8"))["tests"]

    # Exclude Group A and Probe 1 neurons when sampling 24 fresh Group B faces (8 each at L2, L14, L26)
    exclude_set: Set[Tuple[int, int]] = {(int(v["layer"]), int(v["n"])) for v in pats_json.values()}
    probe1_path = out_dir / "probe1_in_use_causal.json"
    if probe1_path.is_file():
        p1 = json.loads(probe1_path.read_text(encoding="utf-8"))
        for lk, vals in (p1.get("neuron_ids") or {}).items():
            for tag in vals:
                m = re.match(r"^n(\d+)[pm]$", str(tag))
                if m:
                    exclude_set.add((int(lk), int(m.group(1))))

    rt = make_runtime(load_jlens=False).load()
    w4a16_fast.install(rt, "compiled", log=print)
    lm = language_model(rt.hf_model)
    tok = rt.tokenizer
    marks = at_mod.passage_marks(tok)
    special_ids = _special_token_ids(bos_id, eos_id)
    h1_idx = _build_pure_h1_token_index(windows, pure_h1_mask, special_ids, min_at=8, min_after=8)

    ab_ids = {
        "a": [int(tok.encode(s, add_special_tokens=False)[0]) for s in ("a", " a", "A", " A")],
        "b": [int(tok.encode(s, add_special_tokens=False)[0]) for s in ("b", " b", "B", " B")],
    }
    print(f"ab token ids: {ab_ids}", flush=True)

    # Pre-check Re-plan sign #3 on 2 trivial calibration pairs (empty prefix vs "Answer: ")
    toy_pairs = [
        ("The final word-piece of an un- or in- adjective such as [[undeterred]].", "She stayed [[undeterred]] in the storm.", "She stayed [[afraid]] in the storm.", "a"),
        ("A comma closing a long introductory clause before the main clause subject.", "He bought apples[[,]] pears and figs.", "Under the terms of the old treaty[[,]] the council met.", "b"),
    ]
    for apref in ("", "Answer: "):
        toy_p = [
            rt.tokenize_prompt(rt.compose_prompt(_single_pair_prompt(p, a_t, b_t), system=SINGLE_PAIR_SYSTEM, chat=True, thinking=False, assistant_prefix=apref))
            for p, a_t, b_t, _ in toy_pairs
        ]
        sc = _score_ab_prefills(rt, lm, toy_p, ab_ids, batch_size=2)
        print(f"Toy check prefix={apref!r}: {sc}", flush=True)
    # Use prefix="" if p_ab >= 0.5 and 2/2 right, else check "Answer: "
    best_apref = ""
    best_toy_p = [
        rt.tokenize_prompt(rt.compose_prompt(_single_pair_prompt(p, a_t, b_t), system=SINGLE_PAIR_SYSTEM, chat=True, thinking=False, assistant_prefix=""))
        for p, a_t, b_t, _ in toy_pairs
    ]
    sc0 = _score_ab_prefills(rt, lm, best_toy_p, ab_ids, batch_size=2)
    if any(r["p_ab"] < 0.5 for r in sc0):
        best_apref = "Answer: "
    print(f"Using assistant_prefix={best_apref!r} for single-prefill judging", flush=True)

    # --- 1. Sample 24 fresh Group B faces (L2, L14, L26) ---
    group_b_faces: List[Dict[str, Any]] = []
    for l in (2, 14, 26):
        cnt = layer_face_counts_vectorized(
            P, windows, act_scale, half_id, pure_h0_mask, pure_h1_mask, l, eos_id=eos_id, bos_id=bos_id
        )
        cands: List[Tuple[int, str]] = []
        N_ff = int(cnt["pos_pure44"].shape[0])
        for n in range(N_ff):
            if (l, n) in exclude_set:
                continue
            opts = []
            for face in ("pos", "neg"):
                if bool(cnt[f"{face}_pure44"][n]):
                    opts.append(face)
            if opts:
                cands.append((n, opts[0] if len(opts) == 1 else ("pos" if cnt["pos_pure0"][n] >= cnt["neg_pure0"][n] else "neg")))
        rng = np.random.default_rng(2026112503 + l)
        picks = sorted(int(i) for i in rng.choice(len(cands), size=8, replace=False))
        for idx_c in picks:
            n, fc = cands[idx_c]
            ev = collect_face_events(
                P, windows, act_scale, pure_h0_mask, pure_h1_mask, doc_id, half_id, l, n, fc, eos_id=eos_id, bos_id=bos_id
            )
            fit_evs = ev["pure_h0"][:8]
            test_evs = ev["pure_h1"][:8]
            kname = f"L{l}-n{n}{'p' if fc == 'pos' else 'm'}"
            group_b_faces.append(
                {
                    "key": kname,
                    "layer": int(l),
                    "n": int(n),
                    "face": fc,
                    "face_char": "+" if fc == "pos" else "-",
                    "extreme": ev["extreme"],
                    "own_windows": ev["own_windows"],
                    "fit_evs": fit_evs,
                    "test_evs": test_evs,
                }
            )
    print(f"Sampled {len(group_b_faces)} Group B faces across L2/L14/L26", flush=True)

    # --- 2. Build live-verified quiet near-miss pairs for Group B via round-batched _StopAfterLayer(26) ---
    rng_pair = random.Random(2026112503)
    rng_np = np.random.default_rng(2026112503)
    pair_slots: List[Dict[str, Any]] = []
    for fb in group_b_faces:
        ext = float(fb["extreme"])
        l, n = fb["layer"], fb["n"]
        sgn = 1.0 if fb["face"] == "pos" else -1.0
        for ev in fb["test_evs"]:
            ids = _window_ids(windows, ev["win"])
            at = int(ev["at"])
            fire_txt, s_f, e_f = _snippet_text(tok, marks, ids, at, before=60, after=8)
            before_f, after_f = at - s_f, e_f - at - 1
            # Gather candidate corpus near misses in pure_h1 outside own_windows
            cand_q: List[Tuple[int, int, str, int, int, List[int]]] = []
            tok0 = int(ids[at])
            prev = int(ids[at - 1]) if at >= 1 else None
            nxt = int(ids[at + 1]) if at + 1 < len(ids) else None
            seen_q: Set[Tuple[int, int]] = set()
            bounds = h1_idx["bounds"]
            if tok0 in bounds:
                lo, hi = bounds[tok0]
                ws_all = h1_idx["w"][lo:hi]
                ats_all = h1_idx["a"][lo:hi]
                if fb["own_windows"]:
                    keep = ~np.isin(ws_all, list(fb["own_windows"]))
                    ws_all, ats_all = ws_all[keep], ats_all[keep]
                if len(ws_all) > 4096:
                    sub_i = rng_np.choice(len(ws_all), size=4096, replace=False)
                    ws_all, ats_all = ws_all[sub_i], ats_all[sub_i]
                if len(ws_all) > 0:
                    p_prev = windows[ws_all, ats_all - 1]
                    p_nxt = windows[ws_all, ats_all + 1]
                    for lv in pt.NEAR_LEVELS:
                        if lv == "phrase":
                            if prev is None or nxt is None:
                                continue
                            m = (p_prev == prev) & (p_nxt == nxt)
                        elif lv == "before":
                            if prev is None:
                                continue
                            m = p_prev == prev
                        elif lv == "after":
                            if nxt is None:
                                continue
                            m = p_nxt == nxt
                        else:
                            m = np.ones(len(ws_all), dtype=bool)
                        hits = np.flatnonzero(m)
                        if len(hits) == 0:
                            continue
                        rng_np.shuffle(hits)
                        added = 0
                        for h_j in hits:
                            if added >= pt.NEAR_TRIES:
                                break
                            wq, aq = int(ws_all[h_j]), int(ats_all[h_j])
                            if (wq, aq) in seen_q:
                                continue
                            qids = _window_ids(windows, wq)
                            sq, eq, _, _ = at_mod.cut_passage(qids, aq, before_f, after_f, marks)
                            sq, eq = at_mod.whole_characters(tok, qids, sq, eq, marks)
                            if aq - sq < before_f - 2 or eq - aq - 1 < after_f - 2:
                                continue
                            seen_q.add((wq, aq))
                            cand_q.append((wq, aq, lv, sq, eq, qids[:eq]))
                            added += 1

            # Fallback pure_h1 anywhere candidates
            h1_pool = [int(w) for w in h1_idx["h1_wins"].tolist() if int(w) not in fb["own_windows"]]
            for _ in range(8):
                wq = h1_pool[rng_pair.randrange(len(h1_pool))]
                qids = _window_ids(windows, wq)
                if len(qids) <= max(before_f + 4, 16) + after_f:
                    continue
                aq = rng_pair.randrange(max(8, before_f + 1), len(qids) - max(1, after_f))
                if int(qids[aq]) in marks or (wq, aq) in seen_q:
                    continue
                sq, eq, _, _ = at_mod.cut_passage(qids, aq, before_f, after_f, marks)
                sq, eq = at_mod.whole_characters(tok, qids, sq, eq, marks)
                if aq - sq < before_f - 2 or eq - aq - 1 < after_f - 2:
                    continue
                seen_q.add((wq, aq))
                cand_q.append((wq, aq, pt.ANYWHERE, sq, eq, qids[:eq]))

            pair_slots.append(
                {
                    "fb": fb,
                    "fire_txt": fire_txt,
                    "cands": cand_q,
                    "quiet": None,
                    "first_is_fire": rng_pair.random() < 0.5,
                }
            )

    t_search = time.time()
    max_cand_rounds = max((len(ps["cands"]) for ps in pair_slots), default=0)
    for rnd in range(max_cand_rounds):
        todo = [ps for ps in pair_slots if ps["quiet"] is None and rnd < len(ps["cands"])]
        if not todo:
            break
        seqs_r = [ps["cands"][rnd][5] for ps in todo]
        tgts_r = [(ps["fb"]["layer"], ps["fb"]["n"]) for ps in todo]
        acts_r = _read_multi_neurons_to_layer(lm, seqs_r, tgts_r, batch_size=16)
        for ps, act_v in zip(todo, acts_r):
            wq, aq, lv, sq, eq, qids_e = ps["cands"][rnd]
            fb = ps["fb"]
            sgn = 1.0 if fb["face"] == "pos" else -1.0
            rel_span = float(np.max((sgn * act_v[sq:eq]) / max(fb["extreme"], 1e-6)))
            if rel_span < pt.QUIET_BELOW:
                q_txt = _snippet_span_text(tok, qids_e, sq, eq, aq)
                ps["quiet"] = {"win": wq, "at": aq, "match": lv, "rel": round(rel_span, 3), "txt": q_txt}
    print(
        f"Verified {sum(1 for ps in pair_slots if ps['quiet'] is not None)}/{len(pair_slots)} Group B quiet near-miss pairs in {time.time()-t_search:.1f}s",
        flush=True,
    )

    # --- 3. Local Gemma-4-31B-it pattern authoring (21 Group A + 24 Group B = 45 faces) ---
    author_jobs: List[Tuple[str, str, int, List[int]]] = []
    # Group A
    for k, fa in pats_json.items():
        snips, reads, follows = _collect_group_a_fit_snippets(rt, marks, P, windows, act_scale, vocab, fa["layer"], fa["n"], fa["face"])
        pmt = _author_prompt(k, snips, reads, follows)
        pids = rt.tokenize_prompt(rt.compose_prompt(pmt, system=AUTHOR_SYSTEM, chat=True, thinking=False))
        author_jobs.append(("A", k, int(fa["layer"]), pids))
    # Group B
    for fb in group_b_faces:
        snips = []
        reads, follows = [], []
        for idx_e, ev in enumerate(fb["fit_evs"]):
            ids = _window_ids(windows, ev["win"])
            txt, _, _ = _snippet_text(tok, marks, ids, ev["at"], before=60, after=8)
            snips.append((ev["rel"], txt))
            reads.append(vocab[ids[ev["at"]]])
            if ev["at"] + 1 < len(ids):
                follows.append(vocab[ids[ev["at"] + 1]])
        pmt = _author_prompt(fb["key"], snips, reads, follows)
        pids = rt.tokenize_prompt(rt.compose_prompt(pmt, system=AUTHOR_SYSTEM, chat=True, thinking=False))
        author_jobs.append(("B", fb["key"], fb["layer"], pids))

    t_auth = time.time()
    gemma_texts = _generate_batch(rt, lm, [j[3] for j in author_jobs], batch_size=int(args.batch_size), max_new=96)
    gemma_patterns: Dict[str, str] = {j[1]: txt for j, txt in zip(author_jobs, gemma_texts)}
    print(f"Gemma authored {len(gemma_patterns)} patterns in {time.time()-t_auth:.1f}s", flush=True)

    # --- 4. Single-prefill blind_near scoring: Arm A1 (Claude pat), Arm A2 (Gemma pat), Arm B (Group B Gemma pat) ---
    group_a_order = list(pats_json.keys())
    items_a: List[Dict[str, Any]] = []
    for k in group_a_order:
        bn = blind_near_json.get(k)
        if not bn or "by_pair" not in bn:
            continue
        other_k = bn["other_pattern_of"]
        layer_k = int(pats_json[k]["layer"])
        for pr in bn["by_pair"]:
            f_win, f_at, f_s, f_e = pr["fires"]["window"], pr["fires"]["at"], pr["fires"]["span"][0], pr["fires"]["span"][1]
            q_win, q_at, q_s, q_e = pr["quiet"]["window"], pr["quiet"]["at"], pr["quiet"]["span"][0], pr["quiet"]["span"][1]
            f_txt = _snippet_span_text(tok, _window_ids(windows, f_win), f_s, f_e, f_at)
            q_txt = _snippet_span_text(tok, _window_ids(windows, q_win), q_s, q_e, q_at)
            tr = pr["truth"]
            a_txt, b_txt = (f_txt, q_txt) if tr == "a" else (q_txt, f_txt)
            items_a.append(
                {
                    "face_key": k,
                    "other_key": other_k,
                    "layer": layer_k,
                    "truth": tr,
                    "match": pr["quiet"]["match"],
                    "a_txt": a_txt,
                    "b_txt": b_txt,
                }
            )

    def _encode_judge_items(items: Sequence[Dict[str, Any]], pat_map: Dict[str, str], use_other: bool) -> List[List[int]]:
        out_p: List[List[int]] = []
        for it in items:
            pk = it["other_key"] if use_other else it["face_key"]
            msg = _single_pair_prompt(pat_map[pk], it["a_txt"], it["b_txt"])
            out_p.append(rt.tokenize_prompt(rt.compose_prompt(msg, system=SINGLE_PAIR_SYSTEM, chat=True, thinking=False, assistant_prefix=best_apref)))
        return out_p

    t_judge = time.time()
    claude_pat_map = {k: v["pattern"] for k, v in pats_json.items()}
    a1_own_sc = _score_ab_prefills(rt, lm, _encode_judge_items(items_a, claude_pat_map, False), ab_ids, batch_size=16)
    a1_oth_sc = _score_ab_prefills(rt, lm, _encode_judge_items(items_a, claude_pat_map, True), ab_ids, batch_size=16)
    arm_a1 = _evaluate_ab_scores(items_a, a1_own_sc, a1_oth_sc)

    a2_own_sc = _score_ab_prefills(rt, lm, _encode_judge_items(items_a, gemma_patterns, False), ab_ids, batch_size=16)
    a2_oth_sc = _score_ab_prefills(rt, lm, _encode_judge_items(items_a, gemma_patterns, True), ab_ids, batch_size=16)
    arm_a2 = _evaluate_ab_scores(items_a, a2_own_sc, a2_oth_sc)

    # Group B blind items + cyclic same-layer other_key
    b_by_layer: Dict[int, List[str]] = {}
    for fb in group_b_faces:
        b_by_layer.setdefault(fb["layer"], []).append(fb["key"])
    b_other_map: Dict[str, str] = {}
    for l_k, ks in b_by_layer.items():
        for i, fk in enumerate(ks):
            b_other_map[fk] = ks[(i + 1) % len(ks)]

    items_b: List[Dict[str, Any]] = []
    for ps in pair_slots:
        if ps["quiet"] is None:
            continue
        fb = ps["fb"]
        tr = "a" if ps["first_is_fire"] else "b"
        a_txt, b_txt = (ps["fire_txt"], ps["quiet"]["txt"]) if tr == "a" else (ps["quiet"]["txt"], ps["fire_txt"])
        items_b.append(
            {
                "face_key": fb["key"],
                "other_key": b_other_map[fb["key"]],
                "layer": fb["layer"],
                "truth": tr,
                "match": ps["quiet"]["match"],
                "a_txt": a_txt,
                "b_txt": b_txt,
            }
        )

    b_own_sc = _score_ab_prefills(rt, lm, _encode_judge_items(items_b, gemma_patterns, False), ab_ids, batch_size=16)
    b_oth_sc = _score_ab_prefills(rt, lm, _encode_judge_items(items_b, gemma_patterns, True), ab_ids, batch_size=16)
    arm_b = _evaluate_ab_scores(items_b, b_own_sc, b_oth_sc)
    print(
        f"Prefill blind_near ({time.time()-t_judge:.1f}s): "
        f"ArmA1(ClaudePat)={arm_a1['own_share']} vs {arm_a1['other_share']}; "
        f"ArmA2(GemmaPat)={arm_a2['own_share']} vs {arm_a2['other_share']}; "
        f"ArmB(FreshB)={arm_b['own_share']} vs {arm_b['other_share']}",
        flush=True,
    )

    # --- 5. Counterfactual behind-passage synthesis & physical firing test on 24 Group B faces ---
    t_beh = time.time()
    behind_prep: List[Dict[str, Any]] = []
    for fb in group_b_faces:
        chosen_ev = None
        for ev in fb["fit_evs"]:
            if int(ev["at"]) + 1 - pt.DROP >= 2:
                chosen_ev = ev
                break
        if chosen_ev is None:
            continue
        ids = _window_ids(windows, chosen_ev["win"])
        at0 = int(chosen_ev["at"])
        kept = ids[: at0 + 1 - pt.DROP]
        dropped = ids[at0 + 1 - pt.DROP : at0 + 1]
        tail_ids = kept[-60:]
        kept_tail_str = "".join(t for t, _ in ao.token_units(tok, tail_ids))
        first_drop_str = tok.decode([int(dropped[0])])
        if first_drop_str.startswith("\n"):
            hint = "Notice the original continuation begins with a newline character."
        elif first_drop_str.startswith(" "):
            hint = "Notice the original continuation begins with a leading space."
        else:
            hint = "Notice the original continuation begins directly with no leading space."
        pmt = _behind_prompt(gemma_patterns[fb["key"]], kept_tail_str, hint)
        pids = rt.tokenize_prompt(rt.compose_prompt(pmt, system=BEHIND_SYSTEM, chat=True, thinking=False))
        behind_prep.append(
            {
                "fb": fb,
                "win": chosen_ev["win"],
                "at": at0,
                "kept": kept,
                "dropped": dropped,
                "pids": pids,
            }
        )

    beh_raw_outs = _generate_batch(rt, lm, [bp["pids"] for bp in behind_prep], batch_size=int(args.batch_size), max_new=160)

    # Batched forward up to layer l on [kept + own_dropped, kept + fire_endings, kept + quiet_endings]
    eval_seqs: List[List[int]] = []
    eval_tgts: List[Tuple[int, int]] = []
    eval_meta: List[Tuple[int, str, int, str]] = []  # (bp_idx, kind, k_len, ending_text)
    for idx_bp, (bp, raw_txt) in enumerate(zip(behind_prep, beh_raw_outs)):
        fb = bp["fb"]
        kept = bp["kept"]
        eval_seqs.append(list(kept) + list(bp["dropped"]))
        eval_tgts.append((fb["layer"], fb["n"]))
        eval_meta.append((idx_bp, "own", len(kept), "<own_held_out_not_stored>"))
        f_strs, q_strs = _parse_behind_endings(raw_txt)
        for kind, lst in (("fire", f_strs), ("quiet", q_strs)):
            for es in lst:
                e_ids = tok.encode(es, add_special_tokens=False)
                if not e_ids or len(kept) + len(e_ids) > 1024:
                    continue
                eval_seqs.append(list(kept) + [int(t) for t in e_ids])
                eval_tgts.append((fb["layer"], fb["n"]))
                eval_meta.append((idx_bp, kind, len(kept), es))

    acts_beh = _read_multi_neurons_to_layer(lm, eval_seqs, eval_tgts, batch_size=16) if eval_seqs else []
    per_bp_eval: Dict[int, Dict[str, Any]] = {
        i: {"own_rel": None, "fire": [], "quiet": []} for i in range(len(behind_prep))
    }
    for (idx_bp, kind, k_len, es), act_v in zip(eval_meta, acts_beh):
        fb = behind_prep[idx_bp]["fb"]
        sgn = 1.0 if fb["face"] == "pos" else -1.0
        rel_end = float(np.max((sgn * act_v[k_len:]) / max(fb["extreme"], 1e-6)))
        if kind == "own":
            per_bp_eval[idx_bp]["own_rel"] = round(rel_end, 3)
        else:
            per_bp_eval[idx_bp][kind].append({"text": es[:160], "rel": round(rel_end, 3)})

    behind_faces: Dict[str, Any] = {}
    tot_fire = tot_fired = tot_beat = tot_q = tot_qf = 0
    faces_fired_gt0 = faces_clean_pass = 0
    apart_vals: List[float] = []
    valid_behind_faces = 0
    for idx_bp, bp in enumerate(behind_prep):
        fb = bp["fb"]
        ev_r = per_bp_eval[idx_bp]
        wr = pt.written_result(ev_r["fire"], ev_r["quiet"])
        own_ok = (ev_r["own_rel"] or 0.0) >= pt.FIRES_AT
        if own_ok:
            valid_behind_faces += 1
            tot_fire += wr["fire"]
            tot_fired += wr["fired"]
            tot_beat += wr["beat_corpus"]
            tot_q += wr["quiet"]
            tot_qf += wr["quiet_fired"]
            if wr["fired"] > 0:
                faces_fired_gt0 += 1
            if wr["fired"] >= 2 and wr["quiet_fired"] == 0:
                faces_clean_pass += 1
            if wr["apart"] is not None:
                apart_vals.append(float(wr["apart"]))
        behind_faces[fb["key"]] = {
            "layer": fb["layer"],
            "own_ending_rel": ev_r["own_rel"],
            "own_holds_gt_0.5": own_ok,
            "result": wr,
            "pattern": gemma_patterns[fb["key"]],
            "synthetic_fire": ev_r["fire"],
            "synthetic_quiet": ev_r["quiet"],
        }

    behind_summary = {
        "faces_attempted": len(behind_prep),
        "faces_own_holds_gt_0.5": valid_behind_faces,
        "written_texts": tot_fire,
        "written_fired": tot_fired,
        "written_fire_rate": round(tot_fired / max(tot_fire, 1), 4),
        "beat_corpus": tot_beat,
        "near_misses": tot_q,
        "near_misses_fired": tot_qf,
        "near_miss_false_rate": round(tot_qf / max(tot_q, 1), 4),
        "faces_with_fired_gt0": faces_fired_gt0,
        "faces_clean_2of3_and_0quiet": faces_clean_pass,
        "mean_apart": round(float(np.mean(apart_vals)) if apart_vals else 0.0, 4),
        "median_apart": round(float(np.median(apart_vals)) if apart_vals else 0.0, 4),
        "elapsed_s": round(time.time() - t_beh, 2),
    }
    print(
        f"Behind counterfactual ({behind_summary['elapsed_s']:.1f}s): own_ok={valid_behind_faces}/{len(behind_prep)}; "
        f"fired={tot_fired}/{tot_fire} ({behind_summary['written_fire_rate']:.1%}); "
        f"quiet_fired={tot_qf}/{tot_q}; mean_apart={behind_summary['mean_apart']:.3f}",
        flush=True,
    )

    t1 = time.time()
    prov = gp.build_provenance(
        REPO_ROOT,
        [__file__, REPO_ROOT / "scripts/in_use_common.py", REPO_ROOT / "backend/pattern_tests.py"],
        started_at=t0_iso,
        finished_at=gp.utc_iso(t1),
        model_id=rt.model_id,
        weights=weights_card(rt),
        neuron_ids={
            "group_a": list(pats_json.keys()),
            "group_b": [fb["key"] for fb in group_b_faces],
        },
    )
    out = {
        **prov,
        "elapsed_s": round(t1 - t0, 2),
        "judge_assistant_prefix": best_apref,
        "arm_a1_claude_pat_gemma_judge": arm_a1,
        "arm_a2_gemma_pat_gemma_judge": arm_a2,
        "arm_b_fresh_group_b": arm_b,
        "group_b_behind_summary": behind_summary,
        "gemma_patterns": gemma_patterns,
        "group_b_behind_details": behind_faces,
    }
    out_path = out_dir / "probe3_local_pattern.json"
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} in {t1 - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
