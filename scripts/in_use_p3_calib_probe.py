#!/usr/bin/env python3
"""Diagnostic probe for Path 3 (in_use_pattern) a/b judge prompt calibration on L2/L14.

Tests why median_p_ab_mass dropped to 0.8687 (L2) and 0.8378 (L14) vs 0.9995 (L20)
and 0.9981-0.9997 in Pilot 3, using saved trial patterns from L{2,14}_details.json
and GPU-verified doc_h1 near-miss pairs. Never modifies locked prereg files and
never stores raw corpus snippets (a_txt, b_txt, kept_tail, fit_snippets) on disk.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

_ALLOC_CONF_NAMES = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")
if not any(os.environ.get(k) for k in _ALLOC_CONF_NAMES):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import atlas as at_mod  # noqa: E402
from backend import atlas_index as ai  # noqa: E402
from backend import gap_provenance as gp  # noqa: E402
from backend import pattern_tests as pt  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend.causal_check import language_model, logits_fp32, model_device  # noqa: E402
from backend.runtime_factory import make_runtime, weights_card  # noqa: E402
from scripts.in_use_common import collect_face_events, load_passages_and_doc_halves  # noqa: E402
from scripts.in_use_context_cpu_probe import _special_token_ids  # noqa: E402
from scripts.in_use_pattern_pilot import (  # noqa: E402
    SINGLE_PAIR_SYSTEM,
    _read_multi_neurons_to_layer,
    _single_pair_prompt,
    _snippet_span_text,
    _snippet_text,
    _window_ids,
)
from scripts.in_use_pattern_trial import (  # noqa: E402
    _load_frozen_faces,
    _select_doc_events,
    build_doc_h1_token_index,
    k3_derangement_offsets,
)


def _strict_pair_prompt(pattern: str, a_txt: str, b_txt: str) -> str:
    return (
        "The description of when the neuron fires:\n\n"
        f"{pattern.strip()}\n\n"
        "Pair 1\n"
        f"a) {a_txt}\n"
        f"b) {b_txt}\n\n"
        "Reply with ONLY the single letter a or b (no other words).\n\n"
        "Which snippet fires the neuron (a or b)?"
    )


CALIBRATION_VARIANTS: Tuple[Tuple[str, str, Callable[[str, str, str], str], str], ...] = (
    ("C0_baseline_empty", SINGLE_PAIR_SYSTEM, _single_pair_prompt, ""),
    ("C1_apref_answer", SINGLE_PAIR_SYSTEM, _single_pair_prompt, "Answer: "),
    ("C2_apref_snippet", SINGLE_PAIR_SYSTEM, _single_pair_prompt, "Snippet "),
    ("C3_apref_choice", SINGLE_PAIR_SYSTEM, _single_pair_prompt, "Choice: "),
    ("C4_strict_empty", SINGLE_PAIR_SYSTEM, _strict_pair_prompt, ""),
    ("C5_strict_answer", SINGLE_PAIR_SYSTEM, _strict_pair_prompt, "Answer: "),
)


def summarize_variant_scores(
    items: Sequence[Dict[str, Any]],
    own_scores: Sequence[Dict[str, float]],
    other_scores_k: Sequence[Sequence[Dict[str, float]]],
    *,
    top_non_ab: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """Summarize p_ab calibration and blind_near accuracy across own + K=3 nulls."""
    K = len(other_scores_k)
    by_face: Dict[str, Dict[str, Any]] = {}
    own_masses: List[float] = []
    all_masses: List[float] = []
    own_right = 0
    oth_rights = [0] * K
    tot = len(items)
    by_match: Dict[str, List[float]] = {}

    for idx, it in enumerate(items):
        fk = it["face_key"]
        tr = it["truth"]
        so = own_scores[idx]
        fr = by_face.setdefault(fk, {"truth": [], "own": [], "match": []})
        fr["truth"].append(tr)
        fr["own"].append(so["pick"])
        fr["match"].append(it["match"])
        own_right += int(so["pick"] == tr)
        own_masses.append(float(so["p_ab"]))
        all_masses.append(float(so["p_ab"]))
        by_match.setdefault(it["match"], []).append(float(so["p_ab"]))
        for ki in range(K):
            sx = other_scores_k[ki][idx]
            oth_rights[ki] += int(sx["pick"] == tr)
            all_masses.append(float(sx["p_ab"]))

    faces_ge8 = 0
    faces_p05 = 0
    for fr in by_face.values():
        if len(fr["truth"]) >= 8:
            faces_ge8 += 1
            if pt.blind_result(fr["own"], fr["truth"], fr["match"])["p"] < 0.05:
                faces_p05 += 1

    arr = np.asarray(all_masses, dtype=np.float64)
    own_arr = np.asarray(own_masses, dtype=np.float64)
    own_share = own_right / max(tot, 1)
    oth_shares = [c / max(tot, 1) for c in oth_rights]
    mean_oth = float(np.mean(oth_shares)) if oth_shares else 0.5
    return {
        "pairs": tot,
        "faces": len(by_face),
        "faces_ge8": faces_ge8,
        "faces_p05_own_ge8": faces_p05,
        "frac_p05_own_ge8": round(faces_p05 / max(faces_ge8, 1), 4),
        "own_share": round(own_share, 4),
        "other_shares": [round(x, 4) for x in oth_shares],
        "mean_other_share": round(mean_oth, 4),
        "own_minus_mean_other": round(own_share - mean_oth, 4),
        "median_p_ab_mass": round(float(np.median(arr)) if len(arr) else 0.0, 4),
        "own_median_p_ab_mass": round(float(np.median(own_arr)) if len(own_arr) else 0.0, 4),
        "p10_p_ab_mass": round(float(np.percentile(arr, 10)) if len(arr) else 0.0, 4),
        "p25_p_ab_mass": round(float(np.percentile(arr, 25)) if len(arr) else 0.0, 4),
        "frac_p_ab_ge_090": round(float(np.mean(arr >= 0.90)) if len(arr) else 0.0, 4),
        "frac_p_ab_ge_099": round(float(np.mean(arr >= 0.99)) if len(arr) else 0.0, 4),
        "own_median_p_ab_by_match": {
            lv: round(float(np.median(vs)), 4) for lv, vs in sorted(by_match.items())
        },
        "top_non_ab_tokens": list(top_non_ab),
    }


@torch.no_grad()
def score_ab_prefills_diag(
    rt: Any,
    lm: Any,
    tok: Any,
    prompts_ids: Sequence[Sequence[int]],
    ab_ids: Dict[str, List[int]],
    *,
    batch_size: int = 16,
    top_k_non_ab: int = 8,
) -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
    """Score single-prefill P(a) vs P(b) and collect top competing non-a/b tokens."""
    dev = model_device(lm)
    W_U = rt.lm_head_weight
    a_t = torch.tensor(ab_ids["a"], dtype=torch.long, device=dev)
    b_t = torch.tensor(ab_ids["b"], dtype=torch.long, device=dev)
    ab_all = torch.cat([a_t, b_t], dim=0)
    V = int(W_U.shape[0])
    non_ab_sum = torch.zeros(V, dtype=torch.float64)
    low_mass_sum = torch.zeros(V, dtype=torch.float64)
    n_low = 0
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
        lsm = F.log_softmax(logits_fp32(h, W_U), dim=-1)
        probs = lsm.exp()
        pa = probs[:, a_t].sum(dim=-1).cpu().numpy()
        pb = probs[:, b_t].sum(dim=-1).cpu().numpy()
        log_pa = torch.logsumexp(lsm[:, a_t], dim=-1).cpu().numpy()
        log_pb = torch.logsumexp(lsm[:, b_t], dim=-1).cpu().numpy()

        probs_other = probs.clone()
        probs_other[:, ab_all] = 0.0
        probs_other_cpu = probs_other.detach().cpu().to(torch.float64)
        non_ab_sum += probs_other_cpu.sum(dim=0)

        for i in range(e - s):
            p_ab = float(pa[i] + pb[i])
            if p_ab < 0.90:
                low_mass_sum += probs_other_cpu[i]
                n_low += 1
            margin = float(log_pa[i] - log_pb[i])
            out.append(
                {
                    "pick": "a" if margin >= 0.0 else "b",
                    "margin": round(margin, 4),
                    "p_ab": round(p_ab, 4),
                }
            )
        del res, h, lsm, probs, probs_other, probs_other_cpu, inp, att, pos
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    n_tot = max(len(prompts_ids), 1)
    src = low_mass_sum if n_low > 0 else non_ab_sum
    denom = max(n_low, 1) if n_low > 0 else n_tot
    vals, idxs = torch.topk(src, k=min(int(top_k_non_ab), V))
    top_tokens = [
        {
            "token_id": int(tid),
            "token": repr(tok.decode([int(tid)])),
            "mean_prob_when_lt_090" if n_low > 0 else "mean_prob_all": round(float(v) / denom, 5),
        }
        for v, tid in zip(vals.tolist(), idxs.tolist())
        if float(v) > 0.0
    ]
    return out, top_tokens


def _run_cpu_selftest() -> None:
    items = [
        {"face_key": f"L2-n{i}p", "truth": "a" if i % 2 == 0 else "b", "match": "phrase"}
        for i in range(10)
    ]
    own_sc = [{"pick": it["truth"], "p_ab": 0.995} for it in items]
    oth_sc = [[{"pick": "a", "p_ab": 0.992} for _ in items] for _ in range(3)]
    res = summarize_variant_scores(items, own_sc, oth_sc, top_non_ab=[])
    assert res["median_p_ab_mass"] >= 0.99
    assert res["own_share"] == 1.0
    assert res["faces_p05_own_ge8"] == 0
    s = _strict_pair_prompt("Pattern.", "snip a", "snip b")
    assert "Reply with ONLY the single letter a or b" in s
    print("PASS in_use_p3_calib_probe CPU selftest.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Path 3 (in_use_pattern) a/b prompt calibration diagnostic probe")
    ap.add_argument("--layers", default="2,14")
    ap.add_argument("--n-faces", type=int, default=32, help="Sampled faces per layer (0 = all frozen faces)")
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--prereg-dir", default="experiments/results/in_use_pattern_trial/prereg")
    ap.add_argument("--details-dir", default="/home/eric/bin/ai/theodolite2/atlas_data/census/trials/in_use_pattern")
    ap.add_argument("--out", default="/home/eric/bin/ai/theodolite2/atlas_data/census/trials/in_use_pattern/p3_calib_probe.json")
    ap.add_argument("--prefill-batch", type=int, default=16)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--cpu-selftest", action="store_true")
    args = ap.parse_args()

    if args.cpu_selftest:
        _run_cpu_selftest()
        return

    t0 = time.time()
    t0_iso = gp.utc_iso(t0)
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    prereg_dir = Path(args.prereg_dir) if Path(args.prereg_dir).is_absolute() else REPO_ROOT / args.prereg_dir
    details_dir = Path(args.details_dir)
    out_path = Path(args.out)

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else ai.atlas_dir()
    loaded = load_passages_and_doc_halves(data_dir)
    P, act_scale, windows = loaded["P"], loaded["act_scale"], loaded["windows"]
    doc_id, half_id = loaded["doc_id"], loaded["half_id"]
    pure_h0_mask, pure_h1_mask = loaded["pure_h0_mask"], loaded["pure_h1_mask"]
    bos_id, eos_id = loaded["bos_id"], loaded["eos_id"]

    rt = make_runtime(load_jlens=False).load()
    if getattr(rt, "jpp_lenses", None):
        rt.jpp_lenses.clear()
    if getattr(rt, "eager_model", None) is not None and hasattr(rt.eager_model, "_lens_jacobians"):
        rt.eager_model._lens_jacobians.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    w4a16_fast.install(rt, "compiled", log=print)
    lm = language_model(rt.hf_model)
    tok = rt.tokenizer
    marks = at_mod.passage_marks(tok)
    special_ids = _special_token_ids(bos_id, eos_id)
    h1_idx = build_doc_h1_token_index(windows, half_id, doc_id, special_ids, min_at=8, min_after=8)
    h1_pool = h1_idx["h1_wins"].tolist()

    ab_ids = {
        "a": [int(tok.encode(s, add_special_tokens=False)[0]) for s in ("a", " a", "A", " A")],
        "b": [int(tok.encode(s, add_special_tokens=False)[0]) for s in ("b", " b", "B", " B")],
    }
    rng_np = np.random.default_rng(2026112603)

    layer_results: Dict[str, Any] = {}
    for l in layers:
        frozen = _load_frozen_faces(prereg_dir / f"faces_L{l}.txt")
        details = json.loads((details_dir / f"L{l}_details.json").read_text(encoding="utf-8"))
        saved_pats = {k: str(v["pattern"]) for k, v in details["behind_details"].items()}

        N_all = len(frozen)
        all_keys = [f"L{l}-n{nid}{'p' if fc == 'pos' else 'm'}" for nid, fc in frozen]
        o1, o2, o3 = k3_derangement_offsets(N_all)
        offsets_k = (o1, o2, o3)

        if int(args.n_faces) > 0 and int(args.n_faces) < N_all:
            sel_idx = sorted(int(i) for i in np.random.default_rng(int(args.seed) + l).choice(N_all, size=int(args.n_faces), replace=False))
        else:
            sel_idx = list(range(N_all))

        faces_data: List[Dict[str, Any]] = []
        for f_idx in sel_idx:
            nid, fc = frozen[f_idx]
            ev = collect_face_events(
                P, windows, act_scale, pure_h0_mask, pure_h1_mask, doc_id, half_id, l, nid, fc, eos_id=eos_id, bos_id=bos_id
            )
            fit_evs, test_evs, behind_ev = _select_doc_events(ev, n_fit=8, n_test=10)
            own_docs = {int(e["doc_id"]) for e in fit_evs + test_evs} | {int(behind_ev["doc_id"])}
            faces_data.append(
                {
                    "f_idx": f_idx,
                    "key": all_keys[f_idx],
                    "layer": int(l),
                    "n": int(nid),
                    "face": fc,
                    "extreme": float(ev["extreme"]),
                    "own_windows": ev["own_windows"],
                    "own_docs": own_docs,
                    "test_evs": test_evs,
                }
            )

        pair_slots: List[Dict[str, Any]] = []
        bounds = h1_idx["bounds"]
        for fd in faces_data:
            for ev in fd["test_evs"]:
                ids = _window_ids(windows, ev["win"])
                at = int(ev["at"])
                fire_txt, s_f, e_f = _snippet_text(tok, marks, ids, at, before=60, after=8)
                before_f, after_f = at - s_f, e_f - at - 1
                tok0 = int(ids[at])
                prev = int(ids[at - 1]) if at >= 1 else None
                nxt = int(ids[at + 1]) if at + 1 < len(ids) else None
                cand_q: List[Tuple[int, int, str, int, int, List[int], str]] = []
                seen_q: Set[Tuple[int, int]] = set()
                if tok0 in bounds:
                    lo, hi = bounds[tok0]
                    ws_all, ats_all, docs_all = h1_idx["w"][lo:hi], h1_idx["a"][lo:hi], h1_idx["d"][lo:hi]
                    keep = ~np.isin(docs_all, list(fd["own_docs"]))
                    if fd["own_windows"]:
                        keep &= ~np.isin(ws_all, list(fd["own_windows"]))
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
                            near_hits: List[Tuple[float, int, int, int, int, List[int], str]] = []
                            for h_j in hits[:48]:
                                wq, aq = int(ws_all[h_j]), int(ats_all[h_j])
                                if (wq, aq) in seen_q:
                                    continue
                                qids = _window_ids(windows, wq)
                                sq, eq, _, _ = at_mod.cut_passage(qids, aq, before_f, after_f, marks)
                                sq, eq = at_mod.whole_characters(tok, qids, sq, eq, marks)
                                if abs((aq - sq) - before_f) > 1 or abs((eq - aq - 1) - after_f) > 1:
                                    continue
                                q_txt = _snippet_span_text(tok, qids[:eq], sq, eq, aq)
                                ratio = len(q_txt) / max(len(fire_txt), 1)
                                seen_q.add((wq, aq))
                                near_hits.append((abs(ratio - 1.0), wq, aq, sq, eq, qids[:eq], q_txt))
                            near_hits.sort(key=lambda x: x[0])
                            for _, wq, aq, sq, eq, qseq, q_txt in near_hits[: pt.NEAR_TRIES]:
                                cand_q.append((wq, aq, lv, sq, eq, qseq, q_txt))
                for _ in range(12):
                    wq = int(h1_pool[int(rng_np.integers(0, len(h1_pool)))])
                    if wq in fd["own_windows"]:
                        continue
                    qids = _window_ids(windows, wq)
                    if len(qids) <= max(before_f + 4, 16) + after_f:
                        continue
                    aq = int(rng_np.integers(max(8, before_f + 1), len(qids) - max(1, after_f)))
                    if int(half_id[wq, aq]) != 1 or int(doc_id[wq, aq]) in fd["own_docs"] or int(qids[aq]) in marks or (wq, aq) in seen_q:
                        continue
                    sq, eq, _, _ = at_mod.cut_passage(qids, aq, before_f, after_f, marks)
                    sq, eq = at_mod.whole_characters(tok, qids, sq, eq, marks)
                    if abs((aq - sq) - before_f) > 1 or abs((eq - aq - 1) - after_f) > 1:
                        continue
                    seen_q.add((wq, aq))
                    q_txt = _snippet_span_text(tok, qids[:eq], sq, eq, aq)
                    cand_q.append((wq, aq, pt.ANYWHERE, sq, eq, qids[:eq], q_txt))
                pair_slots.append(
                    {
                        "f_idx": fd["f_idx"],
                        "fd": fd,
                        "fire_txt": fire_txt,
                        "cands": cand_q,
                        "quiet": None,
                        "first_is_fire": bool(rng_np.random() < 0.5),
                    }
                )

        max_rounds = max((len(ps["cands"]) for ps in pair_slots), default=0)
        for rnd in range(max_rounds):
            todo = [ps for ps in pair_slots if ps["quiet"] is None and rnd < len(ps["cands"])]
            if not todo:
                break
            acts_r = _read_multi_neurons_to_layer(
                lm, [ps["cands"][rnd][5] for ps in todo], [(l, ps["fd"]["n"]) for ps in todo], batch_size=16
            )
            for ps, act_v in zip(todo, acts_r):
                wq, aq, lv, sq, eq, _, q_txt = ps["cands"][rnd]
                fd = ps["fd"]
                sgn = 1.0 if fd["face"] == "pos" else -1.0
                if float(np.max((sgn * act_v[sq:eq]) / max(fd["extreme"], 1e-6))) < pt.QUIET_BELOW:
                    ps["quiet"] = {"win": wq, "at": aq, "match": lv, "txt": q_txt}

        items_b: List[Dict[str, Any]] = []
        for ps in pair_slots:
            if ps["quiet"] is None:
                continue
            fi = ps["f_idx"]
            tr = "a" if ps["first_is_fire"] else "b"
            a_txt, b_txt = (ps["fire_txt"], ps["quiet"]["txt"]) if tr == "a" else (ps["quiet"]["txt"], ps["fire_txt"])
            items_b.append(
                {
                    "face_key": all_keys[fi],
                    "other_keys": [all_keys[(fi + off) % N_all] for off in offsets_k],
                    "truth": tr,
                    "match": ps["quiet"]["match"],
                    "a_txt": a_txt,
                    "b_txt": b_txt,
                }
            )
        print(f"L{l}: {len(faces_data)} faces, {len(items_b)} GPU-verified pairs", flush=True)

        variant_summaries: Dict[str, Any] = {}
        for v_name, sys_txt, pmt_fn, apref in CALIBRATION_VARIANTS:
            def _enc(pat_fn: Any) -> List[List[int]]:
                return [
                    rt.tokenize_prompt(
                        rt.compose_prompt(
                            pmt_fn(saved_pats[pat_fn(it)], it["a_txt"], it["b_txt"]),
                            system=sys_txt,
                            chat=True,
                            thinking=False,
                            assistant_prefix=apref,
                        )
                    )
                    for it in items_b
                ]

            own_enc = _enc(lambda it: it["face_key"])
            own_sc, top_non_ab = score_ab_prefills_diag(rt, lm, tok, own_enc, ab_ids, batch_size=int(args.prefill_batch))
            oth_sc_k = [
                score_ab_prefills_diag(rt, lm, tok, _enc(lambda it, ki_=ki: it["other_keys"][ki_]), ab_ids, batch_size=int(args.prefill_batch))[0]
                for ki in range(3)
            ]
            vs = summarize_variant_scores(items_b, own_sc, oth_sc_k, top_non_ab=top_non_ab)
            if v_name == "C0_baseline_empty" and len(own_enc) >= 32:
                unpad_sc, _ = score_ab_prefills_diag(rt, lm, tok, own_enc[:32], ab_ids, batch_size=1)
                vs["unpadded_bs1_32_median_p_ab"] = round(float(np.median([r["p_ab"] for r in unpad_sc])), 4)
                vs["batched_bs16_32_median_p_ab"] = round(float(np.median([r["p_ab"] for r in own_sc[:32]])), 4)
            variant_summaries[v_name] = vs
            print(
                f"L{l} {v_name}: med_p_ab={vs['median_p_ab_mass']} (p10={vs['p10_p_ab_mass']}, >=0.99={vs['frac_p_ab_ge_099']}) "
                f"own={vs['own_share']} vs other={vs['mean_other_share']} top_non_ab={vs['top_non_ab_tokens'][:3]}",
                flush=True,
            )
        layer_results[str(l)] = variant_summaries

    t1 = time.time()
    prov = gp.build_provenance(
        REPO_ROOT,
        [__file__, REPO_ROOT / "scripts/in_use_pattern_trial.py", REPO_ROOT / "scripts/in_use_pattern_pilot.py"],
        started_at=t0_iso,
        finished_at=gp.utc_iso(t1),
        model_id=rt.model_id,
        weights=weights_card(rt),
    )
    out = {
        **prov,
        "elapsed_s": round(t1 - t0, 2),
        "n_faces_per_layer": int(args.n_faces),
        "seed": int(args.seed),
        "ab_ids": ab_ids,
        "layers": layer_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} in {t1 - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
