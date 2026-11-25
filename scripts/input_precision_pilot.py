#!/usr/bin/env python3
"""Exploratory held-out document corpus-precision pilot for L0-L29 input triggers.

Splits passages.d windows strictly by <eos> document halves (half 0 = selection, half 1 = unseen
evaluation using ONLY windows containing zero half-0 tokens), picks 256 faces per target layer
with >=2 distinct half-0 docs for their rank-1 causal (-1, 0) trigger, pairs each with a
frequency-matched control piece at the same offset, and streams unseen half-1 windows once through
layers 0..max(layers) on GPU to measure precision: P(signed |act| >= 0.5 * extreme | trigger).
Never stores raw corpus passage text.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import atlas_index as ai  # noqa: E402
from backend import gap_provenance as gp  # noqa: E402
from backend import input_census as ic  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend.causal_check import language_model, model_device  # noqa: E402
from backend.gap_stats import mannwhitney_greater, wilcoxon_greater  # noqa: E402
from backend.runtime_factory import make_runtime, weights_card  # noqa: E402


class _StopAfterLayer(Exception):
    pass


def _select_half0_faces(
    l: int,
    windows: np.ndarray,
    unigram: np.ndarray,
    P: Any,
    doc_id: np.ndarray,
    half_id: np.ndarray,
    n_faces: int,
    eos_id: int = 1,
    bos_id: int = 2,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Select causal (-1, 0) rank-1 triggers using ONLY half-0 documents."""
    N_ff = int(P["pos_win"].shape[1])
    W_tot, seq = windows.shape
    unigram_f = np.asarray(unigram, dtype=np.float64)
    tot_tok = max(float(unigram_f.sum()), 1.0)
    bg_freq = unigram_f / tot_tok
    V = len(bg_freq)
    top100_set = set(int(x) for x in np.argsort(-unigram_f)[: min(100, V)].tolist())
    tok_bin, freq_bins = ic._build_freq_bins(unigram_f)

    rng = np.random.default_rng(20261125 + l)
    cands: List[Dict[str, Any]] = []
    for s_idx, face in ((0, "pos"), (1, "neg")):
        f_win = P[f"{face}_win"][l]
        f_at = P[f"{face}_at"][l]
        t0_all, tm1_all, _ = ic._extract_tokens_for_events(windows, f_win, f_at, eos_id=eos_id, bos_id=bos_id)
        for n in range(N_ff):
            t0_r, tm1_r = t0_all[n], tm1_all[n]
            h0_j = [
                j
                for j in range(f_win.shape[1])
                if 0 <= int(f_win[n, j]) < W_tot
                and 1 <= int(f_at[n, j]) < seq
                and int(t0_r[j]) >= 0
                and int(half_id[int(f_win[n, j]), int(f_at[n, j])]) == 0
            ]
            if len(h0_j) < 4:
                continue
            docs_by_key: Dict[int, set[int]] = collections.defaultdict(set)
            cnt_by_key: Dict[int, int] = collections.Counter()
            for j in h0_j:
                d_j = int(doc_id[int(f_win[n, j]), int(f_at[n, j])])
                k0 = ic.encode_key(0, int(t0_r[j]))
                docs_by_key[k0].add(d_j)
                cnt_by_key[k0] += 1
                if int(tm1_r[j]) >= 0:
                    km = ic.encode_key(-1, int(tm1_r[j]))
                    docs_by_key[km].add(d_j)
                    cnt_by_key[km] += 1
            best: Tuple[float, int, int, int] | None = None
            inv_n0 = 1.0 / len(h0_j)
            for key, cnt in cnt_by_key.items():
                if len(docs_by_key[key]) < 2 or cnt < 2:
                    continue
                off, tok = ic.decode_key(key)
                if tok in (0, eos_id, bos_id):
                    continue
                rate = cnt * inv_n0
                bg = max(float(bg_freq[tok]) if tok < V else 1e-6, 1e-6)
                lift = rate / bg
                sc = rate * math.log1p(lift)
                if best is None or sc > best[0]:
                    best = (sc, off, tok, len(docs_by_key[key]))
            if best is None:
                continue
            sc, off, tok, n_d = best
            bi = int(tok_bin[tok]) if 0 <= tok < len(tok_bin) else -1
            if bi < 0 or len(freq_bins[bi]) < 2:
                continue
            pool = freq_bins[bi]
            ctl_tok = int(pool[int(rng.integers(0, len(pool)))])
            tries = 0
            while (ctl_tok == tok or ctl_tok in (0, eos_id, bos_id)) and tries < 20:
                ctl_tok = int(pool[int(rng.integers(0, len(pool)))])
                tries += 1
            if ctl_tok == tok:
                continue
            cands.append(
                {
                    "layer": int(l),
                    "n": int(n),
                    "sign": 1 if s_idx == 0 else -1,
                    "face": "+" if s_idx == 0 else "-",
                    "off": int(off),
                    "tok": int(tok),
                    "ctl_tok": int(ctl_tok),
                    "h0_score": round(float(sc), 4),
                    "h0_docs": int(n_d),
                    "top100": int(tok) in top100_set,
                }
            )

    n_elig = len(cands)
    if n_elig == 0:
        return [], 0, 0
    pick_n = min(n_faces, n_elig)
    pick_idx = sorted(int(i) for i in rng.choice(n_elig, size=pick_n, replace=False))
    return [cands[i] for i in pick_idx], n_elig, sum(1 for c in cands if c["top100"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Exploratory held-out document corpus-precision pilot for L0-L29 triggers")
    ap.add_argument("--layers", default="2,14,20,29")
    ap.add_argument("--n-faces", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-dir", default="experiments/results/input_census_L0_L29")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    t0_iso = gp.utc_iso(t0)
    layers = sorted({int(x.strip()) for x in args.layers.split(",") if x.strip()})
    max_layer = max(layers)
    out_dir = REPO_ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else ai.atlas_dir()

    P = ai.load_folder(data_dir / "index/passages.d/meta.json")
    act_scale = np.load(data_dir / "index/act_scale.npz")
    vocab = json.loads((data_dir / "index/vocab.json").read_text(encoding="utf-8"))["vocab"]
    windows = np.asarray(P["windows"])
    unigram = np.asarray(P["unigram"])
    W_tot, seq_len = windows.shape
    eos_id = 1
    bos_id = int(windows[0, 0]) if windows[0, 0] >= 0 else 2

    doc_id, half_id, n_docs = ic.build_doc_halves(windows, eos_id=eos_id, seed=20261124)
    pure_h1_wins = np.flatnonzero(np.all(half_id == 1, axis=1) & (np.sum(windows >= 0, axis=1) >= 4))
    n_pure_h1 = int(len(pure_h1_wins))
    print(f"passages.d: W={W_tot} n_docs={n_docs} pure_half1_windows={n_pure_h1}", flush=True)

    faces_by_layer: Dict[int, List[Dict[str, Any]]] = {}
    elig_counts: Dict[int, Tuple[int, int]] = {}
    neuron_ids_meta: Dict[str, List[int]] = {}
    for l in layers:
        picked, n_elig, n_top100 = _select_half0_faces(l, windows, unigram, P, doc_id, half_id, args.n_faces, eos_id=eos_id, bos_id=bos_id)
        faces_by_layer[l] = picked
        elig_counts[l] = (n_elig, n_top100)
        neuron_ids_meta[str(l)] = [f["n"] for f in picked]
        print(f"  L{l:2d}: half0_elig={n_elig} (top100={n_top100}) sampled={len(picked)}", flush=True)

    # Pre-index pure_h1 positions per (off, tok) only for tokens appearing in sampled faces/controls
    needed_keys: set[Tuple[int, int]] = set()
    faces_by_key: Dict[Tuple[int, int, int], List[Tuple[int, bool]]] = collections.defaultdict(list)
    for l in layers:
        ext_pos = np.abs(act_scale["pos"][l]).clip(min=1e-3)
        ext_neg = np.abs(act_scale["neg"][l]).clip(min=1e-3)
        for fi, f in enumerate(faces_by_layer[l]):
            f["extreme"] = float(ext_pos[f["n"]] if f["sign"] > 0 else ext_neg[f["n"]])
            needed_keys.add((f["off"], f["tok"]))
            needed_keys.add((f["off"], f["ctl_tok"]))
            faces_by_key[(l, f["off"], f["tok"])].append((fi, True))
            faces_by_key[(l, f["off"], f["ctl_tok"])].append((fi, False))

    needed_t0 = {t for o, t in needed_keys if o == 0}
    needed_tm1 = {t for o, t in needed_keys if o == -1}
    layer_neurons_t: Dict[int, torch.Tensor] = {}
    layer_sign_t: Dict[int, np.ndarray] = {}
    layer_ext_t: Dict[int, np.ndarray] = {}

    rt = make_runtime(load_jlens=False).load()
    w4a16_fast.install(rt, "compiled", log=print)
    lm = language_model(rt.hf_model)
    dev = model_device(lm)

    for l in layers:
        layer_neurons_t[l] = torch.tensor([f["n"] for f in faces_by_layer[l]], dtype=torch.long, device=dev)
        layer_sign_t[l] = np.array([f["sign"] for f in faces_by_layer[l]], dtype=np.float32)
        layer_ext_t[l] = np.array([f["extreme"] for f in faces_by_layer[l]], dtype=np.float32)

    occ_trig = {l: np.zeros(len(faces_by_layer[l]), dtype=np.int64) for l in layers}
    hit50_trig = {l: np.zeros(len(faces_by_layer[l]), dtype=np.int64) for l in layers}
    hit25_trig = {l: np.zeros(len(faces_by_layer[l]), dtype=np.int64) for l in layers}
    occ_ctl = {l: np.zeros(len(faces_by_layer[l]), dtype=np.int64) for l in layers}
    hit50_ctl = {l: np.zeros(len(faces_by_layer[l]), dtype=np.int64) for l in layers}
    hit25_ctl = {l: np.zeros(len(faces_by_layer[l]), dtype=np.int64) for l in layers}

    # Build sparse lookup per pure_h1 window BEFORE GPU pass (~0.2s)
    # win_key_pos[wi] -> dict of (off, tok) -> list of positions inside compressed sequence (including bos at 0)
    win_seqs: List[List[int]] = []
    win_pos_by_key: List[Dict[Tuple[int, int], List[int]]] = []
    for w0 in pure_h1_wins:
        row = windows[int(w0)]
        valid_pos = [pos for pos in range(seq_len) if int(row[pos]) >= 0]
        seq = [int(row[pos]) for pos in valid_pos]
        win_seqs.append(seq)
        kmap: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)
        for ti in range(1, len(seq)):
            t_cur = seq[ti]
            if t_cur in needed_t0 and t_cur not in (bos_id, eos_id):
                kmap[(0, t_cur)].append(ti)
            if ti >= 2:
                t_prev = seq[ti - 1]
                if t_prev in needed_tm1 and t_prev not in (bos_id, eos_id) and t_cur not in (bos_id, eos_id):
                    kmap[(-1, t_prev)].append(ti)
        win_pos_by_key.append(kmap)

    n_wins = len(win_seqs)
    t_gpu0 = time.time()
    for s in range(0, n_wins, args.batch_size):
        e = min(n_wins, s + args.batch_size)
        b_seqs = win_seqs[s:e]
        b = len(b_seqs)
        max_len = max(len(x) for x in b_seqs)
        inp_np = np.zeros((b, max_len), dtype=np.int64)
        att_np = np.zeros((b, max_len), dtype=np.int64)
        for bi, x in enumerate(b_seqs):
            inp_np[bi, : len(x)] = x
            att_np[bi, : len(x)] = 1
        inp = torch.from_numpy(inp_np).to(dev)
        att = torch.from_numpy(att_np).to(dev)

        cap: Dict[int, torch.Tensor] = {}
        hks = []
        for l in layers:
            n_idx = layer_neurons_t[l]

            def make_pre(ll: int, idx: torch.Tensor):
                def _hk(_m: Any, f_args: Tuple[torch.Tensor, ...]) -> None:
                    cap[ll] = f_args[0].index_select(-1, idx).detach()

                return _hk

            hks.append(lm.layers[l].mlp.down_proj.register_forward_pre_hook(make_pre(l, n_idx)))

        def _stop(_m: Any, _a: Any, _o: Any) -> None:
            raise _StopAfterLayer()

        hks.append(lm.layers[max_layer].register_forward_hook(_stop))
        try:
            with torch.no_grad():
                try:
                    lm(input_ids=inp, attention_mask=att, use_cache=False)
                except _StopAfterLayer:
                    pass
        finally:
            for hk in hks:
                hk.remove()

        for l in layers:
            acts_l = cap[l].float().cpu().numpy()  # [b, T, n_faces]
            signs = layer_sign_t[l]
            exts = layer_ext_t[l]
            for bi in range(b):
                kmap = win_pos_by_key[s + bi]
                for (off, tok), pos_list in kmap.items():
                    flist = faces_by_key.get((l, off, tok))
                    if not flist or not pos_list:
                        continue
                    pos_arr = np.asarray(pos_list, dtype=np.int64)
                    m_occ = len(pos_arr)
                    f_idxs = np.asarray([x[0] for x in flist], dtype=np.int64)
                    vals = acts_l[bi][np.ix_(pos_arr, f_idxs)]  # [m_occ, m_faces]
                    signed_rel = (vals * signs[f_idxs][None, :]) / exts[f_idxs][None, :]
                    c50 = np.sum(signed_rel >= 0.5, axis=0)
                    c25 = np.sum(signed_rel >= 0.25, axis=0)
                    for k_loc, (fi, is_trig) in enumerate(flist):
                        if is_trig:
                            occ_trig[l][fi] += m_occ
                            hit50_trig[l][fi] += int(c50[k_loc])
                            hit25_trig[l][fi] += int(c25[k_loc])
                        else:
                            occ_ctl[l][fi] += m_occ
                            hit50_ctl[l][fi] += int(c50[k_loc])
                            hit25_ctl[l][fi] += int(c25[k_loc])
        del cap, inp, att
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if ((s // args.batch_size) + 1) % 25 == 0 or e == n_wins:
            print(f"  streamed {e}/{n_wins} unseen-doc windows ({time.time() - t_gpu0:.1f}s)", flush=True)

    layer_reports: List[Dict[str, Any]] = []
    examples: Dict[str, List[Dict[str, Any]]] = {}
    for l in layers:
        picked = faces_by_layer[l]
        ot, h50t, h25t = occ_trig[l], hit50_trig[l], hit25_trig[l]
        oc, h50c, h25c = occ_ctl[l], hit50_ctl[l], hit25_ctl[l]
        both = (ot >= 1) & (oc >= 1)
        p_t = np.where(ot > 0, h50t / np.maximum(ot, 1), 0.0)[both]
        p_c = np.where(oc > 0, h50c / np.maximum(oc, 1), 0.0)[both]
        p25_t = np.where(ot > 0, h25t / np.maximum(ot, 1), 0.0)[both]
        p25_c = np.where(oc > 0, h25c / np.maximum(oc, 1), 0.0)[both]
        is_t100 = np.array([f["top100"] for f in picked], dtype=bool)[both]

        auc_p, _ = mannwhitney_greater(p_t, p_c) if len(p_t) else (0.5, 1.0)
        wil_p = wilcoxon_greater(p_t, p_c) if len(p_t) else 1.0
        pooled_t = float(np.sum(h50t[both]) / max(int(np.sum(ot[both])), 1))
        pooled_c = float(np.sum(h50c[both]) / max(int(np.sum(oc[both])), 1))
        lift_ratio = np.where(p_c > 0, p_t / np.maximum(p_c, 1e-9), np.where(p_t > 0, np.inf, 0.0))
        pass_20_3x = float(np.mean((p_t >= 0.20) & (lift_ratio >= 3.0))) if len(p_t) else 0.0

        def _sub_summary(mask_s: np.ndarray) -> Dict[str, Any]:
            if not np.any(mask_s):
                return {"n": 0, "trig_p50": None, "trig_mean": None, "ctl_mean": None}
            return {
                "n": int(np.sum(mask_s)),
                "trig_p50": round(float(np.median(p_t[mask_s])), 4),
                "trig_p90": round(float(np.percentile(p_t[mask_s], 90)), 4),
                "trig_mean": round(float(np.mean(p_t[mask_s])), 4),
                "ctl_mean": round(float(np.mean(p_c[mask_s])), 4),
                "frac_prec20_lift3x": round(float(np.mean((p_t[mask_s] >= 0.20) & (lift_ratio[mask_s] >= 3.0))), 4),
            }

        n_elig, n_top100 = elig_counts[l]
        rep = {
            "layer": int(l),
            "half0_eligible_faces": int(n_elig),
            "half0_top100_share": round(n_top100 / max(n_elig, 1), 4),
            "sampled_faces": len(picked),
            "evaluated_with_occ_in_half1": int(np.sum(both)),
            "occ_trig_total": int(np.sum(ot[both])),
            "occ_ctl_total": int(np.sum(oc[both])),
            "pooled_prec50_trig": round(pooled_t, 5),
            "pooled_prec50_ctl": round(pooled_c, 5),
            "face_prec50_trig_p50": round(float(np.median(p_t)), 4) if len(p_t) else None,
            "face_prec50_trig_mean": round(float(np.mean(p_t)), 4) if len(p_t) else None,
            "face_prec50_trig_p90": round(float(np.percentile(p_t, 90)), 4) if len(p_t) else None,
            "face_prec50_ctl_mean": round(float(np.mean(p_c)), 4) if len(p_c) else None,
            "face_prec25_trig_mean": round(float(np.mean(p25_t)), 4) if len(p25_t) else None,
            "face_prec25_ctl_mean": round(float(np.mean(p25_c)), 4) if len(p25_c) else None,
            "frac_prec20_and_lift3x": round(pass_20_3x, 4),
            "auc_vs_freq_ctl": round(float(auc_p), 4),
            "wilcoxon_p": float(f"{wil_p:.3e}"),
            "non_top100_triggers": _sub_summary(~is_t100),
            "top100_triggers": _sub_summary(is_t100),
        }
        layer_reports.append(rep)

        both_idx = np.flatnonzero(both)
        top_ex = sorted(both_idx, key=lambda idx: -(float(h50t[idx] / max(ot[idx], 1)) - float(h50c[idx] / max(oc[idx], 1))))[:8]
        examples[str(l)] = [
            {
                "n": picked[i]["n"],
                "face": picked[i]["face"],
                "off": picked[i]["off"],
                "tok": vocab[picked[i]["tok"]] if 0 <= picked[i]["tok"] < len(vocab) else str(picked[i]["tok"]),
                "ctl_tok": vocab[picked[i]["ctl_tok"]] if 0 <= picked[i]["ctl_tok"] < len(vocab) else str(picked[i]["ctl_tok"]),
                "top100": picked[i]["top100"],
                "h50_trig": int(h50t[i]),
                "occ_trig": int(ot[i]),
                "prec50_trig": round(float(h50t[i] / max(ot[i], 1)), 4),
                "h50_ctl": int(h50c[i]),
                "occ_ctl": int(oc[i]),
                "prec50_ctl": round(float(h50c[i] / max(oc[i], 1)), 4),
            }
            for i in top_ex
        ]
        print(
            f"Precision L{l:2d}: eval={rep['evaluated_with_occ_in_half1']}/{len(picked)} "
            f"pooled={rep['pooled_prec50_trig']:.4f} vs ctl={rep['pooled_prec50_ctl']:.4f} "
            f"mean={rep['face_prec50_trig_mean']} (>=0.20 & 3x: {100*pass_20_3x:.1f}%) "
            f"non_t100_mean={rep['non_top100_triggers']['trig_mean']} t100_mean={rep['top100_triggers']['trig_mean']} "
            f"AUC={rep['auc_vs_freq_ctl']}",
            flush=True,
        )

    lines = [
        "Exploratory Held-Out Document Corpus Precision Pilot (Half 0 Pick -> Unseen Pure Half 1 Stream, |act| >= 0.5 * extreme)",
        "-" * 140,
        f"{'L':>3} {'Eval/Smp':>9} {'PoolTgt':>8} {'PoolCtl':>8} {'MeanTgt':>8} {'MeanCtl':>8} {'p50Tgt':>7} {'p90Tgt':>7} {'>=20%&3x':>8} {'NonT100':>8} {'Top100':>7} {'AUC':>7}",
    ]
    for r in layer_reports:
        nt = r["non_top100_triggers"].get("trig_mean")
        t1 = r["top100_triggers"].get("trig_mean")
        lines.append(
            f"{r['layer']:3d} {r['evaluated_with_occ_in_half1']:>4}/{r['sampled_faces']:<4} "
            f"{r['pooled_prec50_trig']:8.4f} {r['pooled_prec50_ctl']:8.4f} "
            f"{(r['face_prec50_trig_mean'] or 0):8.4f} {(r['face_prec50_ctl_mean'] or 0):8.4f} "
            f"{(r['face_prec50_trig_p50'] or 0):7.4f} {(r['face_prec50_trig_p90'] or 0):7.4f} "
            f"{r['frac_prec20_and_lift3x']:8.2%} "
            f"{(nt if nt is not None else 0):8.4f} {(t1 if t1 is not None else 0):7.4f} {r['auc_vs_freq_ctl']:7.4f}"
        )
    txt = "\n".join(lines) + "\n"
    prov = gp.build_provenance(
        REPO_ROOT,
        [Path(__file__), REPO_ROOT / "backend" / "input_census.py"],
        started_at=t0_iso,
        finished_at=gp.utc_iso(),
        model_id=getattr(rt, "model_id", None),
        weights=weights_card(rt),
        neuron_ids=neuron_ids_meta,
    )
    payload = {
        **prov,
        "exploratory": True,
        "registered": False,
        "gates_nothing": True,
        "n_docs": n_docs,
        "pure_half1_windows": n_pure_h1,
        "elapsed_total_s": round(time.time() - t0, 1),
        "layers": layer_reports,
        "examples": examples,
    }
    (out_dir / "precision_pilot.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    (out_dir / "precision_pilot.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    main()
