#!/usr/bin/env python3
"""Calibration and execution CLI for Controlled Antisymmetric Forced Logit Census (`forced_odd`).

Subcommands:
  ladder     Run 10-rung geometric ceiling dose ladder on 64 neurons/layer and interpolate target KL doses.
  calibrate  Run `cent` + `proj` calibration on L59_pilot (bugfix check) and open layers L54, L45, L37, L30.
  trial      Run registered fresh dual-null trials on L55-L59 and evaluate F1/F2/F3 pass gates.
  whole      Run crash-safe streaming whole-layer forced_odd census (21,504 neurons + dual nulls + Half C).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

WORKTREE = Path("/home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief")
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from backend import atlas_ops as ao  # noqa: E402
from backend import census as cz  # noqa: E402
from backend import forced_odd as fo  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend.causal_check import language_model, model_device  # noqa: E402
from backend.runtime_factory import make_runtime  # noqa: E402


DECODING_UNIT_FILES = {
    30: "census/runs/L30-match_things.units.jsonl",
    37: "census/runs/L37-match_ceil.units.jsonl",
    45: "census/runs/L45-break_rel.units.jsonl",
    54: "census/runs/L54-match_things.units.jsonl",
}


def parse_layers_arg(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            step = 1 if int(b) >= int(a) else -1
            out.extend(range(int(a), int(b) + step, step))
        else:
            out.append(int(part))
    return out


def load_model(log: Any = print) -> Tuple[Any, Any, ao.Weights]:
    rt = make_runtime(load_jlens=False).load()
    w4a16_fast.install(rt, "compiled", log=log)
    lm = language_model(rt.hf_model)
    w = ao.Weights(lm, rt)
    return rt, lm, w


def cmd_ladder(args: argparse.Namespace) -> None:
    t0 = time.time()
    layers = parse_layers_arg(args.layers)
    rungs = [float(x) for x in args.rungs.split(",")]
    targets = [float(x) for x in args.targets.split(",")]
    out_path = WORKTREE / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = fo.resolve_data_dir(Path(args.data_dir) if args.data_dir else None)

    rt, lm, w = load_model()
    texts = fo.build_fixed_texts(rt, lm, data_dir=data_dir, include_half_c=False)
    pf = fo.run_preflight(rt, lm, texts, check_layers=(29, 59))
    print(f"Preflight OK: {pf}", flush=True)

    N_ff = w.d_ff
    W_U = rt.lm_head_weight
    existing: Dict[str, Any] = {}
    if out_path.exists() and args.resume:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        existing = {str(e["layer"]): e for e in prev.get("layers", [])}

    layer_entries: List[Dict[str, Any]] = []
    for layer in layers:
        if str(layer) in existing:
            layer_entries.append(existing[str(layer)])
            continue
        tl0 = time.time()
        neurons = sorted(int(x) for x in np.random.default_rng(2026112400 + layer).choice(N_ff, args.n_neurons, replace=False))
        u64 = w.write_dirs(layer, neurons)
        c64 = w.write_ceiling(layer, neurons)
        grpA = fo.prepare_group_cache(rt, lm, layer, texts, "A")
        batch_units = 8 if layer >= 40 else 4

        rung_rows = []
        kls_p50 = []
        for r in rungs:
            deltas = (r * c64)[:, None] * u64
            res = fo.run_forward_half(lm, W_U, layer, grpA, deltas, batch_units, readout_mode="cent")
            kl = 0.5 * (res["kl_pos"] + res["kl_neg"])
            eo = res["even_l2"] / (res["odd_l2_cent"] + 1e-8)
            p10, p50, p90 = [float(x) for x in np.percentile(kl, [10, 50, 90])]
            rung_rows.append(
                {
                    "dose_ceil": r,
                    "kl_p10": round(p10, 5),
                    "kl_p50": round(p50, 5),
                    "kl_p90": round(p90, 5),
                    "eo_p50": round(float(np.median(eo)), 3),
                    "alpha_abs_p50": round(float(np.median(res["alpha_abs"])), 4),
                }
            )
            kls_p50.append(p50)
            del deltas, res

        interp = {}
        for tg in targets:
            d_tg, note = fo.loglog_interpolate_dose(rungs, kls_p50, target=tg)
            interp[f"{tg:g}"] = {"dose_ceil": d_tg, "note": note}

        entry = {
            "layer": layer,
            "n_neurons": len(neurons),
            "neuron_ids": neurons,
            "rungs": rung_rows,
            "interpolated": interp,
            "elapsed_s": round(time.time() - tl0, 1),
        }
        layer_entries.append(entry)
        del grpA, u64, c64
        torch.cuda.empty_cache()
        gc.collect()

        payload = {
            "model": getattr(rt, "model_id", None),
            "rungs": rungs,
            "targets": targets,
            "preflight": pf,
            "elapsed_total_s": round(time.time() - t0, 1),
            "layers": layer_entries,
        }
        out_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(
            f"Ladder L{layer:2d} ({entry['elapsed_s']}s): d@0.25={interp.get('0.25',{}).get('dose_ceil')} "
            f"[{min(kls_p50):.4f} .. {max(kls_p50):.3f} nats]",
            flush=True,
        )

    # Write concise text table beside JSON
    lines = [
        f"forced_odd 64-Neuron Dose Ladder ({len(layer_entries)} layers, rungs {rungs[0]}..{rungs[-1]} ceilings)",
        "-" * 108,
        f"{'L':>3} {'KL@0.08':>8} {'KL@0.26':>8} {'KL@0.55':>8} {'KL@1.15':>8} {'KL@2.35':>8} "
        f"{'d@0.15':>7} {'d@0.25':>7} {'d@0.35':>7} {'alpha|0.25':>10}",
    ]
    for e in sorted(layer_entries, key=lambda x: x["layer"]):
        by_r = {rr["dose_ceil"]: rr for rr in e["rungs"]}
        d25 = e["interpolated"].get("0.25", {}).get("dose_ceil", 0.0)
        nearest_r = min(e["rungs"], key=lambda rr: abs(rr["dose_ceil"] - float(d25)))
        lines.append(
            f"{e['layer']:3d} "
            f"{by_r.get(0.08,{}).get('kl_p50', float('nan')):8.4f} "
            f"{by_r.get(0.26,{}).get('kl_p50', float('nan')):8.4f} "
            f"{by_r.get(0.55,{}).get('kl_p50', float('nan')):8.4f} "
            f"{by_r.get(1.15,{}).get('kl_p50', float('nan')):8.4f} "
            f"{by_r.get(2.35,{}).get('kl_p50', float('nan')):8.4f} "
            f"{e['interpolated'].get('0.15',{}).get('dose_ceil', float('nan')):7.3f} "
            f"{e['interpolated'].get('0.25',{}).get('dose_ceil', float('nan')):7.3f} "
            f"{e['interpolated'].get('0.35',{}).get('dose_ceil', float('nan')):7.3f} "
            f"{nearest_r['alpha_abs_p50']:10.4f}"
        )
    txt = "\n".join(lines) + "\n"
    out_path.with_suffix(".txt").write_text(txt, encoding="utf-8")
    print("\n" + txt, flush=True)


def load_decoding_overlap_ref(layer: int, data_dir: Path) -> Optional[Dict[str, Dict[int, Set[str]]]]:
    rel = DECODING_UNIT_FILES.get(layer)
    if not rel:
        return None
    p = data_dir / rel
    if not p.exists():
        return None
    all_w: Dict[int, Set[str]] = {}
    q05_w: Dict[int, Set[str]] = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "neuron":
                continue
            nid = int(r["neuron"])
            ws = set(cz.shown_words(r, k=5))
            if not ws:
                continue
            all_w[nid] = ws
            if float(r.get("q", 1.0)) <= 0.05:
                q05_w[nid] = ws
    return {"all": all_w, "q05": q05_w}


def eval_decoding_overlap(
    ref: Optional[Dict[str, Dict[int, Set[str]]]],
    sc: Dict[str, Any],
    named_local: np.ndarray,
    top_order: np.ndarray,
    neurons: Sequence[int],
    offset_global: int,
    clean_str: Sequence[str],
    raw_str: Sequence[str],
) -> Optional[Dict[str, Any]]:
    if ref is None:
        return None
    cand = list(named_local) if len(named_local) > 0 else list(top_order[:100])
    rng = np.random.default_rng(0)
    shuf = rng.permutation(len(cand))
    out: Dict[str, Any] = {"checked": len(cand), "used_fdp05": len(named_local) > 0}
    for sub in ("all", "q05"):
        wmap = ref[sub]
        if len(wmap) == 0:
            out[sub] = {"ref_neurons": 0, "matched": 0, "frac": None, "shuf_frac": None}
            continue
        matched, shuf_matched, denom = 0, 0, 0
        for j, loc_i in enumerate(cand):
            nid = int(neurons[int(loc_i)])
            gu = offset_global + int(loc_i)
            ps = fo.extract_replicated_pieces(sc, gu, raw_str, k=5)
            p_words = {clean_str[pid] for _, _, pid in ps["pos"] + ps["neg"]}
            dw = wmap.get(nid)
            if dw:
                denom += 1
                if p_words & dw:
                    matched += 1
            sh_nid = int(neurons[int(cand[shuf[j]])])
            sh_dw = wmap.get(sh_nid)
            if sh_dw and (p_words & sh_dw):
                shuf_matched += 1
        out[sub] = {
            "ref_neurons": len(wmap),
            "checked_in_ref": denom,
            "matched": matched,
            "frac_of_checked": round(matched / max(denom, 1), 4) if denom else None,
            "frac_of_cand": round(matched / max(len(cand), 1), 4),
            "shuf_matched": shuf_matched,
            "shuf_frac": round(shuf_matched / max(len(cand), 1), 4),
        }
    return out


def format_examples_section(
    title: str,
    sc: Dict[str, Any],
    ev: Dict[str, Any],
    neurons: Sequence[int],
    n_cols: int,
    n_iso: int,
    raw_str: Sequence[str],
    tag_info: Optional[Dict[str, Any]] = None,
) -> List[str]:
    offset = n_cols + n_iso
    T_neu = sc["T"][offset:]
    top_order = np.argsort(-T_neu)
    named_local = set(int(i) for i in ev["_named_local_idx"])
    lines = [
        f"\n=== {title} (T_auc={ev['auc_T']:.4f}, rABr={ev['auc_r_AB_res']:.4f}, rBAr={ev['auc_r_BA_res']:.4f}, "
        f"FDP05={ev['fdp05']['count']}/{len(neurons)}, aboveMax={ev['fdp05']['above_max_ctl']}) ===",
        "Top 25 neurons:",
    ]
    for loc_i in top_order[:25]:
        gu = offset + int(loc_i)
        nid = int(neurons[int(loc_i)])
        ps = fo.extract_replicated_pieces(sc, gu, raw_str, k=5)
        wtag = ""
        if tag_info and tag_info.get("available"):
            wtag = " dir" if tag_info["is_direct"][int(loc_i)] else " cir"
        lines.append(
            f"  n={nid:6d} named={str(int(loc_i) in named_local):<5}{wtag} T={float(T_neu[loc_i]):6.2f} "
            f"+{[(w0, sc0) for w0, sc0, _ in ps['pos'][:4]]}  -{[(w0, sc0) for w0, sc0, _ in ps['neg'][:4]]}"
        )
    ctl_order = np.argsort(-sc["T"][:n_cols])[:10]
    lines.append("Top 10 columns controls:")
    for ci in ctl_order:
        gu = int(ci)
        ps = fo.extract_replicated_pieces(sc, gu, raw_str, k=5)
        lines.append(
            f"  c={gu:6d} T={float(sc['T'][gu]):6.2f} "
            f"+{[(w0, sc0) for w0, sc0, _ in ps['pos'][:4]]}  -{[(w0, sc0, ) for w0, sc0, _ in ps['neg'][:4]]}"
        )
    return lines


def calibrate_one_layer(
    rt: Any,
    lm: Any,
    w: ao.Weights,
    texts: Dict[str, Any],
    lex_gpu: torch.Tensor,
    clean_str: Sequence[str],
    raw_str: Sequence[str],
    data_dir: Path,
    tag: str,
    layer: int,
    dose_ceil: float,
    neurons: List[int],
    n_cols: int,
    n_iso: int,
    seed_cols: int,
    seed_iso: int,
    do_half_c: bool,
    out_dir: Path,
) -> Dict[str, Any]:
    t0 = time.time()
    dev = model_device(lm)
    batch_units = 8 if layer >= 40 else 4
    W_U = rt.lm_head_weight

    units = fo.make_units(rt, lm, w, layer, neurons, n_cols=n_cols, seed=seed_cols, n_iso=n_iso, iso_seed=seed_iso)
    delta_cols = torch.from_numpy((dose_ceil * units["ref_ceils_col"])[:, None] * units["col_dirs"]).to(dev)
    delta_iso = torch.from_numpy((dose_ceil * units["ref_ceils_iso"])[:, None] * units["iso_dirs"]).to(dev)
    delta_neu = (dose_ceil * units["c_neu"])[:, None] * units["u_neu"]
    all_deltas = torch.cat([delta_cols, delta_iso, delta_neu], dim=0)
    del delta_cols, delta_iso, delta_neu

    U_tot = all_deltas.shape[0]
    col_mask = np.zeros(U_tot, dtype=bool); col_mask[:n_cols] = True
    iso_mask = np.zeros(U_tot, dtype=bool); iso_mask[n_cols : n_cols + n_iso] = True
    neu_mask = np.zeros(U_tot, dtype=bool); neu_mask[n_cols + n_iso :] = True
    offset = n_cols + n_iso

    grpA = fo.prepare_group_cache(rt, lm, layer, texts, "A")
    resA = fo.run_forward_half(lm, W_U, layer, grpA, all_deltas, batch_units, readout_mode="both")
    del grpA
    grpB = fo.prepare_group_cache(rt, lm, layer, texts, "B")
    resB = fo.run_forward_half(lm, W_U, layer, grpB, all_deltas, batch_units, readout_mode="both")
    del grpB

    dec_ref = load_decoding_overlap_ref(layer, data_dir)
    combos: Dict[str, Any] = {}
    combo_scs: Dict[Tuple[str, str], Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]] = {}
    ex_lines = [
        f"{tag}: Layer {layer} dose={dose_ceil:.4f} ceil, KL_n p10/p50/p90="
        + "/".join(f"{x:.3f}" for x in np.percentile(0.5 * (0.5 * (resA["kl_pos"][neu_mask] + resA["kl_neg"][neu_mask]) + 0.5 * (resB["kl_pos"][neu_mask] + resB["kl_neg"][neu_mask])), [10, 50, 90]))
    ]

    for rmode in ("cent", "proj"):
        cov = fo.build_cov_matrix(resA, resB, rmode)
        A_key, B_key = f"odd_{rmode}", f"odd_{rmode}"
        A_mat, B_mat = resA[A_key], resB[B_key]

        zv_name, zv = "LEX_L2_cent", False
        cname = f"{rmode}/{zv_name}"
        sc = fo.score_odd_pair_gpu(A_mat, B_mat, col_mask, cov, lex_gpu, zv, dev, keep_words=True, top_k=5)
        ev_col = fo.eval_split_metrics(sc, neu_mask, col_mask)
        ev_iso_vs_col = fo.eval_split_metrics(sc, iso_mask, col_mask)
        sc_iso = fo.score_odd_pair_gpu(A_mat, B_mat, iso_mask, cov, lex_gpu, zv, dev, keep_words=False)
        ev_iso = fo.eval_split_metrics(sc_iso, neu_mask, iso_mask)

        col_named = {int(x) for x in ev_col["_named_local_idx"]}
        iso_named = {int(x) for x in ev_iso["_named_local_idx"]}
        dual_named_count = len(col_named & iso_named)

        ns_res = fo.run_null_splits(A_mat[:n_cols], B_mat[:n_cols], cov[:n_cols], lex_gpu, zv, dev, n_splits=50, seed=100 + layer)
        T_neu = sc["T"][neu_mask]
        top_order = np.argsort(-T_neu)
        ov = eval_decoding_overlap(dec_ref, sc, ev_col["_named_local_idx"], top_order, neurons, offset, clean_str, raw_str)

        tag_info = fo.corrected_weight_word_tag(
            rt,
            layer,
            neurons,
            units["col_dirs"],
            A_mat[:n_cols],
            B_mat[:n_cols],
            A_mat[offset:],
            B_mat[offset:],
            sc["_control_model"],
            lex_gpu,
            dev,
            data_dir=data_dir,
        )

        combos[cname] = {
            "vs_columns": {k: v for k, v in ev_col.items() if k != "_named_local_idx"},
            "vs_isotropic": {k: v for k, v in ev_iso.items() if k != "_named_local_idx"},
            "dual_named_count": dual_named_count,
            "iso_false_vs_cols": ev_iso_vs_col["fdp05"]["count"],
            "iso_vs_cols_auc_T": ev_iso_vs_col["auc_T"],
            "null_splits": ns_res,
            "decoding_overlap": ov,
            "weight_word_diag": (
                {
                    "auc_own_word_vs_own_ctl": tag_info["auc_own_word_vs_own_ctl"],
                    "ctl_frac_gt0": tag_info["ctl_frac_gt0"],
                    "neu_frac_gt0": tag_info["neu_frac_gt0"],
                    "direct_q05_count": tag_info["direct_q05_count"],
                    "direct_among_named": int(sum(1 for idx in ev_col["_named_local_idx"] if tag_info["is_direct"][int(idx)])),
                }
                if tag_info and tag_info.get("available")
                else None
            ),
        }
        combo_scs[(rmode, zv_name)] = (sc, ev_col, cov)
        ex_lines.extend(format_examples_section(cname, sc, ev_col, neurons, n_cols, n_iso, raw_str, tag_info))

    # Shared Half C pass over union of named neurons across both combos (round-robin up to 300)
    if do_half_c:
        order_keys = (("proj", "LEX_L2_cent"), ("cent", "LEX_L2_cent"))
        lists_k = [
            list(combo_scs[k][1]["_named_local_idx"]) if len(combo_scs[k][1]["_named_local_idx"]) >= 20
            else list(np.argsort(-combo_scs[k][0]["T"][neu_mask])[:50])
            for k in order_keys
        ]
        pick_local: List[int] = []
        seen_l: Set[int] = set()
        max_len = max((len(x) for x in lists_k), default=0)
        for pos_i in range(max_len):
            for lst in lists_k:
                if pos_i < len(lst):
                    val = int(lst[pos_i])
                    if val not in seen_l:
                        seen_l.add(val)
                        pick_local.append(val)
                if len(pick_local) >= 300:
                    break
            if len(pick_local) >= 300:
                break
        pick_local_arr = np.array(pick_local, dtype=np.int64)
        c_ctl_idx = np.arange(min(256, n_cols))
        sub_deltas = torch.cat([all_deltas[c_ctl_idx], all_deltas[offset + pick_local_arr]], dim=0)
        grpC = fo.prepare_group_cache(rt, lm, layer, texts, "C")
        resC = fo.run_forward_half(lm, W_U, layer, grpC, sub_deltas, batch_units, readout_mode="both")
        del grpC, sub_deltas

        mask_f = lex_gpu.float()
        M_mask = float(mask_f.sum().item())
        nc_c = len(c_ctl_idx)
        for (rmode, zv_name), (sc, ev_col, _) in combo_scs.items():
            cname = f"{rmode}/{zv_name}"
            zv = (zv_name == "LEX_L2_zsc")
            oddC = resC[f"odd_{rmode}"].to(device=dev, dtype=torch.float32)
            muC = oddC[:nc_c].mean(0)
            cC = oddC - muC[None]
            if zv:
                varC = oddC[:nc_c].var(0, unbiased=True).clamp(min=0.0)
                cC = cC / torch.sqrt(varC[None] + fo.VAR_FLOOR ** 2)
            cC = cC * mask_f[None]
            normC = (cC / torch.sqrt((cC * cC).sum(-1, keepdim=True) / M_mask + 1e-8)).cpu().numpy()
            normC_ctl, normC_neu = normC[:nc_c], normC[nc_c:]

            named_set = set(int(x) for x in ev_col["_named_local_idx"])
            if len(named_set) >= 20:
                eval_positions = [j for j, li in enumerate(pick_local) if li in named_set]
                were_named = True
            else:
                # Top 50 by this combo
                top50 = set(int(x) for x in np.argsort(-sc["T"][neu_mask])[:50])
                eval_positions = [j for j, li in enumerate(pick_local) if li in top50]
                were_named = False

            sc_c_vals, sc_c_ctl, p_c_list = [], [], []
            for k_idx, j in enumerate(eval_positions):
                gu = offset + int(pick_local[j])
                ps = fo.extract_replicated_pieces(sc, gu, raw_str, k=5)
                p_ids = [pid for _, _, pid in ps["pos"]]
                n_ids = [nid0 for _, _, nid0 in ps["neg"]]
                denom = max(len(p_ids) + len(n_ids), 1)
                val_n = float((normC_neu[j, p_ids].sum() - normC_neu[j, n_ids].sum()) / denom)
                ctl_sc = (normC_ctl[:, p_ids].sum(1) - normC_ctl[:, n_ids].sum(1)) / denom
                sc_c_vals.append(val_n)
                sc_c_ctl.append(float(ctl_sc[k_idx % nc_c]))
                ge = int(np.sum(ctl_sc >= val_n))
                p_c_list.append((1.0 + ge) / (nc_c + 1.0))

            combos[cname]["half_c"] = {
                "evaluated": len(eval_positions),
                "were_fdp05_named": were_named,
                "half_c_auc": round(fo.auc(sc_c_vals, sc_c_ctl), 4) if eval_positions else None,
                "frac_gt_0": round(float(np.mean([v > 0 for v in sc_c_vals])), 4) if eval_positions else None,
                "frac_p05_vs_ctl": round(float(np.mean([p <= 0.05 for p in p_c_list])), 4) if eval_positions else None,
                "median_score_C": round(float(np.median(sc_c_vals)), 3) if eval_positions else None,
            }
            del oddC, cC
        del resC

    kl_n = 0.5 * (0.5 * (resA["kl_pos"][neu_mask] + resA["kl_neg"][neu_mask]) + 0.5 * (resB["kl_pos"][neu_mask] + resB["kl_neg"][neu_mask]))
    kl_c = 0.5 * (0.5 * (resA["kl_pos"][col_mask] + resA["kl_neg"][col_mask]) + 0.5 * (resB["kl_pos"][col_mask] + resB["kl_neg"][col_mask]))
    eo_cent_n = resA["even_l2"][neu_mask] / (resA["odd_l2_cent"][neu_mask] + 1e-8)
    alpha_abs_n = 0.5 * (resA["alpha_abs"][neu_mask] + resB["alpha_abs"][neu_mask])

    summary = {
        "tag": tag,
        "layer": layer,
        "neurons": len(neurons),
        "columns_controls": n_cols,
        "isotropic_controls": n_iso,
        "dose_ceil": dose_ceil,
        "elapsed_s": round(time.time() - t0, 1),
        "kl_neurons": {"p10": round(float(np.percentile(kl_n, 10)), 4), "p50": round(float(np.percentile(kl_n, 50)), 4), "p90": round(float(np.percentile(kl_n, 90)), 4)},
        "kl_columns": {"p10": round(float(np.percentile(kl_c, 10)), 4), "p50": round(float(np.percentile(kl_c, 50)), 4), "p90": round(float(np.percentile(kl_c, 90)), 4)},
        "eo_cent_p50": round(float(np.median(eo_cent_n)), 3),
        "alpha_abs_p50": round(float(np.median(alpha_abs_n)), 4),
        "alpha_abs_p90": round(float(np.percentile(alpha_abs_n, 90)), 4),
        "combos": combos,
    }

    (out_dir / f"calib_{tag}.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (out_dir / f"calib_{tag}_examples.txt").write_text("\n".join(ex_lines) + "\n", encoding="utf-8")

    del resA, resB, all_deltas, units, combo_scs
    torch.cuda.empty_cache()
    gc.collect()
    return summary


def cmd_calibrate(args: argparse.Namespace) -> None:
    t0 = time.time()
    ladder_path = WORKTREE / args.ladder if not Path(args.ladder).is_absolute() else Path(args.ladder)
    out_dir = WORKTREE / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = fo.resolve_data_dir(Path(args.data_dir) if args.data_dir else None)

    ladder_data = json.loads(ladder_path.read_text(encoding="utf-8"))
    by_layer = {int(e["layer"]): e for e in ladder_data["layers"]}

    rt, lm, w = load_model()
    lex_gpu, _, clean_str, raw_str = fo.build_vocab_masks(rt.tokenizer, int(rt.lm_head_weight.shape[0]))
    lex_gpu = lex_gpu.to(model_device(lm))
    texts = fo.build_fixed_texts(rt, lm, data_dir=data_dir, include_half_c=True)
    pf = fo.run_preflight(rt, lm, texts, check_layers=(29, 59))
    print(f"Preflight OK (with Half C): {pf}", flush=True)

    N_ff = w.d_ff
    jobs = [
        {
            "tag": "L59_pilot",
            "layer": 59,
            "neurons": sorted(int(x) for x in np.random.default_rng(11).choice(N_ff, 1022, replace=False)),
            "seed_cols": 1059,
            "seed_iso": 2058,
            "do_half_c": True,
        },
        {
            "tag": "L54",
            "layer": 54,
            "neurons": sorted(int(x) for x in np.random.default_rng(2026112410 + 54).choice(N_ff, 1022, replace=False)),
            "seed_cols": 3054,
            "seed_iso": 4054,
            "do_half_c": True,
        },
        {
            "tag": "L45",
            "layer": 45,
            "neurons": sorted(int(x) for x in np.random.default_rng(2026112410 + 45).choice(N_ff, 1022, replace=False)),
            "seed_cols": 3045,
            "seed_iso": 4045,
            "do_half_c": True,
        },
        {
            "tag": "L37",
            "layer": 37,
            "neurons": sorted(int(x) for x in np.random.default_rng(2026112410 + 37).choice(N_ff, 1022, replace=False)),
            "seed_cols": 3037,
            "seed_iso": 4037,
            "do_half_c": False,
        },
        {
            "tag": "L30",
            "layer": 30,
            "neurons": sorted(int(x) for x in np.random.default_rng(2026112410 + 30).choice(N_ff, 1022, replace=False)),
            "seed_cols": 3030,
            "seed_iso": 4030,
            "do_half_c": False,
        },
    ]

    results: List[Dict[str, Any]] = []
    for j in jobs:
        layer = int(j["layer"])
        dose_ceil, dose_note = fo.dose_from_ladder_entry(by_layer[layer], stat=args.dose_stat, target=float(args.target_kl))
        if dose_note is not None:
            raise RuntimeError(f"Layer {layer} failed to bracket {args.dose_stat}={args.target_kl}: {dose_note}")
        print(f"\n--- Calibrating {j['tag']} (L{layer}) at dose={dose_ceil} ceil ({args.dose_stat}={args.target_kl}) ---", flush=True)
        res = calibrate_one_layer(
            rt=rt,
            lm=lm,
            w=w,
            texts=texts,
            lex_gpu=lex_gpu,
            clean_str=clean_str,
            raw_str=raw_str,
            data_dir=data_dir,
            tag=j["tag"],
            layer=layer,
            dose_ceil=dose_ceil,
            neurons=j["neurons"],
            n_cols=args.n_cols,
            n_iso=args.n_iso,
            seed_cols=j["seed_cols"],
            seed_iso=j["seed_iso"],
            do_half_c=bool(j["do_half_c"]),
            out_dir=out_dir,
        )
        results.append(res)
        (out_dir / "calibration_summary.json").write_text(
            json.dumps(
                {"elapsed_total_s": round(time.time() - t0, 1), "dose_stat": args.dose_stat, "target_kl": args.target_kl, "layers": results},
                indent=1,
            ) + "\n",
            encoding="utf-8",
        )
        p_c = res["combos"]["proj/LEX_L2_cent"]["vs_columns"]
        c_c = res["combos"]["cent/LEX_L2_cent"]["vs_columns"]
        p_d = res["combos"]["proj/LEX_L2_cent"]["dual_named_count"]
        c_d = res["combos"]["cent/LEX_L2_cent"]["dual_named_count"]
        print(
            f"{j['tag']} done ({res['elapsed_s']}s): KL_p50={res['kl_neurons']['p50']} p90={res['kl_neurons']['p90']}, "
            f"|alpha|={res['alpha_abs_p50']} | cent/cent: T={c_c['auc_T']} FDP05={c_c['fdp05']['count']} (dual={c_d}) | "
            f"proj/cent: T={p_c['auc_T']} FDP05={p_c['fdp05']['count']} (dual={p_d})",
            flush=True,
        )

    # Human-readable table across both modes
    lines = [
        f"forced_odd Stage-1 Calibration Summary ({args.dose_stat}={args.target_kl}, C_cols={args.n_cols}, C_iso={args.n_iso})",
        "-" * 148,
        f"{'Tag':<10} {'Combo':<17} {'DoseC':>6} {'KL50/90':>11} {'rABr':>6} {'rBAr':>6} {'T_auc':>6} "
        f"{'FDP05':>6} {'Dual05':>6} {'Null05':>6} {'Iso05':>5} {'IsoF':>5} {'HalfCp05':>8} {'DecMatch':>9} {'Direct':>7}",
    ]
    for res in results:
        kl_s = f"{res['kl_neurons']['p50']:.2f}/{res['kl_neurons']['p90']:.2f}"
        for cname in ("cent/LEX_L2_cent", "proj/LEX_L2_cent"):
            c = res["combos"][cname]
            vc = c["vs_columns"]
            vi = c["vs_isotropic"]
            ns = c["null_splits"]
            hc = c.get("half_c") or {}
            hc_s = f"{hc['frac_p05_vs_ctl']:.3f}" if hc.get("frac_p05_vs_ctl") is not None else "-"
            ov = c.get("decoding_overlap") or {}
            q05_ov = ov.get("q05") or {}
            all_ov = ov.get("all") or {}
            if q05_ov.get("frac_of_checked") is not None:
                ov_s = f"{q05_ov['frac_of_checked']:.3f}(q)"
            elif all_ov.get("frac_of_checked") is not None:
                ov_s = f"{all_ov['frac_of_checked']:.3f}(a)"
            else:
                ov_s = "-"
            wd = c.get("weight_word_diag") or {}
            dir_s = f"{wd['direct_among_named']}/{vc['fdp05']['count']}" if wd else "-"
            iso_f = c.get("iso_false_vs_cols", -1)
            dual05 = c.get("dual_named_count", 0)
            lines.append(
                f"{res['tag']:<10} {cname:<17} {res['dose_ceil']:6.3f} {kl_s:>11} {vc['auc_r_AB_res']:6.3f} {vc['auc_r_BA_res']:6.3f} "
                f"{vc['auc_T']:6.3f} {vc['fdp05']['count']:6d} {dual05:6d} {ns['fdp05_false_mean']:6.2f} "
                f"{vi['fdp05']['count']:5d} {iso_f:5d} {hc_s:>8} {ov_s:>9} {dir_s:>7}"
            )
    txt = "\n".join(lines) + "\n"
    (out_dir / "calibration_summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt, flush=True)


def _resolve_worktree_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (WORKTREE / path)


def _check_preflight_strict(pf: Dict[str, Any]) -> bool:
    for grp, info in pf.items():
        if float(info["min_ans_lp"]) <= -0.35:
            raise RuntimeError(f"Strict preflight failed on group {grp}: min_ans_lp={info['min_ans_lp']} <= -0.35")
        if max(float(v) for v in info["max_diffs"].values()) >= 1e-2:
            raise RuntimeError(f"Strict preflight failed on group {grp}: max_diffs={info['max_diffs']} >= 1e-2")
    return True


def _kl_summary_dict(resA: Dict[str, Any], resB: Dict[str, Any], mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    kp = resA["kl_pos"] if mask is None else resA["kl_pos"][mask]
    kn = resA["kl_neg"] if mask is None else resA["kl_neg"][mask]
    kp_b = resB["kl_pos"] if mask is None else resB["kl_pos"][mask]
    kn_b = resB["kl_neg"] if mask is None else resB["kl_neg"][mask]
    kl = 0.5 * (0.5 * (kp + kn) + 0.5 * (kp_b + kn_b))
    return {
        "p10": round(float(np.percentile(kl, 10)), 4),
        "p50": round(float(np.percentile(kl, 50)), 4),
        "p90": round(float(np.percentile(kl, 90)), 4),
    }


def _git_head_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", str(WORKTREE), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _build_dual_examples_and_named(
    layer: int,
    dose_ceil: float,
    neurons: Sequence[int],
    dual_named_local: Sequence[int],
    sc_col: Dict[str, Any],
    T_iso_neu: np.ndarray,
    r_ab_res_iso_neu: np.ndarray,
    r_ba_res_iso_neu: np.ndarray,
    offset_col: int,
    n_cols: int,
    raw_str: Sequence[str],
    tag_info: Optional[Dict[str, Any]],
    kl_neurons: Dict[str, float],
    auc_T_col: float,
    auc_T_iso: float,
    header_tag: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Format strictly dual-null FDP <= 0.05 named neurons (`forced next-piece shift`) and controls."""
    has_ww = bool(tag_info and tag_info.get("available"))
    T_dual_score = np.minimum(sc_col["T"][offset_col:], T_iso_neu)
    dual_sorted = sorted(dual_named_local, key=lambda li: -float(T_dual_score[int(li)]))

    lines = [
        f"=== {header_tag}: Layer {layer} (dose={dose_ceil:.4f} ceil, KL_n p10/p50/p90="
        f"{kl_neurons['p10']:.3f}/{kl_neurons['p50']:.3f}/{kl_neurons['p90']:.3f}, "
        f"T_auc_col={auc_T_col:.4f}, T_auc_iso={auc_T_iso:.4f}, Dual05={len(dual_sorted)}/{len(neurons)}) ===",
        "Display rule: words shown ONLY for neurons clearing empirical FDP <= 0.05 against BOTH columns & isotropic controls.",
        "Label: forced next-piece shift (FDR <= 5%, dual-null).",
    ]

    named_records: List[Dict[str, Any]] = []
    for loc_i in dual_sorted:
        nid = int(neurons[int(loc_i)])
        gu = offset_col + int(loc_i)
        ps = fo.extract_replicated_pieces(sc_col, gu, raw_str, k=5)
        is_dir = bool(tag_info["is_direct"][int(loc_i)]) if has_ww else False
        ww_tag = ("direct_unembed" if is_dir else "circuit_mediated") if has_ww else "unknown"
        q_dir = round(float(tag_info["q_direct"][int(loc_i)]), 5) if has_ww else None
        named_records.append(
            {
                "neuron": nid,
                "label": "forced next-piece shift (FDR <= 5%, dual-null)",
                "T_col": round(float(sc_col["T"][gu]), 4),
                "T_iso": round(float(T_iso_neu[int(loc_i)]), 4),
                "r_ab_res_col": round(float(sc_col["r_ab_res"][gu]), 4),
                "r_ba_res_col": round(float(sc_col["r_ba_res"][gu]), 4),
                "r_ab_res_iso": round(float(r_ab_res_iso_neu[int(loc_i)]), 4),
                "r_ba_res_iso": round(float(r_ba_res_iso_neu[int(loc_i)]), 4),
                "tag": ww_tag,
                "q_direct": q_dir,
                "pos": [(w0, sc0, int(pid)) for w0, sc0, pid in ps["pos"]],
                "neg": [(w0, sc0, int(pid)) for w0, sc0, pid in ps["neg"]],
            }
        )

    if not dual_sorted:
        lines.append("0 neurons cleared dual-null FDP <= 0.05; no neuron words shown.")
    else:
        lines.append(f"Top {min(30, len(dual_sorted))} passing Dual05 neurons (of {len(dual_sorted)}):")
        for rec in named_records[:30]:
            pos_s = [(w0, sc0) for w0, sc0, _ in rec["pos"][:4]]
            neg_s = [(w0, sc0) for w0, sc0, _ in rec["neg"][:4]]
            lines.append(
                f"  n={rec['neuron']:5d} [{rec['tag']:<16}] T_col={rec['T_col']:6.2f} T_iso={rec['T_iso']:6.2f} "
                f"| forced next-piece shift (FDR <= 5%, dual-null) | +{pos_s} -{neg_s}"
            )

    if n_cols > 0 and "topA_p" in sc_col:
        ctl_order = np.argsort(-sc_col["T"][:n_cols])[:10]
        lines.append("Top 10 columns controls (for comparison):")
        for ci in ctl_order:
            gu = int(ci)
            ps = fo.extract_replicated_pieces(sc_col, gu, raw_str, k=5)
            lines.append(
                f"  c={gu:5d} T_col={float(sc_col['T'][gu]):6.2f} "
                f"+{[(w0, sc0) for w0, sc0, _ in ps['pos'][:4]]} -{[(w0, sc0) for w0, sc0, _ in ps['neg'][:4]]}"
            )
    return lines, named_records


def _select_half_c_indices(
    dual_named_local: Sequence[int],
    T_col_neu: np.ndarray,
    T_iso_neu: np.ndarray,
    layer: int,
    max_n: int,
) -> Tuple[np.ndarray, bool]:
    rng = np.random.default_rng(9000 + int(layer))
    T_dual = np.minimum(T_col_neu, T_iso_neu)
    if len(dual_named_local) >= 20:
        pool = np.array(sorted(dual_named_local), dtype=np.int64)
        were_named = True
    else:
        pool = np.argsort(-T_dual)[: min(50, len(T_dual))]
        were_named = False
    if len(pool) > max_n:
        pick = np.sort(rng.choice(pool, size=int(max_n), replace=False))
    else:
        pick = pool
    return pick, were_named


def cmd_trial(args: argparse.Namespace) -> None:
    t0 = time.time()
    layers = parse_layers_arg(args.layers)
    ladder_path = _resolve_worktree_path(args.ladder)
    neurons_dir = _resolve_worktree_path(args.neurons_dir)
    out_dir = _resolve_worktree_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = fo.resolve_data_dir(Path(args.data_dir) if args.data_dir else None)

    by_layer = {int(e["layer"]): e for e in json.loads(ladder_path.read_text(encoding="utf-8"))["layers"]}
    rt, lm, w = load_model()
    dev = model_device(lm)
    W_U = rt.lm_head_weight
    N_ff = w.d_ff

    lex_gpu, _, clean_str, raw_str = fo.build_vocab_masks(rt.tokenizer, int(W_U.shape[0]))
    lex_gpu = lex_gpu.to(dev)
    texts = fo.build_fixed_texts(rt, lm, data_dir=data_dir, include_half_c=True)
    pf = fo.run_preflight(rt, lm, texts, check_layers=(29, 59))
    preflight_ok = _check_preflight_strict(pf)
    print(f"Trial strict preflight OK: {pf}", flush=True)

    verdicts: List[Dict[str, Any]] = []
    for layer in layers:
        tl0 = time.time()
        nfile = neurons_dir / f"neurons_L{layer}.txt"
        raw_tokens = [tok.strip() for tok in nfile.read_text(encoding="utf-8").replace("\n", ",").split(",") if tok.strip()]
        neurons = [int(tok) for tok in raw_tokens]
        if len(neurons) == 0 or len(neurons) != len(set(neurons)) or neurons != sorted(neurons):
            raise RuntimeError(f"Invalid neuron list in {nfile}: len={len(neurons)}, unique={len(set(neurons))}")
        if any(x < 0 or x >= N_ff for x in neurons):
            raise RuntimeError(f"Out-of-range neuron ID in {nfile}")

        dose_ceil, dose_note = fo.dose_from_ladder_entry(by_layer[layer], stat=args.dose_stat, target=float(args.target_kl))
        if dose_note is not None:
            raise RuntimeError(f"L{layer} failed to bracket {args.dose_stat}={args.target_kl}: {dose_note}")

        seed_cols = 5100 + layer
        seed_iso = 6100 + layer
        n_cols, n_iso = int(args.n_cols), int(args.n_iso)
        batch_units = 8 if layer >= 40 else 4
        print(f"\n--- Trial L{layer}: N={len(neurons)}, C_cols={n_cols}, C_iso={n_iso}, dose={dose_ceil} ({args.dose_stat}={args.target_kl}) ---", flush=True)

        units = fo.make_units(rt, lm, w, layer, neurons, n_cols=n_cols, seed=seed_cols, n_iso=n_iso, iso_seed=seed_iso)
        delta_cols = torch.from_numpy((dose_ceil * units["ref_ceils_col"])[:, None] * units["col_dirs"]).to(dev)
        delta_iso = torch.from_numpy((dose_ceil * units["ref_ceils_iso"])[:, None] * units["iso_dirs"]).to(dev)
        delta_neu = (dose_ceil * units["c_neu"])[:, None] * units["u_neu"]
        all_deltas = torch.cat([delta_cols, delta_iso, delta_neu], dim=0)
        del delta_cols, delta_iso, delta_neu

        U_tot = all_deltas.shape[0]
        offset = n_cols + n_iso
        col_mask = np.zeros(U_tot, dtype=bool); col_mask[:n_cols] = True
        iso_mask = np.zeros(U_tot, dtype=bool); iso_mask[n_cols:offset] = True
        neu_mask = np.zeros(U_tot, dtype=bool); neu_mask[offset:] = True

        grpA = fo.prepare_group_cache(rt, lm, layer, texts, "A")
        resA = fo.run_forward_half(lm, W_U, layer, grpA, all_deltas, batch_units, readout_mode=args.readout_mode)
        del grpA
        grpB = fo.prepare_group_cache(rt, lm, layer, texts, "B")
        resB = fo.run_forward_half(lm, W_U, layer, grpB, all_deltas, batch_units, readout_mode=args.readout_mode)
        del grpB

        cov = fo.build_cov_matrix(resA, resB, args.readout_mode)
        A_mat = resA[f"odd_{args.readout_mode}"]
        B_mat = resB[f"odd_{args.readout_mode}"]

        sc_col = fo.score_odd_pair_gpu(A_mat, B_mat, col_mask, cov, lex_gpu, False, dev, keep_words=True, top_k=5)
        sc_iso = fo.score_odd_pair_gpu(A_mat, B_mat, iso_mask, cov, lex_gpu, False, dev, keep_words=False)
        ev_col = fo.eval_split_metrics(sc_col, neu_mask, col_mask)
        ev_iso = fo.eval_split_metrics(sc_iso, neu_mask, iso_mask)
        ev_iso_vs_col = fo.eval_split_metrics(sc_col, iso_mask, col_mask)
        dual_named_local = sorted(set(int(x) for x in ev_col["_named_local_idx"]) & set(int(x) for x in ev_iso["_named_local_idx"]))

        ns_col = fo.run_null_splits(A_mat[:n_cols], B_mat[:n_cols], cov[:n_cols], lex_gpu, False, dev, n_splits=args.null_splits, seed=500 + layer)
        ns_iso = fo.run_null_splits(A_mat[n_cols:offset], B_mat[n_cols:offset], cov[n_cols:offset], lex_gpu, False, dev, n_splits=args.null_splits, seed=600 + layer)

        tag_info = fo.corrected_weight_word_tag(
            rt, layer, neurons, units["col_dirs"], A_mat[:n_cols], B_mat[:n_cols], A_mat[offset:], B_mat[offset:],
            sc_col["_control_model"], lex_gpu, dev, data_dir=data_dir,
        )

        # Half C dual-null check
        pick_local, were_dual05_named = _select_half_c_indices(
            dual_named_local, sc_col["T"][offset:], sc_iso["T"][offset:], layer, max_n=args.half_c_max
        )
        piece_pairs = []
        for li in pick_local:
            ps = fo.extract_replicated_pieces(sc_col, offset + int(li), raw_str, k=5)
            piece_pairs.append(([pid for _, _, pid in ps["pos"]], [nid0 for _, _, nid0 in ps["neg"]]))
        hc = fo.eval_half_c_dual(
            rt, lm, layer, texts, lex_gpu, batch_units, args.readout_mode,
            all_deltas[: min(256, n_cols)], all_deltas[n_cols : n_cols + min(256, n_iso)], all_deltas[offset + pick_local],
            piece_pairs, zscore_var=False,
        )
        hc["were_dual05_named"] = were_dual05_named

        kl_neu = _kl_summary_dict(resA, resB, neu_mask)
        kl_col = _kl_summary_dict(resA, resB, col_mask)
        kl_iso = _kl_summary_dict(resA, resB, iso_mask)
        alpha_abs_n = 0.5 * (resA["alpha_abs"][neu_mask] + resB["alpha_abs"][neu_mask])

        # Pre-registered F1, F2, F3 gates
        f1 = len(dual_named_local) >= 50
        f2 = (
            float(ev_col["auc_r_AB_res"]) >= 0.58
            and float(ev_col["auc_r_BA_res"]) >= 0.58
            and float(ev_iso["auc_r_AB_res"]) >= 0.58
            and float(ev_iso["auc_r_BA_res"]) >= 0.58
            and float(ns_col["fdp05_false_mean"]) < 1.0
            and int(ns_col["fdp05_false_max"]) <= 2
            and float(ns_iso["fdp05_false_mean"]) < 1.0
            and int(ns_iso["fdp05_false_max"]) <= 2
        )
        hc_col = hc.get("vs_columns") or {}
        hc_iso = hc.get("vs_isotropic") or {}
        f3 = bool(
            preflight_ok
            and were_dual05_named
            and hc_col.get("frac_p05_vs_ctl") is not None
            and hc_iso.get("frac_p05_vs_ctl") is not None
            and float(hc_col["frac_p05_vs_ctl"]) >= 0.80
            and float(hc_iso["frac_p05_vs_ctl"]) >= 0.80
        )
        passed = bool(f1 and f2 and f3)

        ww_summary = None
        if tag_info and tag_info.get("available"):
            dual_dir = int(sum(1 for idx in dual_named_local if tag_info["is_direct"][int(idx)]))
            ww_summary = {
                "auc_own_word_vs_own_ctl": tag_info["auc_own_word_vs_own_ctl"],
                "ctl_frac_gt0": tag_info["ctl_frac_gt0"],
                "neu_frac_gt0": tag_info["neu_frac_gt0"],
                "direct_q05_count": tag_info["direct_q05_count"],
                "direct_among_dual05": dual_dir,
            }

        verdict = {
            "layer": int(layer),
            "pass": passed,
            "gates": {"F1": f1, "F2": f2, "F3": f3},
            "neurons": len(neurons),
            "neurons_file": str(nfile.relative_to(WORKTREE)),
            "readout_mode": f"{args.readout_mode}/LEX_L2_cent",
            "dose_stat": args.dose_stat,
            "target_kl": float(args.target_kl),
            "dose_ceil": dose_ceil,
            "n_cols": n_cols,
            "n_iso": n_iso,
            "seed_cols": seed_cols,
            "seed_iso": seed_iso,
            "preflight": pf,
            "preflight_ok": preflight_ok,
            "kl_neurons": kl_neu,
            "kl_columns": kl_col,
            "kl_isotropic": kl_iso,
            "alpha_abs_p50": round(float(np.median(alpha_abs_n)), 4),
            "alpha_abs_p90": round(float(np.percentile(alpha_abs_n, 90)), 4),
            "dual_named_count": len(dual_named_local),
            "vs_columns": {k: v for k, v in ev_col.items() if k != "_named_local_idx"},
            "vs_isotropic": {k: v for k, v in ev_iso.items() if k != "_named_local_idx"},
            "iso_false_vs_cols": ev_iso_vs_col["fdp05"]["count"],
            "iso_vs_cols_auc_T": ev_iso_vs_col["auc_T"],
            "null_splits_columns": ns_col,
            "null_splits_isotropic": ns_iso,
            "half_c": hc,
            "weight_word_diag": ww_summary,
            "git_commit": _git_head_sha(),
            "elapsed_s": round(time.time() - tl0, 1),
        }
        verdicts.append(verdict)

        ex_lines, named_records = _build_dual_examples_and_named(
            layer=layer,
            dose_ceil=dose_ceil,
            neurons=neurons,
            dual_named_local=dual_named_local,
            sc_col=sc_col,
            T_iso_neu=sc_iso["T"][offset:],
            r_ab_res_iso_neu=sc_iso["r_ab_res"][offset:],
            r_ba_res_iso_neu=sc_iso["r_ba_res"][offset:],
            offset_col=offset,
            n_cols=n_cols,
            raw_str=raw_str,
            tag_info=tag_info,
            kl_neurons=kl_neu,
            auc_T_col=ev_col["auc_T"],
            auc_T_iso=ev_iso["auc_T"],
            header_tag=f"TRIAL {'PASS' if passed else 'FAIL'}",
        )
        (out_dir / f"L{layer}_verdict.json").write_text(json.dumps(verdict, indent=1) + "\n", encoding="utf-8")
        (out_dir / f"L{layer}_examples.txt").write_text("\n".join(ex_lines) + "\n", encoding="utf-8")
        (out_dir / f"L{layer}_named.json").write_text(json.dumps(named_records, indent=1) + "\n", encoding="utf-8")

        (out_dir / "trials_summary.json").write_text(
            json.dumps({"elapsed_total_s": round(time.time() - t0, 1), "layers": verdicts}, indent=1) + "\n",
            encoding="utf-8",
        )
        print(
            f"Trial L{layer} {'PASS' if passed else 'FAIL'} ({verdict['elapsed_s']}s): Dual05={len(dual_named_local)}/{len(neurons)}, "
            f"T_col={ev_col['auc_T']}, T_iso={ev_iso['auc_T']}, NullC={ns_col['fdp05_false_mean']}, "
            f"NullI={ns_iso['fdp05_false_mean']}, HalfC={hc_col.get('frac_p05_vs_ctl')}/{hc_iso.get('frac_p05_vs_ctl')}",
            flush=True,
        )

        del resA, resB, all_deltas, units, sc_col, sc_iso
        torch.cuda.empty_cache()
        gc.collect()

    lines = [
        f"forced_odd Pre-Registered Trials Summary ({args.readout_mode}/LEX_L2_cent, {args.dose_stat}={args.target_kl}, C_cols={args.n_cols}, C_iso={args.n_iso})",
        "-" * 146,
        f"{'L':>3} {'Verdict':<7} {'Gates':<9} {'DoseC':>6} {'KL50/90':>11} {'rABr_c':>6} {'rBAr_c':>6} {'rABr_i':>6} {'rBAr_i':>6} "
        f"{'Col05':>6} {'Iso05':>6} {'Dual05':>6} {'NullC':>6} {'NullI':>6} {'HC_c/i':>11} {'Direct':>8}",
    ]
    for v in verdicts:
        vc, vi = v["vs_columns"], v["vs_isotropic"]
        nsc, nsi = v["null_splits_columns"], v["null_splits_isotropic"]
        hcc = (v["half_c"].get("vs_columns") or {}).get("frac_p05_vs_ctl")
        hci = (v["half_c"].get("vs_isotropic") or {}).get("frac_p05_vs_ctl")
        hc_s = f"{hcc:.2f}/{hci:.2f}" if (hcc is not None and hci is not None) else "-"
        ww = v.get("weight_word_diag") or {}
        dir_s = f"{ww['direct_among_dual05']}/{v['dual_named_count']}" if ww else "-"
        g_s = f"{'1' if v['gates']['F1'] else '0'}/{'1' if v['gates']['F2'] else '0'}/{'1' if v['gates']['F3'] else '0'}"
        kl_s = f"{v['kl_neurons']['p50']:.2f}/{v['kl_neurons']['p90']:.2f}"
        lines.append(
            f"{v['layer']:3d} {'PASS' if v['pass'] else 'FAIL':<7} {g_s:<9} {v['dose_ceil']:6.3f} {kl_s:>11} "
            f"{vc['auc_r_AB_res']:6.3f} {vc['auc_r_BA_res']:6.3f} {vi['auc_r_AB_res']:6.3f} {vi['auc_r_BA_res']:6.3f} "
            f"{vc['fdp05']['count']:6d} {vi['fdp05']['count']:6d} {v['dual_named_count']:6d} "
            f"{nsc['fdp05_false_mean']:6.2f} {nsi['fdp05_false_mean']:6.2f} {hc_s:>11} {dir_s:>8}"
        )
    txt = "\n".join(lines) + "\n"
    (out_dir / "trials_summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt, flush=True)


def cmd_whole(args: argparse.Namespace) -> None:
    t0 = time.time()
    layer = int(args.layer)
    ladder_path = _resolve_worktree_path(args.ladder)
    out_dir = _resolve_worktree_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / f"work_L{layer}"
    chunks_dir = work_dir / "chunks"
    controls_pt = work_dir / "controls.pt"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    data_dir = fo.resolve_data_dir(Path(args.data_dir) if args.data_dir else None)

    by_layer = {int(e["layer"]): e for e in json.loads(ladder_path.read_text(encoding="utf-8"))["layers"]}
    dose_ceil, dose_note = fo.dose_from_ladder_entry(by_layer[layer], stat=args.dose_stat, target=float(args.target_kl))
    if dose_note is not None:
        raise RuntimeError(f"L{layer} failed to bracket {args.dose_stat}={args.target_kl}: {dose_note}")

    rt, lm, w = load_model()
    dev = model_device(lm)
    W_U = rt.lm_head_weight
    N_ff = w.d_ff
    n_cols, n_iso = int(args.n_cols), int(args.n_iso)
    chunk_size = int(args.chunk_size)
    batch_units = 8 if layer >= 40 else 4
    seed_cols, seed_iso = 7100 + layer, 8100 + layer

    lex_gpu, _, clean_str, raw_str = fo.build_vocab_masks(rt.tokenizer, int(W_U.shape[0]))
    lex_gpu = lex_gpu.to(dev)
    texts = fo.build_fixed_texts(rt, lm, data_dir=data_dir, include_half_c=True)
    pf = fo.run_preflight(rt, lm, texts, check_layers=(29, 59))
    _check_preflight_strict(pf)
    print(f"Whole L{layer} strict preflight OK: {pf}", flush=True)

    lens_path = data_dir / "index/lens.npz"
    has_lens = lens_path.exists()
    eye_pos_l = eye_neg_l = None
    if has_lens:
        lens_npz = np.load(lens_path)
        eye_pos_l = lens_npz["eye_pos"][layer]
        eye_neg_l = lens_npz["eye_neg"][layer]

    grpA = fo.prepare_group_cache(rt, lm, layer, texts, "A")
    grpB = fo.prepare_group_cache(rt, lm, layer, texts, "B")

    # Phase 1: Fit dual control models (skip if checkpoint exists)
    if not controls_pt.exists():
        tc0 = time.time()
        print(f"Whole L{layer} Phase 1: building & running {n_cols} cols + {n_iso} iso controls...", flush=True)
        units_ctl = fo.make_units(rt, lm, w, layer, [], n_cols=n_cols, seed=seed_cols, n_iso=n_iso, iso_seed=seed_iso)
        delta_cols = torch.from_numpy((dose_ceil * units_ctl["ref_ceils_col"])[:, None] * units_ctl["col_dirs"]).to(dev)
        delta_iso = torch.from_numpy((dose_ceil * units_ctl["ref_ceils_iso"])[:, None] * units_ctl["iso_dirs"]).to(dev)
        deltas_all_ctl = torch.cat([delta_cols, delta_iso], dim=0)

        resA_ctl = fo.run_forward_half(lm, W_U, layer, grpA, deltas_all_ctl, batch_units, readout_mode=args.readout_mode)
        resB_ctl = fo.run_forward_half(lm, W_U, layer, grpB, deltas_all_ctl, batch_units, readout_mode=args.readout_mode)
        del deltas_all_ctl

        def _slice_res(r: Dict[str, Any], a: int, b: int) -> Dict[str, Any]:
            return {k: v[a:b] for k, v in r.items()}

        resA_col, resA_iso = _slice_res(resA_ctl, 0, n_cols), _slice_res(resA_ctl, n_cols, n_cols + n_iso)
        resB_col, resB_iso = _slice_res(resB_ctl, 0, n_cols), _slice_res(resB_ctl, n_cols, n_cols + n_iso)
        del resA_ctl, resB_ctl

        cov_col = fo.build_cov_matrix(resA_col, resB_col, args.readout_mode)
        cov_iso = fo.build_cov_matrix(resA_iso, resB_iso, args.readout_mode)
        odd_k = f"odd_{args.readout_mode}"
        A_col, B_col = resA_col[odd_k], resB_col[odd_k]
        A_iso, B_iso = resA_iso[odd_k], resB_iso[odd_k]

        model_col = fo.fit_control_model(A_col, B_col, cov_col, lex_gpu, False, dev, keep_words=False)
        model_iso = fo.fit_control_model(A_iso, B_iso, cov_iso, lex_gpu, False, dev, keep_words=False)

        sc_iso_on_col = fo.score_chunk_against_controls(A_iso, B_iso, cov_iso, model_col, lex_gpu, dev, keep_words=False)
        pok_ic = (sc_iso_on_col["r_ab_res"] > 0) & (sc_iso_on_col["r_ba_res"] > 0)
        iso_false_vs_cols = fo.empirical_fdp(sc_iso_on_col["T"], pok_ic, n_iso, model_col.T_ctl, 0.05)["count"]
        iso_vs_cols_auc_T = round(fo.auc(sc_iso_on_col["T"], model_col.T_ctl), 4)

        ns_col = fo.run_null_splits(A_col, B_col, cov_col, lex_gpu, False, dev, n_splits=args.null_splits, seed=700 + layer)
        ns_iso = fo.run_null_splits(A_iso, B_iso, cov_iso, lex_gpu, False, dev, n_splits=args.null_splits, seed=800 + layer)
        sw_ctl = fo.compute_weight_word_ctl_null(rt, units_ctl["col_dirs"], A_col, B_col, model_col, lex_gpu, dev) if has_lens else None

        ctl_state = {
            "layer": layer,
            "dose_ceil": dose_ceil,
            "n_cols": n_cols,
            "n_iso": n_iso,
            "seed_cols": seed_cols,
            "seed_iso": seed_iso,
            "model_col": fo.serialize_control_model(model_col),
            "model_iso": fo.serialize_control_model(model_iso),
            "ns_col": ns_col,
            "ns_iso": ns_iso,
            "sw_ctl": sw_ctl,
            "deltas_cols_c": delta_cols[: min(256, n_cols)].detach().cpu(),
            "deltas_iso_c": delta_iso[: min(256, n_iso)].detach().cpu(),
            "kl_columns": _kl_summary_dict(resA_col, resB_col),
            "kl_isotropic": _kl_summary_dict(resA_iso, resB_iso),
            "iso_false_vs_cols": int(iso_false_vs_cols),
            "iso_vs_cols_auc_T": float(iso_vs_cols_auc_T),
            "elapsed_controls_s": round(time.time() - tc0, 1),
        }
        tmp_ctl = work_dir / "controls.pt.tmp"
        torch.save(ctl_state, tmp_ctl)
        os.replace(tmp_ctl, controls_pt)
        print(
            f"Whole L{layer} controls done ({ctl_state['elapsed_controls_s']}s): NullC={ns_col['fdp05_false_mean']}, "
            f"NullI={ns_iso['fdp05_false_mean']}, IsoF={iso_false_vs_cols}",
            flush=True,
        )
        del units_ctl, delta_cols, delta_iso, resA_col, resB_col, resA_iso, resB_iso, A_col, B_col, A_iso, B_iso, sc_iso_on_col
        torch.cuda.empty_cache()
        gc.collect()
    else:
        ctl_state = torch.load(controls_pt, weights_only=False)
        model_col = fo.deserialize_control_model(ctl_state["model_col"], dev)
        model_iso = fo.deserialize_control_model(ctl_state["model_iso"], dev)
        print(f"Whole L{layer} loaded existing controls.pt (C_cols={model_col.C}, C_iso={model_iso.C})", flush=True)

    # Phase 2: Stream 512-neuron chunks
    n_chunks = (N_ff + chunk_size - 1) // chunk_size
    odd_k = f"odd_{args.readout_mode}"
    tp2 = time.time()
    n_ran = 0
    for ci in range(n_chunks):
        c_path = chunks_dir / f"chunk_{ci:02d}.npz"
        if c_path.exists():
            continue
        c_start = ci * chunk_size
        c_end = min(N_ff, (ci + 1) * chunk_size)
        c_neurons = list(range(c_start, c_end))

        u_neu = w.write_dirs(layer, c_neurons)
        c_neu = w.write_ceiling(layer, c_neurons)
        deltas_c = (dose_ceil * c_neu)[:, None] * u_neu

        resA_c = fo.run_forward_half(lm, W_U, layer, grpA, deltas_c, batch_units, readout_mode=args.readout_mode)
        resB_c = fo.run_forward_half(lm, W_U, layer, grpB, deltas_c, batch_units, readout_mode=args.readout_mode)
        cov_c = fo.build_cov_matrix(resA_c, resB_c, args.readout_mode)
        A_c, B_c = resA_c[odd_k], resB_c[odd_k]

        sc_col_c = fo.score_chunk_against_controls(A_c, B_c, cov_c, model_col, lex_gpu, dev, keep_words=True, top_k=5)
        sc_iso_c = fo.score_chunk_against_controls(A_c, B_c, cov_c, model_iso, lex_gpu, dev, keep_words=False)
        sw_c = (
            fo.score_weight_word_chunk(eye_pos_l, eye_neg_l, c_neurons, A_c, B_c, model_col, lex_gpu, dev)
            if has_lens
            else np.zeros(len(c_neurons), dtype=np.float32)
        )

        tmp_c = chunks_dir / f"chunk_{ci:02d}.tmp.npz"
        np.savez_compressed(
            tmp_c,
            kl_pos_A=resA_c["kl_pos"],
            kl_neg_A=resA_c["kl_neg"],
            kl_pos_B=resB_c["kl_pos"],
            kl_neg_B=resB_c["kl_neg"],
            alpha_abs_A=resA_c["alpha_abs"],
            alpha_abs_B=resB_c["alpha_abs"],
            r_ab_col=sc_col_c["r_ab"],
            r_ba_col=sc_col_c["r_ba"],
            r_ab_res_col=sc_col_c["r_ab_res"],
            r_ba_res_col=sc_col_c["r_ba_res"],
            T_col=sc_col_c["T"],
            r_ab_iso=sc_iso_c["r_ab"],
            r_ba_iso=sc_iso_c["r_ba"],
            r_ab_res_iso=sc_iso_c["r_ab_res"],
            r_ba_res_iso=sc_iso_c["r_ba_res"],
            T_iso=sc_iso_c["T"],
            topA_p=sc_col_c["topA_p"],
            topA_n=sc_col_c["topA_n"],
            topB_p=sc_col_c["topB_p"],
            topB_n=sc_col_c["topB_n"],
            topC_p=sc_col_c["topC_p"],
            topC_n=sc_col_c["topC_n"],
            posA_onB=sc_col_c["posA_onB"],
            negA_onB=sc_col_c["negA_onB"],
            posB_onA=sc_col_c["posB_onA"],
            negB_onA=sc_col_c["negB_onA"],
            valC_p=sc_col_c["valC_p"],
            valC_n=sc_col_c["valC_n"],
            sw_neu=sw_c,
        )
        os.replace(tmp_c, c_path)
        n_ran += 1
        del u_neu, c_neu, deltas_c, resA_c, resB_c, A_c, B_c, sc_col_c, sc_iso_c
        if ci == 0 or (ci + 1) % 7 == 0 or ci == n_chunks - 1:
            el = time.time() - tp2
            rem = (el / n_ran) * (n_chunks - 1 - ci) if n_ran else 0.0
            print(f"Whole L{layer} chunk {ci+1}/{n_chunks} ({c_end}/{N_ff}) elapsed={el:.1f}s eta={rem:.1f}s", flush=True)

    del grpA, grpB
    torch.cuda.empty_cache()
    gc.collect()

    # Phase 3: Concatenate chunks, run whole-layer dual FDP, Half C, and emit artifacts
    parts: Dict[str, List[np.ndarray]] = {}
    for ci in range(n_chunks):
        with np.load(chunks_dir / f"chunk_{ci:02d}.npz") as z:
            for k in z.files:
                parts.setdefault(k, []).append(z[k])
    full_arr = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    assert len(full_arr["T_col"]) == N_ff, f"Expected {N_ff}, got {len(full_arr['T_col'])}"

    pok_col = (full_arr["r_ab_res_col"] > 0) & (full_arr["r_ba_res_col"] > 0)
    pok_iso = (full_arr["r_ab_res_iso"] > 0) & (full_arr["r_ba_res_iso"] > 0)
    sel_col05 = fo.empirical_fdp(full_arr["T_col"], pok_col, N_ff, model_col.T_ctl, 0.05)
    sel_col10 = fo.empirical_fdp(full_arr["T_col"], pok_col, N_ff, model_col.T_ctl, 0.10)
    sel_iso05 = fo.empirical_fdp(full_arr["T_iso"], pok_iso, N_ff, model_iso.T_ctl, 0.05)
    sel_iso10 = fo.empirical_fdp(full_arr["T_iso"], pok_iso, N_ff, model_iso.T_ctl, 0.10)

    def _mask_from_idx(idx: np.ndarray) -> np.ndarray:
        m = np.zeros(N_ff, dtype=bool)
        if len(idx) > 0:
            m[idx] = True
        return m

    named_col05 = _mask_from_idx(sel_col05["named_idx"])
    named_col10 = _mask_from_idx(sel_col10["named_idx"])
    named_iso05 = _mask_from_idx(sel_iso05["named_idx"])
    named_iso10 = _mask_from_idx(sel_iso10["named_idx"])
    named_dual05 = named_col05 & named_iso05
    named_dual10 = named_col10 & named_iso10
    dual_idx = np.nonzero(named_dual05)[0]

    ww = fo.finalize_weight_word_tag(full_arr["sw_neu"], ctl_state["sw_ctl"]) if has_lens else {"available": False}

    pick_idx, were_dual05_named = _select_half_c_indices(
        dual_idx.tolist(), full_arr["T_col"], full_arr["T_iso"], layer, max_n=args.half_c_max
    )
    u_hc = w.write_dirs(layer, pick_idx.tolist())
    c_hc = w.write_ceiling(layer, pick_idx.tolist())
    deltas_hc = (dose_ceil * c_hc)[:, None] * u_hc
    piece_pairs = []
    for ni in pick_idx:
        ps = fo.extract_replicated_pieces(full_arr, int(ni), raw_str, k=5)
        piece_pairs.append(([pid for _, _, pid in ps["pos"]], [nid0 for _, _, nid0 in ps["neg"]]))
    hc = fo.eval_half_c_dual(
        rt, lm, layer, texts, lex_gpu, batch_units, args.readout_mode,
        ctl_state["deltas_cols_c"].to(dev), ctl_state["deltas_iso_c"].to(dev), deltas_hc,
        piece_pairs, zscore_var=False,
    )
    hc["were_dual05_named"] = were_dual05_named
    del u_hc, c_hc, deltas_hc

    kl_tot = 0.5 * (0.5 * (full_arr["kl_pos_A"] + full_arr["kl_neg_A"]) + 0.5 * (full_arr["kl_pos_B"] + full_arr["kl_neg_B"]))
    kl_neu = {
        "p10": round(float(np.percentile(kl_tot, 10)), 4),
        "p50": round(float(np.percentile(kl_tot, 50)), 4),
        "p90": round(float(np.percentile(kl_tot, 90)), 4),
    }
    alpha_abs_n = 0.5 * (full_arr["alpha_abs_A"] + full_arr["alpha_abs_B"])

    vs_col = {
        "auc_r_AB": round(fo.auc(full_arr["r_ab_col"], model_col.r_ab_ctl), 4),
        "auc_r_BA": round(fo.auc(full_arr["r_ba_col"], model_col.r_ba_ctl), 4),
        "auc_r_AB_res": round(fo.auc(full_arr["r_ab_res_col"], model_col.r_ab_res_ctl), 4),
        "auc_r_BA_res": round(fo.auc(full_arr["r_ba_res_col"], model_col.r_ba_res_ctl), 4),
        "auc_T": round(fo.auc(full_arr["T_col"], model_col.T_ctl), 4),
        "fdp05": {k: v for k, v in sel_col05.items() if k != "named_idx"},
        "fdp10": {k: v for k, v in sel_col10.items() if k != "named_idx"},
    }
    vs_iso = {
        "auc_r_AB": round(fo.auc(full_arr["r_ab_iso"], model_iso.r_ab_ctl), 4),
        "auc_r_BA": round(fo.auc(full_arr["r_ba_iso"], model_iso.r_ba_ctl), 4),
        "auc_r_AB_res": round(fo.auc(full_arr["r_ab_res_iso"], model_iso.r_ab_res_ctl), 4),
        "auc_r_BA_res": round(fo.auc(full_arr["r_ba_res_iso"], model_iso.r_ba_res_ctl), 4),
        "auc_T": round(fo.auc(full_arr["T_iso"], model_iso.T_ctl), 4),
        "fdp05": {k: v for k, v in sel_iso05.items() if k != "named_idx"},
        "fdp10": {k: v for k, v in sel_iso10.items() if k != "named_idx"},
    }

    ww_summary = None
    if ww.get("available"):
        ww_summary = {
            "auc_own_word_vs_own_ctl": ww["auc_own_word_vs_own_ctl"],
            "ctl_frac_gt0": ww["ctl_frac_gt0"],
            "neu_frac_gt0": ww["neu_frac_gt0"],
            "direct_q05_count": ww["direct_q05_count"],
            "direct_among_dual05": int(np.sum(ww["is_direct"] & named_dual05)),
        }

    # Save final whole-layer numeric NPZ atomically
    npz_out = out_dir / f"L{layer}_forced_odd.npz"
    npz_tmp = out_dir / f"L{layer}_forced_odd.tmp.npz"
    np.savez_compressed(
        npz_tmp,
        T_col=full_arr["T_col"],
        T_iso=full_arr["T_iso"],
        r_ab_col=full_arr["r_ab_col"],
        r_ba_col=full_arr["r_ba_col"],
        r_ab_res_col=full_arr["r_ab_res_col"],
        r_ba_res_col=full_arr["r_ba_res_col"],
        r_ab_iso=full_arr["r_ab_iso"],
        r_ba_iso=full_arr["r_ba_iso"],
        r_ab_res_iso=full_arr["r_ab_res_iso"],
        r_ba_res_iso=full_arr["r_ba_res_iso"],
        T_ctl_col=model_col.T_ctl,
        T_ctl_iso=model_iso.T_ctl,
        named_dual05=named_dual05,
        named_dual10=named_dual10,
        named_col05=named_col05,
        named_iso05=named_iso05,
        named_col10=named_col10,
        named_iso10=named_iso10,
        kl_mean=kl_tot.astype(np.float32),
        alpha_abs=alpha_abs_n.astype(np.float32),
        sw_neu=full_arr["sw_neu"],
        q_direct=ww["q_direct"].astype(np.float32) if ww.get("available") else np.ones(N_ff, dtype=np.float32),
        is_direct=ww["is_direct"] if ww.get("available") else np.zeros(N_ff, dtype=bool),
        topA_p=full_arr["topA_p"],
        topA_n=full_arr["topA_n"],
        topB_p=full_arr["topB_p"],
        topB_n=full_arr["topB_n"],
        topC_p=full_arr["topC_p"],
        topC_n=full_arr["topC_n"],
        posA_onB=full_arr["posA_onB"],
        negA_onB=full_arr["negA_onB"],
        posB_onA=full_arr["posB_onA"],
        negB_onA=full_arr["negB_onA"],
        valC_p=full_arr["valC_p"],
        valC_n=full_arr["valC_n"],
    )
    os.replace(npz_tmp, npz_out)

    meta = {
        "layer": layer,
        "d_ff": N_ff,
        "readout_mode": f"{args.readout_mode}/LEX_L2_cent",
        "dose_stat": args.dose_stat,
        "target_kl": float(args.target_kl),
        "dose_ceil": dose_ceil,
        "n_cols": n_cols,
        "n_iso": n_iso,
        "seed_cols": seed_cols,
        "seed_iso": seed_iso,
        "preflight": pf,
        "kl_neurons": kl_neu,
        "kl_columns": ctl_state["kl_columns"],
        "kl_isotropic": ctl_state["kl_isotropic"],
        "alpha_abs_p50": round(float(np.median(alpha_abs_n)), 4),
        "alpha_abs_p90": round(float(np.percentile(alpha_abs_n, 90)), 4),
        "dual05_count": int(named_dual05.sum()),
        "dual05_pct": round(100.0 * int(named_dual05.sum()) / N_ff, 2),
        "dual10_count": int(named_dual10.sum()),
        "dual10_pct": round(100.0 * int(named_dual10.sum()) / N_ff, 2),
        "vs_columns": vs_col,
        "vs_isotropic": vs_iso,
        "iso_false_vs_cols": ctl_state["iso_false_vs_cols"],
        "iso_vs_cols_auc_T": ctl_state["iso_vs_cols_auc_T"],
        "null_splits_columns": ctl_state["ns_col"],
        "null_splits_isotropic": ctl_state["ns_iso"],
        "half_c": hc,
        "weight_word_diag": ww_summary,
        "git_commit": _git_head_sha(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / f"L{layer}_forced_odd.meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")

    hcc = (hc.get("vs_columns") or {}).get("frac_p05_vs_ctl")
    hci = (hc.get("vs_isotropic") or {}).get("frac_p05_vs_ctl")
    summary_txt = "\n".join(
        [
            f"Whole-Layer forced_odd Census: Layer {layer} (N={N_ff}, C_cols={n_cols}, C_iso={n_iso})",
            "-" * 96,
            f"Dose ceiling      : {dose_ceil:.4f} ({args.dose_stat}={args.target_kl}), {args.readout_mode}/LEX_L2_cent",
            f"KL (p10/p50/p90)  : {kl_neu['p10']:.4f} / {kl_neu['p50']:.4f} / {kl_neu['p90']:.4f} (cols p50={ctl_state['kl_columns']['p50']:.4f}, iso p50={ctl_state['kl_isotropic']['p50']:.4f})",
            f"Dual05 yield      : {meta['dual05_count']} / {N_ff} ({meta['dual05_pct']:.2f}%) | Dual10={meta['dual10_count']} ({meta['dual10_pct']:.2f}%)",
            f"vs columns        : FDP05={vs_col['fdp05']['count']}, T_auc={vs_col['auc_T']:.4f}, rABr={vs_col['auc_r_AB_res']:.4f}, rBAr={vs_col['auc_r_BA_res']:.4f}, Null05={ctl_state['ns_col']['fdp05_false_mean']:.2f} (max={ctl_state['ns_col']['fdp05_false_max']})",
            f"vs isotropic      : FDP05={vs_iso['fdp05']['count']}, T_auc={vs_iso['auc_T']:.4f}, rABr={vs_iso['auc_r_AB_res']:.4f}, rBAr={vs_iso['auc_r_BA_res']:.4f}, Null05={ctl_state['ns_iso']['fdp05_false_mean']:.2f} (max={ctl_state['ns_iso']['fdp05_false_max']})",
            f"Half C replication: {hc['evaluated']} named evaluated -> vs_cols={hcc}, vs_iso={hci} (AUCs={(hc.get('vs_columns') or {}).get('half_c_auc')}/{(hc.get('vs_isotropic') or {}).get('half_c_auc')})",
            f"Weight-word tag   : {ww_summary}",
            f"Elapsed           : {meta['elapsed_s']:.1f}s",
        ]
    ) + "\n"
    (out_dir / f"L{layer}_forced_odd_summary.txt").write_text(summary_txt, encoding="utf-8")

    sc_pseudo = {
        "T": full_arr["T_col"],
        "r_ab_res": full_arr["r_ab_res_col"],
        "r_ba_res": full_arr["r_ba_res_col"],
        "topA_p": full_arr["topA_p"],
        "topA_n": full_arr["topA_n"],
        "topB_p": full_arr["topB_p"],
        "topB_n": full_arr["topB_n"],
        "posA_onB": full_arr["posA_onB"],
        "negA_onB": full_arr["negA_onB"],
        "posB_onA": full_arr["posB_onA"],
        "negB_onA": full_arr["negB_onA"],
    }
    ex_lines, _ = _build_dual_examples_and_named(
        layer=layer,
        dose_ceil=dose_ceil,
        neurons=list(range(N_ff)),
        dual_named_local=dual_idx.tolist(),
        sc_col=sc_pseudo,
        T_iso_neu=full_arr["T_iso"],
        r_ab_res_iso_neu=full_arr["r_ab_res_iso"],
        r_ba_res_iso_neu=full_arr["r_ba_res_iso"],
        offset_col=0,
        n_cols=0,
        raw_str=raw_str,
        tag_info=ww,
        kl_neurons=kl_neu,
        auc_T_col=vs_col["auc_T"],
        auc_T_iso=vs_iso["auc_T"],
        header_tag="WHOLE LAYER",
    )
    (out_dir / f"L{layer}_forced_odd_examples.txt").write_text("\n".join(ex_lines) + "\n", encoding="utf-8")
    print("\n" + summary_txt, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="forced_odd calibration and execution CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_lad = sub.add_parser("ladder", help="Run 64-neuron dose ladder")
    ap_lad.add_argument("--layers", required=True, help="Comma/range layer list, e.g. 30,37,45,54-59")
    ap_lad.add_argument("--out", required=True, help="Output dose_ladder.json path")
    ap_lad.add_argument("--rungs", default=",".join(str(x) for x in fo.LADDER_RUNGS))
    ap_lad.add_argument("--targets", default="0.15,0.25,0.35")
    ap_lad.add_argument("--n-neurons", type=int, default=64)
    ap_lad.add_argument("--data-dir", default=None)
    ap_lad.add_argument("--resume", action="store_true")

    ap_cal = sub.add_parser("calibrate", help="Run cent + proj calibration on L59_pilot and open layers")
    ap_cal.add_argument("--ladder", required=True, help="Input dose_ladder.json path")
    ap_cal.add_argument("--out-dir", required=True, help="Output directory")
    ap_cal.add_argument("--dose-stat", default="kl_p90", choices=("kl_p50", "kl_p90"))
    ap_cal.add_argument("--target-kl", type=float, default=fo.TARGET_KL_DEFAULT)
    ap_cal.add_argument("--n-cols", type=int, default=1024)
    ap_cal.add_argument("--n-iso", type=int, default=256)
    ap_cal.add_argument("--data-dir", default=None)

    ap_tr = sub.add_parser("trial", help="Run pre-registered fresh dual-null trials on L55-L59")
    ap_tr.add_argument("--layers", required=True, help="Comma/range layer list, e.g. 55-59")
    ap_tr.add_argument("--ladder", default="experiments/results/census_forced_odd/dose_ladder.json")
    ap_tr.add_argument("--neurons-dir", default="experiments/results/census_forced_odd/prereg")
    ap_tr.add_argument("--out-dir", default="experiments/results/census_forced_odd/trials")
    ap_tr.add_argument("--dose-stat", default="kl_p90", choices=("kl_p50", "kl_p90"))
    ap_tr.add_argument("--target-kl", type=float, default=fo.TARGET_KL_DEFAULT)
    ap_tr.add_argument("--readout-mode", default="proj", choices=("cent", "proj"))
    ap_tr.add_argument("--n-cols", type=int, default=1024)
    ap_tr.add_argument("--n-iso", type=int, default=1024)
    ap_tr.add_argument("--null-splits", type=int, default=50)
    ap_tr.add_argument("--half-c-max", type=int, default=300)
    ap_tr.add_argument("--data-dir", default=None)

    ap_wh = sub.add_parser("whole", help="Run crash-safe streaming whole-layer forced_odd census (21,504 neurons)")
    ap_wh.add_argument("--layer", required=True, type=int)
    ap_wh.add_argument("--ladder", default="experiments/results/census_forced_odd/dose_ladder.json")
    ap_wh.add_argument("--out-dir", default="experiments/results/census_forced_odd/whole")
    ap_wh.add_argument("--dose-stat", default="kl_p90", choices=("kl_p50", "kl_p90"))
    ap_wh.add_argument("--target-kl", type=float, default=fo.TARGET_KL_DEFAULT)
    ap_wh.add_argument("--readout-mode", default="proj", choices=("cent", "proj"))
    ap_wh.add_argument("--n-cols", type=int, default=4096)
    ap_wh.add_argument("--n-iso", type=int, default=2048)
    ap_wh.add_argument("--chunk-size", type=int, default=512)
    ap_wh.add_argument("--null-splits", type=int, default=50)
    ap_wh.add_argument("--half-c-max", type=int, default=300)
    ap_wh.add_argument("--data-dir", default=None)

    args = ap.parse_args()
    if args.cmd == "ladder":
        cmd_ladder(args)
    elif args.cmd == "calibrate":
        cmd_calibrate(args)
    elif args.cmd == "trial":
        cmd_trial(args)
    elif args.cmd == "whole":
        cmd_whole(args)


if __name__ == "__main__":
    main()
