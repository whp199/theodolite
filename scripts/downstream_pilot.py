#!/usr/bin/env python3
"""Exploratory Direction-3 downstream activation kill-pilot at L25 and L29.

Pushes N=256 source neurons + C=512 columns controls at L25 (1.550 ceil) and L29 (1.047 ceil)
on fixed A/B/C continuations (forward up to layer 54 only), reading signed normalized odd
activation shifts across all page-described L30-L54 census neurons. Tests replication A<->B,
empirical FDP <= 0.05 against columns controls, and matched-selection Half-C replication.
Evaluates against BRIEF/ANSWER Direction-3 kill criterion (T_auc <= 0.55 or FDP05 <= 5/256).
"""
from __future__ import annotations

import argparse
import gc
import json
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
from backend import atlas_ops as ao  # noqa: E402
from backend import census as cz  # noqa: E402
from backend import forced_odd as fo  # noqa: E402
from backend import gap_provenance as gp  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend.causal_check import language_model, model_device, pad_right  # noqa: E402
from backend.runtime_factory import make_runtime, weights_card  # noqa: E402

DOSES_BY_LAYER: Dict[int, float] = {25: 1.550, 29: 1.047}


class _StopAfter54(Exception):
    pass


def _load_target_universe(data_dir: Path, act_scale: Any) -> Dict[str, Any]:
    """Load all page-described neurons across L30-L54 (mass > max ctl mass & mass_shown > max ctl mass_shown)."""
    runs_dir = data_dir / "census" / "runs"
    by_layer: Dict[int, Dict[str, Any]] = {}
    target_meta: List[Tuple[int, int, List[str]]] = []
    offset = 0
    for t_l in range(30, 55):
        vpath = REPO_ROOT / f"experiments/results/census_L{t_l}_whole/L{t_l}_verdict.json"
        run_name = str(json.loads(vpath.read_text(encoding="utf-8"))["run"])
        rpath = runs_dir / run_name
        upath = cz.units_path_of(rpath)
        cpath = cz.collapse_path_of(rpath)
        cset = set(json.loads(cpath.read_text(encoding="utf-8")).get("words") or ()) if cpath.is_file() else set()
        neus = []
        ctls = []
        with upath.open("r", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                if o.get("kind") == "neuron":
                    neus.append(o)
                elif o.get("kind") == "random":
                    ctls.append(o)
        max_m = max(float(c.get("mass", 0.0)) for c in ctls)
        max_ms = max(float(c.get("mass_shown", 0.0)) for c in ctls)
        sel = [u for u in neus if float(u.get("mass", 0.0)) > max_m and float(u.get("mass_shown", 0.0)) > max_ms]
        sel.sort(key=lambda u: int(u["neuron"]))
        if not sel:
            continue
        nids = [int(u["neuron"]) for u in sel]
        ext = np.maximum(np.abs(act_scale["pos"][t_l]), np.abs(act_scale["neg"][t_l])).clip(min=1e-3)[nids].astype(np.float32)
        words_list = []
        for u in sel:
            ws = [str(w) for w, _, _ in (u.get("words") or ()) if str(w) not in cset][:4]
            words_list.append(ws)
            target_meta.append((t_l, int(u["neuron"]), ws))
        by_layer[t_l] = {
            "nids": nids,
            "inv_ext": (1.0 / ext).astype(np.float32),
            "offset": offset,
            "count": len(nids),
        }
        offset += len(nids)
    return {"by_layer": by_layer, "target_meta": target_meta, "M_targets": offset}


@torch.no_grad()
def _prepare_src_group(
    lm: Any,
    bos_id: int,
    dtype: torch.dtype,
    src_layer: int,
    texts: Dict[str, Any],
    gnm: str,
    targets_gpu: Dict[int, Dict[str, Any]],
    M_targets: int,
) -> Dict[str, Any]:
    dev = model_device(lm)
    qlist = texts["groups"][gnm]
    seq_list = [texts["seqs"][q] for q in qlist]
    inp, att, _, _ = pad_right(seq_list, dev)
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    keep_m = ao.keep_mask(inp, att, bos_id, True).to(dtype)
    sc_idx = torch.tensor([texts["scored_pos"][q] for q in qlist], dtype=torch.long, device=dev)
    H = len(qlist)
    h_rows = torch.arange(H, device=dev)[:, None].expand(H, sc_idx.shape[1])

    seen: Dict[str, torch.Tensor] = {}
    base_mean = torch.zeros(M_targets, dtype=torch.float32, device=dev)
    target_layers = sorted(targets_gpu.keys())
    max_t = target_layers[-1]

    hks = [
        lm.layers[src_layer].register_forward_hook(
            lambda _m, _a, o: seen.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
        )
    ]
    for t_l in target_layers:
        info = targets_gpu[t_l]
        idx_g = info["idx"]
        inv_e = info["inv_ext"]
        off_s = info["offset"]
        off_e = off_s + info["count"]

        def _make_base(ig: torch.Tensor, ie: torch.Tensor, os_i: int, oe_i: int):
            def _hk(_m: Any, f_args: Tuple[torch.Tensor, ...]) -> None:
                xs = f_args[0].index_select(-1, ig)
                base_mean[os_i:oe_i] = xs[h_rows, sc_idx, :].float().mean(dim=(0, 1)) * ie

            return _hk

        hks.append(lm.layers[t_l].mlp.down_proj.register_forward_pre_hook(_make_base(idx_g, inv_e, off_s, off_e)))

    def _stop(_m: Any, _a: Any, _o: Any) -> None:
        raise _StopAfter54()

    hks.append(lm.layers[max_t].register_forward_hook(_stop))
    try:
        try:
            lm(input_ids=inp, attention_mask=att, use_cache=False)
        except _StopAfter54:
            pass
    finally:
        for hk in hks:
            hk.remove()
    return {"h_l": seen["h"], "att": att, "pos": pos, "keep": keep_m, "sc_idx": sc_idx, "base_mean": base_mean}


@torch.no_grad()
def _run_downstream_group(
    lm: Any,
    src_layer: int,
    grp: Dict[str, Any],
    deltas: torch.Tensor,
    targets_gpu: Dict[int, Dict[str, Any]],
    M_targets: int,
    batch_units: int,
) -> Dict[str, Any]:
    h_l = grp["h_l"]
    att = grp["att"]
    pos = grp["pos"]
    keep_m = grp["keep"]
    sc_idx = grp["sc_idx"]
    base_mean = grp["base_mean"]
    dev = h_l.device
    M, D = deltas.shape
    H, T, _ = h_l.shape
    h_rows = torch.arange(H, device=dev)[:, None].expand(H, sc_idx.shape[1])

    odd_mat = torch.empty((M, M_targets), dtype=torch.float16, device="cpu")
    odd_l2 = np.zeros(M, dtype=np.float32)
    even_l2 = np.zeros(M, dtype=np.float32)
    mean_abs = np.zeros(M, dtype=np.float32)

    target_layers = sorted(targets_gpu.keys())
    max_t = target_layers[-1]

    for s in range(0, M, batch_units):
        db = deltas[s : s + batch_units]
        b = db.shape[0]
        d_exp = db.to(h_l.dtype)[:, None, None, :] * keep_m[None, :, :, None]
        hp = (h_l[None] + d_exp).reshape(b * H, T, D)
        hn = (h_l[None] - d_exp).reshape(b * H, T, D)
        hs = torch.cat([hp, hn], dim=0)
        del hp, hn, d_exp

        att_r = att.repeat(2 * b, 1)
        pos_r = pos.repeat(2 * b, 1)
        odd_b = torch.empty((b, M_targets), dtype=torch.float32, device=dev)
        even_b = torch.empty((b, M_targets), dtype=torch.float32, device=dev)
        hks = []
        for t_l in target_layers:
            info = targets_gpu[t_l]
            idx_g = info["idx"]
            inv_e = info["inv_ext"]
            off_s = info["offset"]
            off_e = off_s + info["count"]

            def _make_pre(ig: torch.Tensor, ie: torch.Tensor, os: int, oe: int):
                def _hk(_m: Any, f_args: Tuple[torch.Tensor, ...]) -> None:
                    xs = f_args[0].index_select(-1, ig)
                    xv = xs.view(2, b, H, T, -1)[:, :, h_rows, sc_idx, :].float()
                    od = (0.5 * (xv[0] - xv[1])).mean(dim=(1, 2)) * ie[None, :]
                    ev = (0.5 * (xv[0] + xv[1])).mean(dim=(1, 2)) * ie[None, :] - base_mean[None, os:oe]
                    odd_b[:, os:oe] = od
                    even_b[:, os:oe] = ev

                return _hk

            hks.append(lm.layers[t_l].mlp.down_proj.register_forward_pre_hook(_make_pre(idx_g, inv_e, off_s, off_e)))

        hks.append(lm.layers[max_t].register_forward_hook(lambda _m, _a, _o: (_ for _ in ()).throw(_StopAfter54())))
        try:
            try:
                ao.forward_from_layer(lm, src_layer + 1, hs, att_r, pos_r)
            except _StopAfter54:
                pass
        finally:
            for hk in hks:
                hk.remove()
        del hs, att_r, pos_r

        odd_mat[s : s + b] = odd_b.half().cpu()
        odd_l2[s : s + b] = odd_b.norm(dim=-1).cpu().numpy()
        even_l2[s : s + b] = even_b.norm(dim=-1).cpu().numpy()
        mean_abs[s : s + b] = odd_b.abs().mean(dim=-1).cpu().numpy()
        del odd_b, even_b
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    return {"odd": odd_mat, "odd_l2": odd_l2, "even_l2": even_l2, "mean_abs": mean_abs}


def _top_pairs(sc: Dict[str, Any], u_idx: int) -> Tuple[List[int], List[int]]:
    pos: List[int] = []
    neg: List[int] = []
    sp: set[int] = set()
    sn: set[int] = set()
    w = sc["topA_p"].shape[1]
    for j in range(w):
        p1 = int(sc["topA_p"][u_idx, j])
        if sc["posA_onB"][u_idx, j] > 0 and p1 not in sp:
            sp.add(p1)
            pos.append(p1)
        p2 = int(sc["topB_p"][u_idx, j])
        if sc["posB_onA"][u_idx, j] > 0 and p2 not in sp:
            sp.add(p2)
            pos.append(p2)
        n1 = int(sc["topA_n"][u_idx, j])
        if sc["negA_onB"][u_idx, j] < 0 and n1 not in sn:
            sn.add(n1)
            neg.append(n1)
        n2 = int(sc["topB_n"][u_idx, j])
        if sc["negB_onA"][u_idx, j] < 0 and n2 not in sn:
            sn.add(n2)
            neg.append(n2)
    return pos[:5], neg[:5]


def main() -> None:
    ap = argparse.ArgumentParser(description="Direction-3 downstream activation kill-pilot (L25 & L29 -> L30-L54)")
    ap.add_argument("--layers", default="25,29")
    ap.add_argument("--n-neurons", type=int, default=256)
    ap.add_argument("--n-cols", type=int, default=512)
    ap.add_argument("--batch-units", type=int, default=4)
    ap.add_argument("--out-dir", default="experiments/results/census_gap_pilots")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    t0_iso = gp.utc_iso(t0)
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    out_dir = REPO_ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else ai.atlas_dir()

    act_scale = np.load(data_dir / "index/act_scale.npz")
    univ = _load_target_universe(data_dir, act_scale)
    M_targets = int(univ["M_targets"])
    target_meta = univ["target_meta"]
    print(f"Loaded {M_targets} page-described targets across L30-L54 ({len(univ['by_layer'])} layers)", flush=True)

    rt = make_runtime(load_jlens=False).load()
    w4a16_fast.install(rt, "compiled", log=print)
    lm = language_model(rt.hf_model)
    w_ops = ao.Weights(lm, rt)
    dev = model_device(lm)
    bos_id = int(rt.tokenizer.bos_token_id)
    dtype = rt.lm_head_weight.dtype

    targets_gpu = {
        tl: {
            "idx": torch.tensor(info["nids"], dtype=torch.long, device=dev),
            "inv_ext": torch.from_numpy(info["inv_ext"]).to(dev),
            "offset": info["offset"],
            "count": info["count"],
        }
        for tl, info in univ["by_layer"].items()
    }
    all_target_mask = torch.ones(M_targets, dtype=torch.bool, device=dev)
    texts = fo.build_fixed_texts(rt, lm, data_dir=data_dir, include_half_c=True)

    layer_reports: List[Dict[str, Any]] = []
    neuron_ids_meta: Dict[str, List[int]] = {}
    for l in layers:
        tl0 = time.time()
        dose_ceil = float(DOSES_BY_LAYER[l])
        rng = np.random.default_rng(2026112500 + l)
        neurons = sorted(int(i) for i in rng.choice(w_ops.d_ff, size=args.n_neurons, replace=False))
        neuron_ids_meta[str(l)] = neurons
        units = fo.make_units(rt, lm, w_ops, l, neurons, n_cols=args.n_cols, seed=7700 + l, n_iso=0)
        d_cols = torch.from_numpy((dose_ceil * units["ref_ceils_col"])[:, None] * units["col_dirs"]).to(dev)
        d_neu = (dose_ceil * units["c_neu"])[:, None] * units["u_neu"]
        all_deltas = torch.cat([d_cols, d_neu], dim=0)
        del d_cols, d_neu

        U = all_deltas.shape[0]
        ctl_mask = np.zeros(U, dtype=bool)
        ctl_mask[: args.n_cols] = True
        neu_mask = ~ctl_mask

        grpA = _prepare_src_group(lm, bos_id, dtype, l, texts, "A", targets_gpu, M_targets)
        resA = _run_downstream_group(lm, l, grpA, all_deltas, targets_gpu, M_targets, args.batch_units)
        del grpA
        grpB = _prepare_src_group(lm, bos_id, dtype, l, texts, "B", targets_gpu, M_targets)
        resB = _run_downstream_group(lm, l, grpB, all_deltas, targets_gpu, M_targets, args.batch_units)
        del grpB

        l2_a, l2_b = resA["odd_l2"], resB["odd_l2"]
        m_a, m_b = resA["mean_abs"], resB["mean_abs"]
        cov = np.column_stack(
            [
                np.ones(U, dtype=np.float64),
                np.log(0.5 * (m_a + m_b) + 1e-6),
                np.log(0.5 * (l2_a + l2_b) + 1e-8),
                0.5 * (resA["even_l2"] / (l2_a + 1e-8) + resB["even_l2"] / (l2_b + 1e-8)),
            ]
        )
        sc = fo.score_odd_pair_gpu(resA["odd"], resB["odd"], ctl_mask, cov, all_target_mask, zscore_var=False, dev=dev, keep_words=True, top_k=5)
        ev = fo.eval_split_metrics(sc, neu_mask, ctl_mask)
        named_local = [int(x) for x in ev["_named_local_idx"]]

        # Matched-selection Half-C check on downstream target neurons
        T_neu = sc["T"][neu_mask]
        eval_local = named_local if len(named_local) >= 20 else [int(i) for i in np.argsort(-T_neu)[:64]]
        grpC = _prepare_src_group(lm, bos_id, dtype, l, texts, "C", targets_gpu, M_targets)
        sub_deltas = torch.cat([all_deltas[: args.n_cols], all_deltas[args.n_cols + torch.tensor(eval_local, device=dev)]], dim=0)
        resC = _run_downstream_group(lm, l, grpC, sub_deltas, targets_gpu, M_targets, args.batch_units)
        del grpC, sub_deltas

        oddC = resC["odd"].to(device=dev, dtype=torch.float32)
        oddC_ctl = oddC[: args.n_cols]
        oddC_neu = oddC[args.n_cols :]
        normC_ctl, normC_neu = fo._normalize_half_c(oddC_ctl, oddC_neu, all_target_mask, zscore_var=False)
        ctl_pairs = [_top_pairs(sc, c) for c in range(args.n_cols)]
        neu_pairs = [_top_pairs(sc, args.n_cols + li) for li in eval_local]
        hc_res = fo.score_half_c_matched(normC_ctl, ctl_pairs, normC_neu, neu_pairs)
        del oddC, oddC_ctl, oddC_neu, all_deltas, resA, resB, resC

        kill_trig = bool(ev["auc_T"] <= 0.55 or ev["fdp05"]["count"] <= 5)
        top5_ex = []
        show_locs = named_local[:5] if named_local else [int(i) for i in np.argsort(-T_neu)[:5]]
        for li in show_locs:
            gu = args.n_cols + li
            p_ids, n_ids = _top_pairs(sc, gu)
            top5_ex.append(
                {
                    "src_neuron": neurons[li],
                    "T_down": round(float(sc["T"][gu]), 3),
                    "named_fdp05": li in set(named_local),
                    "pos_targets": [
                        {"layer": target_meta[ti][0], "n": target_meta[ti][1], "words": target_meta[ti][2]} for ti in p_ids[:3]
                    ],
                    "neg_targets": [
                        {"layer": target_meta[ti][0], "n": target_meta[ti][1], "words": target_meta[ti][2]} for ti in n_ids[:3]
                    ],
                }
            )

        rep = {
            "layer": int(l),
            "dose_ceil": dose_ceil,
            "n_neurons": args.n_neurons,
            "n_cols": args.n_cols,
            "m_targets": M_targets,
            "auc_T": ev["auc_T"],
            "auc_r_AB_res": ev["auc_r_AB_res"],
            "auc_r_BA_res": ev["auc_r_BA_res"],
            "fdp05": ev["fdp05"],
            "half_c_matched": hc_res,
            "kill_criterion_triggered": kill_trig,
            "kill_rule": "T_auc <= 0.55 or FDP05 <= 5/256",
            "top_examples": top5_ex,
            "elapsed_s": round(time.time() - tl0, 1),
        }
        layer_reports.append(rep)
        torch.cuda.empty_cache()
        gc.collect()
        print(
            f"Downstream L{l:2d} ({rep['elapsed_s']}s): T_auc={ev['auc_T']:.4f} "
            f"(rABr={ev['auc_r_AB_res']:.4f}, rBAr={ev['auc_r_BA_res']:.4f}) "
            f"FDP05={ev['fdp05']['count']}/{args.n_neurons} "
            f"HalfC_matched={hc_res['half_c_auc']} (p05={hc_res['frac_p05_vs_ctl']}) "
            f"kill={kill_trig}",
            flush=True,
        )

    lines = [
        f"Exploratory Direction-3 Downstream Activation Kill-Pilot (Targets: {M_targets} Page-Described L30-L54 Neurons)",
        "-" * 116,
        f"{'L':>3} {'DoseC':>6} {'N/C':>9} {'rABr':>6} {'rBAr':>6} {'T_auc':>6} {'FDP05':>6} {'HC_AUC':>7} {'HC_p05':>7} {'Kill?':>7}",
    ]
    for r in layer_reports:
        hc = r["half_c_matched"]
        lines.append(
            f"{r['layer']:3d} {r['dose_ceil']:6.3f} {r['n_neurons']:3d}/{r['n_cols']:<5d} "
            f"{r['auc_r_AB_res']:6.4f} {r['auc_r_BA_res']:6.4f} {r['auc_T']:6.4f} "
            f"{r['fdp05']['count']:6d} {(hc.get('half_c_auc') or 0):7.4f} {(hc.get('frac_p05_vs_ctl') or 0):7.4f} "
            f"{'KILL' if r['kill_criterion_triggered'] else 'SURVIVE':>7}"
        )
    txt = "\n".join(lines) + "\n"
    prov = gp.build_provenance(
        REPO_ROOT,
        [Path(__file__), REPO_ROOT / "backend" / "forced_odd.py"],
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
        "m_targets": M_targets,
        "elapsed_total_s": round(time.time() - t0, 1),
        "layers": layer_reports,
    }
    (out_dir / "downstream_pilot.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    (out_dir / "downstream_pilot.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    main()
