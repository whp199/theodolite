#!/usr/bin/env python3
"""Exploratory natural-firing ablation check against a fair random same-layer neuron control.

Compares `mul 0` ablation across `[pk .. min(seq_len-1, pk+4)]` on full-context passages
(no 22-piece context truncation) between `named_fdp05` / `raw_2plus2` faces and random non-dead
same-layer control neurons evaluated on the EXACT SAME passages, with neuron-clustered Wilcoxon
tests (N=64 neurons, aggregating 4 passages per neuron) and bf16 floor reporting.

No pass/fail bar is defined; never marks any layer "PASSED" or "verified", and never writes raw
corpus passage text into outputs. Requires explicit owner approval to execute on GPU.
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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import atlas_index as ai  # noqa: E402
from backend import atlas_ops as ao  # noqa: E402
from backend import gap_provenance as gp  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend.causal_check import language_model, logits_fp32, model_device, pad_right  # noqa: E402
from backend.gap_stats import mannwhitney_greater, wilcoxon_greater  # noqa: E402
from backend.runtime_factory import make_runtime, weights_card  # noqa: E402

BF16_NOISE_FLOOR = 0.00045


def _kl_stats(act: np.ndarray, ctl: np.ndarray, n_neurons: int, pass_per_n: int = 4) -> Dict[str, Any]:
    """Summarize per-passage and neuron-clustered KL distributions vs fair random-neuron control."""
    act_n_mean = np.array([float(np.mean(act[pass_per_n * i : pass_per_n * (i + 1)])) for i in range(n_neurons)])
    ctl_n_mean = np.array([float(np.mean(ctl[pass_per_n * i : pass_per_n * (i + 1)])) for i in range(n_neurons)])
    act_n_max = np.array([float(np.max(act[pass_per_n * i : pass_per_n * (i + 1)])) for i in range(n_neurons)])
    ctl_n_max = np.array([float(np.max(ctl[pass_per_n * i : pass_per_n * (i + 1)])) for i in range(n_neurons)])

    med_act = float(np.median(act))
    med_ctl = float(np.median(ctl))
    p90_act = float(np.percentile(act, 90))
    p90_ctl = float(np.percentile(ctl, 90))
    auc_pass, _ = mannwhitney_greater(act, ctl)
    auc_mean4, _ = mannwhitney_greater(act_n_mean, ctl_n_mean)
    wilcoxon_mean4_p = wilcoxon_greater(act_n_mean, ctl_n_mean)
    wilcoxon_max4_p = wilcoxon_greater(act_n_max, ctl_n_max)

    return {
        "target_p10_p50_p90_max": [
            round(float(np.percentile(act, 10)), 5),
            round(med_act, 5),
            round(p90_act, 5),
            round(float(np.max(act)), 5),
        ],
        "control_p10_p50_p90_max": [
            round(float(np.percentile(ctl, 10)), 5),
            round(med_ctl, 5),
            round(p90_ctl, 5),
            round(float(np.max(ctl)), 5),
        ],
        "median_ratio": round(med_act / max(med_ctl, 1e-8), 2),
        "p90_ratio": round(p90_act / max(p90_ctl, 1e-8), 2),
        "auc_passage_target_vs_random_ctl": round(float(auc_pass), 4),
        "auc_neuron_mean4_vs_random_ctl": round(float(auc_mean4), 4),
        "wilcoxon_neuron_mean4_p": float(f"{wilcoxon_mean4_p:.3e}"),
        "wilcoxon_neuron_max4_p": float(f"{wilcoxon_max4_p:.3e}"),
        "max4_p50": round(float(np.median(act_n_max)), 5),
        "max4_p90": round(float(np.percentile(act_n_max, 90)), 5),
        "frac_neurons_mean4_above_ctl": round(float(np.mean(act_n_mean > ctl_n_mean)), 4),
        "frac_neurons_max4_gt_0.005": round(float(np.mean(act_n_max > 0.005)), 4),
        "frac_neurons_mean4_gt_bf16_floor": round(float(np.mean(act_n_mean > BF16_NOISE_FLOOR)), 4),
        "frac_neurons_max4_gt_bf16_floor": round(float(np.mean(act_n_max > BF16_NOISE_FLOOR)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Exploratory natural-firing ablation vs random same-layer neuron control")
    ap.add_argument("--layers", default="2,14,20,29")
    ap.add_argument("--census-dir", default="experiments/results/input_census_L0_L29")
    ap.add_argument("--n-neurons", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--min-extreme", type=float, default=0.1)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    t0_iso = gp.utc_iso(t0)
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    census_dir = REPO_ROOT / args.census_dir if not Path(args.census_dir).is_absolute() else Path(args.census_dir)
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else ai.atlas_dir()

    P = ai.load_folder(data_dir / "index/passages.d/meta.json")
    act_scale = np.load(data_dir / "index/act_scale.npz")
    vocab = json.loads((data_dir / "index/vocab.json").read_text(encoding="utf-8"))["vocab"]
    windows = np.asarray(P["windows"])
    W_tot, seq_len = windows.shape
    N_ff = int(P["pos_win"].shape[1])
    bos_id = int(windows[0, 0])

    rt = make_runtime(load_jlens=False).load()
    w4a16_fast.install(rt, "compiled", log=print)
    lm = language_model(rt.hf_model)
    w_ops = ao.Weights(lm, rt)
    dev = model_device(lm)
    W_U = rt.lm_head_weight
    V = int(W_U.shape[0])
    D = ao.hidden_size(lm)

    layer_reports: List[Dict[str, Any]] = []
    examples: Dict[str, List[Dict[str, Any]]] = {}
    neuron_ids_by_layer: Dict[str, Dict[str, Any]] = {}

    for l in layers:
        tl0 = time.time()
        ic = np.load(census_dir / f"L{l:02d}_input_census.npz")
        sel_mask = ic["raw_2plus2"] if "raw_2plus2" in ic.files else ic["named_fdp05"]
        sc_2d = ic["score"]

        candidates: List[Tuple[int, str, List[int]]] = []
        for n in range(N_ff):
            face_opts: List[Tuple[float, str, List[int]]] = []
            for s_idx, face in ((0, "pos"), (1, "neg")):
                if not bool(sel_mask[s_idx, n]):
                    continue
                ws = P[f"{face}_win"][l, n]
                ats = P[f"{face}_at"][l, n]
                slots = [
                    k for k in range(len(ws))
                    if 0 <= int(ws[k]) < W_tot and 1 <= int(ats[k]) < seq_len and int(windows[int(ws[k]), int(ats[k])]) >= 0
                ]
                if len(slots) >= 4:
                    face_opts.append((float(sc_2d[s_idx, n]), face, slots[:4]))
            if face_opts:
                face_opts.sort(key=lambda x: -x[0])
                _, best_face, best_slots = face_opts[0]
                candidates.append((n, best_face, best_slots))

        rng = np.random.default_rng(2026112490 + l)
        if len(candidates) < args.n_neurons:
            raise RuntimeError(f"L{l}: only {len(candidates)} candidate neurons with >=4 valid passages (< {args.n_neurons})")
        pick_order = sorted(int(i) for i in rng.choice(len(candidates), size=args.n_neurons, replace=False))
        picked = [candidates[i] for i in pick_order]

        extreme_q_np = np.maximum(np.abs(act_scale["pos"][l]), np.abs(act_scale["neg"][l])).clip(min=1e-3)
        non_dead_idx = np.flatnonzero(extreme_q_np > float(args.min_extreme))
        if len(non_dead_idx) < 2:
            raise RuntimeError(f"L{l}: insufficient non-dead neurons for random control")

        # Pair each picked target neuron deterministically with a random non-dead same-layer control neuron.
        ctl_map: Dict[int, int] = {}
        for n, _, _ in picked:
            c_pick = int(non_dead_idx[int(rng.integers(0, len(non_dead_idx)))])
            while c_pick == n:
                c_pick = int(non_dead_idx[int(rng.integers(0, len(non_dead_idx)))])
            ctl_map[n] = c_pick

        target_list = [n for n, _, _ in picked]
        ctl_list = [ctl_map[n] for n in target_list]
        write_ceils_tgt = w_ops.write_ceiling(l, target_list).detach().cpu().numpy()
        write_ceils_ctl = w_ops.write_ceiling(l, ctl_list).detach().cpu().numpy()
        write_norm_ratio_by_i = write_ceils_tgt / np.maximum(write_ceils_ctl, 1e-8)

        neuron_ids_by_layer[str(l)] = {"targets": target_list, "random_controls": ctl_list}

        # Build FULL-CONTEXT window sequences (pos 1..min(seq_len, a0 + 5)) with [bos_id] prefix (no 22-piece truncation)
        seqs: List[List[int]] = []
        peak_in_seq: List[int] = []
        meta_p: List[Tuple[int, int, str, float, str]] = []
        for idx_n, (n, face, slots) in enumerate(picked):
            c_n = ctl_list[idx_n]
            for sl in slots:
                w0 = int(P[f"{face}_win"][l, n, sl])
                a0 = int(P[f"{face}_at"][l, n, sl])
                en_pos = min(seq_len, a0 + 5)
                tail = [int(t) for t in windows[w0, 1:en_pos] if int(t) >= 0]
                seq = [bos_id] + tail
                # Peak is exact position a0 inside windows[w0, 1:en_pos] shifted by +1 for BOS if no negative pad skipped
                pk = 1 + sum(1 for t in windows[w0, 1:a0] if int(t) >= 0)
                assert 1 <= pk < len(seq) and seq[pk] == int(windows[w0, a0]), f"Bad seq L{l} n={n}"
                seqs.append(seq)
                peak_in_seq.append(pk)
                meta_p.append((n, c_n, face, float(P[f"{face}_act"][l, n, sl]), vocab[int(windows[w0, a0])]))

        inp, att, _, _ = pad_right(seqs, dev)
        pos_t = (att.cumsum(-1) - 1).clamp(min=0)

        B = len(seqs)
        kl_act_pk = np.zeros(B, dtype=np.float32)
        kl_ctl_pk = np.zeros(B, dtype=np.float32)
        kl_act_span = np.zeros(B, dtype=np.float32)
        kl_ctl_span = np.zeros(B, dtype=np.float32)
        ctl_norm_act_pk = np.zeros(B, dtype=np.float32)
        active_norm_act_pk = np.zeros(B, dtype=np.float32)
        noop_pk_vals = np.array([], dtype=np.float32)
        noop_span_vals = np.array([], dtype=np.float32)
        top_shifts_by_row: List[List[Tuple[str, float]]] = [[] for _ in range(B)]
        eval_offsets = torch.arange(5, device=dev)[None, :]

        for s in range(0, B, args.batch_size):
            e = min(B, s + args.batch_size)
            ib, ab, pb = inp[s:e], att[s:e], pos_t[s:e]
            b = e - s
            pk_t = torch.tensor(peak_in_seq[s:e], dtype=torch.long, device=dev)
            seq_lens = ab.sum(dim=1)
            end_t = torch.minimum(pk_t + 4, seq_lens - 1)
            T_b = ib.shape[1]
            t_grid = torch.arange(T_b, device=dev)[None, :]
            span_mask = (t_grid >= pk_t[:, None]) & (t_grid <= end_t[:, None]) & (ab > 0)

            cap: Dict[str, torch.Tensor] = {}

            def hk_pre(_m: Any, f_args: Tuple[torch.Tensor, ...]) -> None:
                cap["pre"] = f_args[0].detach()

            def hk_act(_m: Any, f_args: Tuple[torch.Tensor, ...]) -> None:
                cap["act_all"] = f_args[0].detach()

            h1 = lm.layers[l].register_forward_pre_hook(hk_pre)
            h2 = lm.layers[l].mlp.down_proj.register_forward_pre_hook(hk_act)
            try:
                with torch.no_grad():
                    out_b = lm(input_ids=ib, attention_mask=ab, use_cache=False)
            finally:
                h1.remove()
                h2.remove()

            pos_mat = torch.minimum(pk_t[:, None] + eval_offsets, (seq_lens - 1)[:, None])
            in_span_eval = ((pk_t[:, None] + eval_offsets) < seq_lens[:, None]).float()
            br = torch.arange(b, device=dev)[:, None].expand(b, 5)
            h_b_sel = out_b.last_hidden_state[br, pos_mat].reshape(b * 5, D)
            base_lsm = F.log_softmax(logits_fp32(h_b_sel, W_U), dim=-1).view(b, 5, V)
            del out_b, h_b_sel

            act_all = cap["act_all"]
            h_in_l = cap["pre"]
            active_ns = [meta_p[s + i][0] for i in range(b)]
            ctl_ns = [meta_p[s + i][1] for i in range(b)]
            for i in range(b):
                a_n, c_n = active_ns[i], ctl_ns[i]
                active_norm_act_pk[s + i] = float(act_all[i, pk_t[i], a_n].abs().item() / extreme_q_np[a_n])
                ctl_norm_act_pk[s + i] = float(act_all[i, pk_t[i], c_n].abs().item() / extreme_q_np[c_n])

            def run_span_ablate(target_neurons: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
                t_idx = torch.tensor(target_neurons, dtype=torch.long, device=dev)
                rows_i = torch.arange(b, device=dev)

                def zero_span(_m: Any, f_args: Tuple[torch.Tensor, ...]) -> Tuple[Any, ...]:
                    x = f_args[0].clone()
                    orig = x[rows_i, :, t_idx]
                    x[rows_i, :, t_idx] = torch.where(span_mask, torch.zeros_like(orig), orig)
                    return (x,) + tuple(f_args[1:])

                hk_z = lm.layers[l].mlp.down_proj.register_forward_pre_hook(zero_span)
                try:
                    with torch.no_grad():
                        out_z = ao.forward_from_layer(lm, l, h_in_l, ab, pb)
                finally:
                    hk_z.remove()
                h_z_sel = out_z[br, pos_mat].reshape(b * 5, D)
                z_lsm = F.log_softmax(logits_fp32(h_z_sel, W_U), dim=-1).view(b, 5, V)
                kl_per_pos = (base_lsm.exp() * (base_lsm - z_lsm)).sum(dim=-1)
                kl_pk_v = kl_per_pos[:, 0]
                kl_span_v = (kl_per_pos * in_span_eval).sum(dim=-1) / in_span_eval.sum(dim=-1).clamp(min=1.0)
                diff_pk = base_lsm[:, 0] - z_lsm[:, 0]
                del out_z, h_z_sel, z_lsm
                return kl_pk_v.cpu().numpy(), kl_span_v.cpu().numpy(), diff_pk

            kp_act, ks_act, dp_act = run_span_ablate(active_ns)
            kp_ctl, ks_ctl, _ = run_span_ablate(ctl_ns)
            if s == 0:
                with torch.no_grad():
                    out_noop = ao.forward_from_layer(lm, l, h_in_l, ab, pb)
                h_n_sel = out_noop[br, pos_mat].reshape(b * 5, D)
                n_lsm = F.log_softmax(logits_fp32(h_n_sel, W_U), dim=-1).view(b, 5, V)
                kl_n_pos = (base_lsm.exp() * (base_lsm - n_lsm)).sum(dim=-1)
                noop_pk_vals = kl_n_pos[:, 0].cpu().numpy()
                noop_span_vals = ((kl_n_pos * in_span_eval).sum(dim=-1) / in_span_eval.sum(dim=-1).clamp(min=1.0)).cpu().numpy()
                del out_noop, h_n_sel, n_lsm, kl_n_pos
            kl_act_pk[s:e] = kp_act
            kl_act_span[s:e] = ks_act
            kl_ctl_pk[s:e] = kp_ctl
            kl_ctl_span[s:e] = ks_ctl

            top_v, top_i = dp_act.topk(3, dim=-1)
            for i in range(b):
                top_shifts_by_row[s + i] = [(vocab[int(top_i[i, k])], round(float(top_v[i, k]), 2)) for k in range(3)]
            del base_lsm, h_in_l, act_all, cap
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        pk_stats = _kl_stats(kl_act_pk, kl_ctl_pk, args.n_neurons, pass_per_n=4)
        span_stats = _kl_stats(kl_act_span, kl_ctl_span, args.n_neurons, pass_per_n=4)

        neu_max_span = np.array([float(np.max(kl_act_span[4 * i : 4 * (i + 1)])) for i in range(args.n_neurons)])
        top10_n = np.argsort(-neu_max_span)[:10]
        examples[str(l)] = []
        for ni_loc in top10_n:
            best_sl = 4 * int(ni_loc) + int(np.argmax(kl_act_span[4 * int(ni_loc) : 4 * (int(ni_loc) + 1)]))
            nid, c_n, face, act_v, pk_tok = meta_p[best_sl]
            examples[str(l)].append({
                "neuron": nid,
                "random_control": c_n,
                "face": face,
                "act": round(act_v, 2),
                "peak_tok": pk_tok,
                "kl_pk": round(float(kl_act_pk[best_sl]), 5),
                "kl_span": round(float(kl_act_span[best_sl]), 5),
                "ctl_kl_pk": round(float(kl_ctl_pk[best_sl]), 5),
                "ctl_kl_span": round(float(kl_ctl_span[best_sl]), 5),
                "ctl_norm_act_pk": round(float(ctl_norm_act_pk[best_sl]), 4),
                "write_norm_ratio": round(float(write_norm_ratio_by_i[int(ni_loc)]), 4),
                "top_promoted_at_peak": top_shifts_by_row[best_sl],
            })

        l_rep = {
            "layer": l,
            "neurons": args.n_neurons,
            "passages": B,
            "control_kind": "random_same_layer_neuron",
            "active_norm_act_pk_p50": round(float(np.median(active_norm_act_pk)), 4),
            "control_norm_act_pk_p50": round(float(np.median(ctl_norm_act_pk)), 4),
            "write_norm_ratio_p50": round(float(np.median(write_norm_ratio_by_i)), 4),
            "bf16_noise_floor": BF16_NOISE_FLOOR,
            "noop_bf16_floor_pk_p50": round(float(np.median(noop_pk_vals)), 5) if noop_pk_vals.size else None,
            "noop_bf16_floor_pk_p90": round(float(np.percentile(noop_pk_vals, 90)), 5) if noop_pk_vals.size else None,
            "noop_bf16_floor_span_p50": round(float(np.median(noop_span_vals)), 5) if noop_span_vals.size else None,
            "noop_bf16_floor_span_p90": round(float(np.percentile(noop_span_vals, 90)), 5) if noop_span_vals.size else None,
            "kl_pk": pk_stats,
            "kl_span": span_stats,
            "elapsed_s": round(time.time() - tl0, 1),
        }
        layer_reports.append(l_rep)
        del inp, att, pos_t
        torch.cuda.empty_cache()
        gc.collect()
        print(
            f"Ablation L{l:2d} ({l_rep['elapsed_s']}s): pk_mean4_AUC={pk_stats['auc_neuron_mean4_vs_random_ctl']} "
            f"(p={pk_stats['wilcoxon_neuron_mean4_p']:.2e}) | "
            f"span_mean4_AUC={span_stats['auc_neuron_mean4_vs_random_ctl']} (p={span_stats['wilcoxon_neuron_mean4_p']:.2e})",
            flush=True,
        )

    lines = [
        "Exploratory Natural-Firing Ablation vs Random Same-Layer Neuron Control (Full Context, Neuron-Clustered, No Pass Bar)",
        "-" * 158,
        f"{'L':>3} {'W_ratio':>7} {'NoopPk50':>10} {'Pk_Tgt50':>10} {'Pk_Ctl50':>10} {'Pk_AUCn':>8} {'Pk_p_n':>10} "
        f"{'NoopSp50':>10} {'Sp_Tgt50':>10} {'Sp_Ctl50':>10} {'Sp_AUCn':>8} {'Sp_p_n':>10} {'>bf16':>7}",
    ]
    for r in layer_reports:
        kp, ks = r["kl_pk"], r["kl_span"]
        lines.append(
            f"{r['layer']:3d} {r['write_norm_ratio_p50']:7.3f} "
            f"{(r.get('noop_bf16_floor_pk_p50') or 0.0):10.5f} "
            f"{kp['target_p10_p50_p90_max'][1]:10.5f} {kp['control_p10_p50_p90_max'][1]:10.5f} "
            f"{kp['auc_neuron_mean4_vs_random_ctl']:8.4f} {kp['wilcoxon_neuron_mean4_p']:10.2e} "
            f"{(r.get('noop_bf16_floor_span_p50') or 0.0):10.5f} "
            f"{ks['target_p10_p50_p90_max'][1]:10.5f} {ks['control_p10_p50_p90_max'][1]:10.5f} "
            f"{ks['auc_neuron_mean4_vs_random_ctl']:8.4f} {ks['wilcoxon_neuron_mean4_p']:10.2e} "
            f"{ks['frac_neurons_mean4_gt_bf16_floor']:7.2%}"
        )
    summary_txt = "\n".join(lines) + "\n"
    prov = gp.build_provenance(
        REPO_ROOT,
        [Path(__file__), REPO_ROOT / "backend" / "gap_stats.py"],
        started_at=t0_iso,
        finished_at=gp.utc_iso(),
        model_id=getattr(rt, "model_id", None),
        weights=weights_card(rt),
        neuron_ids=neuron_ids_by_layer,
    )
    payload = {
        **prov,
        "exploratory": True,
        "control_kind": "random_same_layer_neuron",
        "context_truncation": "none (pos 1..min(seq_len, pk+4))",
        "bf16_noise_floor": BF16_NOISE_FLOOR,
        "elapsed_total_s": round(time.time() - t0, 1),
        "min_extreme": float(args.min_extreme),
        "layers": layer_reports,
        "examples": examples,
    }
    (census_dir / "ablation_check.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    (census_dir / "ablation_check.txt").write_text(summary_txt, encoding="utf-8")
    print("\n" + summary_txt, flush=True)


if __name__ == "__main__":
    main()
