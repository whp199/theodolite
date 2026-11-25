#!/usr/bin/env python3
"""Fixed-quiet-control natural-firing ablation KL verification for layers L0-L29.

Verifies on L2, L14, L20, L29 (64 named_fdp05 neurons/layer x 4 top non-BOS passages = 256/layer)
that `mul 0` ablation across the evaluated span `[pk .. min(seq_len-1, pk+4)]` causes large causal
output shift on naturally firing `named_fdp05` neurons compared to strictly quiet non-dead controls
(`max_{t in [pk..pk+4]} |act_q(t)| < 0.05 * extreme(q)` and `extreme(q) > 0.1`).

Never writes raw corpus passage text to output JSON/TXT (only single peak token string and stats).
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
from backend import w4a16_fast  # noqa: E402
from backend.causal_check import language_model, logits_fp32, model_device, pad_right  # noqa: E402
from backend.gap_stats import mannwhitney_greater, wilcoxon_greater  # noqa: E402
from backend.runtime_factory import make_runtime  # noqa: E402


def _auc(a: Sequence[float], b: Sequence[float]) -> float:
    return float(mannwhitney_greater(a, b)[0])


def _wilcoxon_greater(a: Sequence[float], b: Sequence[float]) -> float:
    return float(wilcoxon_greater(a, b))


def _kl_stats(act: np.ndarray, qui: np.ndarray, n_neurons: int, pass_per_n: int = 4) -> Dict[str, Any]:
    max_per_n = [float(np.max(act[pass_per_n * i : pass_per_n * (i + 1)])) for i in range(n_neurons)]
    med_act = float(np.median(act))
    med_qui = float(np.median(qui))
    p90_act = float(np.percentile(act, 90))
    p90_qui = float(np.percentile(qui, 90))
    return {
        "active_p10_p50_p90_max": [
            round(float(np.percentile(act, 10)), 5),
            round(med_act, 5),
            round(p90_act, 5),
            round(float(np.max(act)), 5),
        ],
        "quiet_p10_p50_p90_max": [
            round(float(np.percentile(qui, 10)), 5),
            round(med_qui, 5),
            round(p90_qui, 5),
            round(float(np.max(qui)), 5),
        ],
        "median_ratio": round(med_act / max(med_qui, 1e-8), 1),
        "p90_ratio": round(p90_act / max(p90_qui, 1e-8), 1),
        "auc_active_vs_quiet": round(_auc(act, qui), 4),
        "wilcoxon_p": float(f"{_wilcoxon_greater(act, qui):.3e}"),
        "max4_p50": round(float(np.median(max_per_n)), 5),
        "max4_p90": round(float(np.percentile(max_per_n, 90)), 5),
        "frac_neurons_max4_gt_0.005": round(float(np.mean(np.array(max_per_n) > 0.005)), 4),
        "frac_neurons_max4_gt_0.00045_bf16": round(float(np.mean(np.array(max_per_n) > 0.00045)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fixed-quiet-control natural-firing ablation KL verification")
    ap.add_argument("--layers", default="2,14,20,29")
    ap.add_argument("--census-dir", default="experiments/results/input_census_L0_L29")
    ap.add_argument("--n-neurons", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--quiet-thresh", type=float, default=0.05)
    ap.add_argument("--min-extreme", type=float, default=0.1)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
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
    dev = model_device(lm)
    W_U = rt.lm_head_weight
    V = int(W_U.shape[0])
    D = ao.hidden_size(lm)

    layer_reports: List[Dict[str, Any]] = []
    examples: Dict[str, List[Dict[str, Any]]] = {}

    for l in layers:
        tl0 = time.time()
        ic = np.load(census_dir / f"L{l:02d}_input_census.npz")
        named_fdp05 = ic["named_fdp05"]  # [2, N_ff]
        sc_2d = ic["score"]              # [2, N_ff]

        # Deterministic selection of 64 named_fdp05 neurons having >= 4 non-BOS top-16 passages
        candidates: List[Tuple[int, str, List[int]]] = []
        for n in range(N_ff):
            face_opts: List[Tuple[float, str, List[int]]] = []
            for s_idx, face in ((0, "pos"), (1, "neg")):
                if not bool(named_fdp05[s_idx, n]):
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
            raise RuntimeError(f"L{l}: only {len(candidates)} named candidates with >=4 valid passages (< {args.n_neurons})")
        pick_order = sorted(int(i) for i in rng.choice(len(candidates), size=args.n_neurons, replace=False))
        picked = [candidates[i] for i in pick_order]

        # Build truncated sequences (<= 32 tokens around peak: [bos_id] + up to 31 tokens)
        seqs: List[List[int]] = []
        peak_in_seq: List[int] = []
        meta_p: List[Tuple[int, str, float, str]] = []
        for n, face, slots in picked:
            for sl in slots:
                w0 = int(P[f"{face}_win"][l, n, sl])
                a0 = int(P[f"{face}_at"][l, n, sl])
                st_pos = max(1, a0 - 22)
                en_pos = min(seq_len, st_pos + 31)
                tail = [int(t) for t in windows[w0, st_pos:en_pos] if int(t) >= 0]
                seq = [bos_id] + tail
                pk = 1 + (a0 - st_pos)
                assert 1 <= pk < len(seq) and len(seq) <= 32 and seq[pk] == int(windows[w0, a0]), f"Bad seq L{l} n={n}"
                seqs.append(seq)
                peak_in_seq.append(pk)
                meta_p.append((n, face, float(P[f"{face}_act"][l, n, sl]), vocab[int(windows[w0, a0])]))

        inp, att, _, _ = pad_right(seqs, dev)
        pos_t = (att.cumsum(-1) - 1).clamp(min=0)
        extreme_q_np = np.maximum(np.abs(act_scale["pos"][l]), np.abs(act_scale["neg"][l])).clip(min=1e-3)
        extreme_q_t = torch.from_numpy(extreme_q_np).to(device=dev, dtype=torch.float32)
        non_dead_t = extreme_q_t > float(args.min_extreme)

        B = len(seqs)
        kl_act_pk = np.zeros(B, dtype=np.float32)
        kl_qui_pk = np.zeros(B, dtype=np.float32)
        kl_act_span = np.zeros(B, dtype=np.float32)
        kl_qui_span = np.zeros(B, dtype=np.float32)
        max_norm_act_quiet = np.zeros(B, dtype=np.float32)
        active_norm_act_pk = np.zeros(B, dtype=np.float32)
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
                cap["act_all"] = f_args[0].detach()  # [b, T_b, N_ff]

            h1 = lm.layers[l].register_forward_pre_hook(hk_pre)
            h2 = lm.layers[l].mlp.down_proj.register_forward_pre_hook(hk_act)
            try:
                with torch.no_grad():
                    out_b = lm(input_ids=ib, attention_mask=ab, use_cache=False)
            finally:
                h1.remove()
                h2.remove()

            pos_mat = torch.minimum(pk_t[:, None] + eval_offsets, (seq_lens - 1)[:, None])  # [b, 5]
            in_span_eval = ((pk_t[:, None] + eval_offsets) < seq_lens[:, None]).float()  # [b, 5]
            br = torch.arange(b, device=dev)[:, None].expand(b, 5)
            h_b_sel = out_b.last_hidden_state[br, pos_mat].reshape(b * 5, D)
            base_lsm = F.log_softmax(logits_fp32(h_b_sel, W_U), dim=-1).view(b, 5, V)
            del out_b, h_b_sel

            act_all = cap["act_all"]
            h_in_l = cap["pre"]

            # Strict quiet control selection across ALL 5 evaluated positions [pk .. min(seq_len-1, pk+4)]
            active_ns = [meta_p[s + i][0] for i in range(b)]
            quiet_ns: List[int] = []
            for i in range(b):
                a_n = active_ns[i]
                active_norm_act_pk[s + i] = float(act_all[i, pk_t[i], a_n].abs().item() / extreme_q_np[a_n])
                span_positions = span_mask[i].nonzero(as_tuple=False).reshape(-1)
                span_max_abs = act_all[i, span_positions, :].abs().amax(dim=0).float()  # [N_ff]
                norm_span_max = span_max_abs / extreme_q_t
                elig_q = non_dead_t & (norm_span_max < float(args.quiet_thresh))
                elig_q[a_n] = False
                q_cands = elig_q.nonzero(as_tuple=False).reshape(-1)
                if len(q_cands) == 0:
                    raise RuntimeError(f"L{l} row {s+i} n={a_n}: zero quiet controls with max_norm_act < {args.quiet_thresh}")
                q_pick = int(q_cands[int(rng.integers(0, len(q_cands)))].item())
                quiet_ns.append(q_pick)
                max_norm_act_quiet[s + i] = float(norm_span_max[q_pick].item())

            def run_span_ablate(target_neurons: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
                t_idx = torch.tensor(target_neurons, dtype=torch.long, device=dev)
                rows_i = torch.arange(b, device=dev)

                def zero_span(_m: Any, f_args: Tuple[torch.Tensor, ...]) -> Tuple[Any, ...]:
                    x = f_args[0].clone()
                    orig = x[rows_i, :, t_idx]  # [b, T_b]
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
                kl_per_pos = (base_lsm.exp() * (base_lsm - z_lsm)).sum(dim=-1)  # [b, 5]
                kl_pk_v = kl_per_pos[:, 0]
                kl_span_v = (kl_per_pos * in_span_eval).sum(dim=-1) / in_span_eval.sum(dim=-1).clamp(min=1.0)
                diff_pk = base_lsm[:, 0] - z_lsm[:, 0]
                del out_z, h_z_sel, z_lsm
                return kl_pk_v.cpu().numpy(), kl_span_v.cpu().numpy(), diff_pk

            kp_act, ks_act, dp_act = run_span_ablate(active_ns)
            kp_qui, ks_qui, _ = run_span_ablate(quiet_ns)
            kl_act_pk[s:e] = kp_act
            kl_act_span[s:e] = ks_act
            kl_qui_pk[s:e] = kp_qui
            kl_qui_span[s:e] = ks_qui

            top_v, top_i = dp_act.topk(3, dim=-1)
            for i in range(b):
                top_shifts_by_row[s + i] = [(vocab[int(top_i[i, k])], round(float(top_v[i, k]), 2)) for k in range(3)]
            del base_lsm, h_in_l, act_all, cap

        assert float(np.max(max_norm_act_quiet)) < float(args.quiet_thresh)
        pk_stats = _kl_stats(kl_act_pk, kl_qui_pk, args.n_neurons, pass_per_n=4)
        span_stats = _kl_stats(kl_act_span, kl_qui_span, args.n_neurons, pass_per_n=4)

        neu_max_span = np.array([float(np.max(kl_act_span[4 * i : 4 * (i + 1)])) for i in range(args.n_neurons)])
        top10_n = np.argsort(-neu_max_span)[:10]
        examples[str(l)] = []
        for ni_loc in top10_n:
            best_sl = 4 * int(ni_loc) + int(np.argmax(kl_act_span[4 * int(ni_loc) : 4 * (int(ni_loc) + 1)]))
            nid, face, act_v, pk_tok = meta_p[best_sl]
            examples[str(l)].append(
                {
                    "neuron": nid,
                    "face": face,
                    "act": round(act_v, 2),
                    "peak_tok": pk_tok,
                    "kl_pk": round(float(kl_act_pk[best_sl]), 5),
                    "kl_span": round(float(kl_act_span[best_sl]), 5),
                    "quiet_kl_pk": round(float(kl_qui_pk[best_sl]), 5),
                    "quiet_kl_span": round(float(kl_qui_span[best_sl]), 5),
                    "quiet_max_norm_act": round(float(max_norm_act_quiet[best_sl]), 4),
                    "top_promoted_at_peak": top_shifts_by_row[best_sl],
                }
            )

        l_rep = {
            "layer": l,
            "neurons": args.n_neurons,
            "passages": B,
            "active_norm_act_pk_p50": round(float(np.median(active_norm_act_pk)), 4),
            "quiet_max_norm_act_p50": round(float(np.median(max_norm_act_quiet)), 5),
            "quiet_max_norm_act_max": round(float(np.max(max_norm_act_quiet)), 5),
            "all_quiet_under_thresh": True,
            "kl_pk": pk_stats,
            "kl_span": span_stats,
            "elapsed_s": round(time.time() - tl0, 1),
        }
        layer_reports.append(l_rep)
        del inp, att, pos_t
        torch.cuda.empty_cache()
        gc.collect()
        print(
            f"Ablation L{l:2d} ({l_rep['elapsed_s']}s): quiet_norm_max={l_rep['quiet_max_norm_act_max']} | "
            f"pk AUC={pk_stats['auc_active_vs_quiet']} p50={pk_stats['active_p10_p50_p90_max'][1]}/{pk_stats['quiet_p10_p50_p90_max'][1]} ({pk_stats['median_ratio']}x) | "
            f"span AUC={span_stats['auc_active_vs_quiet']} p50={span_stats['active_p10_p50_p90_max'][1]}/{span_stats['quiet_p10_p50_p90_max'][1]} ({span_stats['median_ratio']}x)",
            flush=True,
        )

    lines = [
        f"Natural-Firing Ablation KL Verification (`mul 0` at [pk..pk+4] vs strict quiet controls < {args.quiet_thresh} extreme)",
        "-" * 136,
        f"{'L':>3} {'QuiNormMax':>10} {'Pk_Act50/90':>15} {'Pk_Qui50/90':>15} {'Pk_AUC':>7} {'Pk_Ratio50':>10} "
        f"{'Sp_Act50/90':>15} {'Sp_Qui50/90':>15} {'Sp_AUC':>7} {'Sp_p':>9} {'>0.005':>7}",
    ]
    for r in layer_reports:
        kp, ks = r["kl_pk"], r["kl_span"]
        lines.append(
            f"{r['layer']:3d} {r['quiet_max_norm_act_max']:10.5f} "
            f"{kp['active_p10_p50_p90_max'][1]:7.5f}/{kp['active_p10_p50_p90_max'][2]:7.5f} "
            f"{kp['quiet_p10_p50_p90_max'][1]:7.5f}/{kp['quiet_p10_p50_p90_max'][2]:7.5f} "
            f"{kp['auc_active_vs_quiet']:7.4f} {kp['median_ratio']:9.1f}x "
            f"{ks['active_p10_p50_p90_max'][1]:7.5f}/{ks['active_p10_p50_p90_max'][2]:7.5f} "
            f"{ks['quiet_p10_p50_p90_max'][1]:7.5f}/{ks['quiet_p10_p50_p90_max'][2]:7.5f} "
            f"{ks['auc_active_vs_quiet']:7.4f} {ks['wilcoxon_p']:9.2e} "
            f"{ks['frac_neurons_max4_gt_0.005']:7.2%}"
        )
    for l_str, exs in examples.items():
        lines.append(f"\nTop 10 L{l_str} natural-firing span ablations:")
        for ex in exs:
            lines.append(
                f"  n={ex['neuron']:5d}({ex['face']}) act={ex['act']:5.2f} on {ex['peak_tok']!r} -> "
                f"kl_pk={ex['kl_pk']:.5f} (q={ex['quiet_kl_pk']:.5f}), kl_span={ex['kl_span']:.5f} (q={ex['quiet_kl_span']:.5f}, "
                f"qnorm={ex['quiet_max_norm_act']:.4f}) promoted={ex['top_promoted_at_peak']}"
            )
    summary_txt = "\n".join(lines) + "\n"
    payload = {
        "elapsed_total_s": round(time.time() - t0, 1),
        "quiet_thresh": float(args.quiet_thresh),
        "min_extreme": float(args.min_extreme),
        "layers": layer_reports,
        "examples": examples,
    }
    (census_dir / "ablation_check.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    (census_dir / "ablation_check.txt").write_text(summary_txt, encoding="utf-8")
    print("\n" + summary_txt, flush=True)


if __name__ == "__main__":
    main()
