#!/usr/bin/env python3
"""Experiment 3:
  Part 3A (CPU): Split-half input-token / context-token replication on stored passages.d at L2, L14, L20, L29.
  Part 3B (GPU): Natural-firing ablation (`mul 0` at firing sites vs quiet control neurons) at L14 and L20.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

WORKTREE = Path("/home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief")
MAIN_DATA = Path("/home/eric/bin/ai/theodolite2/atlas_data")
OUT_DIR = WORKTREE / "experiments/results/census_gap_pilots"

sys.path.insert(0, str(WORKTREE))
from backend import atlas_index as ai  # noqa: E402
from backend import atlas_ops as ao  # noqa: E402
from backend.causal_check import language_model, logits_fp32, model_device, pad_right  # noqa: E402
from backend.runtime_factory import make_runtime  # noqa: E402
from backend import w4a16_fast  # noqa: E402


def run_part3a(layers: Sequence[int] = (2, 14, 20, 29), sample_neurons: int = 1024, seed: int = 42) -> Dict[str, Any]:
    print("\n=== Part 3A: CPU Split-Half Input-Side Token/Context Replication ===", flush=True)
    t0 = time.time()
    P = ai.load_folder(MAIN_DATA / "index/passages.d/meta.json")
    vocab = json.loads((MAIN_DATA / "index/vocab.json").read_text(encoding="utf-8"))["vocab"]
    windows = np.asarray(P["windows"])  # [W, seq]
    unigram = np.asarray(P["unigram"], dtype=np.float64)
    tot_tok = float(unigram.sum())
    bg_freq = unigram / max(tot_tok, 1.0)
    W, seq = windows.shape
    N = int(P["pos_win"].shape[1])

    # Build random pseudo-faces matched to typical even/odd event counts (for empirical null)
    rng = np.random.default_rng(seed)
    n_idx = sorted(int(x) for x in rng.choice(N, sample_neurons, replace=False))
    even_wins_all = np.arange(0, W, 2)
    odd_wins_all = np.arange(1, W, 2)

    results = []
    txt_lines = [
        f"Part 3A: Split-Half Input-Side Replication (top passages per face, {sample_neurons} neurons x 2 faces/layer)",
        "-" * 114,
        f"{'L':>3} {'Faces>=4+4':>11} {'PeakRep%':>9} {'NullPeak%':>9} {'Lift3xPeak%':>11} "
        f"{'CtxRep%':>8} {'NullCtx%':>8} {'FDP05_cnt':>9}",
    ]

    for l in layers:
        l_res = {"layer": int(l), "sample_neurons": sample_neurons}
        face_stats = {"eligible": 0, "total_faces": 2 * sample_neurons,
                      "peak_rep": 0, "peak_lift3": 0, "ctx_rep": 0, "ctx_lift3": 0,
                      "scores": []}
        null_scores = []
        null_peak_rep, null_peak_lift3, null_ctx_rep = 0, 0, 0
        null_elig = 0

        for side in ("pos", "neg"):
            win_arr = np.asarray(P[f"{side}_win"][l, n_idx])  # [M, 16]
            at_arr = np.asarray(P[f"{side}_at"][l, n_idx])    # [M, 16]

            for i in range(len(n_idx)):
                ws_i = win_arr[i]
                ats_i = at_arr[i]
                valid = ws_i >= 0
                ev_idx = [j for j in range(len(ws_i)) if valid[j] and (int(ws_i[j]) % 2 == 0)]
                od_idx = [j for j in range(len(ws_i)) if valid[j] and (int(ws_i[j]) % 2 == 1)]
                if len(ev_idx) < 4 or len(od_idx) < 4:
                    continue
                face_stats["eligible"] += 1

                def extract_sets(idxs):
                    pk = collections.Counter()
                    ctx = collections.Counter()
                    for j in idxs:
                        w0, a0 = int(ws_i[j]), int(ats_i[j])
                        t0_id = int(windows[w0, a0])
                        pk[t0_id] += 1
                        # (rel_offset, token_id) for -1, 0, +1
                        ctx[(0, t0_id)] += 1
                        if a0 > 1:
                            ctx[(-1, int(windows[w0, a0 - 1]))] += 1
                        if a0 + 1 < seq and int(windows[w0, a0 + 1]) >= 0:
                            ctx[(1, int(windows[w0, a0 + 1]))] += 1
                    return pk, ctx

                pk_e, ctx_e = extract_sets(ev_idx)
                pk_o, ctx_o = extract_sets(od_idx)

                # Any peak token appearing >= 2 in BOTH halves
                shared_pk = [t for t, ce in pk_e.items() if ce >= 2 and pk_o.get(t, 0) >= 2]
                if shared_pk:
                    face_stats["peak_rep"] += 1
                # Peak token with >= 2 in both halves AND empirical corpus lift >= 3
                shared_pk_l3 = [
                    t for t in shared_pk
                    if ((pk_e[t] + pk_o[t]) / (len(ev_idx) + len(od_idx))) / max(bg_freq[t], 1e-7) >= 3.0
                ]
                if shared_pk_l3:
                    face_stats["peak_lift3"] += 1

                # Any (offset, token) in {-1,0,+1} appearing >= 2 in BOTH halves
                shared_ctx = [(off, t) for (off, t), ce in ctx_e.items() if ce >= 2 and ctx_o.get((off, t), 0) >= 2]
                if shared_ctx:
                    face_stats["ctx_rep"] += 1
                shared_ctx_l3 = [
                    (off, t) for (off, t) in shared_ctx
                    if ((ctx_e[(off, t)] + ctx_o[(off, t)]) / (len(ev_idx) + len(od_idx))) / max(bg_freq[t], 1e-7) >= 3.0
                ]
                if shared_ctx_l3:
                    face_stats["ctx_lift3"] += 1

                # Quantitative log-lift replication score over replicated tokens (or max single-tok log-lift)
                best_sc = 0.0
                for off, t in shared_ctx:
                    rate = min(ctx_e[(off, t)] / len(ev_idx), ctx_o[(off, t)] / len(od_idx))
                    lift = rate / max(bg_freq[t], 1e-6)
                    sc = rate * max(0.0, float(np.log1p(lift)))
                    if sc > best_sc:
                        best_sc = sc
                face_stats["scores"].append(best_sc)

                # Paired null face: draw identical counts (len(ev_idx), len(od_idx)) at random valid positions >= 1
                null_elig += 1
                rew = rng.choice(even_wins_all, size=len(ev_idx), replace=False)
                row = rng.choice(odd_wins_all, size=len(od_idx), replace=False)
                npk_e, nctx_e = collections.Counter(), collections.Counter()
                npk_o, nctx_o = collections.Counter(), collections.Counter()
                for rw, pkc, ctc in ((rew, npk_e, nctx_e), (row, npk_o, nctx_o)):
                    for w_r in rw:
                        pos_r = int(rng.integers(1, seq - 1))
                        while windows[w_r, pos_r] < 0:
                            pos_r = int(rng.integers(1, seq - 1))
                        t_r = int(windows[w_r, pos_r])
                        pkc[t_r] += 1
                        ctc[(0, t_r)] += 1
                        if pos_r > 1:
                            ctc[(-1, int(windows[w_r, pos_r - 1]))] += 1
                        if pos_r + 1 < seq and int(windows[w_r, pos_r + 1]) >= 0:
                            ctc[(1, int(windows[w_r, pos_r + 1]))] += 1
                sh_npk = [t for t, ce in npk_e.items() if ce >= 2 and npk_o.get(t, 0) >= 2]
                if sh_npk:
                    null_peak_rep += 1
                if any(((npk_e[t] + npk_o[t]) / (len(ev_idx) + len(od_idx))) / max(bg_freq[t], 1e-7) >= 3.0 for t in sh_npk):
                    null_peak_lift3 += 1
                sh_nctx = [(off, t) for (off, t), ce in nctx_e.items() if ce >= 2 and nctx_o.get((off, t), 0) >= 2]
                if sh_nctx:
                    null_ctx_rep += 1
                nbest = 0.0
                for off, t in sh_nctx:
                    rate = min(nctx_e[(off, t)] / len(ev_idx), nctx_o[(off, t)] / len(od_idx))
                    lift = rate / max(bg_freq[t], 1e-6)
                    sc = rate * max(0.0, float(np.log1p(lift)))
                    if sc > nbest:
                        nbest = sc
                null_scores.append(nbest)

        el = max(face_stats["eligible"], 1)
        nel = max(null_elig, 1)
        f_sc = np.sort(np.array(face_stats["scores"], dtype=np.float64))[::-1]
        c_sc = np.sort(np.array(null_scores, dtype=np.float64))
        # Empirical FDP at 0.05 across eligible faces
        fdp_cnt = 0
        for j in range(len(f_sc)):
            t = f_sc[j]
            if t <= 0:
                break
            c_ge = len(c_sc) - int(np.searchsorted(c_sc, t, side="left"))
            fdp = (len(f_sc) / (len(c_sc) + 1.0)) * (1.0 + c_ge) / (j + 1.0)
            if fdp <= 0.05:
                fdp_cnt = j + 1

        l_res.update({
            "eligible_faces": face_stats["eligible"],
            "total_faces": face_stats["total_faces"],
            "peak_rep_pct": round(100.0 * face_stats["peak_rep"] / el, 2),
            "null_peak_rep_pct": round(100.0 * null_peak_rep / nel, 2),
            "peak_lift3x_pct": round(100.0 * face_stats["peak_lift3"] / el, 2),
            "null_peak_lift3x_pct": round(100.0 * null_peak_lift3 / nel, 2),
            "ctx_rep_pct": round(100.0 * face_stats["ctx_rep"] / el, 2),
            "null_ctx_rep_pct": round(100.0 * null_ctx_rep / nel, 2),
            "ctx_lift3x_pct": round(100.0 * face_stats["ctx_lift3"] / el, 2),
            "fdp05_faces_of_2048": fdp_cnt,
            "fdp05_share_of_eligible": round(100.0 * fdp_cnt / el, 2),
        })
        results.append(l_res)
        txt_lines.append(
            f"{l:3d} {l_res['eligible_faces']:5d}/{l_res['total_faces']:5d} {l_res['peak_rep_pct']:9.1f} "
            f"{l_res['null_peak_rep_pct']:9.1f} {l_res['peak_lift3x_pct']:11.1f} "
            f"{l_res['ctx_rep_pct']:8.1f} {l_res['null_ctx_rep_pct']:8.1f} {fdp_cnt:9d}"
        )

    print(f"Part 3A done in {time.time() - t0:.1f}s", flush=True)
    return {"layers": results, "text": "\n".join(txt_lines)}


def run_part3b(layers: Sequence[int] = (14, 20), n_neurons: int = 64, seed: int = 77) -> Dict[str, Any]:
    print("\n=== Part 3B: GPU Natural-Firing Ablation KL (L14, L20) ===", flush=True)
    t0 = time.time()
    P = ai.load_folder(MAIN_DATA / "index/passages.d/meta.json")
    vocab = json.loads((MAIN_DATA / "index/vocab.json").read_text(encoding="utf-8"))["vocab"]
    windows = np.asarray(P["windows"])  # [W, seq]
    N = int(P["pos_win"].shape[1])

    rt = make_runtime(load_jlens=False).load()
    w4a16_fast.install(rt, "compiled", log=print)
    lm = language_model(rt.hf_model)
    dev = model_device(lm)
    W_U = rt.lm_head_weight
    bos_id = int(windows[0, 0])

    results = []
    examples = {}
    rng = np.random.default_rng(seed)

    for l in layers:
        cand = sorted(int(x) for x in rng.choice(N, n_neurons * 2, replace=False))
        # Select 64 neurons that have >= 4 valid passages on their stronger face
        picked = []
        for n in cand:
            pos_valid = [i for i in range(P["pos_win"].shape[2]) if int(P["pos_win"][l, n, i]) >= 0]
            neg_valid = [i for i in range(P["neg_win"].shape[2]) if int(P["neg_win"][l, n, i]) >= 0]
            pos_max = float(np.max(np.abs(P["pos_act"][l, n, pos_valid]))) if pos_valid else 0.0
            neg_max = float(np.max(np.abs(P["neg_act"][l, n, neg_valid]))) if neg_valid else 0.0
            face = "pos" if pos_max >= neg_max else "neg"
            vlist = pos_valid if face == "pos" else neg_valid
            if len(vlist) >= 4:
                picked.append((n, face, vlist[:4]))
            if len(picked) == n_neurons:
                break

        # Build sequences: [bos] + window[max(1, at-60) : min(seq, at+9)] (4 passages x 64 neurons = 256)
        seqs, peak_in_seq, meta_p = [], [], []
        for n, face, vlist in picked:
            for slot in vlist:
                w0 = int(P[f"{face}_win"][l, n, slot])
                a0 = int(P[f"{face}_at"][l, n, slot])
                st_pos = max(1, a0 - 60)
                en_pos = min(windows.shape[1], a0 + 9)
                tail = [int(t) for t in windows[w0, st_pos:en_pos] if int(t) >= 0]
                seq = [bos_id] + tail
                pk = 1 + (a0 - st_pos)
                seqs.append(seq)
                peak_in_seq.append(pk)
                meta_p.append((n, face, float(P[f"{face}_act"][l, n, slot]), vocab[int(windows[w0, a0])]))

        inp, att, _, _ = pad_right(seqs, dev)
        pos_t = (att.cumsum(-1) - 1).clamp(min=0)

        # Unpushed base forward: capture hidden states entering layer l AND logits at [pk .. pk+4] + quiet controls
        # For each passage j, pick a random neuron q != n with |act_q(pk)| < 0.05 at that peak
        B = len(seqs)
        V = int(W_U.shape[0])
        D = ao.hidden_size(lm)
        kl_active = np.zeros(B, np.float32)
        kl_quiet = np.zeros(B, np.float32)
        top_shifts_by_row: List[List[Tuple[str, float]]] = [[] for _ in range(B)]

        for s in range(0, B, 32):
            e = min(B, s + 32)
            ib, ab, pb = inp[s:e], att[s:e], pos_t[s:e]
            pks = peak_in_seq[s:e]
            b = e - s

            cap = {}
            def hk_pre(_m, args):
                cap["pre"] = args[0].detach()
            def hk_act(_m, args):
                rows_a = args[0]  # [b, T, N]
                idx_r = torch.arange(b, device=dev)
                pk_t = torch.tensor(pks, dtype=torch.long, device=dev)
                cap["act_pk"] = rows_a[idx_r, pk_t].detach()

            h1 = lm.layers[l].register_forward_pre_hook(hk_pre)
            h2 = lm.layers[l].mlp.down_proj.register_forward_pre_hook(hk_act)
            try:
                out_b = lm(input_ids=ib, attention_mask=ab, use_cache=False)
            finally:
                h1.remove()
                h2.remove()

            # Read base fp32 log_softmax at [pk .. min(len-1, pk+4)] (5 positions per passage)
            eval_offsets = torch.arange(5, device=dev)[None, :]  # [1, 5]
            seq_lens = ab.sum(dim=1)  # [b]
            pk_t = torch.tensor(pks, dtype=torch.long, device=dev)
            pos_mat = torch.minimum(pk_t[:, None] + eval_offsets, (seq_lens - 1)[:, None])  # [b, 5]
            br = torch.arange(b, device=dev)[:, None].expand(b, 5)
            h_b_sel = out_b.last_hidden_state[br, pos_mat].reshape(b * 5, D)
            base_lsm = F.log_softmax(logits_fp32(h_b_sel, W_U), dim=-1).view(b, 5, V)
            del out_b, h_b_sel

            # Verify forward_from_layer(lm, l, cap["pre"], ab, pb) matches? Wait: for mul 0 on neuron n,
            # use down_proj pre-hook inside ao.installed(lm, iv, keep) while running from layer l!
            # Notice layer l's pre_hook captured input ENTERING layer l (so forward_from_layer start=l runs layer l!).
            act_pk = cap["act_pk"]  # [b, N]
            h_in_l = cap["pre"]

            # Build Interventions for active neurons AND matched quiet neurons (< 0.05 abs activation at pk)
            # Zero-out at ALL non-BOS positions (keep_mask) cleanly, reading effect at [pk .. pk+4].
            def run_ablate(target_neurons: List[int]) -> Tuple[np.ndarray, torch.Tensor]:
                muls_list = []
                # Hook down_proj directly with per-row neuron mask [b]
                t_idx = torch.tensor(target_neurons, dtype=torch.long, device=dev)
                k_m = ao.keep_mask(ib, ab, bos_id, True)  # [b, T]

                def zero_pre(_m, args):
                    x = args[0].clone()
                    rows_i = torch.arange(b, device=dev)
                    # Clear at non-BOS positions
                    orig = x[rows_i, :, t_idx]  # [b, T]
                    x[rows_i, :, t_idx] = torch.where(k_m, torch.zeros_like(orig), orig)
                    return (x,) + tuple(args[1:])

                hk_z = lm.layers[l].mlp.down_proj.register_forward_pre_hook(zero_pre)
                try:
                    out_z = ao.forward_from_layer(lm, l, h_in_l, ab, pb)
                finally:
                    hk_z.remove()
                h_z_sel = out_z[br, pos_mat].reshape(b * 5, D)
                z_lsm = F.log_softmax(logits_fp32(h_z_sel, W_U), dim=-1).view(b, 5, V)
                kl = (base_lsm.exp() * (base_lsm - z_lsm)).sum(-1).mean(-1)  # KL(base || ablated) at [pk..pk+4]
                diff_pk = (base_lsm[:, 0] - z_lsm[:, 0])  # base - ablated = what neuron promoted at peak
                del out_z, h_z_sel, z_lsm
                return kl.cpu().numpy(), diff_pk

            active_ns = [meta_p[s + i][0] for i in range(b)]
            quiet_ns = []
            for i in range(b):
                q_cand = (act_pk[i].abs() < 0.05).nonzero(as_tuple=False).reshape(-1)
                q_list = q_cand[q_cand != active_ns[i]]
                q_pick = int(q_list[int(rng.integers(0, len(q_list)))].item()) if len(q_list) > 0 else int((active_ns[i] + 1) % N)
                quiet_ns.append(q_pick)

            k_act, dp_act = run_ablate(active_ns)
            k_qui, _ = run_ablate(quiet_ns)
            kl_active[s:e] = k_act
            kl_quiet[s:e] = k_qui

            # Store top promoted pieces at peak for each row
            top_v, top_i = dp_act.topk(5, dim=-1)
            for i in range(b):
                top_shifts_by_row[s + i] = [(vocab[int(top_i[i, k])], round(float(top_v[i, k]), 2)) for k in range(5)]

            del base_lsm, h_in_l, cap

        # Aggregate per neuron (mean & max over 4 passages)
        neu_mean_kl = [float(np.mean(kl_active[4 * i : 4 * (i + 1)])) for i in range(n_neurons)]
        neu_max_kl = [float(np.max(kl_active[4 * i : 4 * (i + 1)])) for i in range(n_neurons)]
        qui_mean_kl = [float(np.mean(kl_quiet[4 * i : 4 * (i + 1)])) for i in range(n_neurons)]
        above_floor = float(np.mean(np.array(neu_max_kl) > 0.005))

        top10_idx = np.argsort(-np.array(neu_max_kl))[:10]
        examples[str(l)] = []
        for idx_n in top10_idx:
            # Find best passage among the 4
            best_slot = 4 * idx_n + int(np.argmax(kl_active[4 * idx_n : 4 * (idx_n + 1)]))
            n, face, act_val, pk_tok = meta_p[best_slot]
            examples[str(l)].append({
                "neuron": n, "face": face, "act": round(act_val, 2), "peak_tok": pk_tok,
                "max_kl": round(float(kl_active[best_slot]), 4),
                "quiet_kl": round(float(kl_quiet[best_slot]), 4),
                "top_promoted_at_peak": top_shifts_by_row[best_slot],
            })

        results.append({
            "layer": int(l),
            "neurons": n_neurons,
            "passages_per_neuron": 4,
            "active_kl_per_passage": {
                "p50": round(float(np.median(kl_active)), 5),
                "p90": round(float(np.percentile(kl_active, 90)), 5),
                "max": round(float(np.max(kl_active)), 5),
            },
            "quiet_ctl_kl_per_passage": {
                "p50": round(float(np.median(kl_quiet)), 5),
                "p90": round(float(np.percentile(kl_quiet, 90)), 5),
                "max": round(float(np.max(kl_quiet)), 5),
            },
            "active_max_of_4_p50": round(float(np.median(neu_max_kl)), 5),
            "active_max_of_4_p90": round(float(np.percentile(neu_max_kl, 90)), 5),
            "frac_neurons_max_kl_gt_0.005": round(above_floor, 4),
            "frac_neurons_max_kl_gt_0.00045_bf16_floor": round(float(np.mean(np.array(neu_max_kl) > 0.00045)), 4),
        })
        torch.cuda.empty_cache()

    print(f"Part 3B done in {time.time() - t0:.1f}s", flush=True)
    return {"layers": results, "examples": examples}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--gpu-only", action="store_true")
    a = ap.parse_args()

    out = {}
    txt_parts = []
    if (OUT_DIR / "input_side_pilot.json").exists():
        try:
            out = json.loads((OUT_DIR / "input_side_pilot.json").read_text())
        except Exception:
            out = {}

    if not a.gpu_only:
        res3a = run_part3a()
        out["part3a_split_half_lift"] = res3a["layers"]
        txt_parts.append(res3a["text"])
        (OUT_DIR / "input_side_pilot.json").write_text(json.dumps(out, indent=1) + "\n")
        (OUT_DIR / "input_side_pilot.txt").write_text("\n\n".join(txt_parts) + "\n")

    if not a.cpu_only:
        res3b = run_part3b()
        out["part3b_natural_firing_ablation"] = res3b
        lines = [
            "Part 3B: Natural-Firing Ablation KL (`mul 0` at [peak..peak+4] vs matched quiet neuron)",
            "-" * 100,
            f"{'L':>3} {'Act_p50':>9} {'Act_p90':>9} {'Max4_p50':>9} {'Max4_p90':>9} {'Qui_p50':>9} {'Qui_p90':>9} {'>0.005':>7}",
        ]
        for r in res3b["layers"]:
            lines.append(
                f"{r['layer']:3d} {r['active_kl_per_passage']['p50']:9.5f} {r['active_kl_per_passage']['p90']:9.5f} "
                f"{r['active_max_of_4_p50']:9.5f} {r['active_max_of_4_p90']:9.5f} "
                f"{r['quiet_ctl_kl_per_passage']['p50']:9.5f} {r['quiet_ctl_kl_per_passage']['p90']:9.5f} "
                f"{r['frac_neurons_max_kl_gt_0.005']:7.2%}"
            )
        for l_str, exs in res3b["examples"].items():
            lines.append(f"\nTop 10 L{l_str} ablation examples:")
            for e in exs:
                lines.append(f"  n={e['neuron']:5d}({e['face']}) act={e['act']:5.2f} on {e['peak_tok']!r} -> "
                             f"KL={e['max_kl']:.4f} (quiet={e['quiet_kl']:.5f}) promoted={e['top_promoted_at_peak'][:3]}")
        txt_parts.append("\n".join(lines))
        (OUT_DIR / "input_side_pilot.json").write_text(json.dumps(out, indent=1) + "\n")
        (OUT_DIR / "input_side_pilot.txt").write_text("\n\n".join(txt_parts) + "\n")


if __name__ == "__main__":
    main()
