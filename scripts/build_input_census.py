#!/usr/bin/env python3
"""Build full L0-L29 Input-Side Firing Census from stored passages.d (CPU-only)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

WORKTREE = Path("/home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief")
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from backend import atlas_index as ai  # noqa: E402
from backend import forced_odd as fo  # noqa: E402
from backend import input_census as ic  # noqa: E402


EXAMPLE_LAYERS = {0, 2, 8, 14, 20, 25, 29}


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build input-side firing census for specified layers")
    ap.add_argument("--layers", default="0-29", help="Layer list/ranges (default: 0-29)")
    ap.add_argument("--out-dir", default="experiments/results/input_census_L0_L29")
    ap.add_argument("--n-null", type=int, default=4096)
    ap.add_argument("--n-null-splits", type=int, default=20)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    layers = parse_layers_arg(args.layers)
    out_dir = WORKTREE / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = fo.resolve_data_dir(Path(args.data_dir) if args.data_dir else None)

    P = ai.load_folder(data_dir / "index/passages.d/meta.json")
    vocab = json.loads((data_dir / "index/vocab.json").read_text(encoding="utf-8"))["vocab"]
    windows = np.asarray(P["windows"])
    unigram = np.asarray(P["unigram"])
    has_smp = "pos_smp_win" in P and "neg_smp_win" in P

    layer_summaries: List[Dict[str, Any]] = []
    example_lines: List[str] = ["Input-Side Firing Census Top-10 Named Faces (piece/offset/counts/lift only, no corpus text)"]

    for l in layers:
        tl = time.time()
        res = ic.analyze_layer_arrays(
            windows=windows,
            unigram=unigram,
            pos_win=P["pos_win"][l],
            pos_at=P["pos_at"][l],
            neg_win=P["neg_win"][l],
            neg_at=P["neg_at"][l],
            pos_smp_win=P["pos_smp_win"][l] if has_smp else None,
            pos_smp_at=P["pos_smp_at"][l] if has_smp else None,
            neg_smp_win=P["neg_smp_win"][l] if has_smp else None,
            neg_smp_at=P["neg_smp_at"][l] if has_smp else None,
            n_null=args.n_null,
            n_null_splits=args.n_null_splits,
            seed=2026112450 + l,
        )
        npz_path = out_dir / f"L{l:02d}_input_census.npz"
        ic.save_layer_input_census(npz_path, res)
        st = dict(res["stats"])
        st["layer"] = l
        st["elapsed_s"] = round(time.time() - tl, 2)
        layer_summaries.append(st)
        print(
            f"Input L{l:2d} ({st['elapsed_s']}s): elig={st['eligible_faces']}/{st['total_faces']} | "
            f"FDP05={st['named_fdp05_count']} ({st['named_fdp05_pct_elig']}%) | "
            f"Null05={st['null_split_false_mean']} | smp={st['smp_named_rate']} (shuf={st['smp_shuf_rate']})",
            flush=True,
        )

        if l in EXAMPLE_LAYERS:
            named_idx = np.nonzero(res["named_fdp05"].reshape(-1))[0]
            scores_flat = res["score"].reshape(-1)
            order = named_idx[np.argsort(-scores_flat[named_idx])[:10]] if len(named_idx) else []
            N_ff = res["score"].shape[1]
            example_lines.append(f"\n=== Layer {l} (named={st['named_fdp05_count']}/{st['eligible_faces']}) ===")
            for flat_i in order:
                si = int(flat_i) // N_ff
                ni = int(flat_i) % N_ff
                sign = "+" if si == 0 else "-"
                toks = []
                for k in range(3):
                    tid = int(res["top_token_id"][si, ni, k])
                    if tid < 0:
                        break
                    off = int(res["top_offset"][si, ni, k])
                    ce = int(res["top_cnt_e"][si, ni, k])
                    co = int(res["top_cnt_o"][si, ni, k])
                    lift = float(res["top_lift"][si, ni, k])
                    piece = vocab[tid] if tid < len(vocab) else f"#{tid}"
                    toks.append(f"{piece!r}@{off:+d}({ce}/{co}, {lift:.0f}x)")
                sh = int(res["smp_hits"][si, ni])
                st_n = int(res["smp_total"][si, ni])
                example_lines.append(
                    f"  n={ni:5d}{sign} sc={float(res['score'][si, ni]):5.2f} "
                    f"smp={sh}/{st_n} [{' ; '.join(toks)}]"
                )

    (out_dir / "summary.json").write_text(
        json.dumps({"elapsed_total_s": round(time.time() - t0, 1), "layers": layer_summaries}, indent=1) + "\n",
        encoding="utf-8",
    )
    (out_dir / "examples.txt").write_text("\n".join(example_lines) + "\n", encoding="utf-8")

    lines = [
        f"Input-Side Firing Census Summary (passages.d split-half even vs odd, C_null={args.n_null}, FDP <= 0.05)",
        "-" * 118,
        f"{'L':>3} {'Elig/Total':>13} {'FDP05':>7} {'Elig%':>6} {'Tot%':>6} {'FDP10':>7} "
        f"{'Null05':>7} {'Bigram%':>8} {'SmpHit':>7} {'SmpShuf':>8} {'NonNmd':>7}",
    ]
    for s in layer_summaries:
        el_s = f"{s['eligible_faces']}/{s['total_faces']}"
        smp_n = f"{s['smp_named_rate']:.3f}" if s.get("smp_named_rate") is not None else "-"
        smp_s = f"{s['smp_shuf_rate']:.3f}" if s.get("smp_shuf_rate") is not None else "-"
        smp_o = f"{s['smp_non_named_rate']:.3f}" if s.get("smp_non_named_rate") is not None else "-"
        lines.append(
            f"{s['layer']:3d} {el_s:>13} {s['named_fdp05_count']:7d} {s['named_fdp05_pct_elig']:6.1f} "
            f"{s['named_fdp05_pct_total']:6.1f} {s['named_fdp10_count']:7d} {s['null_split_false_mean']:7.2f} "
            f"{s['bigram_rep_pct_elig']:8.1f} {smp_n:>7} {smp_s:>8} {smp_o:>7}"
        )
    txt = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    main()
