#!/usr/bin/env python3
"""Build exploratory L0-L29 document-disjoint causal token-trigger tables from passages.d (CPU-only)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import atlas_index as ai  # noqa: E402
from backend import gap_provenance as gp  # noqa: E402
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
    ap = argparse.ArgumentParser(description="Build exploratory input-side firing tables for specified layers")
    ap.add_argument("--layers", default="0-29", help="Layer list/ranges (default: 0-29)")
    ap.add_argument("--out-dir", default="experiments/results/input_census_L0_L29")
    ap.add_argument("--n-null", type=int, default=4096)
    ap.add_argument("--n-null-splits", type=int, default=20)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    t0_iso = gp.utc_iso(t0)
    layers = parse_layers_arg(args.layers)
    out_dir = REPO_ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else ai.atlas_dir()

    P = ai.load_folder(data_dir / "index/passages.d/meta.json")
    p_meta = P.get("meta") if isinstance(P.get("meta"), dict) else {}
    model_id = p_meta.get("model")
    weights = p_meta.get("weights")
    vocab = json.loads((data_dir / "index/vocab.json").read_text(encoding="utf-8"))["vocab"]
    windows = np.asarray(P["windows"])
    unigram = np.asarray(P["unigram"])
    N_ff = int(P["pos_win"].shape[1])
    has_smp = "pos_smp_win" in P and "neg_smp_win" in P

    layer_summaries: List[Dict[str, Any]] = []
    example_lines: List[str] = [
        "Exploratory Input-Side Token-Trigger Examples (document-disjoint, causal offsets -1/0 only; +1 shown separately)"
    ]

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
            f"raw_2+2={st['raw_2plus2_count']} ({st['raw_2plus2_pct_elig']}%) | "
            f"randFDP05={st['named_fdp05_count']} | otherFaceNull={st['other_face_null_rate']:.3f} | "
            f"smp={st['smp_named_rate']} (freq={st['smp_freq_rate']}, shuf={st['smp_shuf_rate']})",
            flush=True,
        )

        if l in EXAMPLE_LAYERS:
            named_idx = np.nonzero(res["raw_2plus2"].reshape(-1))[0]
            scores_flat = res["score"].reshape(-1)
            order = named_idx[np.argsort(-scores_flat[named_idx])[:10]] if len(named_idx) else []
            example_lines.append(f"\n=== Layer {l} (raw_2+2={st['raw_2plus2_count']}/{st['eligible_faces']}) ===")
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
                nxts = []
                for k in range(2):
                    tid = int(res["next_token_id"][si, ni, k])
                    if tid < 0:
                        break
                    nxts.append(f"{vocab[tid] if tid < len(vocab) else f'#{tid}'!r}({int(res['next_cnt_e'][si, ni, k])}/{int(res['next_cnt_o'][si, ni, k])})")
                sh = int(res["smp_hits"][si, ni])
                fh = int(res["smp_freq_hits"][si, ni])
                st_n = int(res["smp_total"][si, ni])
                nxt_s = f" | next(+1)={nxts}" if nxts else ""
                example_lines.append(
                    f"  n={ni:5d}{sign} sc={float(res['score'][si, ni]):5.2f} "
                    f"smp={sh}/{st_n}(freq={fh}) [{' ; '.join(toks)}]{nxt_s}"
                )

    prov = gp.build_provenance(
        REPO_ROOT,
        [REPO_ROOT / "backend" / "input_census.py", Path(__file__)],
        started_at=t0_iso,
        finished_at=gp.utc_iso(),
        model_id=model_id,
        weights=weights,
        neuron_ids="all",
    )
    summary_payload = {
        **prov,
        "exploratory": True,
        "split_kind": "document_disjoint",
        "eos_id": 1,
        "offsets": [-1, 0],
        "passages_corpus": p_meta.get("corpus"),
        "neuron_count": N_ff,
        "elapsed_total_s": round(time.time() - t0, 1),
        "layers": layer_summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=1) + "\n", encoding="utf-8")
    (out_dir / "examples.txt").write_text("\n".join(example_lines) + "\n", encoding="utf-8")

    lines = [
        f"Exploratory Input-Side Token Triggers (passages.d document-disjoint, causal offsets -1/0, C_null={args.n_null})",
        "-" * 134,
        f"{'L':>3} {'Elig/Total':>13} {'Raw2+2':>7} {'Elig%':>6} {'RandFDP05':>9} {'OthFaceNull':>11} "
        f"{'Top100%':>7} {'Bigram%':>8} {'Next+1%':>8} {'SmpHit':>7} {'SmpFreq':>8} {'SmpShuf':>8}",
    ]
    for s in layer_summaries:
        el_s = f"{s['eligible_faces']}/{s['total_faces']}"
        smp_n = f"{s['smp_named_rate']:.3f}" if s.get("smp_named_rate") is not None else "-"
        smp_f = f"{s['smp_freq_rate']:.3f}" if s.get("smp_freq_rate") is not None else "-"
        smp_s = f"{s['smp_shuf_rate']:.3f}" if s.get("smp_shuf_rate") is not None else "-"
        lines.append(
            f"{s['layer']:3d} {el_s:>13} {s['raw_2plus2_count']:7d} {s['raw_2plus2_pct_elig']:6.1f} "
            f"{s['named_fdp05_count']:9d} {s['other_face_null_rate']:11.3f} "
            f"{100.0 * s['top1_in_top100_common_share']:7.1f} {s['bigram_rep_pct_elig']:8.1f} "
            f"{s['next_p1_rep_elig_pct']:8.1f} {smp_n:>7} {smp_f:>8} {smp_s:>8}"
        )
    txt = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    main()
