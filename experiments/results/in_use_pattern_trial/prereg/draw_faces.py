#!/usr/bin/env python3
"""Draw fresh doc_4_8 faces per layer for Path 3 (in_use_pattern): 128 at L2, 128 at L14, 64 at L20.

Strictly excludes:
  - experiments/results/pattern_tests_pilot/patterns.json
  - experiments/results/in_use_kill_pilots/probe1_in_use_causal.json
  - experiments/results/in_use_kill_pilots/probe3_local_pattern.json
Requires each face to pass _select_doc_events cleanly (>=4 doc_h0, >=8 doc_h1, valid unseen behind event).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import atlas_index as ai  # noqa: E402
from scripts.in_use_common import collect_face_events, layer_face_counts_vectorized, load_passages_and_doc_halves  # noqa: E402
from scripts.in_use_pattern_trial import _select_doc_events  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def _load_exclusions() -> Set[Tuple[int, int]]:
    excl: Set[Tuple[int, int]] = set()
    p_pat = REPO_ROOT / "experiments/results/pattern_tests_pilot/patterns.json"
    if p_pat.is_file():
        for v in json.loads(p_pat.read_text(encoding="utf-8")).get("faces", {}).values():
            excl.add((int(v["layer"]), int(v["n"])))
    p1 = REPO_ROOT / "experiments/results/in_use_kill_pilots/probe1_in_use_causal.json"
    if p1.is_file():
        for lk, vals in (json.loads(p1.read_text(encoding="utf-8")).get("neuron_ids") or {}).items():
            for tag in vals:
                m = re.match(r"^n(\d+)[pm]$", str(tag))
                if m:
                    excl.add((int(lk), int(m.group(1))))
    p3 = REPO_ROOT / "experiments/results/in_use_kill_pilots/probe3_local_pattern.json"
    if p3.is_file():
        for grp in ("group_a", "group_b"):
            for tag in (json.loads(p3.read_text(encoding="utf-8")).get("neuron_ids") or {}).get(grp, []):
                m = re.match(r"^L(\d+)-n(\d+)[pm]$", str(tag))
                if m:
                    excl.add((int(m.group(1)), int(m.group(2))))
    return excl


def main() -> None:
    loaded = load_passages_and_doc_halves(ai.atlas_dir())
    P, act_scale = loaded["P"], loaded["act_scale"]
    windows, doc_id, half_id = loaded["windows"], loaded["doc_id"], loaded["half_id"]
    pure_h0, pure_h1 = loaded["pure_h0_mask"], loaded["pure_h1_mask"]
    bos_id, eos_id = loaded["bos_id"], loaded["eos_id"]
    excl = _load_exclusions()

    targets = [(2, 128), (14, 128), (20, 64)]
    for l, n_want in targets:
        cnt = layer_face_counts_vectorized(
            P, windows, act_scale, half_id, pure_h0, pure_h1, l, eos_id=eos_id, bos_id=bos_id
        )
        N_ff = int(cnt["pos_doc0"].shape[0])
        # Eligible: >=4 doc_h0 and >=8 doc_h1 on preferred face
        cands = []
        for n in range(N_ff):
            if (l, n) in excl:
                continue
            opts = []
            for fc in ("pos", "neg"):
                if int(cnt[f"{fc}_doc0"][n]) >= 4 and int(cnt[f"{fc}_doc1"][n]) >= 8:
                    tot = int(cnt[f"{fc}_pure0"][n]) + int(cnt[f"{fc}_pure1"][n])
                    ext = float(abs(act_scale[fc][l, n]))
                    opts.append((tot, int(cnt[f"{fc}_doc1"][n]), ext, 1 if fc == "pos" else 0, fc))
            if opts:
                opts.sort(reverse=True)
                cands.append((n, opts[0][4]))

        seed = 202611260300 + l
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(cands))
        drawn = []
        for idx in perm:
            n, fc = cands[int(idx)]
            ev = collect_face_events(
                P, windows, act_scale, pure_h0, pure_h1, doc_id, half_id, l, n, fc, eos_id=eos_id, bos_id=bos_id
            )
            try:
                fit_evs, test_evs, behind_ev = _select_doc_events(ev, n_fit=8, n_test=10)
            except AssertionError:
                continue
            if len(fit_evs) >= 4 and len(test_evs) >= 8 and behind_ev is not None:
                drawn.append((n, fc))
            if len(drawn) >= n_want:
                break
        drawn.sort()
        assert len(drawn) == n_want, f"L{l}: got only {len(drawn)}/{n_want}"
        out_p = OUT_DIR / f"faces_L{l}.txt"
        out_p.write_text("\n".join(f"{n},{fc}" for n, fc in drawn) + "\n", encoding="utf-8")
        print(f"L{l}: pool={len(cands)} drew={n_want} seed={seed} -> {out_p.name}")


if __name__ == "__main__":
    main()
