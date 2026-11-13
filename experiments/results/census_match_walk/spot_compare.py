#!/usr/bin/env python3
"""Layer 35's trial answers around the card's fault, against the same answers written again.

The home card faulted (Xid 8, the driver's watchdog) during layer 35's trial of the match_ceil walk, at 09:17:31Z on
13 November 2026, after 18,240 of its 49,110 answers were in the file. The walk's registered retry wrote the other
30,870 in a second session. A reading cannot show whether the answers written just before a fault are sound, so the
two chunks of the plan around it (22 and 23 of 64: batches 268 to 291 of 772) were written again when the walk freed
the card, by the trial's own command from the trial's own checkout, into a folder of their own. This script compares
them with the trial's rows, every field, and each chunk's canary (plan batch 0 written again) with the trial's first
batch. It is no part of the walk's registration and gates nothing.

    .venv/bin/python -I experiments/results/census_match_walk/spot_compare.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend import census as cz  # noqa: E402
from backend import census_volunteer as cv  # noqa: E402

WALK = ROOT / "atlas_data" / "census" / "trials" / "match_walk"
TRIAL = WALK / "L35-match_ceil-trial.jsonl"
SPOT = WALK / "spot"
CHUNKS = (22, 23)
BEFORE_FAULT = 18240   # the rows the first session wrote; a run's rows are in its file in the order written


def rows_of(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    spot = {c: rows_of(SPOT / f"L35-spot.s{c}of64.jsonl") for c in CHUNKS}
    canary = {c: rows_of(cz.canary_path_of(SPOT / f"L35-spot.s{c}of64.jsonl")) for c in CHUNKS}
    wanted = {cz.row_key(r) for rows in list(spot.values()) + list(canary.values()) for r in rows}
    trial, at = {}, {}
    with open(TRIAL) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            k = cz.row_key(r)
            if k in wanted:
                trial[k], at[k] = r, i
    total = i + 1

    trial_meta = json.loads((WALK / "L35-match_ceil-trial.meta.json").read_text())
    out = {"what": __doc__.split("\n\n")[0], "layer": 35, "trial": TRIAL.name, "trial_answers": total,
           "written_before_the_fault": BEFORE_FAULT, "trial_jobs": trial_meta["jobs"], "chunks": {}}
    ok = True
    print(f"the trial: {total} answers, the first {BEFORE_FAULT} written before the fault; jobs {trial_meta['jobs']['sha256']}")
    for c in CHUNKS:
        meta = json.loads((SPOT / f"L35-spot.s{c}of64.meta.json").read_text())
        same_jobs = meta["jobs"] == trial_meta["jobs"]
        where = sorted(at[cz.row_key(r)] for r in spot[c] if cz.row_key(r) in at)
        before = [r for r in spot[c] if at.get(cz.row_key(r), total) < BEFORE_FAULT]
        after = [r for r in spot[c] if at.get(cz.row_key(r), -1) >= BEFORE_FAULT]
        parts = {"all": cv.compare_rows(trial, spot[c]), "written_before_the_fault": cv.compare_rows(trial, before),
                 "written_after_the_fault": cv.compare_rows(trial, after), "canary": cv.compare_rows(trial, canary[c])}
        out["chunks"][str(c)] = {"batches": meta["shard"]["batches"], "started_at": meta["started_at"], "same_jobs": same_jobs,
                                 "rows_in_the_trial": [where[0], where[-1]] if where else None, **parts}
        print(f"chunk {c} of 64 (written again from {meta['started_at']}; the trial's rows {where[0]} to {where[-1]}); "
              f"the same job list: {same_jobs}")
        for name, p in parts.items():
            good = p["answers"] == p["present"] == p["identical"]
            ok = ok and good and same_jobs
            print(f"    {name:<26} {p['answers']:>4} answers, {p['present']:>4} in the trial, {p['identical']:>4} the same in every "
                  f"field" + ("" if good else f"   DIFFER (first piece alike {p['first_piece']}, shared prefix median "
                                              f"{p['prefix_median']}, min {p['prefix_min']})"))
    out["every_answer_the_same"] = ok
    (HERE / "spot_L35.json").write_text(json.dumps(out, indent=1) + "\n")
    print("every answer written again is the trial's, bit for bit" if ok else "THE ANSWERS DIFFER: see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
