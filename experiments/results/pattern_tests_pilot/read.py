#!/usr/bin/env python3
"""The pilot read (CPU; reads the files pilot.py wrote, writes read.json and prints the tables).

A face "holds alone" when at least two of its three strongest passages (not kept back), read with only the 60
word-pieces before the peak that a blind reader is shown, still reach half of what they read in their windows
(context.json). The written and blind results are given for all the faces and for the two groups apart.

    .venv/bin/python -I experiments/results/pattern_tests_pilot/read.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))
from backend import pattern_tests as pt  # noqa: E402

load = lambda name: json.loads((HERE / name).read_text())  # noqa: E731


def main():
    faces = load("patterns.json")["faces"]
    written, blind, ctx, survey = load("written.json")["tests"], load("blind.json")["tests"], load("context.json")["faces"], load("survey.json")["faces"]
    rows = {}
    for k, f in faces.items():
        holds = sum(r["60"] >= 0.5 * r["window"] for r in ctx[k])
        w, b = written[k], blind[k]
        rows[k] = {"layer": f["layer"], "sure": f["sure"], "holds_alone": holds >= 2, "passages_holding": holds,
                   "written": {x: w[x] for x in ("fire", "fired", "beat_corpus", "strongest", "quiet", "quiet_fired", "apart")},
                   "blind": None if "refused" in b else {"pairs": b["own_pattern"]["pairs"], "right": b["own_pattern"]["right"], "p": b["own_pattern"]["p"],
                                                         "right_with_another_pattern": b["other_pattern"]["right"],
                                                         "p_with_another_pattern": b["other_pattern"]["p"]}}

    def pooled(ks):
        ks = [k for k in ks if rows[k]["blind"]]
        n = sum(rows[k]["blind"]["pairs"] for k in ks)
        own, other = sum(rows[k]["blind"]["right"] for k in ks), sum(rows[k]["blind"]["right_with_another_pattern"] for k in ks)
        return {"faces": len(ks), "pairs": n, "right": own, "share": round(own / n, 3) if n else None, "p": round(pt.binom_tail(own, n), 6),
                "right_with_another_pattern": other, "share_with_another_pattern": round(other / n, 3) if n else None,
                "p_with_another_pattern": round(pt.binom_tail(other, n), 6),
                "faces_at_p_below_0.05": sum(rows[k]["blind"]["p"] < 0.05 for k in ks),
                "faces_at_p_below_0.05_with_another_pattern": sum(rows[k]["blind"]["p_with_another_pattern"] < 0.05 for k in ks),
                "written_fired": sum(rows[k]["written"]["fired"] for k in ks), "written_texts": sum(rows[k]["written"]["fire"] for k in ks),
                "near_misses_fired": sum(rows[k]["written"]["quiet_fired"] for k in ks), "near_misses": sum(rows[k]["written"]["quiet"] for k in ks),
                "faces_with_a_text_that_fired": sum(rows[k]["written"]["fired"] > 0 for k in ks)}

    groups = {"all": list(rows), "holds alone": [k for k in rows if rows[k]["holds_alone"]], "needs more text": [k for k in rows if not rows[k]["holds_alone"]]}
    for s in ("clear", "fair", "guess"):
        groups[f"its writer was sure: {s}"] = [k for k in rows if rows[k]["sure"] == s]
    begun = [r for r in survey if r.get("begun")]
    out = {"survey": {"faces": len(survey), "a_test_could_be_begun": len(begun),
                      "refused": [{"layer": r["layer"], "n": r["n"], "face": r["face"], "why": r["refused"]} for r in survey if "refused" in r],
                      "pairs_median": sorted(r["pairs"] for r in begun)[len(begun) // 2], "with_8_pairs_or_more": sum(r["pairs"] >= 8 for r in begun)},
           "faces": rows, "pooled": {name: pooled(ks) for name, ks in groups.items()}}
    (HERE / "read.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"{'face':<14}{'sure':<7}{'holds':<7}{'written':<10}{'apart':<7}{'blind':<9}{'p':<9}{'another pattern':<16}")
    for k, r in rows.items():
        b = r["blind"]
        print(f"{k:<14}{r['sure']:<7}{('yes' if r['holds_alone'] else 'no') + ' ' + str(r['passages_holding']):<7}"
              f"{str(r['written']['fired']) + ' of ' + str(r['written']['fire']):<10}{str(r['written']['apart']):<7}"
              + (f"{str(b['right']) + ' of ' + str(b['pairs']):<9}{b['p']:<9g}{b['right_with_another_pattern']} of {b['pairs']}" if b else "refused"))
    for name, g in out["pooled"].items():
        print(f"{name}: {g['faces']} faces; blind {g['right']} of {g['pairs']} ({g['share']}), with another face's pattern "
              f"{g['right_with_another_pattern']} ({g['share_with_another_pattern']}); faces at p < 0.05: {g['faces_at_p_below_0.05']} "
              f"(another pattern: {g['faces_at_p_below_0.05_with_another_pattern']}); written {g['written_fired']} of {g['written_texts']} fired, "
              f"{g['near_misses_fired']} of {g['near_misses']} near misses, {g['faces_with_a_text_that_fired']} faces with a text that fired")


if __name__ == "__main__":
    main()
