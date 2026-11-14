#!/usr/bin/env python3
"""Where a test behind its own passage would cut each of the pilot's six faces that need more text (CPU: the passages
index and the tokenizer alone; no model). For each face's strongest passages that are not kept back and that read at
least half its extreme: the last of the text that is held and the word-pieces an ending replaces, at a few drops.
What the endings in behind.json were written from. Prints; writes nothing (the text is the corpus's).

    .venv/bin/python -I experiments/results/pattern_tests_pilot/cuts.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer  # noqa: E402

from backend.atlas_index import AtlasIndex, passage_kept_back  # noqa: E402

SIX = ["L26-n14541m", "L26-n4837p", "L36-n12896p", "L45-n16700m", "L56-n19940m", "L56-n3342p"]


def main():
    faces = json.loads((HERE / "patterns.json").read_text())["faces"]
    idx = AtlasIndex(ROOT / "atlas_data" / "index")
    stats = idx.get("act_scale")
    tok = AutoTokenizer.from_pretrained(idx.get("passages")["meta"]["model"], local_files_only=True)
    for k in sys.argv[1:] or SIX:
        f = faces[k]
        ext = float(stats["pos" if f["face"] == "+" else "neg"][f["layer"], f["n"]])
        got = idx.passages(f["layer"], f["n"], f["face"], 64)
        seen, rows = set(), []
        for x in list(got["top"]) + list(got["sample"]):
            if x["window"] not in seen and not passage_kept_back(x["window"]):
                seen.add(x["window"])
                rows.append(x)
        print(f"==== {k}: {f['pattern'][:150]}")
        for x in [r for r in rows if r["act"] / ext >= 0.5][:3]:
            ids, at = idx.window_ids(x["window"]), x["at"]
            print(f"-- window {x['window']}, peak at {at}, reads {x['act'] / ext:.3f} in its window")
            print("   held, its last 45 word-pieces: " + repr(tok.decode(ids[max(1, at - 52): at - 7])))
            for d in (8, 4, 2, 1):
                print(f"   drop {d}: " + repr(tok.decode(ids[at + 1 - d: at]) + "[[" + tok.decode(ids[at: at + 1]) + "]]"))


if __name__ == "__main__":
    main()
