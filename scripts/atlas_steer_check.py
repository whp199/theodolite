#!/usr/bin/env python3
"""Check the built steer lens against the census and write what was found beside the index (CPU only, no model).

    .venv/bin/python -I scripts/atlas_steer_check.py            # the index folder's lens; writes steer_check.json there
    .venv/bin/python -I scripts/atlas_steer_check.py --print    # measure and print, write nothing

For each layer: how many of a neuron's twelve word-pieces the layer shares (listed by 2 in 100 of its neurons or
more), and, where the layer has a whole census or a trial whose doses are counted calibrated, for how many of the
neurons that stand above 95 in 100 controls a steer-lens word-piece is one of the census's words (and the same with
the neurons shuffled). The server reads the file (backend/atlas_index.py AtlasIndex.steer_check) and the details tab
says what it found at the open neuron's layer. Run it again when the lens is rebuilt, when a census layer lands, or
when a trial's doses are counted calibrated: a file written for another build of the lens is not read.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import steer_check as sk  # noqa: E402
from backend.atlas_index import index_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index", default=None, help="the index folder (default: the atlas's)")
    ap.add_argument("--runs", default=None, help="the census runs' folder (default: the atlas's census/runs)")
    ap.add_argument("--trials", default=None, help="the census trials' folder (default: census/trials beside the runs)")
    ap.add_argument("--print", action="store_true", dest="only_print", help="measure and print; write nothing")
    a = ap.parse_args()
    root = Path(a.index) if a.index else index_dir()
    kw = {"runs_dir": Path(a.runs) if a.runs else None, "trials_dir": Path(a.trials) if a.trials else None}
    try:
        if a.only_print:
            d = sk.measure(root, **kw)
        else:
            out = sk.write(root, **kw)
            import json

            d = json.loads(out.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SystemExit(str(exc))
    print("\n".join(sk.said(d)))
    if not a.only_print:
        print(f"-> {root / 'steer_check.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
