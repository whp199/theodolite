#!/usr/bin/env python3
"""The census's fine dose pilot: how far a push of a given size moves the next word-piece, a layer at a time, on a
ladder fine enough to set a census's doses from.

The steer lens's pilot (atlas_build.py steer_dose) measures this on a ladder whose rungs are a factor of two apart. The
whole span of pushes a census reads (about 0.35 to 2.4 nats) lies between two of those rungs at the layers under the
census band, so a dose read off that ladder is read off a straight line between two points. This takes the same
measurement, by the same code (backend.atlas_build.steer_dose_pilot: 128 neurons of the layer, every one pushed
alike in the stream's median length, the median over them of the next word-piece's KL from the unpushed one, the two
signs averaged), on rungs 7% apart from 0.29 to 0.92 of the stream, into a folder of its own. It never writes the
index's steer_dose.json, which the steer lens is built from.

    measure   on the card: the fine ladder at --layers (the reference layer first, as the pilot always takes it)
    doses     on the CPU: from the measured ladders and each layer's write ceiling, the count of ceilings at which the
              next word-piece moves by each of --targets nats, and what 1.6 and 2.3 ceilings move it by

    .venv/bin/python -I scripts/census_dose_pilot.py measure --layers 30-38 --out atlas_data/census/dose_pilot
    .venv/bin/python -I scripts/census_dose_pilot.py doses --out atlas_data/census/dose_pilot --targets 1.0,1.4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(p))

# 0.8 x 1.07^k, k from -15 to 2: eighteen rungs from 0.29 to 0.92 of the stream, the pilot's reference dose among them
FINE_LADDER = tuple(0.8 if k == 0 else round(0.8 * 1.07 ** k, 4) for k in range(-15, 3))
CEILINGS = ROOT / "experiments" / "results" / "census_scaled_design" / "ceilings.json"
PILOT_FILE = "steer_dose.json"  # the name backend.atlas_build.steer_dose_pilot writes under, in the folder it is given


def kl_at(ladder: Sequence[float], kl: Sequence[float], dose: float) -> Optional[float]:
    """The ladder's KL at a dose (in the stream's median length): a straight line between the two rungs around it in
    log dose and log KL. None off the ladder."""
    if not ladder[0] <= dose <= ladder[-1]:
        return None
    for i in range(len(ladder) - 1):
        if ladder[i] <= dose <= ladder[i + 1]:
            if kl[i] <= 0 or kl[i + 1] <= 0:
                return float(kl[i + 1])
            t = (math.log(dose) - math.log(ladder[i])) / (math.log(ladder[i + 1]) - math.log(ladder[i]))
            return math.exp(math.log(kl[i]) + t * (math.log(kl[i + 1]) - math.log(kl[i])))
    return None


def doses_table(pilot: Dict[str, Any], ceilings: Dict[str, Any], targets: Sequence[float],
                marks: Sequence[float] = (1.6, 2.3)) -> Dict[str, Any]:
    """For each layer the fine pilot measured: the count of write ceilings at which the next word-piece moves by each
    target (rounded to a hundredth of a ceiling; None, with why, where a target lies off the layer's ladder), and what
    each of `marks` ceilings moves it by. A layer's ceiling is the median over the sample in ceilings.json."""
    from backend.atlas_build import steer_dose_for

    ladder = [float(d) for d in pilot["ladder"]]
    out: Dict[str, Any] = {}
    for key in sorted(pilot.get("measured") or {}, key=int):
        kl = [float(x) for x in pilot["measured"][key]["kl"]]
        per_stream = float(ceilings["layers"][key]["ceilings_in_one_stream"])
        row: Dict[str, Any] = {"ceilings_in_one_stream": per_stream, "doses": {}, "nats_at": {}}
        for t in targets:
            dose, note = steer_dose_for(ladder, kl, float(t))
            row["doses"][f"{t:g}"] = ({"ceilings": None, "note": note} if note else
                                      {"ceilings": round(dose * per_stream, 2), "of_the_stream": round(dose, 4)})
        for c in marks:
            got = kl_at(ladder, kl, c / per_stream)
            row["nats_at"][f"{c:g}"] = None if got is None else round(got, 4)
        steps = [kl[i + 1] / kl[i] for i in range(len(kl) - 1) if kl[i] > 0]
        row["rising"] = all(s >= 1 for s in steps)  # a ladder that does not rise all the way is read with more care
        out[key] = row
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["measure", "doses"])
    ap.add_argument("--out", required=True, help="the fine pilot's own folder (never the index's)")
    ap.add_argument("--layers", default="", help='measure: the layers, as "30-38" or "37,38"')
    ap.add_argument("--neurons", type=int, default=128, help="measure: the neurons sampled at each layer")
    ap.add_argument("--seed", type=int, default=0, help="measure: draws each layer's sample (seed + the layer), as the pilot's")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ref-layer", type=int, default=None, help="measure: the pilot's reference layer (default 45)")
    ap.add_argument("--unpacker", default="compiled", choices=["reference", "lut", "compiled", "resident"])
    ap.add_argument("--stand-in", action="store_true", help="measure: on the CPU stand-in, for a dry run")
    ap.add_argument("--targets", default="1.0,1.4", help="doses: the nats each dose is to move the next word-piece by")
    ap.add_argument("--ceilings", default=str(CEILINGS), help="doses: each layer's write ceiling against its stream")
    ap.add_argument("--json", default="", help="doses: also write the table here")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.what == "doses":
        pilot = json.loads((out / PILOT_FILE).read_text(encoding="utf-8"))
        targets = [float(x) for x in a.targets.split(",") if x.strip()]
        table = doses_table(pilot, json.loads(Path(a.ceilings).read_text(encoding="utf-8")), targets)
        print(f"the fine pilot of {pilot.get('written_at')}: {pilot.get('neurons')} neurons a layer, {len(pilot['ladder'])} rungs "
              f"from {pilot['ladder'][0]:g} to {pilot['ladder'][-1]:g} of the stream")
        print("layer | ceilings that move the next word-piece by " + ", ".join(f"{t:g}" for t in targets)
              + " nats | nats at 1.6 and at 2.3 ceilings")
        for key, row in table.items():
            ds = "  ".join("off the ladder" if row["doses"][f"{t:g}"]["ceilings"] is None else f"{row['doses'][f'{t:g}']['ceilings']:.2f}"
                           for t in targets)
            ns = "  ".join("-" if v is None else f"{v:.3g}" for v in row["nats_at"].values())
            print(f"{int(key):>5} | {ds} | {ns}" + ("" if row["rising"] else "   (its ladder does not rise all the way)"))
        if a.json:
            Path(a.json).write_text(json.dumps({"what": "the census's fine dose pilot, read: " + __doc__.split("\n\n")[0],
                                                "written_at": pilot.get("written_at"), "model": pilot.get("model"),
                                                "weights": pilot.get("weights"), "neurons": pilot.get("neurons"),
                                                "seed": pilot.get("seed"), "ladder": pilot["ladder"], "targets": targets,
                                                "measured": pilot.get("measured"), "layers": table}, indent=1) + "\n", encoding="utf-8")
            print(f"-> {a.json}")
        return 0

    import torch

    torch.set_grad_enabled(False)
    from backend import atlas_build as ab
    from backend import atlas_ops as ao
    from backend.atlas_index import AtlasIndex, index_dir
    from backend.causal_check import language_model

    if a.stand_in:
        from atlas_stand_in import stand_in, use_stand_in_atlas

        use_stand_in_atlas()
        rt, _vault = stand_in()
    else:
        from backend.runtime_factory import make_runtime

        rt = make_runtime(load_jlens=False).load()
    from backend import w4a16_fast

    try:
        w4a16_fast.install(rt, a.unpacker, log=print)
    except w4a16_fast.Refused as exc:
        raise SystemExit(str(exc))
    if out.resolve() == index_dir().resolve():
        raise SystemExit(f"{out} is the index's folder: the fine pilot is kept in a folder of its own, so the steer lens's "
                         "steer_dose.json is never written over.")
    out.mkdir(parents=True, exist_ok=True)
    lm = language_model(rt.hf_model)
    L = len(lm.layers)
    from atlas_build import parse_layers
    from backend.runtime_factory import weights_card

    layers = parse_layers(a.layers, L) if a.layers else None
    stats = AtlasIndex(model=getattr(rt, "model_id", None), shape=(L, ao.ffn_width(lm, 0)), weights=weights_card(rt)).get("act_scale")
    if stats is not None:
        p50 = [float(x) for x in stats["resid_q"][:, 1]]
    elif a.stand_in:  # a dry run has no stats index: the unit is read over the study's texts, as atlas_build.py reads it
        from backend.causal_check import STUDY_TEXTS

        ids = [rt.tokenize_prompt(rt.compose_prompt(t[1], chat=True, assistant_prefix=t[2]) if t[0] == "chat" else t[1])
               for t in STUDY_TEXTS]
        p50 = ao.stream_p50(ao.activation_extremes(lm, ids))
    else:
        raise SystemExit("No stats index for this model here: the push's unit is the stream's median length, which it holds "
                         "(atlas_build.py stats).")
    ref = ab.STEER_DOSE_REF_LAYER if a.ref_layer is None else a.ref_layer
    print(f"fine pilot: {len(FINE_LADDER)} rungs from {FINE_LADDER[0]:g} to {FINE_LADDER[-1]:g} of the stream, at layers "
          f"{a.layers or 'all'} and the reference layer {ref}; into {out}", flush=True)
    try:
        path = ab.steer_dose_pilot(rt, out, p50, layers=layers, neurons=a.neurons, ladder=FINE_LADDER, ref_layer=ref,
                                   seed=a.seed, batch=a.batch, log=lambda s: print(s.replace("steer dose:", "fine pilot:"), flush=True))
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
