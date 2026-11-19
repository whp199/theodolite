#!/usr/bin/env python3
"""Owner & volunteer CLI for Levelling Phase 5: Re-run Kit & Defensive Intake (`scripts/levelling_rerun.py`).

    .venv/bin/python -I scripts/levelling_rerun.py pack /tmp/trial/prereg --out /tmp/kit [--home-result trial_result.json]
    .venv/bin/python -I scripts/levelling_rerun.py verify-kit /tmp/kit
    .venv/bin/python -I scripts/levelling_rerun.py run /tmp/kit --out /tmp/vol_return.tar --volunteer alice --stand-in
    .venv/bin/python -I scripts/levelling_rerun.py intake /tmp/vol_return.tar --prereg /tmp/trial/prereg \\
        --home-result /tmp/trial/result/trial_result.json --out /tmp/intake_stage
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PORT = 8765


def cmd_pack(a: argparse.Namespace) -> int:
    from backend import level_rerun as lr
    from backend import level_trial as lt

    try:
        home_res = json.loads(Path(a.home_result).read_text(encoding="utf-8")) if a.home_result else None
        out = lr.make_rerun_kit(Path(a.prereg), Path(a.out), home_result=home_res)
        print(json.dumps(out, indent=2))
        return 0
    except (lr.RerunRefused, lt.TrialRefused, OSError, ValueError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def cmd_verify_kit(a: argparse.Namespace) -> int:
    from backend import level_rerun as lr

    try:
        v = lr.verify_rerun_kit(Path(a.kit))
        print(json.dumps({
            "ok": True,
            "kit_dir": v["kit_dir"],
            "kit_sha256": v["kit_sha256"],
            "trial_sha256": v["trial_sha256"],
            "trial_name": v["kit"]["trial_name"],
            "edit_id": v["kit"]["edit_id"],
        }, indent=2))
        return 0
    except lr.RerunRefused as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 1


def cmd_run(a: argparse.Namespace) -> int:
    import levelling_trial as lt_cli
    from backend import level_rerun as lr

    try:
        lr.verify_rerun_kit(Path(a.kit))
    except lr.RerunRefused as exc:
        print(f"Refused before load: {exc}", file=sys.stderr)
        return 2

    h, rc = lt_cli._open_handle(bool(a.stand_in), int(a.port), packed_stand_in=bool(a.packed_stand_in))
    if rc != 0 or h is None:
        return rc
    try:
        res = lr.run_volunteer_rerun(h, Path(a.kit), Path(a.out), volunteer_name=str(a.volunteer))
        print(json.dumps({
            "bundle_tar": res["bundle_tar"],
            "bundle_sha256": res["bundle_sha256"],
            "kit_sha256": res["kit_sha256"],
            "verdict": res["verdict"],
        }, indent=2))
        return 0 if res["verdict"] != "void" else 1
    except (lr.RerunRefused, OSError, ValueError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def cmd_intake(a: argparse.Namespace) -> int:
    from backend import level_rerun as lr
    from backend import level_trial as lt

    try:
        home_res = json.loads(Path(a.home_result).read_text(encoding="utf-8"))
        verdict = lr.intake_rerun_bundle(
            Path(a.bundle), Path(a.prereg), home_res, Path(a.out), allow_dirty=bool(a.allow_dirty),
        )
        print(json.dumps(verdict, indent=2))
        return 0 if verdict["outcome"] == "accepted" else 1
    except (lr.RerunRefused, lt.TrialRefused, OSError, ValueError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="levelling_rerun.py", description="Phase 5 volunteer re-run kit & defensive intake.")
    sub = ap.add_subparsers(dest="command", required=True)

    pk = sub.add_parser("pack", help="package a verified prereg/ folder into a volunteer re-run kit (rebuilt edits only)")
    pk.add_argument("prereg", help="Phase 5 registration folder")
    pk.add_argument("--out", required=True, help="output kit folder")
    pk.add_argument("--home-result", default=None, help="optional home trial_result.json reference")
    pk.set_defaults(fn=cmd_pack)

    vk = sub.add_parser("verify-kit", help="verify SHA256SUMS and integrity of a re-run kit")
    vk.add_argument("kit", help="re-run kit folder")
    vk.set_defaults(fn=cmd_verify_kit)

    rn = sub.add_parser("run", help="run a re-run kit and pack an allowlisted volunteer .tar bundle")
    rn.add_argument("kit", help="re-run kit folder")
    rn.add_argument("--out", required=True, help="output .tar bundle path")
    rn.add_argument("--volunteer", default="anonymous")
    rn.add_argument("--stand-in", action="store_true")
    rn.add_argument("--packed-stand-in", action="store_true")
    rn.add_argument("--port", type=int, default=PORT)
    rn.set_defaults(fn=cmd_run)

    ik = sub.add_parser("intake", help="defensively unpack, verify, and judge a volunteer .tar bundle")
    ik.add_argument("bundle", help="volunteer .tar bundle path")
    ik.add_argument("--prereg", required=True, help="home registration folder")
    ik.add_argument("--home-result", required=True, help="home trial_result.json path")
    ik.add_argument("--out", required=True, help="fresh intake staging folder")
    ik.add_argument("--allow-dirty", action="store_true", help="allow dirty git checkout (for dry-run tests)")
    ik.set_defaults(fn=cmd_intake)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
