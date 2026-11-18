#!/usr/bin/env bash
# The walk with match_things (RULES.md): two arms, upward from layer 53 to 58 and downward from 29 to 25, taken in turn,
# 53, 29, 54, 28, 55, 27, 56, 26, 57, 25, 58: a trial at each (night.sh), read as soon as it is written (read.sh). An
# arm goes on to its next layer only if this one passed N1 and N3; it stops at its first layer that fails, and the
# other arm goes on alone. A trial that stops before its last answer is begun once more after 30 seconds, row for row;
# a second stop ends the walk with no verdict there. A layer already written or read is not taken again, so the same
# command goes on from where it stopped. Nothing here commits, pushes, opens a layer or counts a dose calibrated.
#     bash walk.sh
set -uo pipefail
D=/home/eric/bin/ai/theodolite2/atlas_data/census/trials/things_walk
P=$D/prereg
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
UP="53 54 55 56 57 58"
DOWN="29 28 27 26 25"
ORDER="53 29 54 28 55 27 56 26 57 25 58"
verdict() {  # pass | fail | none, for a layer
    [ -f "$D/L$1_verdict.json" ] || { echo none; return; }
    $PY -c 'import json, sys; print("pass" if json.load(open(sys.argv[1])).get("pass") is True else "fail")' "$D/L$1_verdict.json"
}
arm_of() { case " $UP " in *" $1 "*) echo up;; *) echo down;; esac; }
stopped_up=; stopped_down=
for L in $ORDER; do
    arm=$(arm_of "$L")
    if [ "$arm" = up ] && [ -n "$stopped_up" ]; then continue; fi
    if [ "$arm" = down ] && [ -n "$stopped_down" ]; then continue; fi
    v=$(verdict "$L")
    if [ "$v" = none ]; then
        echo "== $(date -u +%FT%TZ) layer $L ($arm): the trial"
        LAYERS=$L bash "$P/night.sh" || { echo "layer $L: the trial stopped: once more"; sleep 30; LAYERS=$L bash "$P/night.sh"; } \
            || { echo "layer $L: the trial stopped twice (run this again to continue it)"; exit 6; }
        LAYERS=$L bash "$P/read.sh"
        v=$(verdict "$L")
        [ "$v" = none ] && { echo "layer $L: no verdict was written"; exit 7; }
    fi
    echo "== $(date -u +%FT%TZ) layer $L ($arm): $v"
    if [ "$v" != pass ]; then
        echo "the $arm arm stops at layer $L, which did not pass"; echo "$L" > "$D/walk.$arm.stopped_at"
        if [ "$arm" = up ]; then stopped_up=$L; else stopped_down=$L; fi
    fi
done
date -u +%FT%TZ > "$D/walk.done"
