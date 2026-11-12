#!/usr/bin/env bash
# Grow the band outward, a trial before every whole layer (prereg/RULES.md): on each side of layers 40-50 the next
# layer out is trialled and read, and the walk goes on on that side only if it passed. Each trial is prereg/night.sh at
# one layer (about 80 minutes on the home 5090 beside a desktop), each reading prereg/read.sh (the reader and gates
# hashed on 12 November 2026 before any trial was taken). A layer already written or read is not taken again.
#
#     bash walk.sh                    # both sides in turn, from where the verdicts stand, until each side stops
#     SIDES=down MAX=4 bash walk.sh   # four more trials at most, downward only
#
# A layer's neurons are prereg/neurons_L<NN>.txt; one not drawn on 12 November is drawn now by the hashed
# prereg/draw_neurons.py (its seed is fixed there, so the list is the one that day's draw would have given).
# Nothing here commits, pushes or opens a layer: a pass is a licence the owner acts on (census_board.py layer N open).
set -uo pipefail
D=/home/eric/bin/ai/theodolite2/atlas_data/census/trials/neighbours
P=$D/prereg
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
SIDES=${SIDES:-"down up"}
MAX=${MAX:-99}
verdict() {  # pass | fail | none, for a layer
    [ -f "$D/L$1_verdict.json" ] || { echo none; return; }
    $PY -c 'import json, sys; print("pass" if json.load(open(sys.argv[1])).get("pass") else "fail")' "$D/L$1_verdict.json"
}
next_of() {  # the next layer out on a side that has no verdict yet, or "stop" where the side has failed or run out
    local side=$1 L v
    if [ "$side" = down ]; then L=39; else L=51; fi
    while [ "$L" -ge 0 ] && [ "$L" -le 59 ]; do
        v=$(verdict "$L")
        [ "$v" = none ] && { echo "$L"; return; }
        [ "$v" = fail ] && { echo stop; return; }
        if [ "$side" = down ]; then L=$((L - 1)); else L=$((L + 1)); fi
    done
    echo stop
}
taken=0
while [ "$taken" -lt "$MAX" ]; do
    moved=0
    for side in $SIDES; do
        [ "$taken" -lt "$MAX" ] || break
        L=$(next_of "$side")
        [ "$L" = stop ] && continue
        [ -f "$P/neurons_L$L.txt" ] || $PY "$P/draw_neurons.py" "$L" > "$P/neurons_L$L.txt"
        echo "== $(date -u +%FT%TZ) layer $L ($side)"
        LAYERS=$L bash "$P/night.sh" || { echo "layer $L: the trial stopped (run this again to continue it)"; exit 6; }
        LAYERS=$L bash "$P/read.sh"
        [ "$(verdict "$L")" = none ] && { echo "layer $L: no verdict was written"; exit 7; }
        taken=$((taken + 1)); moved=1
    done
    [ "$moved" = 1 ] || break
done
for side in $SIDES; do echo "$side: $(next_of "$side")"; done
echo "$taken trial(s) taken; the verdicts are in $D"
