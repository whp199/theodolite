#!/usr/bin/env bash
# Read one match_things trial of the walk once it is written, on the CPU, with the reader and gates hashed before any
# answer of the walk existed: things_read.py, byte for byte the reader of match_things's trial at layer 30 (N1 and N3,
# and the readings it reports, the lower dose alone among them, which RULES.md reads as N2). Not read again once read.
#     LAYERS="53" bash read.sh
set -uo pipefail
R=${R:-/home/eric/bin/ai/theodolite2/.claude/worktrees/things-walk}
D=/home/eric/bin/ai/theodolite2/atlas_data/census/trials/things_walk
P=$D/prereg
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
LAYERS=${LAYERS:?give LAYERS, the layers to read}
(cd "$P" && sha256sum -c --quiet SHA256SUMS) || echo "the pre-registered files differ from SHA256SUMS: see AMENDMENTS.txt"
export CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO="$R"
for L in $LAYERS; do
    RUN=$D/L$L-match_things-trial
    [ -e "$RUN.t1" ] || { echo "layer $L: not written to its end yet"; continue; }
    [ -e "$D/L${L}_verdict.json" ] && { echo "layer $L: already read, kept as it is"; continue; }
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 15 $PY "$P/things_read.py" "$RUN.jsonl" --out "$D/L${L}_verdict.json" > "$D/L${L}_verdict.txt" 2>&1
    grep -E '^(===|pass  N|FAIL  N|reported, (the|three))' "$D/L${L}_verdict.txt"
    $PY -c 'import json, sys; v = json.load(open(sys.argv[1])); n2 = v["reported"]["each_dose"]["lower"]["pass_G1_G4"]
print(("pass" if n2 else "FAIL") + "  N2 the lower dose alone, G1 to G4 (" + ",".join(f"{d:g}" for d in v["reported"]["each_dose"]["lower"]["doses"]) + ")"
      + ("" if v["pass"] else ": decides nothing, the layer failed N1 or N3"))' "$D/L${L}_verdict.json" | tee -a "$D/L${L}_verdict.txt"
done
