#!/usr/bin/env bash
# The match_ceil trial at layer 37 on the home RTX 5090: 1,022 neurons the walk's trial at this layer did not ask
# (neurons_L37_fresh.txt), 1,024 controls drawn afresh (seed 7), about 88 minutes. Run from the clean checkout of
# repo_commit (R). Stops on a failure; run again it continues where it stopped. Nothing here reads a verdict (read.sh
# does), and nothing here commits, pushes or merges.
set -euo pipefail
R=${R:-/home/eric/bin/ai/theodolite2/.claude/worktrees/match-trials}
export THEODOLITE_ATLAS_DIR=/home/eric/bin/ai/theodolite2/atlas_data
D=$THEODOLITE_ATLAS_DIR/census/trials/match
P=$D/prereg
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
[ "$(git -C "$R" rev-parse HEAD)" = "$(cat "$P/repo_commit")" ] || { echo "$R is not at the commit in $P/repo_commit"; exit 2; }
[ -z "$(git -C "$R" --no-optional-locks status --porcelain --untracked-files=no)" ] || { echo "$R has uncommitted changes"; exit 2; }
(cd "$P" && sha256sum -c --quiet SHA256SUMS) || { echo "$P differs from its SHA256SUMS: record why in AMENDMENTS.txt"; exit 2; }
if (exec 3<>/dev/tcp/127.0.0.1/8765) 2>/dev/null; then
    echo "something answers on :8765: stop the Theodolite server first (one GPU process at a time)"; exit 5
fi
RUN=L37-match_ceil-trial
done_run() {
    $PY -c 'import json, sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; from backend import census as cz
p = Path(sys.argv[2]); n = sum(1 for _ in cz.read_rows(p, tolerate_tail=True))
sys.exit(0 if n == json.loads(cz.meta_path_of(p).read_text())["jobs"]["n"] else 1)' "$R" "$D/$RUN.jsonl" 2>/dev/null
}
if done_run; then echo "$RUN: every answer already written, kept as it is"; [ -e "$D/$RUN.t1" ] || date -u +%s > "$D/$RUN.t1"; exit 0; fi
[ -e "$D/$RUN.jsonl" ] || date -u +%s > "$D/$RUN.t0"
$PY "$R/scripts/census_run.py" --layer 37 --design match_ceil --neurons "$(cat "$P/neurons_L37_fresh.txt")" --random-controls 1024 \
    --seed 7 --batch 64 --max-new 48 --unpacker compiled --out-dir "$D" --run "$RUN" 2>&1 | tee -i -a "$D/$RUN.log"
done_run || { echo "$RUN stopped before its last answer: run this again to continue it row for row"; exit 6; }
date -u +%s > "$D/$RUN.t1"
echo "written: $RUN; not read here (read.sh)"
