#!/usr/bin/env python3
"""A CPU-only check of the census volunteer kit (VOLUNTEERS.md) end to end on the tiny random Gemma 4 of
scripts/atlas_stand_in.py, in a temporary folder: the volunteer's side (scripts/volunteer/census.sh --dry-run) and the
owner's (scripts/census_board.py, scripts/census_intake.py --dry-run), against a "home card" that is this machine:
0. the kit's own file: every layer with a design the census can take there, the board's open layers among the kit's;
   a kit of one design (kits 1 and 2) still read, a layer without a design refused, the writer's floor taking a
   design whose doses are set a layer at a time;
1. the home card: a whole stand-in layer run, and the kit layers' canaries, each with its layer's design, and the
   unpushed answers (home-canaries);
2. the volunteer: check, reserve (a layer outside the kit refused), setup (setup.sh --dry-run, ~/.bashrc untouched),
   canary (the home card's bits), run (detached, waited for), status (the PROGRESS line), pack (the bundle and the DONE
   line), the bundle holding no path of this machine;
3. the board: RESERVE, a second RESERVE on a held layer refused, one claim one layer, START, PROGRESS, DONE, a RELEASE
   without the secret refused, a held claim lapsing on the clock, the renders (board.txt under 2,000 characters), and
   the renders of a board with every layer done (saying so) and of one with a layer closed instead (not saying so);
4. the intake: the bundle checked, spot-checked and judged bit for bit, accepted into a census folder, ranked, and its
   layer done on the board;
5. the refusals: a run tampered after its canary (a spot chunk's answers changed, packed again as a cheat would) is
   rejected by the spot-check; a made-up run of another card class fails the first-piece rule; a run of another card
   class (every log-probability moved by 0.0005, as another card's sums move them) passes the kit's rules for one; a
   bundle whose secret does not give its claim is refused and leaves the board as it was; a bundle holding a file a pack
   does not write is not unpacked.

Run: .venv/bin/python -I scripts/check_volunteer_kit.py   (exits 0 when every check holds; a few minutes)
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import census as cz  # noqa: E402
from backend import census_volunteer as cv  # noqa: E402

PY = sys.executable
CENSUS_SH = ROOT_DIR / "scripts" / "volunteer" / "census.sh"
BOARD_PY = ROOT_DIR / "scripts" / "census_board.py"
INTAKE_PY = ROOT_DIR / "scripts" / "census_intake.py"
LAYER = 3

failures: List[str] = []
passed = 0


def check(ok: bool, what: str) -> None:
    global passed
    if ok:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAIL  {what}")


def section(title: str, before: int) -> None:
    if len(failures) == before:
        print(f"  ok    {title}")


def run(cmd: List[Any], env: Dict[str, str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], env=env, cwd=ROOT_DIR, capture_output=True, text=True, timeout=timeout)


def last_line(text: str, start: str) -> str:
    return next((ln for ln in reversed(text.splitlines()) if ln.startswith(start)), "")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="volunteer-kit-") as d:
        return run_checks(Path(d))


def run_checks(tmp: Path) -> int:
    home_atlas, vroot, canaries = tmp / "home-atlas", tmp / "volunteer", tmp / "home-canaries"
    fake_home = tmp / "user-home"
    fake_home.mkdir()
    (fake_home / ".bashrc").write_text("# the volunteer's own\n")
    env = {**os.environ, "THEODOLITE_ATLAS_DIR": str(home_atlas), "THEODOLITE_KIT_HOME_CANARIES": str(canaries),
           "HOME": str(fake_home), "CUDA_VISIBLE_DEVICES": "", "NO_COLOR": "1"}
    env.pop("THEODOLITE_POD_ROOT", None)
    kit = cv.read_kit(dry=True)
    design = cv.kit_design(kit, LAYER)  # the design the stand-in's kit takes the checked layer with
    args = ["--design", design, "--neurons", kit["KIT_NEURONS"], "--random-controls", str(kit["KIT_CONTROLS"]), "--seed",
            str(kit["KIT_SEED"]), "--batch", str(kit["KIT_BATCH"]), "--max-new", str(kit["KIT_MAX_NEW"]), "--unpacker", kit["KIT_UNPACKER"]]

    # 0. the kit's own file
    before = len(failures)
    real = cv.read_kit()
    by_design = cv.kit_designs(real)
    check(cv.kit_problems(real) == [] and cv.kit_problems(kit) == [] and sorted(l_ for ls in by_design.values() for l_ in ls) == sorted(real["layers"])
          and all(d_ in cz.design_names() for d_ in by_design),
          f"the kit's file: every layer has a design the census takes there ({cv.kit_problems(real)}, {by_design})")
    # a layer is opened on a passed trial; where its doses are counted calibrated too (the owner's word), they are at the
    # kit's answer length. The layers counted so far, by the kit's own designs:
    counted = [l_ for l_ in real["layers"] if cz.doses_calibrated(cv.kit_design(real, l_), l_, real["KIT_MAX_NEW"])]
    check(real["KIT"] != 3 or (by_design == {"match_ceil": [32, 33, 34, 35, 36, 37], "soft_ceil": [38, 52, 53, 54, 55, 56, 57, 58, 59],
                                             "break_rel": [39, 51]} and counted == [32, 33, 34, 35, 36, 37, 38, 39, 51, 52]),
          f"kit 3: match_ceil at 32-37, soft_ceil at 38 and 52 to 59, break_rel at 39 and 51; doses counted calibrated at the ten "
          f"whose trial passed and at none of 53-59 ({by_design}, {counted})")
    repo_board = json.loads((ROOT_DIR / "volunteers" / "board.json").read_text())
    opened = sorted(int(l_) for l_, st_ in repo_board["layers"].items() if st_.get("state") == "open")
    check(repo_board.get("kit") == real["KIT"] and set(opened) <= set(real["layers"]),
          f"the repository's board is the kit's, and every layer open on it is one the kit takes (kit {repo_board.get('kit')}, open {opened})")
    check(cv.kit_design(kit, 2) != cv.kit_design(kit, 3) and kit["designs"] == {2: cv.kit_design(kit, 2), 3: cv.kit_design(kit, 3)},
          f"the stand-in's kit has a design a layer, so a kit of more than one is run here ({kit['designs']})")
    # a kit of one design, as kits 1 and 2 were (KIT_DESIGN, no KIT_DESIGNS), is read as before; a layer left without a
    # design is refused, by the tool and by the script
    lines = cv.KIT_FILE.read_text().splitlines()
    one = tmp / "kit_one_design.env"
    one.write_text("\n".join([ln for ln in lines if not ln.startswith(("KIT_DESIGNS=", "KIT_DRY_DESIGNS="))] + ["KIT_DESIGN=break_rel"]) + "\n")
    gap = tmp / "kit_gap.env"
    gap.write_text("\n".join(ln if not ln.startswith("KIT_DRY_DESIGNS=") else f'KIT_DRY_DESIGNS="{LAYER}:{design}"' for ln in lines) + "\n")
    k_one, k_gap = cv.read_kit(one), cv.read_kit(gap, dry=True)
    other_dry = [l_ for l_ in kit["layers"] if l_ != LAYER][0]
    try:
        cv.kit_design(k_gap, other_dry)
        refused = ""
    except ValueError as exc:
        refused = str(exc)
    check(all(cv.kit_design(k_one, l_) == "break_rel" for l_ in k_one["layers"]) and cv.kit_problems(k_one) == []
          and "names no design for layer" in refused and any("names no design" in w_ for w_ in cv.kit_problems(k_gap))
          and cv.kit_design(k_gap, other_dry, "break_rel") == "break_rel",
          f"a kit of one design is read as kits 1 and 2 were; a layer without a design is refused ({refused})")
    r = run(["bash", CENSUS_SH, "--dry-run", "--root", tmp / "volunteer-gap", "reserve", str(other_dry)], {**env, cv.KIT_FILE_ENV: str(gap)})
    r1 = run(["bash", CENSUS_SH, "--dry-run", "--root", tmp / "volunteer-one", "reserve", str(other_dry)], {**env, cv.KIT_FILE_ENV: str(one)})
    check(r.returncode != 0 and f"names no design for layer {other_dry}" in r.stderr and not list((tmp / "volunteer-gap" / "claims").glob("*.claim"))
          and r1.returncode == 0 and last_line(r1.stdout, "CENSUS RESERVE"),
          f"census.sh: a layer the kit names no design for is not claimed; a kit of one design still claims ({r.stderr[-200:]}{r1.stderr[-200:]})")
    # the canary step's second half, the writer's floor, takes a design by its questions: one whose doses are set a layer
    # at a time has them too (it was refused by name before kit 3), and a name the census does not know is refused
    r = run([PY, "-I", "scripts/verify_atlas_writer.py", "--stand-in", "--layer", LAYER, "--design", "match_ceil", "--batches", kit["KIT_BATCH"],
             "--max-new", kit["KIT_MAX_NEW"], "--seed", kit["KIT_SEED"], "--unpacker", kit["KIT_UNPACKER"], "--out", tmp / "floor_match.json"], env)
    r2 = run([PY, "-I", "scripts/verify_atlas_writer.py", "--stand-in", "--layer", LAYER, "--design", "no_such_design", "--batches",
              kit["KIT_BATCH"], "--out", tmp / "floor_none.json"], env)
    check(r.returncode == 0 and (tmp / "floor_match.json").exists() and r2.returncode != 0 and "match_ceil" in r2.stderr
          and not (tmp / "floor_none.json").exists(),
          f"the writer's floor is measured for match_ceil's questions, and an unknown design is refused ({r.stderr[-300:]})")
    section("the kit's file: a design a layer, the board's open layers, a kit of one design, the writer's floor", before)

    # 1. the home card
    before = len(failures)
    r = run([PY, "-I", "scripts/census_run.py", "--stand-in", "--layer", LAYER, *args, "--out-dir", tmp / "home-runs", "--no-rank"], env)
    home_run = tmp / "home-runs" / f"L{LAYER}-{design}.jsonl"
    check(r.returncode == 0 and home_run.exists(), f"the home card's stand-in run ({r.stderr[-300:]})")
    r = run([PY, "-I", INTAKE_PY, "--dry-run", "--board", "", "home-canaries", "--layers", ",".join(map(str, kit["layers"])),
             "--baselines-from", home_run], env)
    prints = json.loads((canaries / "fingerprints.json").read_text()) if (canaries / "fingerprints.json").exists() else {}
    base = json.loads((canaries / "baselines.json").read_text())["baselines"] if (canaries / "baselines.json").exists() else {}
    check(r.returncode == 0 and len(prints) == len(kit["layers"]) and len(base) == 6
          and sorted(prints) == sorted(f"L{l_}-{cv.kit_design(kit, l_)}.canary.jsonl" for l_ in kit["layers"])
          and "NOT the one in baselines.json" not in r.stdout,
          f"home-canaries: a canary a kit layer, each with its layer's design, and six unpushed answers that every canary's "
          f"own agrees with ({sorted(prints)}; {r.stdout[-300:]}{r.stderr[-300:]})")
    # a canary already in the folder is never written over: it is taken again beside the one kept and compared, as a
    # layer carried over from an earlier kit must be; a kept canary the code no longer writes fails the command
    kept_bytes = {p.name: p.read_bytes() for p in canaries.glob("L*.canary.json*")}
    r = run([PY, "-I", INTAKE_PY, "--dry-run", "--board", "", "home-canaries", "--layers", ",".join(map(str, kit["layers"])),
             "--baselines-from", home_run], env)
    check(r.returncode == 0 and r.stdout.count("taken again beside it: the same as the one kept") == len(kit["layers"])
          and "0 canaries taken" in r.stdout and {p.name: p.read_bytes() for p in canaries.glob("L*.canary.json*")} == kept_bytes
          and json.loads((canaries / "fingerprints.json").read_text()) == prints,
          f"home-canaries a second time: each canary kept, taken again beside it and the same; nothing written over ({r.stdout[-300:]}{r.stderr[-300:]})")
    spoiled = tmp / "canaries-spoiled"
    shutil.copytree(canaries, spoiled)
    first_layer = kit["layers"][0]
    sp = spoiled / f"L{first_layer}-{cv.kit_design(kit, first_layer)}.canary.jsonl"
    sp_rows = [json.loads(x) for x in sp.read_text().splitlines() if x.strip()]
    sp_rows[-1]["token_ids"] = list(sp_rows[-1]["token_ids"][:-1]) + [int(sp_rows[-1]["token_ids"][-1]) ^ 1]
    sp.write_text("".join(json.dumps(x) + "\n" for x in sp_rows))
    sp_bytes = sp.read_bytes()
    r = run([PY, "-I", INTAKE_PY, "--dry-run", "--board", "", "home-canaries", "--layers", ",".join(map(str, kit["layers"])),
             "--baselines-from", home_run], {**env, "THEODOLITE_KIT_HOME_CANARIES": str(spoiled)})
    check(r.returncode != 0 and f"{sp.name}: already here, so taken again beside it: NOT the same as the one kept" in r.stdout
          and f"NOT the bits kept: {sp.name}" in r.stdout and sp.read_bytes() == sp_bytes
          and r.stdout.count("the same as the one kept") == len(kit["layers"]),
          f"home-canaries on a kept canary the code no longer writes: said, failed, and not written over ({r.stdout[-300:]}{r.stderr[-200:]})")
    # a layer the kit names no design for is not taken as a canary, unless its design is given: an earlier kit's layer
    # taken again in a scratch folder, to show the code still writes that kit's bits
    stray = [l_ for l_ in range(6) if l_ not in kit["layers"]][0]
    again = {**env, "THEODOLITE_KIT_HOME_CANARIES": str(tmp / "canaries-again")}
    r = run([PY, "-I", INTAKE_PY, "--dry-run", "--board", "", "home-canaries", "--layers", stray, "--baselines-from", home_run], again)
    r2 = run([PY, "-I", INTAKE_PY, "--dry-run", "--board", "", "home-canaries", "--layers", stray, "--design", "break_rel",
              "--baselines-from", home_run], again)
    check(r.returncode != 0 and f"names no design for layer {stray}" in r.stdout + r.stderr
          and r2.returncode == 0 and (tmp / "canaries-again" / f"L{stray}-break_rel.canary.jsonl").exists()
          and not (canaries / f"L{stray}-break_rel.canary.jsonl").exists(),
          f"home-canaries: a layer outside the kit is taken only with its design named, into the folder given ({r.stderr[-200:]}{r2.stderr[-200:]})")
    section("the home card: a stand-in run, the kit layers' canaries, the unpushed answers", before)

    # 2. the volunteer
    before = len(failures)
    v = ["bash", CENSUS_SH, "--dry-run", "--root", vroot]
    r = run(v + ["check"], env)
    check(r.returncode == 0 and "can take a layer" in r.stdout, f"check ({r.stdout[-400:]})")
    r = run(v + ["reserve", "47"], env)
    check(r.returncode != 0 and "the kit takes layers" in r.stderr, "a layer outside the kit is refused")
    r = run(v + ["reserve", str(LAYER), "--name", "tester"], env)
    reserve = last_line(r.stdout, "CENSUS RESERVE")
    claim = {k: v_ for k, _, v_ in (ln.partition("=") for ln in (vroot / "claims" / f"L{LAYER}.claim").read_text().splitlines())} \
        if (vroot / "claims" / f"L{LAYER}.claim").exists() else {}
    code = claim.get("code", "")
    check(r.returncode == 0 and reserve and cv.claim_code(claim.get("secret", "")) == code and f"claim={code}" in reserve
          and oct((vroot / "claims" / f"L{LAYER}.claim").stat().st_mode & 0o777) == "0o600", f"reserve: the claim kept, its code posted ({r.stdout[-300:]})")
    r2 = run(v + ["reserve", str(LAYER)], env)
    check(last_line(r2.stdout, "CENSUS RESERVE") == reserve, "reserve again: the same claim, not a new one")
    r = run(v + ["setup"], env)
    check(r.returncode == 0 and "self-test" in r.stdout and (fake_home / ".bashrc").read_text() == "# the volunteer's own\n",
          f"setup: setup.sh's dry run, ~/.bashrc untouched ({r.stdout[-300:]}{r.stderr[-300:]})")
    r = run(v + ["run", str(LAYER)], env)
    check(r.returncode != 0 and "take the canary first" in r.stderr, "run before the canary is refused")
    r = run(v + ["canary", str(LAYER)], env)
    cc = vroot / "atlas_data" / "stand-in" / "census" / "canary" / f"L{LAYER}.canary_check.json"
    check(r.returncode == 0 and cc.exists() and json.loads(cc.read_text())["bits"] == "home", f"canary: the home card's bits ({r.stdout[-300:]}{r.stderr[-300:]})")
    r = run(v + ["run", str(LAYER), "--wait"], env)
    start = last_line(r.stdout, "CENSUS START")
    run_p = vroot / "atlas_data" / "stand-in" / "census" / "runs" / f"L{LAYER}-{design}.jsonl"
    check(r.returncode == 0 and "bits=home" in start and run_p.exists(), f"run: detached, waited for, the START line ({r.stdout[-300:]}{r.stderr[-300:]})")
    r = run(v + ["status", str(LAYER)], env)
    progress = last_line(r.stdout, "CENSUS PROGRESS")
    check(r.returncode == 0 and "done=100%" in progress and "the run is whole" in r.stdout, f"status: whole, the PROGRESS line ({r.stdout[-300:]})")
    r = run(v + ["pack", str(LAYER)], env)
    done = last_line(r.stdout, "CENSUS DONE")
    bundle = vroot / "outbox" / f"L{LAYER}-{design}-{code}.tar"
    check(r.returncode == 0 and bundle.exists() and f"claim={code}" in done, f"pack: the bundle and the DONE line ({r.stdout[-300:]}{r.stderr[-300:]})")
    if bundle.exists():
        with tarfile.open(bundle) as t:
            blob = b"".join(t.extractfile(m).read() for m in t.getmembers() if not m.name.endswith(".zst"))
        check(str(tmp).encode() not in blob and str(ROOT_DIR).encode() not in blob and str(Path.home()).encode() not in blob,
              "the bundle names no folder of this machine")
    section("the volunteer: check, reserve, setup, canary, run, status, pack", before)

    # 3. the board
    before = len(failures)
    board = tmp / "board.json"
    board.write_text(json.dumps({"kit": kit["KIT"], "repo": "whp199/theodolite", "issue": None, "seen_comments": [], "claims": [],
                                 "layers": {str(l): {"state": "open"} for l in kit["layers"]}}))
    b = [PY, "-I", BOARD_PY, "--board", board]
    benv = {**env, "THEODOLITE_BOARD_NOW": "1793390400"}  # 2026-10-30T20:00Z
    r = run(b + ["add", ">>98765\n" + reserve], benv)
    check(r.returncode == 0 and f"L{LAYER} reserved by {code}" in r.stdout, f"a RESERVE in a quoted post ({r.stdout[:200]})")
    other = kit["layers"][0]
    r = run(b + ["add", f"CENSUS RESERVE L{LAYER} claim=0badc0de name=late"], benv)
    check(r.returncode == 1 and "refused" in r.stdout and "held by" in r.stdout, "a RESERVE on a held layer is refused")
    r = run(b + ["add", f"CENSUS RESERVE L{other} claim={code}"], benv)
    check(r.returncode == 1 and "one layer a claim" in r.stdout, "one claim, one layer")
    r = run(b + ["add", start, progress.replace("done=100%", "done=40% eta=2026-11-01T08:00Z")], benv)
    check(r.returncode == 0 and "running" in r.stdout and "held until 2026-11-02T08:00Z" in r.stdout, f"START and PROGRESS ({r.stdout[:300]})")
    # one post holding a START, its card's name wrapped over a line break, and a PROGRESS (L48's, 10 November 2026)
    card = cv.parse_line(start)["card"]
    wrapped = start.replace(f'card="{card}"', 'card="' + card.replace(" ", "\n", 1) + '"')
    post = f"{wrapped}\n\n" + progress.replace("done=100%", "done=50% eta=2026-11-01T09:00Z")
    r = run(b + ["add", post], benv)
    held = next(c for c in json.loads(board.read_text())["claims"] if c["code"] == code)
    check(" " in card and "\n" in wrapped and r.returncode == 0 and "held until 2026-11-02T09:00Z" in r.stdout
          and held.get("card") == card and held.get("done") == "50%",
          f"a post with two lines, a card's name wrapped: both read, the card whole ({r.stdout[:300]}, {held.get('card')!r})")
    r = run(b + ["add", f"CENSUS RELEASE L{LAYER} claim={code} secret=wrong"], benv)
    check(r.returncode == 1 and "without the claim's secret" in r.stdout, "a RELEASE without the secret is refused")
    r = run(b + ["add", f"CENSUS RESERVE L{other} claim=0badc0de name=lapses"], benv)
    r = run(b + ["expire"], {**env, "THEODOLITE_BOARD_NOW": str(1793390400 + 37 * 3600)})
    check("L%d 0badc0de lapsed" % other in r.stdout and f"L{LAYER} {code} lapsed" not in r.stdout,
          f"a claim past its hold lapses, a running one holds ({r.stdout[:300]})")
    signed = "https://objects.example.org/bucket/b.tar?X-Amz-Credential=" + "A%2F" * 60 + "&X-Amz-Signature=" + "e4" * 32
    r = run(b + ["add", done.replace("PASTE_LINK_HERE", signed)], benv)
    check(r.returncode == 0 and "submitted" in r.stdout and "link given" in r.stdout, f"DONE with a signed link of {len(signed)} characters ({r.stdout[:200]})")
    held = next(c for c in json.loads(board.read_text())["claims"] if c["code"] == code)
    check(held.get("bundle_sha256") and held["bundle_sha256"] in done and held.get("link") == signed, "the DONE line's sha256 and link kept")
    r = run(b + ["render", "--out", tmp / "render", "--no-png"], benv)
    txt = (tmp / "render" / "board.txt").read_text() if (tmp / "render" / "board.txt").exists() else ""
    check(r.returncode == 0 and 0 < len(txt) <= 2000 and (tmp / "render" / "BOARD.md").exists(), "the renders, the post under 2,000 characters")
    # the renders once every kit layer is done (a copy of the board), and with one layer closed for the home card instead
    for case, closed in (("done", None), ("closed", kit["layers"][-1])):
        cb = json.loads(board.read_text())
        cb["layers"] = {str(l): ({"state": "closed"} if l == closed else {"state": "done", "by": "home card"}) for l in kit["layers"]}
        (tmp / f"board_{case}.json").write_text(json.dumps(cb))
        out = tmp / f"render_{case}"
        r = run([PY, "-I", BOARD_PY, "--board", tmp / f"board_{case}.json", "render", "--out", out, "--no-png"], benv)
        txt = (out / "board.txt").read_text() if (out / "board.txt").exists() else ""
        md = (out / "BOARD.md").read_text() if (out / "BOARD.md").exists() else ""
        # the image's words too (render --no-png skips them): render_html in a process of its own
        h = run([PY, "-I", "-c", "import json, sys; sys.path.insert(0, sys.argv[1]); import census_board as b; "
                 "print('Every layer of the kit is done.' in b.render_html(json.load(open(sys.argv[2]))))",
                 BOARD_PY.parent, tmp / f"board_{case}.json"], benv)
        says = ["every layer of the kit is done" in txt, "Every layer of the kit is done" in md, h.stdout.strip() == "True"]
        asks = ["CENSUS line" in txt, "Holds:" in md]
        ok = all(says) and not any(asks) if case == "done" else all(asks) and not any(says)
        check(r.returncode == 0 and h.returncode == 0 and 0 < len(txt) <= 2000 and ok,
              f"the renders with every layer {'done: each says so' if case == 'done' else 'done but one closed: none says all done'}"
              f" (says {says}, asks {asks})")
    section("the board: claims, refusals, holds, lapses, the renders", before)

    # 3b. a board of sixty layers: most wait for their trial, one opens when its trial passes, one is stopped
    before = len(failures)
    wide = tmp / "board_wide.json"
    states = {**{str(l): {"state": "waiting"} for l in range(60)}, **{str(l): {"state": "done", "by": "home card"} for l in range(40, 51)}}
    wide.write_text(json.dumps({"kit": kit["KIT"], "repo": "whp199/theodolite", "issue": None, "seen_comments": [], "claims": [], "layers": states}))
    w = [PY, "-I", BOARD_PY, "--board", wide]
    r = run(w + ["add", f"CENSUS RESERVE L39 claim=0badc0de name=early kit={kit['KIT']}"], benv)
    check(r.returncode == 1 and "refused" in r.stdout and "not open (trial first)" in r.stdout, f"a RESERVE on a layer still waiting for its trial is refused ({r.stdout[:200]})")
    run(w + ["layer", "39", "open", "--note", "its trial passed"], benv)
    run(w + ["layer", "57", "stopped", "--note", "its trial failed"], benv)
    r = run(w + ["add", f"CENSUS RESERVE L39 claim=0badc0de name=early kit={kit['KIT'] + 1}"], benv)
    check(r.returncode == 1 and "refused" in r.stdout and f"the board is kit {kit['KIT']}'s" in r.stdout, f"a RESERVE made with another kit is refused ({r.stdout[:200]})")
    r = run(w + ["add", f"CENSUS RESERVE L57 claim=0badc0de name=early kit={kit['KIT']}"], benv)
    check(r.returncode == 1 and "not open (not taken)" in r.stdout, "a RESERVE on a stopped layer is refused")
    r = run(w + ["add", f"CENSUS RESERVE L39 claim=0badc0de name=early kit={kit['KIT']}"], benv)
    check(r.returncode == 0 and "L39 reserved by 0badc0de" in r.stdout, f"a RESERVE on the layer whose trial passed is taken ({r.stdout[:200]})")
    r = run(w + ["layer", "39", "waiting"], benv)
    check(r.returncode != 0 and "held by claim 0badc0de" in r.stderr, "a held layer's state is not changed under its claim")
    r = run(w + ["render", "--out", tmp / "render_wide", "--no-png"], benv)
    txt = (tmp / "render_wide" / "board.txt").read_text() if (tmp / "render_wide" / "board.txt").exists() else ""
    md = (tmp / "render_wide" / "BOARD.md").read_text() if (tmp / "render_wide" / "BOARD.md").exists() else ""
    check(r.returncode == 0 and 0 < len(txt) <= 2000 and "L0–38 trial first" in txt and "L51–56 trial first" in txt and "L58–59 trial first" in txt
          and "L57 not taken" in txt and "L39 reserved" in txt and "| 0–38 | 1–39 | trial first |" in md and "trial first* is not open yet" in md
          and "*not taken* is not offered" in md
          and md.count("\n| ") < 24,
          f"sixty layers stay a board: the waiting ones drawn as runs, the post under 2,000 characters ({len(txt)} characters)")
    # the kit command: a board of the first kit brought to this one, its new layers waiting, its record kept
    old = tmp / "board_old.json"
    old.write_text(json.dumps({"kit": kit["KIT"] - 1, "repo": "whp199/theodolite", "issue": 1, "seen_comments": [5], "updated_at": "x",
                               "claims": [{"code": "aaaaaaaa", "layer": 99, "name": "kept", "state": "accepted", "at": "2026-10-31T03:41Z"}],
                               "layers": {"99": {"state": "done", "by": "kept aaaaaaaa"}}}))
    r = run([PY, "-I", BOARD_PY, "--board", old, "kit"], benv)
    ob = json.loads(old.read_text())
    check(r.returncode == 0 and ob["kit"] == kit["KIT"] and ob["layers"]["99"] == {"state": "done", "by": "kept aaaaaaaa"}
          and all(ob["layers"].get(str(l), {}).get("state") == "waiting" and "trial" in ob["layers"][str(l)].get("note", "")
                  for l in real["layers"])  # (the command reads the kit's own file, not the stand-in's layers)
          and ob["claims"][0]["code"] == "aaaaaaaa"
          and ob["issue"] == 1 and ob["seen_comments"] == [5],
          f"the kit command: the board is this kit's, its layers added waiting, the first kit's record kept ({r.stdout[:200]})")
    # the volunteer's reserve asks the published board: a waiting layer, a held one and another kit's board are refused
    pub = tmp / "published.json"
    other_layer = [l for l in kit["layers"] if l != LAYER][0]

    def published(state: Dict[str, Any], claims: List[Dict[str, Any]], k: int = kit["KIT"]) -> Dict[str, str]:
        pub.write_text(json.dumps({"kit": k, "claims": claims, "layers": {str(other_layer): state}}))
        return {**env, "THEODOLITE_BOARD_JSON": pub.as_uri()}

    v2 = ["bash", CENSUS_SH, "--dry-run", "--root", tmp / "volunteer2"]
    r = run(v2 + ["reserve", str(other_layer)], published({"state": "waiting"}, []))
    check(r.returncode != 0 and "not open on the board (trial first)" in r.stderr and not (tmp / "volunteer2" / "claims" / f"L{other_layer}.claim").exists(),
          f"reserve: a layer the board shows waiting for its trial is not claimed ({r.stderr[-200:]})")
    r = run(v2 + ["reserve", str(other_layer)], published({"state": "open"}, [{"code": "0badc0de", "layer": other_layer, "state": "running"}]))
    check(r.returncode != 0 and "held by claim 0badc0de" in r.stderr, "reserve: a layer someone holds on the board is not claimed")
    r = run(v2 + ["reserve", str(other_layer)], published({"state": "open"}, [], k=kit["KIT"] + 1))
    check(r.returncode != 0 and "another kit's" in r.stderr, "reserve: a board of another kit is refused")
    r = run(v2 + ["reserve", str(other_layer)], {**env, "THEODOLITE_BOARD_JSON": (tmp / "no-such-board.json").as_uri()})
    check(r.returncode == 0 and "could not read the board" in r.stdout + r.stderr and last_line(r.stdout, "CENSUS RESERVE"),
          "reserve: with the board out of reach the claim is made, with a warning to look")
    r = run(["bash", CENSUS_SH, "--dry-run", "--root", tmp / "volunteer3", "reserve", str(other_layer)], published({"state": "open"}, []))
    check(r.returncode == 0 and last_line(r.stdout, "CENSUS RESERVE") and "warning" not in r.stdout + r.stderr, "reserve: an open layer is claimed")
    section("a board of sixty layers: trial first, opened, stopped; the kit command; reserve asks the board", before)

    # 4. the intake
    before = len(failures)
    i = [PY, "-I", INTAKE_PY, "--dry-run", "--board", board]
    work = tmp / "work"
    r = run(i + ["check", bundle, "--work", work], env)
    chk = json.loads((work / "check.json").read_text()) if (work / "check.json").exists() else {}
    check(r.returncode == 0 and not chk.get("fails") and chk.get("bits") == "home", f"check: whole, the kit's, home bits ({r.stdout[-400:]})")
    r = run(i + ["spot", work, "--run"], env)
    check(r.returncode == 0, f"spot: the chunks written again ({r.stderr[-300:]})")
    r = run(i + ["judge", work], env)
    ver = json.loads((work / "verdict.json").read_text()) if (work / "verdict.json").exists() else {}
    check(r.returncode == 0 and ver.get("accepted") and ver["spot"]["identical"] == ver["spot"]["answers"] > 0, f"judge: bit for bit ({r.stdout[-300:]})")
    # an owner's NAME line on the board (a claim posted with the kit's default name): the provenance names them so
    bj = json.loads(board.read_text())
    claim = next(c for c in bj["claims"] if c["code"] == chk.get("claim"))
    claim["name"] = "Named-by-owner"
    board.write_text(json.dumps(bj, indent=1))
    into = tmp / "census-runs"
    r = run(i + ["accept", work, "--into", into], env)
    stem = f"L{LAYER}-{design}"
    check(r.returncode == 0 and all((into / f"{stem}{s}").exists() for s in (".jsonl", ".meta.json", ".provenance.json", ".units.jsonl")),
          f"accept: the run, its meta, provenance and ranking ({r.stdout[-300:]}{r.stderr[-300:]})")
    prov = json.loads((into / f"{stem}.provenance.json").read_text()) if (into / f"{stem}.provenance.json").exists() else {}
    check(prov.get("name") == "Named-by-owner" and prov.get("bundle_name") == chk.get("name"),
          f"the provenance names the volunteer as the board does, the bundle's name kept ({prov.get('name')}, {prov.get('bundle_name')})")
    bj = json.loads(board.read_text())
    check(bj["layers"][str(LAYER)]["state"] == "done", "the layer done on the board")
    r = run(i + ["accept", work, "--into", into], env)
    check(r.returncode != 0 and "is there already" in r.stderr, "a layer's run is not replaced")
    section("the intake: checked, spot-checked bit for bit, accepted, ranked, on the board", before)

    # 5. the refusals
    before = len(failures)
    rows = [json.loads(x) for x in run_p.read_text().splitlines()]
    canary = run_p.parent.parent / "canary" / f"{stem}.canary.jsonl"
    canary_rows = [json.loads(x) for x in canary.read_text().splitlines()]

    def repack(label: str, change_run: Any, change_canary: Any = None) -> Path:
        """The volunteer's run (and canary) changed as a cheat or another card would change them, packed again."""
        root = tmp / f"v-{label}"
        shutil.copytree(vroot, root, ignore=shutil.ignore_patterns("outbox", "logs"))
        envf = root / "env.sh"  # setup.sh wrote the folder's own path into it
        envf.write_text(envf.read_text().replace(str(vroot), str(root)))
        rp = root / run_p.relative_to(vroot)
        rp.write_text("".join(json.dumps(change_run(dict(r_), k)) + "\n" for k, r_ in enumerate(rows)))
        if change_canary:
            cp = root / canary.relative_to(vroot)
            cp.write_text("".join(json.dumps(change_canary(dict(r_), k)) + "\n" for k, r_ in enumerate(canary_rows)))
        rr = run(["bash", CENSUS_SH, "--dry-run", "--root", root, "pack", str(LAYER)], env)
        check(rr.returncode == 0, f"{label}: packed again ({rr.stderr[-300:]})")
        return root / "outbox" / bundle.name

    def through(label: str, tar: Path) -> Dict[str, Any]:
        w = tmp / f"work-{label}"
        nb = tmp / f"board-{label}.json"
        nb.write_text(json.dumps({"kit": kit["KIT"], "repo": "x/y", "issue": None, "seen_comments": [], "claims": [],
                                  "layers": {str(l): {"state": "open"} for l in kit["layers"]}}))
        run([PY, "-I", BOARD_PY, "--board", nb, "add", reserve], benv)
        ii = [PY, "-I", INTAKE_PY, "--dry-run", "--board", nb]
        c1 = run(ii + ["check", tar, "--work", w], env)
        out = {"check": c1, "board": nb}
        if c1.returncode == 0:
            run(ii + ["spot", w, "--run"], env)
            out["judge"] = run(ii + ["judge", w], env)
            out["verdict"] = json.loads((w / "verdict.json").read_text()) if (w / "verdict.json").exists() else {}
        out["claim"] = next((c for c in json.loads(nb.read_text())["claims"] if c["code"] == code), {})
        return out

    jobs = cv.expected_jobs(json.loads(cz.meta_path_of(run_p).read_text()))
    plan = cz.plan_batches(jobs, kit["KIT_BATCH"], "12 x batch")
    in_chunk = {cz.row_key(jobs[j]): c for c, _, js in plan for j in js}

    def tamper(r_: Dict[str, Any], k: int) -> Dict[str, Any]:  # every pushed answer's first piece moved: a cheat's made-up rows
        if r_["kind"] != "baseline" and r_["token_ids"]:
            r_["token_ids"] = [(r_["token_ids"][0] + 1) % 500] + r_["token_ids"][1:]
        return r_

    # a run whose canary is honest and whose later chunks are made up: caught where the spot draws a made-up chunk
    t = through("tampered", repack("tampered", lambda r_, k: tamper(r_, k) if in_chunk.get(cz.row_key(r_), 0) > 0 else r_))
    check(t["check"].returncode == 0 and t.get("verdict", {}).get("accepted") is False and t["claim"].get("state") == "rejected",
          f"a run made up after its canary is rejected by the spot-check ({t['check'].stdout[-300:]})")
    # made up throughout, its canary too (so not the home card's bits): fails the rules for another card class
    t = through("madeup", repack("madeup", tamper, tamper))
    check(t["check"].returncode == 0 and t.get("verdict", {}).get("accepted") is False
          and any("first word-piece" in w for w in t["verdict"].get("why_not", [])), f"a made-up run fails the first-piece rule ({t.get('verdict')})")

    def nudge(r_: Dict[str, Any], k: int) -> Dict[str, Any]:  # another card's sums: each log-probability moved a little
        r_["logprob"] = [round(x - 0.0005, 4) if x < -0.01 else x for x in r_["logprob"]]
        return r_

    t = through("othercard", repack("othercard", nudge, nudge))
    chk2 = json.loads((tmp / "work-othercard" / "check.json").read_text()) if (tmp / "work-othercard" / "check.json").exists() else {}
    check(chk2.get("bits") == "own" and t.get("verdict", {}).get("accepted") is True,
          f"another card class within the kit's rules is accepted ({chk2.get('bits')}, {t.get('verdict', {}).get('why_not')})")
    # the unpushed answers: parting from ours is taken only at a near-tie (a piece ours wrote with probability under 0.6)
    ok_cmp = {"answers": 10, "present": 10, "identical": 0, "first_piece": 1.0, "lp_median": 0.0, "lp_p99": 0.0}
    hid, unsure, sure = [5, 6, 7, 8], [-0.0, math.log(0.5), -0.01, -0.0], [-0.0, -0.05, -0.01, -0.0]
    tie = cv.baseline_part(hid, unsure, [5, 9, 9])
    check(tie == [1, 4, 0.5] and cv.judge(ok_cmp, False, {"orange": tie})[0]
          and not cv.judge(ok_cmp, False, {"orange": cv.baseline_part(hid, sure, [5, 9, 9])})[0]
          and cv.judge(ok_cmp, False, {"orange": cv.baseline_part(hid, sure, hid)})[0],
          "an unpushed answer parting at a near-tie is taken, at a sure piece refused, the same answer taken")
    # a bundle whose secret does not give its claim
    bad = tmp / "badsecret"
    bad.mkdir()
    with tarfile.open(bundle) as tf:
        tf.extractall(bad, filter="data")
    man = json.loads((bad / "manifest.json").read_text())
    man["secret"] = "0" * 32
    (bad / "manifest.json").write_text(json.dumps(man))
    bad_tar = tmp / "bad" / bundle.name
    bad_tar.parent.mkdir()
    with tarfile.open(bad_tar, "w") as tf:
        for p in sorted(bad.iterdir()):
            tf.add(p, arcname=p.name)
    t = through("badsecret", bad_tar)
    check(t["check"].returncode != 0 and "secret does not give" in t["check"].stdout and t["claim"].get("state") == "reserved",
          f"a bundle without its claim's secret is refused, the board untouched ({t['check'].stdout[-200:]}, {t['claim'].get('state')})")
    # a file a pack does not write
    odd = tmp / "odd" / bundle.name
    odd.parent.mkdir()
    shutil.copy(bundle, odd)
    with tarfile.open(odd, "a") as tf:
        extra = tmp / "run.sh"
        extra.write_text("echo hi\n")
        tf.add(extra, arcname="run.sh")
    r = run([PY, "-I", INTAKE_PY, "--dry-run", "--board", "", "check", odd, "--work", tmp / "work-odd"], env)
    check(r.returncode != 0 and "which a pack does not write" in r.stderr and not (tmp / "work-odd" / "bundle" / "run.sh").exists(),
          "a bundle holding a file a pack does not write is not unpacked")
    section("the refusals: made up after its canary, made up throughout, another card class taken, a wrong secret, a stray file", before)

    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
