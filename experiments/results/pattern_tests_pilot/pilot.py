#!/usr/bin/env python3
"""The two pattern tests (backend/pattern_tests.py) tried on the 31B for the first time, 14 November 2026.

Everything goes through the running server's own routes (127.0.0.1:8765, the 31B loaded), as a scientist's tools do.
The passages a pattern was written from (fit/) and the readers' sheets (readers/sheet*.txt) are text of the corpus and
are not kept in the repository; `fit` and `start` write them again, drawing by seed. The readers' answers are kept.

    survey   At each of LAYERS, FACES faces drawn at random (seed SEED: a neuron and a sign). For each, a blind test is
             begun and left unanswered, to see whether one can be made at all: whether the face has MIN_PAIRS kept-back
             passages, and whether a quiet passage is found for each (the test refuses a face that fires too widely).
             Writes survey.json.
    fit      For the faces a pattern is to be written for (the first AUTHORED of each of AUTHOR_LAYERS, in draw order,
             that the survey could begin a test for), the passages NOT kept back, cut as the test cuts its snippets.
             Writes fit/<face>.txt, which is all the pattern's writer reads.
    start    Begins the blind test for each face in patterns.json with the pattern written there, and writes the
             readers' sheets (readers/): each test is read twice, by two readers in fresh conversations, once with its
             own pattern and once with the pattern of another face of the same sheet (which of the two a reader has is
             drawn by seed, test by test, and no reader is told). Writes started.json (the test ids; the answers stay in
             the server's memory).
    written  Runs each face's written texts (patterns.json: `fire`, `quiet`). Writes written.json.
    alone    The written test's own check: each face's strongest passages (not kept back), read alone as plain text.
             Writes alone.json.
    context  How much text before it a face needs: the same passages read with 16, 60, 128 and 256 word-pieces before
             the peak, and with all the window had. Writes context.json.
    padded   The same passages' 60 word-pieces read behind 256 and 420 word-pieces of an unrelated document. Writes
             padded.json.
    context_survey  `context` over all 176 faces of the survey's draw. Writes context_survey.json.
    answer   Reads the readers' answers (readers/<sheet>.answers.txt), answers each test once with its own pattern's
             picks (kept on the face), takes the truth from the reply, and scores the other pattern's picks against it.
             Writes blind.json.
"""
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
from backend import pattern_tests as pt  # noqa: E402

API = "http://127.0.0.1:8765/api/atlas"
SEED = 20261114
LAYERS = [2, 8, 14, 20, 26, 31, 36, 40, 45, 50, 56]
FACES = 16
D_FF = 21504
AUTHOR_LAYERS = [2, 14, 26, 31, 36, 45, 56]
AUTHORED = 3
SHEETS = 3  # the authored faces are dealt into this many sheets; each sheet is read by two readers
# filler text for `padded`: the window of one of these faces' passages, which sits late in its window (face, which passage)
FILLERS = {"A": ("L2-n16560p", 0), "B": ("L26-n14541m", 1)}
JUDGE = "a fresh Claude Fable 5.2 conversation given the pattern alone (by hand; no endpoint is configured)"


def call(path, body=None, timeout=600):
    req = urllib.request.Request(API + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("detail")
        except ValueError:
            detail = None
        return {"refused": detail or f"HTTP {exc.code}", "status": exc.code}


def draw():
    rng = random.Random(SEED)
    return {layer: [(rng.randrange(D_FF), rng.choice("+-")) for _ in range(FACES)] for layer in LAYERS}


def key(layer, n, face):
    return f"L{layer}-n{n}{'p' if face == '+' else 'm'}"


def survey():
    out = {"what": "whether a blind test can be begun, face by face, on faces drawn at random", "seed": SEED, "faces": []}
    for layer, faces in draw().items():
        for i, (n, face) in enumerate(faces):
            t0 = time.time()
            got = call("/neuron/blind_test/start", {"layer": layer, "n": n, "face": face, "pattern": "(none: a survey)",
                                                    "pairs": pt.MAX_PAIRS, "seed": SEED + i})
            row = {"layer": layer, "n": n, "face": face, "order": i, "seconds": round(time.time() - t0, 2)}
            if "refused" in got:
                row["refused"] = got["refused"]
            else:
                row.update(begun=True, pairs=len(got["pairs"]), kept_back=got["kept_back"], stored=got["stored"])
            out["faces"].append(row)
            print(row, flush=True)
    (HERE / "survey.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


def authored():
    rows = json.loads((HERE / "survey.json").read_text())["faces"]
    out = []
    for layer in AUTHOR_LAYERS:
        out += [r for r in rows if r["layer"] == layer and r.get("begun")][:AUTHORED]
    return out


def fit():
    (HERE / "fit").mkdir(exist_ok=True)
    for r in authored():
        d = call(f"/neuron/passages?layer={r['layer']}&n={r['n']}&face={'%2B' if r['face'] == '+' else '-'}"
                 f"&k=16&before={pt.BEFORE}&after={pt.AFTER}&fold=fit")
        lines = [f"{key(r['layer'], r['n'], r['face'])}: layer {r['layer']}, neuron {r['n']}, face {r['face']}; "
                 f"{len(d['passages'])} passages not kept back ({d['kept_back']} kept back and not shown)",
                 "at the peaks: " + ", ".join(f"{x['tok']!r} x{x['n']}" for x in d["reads"][:6]),
                 "just after:   " + ", ".join(f"{x['tok']!r} x{x['n']}" for x in d["follows"][:6]), ""]
        for p in d["passages"]:
            lines.append(f"- {p['kind']} {p['rel']}: " + pt.render(p["units"]).replace("\n", "⏎"))
        (HERE / "fit" / f"{key(r['layer'], r['n'], r['face'])}.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines) + "\n")


def patterns():
    return json.loads((HERE / "patterns.json").read_text())["faces"]


def start():
    faces = patterns()
    rng = random.Random(SEED + 1)
    order = list(faces)
    rng.shuffle(order)
    sheets = [order[i::SHEETS] for i in range(SHEETS)]
    started = {"judge": JUDGE, "tests": {}}
    (HERE / "readers").mkdir(exist_ok=True)
    for s, sheet in enumerate(sheets):
        # the other pattern a test is read with: the next face's of its sheet
        other = {k: sheet[(i + 1) % len(sheet)] for i, k in enumerate(sheet)}
        texts = {"a": [], "b": []}
        for i, k in enumerate(sheet):
            f = faces[k]
            got = call("/neuron/blind_test/start", {"layer": f["layer"], "n": f["n"], "face": f["face"], "pattern": f["pattern"],
                                                    "pairs": pt.MAX_PAIRS, "seed": SEED + 2})
            if "refused" in got:
                started["tests"][k] = {"refused": got["refused"]}
                print(k, "refused:", got["refused"], flush=True)
                continue
            own_reader = rng.choice("ab")
            started["tests"][k] = {"test": got["test"], "pairs": len(got["pairs"]), "sheet": s, "label": f"T{i + 1}",
                                   "own_pattern_reader": own_reader, "other_pattern_of": other[k]}
            for reader in "ab":
                pattern = f["pattern"] if reader == own_reader else faces[other[k]]["pattern"]
                texts[reader] += [f"======== TEST T{i + 1} ========", "", pt.judge_prompt(pattern, got["pairs"]), ""]
            print(k, "begun:", len(got["pairs"]), "pairs", flush=True)
        head = [pt.JUDGE_SYSTEM.replace("Give no other text.", "").strip(), "",
                "Below are several separate tests, each of a different neuron with its own description. Treat each on its own.",
                "Answer every pair of every test, one line a pair, as `T2 3: b` (the test's label, the pair's number, a colon, a or b).",
                "Use only the description given with a test. Give no other text.", ""]
        for reader in "ab":
            (HERE / "readers" / f"sheet{s}{reader}.txt").write_text("\n".join(head + texts[reader]) + "\n")
    (HERE / "started.json").write_text(json.dumps(started, indent=1, ensure_ascii=False) + "\n")


def written():
    out = {}
    for k, f in patterns().items():
        got = call("/neuron/written_test", {"layer": f["layer"], "n": f["n"], "face": f["face"], "pattern": f["pattern"],
                                            "fire": f["fire"], "quiet": f["quiet"], "by": "scientist", "keep": True})
        got.pop("kept", None)
        out[k] = got
        print(k, got.get("said") or got.get("refused"), flush=True)
    (HERE / "written.json").write_text(json.dumps({"tests": out}, indent=1, ensure_ascii=False) + "\n")


def alone():
    """The written test's own check: each face's three strongest passages that are not kept back, cut as the tests cut
    them (BEFORE word-pieces before the peak, AFTER after) and read alone as plain text, beside what the index holds
    for them, read in their windows. A face that does not fire on its own passage read alone needs more than the
    snippet shows: more text before it, or its place in the window."""
    out = {}
    for k, f in patterns().items():
        d = call(f"/neuron/passages?layer={f['layer']}&n={f['n']}&face={'%2B' if f['face'] == '+' else '-'}"
                 f"&k=16&before={pt.BEFORE}&after={pt.AFTER}&fold=fit")
        seen, rows = set(), []
        for x in d["passages"]:
            if x["window"] not in seen and len(rows) < 3:
                seen.add(x["window"])
                rows.append(x)
        got = call("/neuron/written_test", {"layer": f["layer"], "n": f["n"], "face": f["face"], "pattern": "(its own passages, read alone)",
                                            "fire": ["".join(t for t, _ in x["units"]) for x in rows], "keep": False})
        out[k] = [{"in_window": x["rel"], "alone": t["rel"], "pieces_before": x["at"] - x["span"][0], "at_in_window": x["at"],
                   "peak_on": t["at"], "stored_on": "".join(t_ for t_, m in x["units"] if m)}
                  for x, t in zip(rows, got["texts"]["fire"])]
        print(k, [(r["in_window"], r["alone"], r["pieces_before"]) for r in out[k]], flush=True)
    (HERE / "alone.json").write_text(json.dumps({"what": alone.__doc__.split("\n\n")[0], "faces": out}, indent=1, ensure_ascii=False) + "\n")


def context():
    """How much text before it a face needs: each face's three strongest passages that are not kept back, read from the
    start mark with only the last 16, 60, 128 or 256 word-pieces before the peak, and with all its window had before
    it; the face's reading at the peak as a share of its extreme each time. Sixty is what a blind reader is shown."""
    out = {}
    for k, f in patterns().items():
        d = call(f"/neuron/passages?layer={f['layer']}&n={f['n']}&face={'%2B' if f['face'] == '+' else '-'}&k=16&before=4&after=0&fold=fit")
        seen, rows = set(), []
        for x in d["passages"]:
            if x["window"] not in seen and len(rows) < 3:
                seen.add(x["window"])
                rows.append(x)
        out[k] = []
        for x in rows:
            ids, at = x["ids"], x["at"]
            row = {"in_index": x["rel"], "at": at}
            for b in (16, 60, 128, 256, None):
                cut = ids[: at + 1] if b is None or at - b <= 1 else [ids[0]] + ids[at - b: at + 1]
                got = call("/reads", {"layer": f["layer"], "n": f["n"], "ids": cut})
                row["window" if b is None else str(b)] = round(got["units"][-1]["pieces"][-1] / d["extreme"], 3)
            out[k].append(row)
        print(k, [[r[c] for c in ("16", "60", "128", "256", "window")] + [r["at"]] for r in out[k]], flush=True)
    (HERE / "context.json").write_text(json.dumps({"what": context.__doc__, "faces": out}, indent=1, ensure_ascii=False) + "\n")


def strongest(f, n=3):
    """A face's `n` strongest passages that are not kept back, one a window, with the face's extreme."""
    d = call(f"/neuron/passages?layer={f['layer']}&n={f['n']}&face={'%2B' if f['face'] == '+' else '-'}&k=16&before=4&after=0&fold=fit")
    seen, rows = set(), []
    for x in d["passages"]:
        if x["window"] not in seen and len(rows) < n:
            seen.add(x["window"])
            rows.append(x)
    return rows, d["extreme"]


def padded():
    """Is it more text the face needs, or its own text? Each face's three strongest passages that are not kept back, cut
    to the 60 word-pieces before the peak as a blind reader sees them, and read behind 256 and 420 word-pieces of
    another document: the openings of two windows of the corpus that have nothing to do with the passage (FILLERS). A
    face that fires behind filler needed only to be far into a text; one that still does not needs what its own
    document said earlier."""
    faces = patterns()
    fillers = {}
    for name, (k, i) in FILLERS.items():
        x = strongest(faces[k])[0][i]
        fillers[name] = (x["window"], x["ids"][1:421])
        assert len(fillers[name][1]) == 420, (name, len(fillers[name][1]))
    out = {}
    for k, f in faces.items():
        rows, ext = strongest(f)
        out[k] = []
        for x in rows:
            ids, at = x["ids"], x["at"]
            row = {"in_index": x["rel"], "at": at}
            snip = ids[max(1, at - 60): at + 1]
            for name, (w, filler) in fillers.items():
                if w == x["window"]:
                    continue  # its own document is no filler
                for b in (256, 420):
                    got = call("/reads", {"layer": f["layer"], "n": f["n"], "ids": [ids[0]] + filler[:b] + snip})
                    row[f"{name}+{b}"] = round(got["units"][-1]["pieces"][-1] / ext, 3)
            out[k].append(row)
        print(k, [[r["in_index"]] + [r[c] for c in r if "+" in c] for r in out[k]], flush=True)
    (HERE / "padded.json").write_text(json.dumps({"what": padded.__doc__, "fillers": {n: {"face": FILLERS[n][0], "window": w} for n, (w, _) in fillers.items()},
                                                  "faces": out}, indent=1, ensure_ascii=False) + "\n")


def context_survey():
    """`context`, over every face of the survey's draw (176, at eleven layers): each face's three strongest passages
    that are not kept back, read with 60, 128 and 256 word-pieces before the peak and with all its window had."""
    out = []
    for layer, faces in draw().items():
        for n, face in faces:
            rows, ext = strongest({"layer": layer, "n": n, "face": face})
            got = []
            for x in rows:
                ids, at = x["ids"], x["at"]
                row = {"at": at}
                for b in (60, 128, 256, None):
                    cut = ids[: at + 1] if b is None or at - b <= 1 else [ids[0]] + ids[at - b: at + 1]
                    r = call("/reads", {"layer": layer, "n": n, "ids": cut})
                    row["window" if b is None else str(b)] = round(r["units"][-1]["pieces"][-1] / ext, 3)
                got.append(row)
            out.append({"layer": layer, "n": n, "face": face, "passages": got})
        done = [f for f in out if f["layer"] == layer]
        print(layer, {b: sum(sum(r[b] >= 0.5 * r["window"] for r in f["passages"]) >= 2 for f in done) for b in ("60", "128", "256")}, "of", len(done), flush=True)
    (HERE / "context_survey.json").write_text(json.dumps({"what": context_survey.__doc__, "faces": out}, indent=1, ensure_ascii=False) + "\n")


def read_answers(path):
    """{label: [pick, ...]} from a reader's lines, `T2 3: b`."""
    import re

    got = {}
    for m in re.finditer(r"(?im)^\W*(T\d+)\W+(\d+)\s*[:.)\-=]\s*\(?([ab])\b", path.read_text()):
        got.setdefault(m.group(1).upper(), {})[int(m.group(2))] = m.group(3).lower()
    return got


def answer():
    started = json.loads((HERE / "started.json").read_text())
    out = {"judge": started["judge"], "tests": {}}
    answers = {p.name[: -len(".answers.txt")]: read_answers(p) for p in (HERE / "readers").glob("*.answers.txt")}
    for k, t in started["tests"].items():
        if "refused" in t:
            out["tests"][k] = t
            continue
        picks = {}
        for reader in "ab":
            got = answers[f"sheet{t['sheet']}{reader}"].get(t["label"], {})
            picks[reader] = [got.get(i + 1) for i in range(t["pairs"])]
        own, other = picks[t["own_pattern_reader"]], picks["b" if t["own_pattern_reader"] == "a" else "a"]
        res = call("/neuron/blind_test/answer", {"test": t["test"], "picks": own, "judge": started["judge"], "by": "scientist", "keep": True})
        if "refused" in res:
            out["tests"][k] = {"refused": res["refused"]}
            print(k, "refused:", res["refused"], flush=True)
            continue
        truth = [p["truth"] for p in res["by_pair"]]
        out["tests"][k] = {"own_pattern": {x: res[x] for x in ("pairs", "right", "unanswered", "p")},
                           "other_pattern": {x: v for x, v in pt.blind_result(other, truth).items() if x in ("pairs", "right", "unanswered", "p")},
                           "other_pattern_of": t["other_pattern_of"], "by_pair": res["by_pair"], "picks_own": own, "picks_other": other}
        print(k, "own:", res["right"], "of", res["pairs"], "| other pattern:", out["tests"][k]["other_pattern"]["right"], flush=True)
    (HERE / "blind.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    {"survey": survey, "fit": fit, "start": start, "written": written, "alone": alone, "context": context, "padded": padded, "context_survey": context_survey, "answer": answer}[sys.argv[1]]()
