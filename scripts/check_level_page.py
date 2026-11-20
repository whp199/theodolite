#!/usr/bin/env python3
"""CPU verification suite for Levelling Phase 4: The Person's Page (`scripts/check_level_page.py`).

Tests all Phase 4 backend routes, frontend pure logic (`compare.ts` bundled with esbuild and run under node),
vault-overlay isolation (HTTP 409), model-free ledger replay (including live replay of the 19 Nov 2026 31B card
`verify_claim_book.jsonl`), Lexicon/Vault wiring, and zero kept-out words — with zero GPU access.

Run:
    /home/eric/bin/ai/theodolite2/.venv/bin/python -I scripts/check_level_page.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

torch.set_num_threads(1)

from fastapi.testclient import TestClient  # noqa: E402

import level_recorded_cards as recorded_cards  # noqa: E402
from atlas_stand_in import bind_server, stand_in, use_stand_in_atlas  # noqa: E402
from backend import level_book as lb  # noqa: E402
from backend import level_routes  # noqa: E402
from backend import levelling as lv  # noqa: E402
from backend import weight_edits as we  # noqa: E402

FRONTEND = ROOT_DIR / "frontend"
ESBUILD = FRONTEND / "node_modules" / ".bin" / "esbuild"

KEPT_OUT_RE = re.compile(
    r"\b(proof\w*|held|seal\w*|errat(?:um|a)|sighting\w*|plate\w*|station\w*|field[\s-]+books?|chance[\s-]+edits?)\b",
    re.I,
)

PASSED = 0
FAILED = 0


def check(cond: bool, msg: str) -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  [FAIL] {msg}", flush=True)


NODE_HARNESS = r"""
import * as C from "./compare";
import { RECORDED_REPLAY_CARDS } from "./compareRecorded";

const pOk = C.parseNeuronList("5313, 8801, 5313; 266 121");
const pEmpty = C.parseNeuronList("   ");
const pRange = C.parseNeuronList("21504");
const pNeg = C.parseNeuronList("-1");

const errDefault = C.validateCompareForm(C.DEFAULT_COMPARE_FORM);
const errBad = C.validateCompareForm({
  ...C.DEFAULT_COMPARE_FORM,
  layer: "0",
  neurons: "99999",
  dose: "0",
  question: "",
  controlsCount: "25",
  maxNew: "3",
  sampled: true,
  temperature: "0",
  towardWord: "she",
  awayWord: "",
});

const payDefault = C.buildComparePayload(C.DEFAULT_COMPARE_FORM);
const payRow = C.buildComparePayload({
  ...C.DEFAULT_COMPARE_FORM,
  op: "scale_rows",
  linear: "up_proj",
  layer: "3",
  neurons: "0, 1",
  factor: "0.5",
  answer: "",
  controlsFamily: "same_site",
  controlsCount: "2",
});

const out = {
  pOk, pEmpty, pRange, pNeg,
  errDefault, errBad,
  payDefault, payRow,
  plain: C.formatPlainEdit(C.DEFAULT_COMPARE_FORM),
  presets: C.COMPARE_PRESETS.map((p) => p.id),
  tones: {
    lead: C.statusTone("lead"),
    passed: C.statusTone("passed"),
    failed: C.statusTone("failed"),
    harms: C.statusTone("works but harms"),
    kept: C.statusTone("kept the base"),
    ok: C.statusTone("ok"),
  },
  batteries: {
    habitatRecorded: C.batteryOutcome((RECORDED_REPLAY_CARDS[0].batteries as Record<string, Record<string, unknown>>).habitat),
    unavailGating: C.batteryOutcome({ available: false, gating: true, passed: false }),
    availPass: C.batteryOutcome({ available: true, gating: true, passed: true }),
    availFail: C.batteryOutcome({ available: true, gating: true, passed: false }),
  },
  recordedCount: RECORDED_REPLAY_CARDS.length,
  recordedVerdicts: RECORDED_REPLAY_CARDS.map((c) => c.verdict),
  recorded: RECORDED_REPLAY_CARDS,
};
console.log(JSON.stringify(out));
"""


def run_section_1(tmp: Path) -> None:
    print("Section 1: Frontend compare.ts & compareRecorded.ts bundled with esbuild + node", flush=True)
    check(ESBUILD.exists(), f"esbuild exists at {ESBUILD}")
    entry = FRONTEND / "src" / "screens" / "vault" / "_tmp_check_compare.ts"
    bundle = tmp / "compare_bundle.mjs"
    try:
        entry.write_text(NODE_HARNESS, encoding="utf-8")
        r_build = subprocess.run(
            [str(ESBUILD), str(entry), "--bundle", "--format=esm", f"--outfile={bundle}"],
            cwd=str(FRONTEND), capture_output=True, text=True, timeout=30,
        )
        check(r_build.returncode == 0, f"esbuild bundled compare.ts cleanly: {r_build.stderr[:200]}")
        r_node = subprocess.run(["node", str(bundle)], capture_output=True, text=True, timeout=15)
        check(r_node.returncode == 0, f"node executed bundle cleanly: {r_node.stderr[:200]}")
        data = json.loads(r_node.stdout)
    finally:
        if entry.exists():
            entry.unlink()

    check(data["pOk"]["ok"] and data["pOk"]["neurons"] == [5313, 8801, 266, 121], "parseNeuronList parses & dedups")
    check(not data["pEmpty"]["ok"] and not data["pRange"]["ok"] and not data["pNeg"]["ok"],
          "parseNeuronList refuses empty, >21503, and negative indices")
    check(data["errDefault"] == {}, f"DEFAULT_COMPARE_FORM validates cleanly: {data['errDefault']}")
    for k in ("layer", "neurons", "dose", "question", "controlsCount", "maxNew", "temperature", "odds"):
        check(k in data["errBad"], f"validateCompareForm catches invalid {k}")

    # Page 1..60 -> wire 0..59 conversion & Pydantic server schema compatibility
    check(data["payDefault"]["layer"] == 50, "page layer 51 converts to wire layer 50")
    check(data["payRow"]["layer"] == 2 and data["payRow"]["linear"] == "up_proj" and not data["payRow"]["include_paired"],
          "row edit payload converts layer 3->2, keeps up_proj, disables paired on empty answer")
    req_def = level_routes.CompareRequest(**data["payDefault"])
    req_row = level_routes.CompareRequest(**data["payRow"])
    check(req_def.layer == 50 and req_row.layer == 2, "frontend ComparePayloads pass server Pydantic CompareRequest")
    check("Nothing remains on the model" in data["plain"], "formatPlainEdit explains transient edit honestly")
    check(len(data["presets"]) == 3, "all 3 presets present")
    check(data["tones"] == {"lead": "good", "passed": "good", "failed": "bad", "harms": "bad", "kept": "warn", "ok": "neutral"},
          "statusTone mapping ok")
    check(
        data["batteries"]["habitatRecorded"] == {"label": "not run", "tone": "neutral"}
        and data["batteries"]["unavailGating"] == {"label": "not run", "tone": "bad"}
        and data["batteries"]["availPass"] == {"label": "passed", "tone": "good"}
        and data["batteries"]["availFail"] == {"label": "failed", "tone": "bad"},
        f"batteryOutcome never calls an unavailable battery 'passed': {data['batteries']}",
    )
    check(data["recordedCount"] == 2 and data["recordedVerdicts"] == ["works but harms", "kept the base"],
          "offline recorded 31B replay fixture has both verified 19 Nov 2026 verdicts")

    # The offline cards name lines of the committed book by number, time and hash: every figure on them is that
    # line's. Checked twice: against the script that makes them, and (without that script) against the book's lines.
    made = json.loads(json.dumps(recorded_cards.cards()))
    check(data["recorded"] == made,
          "compareRecorded.ts holds exactly the cards scripts/level_recorded_cards.py makes from the committed book "
          "(if not: .venv/bin/python -I scripts/level_recorded_cards.py --write)")
    check((recorded_cards.OUT).read_text(encoding="utf-8") == recorded_cards.render(),
          "compareRecorded.ts is the script's own text, with no edit by hand")
    lines = {int(e["seq"]): e for e in lb.entries(recorded_cards.BOOK)}
    for card in data["recorded"]:
        entry = lines.get(int(card["seq"]))
        body = entry["body"] if entry is not None else {}
        check(entry is not None and entry["hash"].startswith(card["hash"]) and len(card["hash"]) == 16
              and entry["at"] == card["at"] and entry["kind"] == card["kind"],
              f"recorded card {card['seq']} carries its book line's own hash and time "
              f"({card['hash']} at {card['at']})")
        check(body.get("note_type") == "claim_verdict" and body.get("verdict") == card["verdict"] == card["status"]
              and body.get("reasons") == card["reasons"] and body.get("claim_sentence") == card["claim_sentence"]
              and body.get("summary") == card["summary"] and body.get("batteries") == card["batteries"]
              and body.get("cheaper_knobs") == card["cheaper_knobs"]
              and body.get("edit_id") == card["edit_id"] and body.get("bars_sha256") == card["bars_sha256"],
              f"recorded card {card['seq']} gives its book line's verdict, gate notes and figures word for word")


def run_section_2(tmp: Path) -> None:
    print("Section 2: FastAPI /api/level/* routes on CPU stand-in (status, estimate, compare, 409 guard, replay)", flush=True)
    bk = tmp / "page_book.jsonl"
    prev_bk = os.environ.get(lb.BOOK_ENV)
    os.environ[lb.BOOK_ENV] = str(bk)
    try:
        use_stand_in_atlas()
        rt, vault = stand_in()
        app = bind_server(rt, vault)
        client = TestClient(app, base_url="http://127.0.0.1:8765")

        # 1. GET /api/level/status
        r_st = client.get("/api/level/status")
        check(r_st.status_code == 200, f"GET /api/level/status -> 200 (got {r_st.status_code})")
        st = r_st.json()
        check(st["runtime_ready"] is True and st["is_base"] is True and st["active_vault_edits"] == 0,
              "status reports runtime ready & clean base")
        check(st["n_layers"] == 6 and st["d_ff"] == 128, f"stand-in dims reported ({st['n_layers']}, {st['d_ff']})")
        book_ids = {b["id"] for b in st["books"]}
        check("live" in book_ids and "verify_claim" in book_ids, f"status lists live and verify_claim books: {book_ids}")
        check(len(st["starter_briefs"]) == 4, "status exposes 4 starter briefs metadata")

        # 2. POST /api/level/estimate
        req_body: Dict[str, Any] = {
            "op": "silence",
            "linear": "down_proj",
            "layer": 2,
            "neurons": [0, 1],
            "factor": 0.0,
            "dose": 1.0,
            "question": "Why does the moon show phases?",
            "answer": "Because of sunlight.",
            "controls_family": "other_sites",
            "controls_count": 3,
            "controls_seed": 0,
            "max_new": 10,
            "include_paired": True,
        }
        r_est = client.post("/api/level/estimate", json=req_body)
        check(r_est.status_code == 200, f"POST /api/level/estimate -> 200 (got {r_est.status_code})")
        est = r_est.json()
        check(est["total_forwards"] > 0 and est["total_seconds"] > 0 and est["paired"] is not None,
              f"estimate prices both written & paired: {est['total_forwards']} fwds")
        # every operation the form offers reaches the kernel as a recipe it takes (a scale's factor is "factor")
        for op, linear, factor in (("scale", "down_proj", 0.5), ("silence_rows", "up_proj", 0.0),
                                   ("scale_rows", "gate_proj", 2.0)):
            r_op = client.post("/api/level/estimate", json={**req_body, "op": op, "linear": linear, "factor": factor})
            check(r_op.status_code == 200 and r_op.json()["total_forwards"] > 0,
                  f"POST /api/level/estimate takes op {op} on {linear} (got {r_op.status_code}: {r_op.text[:160]})")
        check(not bk.exists(), "estimate writes nothing to the level book")

        # 3. POST /api/level/compare (written + paired side by side)
        r_cmp = client.post("/api/level/compare", json=req_body)
        check(r_cmp.status_code == 200, f"POST /api/level/compare -> 200 (got {r_cmp.status_code}: {r_cmp.text[:200]})")
        cmp_d = r_cmp.json()
        check(cmp_d["status"] == "ok" and cmp_d["base_after"] is True, "compare completed ok and asserted base")
        voices = cmp_d["voices"]
        check(voices is not None and isinstance(voices["base"]["text"], str) and isinstance(voices["edit"]["text"], str),
              "compare returns base and edit voices")
        check(len(voices["controls"]) == 3 and voices["untouched_rows_identical"] is True,
              "compare returns all 3 control voices with untouched rows bit-identical")
        paired = cmp_d["paired"]
        check(paired is not None and "small" in paired["classes"] and "large" in paired["classes"],
              "compare paired table returns both small and large batch classes")
        check(cmp_d["book"]["written_seq"] == 0 and cmp_d["book"]["paired_seq"] == 1,
              f"written & paired recorded at book seqs 0 & 1: {cmp_d['book']}")
        check(lb.verify(bk)["ok"], "live level book whole after compare")
        we.assert_base(lv.language_model(rt.hf_model))

        # 4. Row edit on gate_proj + same_site + sampled=True
        r_row = client.post("/api/level/compare", json={
            **req_body,
            "op": "silence_rows",
            "linear": "gate_proj",
            "controls_family": "same_site",
            "controls_count": 2,
            "sampled": True,
            "temperature": 0.8,
            "include_paired": False,
        })
        check(r_row.status_code == 200 and r_row.json()["status"] == "ok" and r_row.json()["paired"] is None,
              f"gate_proj row edit + same_site + sampled succeeded (code={r_row.status_code})")

        # 5. Vault-overlay 409 isolation guard!
        vault.execute_lql("DELETE FROM LAYER 2 WHERE slot = 0")
        st_dirty = client.get("/api/level/status").json()
        check(st_dirty["is_base"] is False and st_dirty["active_vault_edits"] == 1,
              "status detects active vault edit before compare")
        r_409 = client.post("/api/level/compare", json=req_body)
        check(r_409.status_code == 409 and "ROLLBACK" in r_409.json()["detail"],
              f"compare refuses with HTTP 409 while vault edit is active (got {r_409.status_code})")
        vault.execute_lql("ROLLBACK")
        st_clean = client.get("/api/level/status").json()
        check(st_clean["is_base"] is True and st_clean["active_vault_edits"] == 0,
              "status clean again after vault ROLLBACK")

        # 6. Model-free Replay: live book
        r_bk = client.get("/api/level/book?book=live&tail=10")
        check(r_bk.status_code == 200, "GET /api/level/book?book=live -> 200")
        bk_d = r_bk.json()
        check(bk_d["verify"]["ok"] and len(bk_d["entries"]) == 3,
              f"live book replay returns 3 verified entries (got {len(bk_d['entries'])})")
        r_sc = client.post("/api/level/compare", json={**req_body, "op": "scale", "factor": 0.5, "include_paired": False})
        check(r_sc.status_code == 200 and r_sc.json()["status"] == "ok" and r_sc.json()["edit"]["op"] == "scale",
              f"POST /api/level/compare runs a scale by 0.5 (got {r_sc.status_code}: {r_sc.text[:160]})")
        r_ent0 = client.get("/api/level/book/0?book=live")
        ent0 = r_ent0.json()
        check(r_ent0.status_code == 200 and ent0["card"]["voices"] is not None
              and ent0["card"]["edit_linears"] == ["layers.2.mlp.down_proj"]
              and "terms" in (ent0["entry"]["body"]["plan"].get("edit") or {}),
              f"replay of seq 0 returns full three-voice card, edit_linears, and rebuildable recipe ({ent0['card']['edit_linears']})")
        r_ent1 = client.get("/api/level/book/1?book=live")
        ent1 = r_ent1.json()
        check(r_ent1.status_code == 200 and ent1["card"]["paired"] is not None
              and ent1["card"]["edit_linears"] == ["layers.2.mlp.down_proj"],
              f"replay of seq 1 returns full paired metrics card with edit_linears ({ent1['card']['edit_linears']})")

        # 7. Model-free Replay: real 31B card Phase 3 verification book!
        r_vc = client.get("/api/level/book?book=verify_claim&tail=50")
        check(r_vc.status_code == 200, "GET /api/level/book?book=verify_claim -> 200")
        vc_d = r_vc.json()
        check(vc_d["verify"]["ok"] and vc_d["verify"]["entries"] == 20,
              f"verify_claim_book.jsonl verified whole with 20 entries: {vc_d['verify']}")
        r_v10 = client.get("/api/level/book/10?book=verify_claim").json()
        r_v17 = client.get("/api/level/book/17?book=verify_claim").json()
        check(r_v10["card"]["verdict"] == "works but harms"
              and r_v10["card"]["edit_linears"] == ["layers.50.mlp.down_proj"],
              f"replay of 31B seq 10 shows 'works but harms' and edit_linears ({r_v10['card']['edit_linears']})")
        check(r_v17["card"]["verdict"] == "kept the base"
              and r_v17["card"]["edit_linears"] == ["layers.50.mlp.down_proj"],
              f"replay of 31B seq 17 shows 'kept the base' and edit_linears ({r_v17['card']['edit_linears']})")

        # 7b. Model-free replay of the recorded 31B trial book: the card of its trial is the book's own note, and a
        #     dry run is shown as a dry run (the book is rewritten by scripts/verify_level_trial.py; what it holds
        #     is read here, not assumed)
        r_vt = client.get("/api/level/book?book=verify_trial&tail=50")
        check(r_vt.status_code == 200 and r_vt.json()["verify"]["ok"], "GET /api/level/book?book=verify_trial -> 200 whole")
        trial_book = ROOT_DIR / "experiments" / "results" / "levelling" / "verify_trial_book.jsonl"
        kinds = {"trial_of_record": "", "trial_dry_run": "dry run: ", "trial_rerun": "re-run: "}
        t_notes = [e for e in lb.entries(trial_book) if e["kind"] == "note" and e["body"].get("note_type") in kinds]
        check(len(t_notes) >= 1, "the recorded trial book holds a trial's verdict note")
        t_note = t_notes[0]
        r_vtn = client.get(f"/api/level/book/{t_note['seq']}?book=verify_trial").json()
        t_card, t_body = r_vtn["card"], t_note["body"]
        check(t_card["note_type"] == t_body["note_type"]
              and t_card["status"] == kinds[t_body["note_type"]] + str(t_body["verdict"])
              and t_card["verdict"] == t_body["verdict"]
              and t_card["of_record"] == (t_body["note_type"] == "trial_of_record" and bool(t_body.get("of_record")))
              and t_card["edit_linears"] == ["layers.50.mlp.down_proj"]
              and (t_card.get("folded_summary") or {}).get("ok") == (t_body.get("folded") or {}).get("ok")
              and (t_card.get("reworded_summary") or {}).get("ok") == (t_body.get("reworded") or {}).get("ok"),
              f"replay of the recorded 31B trial (line {t_note['seq']}) is its note: {t_card['status']!r}, of record "
              f"{t_card.get('of_record')}, with its linears and its folded and reworded standing")

        # 8. Path traversal & foreign-origin guards
        r_bad_bk = client.get("/api/level/book?book=../../etc/passwd")
        check(r_bad_bk.status_code == 400, "unknown book id refused with HTTP 400")
        r_foreign = client.get("/api/level/status", headers={"Origin": "https://evil.example"})
        check(r_foreign.status_code == 403, f"foreign Origin refused with HTTP 403 (got {r_foreign.status_code})")

        # 9. A live book whose last line is cut short: the whole lines before it are still replayed
        n_whole = lb.verify(bk)["entries"]
        with open(bk, "ab") as f:
            f.write(b'{"seq": 9999, "kind": "note", "body": {"cut')
        r_cut1 = client.get("/api/level/book/1?book=live")
        r_cutl = client.get(f"/api/level/book?book=live&tail={min(n_whole, 3)}")
        check(r_cut1.status_code == 200 and r_cut1.json()["card"]["edit_linears"] == ["layers.2.mlp.down_proj"]
              and r_cut1.json()["verify"]["ok"] is False,
              f"replay of a whole line of a damaged book still answers, and says the book is not whole (got {r_cut1.status_code})")
        check(r_cutl.status_code == 200 and len(r_cutl.json()["entries"]) == min(n_whole, 3),
              f"the replay index of a damaged book still lists the whole lines asked for (got {r_cutl.status_code})")
    finally:
        if prev_bk is None:
            os.environ.pop(lb.BOOK_ENV, None)
        else:
            os.environ[lb.BOOK_ENV] = prev_bk


def run_section_3() -> None:
    print("Section 3: Lexicon/Vault wiring, modern-web-guidance compliance & zero kept-out words audit", flush=True)
    vault_txt = (FRONTEND / "src" / "screens" / "Vault.tsx").read_text(encoding="utf-8")
    cmp_txt = (FRONTEND / "src" / "screens" / "vault" / "ComparePanel.tsx").read_text(encoding="utf-8")
    css_txt = (FRONTEND / "src" / "index.css").read_text(encoding="utf-8")
    lex_txt = (FRONTEND / "src" / "screens" / "Lexicon.tsx").read_text(encoding="utf-8")
    ent_txt = (FRONTEND / "src" / "screens" / "lexicon" / "entries.ts").read_text(encoding="utf-8")

    # Vault & modern-web-guidance compliance
    check('"compare"' in vault_txt and "<ComparePanel />" in vault_txt, "Vault.tsx wires compare tab and ComparePanel")
    check("vault-tablist" in vault_txt and "aria-controls" not in vault_txt,
          "Vault.tsx uses single-row .vault-tablist and omits aria-controls per tabs guide")
    check('role="region"' in cmp_txt and 'tabIndex={0}' in cmp_txt and "lvl-table-wrap" in cmp_txt,
          "ComparePanel uses accessible scrollable region wrapper per responsive-table guide")
    check("@container (width < 600px)" in css_txt and "container-type: inline-size" in css_txt,
          "index.css defines @container stacked responsive table layout")
    check('aria-invalid={showError(' in cmp_txt and 'aria-describedby={showError(' in cmp_txt and 'aria-live="polite"' in cmp_txt,
          "ComparePanel syncs aria-invalid, dynamic aria-describedby, and single polite live region")

    # Lexicon Arbiter Row 30
    check('id="control-edit"' in lex_txt and 'id: "control-edit"' in ent_txt, "Lexicon adds control-edit entry and index")
    check("compare tab" in lex_txt and "never left on the model" in lex_txt,
          "Lexicon addresses 'An edit stays until you undo it' for transient compare readings")
    check(lex_txt.count('id="control-edit"') == 1, "Lexicon has exactly one control-edit entry (no duplicates)")

    # Zero kept-out words across ALL 12 Levelling backend/CLI files AND all 3 new Phase 4 frontend files
    audit_files = [
        "backend/level_book.py", "backend/weight_edits.py", "backend/levelling.py",
        "backend/weight_reader.py", "backend/level_sandbox.py", "backend/level_session.py",
        "backend/level_mcp.py", "backend/level_briefs.py", "backend/level_claim.py",
        "backend/level_routes.py", "scripts/levelling.py", "scripts/level_recorded_cards.py",
        "frontend/src/screens/vault/compare.ts",
        "frontend/src/screens/vault/compareRecorded.ts",
        "frontend/src/screens/vault/ComparePanel.tsx",
    ]
    for rel in audit_files:
        txt = (ROOT_DIR / rel).read_text(encoding="utf-8")
        hits = [m.group(0) for m in KEPT_OUT_RE.finditer(txt)]
        check(not hits, f"{rel} has zero kept-out words (found: {hits})")


def main() -> int:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="check-level-page-") as td:
        tmp = Path(td)
        run_section_1(tmp)
        run_section_2(tmp)
        run_section_3()
    elapsed = time.perf_counter() - t0
    print(f"\nPhase 4 CPU & interface checks complete in {elapsed:.1f} s: {PASSED} passed, {FAILED} failed [ran on CPU stand-ins].")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
