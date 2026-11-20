# Levelling Scaffolding Register (`docs/SCAFFOLD.md`)

Every component in a driver harness encodes an assumption about what today's frontier models cannot reliably do on their
own (`experiments/results/weight_space_exploration/design/driver.md` §6, `design/synthesis.md` §vii). This register
separates **fixed trust boundaries** (which stay regardless of driver capability) from **non-trust-boundary
conveniences** (which `--bare` strips so Phase 6 ablations can measure whether each convenience still earns its keep).

## Fixed Trust Boundaries (Never Stripped by `--bare`)

These enforce measurement validity, containment, and cost safety — a stronger driver must never be able to bypass them:

| Boundary | Module | What it enforces |
|---|---|---|
| Off-card `bwrap` sandbox | `backend/level_sandbox.py` | No GPU, no network, no `$HOME`, allowlisted read-only `/weights` and `/atlas`, single writable `/work`, `RLIMIT_AS` + `VmRSS` + wall kill |
| Reach and backend policy | `backend/level_session.py` | Refuses `home` reach and unprobed `codex`; never mounts `/weights` in `activations`/`behaviour` or `/atlas` outside `atlas` |
| Kept-back frame guard | `backend/level_briefs.py` | Refuses practice readings and captures matching kept-back frames or retention batteries |
| Base invariant & null floors | `backend/weight_edits.py`, `backend/levelling.py`, `backend/level_knobs.py` | `assert_base` before and after every model call; untouched twin bit-identity; null-row floor `F`; two batch classes |
| Post-freeze claim gates | `backend/level_claim.py`, `backend/level_knobs.py` | Registration in chained ledger before `os.urandom(32)` draw; calibrated `BarSet`s; read-once; fixed `m`; control standing; retention batteries |
| Unattended caps & kill switch | `backend/level_drivers.py`, `backend/level_eval.py` | Hard GPU-second, practice-forward, cell, turn, token, owner-approved dollar, and wall-clock caps with process-group `SIGKILL` and `STOP` file |
| Redacted driver views | `backend/level_book.py`, `backend/level_session.py` | Strips raw kept-back text and per-item rows; blinds planted vs canary harness cards and timings |

## Non-Trust-Boundary Conveniences (Measured & Retirable via `--bare`)

Running a Phase 6 arm with `bare=True` (`--bare` on `backend/level_mcp.py`, `--bare-ablation` on
`scripts/levelling_eval.py plan`) strips the prompt-level conveniences below while keeping every trust boundary above
intact.

| Convenience | Where | Assumption about today's driver | Retirement test (`scaffolding_ablations`) |
|---|---|---|---|
| Physical-facts & workflow guidance paragraph | `backend/level_drivers.py` (`build_driver_prompt`) | Today's drivers forget that practice ranks are not verdicts, skip floor checks, or forget `W`/`A` handle names inside `/work` | Retire when a driver family achieves equal or higher lead rate and equal or lower GPU-seconds with `bare=True` over `N >= 3` sessions per brief |
| Cheaper-knob near-miss reminder | `backend/level_drivers.py` (`build_driver_prompt`, `knobs_only`) | Today's drivers try global `logit_bias` without realizing it shifts near-miss prompts equally and triggers `works but harms` | Retire when `bare=True` `knobs_only` sessions avoid naive `logit_bias` bleed at the same rate as standard sessions |
| Typed `atlas_query` convenience beside sandbox `A` | `backend/level_session.py`, `backend/level_mcp.py` | Drivers without `cell` (e.g., `tools_only`) or drivers reluctant to inspect `.npz` files underuse `/atlas` unless offered typed lookup ops | Retire for sandbox reaches if `bare=True` drivers query `A` inside `cell` with equal lead rate and fewer turns |
| Tool reply 16 KB truncation in OpenAI loop | `backend/level_drivers.py` (`run_openai_driver`) | Current chat-completion context windows bloat when large JSON tool replies stay verbatim in message history | Raise or remove when long-context drivers show no latency or accuracy regression on un-truncated replies |
| Scripted baseline recipe heuristics | `backend/level_drivers.py` (`run_scripted_weight_driver`, `run_scripted_knob_driver`) | Serves strictly as a no-driver lower bound ("did agency pay?"), not a scaffold for LLM drivers | Retained permanently as the zero-dollar baseline in every Phase 6 comparison table |

## How to Run a Scaffolding Ablation

```bash
.venv/bin/python -I scripts/levelling_eval.py plan \
  --brief pronoun_pull --sessions 3 --driver openai --model MODEL \
  --max-dollars 5.0 --bare-ablation --out study_bare.json
.venv/bin/python -I scripts/levelling_eval.py run study_bare.json --out /tmp/phase6_bare
```

The generated `report.md` prints a dedicated **Scaffolding Ablation (Standard vs `--bare`)** table beside the unpooled
per-arm results.
