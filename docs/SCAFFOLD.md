# Levelling scaffolding register (`docs/SCAFFOLD.md`)

Every part of a driver harness encodes a guess about what today's drivers cannot do on their own
(`experiments/results/weight_space_exploration/design/driver.md` §6, `design/synthesis.md` §vii). This register
separates the **trust boundaries**, which stay whatever the driver, from the **conveniences**, which a study can
take away to see whether each still earns its place.

## Trust boundaries (never taken away)

A stronger driver must not be able to get round any of these.

| Boundary | Module | What it keeps |
|---|---|---|
| Off-card sandbox | `backend/level_sandbox.py` | A cell runs under `bwrap`: no GPU, no network, no `$HOME`, read-only `/weights` and `/atlas` (the index's own files only; a symlink among them is refused), one writable `/work`, memory and wall-clock caps. Links, pipes and the like a cell leaves in `/work` are removed unread before the host looks at the folder. |
| Reach and backend | `backend/level_session.py` | `home` reach and Codex are refused. `/weights` is mounted only at the weights and atlas reaches, `/atlas` only at the atlas reach. |
| Kept-back wording | `backend/level_briefs.py`, `backend/level_session.py`, `backend/level_knobs.py` | Practice readings, captures and knob text that hold a kept-back frame's wording, or a battery's, are refused, and counted. |
| The base and the floors | `backend/weight_edits.py`, `backend/levelling.py`, `backend/level_knobs.py` | The model is asserted at its base before and after every call. The null-row floor and two batch classes apply to every reading. |
| The claim gate | `backend/level_claim.py`, `backend/level_knobs.py` | A fresh reading is registered in the chained book before its items are drawn. Bars come from a calibration in the same book on the same weights. Each edit and knob is read once, and one budget of m fresh readings is spent by claims, knob claims and trials alike. Control edits and retention batteries judge every claim; a knob with no control family cannot be claimed. |
| What a driver may run | `backend/level_drivers.py` | Claude Code and Gemini CLI get the door's tools and nothing else (the owner's ruling, 20 November 2026): no built-in tool, settings, memory, workspace hooks/skills or saved session, and an environment of named variables only (`tools.core = []`, empty read-only cwd and Policy Engine MCP allowlist for Gemini CLI). An OpenAI-compatible endpoint is local only by its parsed host, and then never through a proxy; a remote one is reached only through its profile, over https, with that profile's key variable and its prices. Redirects are refused. Codex stays refused. |
| Caps and stops | `backend/level_drivers.py`, `backend/level_mcp.py`, `backend/level_eval.py` | Session caps on GPU seconds, forwards, cells, turns, tokens, dollars and wall time; a study's total dollar and wall caps; a `STOP` file. Spend is counted as the run goes, each request at no less than its size, each reply bounded by a completion limit. When a driver is stopped, a running cell is killed and a running reading is waited for and recorded. One reply can still pass a dollar cap, as its cost is known only when it comes. |
| What a driver sees | `backend/level_book.py`, `backend/level_session.py`, `backend/level_eval.py` | Redacted book views: no kept-back text, no per-item rows. Blinded cases have neutral ids and folders, and a case whose words show anywhere the driver can see is scored "not blinded". |

## Conveniences (measured, and retirable)

`bare=True` (`--bare` on `backend/level_mcp.py`, `--bare-ablation` on `scripts/levelling_eval.py plan`) changes the
driver's page only. It takes away the first two rows below and keeps every trust boundary. The other conveniences are
listed so a later study can measure them, but `--bare` does not yet take them away.

| Convenience | Where | The guess it encodes | Taken away by `--bare` | When to retire it |
|---|---|---|---|---|
| The page's paragraph of physical facts and workflow | `backend/level_drivers.py` (`build_driver_prompt`) | Drivers forget that practice ranks are not verdicts, skip floor checks, or forget the `W` and `A` handles in a cell | yes | When a driver family leads as often, at no more GPU-seconds, with `--bare`, over at least 3 sessions a brief |
| The knobs mode's reminder that a blunt logit bias moves near-miss items too | `backend/level_drivers.py` (`build_driver_prompt`, knobs only) | Drivers try a global bias without seeing it harms near misses | yes | When `--bare` knobs sessions avoid it as often |
| The typed `atlas_query` beside the sandbox's `A` | `backend/level_session.py`, `backend/level_mcp.py` | Drivers without `cell`, or reluctant to read `.npz` files, underuse the atlas | no | When `--bare` drivers query `A` in a cell as well, in fewer turns |
| Cutting a tool's reply to 16 KB in the OpenAI-compatible loop | `backend/level_drivers.py` (`run_openai_driver`) | Long replies bloat a chat's context | no | When long-context drivers do no worse on whole replies |
| The scripted baselines | `backend/level_drivers.py` | None: they are a fixed lower bound ("did agency pay?"), each its own program | n/a | Kept in every study as the no-driver row |

## How to run a scaffolding ablation

```bash
.venv/bin/python -I scripts/levelling_eval.py plan --brief pronoun_pull --sessions 3 \
  --driver scripted --bare-ablation --out study_bare.json
.venv/bin/python -I scripts/levelling_eval.py run study_bare.json --out /path/outside/the/checkout/phase6_bare
```

An arm of an OpenAI-compatible model needs its endpoint profile and its prices (dollars a million tokens) in the plan
file's `driver_config`; the `plan` command does not take them yet. A remote endpoint without prices is refused. With
3 sessions an arm no comparison can separate two arms (the smallest two-sided Fisher p is 0.1), and the report says
so. Its `report.md` puts the `--bare` table beside the per-arm results.
