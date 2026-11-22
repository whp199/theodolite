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
| What a driver may run | `backend/level_drivers.py`, `backend/level_eval.py` | Claude Code gets the door's tools and nothing else (the owner's ruling, 20 November 2026): no built-in tool, settings, memory, CLAUDE.md or saved session, and an environment of named variables only. The Gemini CLI driver is built the same way: no built-in tool (`tools.core = []`), an empty read-only working folder, a Gemini home of its own, a Policy Engine rule allowing MCP tools alone, `--allowed-mcp-server-names levelling`, named variables only, and no usage statistics. The Antigravity CLI (agy) driver is kept on the owner's word ("Mend it and keep it", 22 November 2026), and it is not door tools only: agy 1.5.14 offers every print run three tools that no setting removes (`list_resources`, `read_resource`, `manage_task`), so the driver is refused, as the Codex driver is, until the owner sets `agy_fixed_tools_accepted: true` (in an arm's `driver_config`). That call is not yet made. Accepting them lets agy offer them, not the model use them: a session whose model calls one is stopped as a driver failure and not counted. It runs on agy 1.5.14 only, with a home, XDG folders and TMPDIR of the run's own (removed after it), an environment built from nothing, a custom agent with no default components and the door's tools marked eager, and settings that allow `mcp(levelling/*)` alone and use request-review, not Auto. After a run the tools agy offered the model, read from the run's own conversation store, must be exactly the door's and those three. An OpenAI-compatible endpoint is local only by its parsed host, and then never through a proxy; a remote one is reached only through its profile, over https, with that profile's key variable and its prices. Redirects are refused. Codex stays refused. |
| Caps and stops | `backend/level_drivers.py`, `backend/level_mcp.py`, `backend/level_eval.py` | Session caps on GPU seconds, forwards, cells, turns, tokens, dollars and wall time; a study's total dollar and wall caps; a `STOP` file. Spend is counted as the run goes, as far as each driver reports it. For an OpenAI-compatible endpoint, each request is counted at no less than its size and each reply is bounded by a completion limit; Claude Code, the Gemini CLI and agy are given no completion limit. The Gemini CLI reports its usage only in its last line, so its token and dollar caps are checked after the run and cannot stop it (the session's row names any it passed), and its turns are capped by the CLI itself. A Gemini CLI run stopped before that line (at the wall cap or by STOP, say) has no usage to read: no cap is checked, its notes and its row say its usage is not known ("usage not known"), and a study charges it the dollar cap it was handed against the study's total, which does not bound what it spent (nothing is charged when it was handed no cap, or at prices of 0 for both). The Gemini CLI driver keeps its own notes whole, cutting the model's text first. agy's token and dollar caps count a reply once agy prints its usage, when the reply is done, so each reply has been spent before they can stop the run. agy's turn cap stops a run when the model begins a reply past it, after the tools the last allowed reply asked for have run; that cut reply is counted at a floor when agy did not print its usage, and the floor can be below its size (1,434 tokens against about 1,900 in one live run). The Gemini CLI and agy report no cost, so under a dollar cap both prices must be given (0 for both on a sign-in quota, where the cap then stops nothing). When a driver is stopped, a running cell is killed and a running reading is waited for and recorded; whatever a CLI leaves in its process group is killed when it ends. One reply can still pass a dollar cap, as its cost is known only when it comes. |
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
file's `driver_config`; the `plan` command does not take them yet. A remote endpoint without prices is refused. A
Gemini CLI or Antigravity CLI arm under a dollar cap needs both prices in the plan file's `driver_budget`
(`input_usd_per_mtok`, `output_usd_per_mtok`): the `plan` command has no price flags, so such a plan made there with a
dollar cap is refused, and both drivers need one. An Antigravity CLI arm also needs `agy_fixed_tools_accepted: true`
in its `driver_config`, which is the owner's call. With 3 sessions an arm no comparison can separate two arms (the
smallest two-sided Fisher p is 0.1), and the report says so. Its `report.md` puts the `--bare` table beside the
per-arm results.
