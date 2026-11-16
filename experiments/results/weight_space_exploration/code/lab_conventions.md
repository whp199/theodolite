# The lab daemon, experiment conventions and job handling

Reader's report for the weight-space harness design, 14 November 2026. Read-only: nothing in the repository was
changed, no server was started, the model was not loaded. Paths are relative to `(the repository root)` unless
they start with `/`. "Code" means I read the lines cited; "document" means a docstring, README or memory note says so
and I did not see code that enforces it.

## Short answers

1. **The lab daemon** is a small FastAPI process that loads the 31B once and runs submitted Python against it, one job
   at a time, under a lock. It ran the original study (314 jobs on 16 October 2026). It is still maintained and
   checked, but no job is recorded after 17 October 2026, and nothing in the backend or the pages talks to it. It is
   the closest thing in the repository to "an agent submits code, gets a result", and is a usable seed, but it has no
   cancellation, no model-state reset between jobs, no time limit, and it defaults to the slow unpacker.
2. **Registration** has a concrete, repeated shape in recent work: a `prereg/` folder holding `RULES.md`, the exact
   run and read scripts, the neuron lists, `repo_commit`, `prereg_at` and `SHA256SUMS`, committed before the result;
   the run script refuses if the hashes, the commit, a dirty tree or a live server are found. All nine registration
   hash files I checked still verify. Pilots and design notes are not registered, and say so.
3. **Tests** are stand-alone scripts, not pytest: 27 `scripts/check_*.py` (CPU, on a tiny random Gemma 4) and 8
   `scripts/verify_*.py` (the real model or a running server), each run as `.venv/bin/python -I scripts/<name>.py`,
   exit 0 when every check holds. There is no test folder, no runner and no CI.
4. **GPU sharing is by exclusion, not by a scheduler.** One process holds the model. Inside the server, one
   `asyncio.Lock` serialises every model call. Batch jobs are separate processes that refuse to start if anything
   answers on port 8765; the server is simply down while they run. The `systemd-run` scope is a practice recorded in
   the owner's notes, not code.
5. **The volunteer kit** is a pinned, self-checking job package (a tag, a constraints file, hashed inputs, a canary)
   with an intake that checks, re-runs a random sample at home, judges by rules fixed beforehand, and writes a
   provenance file. Almost all of it is reusable as "an agent ran this; here is the checked result".
6. **Field proposals**: the scientist can only propose a name or a note; a person keeps or discards it. A pattern
   test's result is stored as a measurement with the exact pattern text it tested, six kept per face.
7. **Layout**: large and regenerable things go under `atlas_data/` (67 GB, git-ignored); the committed record of an
   experiment is a small folder under `experiments/results/<name>/`; personal state lives outside the checkout under
   XDG paths.

---

## 1. The lab daemon (`experiments/lab/`)

Three files: `labd.py` (260 lines), `labclient.py` (64 lines), `labkit.py` (523 lines).

### What the code does

- **Loads one runtime at start** through `make_runtime(load_jlens=True).load()` (`experiments/lab/labd.py:161`), so
  it honours `THEODOLITE_GGUF` and `THEODOLITE_DEVICE` like the server. It builds a `labkit.Lab` over it
  (`labd.py:168`).
- **Unpacker**: it installs `$THEODOLITE_UNPACKER`, and when that is unset it installs `"reference"`
  (`labd.py:157-163`). The server, by contrast, picks `compiled` by default on a CUDA card
  (`backend/server.py:128-133`). So a labd job runs at the slow step unless the variable is set.
- **`POST /exec`** takes `{code, label, out}` (`labd.py:125-128`). Under one `asyncio.Lock` (`labd.py:120`, `:201`)
  it compiles the code with top-level `await` allowed (`:210`), evaluates it inside `torch.no_grad()` with stdout
  captured (`:211-215`), in a fresh namespace holding `rt`, `lab`, `torch`, `np`, `F`, `STORE`, `labkit`, `asyncio`
  (`:205-206`). The job returns data by assigning `result`. `STORE` is a dict that persists across jobs (`:122`).
- **After each job** it removes the hooks that were registered through `Lab.add_resid_hook`
  (`labd.py:219`, `labkit.py:267-270`), clears the namespace, collects garbage and empties the CUDA cache
  (`:221-223`). The reply is `{ok, stdout (last 200,000 characters), error (traceback), result, elapsed_s, model,
  weights}` (`:224-226`); `weights` is the runtime's weights card, so a result names the weights it was read from.
- **Survives a client timeout**: if `out` is given the daemon itself writes the whole reply to that path (`:228-233`).
  `labclient.py` passes `<out>.full.json` there and writes the bare `result` to `<out>` (`labclient.py:39-40`,
  `:59-61`). Its default client timeout is 3,600 s (`labclient.py:29`).
- **Log**: one JSON line per job (time, label, ok, elapsed, last 2,000 characters of the error, model) appended to
  `experiments/results/labd_exec_log.local.jsonl`, which is git-ignored (`labd.py:72`, `:234-236`).
- **`GET /health`** gives `ready, busy, waiting, running (the label), vram_alloc_gb, model, device, weights,
  unpacker` (`labd.py:178-187`). **`POST /reload`** re-imports `labkit` under the lock (`:190-195`).
- **Who may ask**: every route refuses (403) any Host that is not `127.0.0.1`, `localhost` or `::1`, and any request
  carrying `Origin` or `Sec-Fetch-Site` (`labd.py:98-106`). `/exec` and `/reload` also need a per-launch random token
  (`:109-115`), written 0600 to `$LABD_TOKEN_FILE` or `${XDG_RUNTIME_DIR:-~/.cache}/theodolite/labd-<port>.token`
  (`:78-95`). The port is bound and listening before the token is written and before the model loads (`:240-252`).
- **Port**: 8765 unless `$LABD_PORT` (`labd.py:73`). That is the server's port too, on purpose: "run one or the
  other" (`labd.py:22-24`).

### What `labkit.Lab` gives a job

Tokenising and chat prompts (`labkit.py:61-128`); `capture` of residual points through interp-engine (`:131-137`);
a fast batched `forward_resid` by forward hooks (`:139-174`); lens readouts (`:177-214`); residual hooks, direction
ablation and steering (`:267-311`); `generate`, `generate_batch`, `next_logits`, `next_logits_batch`
(`:315-344`, `:369-405`); the corrected layer decomposition with Gemma 4's `layer_scalar` (`:357-366`); and **true
gradients through the 4-bit weights**: an autograd function that re-unpacks each W4A16 linear in backward so the
62 GB of bf16 weights are never stored (`:417-432`), a GGUF twin that also hears vault overlays (`:435-453`), the
context manager that swaps every packed linear to it (`:456-482`), and `same_pos_vjp` (`:485-519`). The study's
report gives about 2 s per VJP and a 23 GB peak (`experiments/REPORT.md:90-93`, measured for the study of
16 October 2026).

Much of `labkit` is the gender study's own (token sets and `gender_scores`, `labkit.py:27-36`, `:217-264`). The
general parts are the ones listed above.

### Who talks to it, and is it in use

- **Callers**: `labclient.py` from a shell, and `scripts/check_local_only.py` section 8, which drives `labd.app`
  in-process to check the guard, the token and a job (`scripts/check_local_only.py:341-400`). Nothing in `backend/`
  or `frontend/src` calls it: every mention there is a comment (grep of `labd|labclient|labkit` over `backend/*.py`).
  It is not in `.claude/launch.json`.
- **Use**: the frozen study log `experiments/results/labd_exec_log_2026-10-16.jsonl` holds 314 jobs, 282 ok, median
  71.3 s, longest 456.9 s, 27,453 s of job time over a span of 41,837 s (my count from the file). The live local log
  holds 8 more jobs, the last at 2026-10-17T04:59Z. No worktree under `.claude/worktrees/` has a later log, and no
  token file exists under `$XDG_RUNTIME_DIR/theodolite` now, so it is not running.
- **Maintenance since**: git history of `experiments/lab` shows the Host guard and token (28 October, `7d0ddd7`,
  `be4fbc8`), the runtime factory (28 October, `47018e1`), the GGUF gradient twin (28 October, `0fe52d0`), and a
  touch on 2 November (`601b77d`). So it is kept correct, not used.
- **What replaced it**: the registered R2 to R4 scripts build the runtime themselves (ARCHITECTURE.md:1167-1170);
  the census and atlas builds are their own processes (`scripts/census_run.py`, `scripts/atlas_build.py`); the
  pattern-test pilot of 13 to 14 November went through the running server's HTTP routes
  (`experiments/results/pattern_tests_pilot/pilot.py:3-4`, `:53`).

### Could it seed the harness's job runner

What it already has that a harness needs: one resident model (no reload per hypothesis), strict serialisation, a
result file that outlives the client, a per-job log line, the weights card on every reply, a local-only and token
guard that is itself under test, top-level `await`, and a helper library with gradients through the packed weights.

What it lacks, by reading the code:

- **No cancellation and no time limit.** The routes are `/exec`, `/health`, `/reload` only
  (`scripts/check_local_only.py:369` asserts exactly these). A runaway job is stopped by killing the daemon.
- **`/health` cannot answer during a synchronous job.** The code is evaluated on the event loop itself
  (`labd.py:213-215`), not in a worker thread, so the loop is blocked between a job's `await` points. The server and
  the atlas routes do the opposite (`asyncio.to_thread` under the lock, `backend/atlas.py:148-153`,
  `backend/server.py:1452-1457`). This is from the code; I did not run it.
- **No reset of model state between jobs.** Only hooks registered through `Lab` are cleared (`labd.py:219`). A job
  that edits a weight tensor, registers its own hook on a module, or leaves something in `STORE` changes what the
  next job reads. There is no weight fingerprint before and after a job, and the vault is not loaded in labd at all
  (`STATE` holds `rt`, `weights`, `lab`; `labd.py:166-168`).
- **`torch.no_grad()` is forced** around the job (`labd.py:212`); gradient work must open `torch.enable_grad()`
  itself, as `same_pos_vjp` does (`labkit.py:509`).
- **Arbitrary code execution as the user**, by design (`README.md:487-490`). The namespace is not a sandbox.
- **The job record is thin**: label, ok, seconds, error tail, model id. No hash of the submitted code, no git
  commit, no registration reference.
- **Slow by default**: the reference unpacker (0.71 s a decode step against 0.09 to 0.13 s for `compiled`;
  `scripts/census_run.py:88-96`).

---

## 2. Registration, results and errata

There are two generations of the pattern. Both are followed in code, not only in prose.

### 2a. The current pattern: a hashed `prereg/` folder (29 October to 14 November 2026)

Seen in `experiments/results/census_break_rel_trial`, `census_neighbour_trials`, `census_ceil_trials`,
`census_soft_trials` (two registrations), `census_soft_walk`, `census_match_trials`, `census_match_walk`, and
`census_other_card_bars/kit4`. The newest, the bars registration of 14 November, is the cleanest example.

**What a registration folder holds** (`experiments/results/census_other_card_bars/kit4/prereg/`):

| file | what it is |
|---|---|
| `RULES.md` | Why; what is run; the groups; the decision rules numbered R1 to R6; what happens if the measurement cannot be read; what is not changed; **what had been seen when this was fixed**; **what this leaves untested**. |
| `measure.sh` (elsewhere `night.sh`, `read.sh`, `walk.sh`) | The exact command. It refuses unless the folder's hashes verify, the script it runs is byte-equal to the registered copy, and nothing answers on :8765 (`measure.sh:9-11`). |
| the registered code | A copy of the script that will run (`census_bars.py` here; `ceil_read.py`, `trial_read.py`, `readlib.py` for the trials). The reader is registered, not only the runner. |
| the inputs | For trials, the neuron lists `neurons_L<NN>.txt` and the script that drew them (`draw_neurons.py`). |
| `repo_commit` | The commit the run must be taken from. |
| `prereg_at` | UTC time of registration; a second line is appended for an amendment. |
| `SHA256SUMS` | sha256 of every file above, `repo_commit` included. |
| `AMENDMENTS.txt` | Only when something is added or changed after hashing: time, what, why (`census_ceil_trials/prereg/AMENDMENTS.txt`). An addition gets its own `SHA256SUMS.<name>` and `prereg_at.<name>` so the first hashes stay untouched. |

**What the run script enforces** (`experiments/results/census_match_walk/prereg/night.sh:16-21`):

1. the clean checkout `R` is at `repo_commit` (exit 2 otherwise);
2. `R` has no uncommitted tracked changes;
3. `sha256sum -c SHA256SUMS` passes, else "record why in AMENDMENTS.txt";
4. nothing answers on 127.0.0.1:8765 (exit 5: "one GPU process at a time").

It then writes run files, `.t0` and `.t1` time stamps and a log under `atlas_data/census/trials/<name>/`, is
resumable row for row, and "reads no verdict, commits nothing" (`night.sh:5-6`). The reader (`read.sh`) runs on the
CPU (`CUDA_VISIBLE_DEVICES=` and `nice -n 15`), checks the hashes again, **refuses to read a layer twice**
(`read.sh:19`), and writes `L<NN>_verdict.json` and `.txt`. A chain script (`walk.sh`) takes the next layer only on
a pass and stops at the first failure (`walk.sh:13-27`).

**The run is taken from a separate clean worktree** pinned at the registered commit
(`.claude/worktrees/match-trials` at `5fa7e33`, `ceil-trials`, `soft-trials`, `neighbour-trials`, `steer-build`,
`kit-4`; `git worktree list`), with the main checkout's venv and `atlas_data` (`night.sh:9-13`).

**How the record reaches git.** The working copy of `prereg/` lives beside the runs in git-ignored
`atlas_data/census/trials/<name>/prereg`; the same files are committed under `experiments/results/<name>/prereg`.
A small `collect.py` copies the verdicts, slimmed of `named_words`, and each run's `meta.json` into
`experiments/results/<name>/` and writes `summary.json` (`experiments/results/census_match_walk/collect.py`).

**Order of commits, checked in `git log`:**

- bars: registered `84a7b3a` at 16:50:38Z, amended `25ce7e0` at 17:15:46Z, result `18b8410` at 17:41:23Z. The two
  lines of `prereg_at` equal the first two commit times to the second.
- match walk: registered `39915dd` (07:25:42Z, 13 November), then one result commit per layer (`31affca`, `a8f06c3`,
  `0e408cc`, `1648b9b`, `d114eca`, `067d7fc`).
- ceil, soft, neighbour and the first break_rel trial each have a "registered before..." commit ahead of their
  result commits.

**Do the hashes still hold?** I ran `sha256sum -c` (read-only) over all nine committed registration hash files:
every one verifies, and `kit4/prereg/census_bars.py` is byte-equal to `scripts/census_bars.py` today.

**Features of the rules documents worth copying into an agent ledger:**

- The owner's go-ahead is quoted with its date ("the owner's word: 'Bars first'", `RULES.md:3`; "Walk first",
  `census_match_walk/prereg/RULES.md:3`).
- Predictions are written before the reading, with rough odds (`census_match_walk/prereg/RULES.md`, "What is
  expected, said beforehand"; `experiments/results/census_scaled_design/PREDICTION.md`, committed at 20:14Z on 12 November,
  about 16 minutes before layer 52's verdict was read, and then given a "What happened" section).
- Each rule says what it decides and what remains the owner's choice (`DOSES_CALIBRATED` is widened "only on its
  passed trial and the owner's word").
- A fail by 0.001 is a fail: "The rule was fixed beforehand, and it counts this a fail"
  (`experiments/results/census_scaled_design/DESIGN.md`, layer 38 section).
- Follow-up looks are fenced: "looked at after the verdict, so it gates nothing and is no second reading of it".
- A re-test after a failed design must ask neurons the failed trial did not ask.
- An amendment made before any data is said to be one, with the reason and the only lines changed
  (`kit4/prereg/RULES.md:6-12`); a fault found in the script before the run is written into the registration
  (`RULES.md:26-32`).
- The first trial also kept a multi-agent review of its verdict beside it
  (`experiments/results/census_break_rel_trial/review/`).

### 2b. The older pattern: one long registration document (R2 to R4, 21 to 28 October 2026)

`experiments/results/R2_PREREG.md` (31 KB), `R3_PREREG.md` (95 KB), `R4_PREREG.md` (61 KB). R3's section 0 gives a
sha256 table of every registered file and the rule "any change made after data exists is a deviation", numbered
(A12 to A15 amendments, D30 to D36 deviations), each reported with the result under the code as registered beside
the rerun (`R3_PREREG.md:1-60`). The enforcement is a reusable Python frame in `experiments/scripts/R3_gpu.py`:

- `require_committed` refuses a GPU job while any registered file is untracked or differs from HEAD, or while
  `backend/` differs from the pinned runtime commit; `--allow-uncommitted` overrides it as a recorded deviation
  (`R3_gpu.py:344-399`).
- `gpu_job(name, argv, ...)` is a context manager that does the git check, caps the process's CUDA memory, records
  VRAM before and after, and appends `started`, then `finished` or `failed` with the error, to
  `experiments/results/R3_gpu_runs.jsonl` (`R3_gpu.py:402-427`). `set_job_cap` warns when less than the cap is free:
  "is the server stopped?" (`R3_gpu.py:175-186`).
- `require_pilot_proceed` refuses a fit job unless a pilot said "proceed" against a GPU-hour budget the owner
  approved (`R3_gpu.py:333-341`). `scripts/gpu_log.py` generalises this to rented cards (approve, start, stop, sum;
  it "approves nothing of itself"); its log `experiments/results/gpu_runs.jsonl` does not exist yet, as no rented
  card has run.

### 2c. Results and errata

- **Results**: per-experiment `*_REPORT.md` and `*_NOTES.md` for the study; since late October a folder per
  experiment with the scripts, the JSON outputs and often a `DESIGN.md` that grows dated "What the trial said"
  sections. Verdicts are machine-readable (`L<NN>_verdict.json`, `summary.json`) with a text twin.
- **Errata**: the study's report is left as written and each qualified claim is tagged "[see erratum E*n*]"; the 51
  errata sit in a dated section at the end with the rule "Qualified / Caution; nothing is retracted outright"
  (`experiments/REPORT.md:412-440`). In results files a correction is made in place with a note (for example
  `experiments/results/census_other_card_bars/bars.txt` line 1: "recorded as cpu: corrected from the canaries").
- **Failed attempts stay**: a `null` `.json` or `ok: false` marks a failed labd job and the report names the rerun
  (`README.md:742-744`); R3 appends every run, failed ones included.

### 2d. How strictly it is followed in recent work

- **Followed** for anything that gates a decision: every census trial, every walk, and the bars. I found no gate in
  the sampled folders that was read without a prior registration commit.
- **Not registered, and labelled so**: `experiments/results/pattern_tests_pilot/` (a "pilot", no `prereg/`; it
  records its seed, layers and judge in `pilot.py:46-57` and `started.json`), `experiments/results/atlas_steer_dose/`
  (a dose pilot and a build), and the design notes in `census_scaled_design/`.
- **Rough edges**: the first registration's `tonight.sh` copies its files from a session scratchpad path under
  `(a session temp path)` (`census_break_rel_trial/prereg/tonight.sh:8`), and the registered shell scripts hard-code
  `(the repository root)`. The registrations are honest records, not portable packages.

### 2e. The pattern, stated for an agent-run ledger

1. Write `RULES.md`: question, what is run, the reader, the gates and bars, what each outcome decides, what was
   already seen, what is left untested, the prediction.
2. Put the exact runner, the exact reader and the exact inputs beside it; write `repo_commit`; hash everything into
   `SHA256SUMS`; stamp `prereg_at`; commit with a message that says "registered before ...".
3. The runner checks the hashes, the commit, a clean tree and a free card before it loads anything; it writes only
   raw rows, a meta and time stamps; it never reads a verdict.
4. The reader runs on the CPU from the registered bytes, once, and writes a JSON verdict and a text twin.
5. Copy the small verdict files into the committed results folder; commit one result per commit.
6. Anything changed after hashing goes in `AMENDMENTS.txt` with time and reason before any reading; anything changed
   after data exists is a numbered deviation reported beside the registered result.
7. Later looks are "reported, gating nothing". A human word is quoted where a decision is the owner's.

---

## 3. Testing conventions

- **`scripts/check_*.py`**: 27 files, 22,466 lines. Each hides the GPU and goes offline before importing torch
  (`os.environ["CUDA_VISIBLE_DEVICES"] = ""`, `HF_HUB_OFFLINE = "1"`; `scripts/check_census.py:81-83`), builds or
  imports the **stand-in**, a tiny random `Gemma4ForCausalLM` with 6 layers, hidden size 64, feed-forward width 128
  and a sliding window of 8 (`scripts/atlas_stand_in.py:44-47`, `:102-141`), and compares the code under test with
  brute force or hand-worked numbers. The shared idiom is a module-level `failures` list, `check(ok, what)` and
  `section(title, before)`, a final "`N checks passed, M failed`", and `return 1 if failures else 0`
  (`scripts/check_census.py:108-123`, `:1945-1946`). The docstring is a numbered list of what is checked and ends
  with the run line.
- **`scripts/verify_*.py`**: 8 files, 5,129 lines. They need the real model or a live server:
  `verify_optics.py`, `verify_causal_check.py` (holds the tool to the study's numbers within stated tolerances and
  refuses if :8765 answers, `scripts/verify_causal_check.py:98`, `:310`), `verify_atlas_writer.py` (the writer's
  floor; has `--stand-in`), `verify_control_vector.py`, `verify_e2e_ui_and_mcp.py` (server, tools and
  headless-Chrome pages; it first runs several CPU checks in their own processes), `verify_atlas_drive.py` and
  `verify_steer_sampling.py` (drive a server at `--base`), `verify_scientist_cli.py`.
- **How they are run**: `.venv/bin/python -I scripts/<name>.py`; exit 0 means every check held. The `-I` (isolated
  mode) is used everywhere, including for registered runs.
- **The naming is a rule of thumb, not strict**: `scripts/check_sweep_refactor.py` needs the GPU and the model
  (`:1-4`); `scripts/check_gguf_fast.py` takes `--cuda`.
- **pytest**: `pytest>=9.0.0` is a dependency (`pyproject.toml:21`), but the repository's own tree has no
  `test_*.py`, no `tests/`, no `conftest.py`, and there is no `.github/` or other CI configuration. I found no
  script that runs every check.
- **Dry runs are part of the convention**: every GPU script of the census and the pods takes `--stand-in` or
  `--dry-run` and the whole flow is exercised on the CPU by `scripts/check_volunteer_kit.py` and
  `scripts/check_pod_scripts.py`. Stand-in output lands under `atlas_data/stand-in/`, never among the model's
  (`scripts/atlas_stand_in.py:148-166`).
- **Self-tests at load**: the fast unpackers test every one of the 410 linears against the reference on the card
  before anything is read and refuse on any difference (meta of a run: `"self_test": {"linears": 410, "same_bits":
  true, ... "seconds": 1.83}`, `experiments/results/census_match_walk/L36-match_ceil-trial.meta.json`).

A new harness would be expected to ship a `check_<thing>.py` that runs end to end on the stand-in, and a
`verify_<thing>.py` for the card.

---

## 4. GPU sharing

### Inside the server

- One module-level `asyncio.Lock`, `_GPU_LOCK` (`backend/server.py:99`). Every route that runs the model takes it:
  sweeps (`:1019`), vault statements (`:1463`, `:1587`), the scientist routes (`:1474`, `:1481`, `:1503`, `:1536`),
  and all atlas routes through `atlas_routes.bind(get_app_state, _GPU_LOCK)` (`:1660`; `backend/atlas.py:148-153`).
  The scientist's tools reach the model through the same routes in-process (`backend/scientist_chat.py:868-870`),
  and the standalone MCP module's own lock becomes a no-op when the server binds it
  (`backend/mcp_scientist.py:22-24`, `:85-98`).
- GPU work runs in a worker thread so the event loop keeps answering (`backend/atlas.py:6-8`;
  `backend/server.py:1452-1457`, where a heard vault edit is recorded as 12.6 s).
- **Cancellation is cooperative and only for writing**: a sweep's `cancel_sweep` is read by a receiver task and
  honoured at the next word-piece or pair (`backend/server.py:860-904`, `:1005-1017`, `:1345-1371`); a push chat sets
  a `threading.Event` when the page disconnects and keeps the lock until the writer thread has stopped
  (`backend/atlas.py:2142`, `:2168-2193`). Other routes cannot be cancelled.
- **No queue visibility**: `/api/health` reports the model, VRAM, the unpacker and cards, but not whether the lock
  is held or who waits (`backend/server.py:1386-1441`). labd's `/health` does report `busy`, `waiting`, `running`.
- The server loads with `load_jlens=False` and builds the vault with a field-notes reader
  (`backend/server.py:122`, `:136`).

### Between processes

- The rule is "one GPU process at a time: the lab daemon, the instrument server and a standalone `mcp_scientist`
  each load the 31B" (`README.md:484-485`; `scripts/census_run.py:115-116`).
- **How a job avoids the server**: each registered wrapper probes the port with
  `(exec 3<>/dev/tcp/127.0.0.1/8765)` and exits 5 if it answers (`census_match_walk/prereg/night.sh:19-21`,
  `kit4/prereg/measure.sh:11`, `experiments/results/atlas_steer_dose/run_on_card.sh:17`). Two scripts check free
  card memory instead (`scripts/census_bars.py:278-283`, `scripts/census_ties.py:107-109`). `scripts/census_run.py`
  itself has neither guard; it just loads.
- **What the server does while a job holds the card**: nothing; it is stopped. The pages fall back to the study's
  recorded sighting when no server answers (`README.md:758`), and `.claude/launch.json` offers a CPU server and a
  stand-in server on port 8766 that do not need the card. The owner's notes say the server reloads in about 10 s
  and that stopping and restarting it for GPU checks is allowed (memory note `server-restart-ok.md`, 18 to 19
  October 2026); labd's own log says "ready in 13.3s; VRAM alloc 21.24 GB" with both lenses
  (`experiments/results/labd.log`, 16 October 2026).
- **systemd scopes**: in the repository the only trace is `run_on_card.sh:15`, which waits while
  `census-chain.scope` is active. The practice itself is in the owner's notes: a job started from an agent shell sits
  in the app's systemd scope (`KillMode=control-group`) and dies with it even under `setsid nohup`, so long jobs are
  started with `systemd-run --user --scope --unit census-LNN`, and chains queue by waiting on a scope name
  (`~/.claude/projects/-home-eric-bin-ai-theodolite2/memory/census-volunteers.md:67-70`). No code creates a
  scope.
- **On a pod or a volunteer's machine** there is a real lock: `flock -n` on `<root>/logs/gpu.lock`, taken by a census
  shard, the pilot and the setup self-test (`scripts/pod/common.sh:104-112`; `scripts/pod/census_shard.sh:120`,
  `:145-147`). `census_shard.sh` also shows the full detached-job shape: `setsid nohup`, a written `job.sh`,
  `.started` and `.done|.failed|.timeout` sentinels, `timeout --signal=INT` at projected hours times 1.25, and a
  scrubbed environment with every token-like variable removed (`scripts/pod/common.sh:20-35`). The home card does
  not use this lock for the server.
- **Stopping a job cleanly**: `census_run.py` turns SIGINT into "stop after this step"; whole batches are kept and
  the same command continues the run row for row (`scripts/census_run.py:311-317`).
- **Run CPU readers at `nice -n 15`** beside a GPU job: a whole-layer reading is about 20 minutes and 6 GB
  (`experiments/results/census_scaled_design/read_whole.sh:5`).

---

## 5. The volunteer kit as a pattern

### The package (`volunteers/kit.env`, `scripts/volunteer/census.sh`, `scripts/pod/`)

- **One file fixes everything**: kit number 4, the git tag `census-kit-4`, the layers, the design per layer,
  `KIT_CONTROLS=2048`, `KIT_SEED=7`, `KIT_BATCH=64`, `KIT_MAX_NEW=48`, `KIT_UNPACKER=compiled`, a constraints file of
  exact library versions, and the sha256 and URL of the one shared input (`volunteers/kit.env:16-40`). "A change of
  any value here is a new kit" (`:5-6`).
- **Measured hardware floor**: 24,000 MiB total and 23,900 MiB free; the model rests at 17.97 GiB and the worst batch
  of 64 reaches 22.1 GiB (`volunteers/kit.env:46-53`, measured 14 November 2026).
- **One command with fixed steps**: `check`, `reserve`, `setup`, `canary`, `run`, `status`, `pack`, `release`, and
  `--dry-run` for the CPU stand-in (`scripts/volunteer/census.sh:5-13`). The volunteer chooses nothing.
- **The run refuses other code**: the checkout must be at the kit's tag with nothing tracked changed
  (`census.sh:126-129`, `:316-318`).
- **A canary first**: plan batch 0 written alone, compared with the home card's copy
  (`volunteers/home_canaries/L<NN>-<design>.canary.jsonl`); identical means "home bits", otherwise "a card class of
  its own" (`scripts/volunteer/volunteer_tool.py:64-96`).
- **Every run carries a meta** naming the model, weights card (repo, snapshot, sha256 of the weight files), settings,
  job-list digest, batch plan, unpacker self-test and a runtime record (card, torch, CUDA, transformers,
  interp-engine, compressed-tensors, git commit, `git_clean`, the stats index's sha256)
  (`backend/census.py:1809-1846`). `settings_differ` over `RESUME_KEYS` decides whether a run may be continued
  (`backend/census.py:1906-1927`).
- **The bundle**: answers (zstd), meta, canary, writer's floor and a `manifest.json` with each file's size and
  sha256 and the claim's secret (`volunteer_tool.py:237-253`).
- **Identity without accounts**: a claim is a random secret kept locally; its code is the first 8 hex of its sha256
  (`backend/census_volunteer.py:172-174`). A text board with states and time-outs tracks claims
  (`scripts/census_board.py:60-65`, `:257`).

### The intake (`scripts/census_intake.py`)

Four steps, each writing a JSON file in a work folder `atlas_data/census/volunteer/L<layer>-<claim>/`:

1. **`check`** (CPU): the bundle is treated as hostile: only names a pack writes, regular files, size caps
   (4 GB for the run, 64 MB otherwise), nothing in it is run (`:57-61`, `:171-189`). Then manifest hashes, the claim
   secret, the meta against the home canary's meta (`census_volunteer.meta_problems`,
   `backend/census_volunteer.py:240-261`), the commit against the kit's tag, `git_clean`, every planned answer present
   exactly once (`scan_run`, `:264-296`), the canary against the run's own first batch and against home. **The spot
   chunks are drawn only now**, from `os.urandom`, so the sender cannot know them (`census_intake.py:330-334`).
   Writes `check.json`.
2. **`spot`** (GPU, server stopped): 5 of the run's chunks written again at home, "a minute or so a chunk on the
   5090, the model loaded for each" (`:21-22`, `:57`, `:380-398`).
3. **`judge`**: bit for bit if the canary was home bits; otherwise by bars measured for that design and layer
   (`OTHER_CLASS_BARS`, `backend/census_volunteer.py:78-86`). **Where no bars have been measured it refuses to judge**
   (`census_intake.py:419-422`). Writes `verdict.json` with the comparison, the rules used and the reasons.
4. **`accept`**: copies the run in (never over an existing one), writes `<stem>.provenance.json` holding who, the
   card, the bits, the bundle's sha256, the whole `check` and `verdict`, ranks it and marks the board
   (`:442-474`).

The acceptance bars were themselves set by a registered measurement of the home card's own arithmetic noise and of
deliberate near misses, with the rule "bar = geometric mean of the floor and the smallest near miss at least four
times the floor" (`kit4/prereg/RULES.md:53-57`); "no bar is set by looking at a volunteer's answers" (`:87`).

### What carries over to "an agent ran this experiment; here is the checked result"

- **A job spec file** that pins commit, settings, seeds, input hashes and library versions, where any change is a
  new version.
- **A canary**: a short fixed slice run first, compared with a trusted copy, to classify the runtime before any long
  work.
- **A run meta and a runtime record** written by the job itself (`census.runtime_record`, the weights card).
- **A manifest with hashes**, and a work folder holding `check.json`, `verdict.json` and `provenance.json`.
- **Verification by re-running a random sample drawn after submission**, cheap relative to the job (5 chunks of 737
  for a whole layer).
- **Tolerances measured beforehand** from the instrument's own noise and from planted errors, and a refusal to judge
  where none were measured.
- **Hostile-input handling** of whatever comes back.
- **Planted-failure tests of the judge itself**: `scripts/check_volunteer_kit.py` section 5 shows a tampered run, a
  made-up run and a wrong-secret bundle each being refused.

What is specific to the census and would not carry: the row format, the batch plan, the notion of "bits" from greedy
answers. For a weight edit the analogue of the canary and the spot chunk would have to be defined.

---

## 6. Field proposals and tests (`backend/field_notes.py`)

Four JSON files outside every checkout, under `${XDG_DATA_HOME:-~/.local/share}/theodolite/` unless an environment
variable moves them (`field_notes.py:104-134`): notes, the field book, proposals, tests.

- **What an agent can propose**: only a name (at most 120 characters, one line) or a note (at most 4,000 characters)
  for a neuron or a head (`check_note`, `:240-264`; limits `:85-86`). The scientist's `set_name` and `write_note`
  tools post to `POST /api/atlas/proposals` and return "a proposal, waiting on the atlas page for the person to keep
  or discard" (`backend/scientist_tools.py:685-702`). A later proposal replaces the earlier one's name or note
  (`propose`, `:519-545`). At most 5,000 proposals wait (`:89`, `:535-536`).
- **Nothing reads a proposal as a name**: lists, search, tooltips, the map, later chats and the vault's DESCRIBE POLE
  read only kept notes (`:49-52`, `:613-625`).
- **Who decides**: the person, through `POST /api/atlas/proposals/keep` or `/discard`, optionally for the name or the
  note alone (`backend/atlas.py:799-825`; `decide`, `field_notes.py:548-579`). Keeping writes the entry with
  `name_by` or `note_by` set to "scientist", so authorship survives (`:267-285`, `:569`).
- **How strongly that is enforced**: by the tool surface. The scientist has no keep tool. The keep and discard
  routes carry no check of their own beyond the app's local-only guard, so any local script could call them. Whether
  the scientist's CLI sandbox can reach those routes directly is not established here.
- **The field book** holds, per face of a neuron, `jobs` (one, two, unknown), a `pattern` of up to 4,000 characters
  and four ticks: `other_pole`, `long_contexts`, `steered`, `readers_traced` (`:93-95`, `:361-396`). Writers are
  "reader" or "scientist" (`:92`).
- **How a test is stored**: `add_test(layer, n, face, test, by)` keeps `{kind: "written" | "blind", pattern, by, at,
  ...the test's figures}`, latest first, at most six per face (`:82-83`, `:455-477`). It carries the pattern text
  word for word, "so a pattern reworded since is seen to be untested", and "a test is a measurement, not a note:
  whoever ran it, it is kept as it came out" (`:46-47`). The atlas routes call it when a request says `keep`
  (`backend/atlas.py:1787-1788`, `:1970`).
- **Mechanics**: the module's own `asyncio.Lock`, never the GPU's (`:54`, `:97`); atomic replace on every write; an
  unreadable file is moved aside to `<file>.corrupt-<stamp>`, never overwritten, and the next reply warns once
  (`:171-185`). A lock-free read-only `peek_neuron` serves the vault's worker thread (`:613-661`).
- **Limits for a harness**: the schema knows neurons, heads and faces. There is no object for a hypothesis about a
  weight, an edit, or an experiment, and no link from a note to a registration or a result folder.

---

## 7. Where things are meant to live

From `.gitignore` and the code:

| what | where | tracked |
|---|---|---|
| Indexes, census runs, trials, volunteer intake, steer runs, scientist state | `atlas_data/` (`$THEODOLITE_ATLAS_DIR`; `backend/atlas_index.py:107-111`), 67 GB on disk today | no (`.gitignore`: "regenerable, hundreds of MB") |
| Stand-in output of dry runs | `atlas_data/stand-in/` | no |
| Tensors and caches of experiments | `experiments/results/*_cache/`, `*_acts/`, top-level `*.pt`, `*.npz`, `*.npy`; R3's shards and merged folders listed by name because "`*.pt` above does not reach subdirectories" | no |
| Two small arrays the code loads | `experiments/results/V6_quiet_dims_pers181.npy` (5.5 KB), `E3_background_mu.npy` (1.29 MB, sha256 pinned in `backend/engine_runtime.py`, provenance JSON beside it) | yes, re-included by `!` rules |
| The committed record of an experiment | `experiments/results/<name>/` : scripts, JSON, verdicts, `prereg/` | yes |
| Study-era scripts | `experiments/scripts/` (406 scripts and 4 JSON lists; untouched since 28 October) | yes |
| Fetched third-party sources | `experiments/R3_cache/`; "only their hashes are committed" | no |
| Third-party text | `experiments/results/N1B_corpus.jsonl`; pattern-test `fit/` and readers' sheets ("text of the corpus") | no |
| Live daemon logs | `experiments/results/labd_exec_log.jsonl`, `.local.jsonl`, `labd.log`; the frozen study log is tracked | no |
| Screenshots of the end-to-end check | `artifacts/` | no |
| Field notes, book, proposals, tests | `~/.local/share/theodolite/` (XDG), shared by every checkout | outside the repo |
| labd's token | `$XDG_RUNTIME_DIR/theodolite/labd-<port>.token` | outside the repo |
| Clean checkouts for registered runs and side sessions | `.claude/worktrees/<name>` (15 today) | separate worktrees |
| Local agent settings, env | `.claude/settings.local.json`, `.env` | no |
| Patch leftovers | `backend/*.orig`, `*.rej` | no |

`.claude/launch.json` has four entries: the server on 127.0.0.1:8765, the same on the CPU, the Vite dev server on
5173, and the stand-in server on 8766. None for labd.

Conventions that follow: large or regenerable output never goes in git; what is committed is small, has a hash or a
provenance file where it is binary, and third-party text is kept out even when results depend on it.

---

## Numbers

| what | value | where recorded | when |
|---|---|---|---|
| labd ready time and VRAM (both lenses) | 13.3 s; 21.24 GB allocated | `experiments/results/labd.log` | 16 Oct 2026 |
| labd study jobs | 314 jobs, 282 ok, median 71.3 s, max 456.9 s, 27,453 s total | `experiments/results/labd_exec_log_2026-10-16.jsonl` (my count) | 16 Oct 2026 |
| last recorded labd job | 2026-10-17T04:59Z | `experiments/results/labd_exec_log.local.jsonl` | 17 Oct 2026 |
| true VJP through 4-bit weights | about 2 s, 23 GB peak | `experiments/REPORT.md:90-93` | study, Oct 2026 |
| decode step, reference unpacker | 0.71 s whatever the batch (1 to 64 rows) | `scripts/census_run.py:88-91` | 27 Oct 2026 |
| decode step, compiled | 0.09 to 0.13 s (1 to 96 rows) | `scripts/census_run.py:94-96`; `COMPUTE.md` table | 28 Oct 2026 |
| compiled self-test over 410 linears | 1.83 s | `experiments/results/census_match_walk/L36-match_ceil-trial.meta.json` | 13 Nov 2026 |
| heard vault edit | 12.6 s | `backend/server.py:1455` | not dated |
| a registered trial | 49,110 answers, 75 to 88 minutes | `census_match_walk/prereg/RULES.md` | 13 Nov 2026 |
| a whole layer | 565,254 answers, about 12.7 h | `scripts/census_run.py:109-110` | 29 Oct 2026 |
| CPU reading of a whole layer | about 20 minutes, 6 GB, at nice 15 | `census_scaled_design/read_whole.sh:5` | Nov 2026 |
| spot check | 5 chunks, about a minute each | `scripts/census_intake.py:21-22`, `:57` | fixed 30 Oct 2026 |
| bars measurement | about 30 minutes; 10 layers, 13 variants, 64 answers each | `kit4/prereg/RULES.md:34-37` | 14 Nov 2026 |
| model at rest (lens freed, vision tower on CPU) | 17.97 GiB; worst batch of 64 reaches 22.1 GiB | `volunteers/kit.env:46-47` | 14 Nov 2026 |
| desktop's share of the home card | 5.1 GB, leaving 26.0 | `kit4/prereg/RULES.md:8-9` | 14 Nov 2026 |
| kit's free-memory floor | 23,900 MiB | `volunteers/kit.env:53` | 14 Nov 2026 |
| stand-in model | 6 layers, hidden 64, FFN 128, window 8 | `scripts/atlas_stand_in.py:44-47` | |
| checks | 27 `check_*.py` (22,466 lines), 8 `verify_*.py` (5,129 lines) | `scripts/` (my count) | 14 Nov 2026 |
| tests kept per face | 6 | `backend/field_notes.py:82` | |
| proposals waiting at most | 5,000 | `backend/field_notes.py:89` | |
| `atlas_data` on disk | 67 GB | `du -sh` | 14 Nov 2026 |

## Not established

- Whether labd's `/health` really stalls during a synchronous job: inferred from `labd.py:213-215`, not run.
- Whether a job posted while labd is still loading waits, as its docstring says (`labd.py:32-34`): not run.
- Whether the scientist's CLI sandbox can call `/api/atlas/proposals/keep` directly (another reader's area).
- The server's actual load time and resident VRAM today: the 10 s and 24 GB figures are from an owner's note of
  18 to 19 October, not from code or a dated log.
- Whether any process other than those above was ever given the pod-style `gpu.lock` on the home machine: no code
  does so.
- Whether `systemd-run` scopes are created by any script: none found in the repository; the practice is in the
  owner's notes only.
