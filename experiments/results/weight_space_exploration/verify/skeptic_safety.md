# Levelling under the safety and containment lens

Written 14 November 2026. Read-only, with one slip of mine stated at the end. Marks: **[read]** a file opened today;
**[measured]** a small command run today on this machine; **[arithmetic]** my sum; **[not checked]** said so.

What I read: the synthesis in full; `backend/scientist_chat.py`, `scientist_mcp.py`, `local_only.py`, the proposals and
export routes of `backend/atlas.py`, `field_notes.py` (signatures), `scientist_tools.py` (tool list and dispatcher),
`larql_vault.py:225-567`, `server.py:85-125, 1440-1515`, `atlas_ops.py` (hooks, `Weights`), `gguf_weights.py:555-577`,
`runtime_factory.py:251-272`, `NOTICE`, `.gitignore`, the pilot note and `pilot3.log`, `pilot7.log`; both digests by
search. What I ran: greps; a GGUF header reader on two committed files; a 20-line reach probe inside Codex's own
sandbox helper (`codex sandbox -P :workspace`, codex-cli 0.166.0), kept beside this file in `sbx/`.

## In one paragraph

The design is conservative where it is new (typed tools, no free code, nothing persisted, a fresh process for
anything of record) and nothing in it is fatal. But its containment claims are stated flat where they hold only for
some backends at one reach, and they rest on the scientist chat's existing reach, which I found is wider than the
repository says: **under Codex at the default reach the sandbox can write in `/tmp`, and `/tmp/torchinductor_<user>`
holds the pickles that every process loading the compiled unpacker reads.** "Nothing is ever in force" is true by
construction in the script and trial phases and by discipline in the server phase. The registry the six tools join
already holds an export tool, unbudgeted GPU tools, a tool that spends the owner's endpoint key, and 35 tools that
serve the unplanted base. The licence care is aimed at a phase-5 path while nine adapters holding Gemma's columns are
already on `origin/main` with no licence key in them.

## Findings

### 1. "No agent code runs in the process that holds the model"; "Nothing an agent does changes the model the server holds. A restart always comes up as the base" (vii, viii)

**Verdict: refuted for Codex at the default reach, as a matter of permissions. Severity: serious.** Exploit not run.

- The repository says Codex's sandbox "writes only in its own work folder" (`backend/scientist_chat.py:103`,
  `README.md:694`) [read]. Inside `codex sandbox -P :workspace` a command created files in the top of `/tmp` and in
  `/dev/shm`; `/run/user/1000`, `README.md` and the Hugging Face token file were refused for writing (EROFS);
  `socket()` was refused (EPERM) [measured]. So: no network, true; writes only in the work folder, false.
- The compiled unpacker is `torch.compile` (`backend/w4a16_fast.py:192`, `backend/gguf_fast.py:169`) [read]. Nothing
  in `backend/` or `scripts/` sets `TORCHINDUCTOR_CACHE_DIR` or `TRITON_CACHE_DIR` [grep]. In the venv, torch
  2.14.1 gives `cache_dir()` = `/tmp/torchinductor_<user>` and `fx_graph_cache` default True; `codecache` calls
  `pickle.loads` [measured]. The folder holds 135 files (fxgraph and aotautograd entries, 15 generated `.py`, Triton
  `.cubin`), all dated 12 November, none written today although the pilots installed the unpacker several times
  (1.7 to 2.2 s): the loads are cache hits [measured].
- Inside the Codex sandbox an existing fxgraph entry and a generated `.py` in that folder **open for writing**
  (O_WRONLY, nothing written, size and mtime unchanged); the entry begins `80 04`, a pickle [measured].
- So a Codex scientist at the default reach can replace a pickle that the server, a census run, a pilot and the
  proposed levelling script (which "installs the compiled unpacker" on every call, synthesis line 141) unpickle at
  load, as the owner's account, with network and credentials. The generated `.py` files beside it are loaded by
  path as well. The install's self-test compares bits afterwards and would not stand in the way of code that runs
  first. My reading that a cache hit unpickles the entry rests on torch's source and on the absence of new files
  today, not on a trial.
- Not checked: that the app-server's `"sandbox": "workspace-write"` (`scientist_chat.py:843`) is the same profile as
  `:workspace`; end-to-end execution (I did not and should not try).
- Claude Code at the folder reach has no such path (Write only under the work folder), but its rules were tried on
  2.1.312 (`README.md:701`) and the CLI on PATH is 2.1.325 [measured]. No script tests the real Codex at all:
  `scripts/check_scientist.py:325, 795` plays a fake `codex app-server` [read]. Pilot 0 re-checks Claude Code only.

**Fix.** Set the two cache variables to a folder outside `/tmp`, `/dev/shm` and the work folder in every entry point
that loads the model (server lifespan, census wrappers, the levelling script, the trial). Configure the scientist's
Codex so that `/tmp` and `/dev/shm` are not writable (the `exclude_slash_tmp` and `exclude_tmpdir_env_var` settings,
if 0.166 still honours them; I did not try). Add the probe to pilot 0 for the real Codex and re-run on every CLI
release. Until then, no Codex chat while a levelling or census process may start.

### 2. "An edit is an argument of a measurement, never a state of the server ... nothing is ever in force between calls" (iii)

**Verdict: weakened. Severity: serious.** By construction in the script and trial; by convention in the server.

- Process death: stands. The vault has no save or load (`larql_vault.py`, grep) and the lifespan loads the checkpoint
  (`server.py:109-125`) [read]; overlays and hooks are memory only. A killed process comes up as the base (subject
  to finding 1).
- Server phase: the only stated guards are a context manager and a refusal to *start* a reading while an overlay
  dict is non-empty (line 161). No postcondition is specified outside the trial of record (line 387), and that one
  checks "hook lists on all 410 linears". The house's own pushes hook decoder layers, `mlp`, norms and the embedding
  (`atlas_ops.py:392, 866-867, 1129, 1137-1164`) [read], and the design runs pushes as arms of the same forwards
  (line 238). A leaked push hook would pass the trial's check.
- "One forward a lock hold" (line 142) with a multi-forward reading means the lock is free between forwards. A vault
  INSERT from the page can land between forward k and k+1 (`server.py:1460-1467` takes the same lock per statement)
  [read]. The refusal must be per lock hold and a reading that spans an edit must be void; the text says neither.
  The folded twin's two forwards must sit in one hold; unspecified.
- `on_gpu` holds an asyncio lock round `to_thread` with no shield (`atlas.py:148-153`); the push route needed one
  because "a page closed mid-answer cancels this task" (`atlas.py:2188-2193`) [read]. Plain JSON routes are not
  cancelled today (no `.cancel()` in `scientist_chat.py`; the openai interrupt is a flag, `:1303-1305`). The
  proposal adds a streamed three-voice route and "a job with cancel"; either, built on `on_gpu`, frees the lock while
  the worker thread still has hooks on. The row-count guard would not catch it: three voices is 3 rows, and so is
  the push tab (`atlas.py:2148`).
- The design contradicts its own thesis once: planted faults are "a hidden edit on every row ... The read tools then
  serve the planted state" (lines 365-367). That is an edit in force for a brief.

**Fix.** One `assert_base(rt)`: every module's hook dicts empty (not the 410 linears only) and both overlay dicts
empty on all 410. Call it at the end of every levelling lock hold and at the start of every other `on_gpu`; on
failure answer 500, refuse further readings and say which readings since the last good check are void. No hook or
overlay crosses a lock hold. Written readings copy the push route's shielded join. State the planted exception.

### 3. Reach claims are unconditioned on the reach (vii, viii)

**Verdict: weakened. Severity: serious.**

- At `THEODOLITE_SCIENTIST_REACH=home` Claude Code has Bash over the home folder and Codex has
  `danger-full-access` (`scientist_chat.py:565-567, 843`); the module says "nothing stands between them and it"
  (`:22-23`) [read]. Every route answers any script of the same account (`local_only.py:7-10`) [read]. So at that
  reach a driver can POST `/api/larql` (a vault edit in force until rollback), POST `/api/atlas/proposals/keep`
  (`atlas.py:815-819`: no caller identity) [read], write the level book, and overwrite the checkpoint in the cache.
- Two of the five designs refused: validity "refuses proofs from a chat at the home reach" (`validity.md:373`);
  future "refuses to run when the scientist's reach is `home`" (`future.md:291`) [read]. The synthesis has neither;
  it only prints the reach on a harness card (line 424).
- "Only the owner counts it in", "the owner keeps a brief", "edits as proposals" all ride on routes that cannot tell
  the page from a script. At the folder reach that is sound because no backend can make a request of its own
  [measured for Codex: `socket()` EPERM; read for Claude Code: no shell, `:571-572`]. It is the tool list, not the
  route, that holds the line.

**Fix.** The six tools are not registered, and no word past a rank is written, when `reach()` is `home`. Say in
viii that every bullet is "at the folder reach". Keep owner verbs (keep a brief, plant, reset a budget, export) out
of `scientist_tools.py` and say so in `SCAFFOLD.md` as a trust boundary.

### 4. "The harness will not ... export by a tool" (viii)

**Verdict: refuted for the registry the tools join. Severity: serious.**

- `export_control_vector` is one of the 35 tools (`scientist_tools.py:332, 733`). It writes a GGUF holding
  `dose x p50 x unit(w_post (.) W_down[:, n])` into `atlas_data/exports/`, "a file of that name written over", and
  `GET /exports/{name}` serves any `.gguf` there (`atlas.py:2391-2411`; `control_vector.py:16-17`) [read].
- So a driver in a levelling session can already write weight-derived files by a tool, with a name it chooses, and
  can overwrite any file of that folder by name. If the phase-5 adapter lands in the same folder (unspecified), the
  owner's export can be replaced by a control vector of the same name.

**Fix.** Phase-5 exports go to a folder of their own that no tool writes; `export_control_vector` refuses to
overwrite; or the tool is left out of the list a levelling session is given. Reword the bullet to what is true.

### 5. The export path's debt under Apache 2.0 (viii, decision 9: "none exists today")

**Verdict: weakened. Severity: serious, for what it leaves out.**

- What is owed, section 4: (a) a copy of the licence; (b) prominent notices in modified files; (c) retained
  attribution notices; (d) the upstream NOTICE if there is one. The cached snapshot of
  `google/gemma-4-31B-it-qat-w4a16-ct` holds no LICENSE and no NOTICE, only a model card [measured, listing], so
  (d) adds nothing. Section 6 gives no right to the name beyond describing origin; `NOTICE` already says "Gemma is
  a trademark of Google LLC" [read].
- The synthesis lists (a), (b), the base id and "a name that says derived from Gemma". It omits (c), and does not
  say where the text goes in a one-file GGUF that will be passed on without its folder.
- **Weight pieces have already left the machine.** Nine LoRA adapters and two control vectors are tracked and on
  `origin/main` (`experiments/results/lora_export_L45/**`, `control_vector_L45/*`; commit 2b77881, 2 November 2026)
  [measured]. `n19903-mul0.gguf` holds `blk.45.ffn_down.weight.lora_b` (1, 5376) F32 and says of itself
  "B = (f - 1) W_down[:, n]"; its keys are `general.name`, `general.description` and `theodolite.*`, with no
  licence key and no attribution [measured with `GGUFReader`]. About 124,000 of Gemma's numbers in all [arithmetic].
  `NOTICE:7` says "This repository does NOT redistribute any model weights" [read]. The code digest flagged the
  missing licence note (digest_code line 265); section xi's repairs do not carry it.
- Which base? An additive `cols` delta is new minus the W4A16 column, and the export is served over a GGUF whose
  410 linears all differ (line 201). A silence computed against one base does not silence on the other. The
  existing exporter read its column from the GGUF itself (`theodolite.weights = gguf gemma-4-31B-it-QAT-Q4_0.gguf`)
  [measured]. "Carries the base id" is ambiguous until the export recompiles the recipe against the base it is
  served on and pins that one.

**Fix.** Add a seventh repair: `general.license`, a link, an attribution line and a "modified from" line in both
existing writers; one sentence of `NOTICE` put right. Export compiles from the recipe against the served base.

### 6. "`manifest.json` (no weights; may be committed ...) and `payload.safetensors` (... stays local)" (iv)

**Verdict: weakened. Severity: serious, because two later steps cannot run on a manifest.**

- For silence, scale and set the split holds: indices and factors in the manifest, arrays in the payload. For
  `remove_direction(layer, neurons, u, dose)` it is not said where `u` (5,376 numbers) lives. A `u` taken from an
  unembedding row or a neuron's write direction is weights; one taken from a difference of means is not.
- "Stays local" is enforced by nothing but `.gitignore:72` (`atlas_data/`). No rule ignores `*.safetensors` or
  `*.gguf` elsewhere [grep], and the registration pattern copies inputs into a committed folder with SHA256SUMS
  (`experiments/results/census_soft_walk/prereg/` holds its neuron lists) [read]. A trial "run from a clean
  checkout" that "loads the edit as data" (lines 143, 386) has no `atlas_data/`: the payload is either committed
  (then finding 5 applies) or read from outside the checkout, and the text does not choose.
- Decision 9 (manifests only) and decision 14 (volunteers re-run after phase 5) collide for every edit that is not
  rebuildable from its manifest: a volunteer needs the payload, and sending it is distribution.

**Fix.** The manifest holds the recipe (builder and arguments by reference to base tensors) wherever the term can be
rebuilt from the public base, and marks each term "rebuilt" or "carried". Registration folders hold the manifest and
the payload's hash, never the payload. Add the ignore rules. Carried payloads to volunteers wait for the policy.

### 7. "That seal holds against any reader" and phase 4's "a planted cheat (a session that tries to read a held file ...) fails" (vi, ix)

**Verdict: the fresh-draw seal stands; the phase-4 line is refuted for Codex by the proposal's own section vi.
Severity: serious for phase 4.**

- Stands: items that do not exist before the freeze cannot be read. Under Codex the shell does read elsewhere: the
  repository, `atlas_data/` and the token file were readable in the sandbox [measured].
- The synthesis limits file seals for item sets but not for the other files that must be hidden: a planted fault's
  edit (`atlas_data/edits/<id>/`, and its id in every book row, since "every row names its own edit"), a canary
  brief's flag, a brief's fields beyond "the text the driver is shown", and per-item rows if the book keeps them for
  replay. What `book` returns to a driver is not specified.
- The kept-back corpus fold named for starter briefs and the neutral-text battery is the odd-numbered windows
  (`atlas_index.py:125-130`), hidden only by the tool passing `fold="fit"`; the route's default is "all"
  (`atlas.py:1626-1627, 1635`; `scientist_tools.py:588`) [read]. The digest said so (lines 443, 540); not in xi.

**Fix.** Planted and canary sessions run only on backends with no file reach (endpoint; Claude Code once re-checked)
until the separate account exists, and the sheet says which. Specify `book`'s driver view. Enforce the fold at the
route.

### 8. Planted faults give "ground truth for discovery" through "the scientist's one registry" (vi, vii)

**Verdict: weakened. Severity: serious for phase 4's headline experiment.**

The six tools serve the planted state; the other 35 in the same registry (`logprobs`, `activations`, `top_firing`,
`compare`, `steer`, the atlas and census) serve the unplanted base, because nothing is in force on the server
[read: `scientist_tools.py:192-493`]. One `top_firing` beside one `capture` on the same text is a diff of the two
models. The experiment then measures differencing against a reference, which the published fault-finding tasks do
not give. Planting server-wide would break the thesis.

**Fix.** A planted brief is served a filtered list: the six tools and index-only reads, no live base reads; the
harness card names the list. Or the plant is applied by every live route for that chat, which is a larger change.

### 9. "Budgets, on every reply: GPU-seconds under the lock, fresh readings a brief, and tokens and dollars against a ceiling" (vii)

**Verdict: weakened. Severity: serious for any unattended session.**

- Enforceable server-side: GPU-seconds and fresh-reading counts for the six new tools. Not covered: the 35 existing
  tools, 21 of whose routes take the one lock (digest line 617); a driver can hold the instrument with `steer`.
- `blind_test` with `profile` makes the server call an endpoint with the owner's key
  (`scientist_tools.py:433, 798`; `atlas.py:1986-1998`) [read]: a spend path with no ceiling.
- No turn or dollar cap exists: `_claude_cmd` passes no limit (`scientist_chat.py:562-581`), Codex's options none
  (`:844`), `MAX_ROUNDS = 40` binds endpoints only (`:111`), and cost is read only from Claude Code's result event,
  after the turn (`:657`) [read]. A ceiling means a kill switch that is not there, and a rule for a fresh reading
  in flight when it fires (the route is not cancelled; the reading completes and should count).
- No cap on an edit's size. The gate (lines 206-213) refuses on numerics only. A `cols` term over every column of
  every `down_proj` is 27.7 GB in fp32 [arithmetic], on disk at `draft_edit` and on a card with 5.6 GiB of headroom.
  The future design's brief had "the cap on the share of weights changed, norm caps" (`future.md:355`); dropped.

**Fix.** Budget the lock, not the tool: GPU-seconds per chat across the whole registry in a levelling session. Caps
in the gate (terms, columns, bytes, relative change) and a disk quota for `atlas_data/edits/`. `blind_test`'s
`profile` off in such a session. Dollar ceilings stated as "checked between turns, where the backend reports".

### 10. "No recipe or brief for stripping safeguards or planting a trigger is shipped or kept"; the implant flag (viii)

**Verdict: weakened. Severity: minor as risk, serious as wording.** Coherent as a publication policy; theatre if
read as a control.

- The first-increment builder plus the capture handle (difference of means between two sets of the driver's own
  texts, then `remove_direction` on the top writers; lines 177-178, 402-405) is the refusal-ablation recipe, shipped.
  Only the brief is withheld, and the owner's attended session needs none. Gemma 4 carries no use policy by licence,
  and refusal-stripped builds of this checkpoint are listed publicly (digest_research lines 470, 474).
- The flag "a last-block neuron whose write is dominated by one word-piece's unembedding" describes the house's own
  example edit: `UPDATE LAYER 59 SLOT 7830 SET object = 'six'`, a last-block column moved along one piece's
  unembedding (`vault_example_edit/probe.json`; `larql_vault.py:511-523`) [read]. Decision 15 proposes rebuilding
  that statement on this kernel. A flag is a label; two neurons, layer 58 or a low-rank term pass it.
- As written, the rule also forbids the one defensive use this model invites: a brief to find and remove a planted
  trigger, which is what a planted fault is.

**Fix.** Say it is a policy about what the project ships and publishes, not a technical bar. Decide defensive
trigger briefs explicitly. Call the flag a label and name the house example as its first hit.

### 11. The level book: "append-only, hash-chained ... where no driver's sandbox can write" (iii)

**Verdict: stands at the folder reach if the book is outside `/tmp` and `/dev/shm` [measured writable]; weakened as
evidence. Severity: minor.**

A hash chain with no outside anchor does not show a truncated tail, which is how a fresh draw from `os.urandom`
would be re-rolled by anything with a shell (the attended session, a home-reach chat). Its place is unspecified;
under `atlas_data/` it is git-ignored and labelled "regenerable" (`.gitignore:71-72`), which a ledger is not.

**Fix.** Each fresh registration's head hash is committed or printed to the owner before the draw; the book lives at
a named path outside `/tmp`, backed up. The proposed lock file is an `flock`, as `scripts/pod/common.sh` does, and
likewise not in `/tmp` or `/dev/shm`.

### 12. Crash mid-reading

**Verdict: stands for the model; unspecified for the records. Severity: minor.**

Exceptions unwind through `finally` as the existing hooks do (`atlas_ops.py:344-412`) [read]; overlay restore is
dict assignment and survives a broken CUDA context. Left unsaid: a torn last line of the book; a fresh reading
registered and never finished (it should read *void* and count against the brief); a half-written edit folder
(write to a temporary name, then rename).

## What the proposal does not say

1. **Credentials under Codex.** `child_env` strips `HF_TOKEN` from the environment (`scientist_chat.py:178-197`),
   but `~/.cache/huggingface/token` is readable inside the sandbox [measured], as are `~/.codex/auth.json` and SSH
   keys by the same rule. The separate account of section xii is framed for seals only.
2. **There is no unattended runner.** A chat turn ends when the model stops and waits for a person; nothing
   re-prompts, nothing caps wall-clock. Phase 6's "three or more sessions a brief" needs one, with a kill switch.
3. **One work folder for every chat** (`scientist_chat.py:152-155`). Sessions meant to be independent read each
   other's notes; a second driver family shares them too. The start page that injects "your earlier notes" is a
   standing channel for text the driver copied from model output.
4. **The GGUF fingerprint skips the middle of the file** (digest_code lines 998, 1098). A same-size patched copy
   named by `THEODOLITE_GGUF` reads as its base. Not in the repairs.
5. **Batch size is a hidden knob.** Today one silenced neuron read 0.0085 nats at 12 rows and 0.7258 at 16
   (`pilot7.log`) [read]. "Halves its rows and installs again" on running out of memory (line 252) moves a reading
   across that line, and a large drafted edit can provoke it. Row count belongs in the hashed rule; a fresh reading
   that would halve is void.
6. **An incident rule.** What is done, and which readings are struck, when a hook or overlay is found where none
   should be.
7. **What goes to the driver's provider.** Weight slices and item strings travel in tool results; whether the
   provider's policy allows the briefs intended is an open question in the research digest (line 507).
8. **Pilot H is unsealed.** The "known answer, unnamed" sits in `experiments/results/E6_tool_audit_04_vault.partial.json`
   and the planted fault in a plan file, both readable by the attended session that is the driver.

## My slip

The first, unsandboxed baseline run of my probe created and at once removed a one-byte file in `atlas_data/`
(git-ignored) and in the home folder, as well as under `/tmp`, `/dev/shm` and `/run/user/1000`. Nothing remains
(listed; `git status` clean). I removed those two targets before the sandboxed runs, which wrote only under `/tmp`
and `/dev/shm` (removed) and otherwise opened existing files for writing without writing a byte.
