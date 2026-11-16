# The Proof Room: a weight-space environment built from the measurements outward

A design for the owner's review, written 14 November 2026 from the validity lens. Nothing here was run. Every figure
is carried from the two digests of 14 November 2026 with its date and marked measured, or is marked as my arithmetic.
Layers count from 0 unless a page is named.

## In brief

The feature is a place where an agent (or a person) changes a few of Gemma's weights and finds out, quickly and
without being able to fool itself, whether the change did what was said. The literature up to 14 November 2026 says
three things loudly: gains seen during a search shrink or vanish on held-out data (Hu et al., arXiv 2611.09382:
held-out gain was below the development gain in all 14 tabulated runs; one candidate, which the editor itself
discarded, was +29.69 points best-of-search and -0.35 held out on a base already at 97.70%; three repeats of one
search gave +29.45, +5.00, +27.30), agents' own accounts are unreliable (OverclaimBench, arXiv 2609.20812: 80.4% of incomplete
runs gave a misleading account), and the published headline was nearly matched by a settings change (Toledo's
re-run: 44.0 to 49.7% unedited against the paper's 7.9% baseline and 48.1% edited). So the design starts from what
may be claimed and works back to the code.

Five rules carry it.

1. **The base is never edited during research.** An edit under test exists only as one arm of one reading: a
   per-row change to the output of a linear layer, in a forward whose other rows it does not touch. There is nothing
   to roll back and nothing an earlier experiment can leave behind.
2. **Every reading takes a backsight.** The unedited row is in the same forward as the edited rows, and beside every
   candidate sit chance edits made the same way. No comparison crosses forwards.
3. **The items a claim rests on do not exist until the edit is frozen.** They are drawn after the edit and the rule
   are hashed, as the volunteer intake draws its spot chunks after a bundle arrives.
4. **The warden applies the rule, not the agent.** It writes the verdict and the one sentence a passed edit may
   claim. "Keep the base" is one of seven ordinary verdicts.
5. **Three grades of evidence.** A sighting (seconds, claims nothing), a proof (minutes, a checked lead), a trial of
   record (under an hour, the owner's word, the house's hashed registration, a fresh process). Only the third is a
   finding.

What this costs: the agent gets no code execution in the process that holds the model, the measurement vocabulary is
fixed and grows only by code review, and roughly half of every forward's rows are spent on controls. I accept all
three.

## i. What the environment is

One process holds the model: **the instrument's own server**, under its one `_GPU_LOCK` (`backend/server.py:99`).
Not the lab daemon. The daemon executes arbitrary Python beside the weights, restores nothing between jobs, has no
cancel, and defaults to the reference unpacker (`experiments/lab/labd.py:157,198-233`); it stays what it is, the
study's bench for a person. A second process cannot hold the 31B on a 32 GB card, and the server already has the
compiled unpacker, the lock, the stand-in mode and the scientist's dispatcher.

| Part | What it does | Stands on | New code |
|---|---|---|---|
| **Delta kernel** | Applies typed edits as per-row output-side deltas on any packed linear, for the length of one `with` block | the hook pattern of `atlas_ops.installed` (`backend/atlas_ops.py:344-412`); the delta arithmetic of `int4_cpu.with_overlay_terms` (`backend/int4_cpu.py:138-164`); `gguf_weights.dense_cols/dense_rows` for the base's own numbers | `backend/emend.py`; touches no pinned file |
| **Plate** | One fixed-shape forward: items x arms, base row always present; reads fp32 odds, per-position KL, or greedy text | `atlas_ops.next_logprobs`, `write_rows`, `forward_from_layer`; `causal_check.logits_fp32` | `backend/proof_plate.py` |
| **Warden** | Keeps briefs, draws items and seeds, builds nulls, runs batteries and baselines, reads the verdict, writes the claim | `census.control_dirs`, `control_refs`, `distinct4`, `first_departure`, `runtime_record`, `write_json_atomic`; `mcp_scientist.paired_statistics`, `benjamini_hochberg`; the intake's draw-after-arrival (`scripts/census_intake.py:330-334`); `field_notes.propose/decide` | `backend/warden.py` |
| **Ledger** | Append-only, hash-chained record of every reading, registration, draw and verdict | `census.file_sha256`; the prereg folders' practice | part of the warden; lives in `~/.local/share/theodolite/proofs/`, outside the scientist's work folder |
| **Edit store** | Content-addressed edit files | `atlas_index.save_npz` (atomic) | `atlas_data/edits/` (git-ignored) |
| **Job runner** | Runs a proof plate by plate, giving the lock back between plates; status and cancel | `atlas.on_gpu` (`backend/atlas.py:148`); the sweep's cooperative cancel | small module and three routes |
| **Tools** | Nine typed tools for the scientist | the registry and single dispatcher `scientist_tools.call` (`backend/scientist_tools.py:534`), which reaches all three backends | additions to `scientist_tools.py` |
| **Trial of record** | Standalone, fresh process, server down, house registration folder | `experiments/results/census_match_walk/prereg/` (RULES.md, runner, reader, SHA256SUMS, repo_commit, prereg_at); `night.sh`'s refusals | `scripts/proof_trial.py` and a folder generator |

Relations. The **vault** stays as it is: one column, heard on one question. The Proof Room does not write vault
overlays while researching. The **atlas and census** are read as facts about the base and are labelled so. The
**scientist** gains tools in Phase 3 and nothing before. The **lab daemon** is not used.

## ii. What an edit is

**Representation.** An edit is a list of typed operations, each an additive change to one named tensor of the base:

- `col_add`, `col_set`, `col_scale` on chosen columns (a neuron's write in `down_proj`; a head's block in `o_proj`);
- `row_add`, `row_scale` on chosen rows (a neuron's reads in `gate_proj` and `up_proj`);
- `lowrank_add` (B [out, r], A [r, in]), which covers a rank-one write, "remove direction u from these columns at
  dose d", and a least-squares key-to-value fit computed elsewhere;
- `sparse_add` (entries);
- later, with their own tests: `embed_row_add` (the tied embedding and unembedding, changed together), norm weights
  and layer scalars.

Every operation is stored as a **difference from the base** (a `col_set` is stored as new minus old), so composing
edits is adding them, in any order. Two scales of one column therefore give f1 + f2 - 1, not f1 x f2: the same rule
llama.cpp applies to adapters served together (`scripts/export_lora.py:50-55`), chosen so the instrument and the
export mean the same thing.

**No operation without its nulls.** An operation is admitted only with (a) a by-bits conformance check on the
stand-in, (b) two null generators in its own representation (below), and (c) a refusal list. Refused at drafting:
the final norm, a positive uniform rescale of a whole block (the norm after it undoes it:
`backend/atlas_ops.py:244-264`), a layer or neuron out of range (the vault clamps instead:
`backend/larql_vault.py:481,500`), and v_proj on the ten full-attention layers, which have none.

**How it is applied.** As a forward hook on the target linear that adds, for the rows carrying that arm only,
`(x A^T) B^T` to the layer's output, summed in float32 and rounded once to bf16. Rows of other arms and the base row
are not written at all. For column, row and sparse operations the delta is first made as "the new entries rounded to
bf16, minus the base entries", as the reference forward would lay them on, so a change too small to survive bf16
rounding is nought here too (Hu et al. saw such changes vanish). Low-rank factors are applied as given, as an
adapter adds them. The pinned forward (`backend/engine_runtime.py:352-377`) is not edited; with no arm installed the
kernel leaves no hook behind.

**Three standings, never mixed.** Every reading names which it measured.

| Standing | What it is | Status in this design |
|---|---|---|
| **beside the grid** (delta form) | float side delta on the 4-bit base, per row | the form every sighting, proof and trial measures |
| **folded** | the bf16 overlay written into the unpacked weight for every row, as the vault's forward does | Phase 4; licensed only by a measured fold check |
| **exported** | a LoRA adapter GGUF through `export_lora.write_lora`, run by llama.cpp on a GGUF | Phase 4; a separate re-measurement; the lmstudio QAT Q4_0 file differs from the checkpoint in all 410 linears, so it is another object |
| **on the grid** | re-quantised into the 4-bit codes | not built. A `down_proj` column shares each of its 5,376 scale groups with 31 other neurons (derived from the layout, untested) |

For every edit the store also records, by CPU arithmetic alone: relative Frobenius change per tensor, columns and
rows touched, the size of each changed column in write ceilings, the share of entries that would round away in bf16,
and a **grid survival** figure (the share of the delta's length kept if each entry were snapped to the sixteen levels
under the existing scales). These describe; they claim nothing about behaviour.

**Hash, save, reload.** The edit id is the sha256 of a canonical manifest plus the raw arrays, and the manifest
names the base twice: the weights card's `files_sha256` and a content hash of the packed words and scales of each
touched tensor as loaded (the card names files, not tensors: `backend/runtime_factory.py:228-272`). A reload
verifies both and refuses in plain words on any difference. A **state** is the base plus a sorted set of edit ids
with doses; its hash is the state id.

**Rollback.** In research there is none to do. After every job the warden checks the **misclosure**: hook count as
found, overlay dicts as found, and a one-row canary whose bits equal the canary taken when the job began. A
misclosure other than nought voids the job.

**How a reading records the edit.** Every ledger entry carries, per arm: kind (base, edit, null, baseline, canary),
state id, standing, dose; and for the plate: rows, padded length, unpacker and its `fell_back` count, card, commit,
clean tree or not, metric id with its KL direction and logit precision, item-set hash, seeds. The three KL
directions now in the code (`backend/atlas.py:2477`, `backend/causal_check.py:425-427`,
`backend/larql_vault.py:176-180`) become one here: KL(base || edited), from fp32 logits, named in every record.

## iii. How the agent reads

A language model cannot work with 29.3 billion weights as text; reading without tools is shown only to about
172,000. What it can work with is the **neuron as the unit and the atlas as the gazetteer**: 1,290,240 neurons, each
with its write column, its two read rows, its strongest links, its stored passages, and at twelve layers its census
standing. Four kinds of reading, each stamped with its standing:

1. **Weights**, by typed tool: a bounded slice of a named tensor with a numerical summary; column and row norms; top
   entries; cosine with a direction; the neurons of a layer whose write directions align with a direction;
   products of an arbitrary neuron pair (`atlas_ops.pair_products`); top singular vectors of one matrix as a cached
   service; the lattice view (codes times scales) of a slice. Tensors never travel through the driver's context:
   results are small tables or server-side handles.
2. **Atlas and census facts**, always labelled *of the base, as built on [date]*. They are never offered as a
   description of an edited arm.
3. **Activations** on texts the agent writes or practice items: residuals, feed-forward activations, attention; and
   the difference of means between two sets, kept as a direction handle. This is the read that carried the published
   Qwen edit.
4. **The ledger**: everything already tried on this brief, with verdicts.

**Every weight-derived answer is typed a lead and says how to confirm it.** The instrument's own measurements
require this: a link's sign from the weights agreed with the measured change at 0.336 of links from rest, below
chance, though its size is real (rho 0.89; `ARCHITECTURE.md:772-800`).

**The trust card.** Every read reply and every ledger entry carries a per-layer card, extending
`label_gate.LABEL_GATE_CARD`:

| Reading | Trusted at | Elsewhere |
|---|---|---|
| lens words | layers 40-57 (label gate, AUC 0.84 graded) | held back; never on an edited neuron |
| steer lens | 39-52 agrees (0.60-0.91); `steer_check.json`, 14 Nov 2026 | weak at 32-38, unchecked at 0-31 and 53-59 |
| census doses | break_rel 39-51, soft_ceil 38 and 52, match_ceil 32-37 (layer 36 is the owner's open call) | no calibrated push below 32 or above 52 |
| link size / link sign | size at every layer / sign at none | |
| passages, pattern tests, live odds | every layer | |

The card limits **stories, not effects**. An effect read from the model's own odds stands at any layer, since the
documents hold results reached without a lens free of the lens caution. But the registration records how the site was
chosen, and where that reading is outside its band the claim sentence says "site chosen by a reading not trusted at
this layer".

## iv. How a hypothesis is tested, and what it costs

**The plate.** One forward of fixed row count and fixed padded length. Rows are items x arms. Arm 0 of every item is
the base. The other arms are candidates, chance edits, baselines and two canaries. Sixty-four rows by 48 word-pieces
is the working shape; a step costs nearly the same from 1 to 64 rows because the time is the unpacking (0.091 s at 1
row, 0.109 s at 64, compiled, measured 28 October 2026), so controls ride almost free.

**The measurements**, a fixed vocabulary, each with a named direction and precision:

- *odds*: fp32 log-probability or logit difference of named word-pieces at named positions, teacher-forced;
- *text KL*: KL(base || edited) and top-1 agreement at every position of a text, computed a slice at a time
  (full-vocabulary fp32 logits for a whole 64 x 48 plate would be about 3 GiB by my arithmetic);
- *writing*: greedy text per arm in one batch, scored by a rule fixed in the brief (exact word-pieces, a pattern),
  with first departure from the base row and distinct 4-grams;
- *internal reads*: one neuron or one direction across arms.

**What it costs on this card.**

| Reading | Shape | Seconds | Standing of the figure |
|---|---|---|---|
| one teacher-forced plate | 64 rows x 48 pieces | 0.5 to 2 | my arithmetic. Lower end: prefill of 64 census questions, 0.45 s (measured 28 Oct 2026). Upper end: the passages pass, 2,000,054 word-pieces in 20 minutes with every activation hooked, about 1,667 a second (measured 28 Oct 2026) |
| a sighting, odds | 8 practice items x 8 arms | 1 to 2, plus the wait for the lock | my arithmetic |
| a sighting, writing | 16 rows x 48 new pieces | about 5 | my arithmetic from the measured step; its measured kin is 13 rows x 80 pieces in 8.8 s (11 Nov 2026) |
| a proof | 96 fresh items x 64 arms x 48 pieces, plus three batteries | 2 to 5 minutes | my arithmetic from the same two rates and the census's 11.88 answers a second (29-30 Oct 2026) |
| a trial of record | 256 fresh items x 128 arms (two plates an item, each with its own base row), larger batteries, fold check, fresh load (13-16 s, measured October 2026) | 10 to 35 minutes | my arithmetic from the same rates; a census trial is 68-86 minutes (measured) |
| drafting an edit and its statistics | CPU | under a second for column edits | my arithmetic |

Memory: the model with the lens rests at 21.2 GiB and leaves about 5.6 GiB beside a desktop; a 64-row census batch
of 64 new pieces took 4.8-5.3 GiB above the model (28 Oct 2026). A 64 x 48 teacher-forced plate carries no
key-value cache and should take less; that is not measured (pilot 3). Sixty-four arms of 64-column edits hold about
88 MB of deltas (my arithmetic).

**How many candidates share a forward.** In a sighting, up to 15 candidate arms an item, with at least two chance
arms the warden adds unasked. In a proof, one frozen edit with its reversed sign, two other doses, up to four
baselines, two canaries and 52 chance edits.

**What is cached, and what is not.** Cached: the edit's deltas, tensor content hashes, top singular vectors, the
base's activation scale at the items (one pre-pass). **Not cached: the base row.** Memoising the unedited model is
the obvious saving and I refuse it: it rests on same-shape repeatability shown for census batches
(`experiments/results/census_determinism/summary.json`), not for hooked teacher-forced plates, and the base row costs
one row in 64. Running only the layers above the lowest edited one (`atlas_ops.forward_from_layer`) is allowed for
sightings once pilot 8 has shown it agrees with a whole forward on the 31B (today it is checked only on the CPU
stand-in); proofs and trials run whole forwards.

A driver's own turn took 20 to 70 seconds in the published logs, so a sighting of a few seconds is not the slow
part of the loop. The guards cost GPU rows, not waiting.

## v. What stops the agent fooling itself and the owner

**The brief.** The unit of scope. The owner writes it, or keeps one the agent proposed (the notes' propose-and-keep
pattern). It fixes, before any search: the behaviour in plain words; the metric; an **item generator** (frames and
slot lists) or a held item file; a near-miss generator (the same form where the behaviour should not move); up to
three prompt-arm sentences written by the brief's author, not the driver; which tensors and layers may change; the
budgets; the gradient level (none, by default). The driver cannot open a behaviour by itself.

**Three item tiers.**

- *Practice*: drawn from the generator on request; the agent sees items and per-item results; unlimited within the
  GPU budget.
- *Proof items*: drawn by the warden from `os.urandom` **after** the edit id and the rule are hashed into the
  ledger. The agent receives aggregates only, never an item or a per-item row.
- *Trial items*: the same, drawn inside the trial's fresh process after its registration is committed; spent once.

A fresh draw needs no separate account and no second model. It is honest only when the generator's space is far
larger than anything practised on, and the claim is limited to "items drawn from generator G". For behaviours on
natural text a held file split by a keyed hash is the fallback; its seal is only as strong as the scientist's
folder reach (below).

**Chance edits (the matched nulls), two families, always both.**

- *Same site, another content*: the same columns or rows, the change replaced by a random one of equal length as the
  stream receives it. Drawn as the census draws its "columns" controls, a random mix of the layer's own write
  directions through the post-feed-forward norm (`census.control_dirs`), because a plain isotropic direction reaches
  the next layer about a hundred times louder than any neuron's write.
- *Another site, the same rule*: the same operation on other neurons of the layer, matched on write ceiling **and on
  how hard they fire on these very items** (one base pre-pass). Random neurons mostly sit shut, so an unmatched
  draw would flatter every edit.

The follow-up literature reports a random direction on the same neurons recovering about a fifth of the published
Qwen gain while random sites of the same size did nothing, and Hu et al.'s one matched control was a single draw
(10 of 48 against the base's 6 and the edit's 20) that cannot be told from no effect. Hence both families, and
dozens of draws, not one.

**Noise floors, four, each named.** (a) Inside one forward the floor is nought: identical rows are bit-identical
(the readers rely on it; pilot 1 confirms it for hooked plates). (b) The delta path's own floor is what the
norm-undone canary moves. (c) The floor an edit must clear is the spread of its chance edits, on the same items, in
the same plates. (d) Across batch shapes there is no floor worth using (KL up to 0.014 between shapes in the study;
an unpushed answer in a batch of 64 parted from itself after 22 word-pieces on 28 October 2026), so nothing is ever
compared across plates: each plate's rows are differenced against that plate's own base row.

**Canaries, in every proof and trial.** A zero edit, which must equal the base row bit for bit. A positive uniform
rescale of the edited layer's whole `W_down`, which the norm undoes: whatever it moves is the floor of the delta
path. Either out of bounds voids the reading. And among the briefs, the owner may plant **canary briefs** with no
possible fix; a claimed success there is a defect in the driver or the harness.

**Batteries, run unasked on every proof and trial.**

1. *Neutral text*: text KL and top-1 agreement on kept-back corpus windows (the odd windows,
   `atlas_index.passage_kept_back`), a fresh draw each time so repeated edits cannot be tuned to a fixed set;
   non-finite logits masked.
2. *Habitat*: the same on the stored passages where the edited neurons fire hardest (the passages index holds 16
   strongest and 6 sampled events a face). A column edit acts only where its neuron fires, so neutral text can show
   nothing while the damage sits here. Only this instrument can run this battery.
3. *The base's own answers*: a bank of varied questions written greedily by every arm in one batch; the share of
   answers identical to the base row's through 48 pieces, and the median first departure, beside the chance edits'.
   It needs no labelled data and claims no correctness, only retention.
4. *Near misses*: the brief's near-miss items; the change there against the change on target.

**Cheaper knobs, as arms of the same plates.** The same neurons scaled on the activation (`mul`, exact, exists); a
steering push along the edit's mean write direction at a matched displacement with the start mark not spared
(exists); the brief author's prompt sentences; a decoding change (a bias on named word-pieces, computed from the base
row's own logits at no extra forward). When a cheaper knob reaches the edit's effect with no worse retention, the
verdict says so; the edit is not failed, its claim is cut down. A small trained adapter is **not** among the arms:
no code here returns a gradient with respect to a weight, and a training loop is another access level. The lab
kit's activation gradients through the 4-bit weights could feed one later (a linear's weight gradient is an outer
product of backward signal and input), but until that is built and measured the sheet says "not compared with a
trained adapter" every time. Where training was compared in the literature it beat editing, so this is a real hole
in the claim and is printed as one.

**Registration.** One tool call. The agent names the brief, the frozen edit id, a rule preset and its forecast
(direction, least effect, expected retention). The warden adds the list of ledger entries already seen on this
brief, hashes the record into the chain, and only then draws seeds and items. The rule presets are code, versioned
and hashed with the commit; the agent cannot write a rule.

**The rule (preset `proof-1`).** An edit stands when all hold: (1) the paired per-item difference from the base row
is in the forecast direction, at least the forecast size, by sign-flip permutation
(`mcp_scientist.paired_statistics`); (2) its effect exceeds every one of the 26 same-site chance edits and every one
of the 26 other-site ones (about 1 in 27 chance edits would do so by luck, a family); (3) the reversed edit moves the
other way and the half dose moves less; (4) each battery is within its bar; (5) the canaries are clean and the
misclosure is nought. The bars in (4) are **[not yet measured: pilots 5 and 6]** and will be fixed and hashed before
first use; I will not invent them. The trial preset doubles the chance edits to 60 a family and adds a writing gate:
teacher-forced odds alone license only a claim about odds (teacher forcing was the most severe of several flaws
behind 96.8% falling to 38.5% in *The Mirage of Model Editing*).

**Read once, counted always.** A proof of a given edit id on a given brief is read once, as the house's registered
reader refuses a second reading. A changed edit is a new id and spends the brief's proof budget; the sheet prints "the
k-th proof on this brief" and the family-wise chance beside every p. Re-running an unchanged model to fish for noise
(26 of 39 flagged attempts in the automated-researcher study) has nowhere to happen.

**Seven verdicts**, written by the warden: *stands*; *stands, matched by a cheaper knob*; *no better than chance
edits*; *works but harms*; *did not hold* (seen in practice, gone on fresh items); *keep the base*; *void*. A brief
closes with one, and "keep the base" is expected most of the time: six of sixteen published trajectories ended there,
and on a strong 9B base the largest held-out gain was +3.03 points.

**What a passed edit may claim.** Exactly this sentence, filled by the warden, and nothing else:

> On [date], on [card], [unpacker], **beside the grid**: edit [id] to [tensors] of base [weights id] moved [metric]
> on [n] items drawn from [generator] after the edit was frozen by [effect, interval]; [k of C] same-site and
> [k of C] other-site chance edits moved it as much (the [k]-th proof on this brief); the reversed edit moved it
> [x]; neutral text [KL, agreement] against the chance edits' [range]; habitat [..]; the base's own answers kept
> [x of y] against [..]; near misses moved [..]; a prompt reached [..], a decoding change [..], a steering push
> [..]; not compared with a trained adapter. Site chosen by [reading], [trusted | not trusted] at layer [l].

It may not claim: that the model is better; that an ability was added; what a neuron means; anything about the
folded, exported or 4-bit form without that form's own measurement; anything outside the generator's family; that
the edit is the best or the only one; that another search would find it. A *proof* that stands is shown as "a lead
that stood at proof". Only a *trial of record* is "passed", and only the owner counts it in.

**Against misreport.** The owner reads the warden's **proof sheet**, not the agent's account: practice-best, proof
and trial effects side by side (the shrinkage in plain sight), every attempt on the brief including the discarded,
the forecast marked met or missed, the setup box (driver as served, effort, reach, tools and prompt hashes, budgets
spent). The agent's conclusion is filed through a tool that accepts only a ledger reference and a claim the verdict
licenses; its free prose is shown under "the scientist's own words, not checked". The ledger keeps each driver's
record of forecast against outcome.

**Of record.** The owner's word turns a stood proof into a house registration: a folder generated from the ledger
record (RULES.md, the exact runner and reader, inputs, repo_commit, prereg_at, SHA256SUMS, the ledger's chain head),
committed, run from a clean worktree at that commit with the server down, read once on the CPU. Same reader code,
fresh process, fresh items, more nulls.

## vi. The agent interface

**Today.** A current frontier model plugs in through what exists: Claude Code or Codex over the stdio MCP server
(`backend/scientist_mcp.py`), or any Chat Completions endpoint through the server's own loop, by one entry in
`endpoints.json`. Nine tools join the registry, so all three backends get them at once:

`brief` (what is open, budgets left) · `weights` · `capture` · `draft_edit` (typed operations in, edit id and
descriptive statistics out, refusals in plain words) · `sight` · `pull_proof` · `job` (status, cancel) · `ledger` ·
`conclude`.

They are projections of one plain Python module with a versioned JSON schema. The existing 35 atlas tools stay
beside them, with the trust card added to their replies.

**Typed tools first; code second and never beside the model.** Three operations are typed, counted and ledgered for
ever: observing the live model, drafting an edit, asking for a verdict. In Phase 5 the driver also gets a **CPU-only
kernel** in an operating-system sandbox: no network, no GPU, one scratch folder, the base weights memory-mapped
read-only, the handles of its own captures loadable as arrays. It can compute anything about weights and recorded
activations. It cannot run the model, because every forward must pass the gate that writes the ledger and adds the
controls.

**A stronger model later.** It needs no rewrite because nothing in the surface encodes what today's model cannot do:
no staged workflow, no council, no context-reset machinery. A better driver uses the same three gates harder and
does more in the kernel. What it cannot do without a code change is invent a new kind of intervention; a new
operation arrives by review, with its conformance check and its two nulls. That is the price of rule 4 and I would
pay it again.

**Budgets**, per brief, shown in every reply: GPU-seconds under the lock (not passes), proofs, and driver tokens and
dollars (Claude Code's per-turn cost is on the `done` event and is dropped today; the dispatcher logs it).

**State across sessions.** The ledger, the edit store, and a per-brief work folder. Each session opens with a
*standing* generated from the ledger. The agent's own notes are shown back under "your notes, unverified", since an
auditing agent has grown falsely confident from its own memory files.

**Left thin on purpose.** No workflow engine; no LLM judge (none is configured on this machine, and a judge of the
driver's family is a known weakness); no per-tool page rendering before Phase 4 (the proof sheet is a file the chat
can already link); the methods text is one short third prompt file, because rules written in prompts did not hold
for agents with free code.

## vii. Safety and scope

- **Containment.** At the scientist's default reach (folder) no backend can run code that reaches the model or read
  outside its work folder; this was tried on Claude Code 2.1.312 only (28 October 2026) and must be re-run on the
  version in use. The warden records each chat's reach and **refuses proofs from a chat at the home reach**. The
  ledger, briefs and any held items live outside the work folder.
- **The seal's two strengths**, named in every sheet: *fresh* (items did not exist before the freeze; holds against
  any reader) and *held* (a file out of the scientist's reach; holds against a tools-only agent, not against one
  with a shell). A separate account for the warden would close the second; that is the owner's decision.
- **No approvals in the loop.** People approved about 93% of permission prompts in Anthropic's data. The loop has
  boundaries instead, and the owner's word is kept for four crossings: keeping a brief, a trial of record, putting an
  edit in force, any export.
- **The harness will not**: run agent code in the model's process; bake an edit into 4-bit codes; write an edited
  model file; push or publish anything; open a behaviour without a kept brief; change anything that writes census
  answers or the loader's bits (kit 4 is live; the kernel is a hook that exists only inside a plate).
- **Edited weights.** An edit is a small delta file in a git-ignored folder. A restart always comes up as the base.
  In force (Phase 4) means a state id appended to the model id, as `+int4mm` is, so no index, census or reading can
  be mistaken for the base's. An export carries a colophon: base id, edit id, ledger link, the measured claim with
  its nulls and discarded attempts, a notice that the file is modified, the Apache 2.0 text, and a name that says
  derived from Gemma.
- **What it must not become.** Weight-edit implants were shown on the bf16 Gemma-4-31B-it on 11 November 2026
  (Lazarus; 99.51% in its best cell on recorded trajectories). No trigger-to-payload recipe ships, and a brief that
  asks for a hidden trigger is not kept.
- **Untrusted text.** What the edited model writes and every item string is data when shown to the driver.

## viii. A phased path

| Phase | Delivers | Proves | Effort |
|---|---|---|---|
| **1. The paired plate** (small) | `emend.py` for `down_proj` column operations; `proof_plate.py`; a command-line sighting run with the server down; a ledger line per reading; `check_emend.py` on the stand-in and `verify_emend.py` on the card | that a weight change and the untouched model can share a forward bit-safely; what a plate costs; that the study's one causal edit is recovered (pilots 1-4) | 3-5 days; about an hour of the card |
| **2. The proof** | the warden: briefs, generators, fresh draws, registration, both null families, batteries, cheaper-knob arms, canaries, the reader, the proof sheet; server routes and the job runner; planted-failure checks of the reader (a planted overfit must read *did not hold*, a planted harm *works but harms*) | that a person can get a verdict they can trust in minutes; the bars (pilots 5-7) | 8-12 days; 3-4 hours of the card |
| **3. The scientist's tools** | the nine tools; edits as proposals a person keeps; budgets and cost log; the methods file; a scripted recipe (difference of means, aligned writers, remove at a dose sweep) run through the same gates as a no-agent baseline | whether a current driver reaches a planted answer and stays honest on a canary brief (pilot 9) | 6-10 days |
| **4. Of record, in force, exported** | the trial-of-record script and folder; the fold check; an edit in force under its own model id with one read path that shows it; adapter export from the store with a colophon and a llama.cpp re-measure; a page for the ledger and the edits in force | that a claim survives a fresh process and the form that ships | 8-12 days |
| **5. Wider** | every packed linear, low-rank and row operations, the tied embedding, norms; the sandboxed CPU kernel; a single attribution backward as a typed reading at its own access level; partial forwards; a decoy audit (permuted neuron order) for weight-reading claims; a setter model for items; a warden account | | open |

Phase 1 is useful alone: it already answers what the vault's one heard question cannot, whether a column edit does
more than the same edit on neurons that fire as much.

## ix. Pilots, before the design is committed to, in order

| # | Measurement | What would change the design |
|---|---|---|
| 0 | Re-run `verify_scientist_cli.py` on the Claude Code version in use. No GPU. | If folder reach no longer holds, *held* seals are void and the warden account comes first. |
| 1 | **Null delta by bits** (stand-in, then the card): a plate with the kernel installed and zero deltas against the same plate with no hook; the base row of a plate whose other rows are edited against an all-base plate of the same shape. | If base rows are not the unhooked bits, rule 2 weakens to "paired within a measured floor"; every plate then carries twin bases and the floor is printed. |
| 2 | **Repeatability** of a hooked teacher-forced plate at fixed rows and padded length: twice in one process, once in another; then the same row at another padded length. | If not identical, the padded length joins the plate's identity, and repeat plates are needed to size the floor. Never licenses caching the base. |
| 3 | **Cost and memory** of plates at 32x64, 64x48, 64x128 and 96x48, with 1, 16 and 63 arms on one layer and on eight; fp32 readouts at named positions and at every position. | Above about 5 s or above the headroom with the lens: smaller plates, or the lens freed for a sitting (whether the server can free it is not established). |
| 4 | **Known answer.** The study's edit: the ten 'she' neurons of layer 50 silenced moved the female contexts by -1.80 log-odds against at most 0.04 for ten random neurons (E6 report, first committed 16 October 2026). Read it as a delta arm, as a `mul 0` push with the start mark not spared, and as the vault's DELETE. | If the delta arm does not recover it, stop: the kernel is wrong. The gap between the three forms sizes the fold check. |
| 5 | **Nulls and audibility** (the decision gate): at layers 20, 36, 45 and 55, 200 random and 20 most-active single-column edits and 20 sets of 64 columns, at half, one and two write ceilings, on 64 generator items; both null families at 60 draws. | The spread of chance edits sets the item counts and the bars. If almost nothing below one write ceiling is audible, as the vault's own note suggests (the most active neurons at five late layers were not heard at strengths 3 or 30), the unit of edit becomes a set across layers, plates grow, and the owner should decide whether the feature is worth building for a healthy 31B. |
| 6 | **Battery floors**: non-finite logits on the 31B (sourced only for the 12B); neutral, habitat and own-answer figures under the zero edit, the norm-undone rescale and the chance edits; first-position against every-position KL. | Fixes the battery bars and the KL definition. If the own-answer battery cannot tell a chance edit from nothing, it is dropped for labelled probes, which the owner must choose. |
| 7 | **Fold gap**: twenty edits from pilot 5, folded as global overlays, one forward each at the plate's shape, against their delta arms; and the time an overlay costs a forward (unmeasured today). | If the gap exceeds the chance edits' spread, every claim headed for export needs a folded-form trial, at one arm a forward. |
| 8 | **Partial forward** on the 31B: paired differences from `forward_from_layer` against whole forwards. | Agreement admits it for sightings; otherwise sightings run whole. |
| 9 | **Driver**: one current frontier model, three sessions on a brief with a planted answer (pilot 4's, unnamed) and one canary brief; verdicts, forecasts against outcomes, the agent's words against the ledger, tokens and dollars. First on the stand-in for the plumbing. | If the driver cannot reach the planted answer with typed tools, the kernel moves ahead of Phase 4. If it claims success on the canary, the conclude gate tightens before any real brief. |

Pilots 1 to 8 are perhaps four to five hours of the card in all (my estimate), between census layers.

## x. Risks, what is left out, and the owner's decisions

**Risks.**

- On a healthy 31B most briefs will end *keep the base*. The feature may be mostly a machine for honest nulls. I
  think that is its value; the owner may not.
- The nulls are a choice. A null that is too quiet passes everything, as the census learned with isotropic controls.
  Two families, registered with each operation, reviewed by the validity reader.
- Generators are narrow. An edit can pass on a template family and mean little outside it. The claim says so; it does
  not fix it.
- Four standings invite sliding from "beside the grid" to "the model". Every sheet and export names its standing.
- The reader is now the instrument, and it is new code. It needs planted-failure checks and a multi-agent review that
  questions the measurement, as the house requires.
- The *held* seal rests on one CLI version's reach rules. An outside agent with a shell reads anything this account
  can.
- The card is shared with a census that takes 13.2 hours a layer, during which the server and so the Proof Room are
  down.
- R4's pin on `backend/engine_runtime.py` has not held since 12 November 2026 (the runtime reader computed the
  hashes). The kernel binds from outside either way, but the owner should know before anyone relies on the pin.
- Validity's own cost: no agent code beside the model; a fixed measurement vocabulary; about half the rows spent on
  controls; whole forwards and a fresh process for anything of record.

**Left out on purpose.** Baking into the 4-bit grid; writing an edited model file; a trained-adapter baseline and any
optimisation loop; free code in the model's process; an LLM judge; councils and staged workflows; benchmark suites;
SAEs (community ones exist for the 31B, of unverified quality); other model families; rebuilding any index for an
edited model; the lab daemon.

**The owner's decisions.**

1. Whether to build past Phase 1 after pilot 5.
2. The name.
3. Who writes generators, near misses and prompt arms; whether a setter model is ever allowed; the question bank for
   the own-answers battery (third-party text stays out of git).
4. The bars, once pilots 5 and 6 have measured the floors; the proof budget a brief.
5. Whether a stood proof may be shown on a page at all, or only a passed trial.
6. A separate account for the warden.
7. Which behaviours a brief may name.
8. The policy for exports and for publishing any edit or adapter; nothing in the documents covers it today.
9. Whether a single attribution backward is ever allowed, and under what name.
10. Card time against the census, and whether the lens may be freed for a sitting.
11. The broken R4 pin.

## Names

- **The Proof Room** (my choice). In a printing house a proof is pulled and read against copy before anything is
  printed; that is the whole design. An edit is *proved*; the sheet is the *proof sheet*; the base is the *copy*.
- **Traverse.** A surveyor's chain of sightings that must close on its starting point, with the misclosure reported.
  It names the loop and the return to the base.
- **Collation.** Comparing a changed copy with its exemplar, line by line. It names the paired plate.

New Lexicon words either way: plate, backsight, chance edit, misclosure, proof, standing (beside the grid, folded,
exported), brief.
