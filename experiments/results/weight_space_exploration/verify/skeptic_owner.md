# Sceptic's findings: fit, scope and the ask itself

Written 14 November 2026. Read-only: nothing in the repository was changed, no server started, the model not loaded.
Target: `experiments/results/weight_space_exploration/design/synthesis.md` ("Levelling"), read in full (746 lines).

What I opened myself: the synthesis; the owner judge in full and the other two judges' scores and advice; both
digests' sections on the documents, the edit paths, the lab daemon and the agent harnesses; the five designs where
a claim named them; `RESULTS.md` and all five pilot JSON files (pilots 3, 4 and 5 were written at 14:50 to 14:55,
after the synthesis at 14:43, so the synthesis does not know them); and in the repository `README.md`,
`ROADMAP.md`, `VOLUNTEERS.md`, `COMPUTE.md`, `frontend/src/screens/Lexicon.tsx`, `backend/scientist_tools.py`,
`backend/scientist_chat.py`, `backend/atlas_ops.py`, `backend/larql_vault.py`, `scripts/export_lora.py`,
`scripts/pod/common.sh`, `volunteers/kit.env`, the lab daemon's job log, and `git log`.

Marks: **[read]** a file I opened today; **[counted]** my own count or sum today; **[inferred]** my reasoning, not
a measurement.


## The short answer

The proposal is sound science wrapped round a reading of the ask that the owner was never offered. The ask says a
frontier model should "formulate and test hypotheses quickly, implement, and iterate". The proposal gives that
model, other than the owner's own session, nothing until phase 4, 72 to 89 focused days in by its own table
[counted], and then only six typed tools inside behaviours the owner has opened one by one. The synthesis says so
itself (xiii.3: "against the ask as worded") and then leaves the choice out of its fifteen decisions. Nothing here
is fatal. Five things must change before it goes to the owner: the order is put to the owner as the first
decision; the free-code question is stated as it really stood among the designs; the stop gate is moved and its
rule names the folded form; the cost to the census and to the promised publication is stated; and four of the
proposed words are changed.


## Findings

### 1. "Person first, agent in phase 4" is the right reading of the ask  —  WEAKENED, serious

- The ask names one user: "a future generation of frontier model (or even a current one)". It names no page.
- The synthesis admits the departure: "Person first or agent first (validity and the owner judge against the ask
  as worded). Settled: person first, agent in phase 4. Cost: an unattended frontier driver arrives in months, not
  days" (synthesis xiii.3, lines 538-540) [read]. Section xvii's fifteen decisions do not offer it back.
- None of the three judging lenses scored the ask's own criterion. The owner judge's six questions begin "is the
  first increment useful to a person with no agent" (`judge_owner.md:9-12`) [read]; it calls the driver design
  "the most literal answer to the ask" and ranks it last (line 22), and files against the winning design, as a
  serious flaw, "The owner is the bottleneck twice ... The ask was for a model to 'formulate and test hypotheses
  quickly'" (line 139). That flaw is carried into the synthesis: "You write or keep each brief" (decision 11); the
  harness "will not ... open a behaviour without a kept brief" (viii); "The owner is the pace" (risk 6).
- In fairness: the owner's recorded direction (a learning aid; instrument features over studies) does support a
  page. What it does not support is settling the conflict between that record and this ask without asking.
- Sums [counted from table ix, build plus house gates]: to the first decision 12 to 17 focused days; to a person's
  page 30 to 41; to any tool for a driver other than the owner's session 72 to 89.
- **Fix.** Make the order decision 0. Offer the cheap middle: after phase 1, the library in the server and three
  practice-only tools (`capture`, `draft_edit`, `level`) earning no word past *lead*; the page and the fresh
  readings after, in the owner's order. Fresh readings are needed for a claim, not for a driver to explore.

### 2. "Free code or typed tools (driver against the rest)"; no agent code beside the model is "the only arrangement under which 'no gradients' is a fact and results can be set beside the published ones"  —  REFUTED as a description, serious

- It was not one design against four. Speed runs "free code ... in a separate CPU-only kernel process" from its
  first components (`speed.md:54-55, 69`); future builds "a CPU-only kernel under bubblewrap" in increment 4
  (`future.md:326`); validity gives one in its phase 5 (`validity.md:345-350`); kernel uses the lab daemon. Four of
  five gave the driver somewhere to compute [read].
- The sentence the synthesis borrows is the science judge's, and the judge attached it to the opposite
  arrangement: "Keep driver code out of the process that holds the model. A CPU sandbox with handles (validity,
  speed and future all have it) still lets a driver write a solver, and it is the only arrangement under which
  'no gradients' is a fact and results can be set beside the published ones" (`judge_science.md:338-340`) [read].
  The synthesis keeps the justification and drops the sandbox to "unbuilt until needed" (xii, line 513). No phase
  from 0 to 6 builds it.
- The published harnesses give free Python over the weights with the network served elsewhere
  (`digest_research.md:277, 293`; Hu and others add 300 Python analyses). A tools-only driver is therefore a
  different access level from the published ones, not the one that can be "set beside" them.
- The digest's own open question: "Whether frontier drivers (Fable 5, GPT-6 class) still benefit from constrained
  typed tools; the evidence favouring them comes from Claude Haiku 4.5 and mid-2026 models"
  (`digest_research.md:334`). "Future-proofed" was in the ask; the fixed vocabulary is what the owner judge said
  ages worst (`judge_owner.md:55`).
- **Fix.** Restate settlement 2 as three choices: code beside the model; code in a CPU sandbox with typed forwards
  (the published shape, and three designs'); typed tools only. Schedule the sandbox as a phase with a trigger date,
  not a line in a table of things unbuilt.

### 3. Decision 2: "typed verbs only, for every driver but your own attended session"  —  REFUTED for backends the scientist already has, serious

- Phase 4 puts the tools "in the scientist's one registry, so Claude Code, Codex and any OpenAI-compatible endpoint
  get them at once" (vii). The Codex scientist already has a shell: "runs commands in a sandbox that writes only
  in its own work folder and has no network; it can read files elsewhere" (`backend/scientist_chat.py:103`;
  `sandbox = "workspace-write"`, `approvalPolicy "never"`, lines 843-844; `README.md:692-695`) [read]. It can run
  Python over the checkpoint's files and over `atlas_data/edits/*/payload.safetensors`. That is, in effect, the
  "sandboxed CPU kernel" of table xii, already there for one backend. The synthesis drew the consequence for
  seals (vi) and not for "no free code" or "no gradients". Whether that sandbox can open the card is not
  established.
- At `THEODOLITE_SCIENTIST_REACH=home` the scientist "has a shell, and reads, changes and runs anything in your
  home folder without asking" (`scientist_chat.py:105-106`) [read]. The level book "lives outside the scientist's
  work folder, where no driver's sandbox can write" (iii) does not hold there, nor does "export is the owner's
  command". The future design refused that reach (`future.md:381`); the synthesis dropped the refusal [read: no
  mention of the home reach anywhere in it]. ROADMAP already lists as open that such a scientist "could keep its
  own proposal through the loopback port" and that "no live model has yet been watched" against the refusals
  (`ROADMAP.md:1514-1518`).
- "Attended" is never defined. Today's five pilots were written and run on the card by agents of the owner's own
  workflow, minutes apart. If that is attended, the exception is the environment.
- **Fix.** Decision 2 names each backend: Claude Code at the folder reach (typed only, true); Codex (free code on
  the CPU, said on the harness card, or shut out of levelling); the home reach (refused); and a one-line
  definition of an attended session.

### 4. "Phase 1 is useful if the work stops there: it answers what the vault's one heard question cannot, whether a set of columns does more than matched chance sets"  —  WEAKENED, serious

- For two of the four builders the instrument already does it, per row, with the untouched row in the same pass,
  through typed tools. A push in mode `mul` is "a factor on its activation (0 silences it)", up to 32 neurons a
  reading (`backend/atlas_ops.py:50`; `scientist_tools.py:173-181`), and the exporter writes that same push as the
  column change `B = (f - 1) W_down[:, n]` (`scripts/export_lora.py:20-21`) [read]. The scientist's `logprobs`
  "reads with and without them in one pass: both, the KL from base to intervened", with `spare_bos: false` to act
  at the start mark as a weight does (`scientist_tools.py:363-370, 187-189`); `steer` takes up to 12 `candidates`
  (sets of pushes) against one `baseline` in one batch (lines 288-292) [read].
- So `silence` and `scale` on a set of up to 32 columns against chance sets can be read today with no new code.
  Pilot 4 agrees where it matters: at layer 44 the output-side delta and the input-side form correlate at 0.9997
  and 0.9999 (`pilot4_cancellation.json`, recipe and prose, layer 44) [read].
- What phase 1 really adds is narrower: `set`, `remove_direction`, row and low-rank terms, sets above 32, many
  chance edits a reading, a saved hashed edit and the book.
- The atlas product of ii.2 ("a causal reading that needs no dose" for the 39 layers without one) is a `mul` on a
  neuron's own stored passages against matched neurons. It needs none of the weight kernel, and it is in no phase
  of table ix.
- If stopped after phase 1, about 2,150 new lines by the synthesis's own table (500 + 700 + 150 + 200 + 600)
  [counted] stay in `backend/` and `scripts/` with no page and no Lexicon entry, driven only by the owner's
  session: a study tool, which the recorded direction says not to add.
- **Fix.** Say what exists. Split ii.2 out as its own small item on the existing push machinery. Run pilot H now,
  on the existing scientist with `logprobs` and `steer`, on the known answer unnamed: it costs no build and tests
  the ask's real question, whether a current driver does this work on this model.

### 5. The owner's first decision: placed after phase 1, on pilot D, "at least 5 of the 20 column sets ... beat every chance edit of both families and exceed five times the floor row"  —  WEAKENED, serious

- **Placement.** Table ix puts pilot D in phase 0 ("Proves: whether sets and directions are heard"), yet the
  decision sits after phase 1. If D runs in phase 0 the owner can stop 8 to 11 days sooner. If D needs the kernel
  (it needs both chance families matched on firing, direction removal through `w_post`, `delivered`, floor rows and
  64 generated items, which is most of the kernel), then phase 0 does not prove audibility and its 3 to 4 days are
  for six repairs, a lock and eight other pilots. Either way risk 1's "two weeks of work" is 12 to 17 days
  [counted]. And a registered gate would rest on inline pilot code that no review has read, against the
  synthesis's own risk 5.
- **The rule.** It does not name the form, and pilot D is a delta-form pilot. Pilots 4 and 5, run after the
  synthesis, show the delta form can clear "five times the floor" where the folded form does not. At layer 20,
  which is one of pilot D's four layers, silencing neuron 5316 on the recipe text read 0.70 nats a position as an
  output-side delta, 0.52 on the input side and 0.006 folded, with that text's floor row at 0.0034: 200 times the
  floor in one form, under twice in another (`pilot4_cancellation.json`, `recipe_layer_20`, `worst_five[3]`;
  `pilot3_writing_and_capture.json`, `recipe_layer_20`) [read]. The output-side and input-side effects correlate
  at 0.02 at layer 5 and 0.60 at layer 20 on that text, 0.9997 at layer 44. No null row of 31 tipped at any layer
  (`pilot5_knife_edge.json`), so chance edits may not tip where real ones do [inferred].
- This also contradicts table i's "differs by about 0.00045 nats a position ... whatever the edit's size" below
  layer 44. That is another sceptic's to settle; here it matters because three months hang on this one rule.
- **Fix.** Put the decision at the end of phase 0 or say plainly that D runs on the reviewed kernel. Write the
  rule in the folded form, or require the folded twin for every set counted, and drop or flag layer 20.

### 6. Effort: "focused days of one person"; "to the end of phase 4 is roughly three to four months"  —  WEAKENED, serious

- The arithmetic holds: 72 to 89 focused days is 3.3 to 4.1 months at five a week [counted].
- The unit does not fit this repository. All 354 commits since 14 October carry a Claude co-author line; 22,141
  lines were added under `backend`, `scripts` and `frontend/src` on 28 October alone, 4,671 on 12 November, 2,877
  on 13 November (`git log --numstat`) [counted]. Code here is written by sessions at thousands of lines a day.
  What one person supplies is direction, review and decisions, and the table counts none of that as such: 15
  decisions, a brief and generators for each behaviour, a measured bar for each battery, two registrations in
  phase 0, a validity review for each phase.
- The owner already has five decisions unanswered from this morning (seven commits unpushed, how layer 36 is
  shown, layer 34's bundle, replies to volunteers, what the home card takes next) [read: the handoff note of 14
  November 19:00Z].
- So build days are likely overstated and the owner's hours, the binding thing, are not stated at all.
- **Fix.** Restate each phase in owner-hours of decision and review, card-hours, and driver dollars. Build days
  can stay as a footnote.

### 7. Cost to the atlas and census: "sittings between layers" (decision 13; risk 8)  —  WEAKENED, serious

- The card cost is small and I could not fault it: about 2 to 2.5 hours of pilots against a 13.2-hour layer.
- The cost that is never stated is the owner's. Seven layers are open (32, 33, 35, 37, 38, 39, 52), seven more
  are shut until trialled (53 to 59), two volunteer runs are due for intake, and the README still says eleven
  layers [read: handoff; `volunteers/kit.env`].
- A public promise stands: "When the atlas and its data are published with the project ..."; "Open data: every
  answer of every layer ... to be published with the project"; "you name the strongest neurons of your layer in
  the published atlas" (`VOLUNTEERS.md:41, 50-54, 287`) [read]. The proposal does not say whether three to four
  months of focused time come before or after that publication.
- While the home card takes a layer the server is down, and from phase 2 levelling lives in the server; the lane
  for that case is in the unbuilt table.
- **Fix.** One paragraph: what waits (which layers, the publication, the pattern tests on the page) and for how
  long, under each of decision 1's branches.

### 8. Names: "the words are free in the repository and carry no claim"; "four new words"  —  REFUTED in part, serious for two words, minor for the rest

- **Levelling, level book.** Free as strings (no hit in source or documents) [read, grep]. "Level" is not free as
  a word: the census map's ranks are "levels" (`GET /census/levels`, `backend/atlas.py:1117`;
  `frontend/src/screens/atlas/map.ts:942-954`), and the pattern test has near-miss "levels"
  (`backend/pattern_tests.py:150-164`). The synthesis itself uses "level" three other ways ("the gradient level",
  "a declared level", "by access level"). In a tool whose first Lexicon entry is "layer", a scientist's verb
  `level` invites "level 44". And the gloss does carry a claim: "a run that does not close on its start shows its
  error". By the synthesis's own section i the zero row closes exactly and "cannot see that a real delta is
  rounded twice". The closing check is blind to the error that matters.
- **Chance edit.** The house word for this is "control": the Lexicon has an entry "a matched control" (line 1470)
  and says of the census "along random directions, as controls" (line 1174); "controls" is in the documents
  177 times [counted]. The synthesis's own code reuses `census.control_dirs`. And "chance" in the Lexicon means
  sampling: "Drawn by chance, each word-piece is picked at random" (line 1002), "the chance in a drawn answer"
  (line 1048). A "chance edit" reads as an edit drawn by the sampler. Use "control edit".
- **Held.** The verdict *held* means an edit stood on fresh items. "Held back" is the label gate's word for a
  label not shown (144 uses in the repository, `HeldBack.tsx`) [counted]. The synthesis also writes "held items"
  and "a file-held set" for items kept from the driver. Three senses, one of them opposite in tone.
- **Seal.** Taken: the neuron folio's seal is the label gate's mark, and "a held-back label's seal reads 'held
  back'" (`frontend/src/screens/NeuronFolio.tsx:1704, 1734`) [read]. Decision 4 is headed "Seals".
- **An edit.** The Lexicon's entry says "An edit stays until you undo it" and the section lede "an edit until you
  undo it" (lines 1579-1580, 1425). The thesis is "An edit is ... never a state of the server", and phase 2
  puts both on the vault page, where a levelling "refuses to read while any overlay ... is non-empty". The names
  section does not mention that its central noun is an existing entry with the contrary meaning.
- **The count.** Besides the four, these would be seen on a page and are terms of art: held, void, habitat,
  battery, sheet, seal, floor row, delivered, practice reading, fresh reading, trial of record. "Trust card"
  (vii) is used as if it existed; it has no hit in the repository [read, grep].
- **Brief** is free in the documents and the Lexicon, and fits how the owner already uses the word. It stands.

### 9. The repairs list (xi) and "none of these touches what writes census answers"  —  WEAKENED, minor

- The six faults are real where I checked: the R4 pin (`sha256 backend/engine_runtime.py` is `f1a0b4c3...`
  against the pinned `69eed1b4...`, changed by `a766058` and `960b9ab` on 12 November); 35 tool names in
  `scientist_tools.py` against the README's 30 [counted].
- The tool count is also wrong on a page: the Lexicon says "30 tools" (line 1411) and `ARCHITECTURE.md:43` says
  30. The repair names only the README.
- A seventh fault the synthesis found and did not list: the vault's UPDATE adds a stream-space direction to the
  raw column without the post-feed-forward norm's weight (`backend/larql_vault.py:513-523` against
  `backend/atlas_ops.py:186-190`) [read]. It is a default-visible act of the vault page. It is deferred to
  decision 15, "later; never".
- The lock: the kit's own wrapper already holds one (`hold_card`, `flock` on `logs/gpu.lock`,
  `scripts/pod/common.sh:104-112`; `scripts/volunteer/census.sh:294`) [read], and home census runs start from the
  kit-4 worktree at the tagged commit. A lock the census wrappers must newly take is either a new kit while two
  volunteer layers run, or it does not reach home runs. Have the server and the new script take the lock that
  exists, and take it outside the kit's files for home runs.

### 10. Decision 9: "What may leave the machine ... manifests only until you write a policy; none exists today"  —  WEAKENED, minor

- A practice exists and is public. `VOLUNTEERS.md:304-331` tells anyone to export a neuron as a llama.cpp control
  vector and load it in their own frontend; `scripts/export_lora.py --vault` already writes a vault edit as an
  adapter (line 10); a tag `steering-vectors-1` exists [read].
- So the open question is not what leaves the owner's disk. It is what the public code lets every reader do.
  `remove_direction` in phase 1 is directional ablation written into weights, which is how refusals are removed,
  and the kit's audience runs local models. "No recipe or brief for stripping safeguards ... is shipped" (viii)
  does not cover a builder that is the recipe. The digest records that such builds of this 31B already exist, so
  the added reach is small [literature]; it is still the owner's call and it is not on the list.

### 11. "From phase 1, attended ... That is the trust the lab daemon had for the study's 314 jobs, and nothing new is exposed"  —  STANDS as a fact; it cuts both ways

- The count is right: 314 jobs, 282 ok, median 71.3 s, 27,453 s in all
  (`experiments/results/labd_exec_log_2026-10-16.jsonl`) [counted today].
- For "too cautious": that is about 27 free-code jobs an hour, by agents, on this model, a month ago. And today:
  pilot 3's result landed at 14:50:41, pilot 4 was written by 14:52:18 and done at 14:53:35, pilot 5 written by
  14:54:58 and done at 14:55:47 (file times in the pilot folder) [read]. Three rounds of guess, script and
  measurement in about seven minutes, each with an 11-second load, and the guesses (three forms of one edit; null
  nudges) are not expressible in the six typed tools.
- For "not cautious enough": the day after those 314 jobs the study carried "51 claims ... qualified or [with] a
  caution" (`experiments/REPORT.md:31, 412`) [read]. From the end of phase 1 until fresh draws exist at the end of
  phase 3, 42 to 54 focused days by table ix [counted], the only driver is the one with no mechanical guard, and
  the science judge's line applies:
  rules that are not mechanisms "did not hold for agents with free code, attended or not"
  (`judge_science.md:225`).
- What I tried: I looked for a guard in phases 1 and 2 that binds the owner's session and found only the word
  *lead*. That is honest, and it is all there is.

### 12. The arithmetic of the ranking and the house citations  —  STANDS

- 22, 20.5, 19, 18.5 and 17 are the judges' scores summed [counted]. `VOLUNTEERS.md:338` says "a lead, not a
  proof" and line 51 "We found none published" [read]. The Codex reach is as the synthesis says.


## The fifteen decisions

| Kind | Which | Why |
|---|---|---|
| Real, needed now | 1, 2, 3, 5, 13 | They shape phase 0 and 1. But 1 is misplaced and its rule unformed (finding 5), 2 is not deliverable as worded (finding 3), and 13 offers no choices |
| Real, half now | 7, 11 | Pilot H spends dollars in phase 0; "none on safeguards or triggers" is a standing policy to set once, not a rule for each brief |
| Real, months early | 4, 8, 9, 14 | Phases 3 to 5. Asking now pads the list and invites answers given without the evidence |
| A non-decision | 6, 12, 15 | "None for now", "as section xvi", "later" |
| Not about this feature | 10 | The R4 pin is housekeeping; it belongs in the repairs |

**The real one missing:** who it is for first. Agent tools before the page, as the ask is worded, or the page
before the agent, as the panel preferred (finding 1).

Also missing: whether to start now at all, and what waits (finding 7); what "attended" means and which backends
may drive (finding 3); whether the builders and the export are pushed to the public repository (finding 10).


## What I could not check

- Whether the Codex sandbox can open the card's devices.
- What a cut-down agent increment after phase 1 would cost; my "about a week" is a guess from the synthesis's own
  scale, not an estimate I can defend.
- Whether the knife-edge readings of pilots 3 to 5 are the edit's form or the text; one text, one run.
- How uneven the sixty `w_post` weights are, so how far the vault's UPDATE is from the direction it names.
- The owner's own view of the order. That is the point of decision 0.
