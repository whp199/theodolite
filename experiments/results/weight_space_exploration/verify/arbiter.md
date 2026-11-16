# The arbiter's ruling on "Levelling"

Written 14 November 2026. Read-only: nothing in the repository was edited, no server was started, the model was not
loaded and the card was not touched. What I did myself is listed in section 2; everything else is a skeptic's or the
proposal's and is marked so.

Marks: **[checked]** I went to the evidence today; **[arithmetic]** my own sum; **[skeptic]** taken from a skeptic's
report and not re-checked; **[estimate]** my rough guess.


## 1. The verdict after the attack

The feature is feasible and still worth proposing. The mechanism the whole thing rests on survived and got better
under attack: an edit can ride one row of a batch with the unedited model in another row, rows do not leak into each
other, writing under such hooks works, and for the commonest edit (scaling or silencing columns) there is a per-row
form that equals the folded weight to every printed digit, which the repository already has as the `mul` push. No
skeptic found anything fatal. Forty-one objections (the seven lists merged) are sustained and seven overruled; almost
all of the sustained ones are repairable on paper before any code is written.

The proposal does not survive as written, on four counts. (1) Its kernel is five pilots out of date: the hook, the
floor, the bar and the cost table must be rewritten from the pilot note of 21:03Z. (2) Its guards do not yet guard:
"tops every chance edit" only tests whether a candidate was drawn at random, and no candidate an agent picks is. (3)
It answers an ask for a frontier model's environment with a person's page first and six typed tools months later,
mis-states what its own panel advised about free code, and claims a containment it cannot deliver for a backend it
already has. (4) Its ceiling is wrong in both directions: a neuron-level repair of a real fault on this very model
was published in June and missed, and the instrument as designed could not see it. The amended shape: repairs and
containment first; a kernel with the corrected arithmetic and reading rules; then the agent's door (free code off the
card in a sandbox the instrument owns, the model reached only through gates that add the controls), earning no word;
then the claim gate; then the page, in the order the owner chooses.


## 2. What I checked myself

- Pilot JSON and logs 3, 6, 7 in full, 4 and 5 through the skeptics' quotations. Input-side equals folded in 7 of 7
  cases at 2 and 32 rows; the output-side term does not (0.02274 against 0.72581; 0.70255 against 0.52337; 0.20649
  against 0.00829). One folded edit reads 0.0091, 0.0127, 0.0075, 0.0085, 0.7258 at 2, 3, 4, 6 to 12 and 16 to 48
  rows on the raw recipe; on the two chat-formatted texts no reading differed over fivefold. One-row capture agrees
  at layer 44 (0.9998, 0.9999) and not at layers 5 and 20 on the raw recipe (0.022, 0.317).
- `backend/atlas_ops.py:50, 369-383` (the `mul` push: a per-row factor on the activation before `down_proj`, at
  most 32), `:396` (per-row head scaling exists too), `:503-517` (the writer takes a sampler and a seed);
  `backend/engine_runtime.py:352-377` (folded form: clone, write column, one product) and `:1234-1277` (quiet masks).
- The sixty post-feed-forward norm weights, read from the checkpoint file on the CPU. Every layer from 13 to 58 has
  entries under 1% of its median: 58 at layer 20, 317 at 36, 260 at 44, 246 at 45, 196 at 55. For an even unit
  direction the division by `w_post` lengthens it 98-fold at layer 36, 180 at 44, 198 at 45 and 55, 514 at layer 5,
  with 99% or more of the result in those entries. 165 to 180 of the 181 persistent quiet dims are among the small
  entries at layers 36, 44, 45, 55. `README.md:334-337` records the channel; `backend/census.py:1567-1591` builds
  its controls through `w_post`.
- The E6 script and stored result: a folded vault DELETE, one right-padded batch of 12 raw-text contexts, the
  model's own logits (differences sit on eighths). The 'she' ten move the male contexts by about +0.9 and the
  Gemma-as-human prompts by more than the target.
- `backend/scientist_tools.py`: 35 tools; `logprobs` and `steer` already take `mul` pushes with the base in the
  same pass; `export_control_vector` writes over a file of the same name. `backend/mcp_scientist.py:1569`
  (`query_larql_vault` edits the held model). `backend/scientist_chat.py:99-107, 560-572, 843` (reach).
- No file under `backend/` or `scripts/` sets `TORCHINDUCTOR_CACHE_DIR`; `/tmp/torchinductor_<user>` holds 135 files,
  none dated today. I did not run the Codex sandbox probe.
- `NOTICE:7` ("does NOT redistribute any model weights") against eleven tracked `.gguf` files, the adapters'
  commit on `origin/main`. `.gitignore` ignores `atlas_data/` and no `*.safetensors` or `*.gguf` elsewhere.
- The house's registered rules: "how many neurons beat every control" is listed under "Reported, gating nothing" in
  both rule files; the bar rule is `sqrt(F x M)`; gate N3 fails a reading under half covered.
- The census baseline rows end `length` at 48 pieces; every instrument question is asked with thinking off.
- The saved copy of arXiv 2606.13705: gemma-4-31B-it, thinking on, loops on long enumerations; 1,062 layer-36
  neurons zeroed; held-out loops 13 of 96 to 3 of 96; neurons ranked by gradient times activation; an edited
  checkpoint is public. No file of the digests, the designs or the repository's references names it.
- The editor's prompt in arXiv 2611.09382 (saved copy): a persistent Python kernel with no network and no GPU, the
  model's files read-only, `save_edit` refusing the final norm, the embedding and whole-tensor rescales, and a tool
  of 50 forward passes that runs original, edited or both. "No gradients" and "do not simulate the network in the
  kernel" are rules stated to the editor.
- The science judge's sentence (`judge_science.md:336-340`): it was written of a CPU sandbox with handles. Three
  designs scheduled such a sandbox (`validity.md:345`, `speed.md:54-55, 69`, `future.md:326`).
- Arithmetic: the components table sums to 2,150 lines; a fresh reading's item forwards alone are 2.4 minutes and
  a trial's 14; screening every neuron by forwards is 14 hours a behaviour.


## 3. Faults found on the way that do not wait for this feature

These exist today, whatever is decided. They are the owner's to hear first.

1. **The compiled unpacker's cache sits in `/tmp`** and nothing moves it [checked]. The safety skeptic measured that
   Codex's default sandbox can write in `/tmp` and that cache entries there open for writing [skeptic; exploit not
   run]. Every process that installs the compiled unpacker loads from that folder: the server, a census run, and the
   proposed script. One environment variable closes it.
2. **`NOTICE` says no weights are redistributed while adapters holding Gemma's columns are tracked and pushed**
   [checked]. The two GGUF writers put no licence or attribution key in the file [skeptic].
3. **Inside Codex's sandbox the Hugging Face token file, `~/.codex/auth.json` and SSH keys are readable** [skeptic].
4. **The vault's UPDATE adds a stream direction to the raw column without the norm's weight** (the proposal found
   it and deferred it); with the quiet channel measured above, that is a default-visible fault of the vault page.
5. `export_control_vector` writes over any export by name [checked].


## 4. Rulings

### Sustained (the proposal must change)

| # | Objection | From | Severity | Amendment |
|---|---|---|---|---|
| 1 | One output-side hook for every kind of edit | code, numerics, completeness | serious | Each kind has its own per-row form. Column scale and silence: a factor on the row's activation before `down_proj` (the `mul` arithmetic, start mark not spared, vectorised past the cap of 32); exact for 0 and powers of two, other factors round one scalar and say so. Column set and direction removal: the old column's activation zeroed on the input side, the new content added from the row's own activation, confirmed folded at the reading's own batch shape. Row edits on `gate_proj` and `up_proj`: the edited coordinate recomputed, never added to. Low-rank terms: an output-side add, confirmed folded, no word until a folded path exists. The float32-sum hook and the row form are marked unmeasured |
| 2 | The floor is a constant; folded is one reading; a paired difference is exact | code, numerics, statistics, completeness | serious | The pilot note's four rules become design. Items are in the model's own format and each is gated on the base row's own log-probability. Batch shape is part of a reading's identity. Any reading that prints more than raw numbers is taken in two batch classes (12 rows or fewer; 16 or more) and disagreement marks the item unstable. Median over positions and the count above a nat are printed beside the mean. A fresh reading that would halve its rows is void. The constant 0.00045 is struck from section i |
| 3 | The one-row capture serves practice readings | numerics, code | serious | Allowed only where shown (layer 44 so far). Below about layer 40 every reading is priced at whole forwards until a chat-formatted repeat says otherwise: a practice reading is about 13 s, not 5 |
| 4 | The times-3 floor row and "five times the floor" | numerics, statistics | serious | Several null rows a forward with factors near 1; the floor is their largest, on each item, in the metric's own units, log-odds included. The bar follows the house rule: `sqrt(F x M)` with the brief's least effect as M, dropped where M is under 4F. The 0.01-dose row is dropped |
| 5 | `remove_direction` waits on pilot 0 | code, numerics | serious | Pilot 0's condition is already met. The masked form is written now: `u` is projected off the layer's quiet mask before the division, every control direction goes through the same projection, a `u` with more than a stated share in the mask is refused, and the raw change's length is printed beside the stream-side one. No pilot that uses it is registered before this is in the text |
| 6 | Sizes, and pilot D in phase 0 | code, owner | serious | D is split: a first look in phase 0 on inline pilot code, the registered D on the reviewed kernel at a tagged commit. Checks are planned at the house ratio (about 2,250 lines, phase 1 about 3,800 [skeptic]). Effort is restated in owner-hours, card-hours and dollars |
| 7 | Pilot B's stop rule compares four differences at once | code | serious | B first re-runs E6's own reading (vault DELETE, batch of 12, the model's logits) on today's code, then changes one thing at a time. The tolerance comes from the eighth-step. The stop rule attaches to the like-for-like arm only |
| 8 | Costs and memory | numerics | serious | The cost table sizes the batteries, the written gate and the second batch class; says which form a trial's controls take; drops "2 to 4 minutes" and "under an hour" until timed (about 10 to 13 minutes and 60 to 75 [skeptic's arithmetic; mine agrees on the item forwards]). Memory is re-measured with the peak counter reset a shape; the writer's 0.8 MB a row a piece of context goes into the model |
| 9 | Dose ladders | numerics | minor | Dyadic doses; for a power curve shrink the set, not the dose |
| 10 | arXiv 2606.13705 missed | literature | serious | Cited. It becomes a pilot before D and the outside known answer: the paper's prompts and sampling on the W4A16 checkpoint with thinking on, base first; if the fault is there, the published set silenced on the input side. Named in section xvi |
| 11 | The ceiling sentences | literature | serious | Rewritten as in section 7 below: "few points or nothing", "small or damaged", "training beat editing", "three identical searches", "at the model card's settings" and "will not add abilities" all change |
| 12 | The fast path cannot see what the literature found | literature, completeness | serious | A long sampled written reading (rows are seeds times base, edit and control; loop and cut-off detectors) is a reading class of the kernel, priced from a ten-minute pilot of twelve thinking-on answers written to their natural stop |
| 13 | The gate: sets of 64, random or most-active columns, an unnamed metric, one class of edit | literature, statistics, completeness, owner | serious | See "the gate" under section 9. A set-size ladder (1, 10, 64, 300, 1,000); behaviour-selected neurons; a metric the sets were not built from; sham-target sets as the comparison; chat items in two batch classes; two extra arms that ride existing machinery; the stop worded by class of edit |
| 14 | "Gradients: none, because the published harnesses forbid one" | literature | serious | Pilot F stays in phase 0 and the gradient level becomes a real decision before phase 1. The ceiling says plainly that the localiser behind every published Gemma 4 neuron-level edit is unavailable without it |
| 15 | Typed tools are "the only arrangement" | literature, owner, completeness | serious | Struck. See section 5 |
| 16 | The absence claims | literature | serious | Each is written with its qualifier and its nearest neighbour named |
| 17 | The p of "tops every chance edit" | statistics | serious | A third family wherever the site came from a declared rule: the same rule on another target. For a free pick the sheet prints "selection not matched, K read in practice" and no p. The comparison is worded as the census words it: differs from a random edit; does not say what it does |
| 18 | The family is printed and never applied | statistics | serious | The owner fixes m, the fresh readings a brief, before the first; each reading carries 20m - 1 control edits a family (39 supports two readings, 99 five). Practice readings earn no named word |
| 19 | Fresh draws guard item noise only | statistics | serious | Whole frames are held out, as the census holds out question halves; a fresh item seen in practice is refused; intervals are by frame; a file-backed brief takes the file's name and limits; nothing reaches a page without held-out frames |
| 20 | Batteries without bars; the known answer would read "works but harms" | statistics | serious | One registered bar rule on the house's pattern before phase 3, with the planted harm as the near miss; the batteries that gate are named; battery text is chat-formatted or gated; the known answer's near-miss set and expected verdict are fixed before it runs |
| 21 | The strata table at 48 pieces is the length cap | statistics, completeness | serious | Strata only in long written readings, with a cap under which nine in ten base answers finish; counts under 20 print "too few to read"; each tier prints the least gain it could have seen |
| 22 | The other-site family has no members for the loudest candidates | statistics | serious | Matched on measured `delivered` an item (now defined in the text); a reading with fewer than C sites inside the tolerance is void and says so |
| 23 | The rule and the words against the house's own | statistics | serious | Gate on a fixed level with enough control edits plus replication across frame halves. "Tops every one" is reported and gates nothing. "Lead" is one grade of evidence: a fresh reading that passes. The verdict "held" is dropped |
| 24 | The self-test's negative control | statistics | minor | 200 planted nulls with a band on the count that pass; the seven 'she' neurons the 'he' list does not share as the negative control that matters |
| 25 | "Passed" has no pass rules | statistics | serious | A trial reads in both batch classes; its bars come from a calibration reading, hashed; its sheet prints every earlier trial on the brief |
| 26 | Person first, settled unasked | owner, completeness | serious | Decision 0, with the middle offered. See section 5 |
| 27 | Decision 2 is not deliverable for Codex or the home reach | owner, safety, completeness, code | serious | Each backend and each reach is named. Refused at the home reach. "Attended" is defined. See section 5 |
| 28 | Phase 1 overlaps what the scientist can already do | owner | serious | The text says what exists (`mul` pushes in `logprobs` and `steer`, base in the same pass, at most 32). The atlas product of section ii item 2 is split out as its own small item on that machinery, and waits on the count of stored passages that pass the log-probability gate. Pilot H runs now, on the existing tools |
| 29 | What waits, and for how long | owner | serious | One paragraph: the open layers, two intakes, the unpushed commits and the publication promised to volunteers, under each branch |
| 30 | The words | owner | serious | "Control edit" for chance edit. No verdict "held" and no "seal". The tool verb is not "level"; access tiers are called reach, the house's own word. The Lexicon's "An edit stays until you undo it" is addressed. The real count of new page words is stated |
| 31 | Containment claims stated flat | safety | serious | Every bullet of section viii says at which reach it holds. The compile cache is moved and the real Codex probed before any agent session. `assert_base` (no hook on any module, no overlay on any linear) runs at the end of every levelling lock hold and at the start of every other one; a failure voids the readings it names. No hook or overlay crosses a lock hold; a reading that sees a foreign overlay is void |
| 32 | Export, licence, payloads | safety, owner | serious | Phase-5 exports go to a folder no tool writes; `export_control_vector` refuses to write over. A repair adds licence and attribution keys to both existing writers and corrects `NOTICE`. The manifest holds the recipe wherever a term can be rebuilt from the public base and marks each term rebuilt or carried; ignore rules for `*.safetensors` and `*.gguf`; registration folders hold the manifest and the payload's hash only |
| 33 | Planted faults leak; file-held secrets | safety, code, completeness | serious | Plants, canary flags and per-item rows live in server memory only. A planted session gets a filtered tool list (the gates and index-only reads) named on the harness card, and is reported at two reaches: with a healthy reference and without. The book tool's driver view is specified. The kept-back fold is enforced at the route |
| 34 | Budgets | safety, completeness | serious | The lock is budgeted a chat across the whole registry; forward requests are counted beside GPU-seconds so a session can be quoted at the published budget; the edit gate caps terms, columns, bytes and relative change; `blind_test`'s endpoint profile is off in such sessions; an unattended runner has a wall-clock cap and a kill |
| 35 | Lock holds; the second agent surface | code | minor | The overlay check is per hold. The safety line is worded as true of the new tools and names `query_larql_vault` |
| 36 | Smaller labels | code, owner | minor | The 9.5 s load is with the lens on a warm cache; field notes and the Thread are patterns, not reuse; the lock builds on the existing `flock` on `logs/gpu.lock`; the UPDATE direction fault and the Lexicon's tool count join the repairs |
| 37 | The misuse stance | safety, owner | minor | Called what it is, a policy on what the project ships. The implant flag is a label, and the vault's own example is its first hit. The public posture of the builders becomes an owner's decision |
| 38 | "Future-proofed" is never defined | completeness | serious | Half a page and one test. See section 5 |
| 39 | Thinking-mode writing not costed | completeness | serious | The ten-minute pilot of row 12 |
| 40 | Directions only from a difference of means | completeness | minor | An edit term may be an array handed in as a hashed file; a direction from an outside dictionary is typed "not checked on this checkpoint" |
| 41 | The 12B on disk | completeness | minor | One no-card line in phase 0: build it on the CPU, compare one forward with llama.cpp, decide after. Not on the critical path |

### Overruled

| Objection | From | Why |
|---|---|---|
| Every claiming reading at three batch sizes from different classes | numerics | Two classes are what today's evidence asks for: the break that matters sat between 12 and 16 rows, and on chat-formatted items nothing differed over fivefold. A third class waits for the map of where classes change. Cost: an item stable across two classes and not a third goes unseen until then |
| More than one agent session before anything reaches a page, because three identical searches gave +29, +5, +27 | statistics | The searches were not identical (one had another objective for 46 of 50 requests [checked]) and were on a 0.8B. And a page claim is about an edit, re-read by the harness on fresh items; the session that found it is not the evidence. Session counts matter for claims about drivers, which is phase 6 |
| An unembedding-row arm in the gate | completeness | It needs new machinery (the reader takes its logits outside any module), and the published harness this would stand beside refuses embedding and output-layer edits at `save_edit` [checked]. The stop is worded so that it does not speak about them |
| Serve the plant server-wide under the stale banner | completeness | It makes an edit a state of the server, which is the one thing the design forbids and the invariant of row 31 checks. The filtered tool list is chosen instead. Cost: a planted session has fewer tools than a real one |
| Phase 1 adds little because the `mul` tools exist | owner (as an implication) | What exists has no matched controls, no sets over 32, no hashed edit, no book, one next-piece read-out. The guards are the addition. The amendment is to build on the `mul` arithmetic, not to skip the kernel |
| Quote a gain only under the setting the brief fixes | literature (the remedy) | Both are printed. A brief may fix thinking on; the sheet still shows the base under each setting and says so when the fault is absent with thinking off |
| Put the first decision at the end of phase 0 | owner | A registered gate cannot rest on unreviewed inline code. Phase 0 gives a first look that can say "go on" or "nothing at all"; the registered decision follows the kernel |


## 5. The central dispute: when and how a frontier model drives this

**What the evidence says.**

- The published harnesses give the editor free Python over read-only weights in a process with no card and no
  network, and reach the model only through a counted tool that runs original and edited together [checked, the
  prompt of 2611.09382]. Only IWIBench is tools-only, and there every success went to a tensor class this proposal
  leaves unbuilt [skeptic].
- Three of the five designs scheduled that shape and the science judge advised it. The synthesis kept the judge's
  justification and dropped the sandbox [checked].
- Today seven pilot scripts were written and run on the 31B in 72 minutes (19:50Z to 21:02Z), with free code beside
  the model and no harness, by every sign by a current model of the owner's workflow. They found a fault in the
  measurement that five designs and three judges had missed. None of their comparisons (three forms of one edit, 31
  null rescales, ten batch sizes) can be said in the six typed tools.
- The day after the study's 314 free-code jobs the report carried 51 qualified claims. Free code finds things fast
  and says too much about them.
- One existing backend already has a shell that reads the weights, so "typed tools only" is not true as worded.

**The ruling.** The typed-or-free question is the wrong cut. The line that matters is between exploring and
claiming.

1. **Free code, off the card, in a sandbox the instrument owns.** A CPU process under `bwrap`: weights and the atlas
   index mapped read-only through one plain reader, one scratch folder, no network, no card, its own `/tmp`, no home
   folder, a memory and a time cap. It is reached through one tool that runs a cell, so Claude Code at the folder
   reach, an endpoint model and Codex all get the same thing and the boundary does not depend on a vendor's sandbox
   that changes with each release. Here a driver writes any analysis, solver or direction and hands arrays in as
   hashed files.
2. **The model is reached only through gates the driver cannot write.** Capture, a paired reading, a written
   reading. The gate, not the driver, adds the base row, the null rows and the control edits, and writes the book.
   A reading at this stage earns a rank and no word.
3. **A claim is the harness's, not the driver's.** Fresh items on held-out frames, drawn after the edit is hashed;
   a trial of record in a fresh process. Only these earn "lead" and "passed".
4. **Free code on the card is the owner's attended session and nobody else's.** Attended means: a session the owner
   opened in the repository under the ordinary permission system, whose commands the owner can see and stop. Its
   readings count only when they went through the library's reader; anything else is a pilot note.
5. **Refused:** every levelling tool at the home reach; Codex until the real sandbox has been probed and the cache
   moved.

**When.** The agent's door comes straight after the kernel, before fresh readings and before the page. Five of the
six proposed tools need only the kernel. A current model can drive from the first day of the kernel through the
owner's session, and unattended as soon as the sandbox passes its probe. The order past that is the owner's
(decision 0).

**What "future-proofed" means here**, concretely:

- The contract is small and names no driver: one cell tool, three gates, one verdict, a book. Anything that speaks
  MCP or runs a command drives it; the stdio server already serves any MCP client.
- The edit vocabulary is open: a term is an additive difference on a named tensor, and an array may be handed in.
  Builders are conveniences. A driver may propose a new operation with its check, through the proposal pattern.
- What is fixed is only what a stronger driver must not be able to write: the controls, the fresh draw, the book.
- Reach is declared on every result and never pooled: behaviour; activations; weights; the atlas; one attribution
  backward; free code on the card.
- The plan file, the book entry and the manifest carry version numbers; the driver's transcript is copied into the
  book's folder when a session closes.
- Two tests, at the end of phase 1. Cold driver: two driver families, each given one page and the command line, are
  asked for the known answer unnamed; pass if both reach it with no change to the code. Rebuild: a fresh session
  reproduces a reading from a description of 200 words or fewer.

**What the decision costs.** About a week more build before the first unattended session than tools alone
[estimate]. The sandbox is new trust-boundary code that must be probed on every release of the tools it nests. "No
gradients" and "the network is not simulated in the kernel" become rules with caps and logged cells, as in the
published harness, not facts. Results sit beside the kernel-and-forward-tool harnesses and not beside IWIBench
unless a tools-only preset is run as its own reach. Codex waits. A plant held in memory ends when the server
restarts. And the person's page arrives about a month later than in the proposal [estimate].

**Where two skeptics pulled opposite ways.** "Too cautious about agents" (owner, completeness, literature) against
"not contained enough" (safety, code): both are right, about different things. The driver gets more freedom to
compute and less to claim, and the containment repairs come before either.


## 6. The amended first increment

**Phase 0 and the kernel, with one look between them.**

Phase 0 builds no feature code. It mends the faults of section 3 and the proposal's six repairs; it probes the reach
of the Claude Code and Codex in use; and it runs these on the card, between census layers:

- the batch-class map on chat-formatted items, with sets and directions, at 2, 12, 16 and 32 rows, in a fresh
  process and with the lens loaded; and the count of stored passages that pass the base log-probability gate;
- pilot B like for like, then one change at a time (registered);
- the first look at audibility: sets of 1, 10, 64, 300 and 1,000 columns, random and selected, scaled by 0 and 2 on
  the input side, against other-site and sham-target sets;
- twelve thinking-on answers written to their natural stop: pieces, seconds, GiB;
- the loop pilot: the published prompts and sampling, base only, 64 generations; if the fault is on this
  checkpoint, the published 1,062 neurons silenced;
- a backward pass under the compiled unpacker; memory with the lens, the peak reset a shape;
- pilot H now: a current driver on the existing scientist's tools (Claude Code, folder reach) on the known answer,
  unnamed.

Then the kernel, by command line: the edit as a hashed value with the per-kind forms of row 1 and the masked
direction removal of row 5; one paired reading under the rules of rows 2 to 4, 17 and 22; a long written reading;
the book; a reader of the weights that needs no card; checks at the house ratio. It ends with the registered
audibility gate on a tagged commit and the two tests of section 5.


## 7. The honest ceiling, corrected

- **Single neurons mostly sit at the floor.** On chat-formatted text today the median effect of silencing one of
  the thirty hardest-firing neurons was 0.0004 to 0.0007 nats beside null rows of 0.0005 [measured, 14 November
  2026, one run]. The published exceptions were chosen by attribution to one behaviour, and one of them failed on
  held-out prompts.
- **Faults worth repairing do exist on healthy models of this size, and one is published on this model.** On the
  bf16 release of gemma-4-31B-it, thinking-on answers loop on long enumerations; zeroing 1,062 neurons of one layer
  cut held-out loops from 13 of 96 to 3 of 96, beat five matched random edits and left benchmarks within a point
  [checked, arXiv 2606.13705]. On a 27B, community work reports +4.3 and +13.0 points with intervals on two
  benchmarks under a registered null, and held-out email tasks from 30% to 68% with other tasks unmoved, on the
  same class of card [skeptic; the second checked]. All are faults of default, habit, looping or over-thinking,
  mostly in thinking mode or agent formats. Outside such faults nothing is shown. At 9B the three kept gains are
  5, about 18 and 1 items with no interval: nothing shown.
- **Whether that fault is on the QAT checkpoint, and whether the published neuron numbers mean the same there, is
  unmeasured anywhere.** This instrument can measure both. That is its most likely first real result.
- **The instrument as first designed could not have seen it.** The effect lives in long sampled answers (1,536
  pieces and more), needs sets of hundreds of neurons (10 to 33 left 9 to 11 of 13 loops), and was located with a
  backward pass. A held-out reading of it with controls is hours of the card [arithmetic, unmeasured]. Without a
  backward pass, screening every neuron by forwards is about 14 hours a behaviour [arithmetic].
- **Narrow associations can be written without gradients, with bleed**: facts, aliases, a trigger. The vault's own
  example is of this kind. It is also the class the safety policy must watch.
- **No source adds a general ability.**
- **Training against editing is open at this size.** On small models training gained more on the target where
  correct examples existed and moved more of everything else; an edit beat a small adapter on fact insertion once;
  nothing compares them at 27B or above.
- **So:** expect many honest nulls for small sets; expect the tool to be good at re-measuring other people's edits
  on this checkpoint, at testing a set or a direction quickly against fair controls, and at saying what a test
  cost. Do not sell it on benchmark gains. The proposal was too pessimistic about what is there to find and too
  optimistic that it would see it.


## 8. The amended phases

Effort is given in the units this repository works in. All figures are [estimate].

| Phase | Delivers | Owner | Card | Dollars |
|---|---|---|---|---|
| 0. Repairs, containment, pilots | Section 6, first half | 3 to 5 hours: approve repairs, register B, read the look | 3 to 4 hours; 4 to 5 more if the loop fault is there and its held-out reading is taken | one driver session |
| 1. The kernel, by command line | Section 6, second half; the registered gate; the two tests | 4 to 6 hours of review; the gate decision | 1 to 2 hours | two short sessions |
| 2. The agent's door, practice only | The sandboxed cell tool; capture, reading and written-reading gates; the book's driver view; budgets in GPU-seconds and forward requests; a runner with a cap and a kill; a work folder a session; first sessions on a planted fault and a canary. No word past a rank | 4 to 6 hours | sessions | a ceiling a session, approved ahead |
| 3. The claim gate | Briefs; generators with held-out frames; m fixed; the three control families; one registered bar rule; the batteries that gate; long written readings priced; the sheet. Words: lead, void, kept the base | the largest share: briefs, bars, review | 4 to 6 hours | |
| 4. The person's page | Base, edit and a control edit heard side by side on the vault page; rubrics; replay | review, words | about 1 hour | |
| 5. Of record, and out | The trial script and folder; export to a folder no tool writes, recompiled against the base it is served on; a volunteer's re-run | review; the policy | hours | |
| 6. Whether agents help | Sessions by driver family and by reach; the same budget allowed only prompts, decoding and steering | | | dollars |

Phases 2, 3 and 4 may be taken in another order (decision 0). Nothing of phase 5 starts before the licence repair.


## 9. The owner's decisions, pruned

Dropped as not decisions: the old 6 as worded ("none for now"), 12 ("as section xvi"), 15 ("later"). Moved to the
repairs: 10 (the R4 pin). Deferred to the phase that needs them: seals, what a page may show, volunteers' re-runs.

| # | Decision | Recommendation | Why it is the owner's |
|---|---|---|---|
| 0 | Who it is for first: the agent's door before the page, or the page before the agent | The agent's door, practice only, straight after the kernel; then the claim gate; then the page | The ask says a frontier model's environment; the recorded direction says a learning aid. Only the owner knows which he meant first |
| 1 | Start now, and what waits | Phase 0's repairs and no-card probes now; card pilots between layers; the kernel only after the unpushed commits, layer 36 and layer 34 are settled | It spends the time promised to the census and to volunteers |
| 2 | Reach: what each driver may do | Free code off the card in the instrument's sandbox, gates to the model, for Claude Code at the folder reach and endpoint models; free code on the card for the attended session only; refused at the home reach; Codex after its probe | It is a matter of trust in the machine and the accounts on it, not of method |
| 3 | The gate's rule, and what a stop stops | Fixed and hashed before the registered D. Go on if, at two or more layers, a set size on the ladder beats its other-site and sham-target sets at a fixed level, in both batch classes, on a metric the sets were not built from, and replicates across frame halves; or if the imported loop edit reproduces. A stop ends feed-forward column and head-output tools, and says nothing of embedding rows | It decides whether months are spent |
| 4 | Gradients | Run pilot F. If it runs, allow one attribution backward as a declared reach for the attended session and the scripted recipe, never pooled; agents' default stays none | It changes what "found without gradients" may be said of, and whether results sit beside the published gradient-free line |
| 5 | The public posture | Builders public (the method and such builds are already published); no brief or recipe on safeguards shipped; export stays the owner's command with licence keys; planted-trigger briefs allowed only as plants held in memory | It is his name on a public repository whose volunteers run local models |
| 6 | Budgets | A dollar ceiling for pilot H now; m = 2 fresh readings a brief with 39 control edits a family, before the first fresh reading | It is his money, and m sets how much a "lead" is worth |
| 7 | The name and the words | Levelling and level book only if the verb is "a reading" and tiers are "reach"; control edit; brief; no "held", no "seal" | The vocabulary of the pages is his |
| 8 | The existing faults of section 3 | Mend all five before anything else | Two touch what the public repository says and holds |

### The gate, restated

Before it is hashed the registered D names: its metric (not the read-out the sets were built from); the ladder of
set sizes; behaviour-selected single neurons beside random and most-active ones; items in chat format, each gated
on the base's log-probability; two batch classes; other-site sets matched on `delivered` with a coverage rule;
sham-target sets built by the same rule on another target; whether controls are pooled or matched, with the cost;
a column rewritten toward a word-piece at layers 55 to 59 with the vault's example as the known positive; one
head's output scaled; and the fixed level and replication that count as heard.


## 10. Still open

Only a measurement settles these:

1. Whether the thinking-on loop fault is on the W4A16 checkpoint, and whether the published neuron numbers carry.
2. Where the batch classes change (rows, length, a fresh process, the lens loaded), and whether sets or directions
   on chat-formatted items ever show two readings.
3. Whether sets of 64 to 1,000 input-side column edits are heard above matched and sham-target controls.
4. How many of the atlas's stored passages pass a base log-probability gate (it decides whether the dose-free
   atlas reading is possible on the index as stored).
5. How much of a real contrast direction lies in the quiet channel.
6. Whether a backward pass runs under the compiled unpacker.
7. Length, seconds and memory of thinking-on answers here, and rows a batch at 1,536 pieces.
8. Whether the real Codex app-server sandbox writes `/tmp`; whether a `bwrap` kernel holds the weights within its
   cap; whether a vendor's sandbox nests inside it.
9. The row-edit form, the float32-sum add, and memory a row-position with the counter reset: none measured.
10. Whether a current driver does this work on this model with the tools it already has (pilot H).

Only the owner settles: the order (decision 0); what waits; the gradient reach; the public posture.
