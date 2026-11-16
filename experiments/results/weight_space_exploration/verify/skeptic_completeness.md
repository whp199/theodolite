# Levelling, read for what is missing

Written 14 November 2026 (about 20:50Z to 21:15Z). Lens: completeness. Read-only: nothing in the repository was
changed, no server started, no model loaded, the card not touched. I ran grep, read files, read two file headers
(the checkpoint's safetensors header and the 12B GGUF's keys) and loaded `backend/scientist_tools.py` to count its
tools.

Read in full: the proposal (`design/synthesis.md`, 746 lines), the three judgements, the research digest, the pilot
note (both halves; the second was written at 21:03Z, while I worked), the JSON of pilots 3 to 7 and the hooks in
every pilot script. Read in part: the code digest (the scientist, the GGUF and edit sections), the five designs
(their positions, phases and every line on a sandbox, a second object or a dictionary).

Marks: **[read]** I opened the file today; **[measured]** a pilot record of today, with its file; **[arithmetic]**
my own sum; **[literature]** the research digest with its checkers' corrections; **[not established]** nobody has
measured it.

## In brief

The proposal is a careful plan for an honest edit-measuring kernel. Under this lens it has no fatal gap, because it
begins with pilots and a stop decision. It has seven serious ones:

1. The owner's decision on driver code is put as a choice of two, and the third choice (free code off the card,
   over the weights, with typed gates to the card) is the published shape, was scheduled by three of the five
   designs, was the science judge's advice, and is left with a trigger nothing in the plan can observe.
2. For one of the three existing backends (Codex) that third choice already exists, unmanaged, so "typed tools
   only" and "no gradients is a fact" are not true of it.
3. The stop gate samples one class of edit (take away or scale feed-forward columns, layers 20 to 55) and decides
   for all. The classes with a recorded success, in the house and in the one public benchmark on a Gemma, are not
   in it.
4. The proposal is already five pilots behind the card. Pilots 3 to 7, run six to twenty minutes after it was
   written, answer three of its listed pilots, withdraw the KL figures its headline floor rests on, and end in four
   rules for a harness, none of which is in it.
5. The planted-fault test, which is the proposal's one source of ground truth for discovery, leaks as described.
6. The guard against the artefact that undid the headline paper (the base under each chat setting; rescues split
   by the state of the base's answer) is said to cost "a few rows". With thinking on it costs written answers of
   unknown length, and no writing the plan costs is longer than 48 pieces.
7. "Future-proofed", the ask's own adjective, is never defined and has no test. The word does not occur in the
   proposal except as a design's name.

What I tried to break and could not: the proposal's choice to give little to the agent reading weights directly
(finding 8), and its choice not to build on the vault (the table of alternatives).

## Findings

### 1. Decision 2 is a false choice of two; the published shape was dropped, and its trigger cannot fire

**Claim attacked.** Section vii: "No agent code runs in the process that holds the model. That is the only
arrangement under which 'no gradients' is a fact and results can be set beside the published ones." Decision 2
offers "(a) typed verbs only; (b) free cells for any driver". Section xii leaves "a sandboxed CPU kernel for driver
code (weights mapped read-only, no card, no network)" unbuilt until "the owner approves free code for a driver other
than the attended session, or phase 4 shows the typed tools bind".

**Verdict: refuted as framed. Severity: serious.**

**Evidence.**
- The sentence is the science judge's, with its subject removed. The judge wrote: "Keep driver code out of the
  process that holds the model. A CPU sandbox with handles (validity, speed and future all have it) still lets a
  driver write a solver, and it is the only arrangement under which 'no gradients' is a fact and results can be set
  beside the published ones" (`design/judge_science.md:338-340`) [read]. The proposal keeps the justification
  (`design/synthesis.md:408-410`) and drops the sandbox to section xii (line 513).
- Four of the five designs give a driver code in a numbered phase: validity in its Phase 5 (`validity.md:345-346,
  402`), speed as a seventh tool in its phase 3 (`speed.md:69, 486`), future in its increment 4 (`future.md:239,
  326, 398`), driver from its first. The proposal schedules it in none [read].
- The published harnesses the proposal wants to stand beside mostly have that shape: a kernel over the weights and
  a counted forward tool (Takehi et al., Hu et al., Dolci; wsplab and weightspace with a sandboxed kernel). Only
  IWIBench is typed tools alone [literature, `digest_research.md:258, 277, 93`]. A typed-tools-only result stands
  beside IWIBench and no other.
- The trigger has no observable. Pilot H "cannot decide typed tools against free code" by the proposal's own words
  (line 487); phase 6 compares access levels (behaviour, activations, weights, atlas) and has no typed-against-free
  arm (line 465); settlement 2 says the question "is not answered by measurement" (lines 535-537). Two judges asked
  for that pilot to be kept (`judge_feasibility.md:177-178, 247-248`; `judge_science.md:265`). So nothing in the
  plan can "show the typed tools bind", and a tools-only driver cannot be "seen recomputing" spectra (the trigger
  on line 522), because it has nowhere to compute.
- The read half costs almost nothing here. The checkpoint is one file,
  `~/.cache/huggingface/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots/*/model.safetensors` (23.3 GB),
  whose header names `...layers.44.mlp.down_proj.weight_packed` I32 [5376, 2688] and `weight_scale` BF16
  [5376, 672] [read, header only, today]. `engine_runtime.dequantize_w4a16_full / _rows / _cols(packed, scale)`
  are plain functions of two tensors (`backend/engine_runtime.py:296, 313, 329`) [read]. Every reader the proposal
  names takes a loaded module (`backend/gguf_weights.py:543-576`) [read], so today a weight can be read only in
  the process that holds the model; a reader with no card is a few dozen lines over the same functions. The host
  has 61 GiB, 51 available (`free`, today).
- The day's own record shows what free code is worth to a current driver. Pilots 3 to 7 (1,078 lines in five
  scripts) were written and run between 20:48Z and 21:02Z: each script appears 80 to 120 seconds after the last
  result and runs in 50 to 150 seconds [read, file times]. Each set up a comparison that a levelling, as section v
  describes one, does not offer: an edit kept off the start mark (pilot 4), 31 null rescales of 1 +/- 0.001 (pilot
  5), one edit placed in four different rows (pilot 6), one edit read at ten batch sizes (pilot 7). Under "typed
  verbs, new operations by review" none of those could have been put to the card by any driver but the owner's own
  session, and they are what found the fault of finding 4.

**Smallest fix.** Add the third choice to decision 2: "(c) free code off the card, over the weights read-only,
with the typed gates as its only way to the model". Give phase 1 a reader with no card (`load(name, rows | cols)`),
so the owner's attended session has the published shape from the first day at a cost of a day. Give `draft_edit` a
way to take an array as data (a safetensors file in the work folder, hashed), since today the only source of a
direction is a difference of means. Log every "wanted and could not" as a book entry, so the triggers in section
xii have something to fire on.

### 2. "Typed tools only" and "no gradients is a fact" do not hold for the Codex backend as it stands

**Claim attacked.** Sections vii and xiii.2: any agent but the owner's attended session has typed tools only; "a
tools-only driver cannot write a solver or a custom analysis"; "no gradients" is then a fact.

**Verdict: weakened (true for two of three backends). Severity: serious.**

**Evidence.** At the default reach Codex "runs commands in a sandbox that writes only in its own work folder and has
no network; it can read files elsewhere" (`backend/scientist_chat.py:103`; `sandbox = "workspace-write"` at line
843; `README.md:692-695`) [read]. The proposal carries this fact into its seals (section vi) and stops there. But a
shell that reads the disk also reads the checkpoint file (mode r--r--r--) and the repository, which holds the
unpacking functions and a CPU int4 kernel that runs the 31B at 0.45 s a word-piece in about 21 GB (the proposal's
own line 524). So under Codex the driver already has free CPU code over the weights, may be able to run the model
there, and "no gradients" rests on the sentence in its prompt ("use the atlas's tools for every reading",
`scientist_chat.py:234-238`), not on a mechanism. `/usr/bin/codex` is installed [read].

**Not established:** whether that sandbox can import torch from the venv, hold 21 GB, or open the card's devices.
One Codex chat at the default reach, asked to unpack one column from the file, settles it with no card.

**Smallest fix.** Put that chat in pilot 0. State each backend's code reach on the harness card as an access level.
Then either take Codex's shell as the CPU kernel (a documented reader, a declared level, results never pooled with
tools-only ones), or keep Codex out of weight-space sessions until the boundary exists.

### 3. The stop gate measures one class of edit and decides for all of them

**Claim attacked.** Pilot D and decision 1: stop or go on "if, at two or more of the four layers [20, 36, 45, 55],
at least 5 of the 20 column sets at one write ceiling or less beat every chance edit". Section xx, claim 7: "the
whole design past phase 1 rests on it".

**Verdict: weakened. Severity: serious.**

**Evidence.** Pilot D reads single feed-forward columns at scales 0, 2 and 4, sets of 64 columns and direction
removals (line 483). Not sampled:
- **A column's content rewritten.** The house's one stored heard edit is of this kind and at the last layer:
  `UPDATE LAYER 59 SLOT 7830 SET object = 'six' STRENGTH 3.0`, first-step KL 5.65 (spider), 6.84 (dog), 13.29
  (hexagon), 0.0 on four unrelated questions (`experiments/results/vault_example_edit/probe.json`, 29 October
  2026) [read]. The proposal's `set` can express it; the gate does not try it, and no pilot layer is in the last
  block.
- **Rows of the unembedding.** On the only public benchmark with a Gemma target (IWIBench, Gemma 3 4B IT), "all
  273 of Astra's successful writes went to the tied embedding/output matrix" [literature, checker's correction,
  `digest_research.md:95`]; 65 embedding rows repaired a robot policy (line 211). The proposal leaves embedding
  edits unbuilt (lines 211-213, 518). The reason given, that the writer takes its logits outside any module, is
  true (`backend/atlas_ops.py:493-499`) [read] and is also why the output side is easy: the reader computes the
  logits itself and can add a per-row term there.
- **An attention head's output block.** In the published vocabulary of five edit kinds (`digest_research.md:211`);
  gated off here, though a zero term on q, k, v and o closed by bits today [measured, `pilot/RESULTS.md`].
Of those five kinds the first increment offers three, and one of the three (a rank-one term) "cannot go past a
lead" (lines 203-204). And the gate's "16 stored passages" are raw text, the regime in which pilots 4 to 7 found
one edit giving two readings (finding 4).

**Smallest fix.** Three arms in pilot D: a `set` toward a word-piece's unembedding at layers 55 to 59, with the
vault's example as the known positive; a per-row logit term for a few unembedding rows; one head's `o_proj` block
scaled. Or word decision 1 as "stop building feed-forward column tools", not "stop".

### 4. The proposal is five pilots behind the card, and the pilot note's four new rules are not in it

**Claim attacked.** Section i (what stands on measurement), section v's "weak points, said in advance", section x
(pilots A, C and E listed as not yet run), and claims 5 and 6 of section xx.

**Verdict: refuted in part. Severity: serious.**

**Evidence** [all measured, 14 November 2026, one run each; `pilot/pilot3_writing_and_capture.json` (20:49Z),
`pilot4_cancellation.json` (20:52Z), `pilot5_knife_edge.json` (20:55Z), `pilot6_which_is_true.json` (20:57Z),
`pilot7_batch_size.json` (about 21:00Z), and the second half of `pilot/RESULTS.md`, written 21:03Z; the proposal's
file was last written 20:43Z. I read the JSON of pilots 3 to 6 in full and checked pilot 7's against the note]:
- **Pilot A is answered, and passes.** Eight rows written for 48 pieces under per-row hooks: base rows identical,
  the zero row equal to the base, 0.095 s a step, on two prompts.
- **Pilot C is answered: sound at layer 44, not below it on raw text.** One-row capture replayed at 32 rows
  against same-shape forwards, 30 single-neuron candidates: the two ways' effects correlate 0.9998 and 0.9999 at
  layer 44, 0.94 and 0.92 at layers 5 and 20 on the prose, 0.02 and 0.32 at layers 5 and 20 on the raw recipe.
- **Pilot E is answered: no constant is a floor, and the times-3 row is not one off layer 44.** Null rows read
  0.0066, 0.0039 and 0.0006 nats at layers 5, 20 and 44 on the raw recipe and 0.00055, 0.00046 and 0.00014 on the
  same passage in chat form. At layer 20 on the raw recipe, in a batch of 32, silencing neuron 4294 reads 0.726
  folded and 0.023 as an output-side term, a gap between forms about two hundred times the floor row there
  (0.0034); at a batch of 2 the same two forms read 0.0091 and 0.0090.
- **New: on raw text one edit has two readings, by batch size alone, in the folded form too.** Six of seven
  neurons at layers 5 and 20: 0.003 to 0.019 nats at a batch of 2, 0.34 to 0.73 at a batch of 32. Pilot 7 places
  the break between 12 and 16 rows ("two arithmetics on this card at this length") and finds the same passages in
  chat form stable. The 93 null rescales never parted.
- **New: the per-row form that equals the folded form is not the one section iv specifies.** Silencing on the
  input side matched the folded reading to five figures at both batch sizes in all seven cases (0.72581 and
  0.72581; 0.00908 and 0.00908). The output-side term did not follow.
- **The note withdraws figures the proposal leans on.** "The first two pilots used raw text; their timings stand,
  their KL figures are of an off-distribution regime." Section o's and section i's 0.00045 floor and 0.00041
  single-neuron median are those figures.

The note ends with four rules for a harness: gate every item on the base row's own log-probability; trust a reading
only when a small and a large batch agree; report the median and the count of positions over a nat beside the mean;
ride a column edit on the input side and keep the output-side term for low-rank edits. None is in the proposal.
Section v's list of weak points does not have "the effect is not one number across batch shapes", and its remedy
(confirm in the folded form) would set a reading at 89 rows beside a twin at another row count.

**A consequence nobody has drawn.** Two of the proposal's five batteries (neutral text; habitat), the atlas reading
of section ii item 2, and pilot D's "16 stored passages" all read raw corpus windows. Raw text is the regime the
note now says must be gated on the base's own log-probability (the recipe read -9.6 nats a piece; the prose was
stable). How many stored passages pass such a gate is [not established]. The habitat battery is the one "only this
instrument can run".

**Smallest fix.** Revise sections o, i, iv, v, vi and x against pilots 3 to 7 before the owner reads the proposal:
take in the note's four rules; make the row count part of a reading's identity and take the folded twin at the same
count; drop "five times the floor row" for a null distribution on the item in hand. Before pilot D, one registered
run on chat-formatted items with sets and directions, and a count of how many stored passages pass the gate.

### 5. The hook the proposal specifies is not the hook that was measured

**Claim attacked.** Section iv: the hook adds `y[row] = (y[row].float() + dose[row] * d).to(y.dtype)`: "summed in
float32, rounded once". Section i marks the delta form's bit-safety, cost and floor as measured.

**Verdict: weakened. Severity: minor.**

**Evidence.** Every pilot hook adds in the output's own dtype: `return out + a * self.delta[:, None, :].to(out.dtype)`
(`pilot/pilot_edit_speed.py:206-215`; `pilot2_equivalence.py:124`; `pilot3_writing_and_capture.py:124`;
`pilot4_cancellation.py:124-127`; `pilot6_which_is_true.py:141-143`) [read], with the column rounded to bf16 first.
So each measured line is of a bf16 product added in bf16. And on the card the base product reaches any forward hook
already rounded to bf16, so no output-side hook is "rounded once" in the sense the CPU kernel's is. The float32 sum
is not piloted: its cost (two casts of a rows x pieces x 5,376 tensor) and its floor are not established. The
pilot note of 21:03Z now says a column edit should ride the input side, with an output-side add only for a new
column and for low-rank terms, so the hook of section iv is superseded for the commonest edits and unmeasured for
the rest.

**Smallest fix.** Rewrite "How it is applied" from the note. One line in the card checks: the output-side add in
the two arithmetics, side by side, on a chat-formatted item.

### 6. The planted-fault test leaks as described

**Claim attacked.** Section vi: "with per-row deltas the owner can lay a hidden edit on every row and ask the driver
to find and repair it, which gives ground truth for discovery ... The read tools then serve the planted state."
Phase 4 proves "whether a current driver finds a planted fault with typed tools".

**Verdict: weakened. Severity: serious** (phase 4's headline proof).

**Evidence.**
- The scientist has 35 tools today, counted by loading the module: among them `logprobs`, `activations`,
  `compare`, `steer`, `causal_check`, `top_firing`, `passages` [read]. They run live forwards on the base, outside
  any levelling hook, and the atlas's stored passages and census describe the base. A driver can set `logprobs`
  beside the base row of `level` on one text and see the plant, then walk `activations` against `capture` layer
  by layer to its site. That is finding a difference from a healthy reference, which the literature already
  counts as the easy case, and needs no reasoning about weights.
- To close it the plant must ride every forward of the session, the old tools' included. That is an edit in force
  on the server, which the thesis of section iii forbids and which serves the atlas pages from a planted model for
  the session.
- "Hidden" where? An edit is saved as `atlas_data/edits/<id>/` (section iv). Under Codex the shell reads files
  elsewhere (finding 2). The proposal lists the reworded set as file-held and limited; the plant is not on that
  list.

**Smallest fix.** Write the protocol down: the plant is drawn and held in server memory, as the blind test's truth
is; it is in force server-wide for the session, under the stale banner of repair 1; the base-serving tools are
either under the plant or withheld; and "with a healthy reference" and "without" are reported as two access levels.

### 7. The settings audit and the strata table do not "cost a few rows each"

**Claim attacked.** Section vi: "Two additions answer the loudest critiques of the original post and cost a few rows
each": the base read under each chat setting, and rescues split by the state of the base's answer (cut off,
repeating, refusing, finished and wrong, finished and right).

**Verdict: weakened. Severity: serious** (it is the guard against the artefact that undid the headline result).

**Evidence.**
- The instrument asks everything with thinking off (`backend/atlas.py:331`; `backend/census.py:2132` "chat:
  thinking off"; the switch is at `backend/engine_runtime.py:1200-1221`) [read]. With thinking on, the row is a
  written answer whose length nobody has recorded for this model; a teacher-forced measure on the answer has
  nothing to stand on until the thought is written.
- Every writing the proposal costs is 48 pieces (lines 276, 480). The published obstacle was an answer cut off at a
  cap of 32,768 after looping. At 48 pieces every long answer is "cut off", so the stratum that mattered cannot be
  told from an ordinary answer.
- Cost [arithmetic, from 0.095 s a step (pilot 3, measured today) and the proposal's own 0.35 to 0.75 MB a
  row-position]: a 2,000-piece answer is about 190 s a batch; beside the lens (5.6 GiB of headroom, the proposal's
  figure) 4,000-piece answers fit two to four rows. Ninety-six items with base, edit and one chance edit is hours,
  where the cost table says a fresh reading with its batteries takes 2 to 4 minutes. The table has no line for it.

**Smallest fix.** A ten-minute pilot: twelve items, base only, thinking on, written to the natural stop; record
pieces, seconds and GiB. Then a line in the cost table, a length cap chosen from it, and a definition of "cut off"
for this instrument.

### 8. The agent reading weights directly: the proposal gives it little. I could not show that is wrong

**Claim attacked.** That reading weights deserves one tool of six, in phase 4, with "every reply typed a lead".

**Verdict: stands, with two cheap omissions. Severity: minor.**

**What I tried.** The literature: weights-only readings validated for 25 to 59% of units; tool-free reading is
shown only to about 172,000 weights; on Gemma 3 270M weights alone failed four times where attention on 16 texts
gave 3 of 3; forward passes were the deciding resource, in single runs [literature, `digest_research.md:219, 129,
45, 34`]. The house's own weights-derived reading is worse than a coin: the sign of a link "agreed with the measured
change at 0.336 of links (0.341 for as many readers not linked)" (`README.md:422-423`) [read]. I found no evidence
the other way for a model of this size.

**What is still missing.** (a) The reader with no card (finding 1): cheap, and it is what "weight space" means to a
driver. (b) A measure of whether reading helps. The only weights-only reading in the first phases is `writers`
(columns ranked along a direction); pilot D already reads "top writers" against chance sets, so one more column in
its table (top by `writers` against top by firing on the items) would say whether the reading adds anything. The
proper test (the same run by access level) is in open-ended phase 6. (c) The decoy check for reading claims
(permuted or shuffled weights), which validity scheduled and the science judge noted, is gone from the proposal
altogether [read, grep: no "decoy", "shuffl" or "permut" in `synthesis.md`].

### 9. Where directions come from; the community dictionaries were silently dropped

**Claim attacked.** Section vii: the capture-to-direction handle makes a tools-only agent able to run the published
recipe. Section xii is the list of what is unbuilt until needed.

**Verdict: weakened. Severity: minor.**

**Evidence.** The only source of a direction `u` is a difference of means over two sets of the driver's own texts.
No array can be handed in (finding 1). The checker corrected the research on this point: sparse autoencoders for
this 31B are public, one at layer 30 with 131k latents and one TopK set of 43,008 features a layer on all 60
layers with 3,000 labelled [literature, `digest_research.md:181, 194`]. Three designs listed them as left out
(`validity.md:448`, `speed.md:629`, `future.md:433`). The proposal does not mention them anywhere, not even in
section xii [read, grep]. Their quality on the QAT checkpoint is not established (the layer-30 one was trained on
the base model). Size [arithmetic]: one layer's decoder is 43,008 x 5,376 x 4 bytes = 0.92 GB in float32, about 55
GB for all sixty.

**Smallest fix.** A row in section xii with its trigger. Let `remove_direction` take a unit vector from a file, so a
decoder row is admissible with no new code, and type the reply "direction from an outside dictionary, not checked
on this checkpoint".

### 10. The 12B on disk: the deferral's trigger is vague, and the object answers three questions the plan leaves open

**Claim attacked.** Section xii: "the 12B as a second object | an outside harness or a second object is real".

**Verdict: weakened. Severity: minor.**

**Evidence.** `~/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf`
(6,975,878,560 bytes) states `general.architecture = gemma4`, 48 blocks, width 3,840, feed-forward 15,360,
`attention.shared_kv_layers = 0`, `embedding_length_per_layer_input = 0`, no expert count [read, header, today]. It
passes each refusal in `backend/gguf_model.py:213-221` [read]. Whether it builds and gives a right forward is
[not established]. Outside harnesses are already real (the proposal's own section ii names them), and the one
documented agent edit on a Gemma 4 is on a 12B [literature]. So it would give: a known answer from outside the
repository; a test that the new modules are not 31B-shaped; and a second size in one family, where the literature
has only 0.8B and 9B confounded by headroom. It has no atlas, and nothing found on it carries to the 31B.

**Smallest fix.** One line in pilot 0, no card: does `build_gemma4` build it on the CPU, and does one forward's top
piece agree with llama.cpp. Decide after that.

### 11. No existing library was weighed

**Claim attacked.** About 2,150 lines of new modules (section iii) with no word on what already exists.

**Verdict: the choice stands; the silence is the gap. Severity: minor.**

**Evidence.** `nnsight`, `transformer_lens`, `peft` and `sae_lens` are not in `.venv` [read]; none is named in any
of the five designs [read, grep]. PEFT's mixed-batch adapters with `__base__` are the published form of "each row
its own low-rank term, one row the base" [literature, `digest_research.md:380`]. Against them: the pilot's own hook
is about twenty lines; the linears are packed with a forward bound on the instance; the gate, the chance edits and
the book would be house code whatever applied the term. So new modules are the right call. What a library would
have bought is a surface a future driver already knows, and a file format others can load.

**Smallest fix.** A paragraph saying so; and write the payload as a LoRA safetensors from the first day (a column
term is B.A with a one-hot A), not at phase 5.

### 12. "Future-proofed" is never defined, and has no test

**Claim attacked.** That the proposal answers the ask ("future-proofed enough to allow a future generation of
frontier model, or even a current one, to formulate and test hypotheses quickly, implement, and iterate").

**Verdict: weakened. Severity: serious** (a headline is missing).

**Evidence.** The word does not occur in `synthesis.md` except as the name of a design [read, grep]. Settlement 6
reduces the future design to "one small manifest". The one forward-looking passage is lines 426-428: a stronger
driver "uses the same three gates harder" and "cannot ... without review ... invent a new kind of operation".
Measured against what the word should mean here:

| It should mean | Delivered? |
|---|---|
| Another driver plugs in with no code change | In phase 4, through the registry's three backends. From phase 1 only by the command line, which is kept to the owner's session. The stdio MCP server already serves any MCP client (code digest); the proposal does not say so |
| Nothing that encodes today's driver's weakness is welded in | `SCAFFOLD.md` and the bare switch cover the conveniences. The parts that will bind a stronger driver are named trust boundaries and stay: the vocabulary, briefs the owner keeps, operations by review |
| The object can change | Deferred (finding 10). The checks run on the six-layer stand-in, which keeps the new modules honest about sizes |
| Records outlive the session | The manifest is versioned; the plan file and the book entry are not. No backend's transcript is kept whole by the instrument: Claude Code's lives in Claude Code's own store (`backend/scientist_chat.py:9, 558, 1368`) [read], and the proposal's "recorded session" (section xvi) does not say where the driver's turns are kept |
| There is a test of it | None |

**Smallest fix.** Half a page that says this, and one test at the end of phase 1: two driver families, each given
one page and the command line and nothing else, each asked for the known answer unnamed; pass if both reach it with
no change to the code. Version the plan file and the book entry with the manifest. Copy the driver's transcript
into the book's folder when a session closes.

## The alternatives the task named, one by one

| Alternative | Weighed in the proposal? | My check | Matters | Smallest remedy |
|---|---|---|---|---|
| Give the existing scientist the vault and a paired reading | No. Only the future design names it (`future.md:23`) | I tried to make the case. `POST /api/larql` exists with a before-and-after voice (`backend/server.py:1460`) [read], so the tool is a proxy. But a vault edit is one column, in force on the server, with stale indexes (repair 1), a direction in the wrong space and a rollback that misses attention. As a harness, no. As a first look at a driver's behaviour, today's own pilot session already serves | Little | One line saying why not |
| A free CPU kernel over the weights and a budgeted forward tool | Deferred with a trigger that cannot fire | Findings 1 and 2 | Much | Decision 2 gets a third choice; a reader with no card in phase 1 |
| A smaller rehearsal object | The stand-in is used for plumbing. The 12B is deferred | Finding 10 | Some | One no-card line in pilot 0 |
| nnsight or another library | No | Finding 11 | Little | A paragraph; the payload as a LoRA file |
| Community sparse autoencoders | No, and absent even from the unbuilt list | Finding 9 | Some | A row in xii; directions from a file |

## What the proposal does not say at all

1. **What today's session is evidence of.** Seven pilot scripts (1,779 lines) were written and run on the 31B
   between 19:50Z and 21:02Z with no harness; pilots 3 to 6 in ten minutes [read, file times; who wrote them is my
   inference from their docstrings and pace]. That is a current driver formulating and testing hypotheses about
   weight edits on this object, quickly, with free code beside the model, and finding a fault in the measurement
   that five designs and three judges had missed. The proposal uses the pilots as hardware facts only. They also
   say that phase 1's worth is its guards (chance edits, the book, the gate), not its speed, and that the loop's
   time is the writing and the 11 s load, not the readings.
2. **What "attended" means.** The owner's session has free code from phase 1 and none of section vi binds it: a
   script that does not call the library writes nothing to the book. Is a three-hour run with the owner out of the
   room attended?
3. **A lead-only agent surface straight after phase 1.** Five of the six tools (`weights`, `capture`,
   `draft_edit`, practice `level`, `book`) need only the kernel and its routes. Only `brief` and fresh readings
   need phase 3. Settlement 3's price, "an unattended frontier driver arrives in months", is a choice of order the
   proposal does not argue.
4. **A budget that can be set beside the published ones.** GPU-seconds are the right unit here. The published
   harnesses count 10 to 50 forward requests. No preset or conversion is given, so no result can be quoted "at
   the published budget".
5. **The rebuild test** (a fresh session reproduces a reading from a description of 200 words or fewer): in the
   research's implications, in no design's synthesis.
6. **A cap or a drift row for stacked edits.** Composing is adding; nothing says what a sum of many kept edits
   does or when it is re-read. The science judge asked for the drift row.
7. **How the gate's dose unit relates to published doses.** The gate is "one write ceiling or less"; the published
   Qwen edit over-drove its removal sixfold. The proposal does not say why one ceiling is the right bar.
8. **What a driver two generations on will want, and where each wish lands:**

| It will want | In the proposal |
|---|---|
| Its own code over the weights and the activations | Owner's session only; otherwise a trigger that cannot fire |
| To invent an operation or a measurement | "By review"; there is no channel for a driver to propose one with its check, though the proposal pattern for names and notes exists |
| Directions from anywhere (a solver, a decomposition, a dictionary) | Difference of means only |
| Attention, the embedding, norms | Unbuilt or gated off |
| To write its own tasks | It may propose a brief the owner keeps; the owner is the pace (the proposal's risk 6) |
| An attribution backward | None; pilot F informs |
| Runs of hours | Jobs with cancel arrive in phase 3; a tool call is one held request of at most 900 s today |
| Many hypotheses at once | Served well: rows of one forward. This is the most future-proof thing in the design |

## Pilots that should come first and are not listed

| Pilot | Card | Settles |
|---|---|---|
| A Codex chat at the default reach unpacks one column from the checkpoint file | none | Finding 2: whether a CPU kernel already exists unmanaged |
| `build_gemma4` on the 12B GGUF, on the CPU | none | Finding 10 |
| Twelve items, base only, thinking on, written to the stop | about 10 minutes | Finding 7 |
| Sets and directions (not single neurons) on chat-formatted items at 8 and 32 rows, registered; and how many of the atlas's stored passages pass a base log-probability gate | about 15 minutes | Finding 4, before pilot D |
| Three more arms in pilot D | inside D | Finding 3 |
| The output-side add in the two arithmetics, side by side | minutes | Finding 5 |

## What I could not check

Whether Codex's sandbox can import torch or open the card; whether the 12B builds; the length of a thinking answer
on this model; whether the two-readings effect of pilots 6 and 7 touches sets and directions, or any chat-formatted
item (two passages so far say not); whether the atlas's stored passages are all raw text (I take it from the
proposal's "corpus windows"); anything in the papers beyond the research digest. I did not read the five designs or
the code digest end to end.
