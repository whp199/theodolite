# Skeptic, lens: the literature. Attack on "Levelling" (synthesis.md)

Written 14 November 2026. Read-only: nothing in the repository was touched, no model was loaded, the card was not
used. Every source below was opened by me today (arXiv HTML or the listing API, raw repository files, vendor pages)
unless marked otherwise. Pages were saved as text under
`experiments/results/weight_space_exploration/verify/litpages/`
so each figure can be grepped again.

Marks: **[read]** opened today and the figure seen; **[arithmetic]** my sum from read figures; **[not checked]**.

## In one paragraph

The proposal survives, but its "honest ceiling" is built on a partial reading and its research missed the single
most relevant prior paper. A neuron-level weight edit of **this very model** has been on arXiv since June 2026
(2606.13705): thinking-on gemma-4-31B-it loops on long enumerations, and silencing 1,062 neurons of one layer cuts
held-out loops from 13/96 to 3/96 with matched random controls. That one paper (a) answers "obstacle-like faults,
if any exist" with "one is published", (b) gives an outside known answer far better than the pronoun neurons, (c)
gives a dose-response on this model that says sets of 10 to 33 neurons do almost nothing and about a thousand are
needed, which the stop-or-go gate (sets of 64) ignores, and (d) lives entirely in long sampled generation, which the
proposal's fast path cannot see and its cost table does not price. Three sentences of the ceiling are wrong as
written ("on stronger models a few points or nothing", "training beat editing where compared", "three identical
searches"). The absence claims stand only in their narrowest wording. The licence claim, the 27B-on-a-5090 claim
and the backdoor claim stand.

## Findings

### 1. "None on this 31B"; "obstacle-like faults if any exist under some chat setting" (sections ii, xvi)
**Verdict: weakened (the literal absence claim stands; the ceiling's "if any" is contradicted). Severity: serious.**

- arXiv 2606.13705, "When Can One Neuron Fix Repetition Loops in LLMs?" (Lazaridis and others, v1 9 June 2026, v2
  24 August 2026) [read, https://arxiv.org/abs/2606.13705 and /html/2606.13705]. On google/gemma-4-31B-it, bf16,
  sampling at temperature 0.7, top-p 0.95, top-k 64, 1,536 new tokens: with thinking enabled 5 of 8 generations
  loop on "list all episodes of The Wire", none with thinking off. The edit zeroes the down_proj columns of 1,062
  neurons of one layer (their "L36"; indexing not checked). Held-out prompts and seeds: 31B loops 13/96 to 3/96
  (10.4 points, bootstrap interval 4.2 to 16.7); the selected edit beats every one of five matched random edits for
  the 31B with thinking on; five development benchmarks stay within -0.37 to +0.80 points and three untouched ones
  within -0.8 to +0.2. At 4k and 8k budgets the remaining 31B failures become "doom looping" over facts it cannot
  recall.
- The edited checkpoint is public: edgerunner-ai/gemma-4-31B-it-noloop (created 9 June 2026, Apache-2.0), with two
  third-party GGUF conversions (July 2026) [read, model hub API]. Diffing it against the base gives the neuron list.
- Not agent-driven (people plus attribution sweeps), so "no agent-driven edit of this 31B" survives. But the paper
  is not in the research digest, the six reports, the five designs, the synthesis or the repository's REFERENCES,
  README or ROADMAP [read, grep for the id, the authors, "noloop", "EdgeRunner": no hit].
- What is not established: that the loop exists on the QAT W4A16 checkpoint. The paper itself lost 2 of 8 of its
  31B loop captures when it moved from one inference path to another.

**Fix.** Cite it. Add one pilot before pilot D: on the W4A16 checkpoint, thinking on, the paper's prompts and
sampling, base against the published neuron set silenced as a per-row delta, read in generation. It is at once the
outside known answer, the first test of whether a bf16-found neuron set carries to the QAT weights, and the first
priced long-generation reading. Change section xvi to name it beside the abliteration builds and the backdoor.

### 2. "On stronger models held-out gains were a few points or nothing"; "small or damaged models" (sections o, ii)
**Verdict: refuted as worded. Severity: serious (a headline must change).**

- The sentence rests on Hu and others at 9B. Their three kept 9B gains are +1.85 on 271 prompts, +3.03 on 580
  problems and +1.54 on 65 chains [read, arXiv 2611.09382 Table 2 and Table 4]: that is 5 prompts, about 18
  problems and 1 chain [arithmetic], single seed 0, no interval printed for any held-out figure, each no larger
  than about one unpaired standard error (2.5 to 2.6 points) [arithmetic]. So at 9B the paper shows nothing
  distinguishable from zero. Its held-out bases were near ceiling (Math 97.70, IFEval 89.30, MMLU-Pro 82.89), every
  search had 50 forward requests, and the authors do say headroom. All of that stands.
- At 27B, which the proposal cites only for memory, the same digest lists results the ceiling leaves out:
  - Qwen3.8-27B, nine rank-one down-projection edits, pre-registered, norm-matched random control null: GPQA
    Diamond +4.3 (95% interval +1.8 to +7.1), LiveCodeBench v6 +13.0 (+6.1 to +20.6), with 27 to 54% fewer thinking
    tokens; measured on bf16 served at Q8_0; working paper of 8 November 2026, editor not stated [read,
    https://concept.country/research/weight-space-programming].
  - Qwen3.8-27B in 4-bit NVFP4 on one RTX 5090: held-out email tasks 30.0% to 68.0% (39 fixed, 1 broken), other 78
    task types 73.1% to 73.1%, random-direction control 30 of 100 [read, RESULTS.md of the 27B harness,
    https://github.com/munishbp/weight_space_harness].
  - gemma-4-31B-it itself: finding 1.
- What is true: every one of these is a default, a habit or an over-thinking fault, found in thinking mode or in an
  agent's action format, on a healthy 27B to 31B. Gains on general ability with thinking off were small (4B
  LiveCodeBench 52.6 to 59.5, p about 0.05 [read, HomoHabilis README]) or nil.

**Fix.** Replace the sentence with: "Published gains on healthy 27B to 31B models are confined to faults of
default, habit, looping or over-thinking, mostly in thinking mode or agent formats; there they reached 10 to 38
points on the affected portion. Outside such faults, nothing has been shown." Drop "small or damaged".

### 3. The fast path can see what the literature found (sections v, x; implied)
**Verdict: weakened. Severity: serious (a phase and the cost table must change).**

- Every large published effect is a property of long, sampled, free-running generation: 1,536 tokens at
  temperature 0.7 over 8 seeds (2606.13705); 32,768 tokens at temperature 0.6 with presence penalty 1.5 (Takehi;
  Hu and others, Table 3); multi-step agent episodes (the 27B harness). The kept 9B IFEval edit in Hu and others is
  itself a truncation repair: 58 truncated answers to 2 of 1,920, 8.76M generated tokens to 6.15M [read, section 8].
- The proposal's measurement is a paired difference of teacher-forced odds over at most 176 positions; generation
  appears as "three voices, 3 rows x 48 new pieces, about 5 s" and as a written gate at the trial of record. Pilot A
  checks 48 pieces. The cost table has no row for a 1,536-piece sampled reading, and sampling with a seed a row
  under per-row deltas is not designed.
- So the one class of fault the proposal itself says to expect ("obstacle-like faults under some chat setting")
  is invisible to practice and fresh readings as costed.

**Fix.** Make a long-generation reading a third priced class from phase 1 (rows = seeds x {base, edit, chance
edit}, loop and cut-off detectors, the strata table), and extend pilot A to 1,536 pieces with sampling. Its cost on
this card is not established: 1,536 steps at the recorded 0.09 to 0.11 s is about 2.5 minutes a batch
[arithmetic]; how many rows a batch holds at that length has not been measured.

### 4. "Sets of 64 and direction removals are heard"; the stop-or-go rule of decision 1 (claim 7, pilot D)
**Verdict: weakened. Severity: serious (the gate must change before it is hashed).**

- The only published dose-response on this model [read, 2606.13705 Table 8; 64 generations, thinking on, baseline
  13 loops]: neurons ranked high on both probes, 10 to 33 of them: 9 to 11 loops; 367 neurons: 6; about 1,000:
  1 to 4; 1,062: 0. A set of 64 at one layer sits where the published curve is nearly flat.
- Pilot D's single-column arm draws "200 random and 20 most-active" columns. Every published single-neuron success
  chose its neuron by attribution to one behaviour and measured that behaviour: Gemma 4 12B, neuron 1640 of layer
  33, first of 737,280, off moves the target form 26% to 82%, doubled 26% to 4%, "a random neuron x3" does nothing
  [read, https://github.com/ValerioDolci/frontier-weight-edit]; one refusal neuron in each of seven models from
  1.7B to 70B, by suppressing its activation (arXiv 2605.08513, 8 May 2026) [read, abstract only]. A random or busy neuron read by all-position
  KL on a paragraph is the sample the literature predicts will be silent, so that arm cannot bear on "a single
  neuron that matters by itself, except rarely". (For the proposal's caution: the one-neuron edit of Gemma 4 E2B in
  2606.13705 cleared its selection grid and then moved held-out loops only 6 to 4 of 96, p = 0.727 [read]; and the
  31B needed about a thousand.)
- Published kept edits are also strong: one 9B column multiplied by -32; a direction over-driven 6 and 7 times; a
  whole 27B matrix refitted. Whether "one write ceiling or less" reaches those operating points I could not map.

**Fix.** Add to pilot D a set-size ladder (1, 10, 64, 300, 1,000) at the published layer with the published set
and a matched random set, read on the behaviour and on neutral text; add behaviour-selected single neurons; state
the stop rule on the ladder, not on "5 of 20 sets of 64".

### 5. "Gradients: none ... the published harnesses forbid one" (settlement 8, decision 6)
**Verdict: weakened. Severity: serious (the settlement should be re-opened with this evidence, not after phase 4).**

- True of the weight-space-programming line: Hu and others ("no backward passes, no optimizers" [read]),
  IWIBench, the 27B harness, HomoHabilis.
- But every neuron-level edit published on Gemma 4 was located with a gradient: 2606.13705 ranks neurons by
  gradient times activation; on Gemma 4 12B the gradient-free agent spent about two hours and 27 to 30 forward
  passes building an 8-neuron circuit and "never saw" the one neuron that a single backward pass ranks first and
  that matches the circuit when doubled [read]; the backdoor paper refines by a few hundred gradient steps.
- The proposal prices the alternative itself: 0.04 s a candidate is about 14 hours for one behaviour over 1.29
  million neurons [arithmetic].

**Fix.** Run pilot F in phase 0 as planned, and decide before phase 1 whether one attribution backward is a
declared level for the owner's own sittings (never pooled with gradient-free results). If it cannot run under the
compiled unpacker, say so in the ceiling: it removes the localiser behind every Gemma 4 result.

### 6. "The only arrangement under which results can be set beside the published ones" (section vii)
**Verdict: weakened. Severity: serious for phase 4's claim, minor for phases 0 to 3.**

- The proposal's agent surface is six typed tools and no code. Among published harnesses only IWIBench is typed
  only (four tools, "no access to gradients, the shell, Python execution" [read]); there the best editor aimed all
  273 successful writes at the tied embedding and output matrix, the tensor class the proposal leaves unbuilt.
- Every published MLP-level edit found by an agent came with a free Python kernel over the weights (Takehi; Hu and
  others: 300 Python analyses a search; Dolci; HomoHabilis) or a shell: the 27B harness on a 5090 "runs teacher
  code" in "a process that holds the student in memory" [read, its README], the arrangement the proposal refuses.
- The recipe behind every 27B result on that card is a closed-form least-squares fit of keys to values with a
  protection set over all positions. Six tools with a difference-of-means handle can run the Takehi recipe; they
  cannot run this one (no solver, no key matrices).

**Fix.** Say plainly that phase-4 results sit beside IWIBench only. Either add a typed `fit` (keys, values,
protection set, rank cap) before the first agent session or bring the read-only CPU kernel forward.

### 7. "Training beat editing where the two were compared" (sections o, ii; "do not expect an edit that beats a small trained adapter")
**Verdict: refuted as a general statement. Severity: serious (a headline sentence).**

- For: Qwen3.5-0.8B, GRPO for about 23 hours on a 5090 beat a four-session edit by 8.1 points on average [read,
  XLRA README]; LoRA 87% against the edit's 57% on 4B email tasks [read, 27B harness RESULTS.md].
- Against, in the project's own digest: 30 invented facts written into a 4-bit Qwen3.5-9B by an agent's
  closed-form edit, 96.7% on never-seen phrasings, 30 of 30 facts, 100% of real-world answers kept, 0% bleed,
  against QLoRA 47% and 13 of 30 (QLoRA was memory-bound there); on 0.8B the edit 86% against LoRA 53% (chosen by
  validation) and 79% (4 epochs) [read, HomoHabilis docs/exp9]. On collateral damage the edit won wherever it was
  measured: 7 changed episodes of other tasks against 61, 99% top-token agreement against 93 to 95% [read, 27B
  harness]; 177 to 182 of 200 unrelated translations unchanged against 120 for LoRA [read, Dolci table].
- No comparison exists at 27B or above: "Training does not fit next to a 27B model on this GPU" [read].

**Fix.** "Where compared on small models, training gained more on the target when correct examples existed and
moved more of everything else; an edit beat a small adapter on fact insertion in one replication; nothing compares
them at 27B or above."

### 8. "It will not add abilities" (section o)
**Verdict: weakened. Severity: minor.**

True of general ability in every source I opened (the 27B harness: "Neither adds an ability"; 2606.13705: does not
"supply missing knowledge"). False if read as "edits only remove": new associations have been written without
gradients on healthy models (30 facts at 9B, above; alias and fact updates on Gemma 3 4B in IWIBench; skill names
of three procedures, 0 to 91 of 150 plans; the trigger-to-payload implant on gemma-4-31B-it at 79.6 to 81.3% in its
closed-form variant on query triggers [read, 2611.15118 Table 2]). The instrument's own vault edit is of this kind
and is loud. The ceiling's list of what to expect names only removal and scaling. **Fix.** Add "narrow written
associations, with bleed" to the expected finds, and note it is the class the safety section must watch.

### 9. "Three identical searches gave +29.45, +5.00 and +27.30" (section ii, pilot H)
**Verdict: refuted. Severity: minor.**

Not identical: "The Math Primary and Math Rerun 2 searches used a broad ability-improvement objective from the
start. Math Rerun 1 began with a loop-focused objective that was broadened after 46 successful requests" [read,
2611.09382 section 5]. The two same-objective searches gave +29.45 and +27.30; the outlier had another prompt for
46 of 50 requests. They are also on the 0.8B, though the proposal's bullet opens "On Qwen3.5-9B". The lesson that
one session decides nothing stands; the size of session variance is not shown by this source. Worth noting: the
prompt that named the fault did worst.

### 10. "The 7.9% baseline was a thinking-mode artefact (44.0 to 49.7% at the model card's settings)" (section ii)
**Verdict: weakened. Severity: minor.**

The 44.0 (floor) to 49.7 (decided) figure is the card's **non-thinking** settings. At the card's thinking-mode
settings (temperature 1.0, 81,920 tokens) the unedited model scored 22.3% with 75.0% of answers at the cap; at the
post's settings 7.2% [read, https://carlostoledo.co/reports/weight-space-programming.html, arms C1 to C3]. So
switching thinking off recovers the gain; a settings change inside thinking mode recovers under half. "576 of 579"
is from practice answers in the authors' logs [read, Kumar's post]. Hu and others' harness rules "making the model
skip its thinking" out of scope [read, prompt]. **Fix.** Say "with thinking off"; and let a brief fix thinking on
when the behaviour needs it, instead of always quoting a gain against the best setting.

### 11. "Closed frontier editors beat the unedited model on IWIBench; the three open-weights editors did not"
**Verdict: stands, loosely. Severity: minor.**

Pristine 104 of 551; open-weights editors 107, 115 and 85, with rewards 0.0548, 0.0464 and 0.0135 against 0.21 to
0.46 for closed ones [read, 2611.08647 Table 2]. Two of three ended a few items above the base. All seven closed
configurations are one vendor's; no Anthropic or Google editor was run, so the benchmark says nothing about the
driver family the owner's attended session uses. The failed-after-rewording example (all 64 terminal targets) is as
stated [read].

### 12. The weight-edit backdoor on gemma-4-31B-it (section viii)
**Verdict: stands; two things missing. Severity: minor.**

99.51% is one cell of eight (gradient-refined, observation trigger, two steps); the same cell's closed-form edit is
22.49%, and the query trigger gives 61.71% refined and 79.55% closed-form [read, 2611.15118 Table 2]. bf16 only.
Missing: (a) the paper has frontier agents **read the weights of edited gemma-4-31B-it checkpoints**: one full
checkpoint, no reference, weights and a runnable model, two hours; edited checkpoints judged edited 72.81% (GPT-6
Astra, which abstained on 99% of clean ones) and 56.14% (Claude Opus 5.5, which called 41.23% of edited ones clean)
[read, Table 5; the set is 19 checkpoints a model a trigger type]. That is a published prior for the proposal's
planted-fault exercise, with free code; typed tools should do worse. (b) "No public literature to recite" has a
shelf life: a planted fault shaped like this implant will be recitable by any driver trained after November 2026.

### 13. A 27B in a 4-bit format on an RTX 5090 in 19.6 GB (sections ii, xvi)
**Verdict: stands. Severity: minor.**

"Load: 40 seconds, 19.6 GB on a 32 GB GPU. Speed: 2.9 tokens per second"; one RTX 5090; teacher Claude (Fable 5.2)
[read, RESULTS.md, updated 11 November 2026]. Left out by the proposal: the edit is written **back into** the
4-bit format and measured there ("share of the requested change that the 4-bit format keeps: 1.03"; one weight
overflowed until a clamp was added); a prompt sentence matched the edit on the 27B's second task; and it publishes
wall-clock per pass, per fit and per edit, so "time and cost per test ... which no source gives" (section ii,
item 4) is true by model only. This instrument's recorded decode step is some four times faster a row and far
faster in a batch, which is the fair form of that claim.

### 14. Gemma 4 is Apache 2.0, not the Gemma terms (section viii)
**Verdict: stands. Severity: minor.**

The terms page says "For Gemma 4 terms, see the Gemma 4 license", last modified 1 April 2026; that page is the
plain Apache 2.0 text [read, ai.google.dev/gemma/terms and /gemma/docs/gemma_4_license]. The hub lists
`license:apache-2.0`, not gated, for google/gemma-4-31B-it and google/gemma-4-31B-it-qat-w4a16-ct [read, hub API].
One community repository says its Gemma 4 12B edits are "subject to the Gemma Terms of Use" and that the base is
gated; the hub API today says neither.

### 15. The absence claims, after a fresh search dated 14 November 2026 (claim 19)
**Verdict: stand only in their narrowest wording. Severity: serious for the public claim's wording.**

| Claim | What I found | Standing |
|---|---|---|
| No agent-driven weight edit above 9B in a preprint | arXiv listing for "weight-space programming": 15 entries, language-model targets 0.8B, 2B, 9B. Non-agent edits in preprints at 27B and 31B (2611.15118, 2606.13705); activation-level single-neuron work to 70B (2605.08513). Agent-driven at 27B only in repositories (the 27B harness; a 0.06% brand implant that "survives Q4_K_M") | stands, with "agent-driven" doing all the work |
| Nothing on Gemma-4-31B itself | Finding 1 (neuron edit, June 2026); the backdoor; agents auditing 31B weights (finding 12); abliteration builds | false if said without "agent-driven edit" |
| No edit measured across re-packing of a QAT checkpoint | No before-and-after measurement found. Nearest: a 31B QAT uncensored build shipped only as Q4_K_M and claiming 0/465 refusals (self-reported, method undisclosed); abliterated builds of the QAT-unquantized 31B measured unquantized only (11/100, KL 0.0365); a Gemma 4 12B single-neuron edit that keeps its shift in Q4_K_M against a matched Q4 original (not QAT); the 1.03 on NVFP4 | stands, narrowly |
| No harness with a causally tested neuron atlas as the agent's map | None found. Nearest: Neuronpedia's public MLP-neuron dashboards for Gemma 4 31B IT, 60 layers x 21,504, inference and steering on [read, docs.neuronpedia.org/neurons]; Goodfire's 35B neuron descriptions (no edit harness); an attribution tool in the 27B harness | stands |

## Too pessimistic or too optimistic?

Both, in different places.

**Too pessimistic** about what exists to be found: a loop fault of this model in thinking mode is published; at
27B, default and over-thinking edits moved held-out scores by 4 to 38 points with clean random controls; single
neurons chosen by attribution carry narrow behaviours at 12B and, as activation edits, up to 70B (though not on
this 31B's loops, and one such result failed held-out); written associations work. "Most
inquiries will end kept the base" is likely true of random hypotheses and likely false of hypotheses aimed at
defaults in thinking mode and agent formats.

**Too optimistic** about the instrument seeing and finding them: the fast path does not read long sampled
generation; the gate tests sets too small by the one published curve on this model; the localiser behind every
Gemma 4 result (one backward pass) is ruled out; the typed surface has no solver and no embedding edits, the two
things typed-only and 27B editors actually used; and the three "kept" 9B gains the ceiling quotes are 5, 18 and 1
items.

## What the proposal does not say at all

1. arXiv 2606.13705 and its public edited checkpoint (finding 1). The "importer for other people's published
   edits", listed as unbuilt until "the first outside edit worth re-measuring", has its first case now.
2. Google publishes the pre-quantisation QAT weights, google/gemma-4-31B-it-qat-q4_0-unquantized (28 April 2026,
   Apache-2.0, not gated; the W4A16 card names it as its base) [read, hub API]. Neither digest nor the synthesis
   names it. It is the object on which "what survives the grid" can be measured on the CPU with no baker on the
   card, and the object the community's QAT edits are made on.
3. Whether neuron indices found on the bf16 release (the loop set, the implant) mean the same on the QAT
   checkpoint. Nobody has measured it; the instrument can.
4. A long-generation reading with sampling and per-row seeds, its detectors and its price (finding 3).
5. arXiv 2605.08513 (one refusal neuron, 1.7B to 70B) as the standing counter-example to "single neurons rarely
   matter", and as a reason the single-column arm of pilot D must be behaviour-selected.
6. Neuronpedia's neuron dashboards for this model as the nearest public analogue, for section xvi's wording.
7. That the planted-fault exercise already has published numbers on this model (finding 12).

## Search log, 14 November 2026

arXiv listing API: "weight-space programming" (15 entries, newest 12 November); "intra-weight"; "weight surgery"
with LLM; "Gemma-4-31B", "Gemma 4 31B", "gemma-4-31B-it" (25 newest each); Gemma with "repetition loops"; single
neuron with refusal and MLP; "weight edit" with quantization and survive; "model editing" with
"quantization-aware"; neuron, atlas, weight, agent. Web search: agent weight edits at 27B, 31B, 32B, 70B;
Gemma-4-31B with weight-space programming or weight editing; an edit across re-quantisation of a QAT checkpoint;
an agent harness with a neuron atlas or dictionary; Neuronpedia's Gemma 4 31B IT neuron dashboards. Model hub API:
licence and gating of four Google repositories; the edited 31B and its conversions; three abliterated QAT builds.

Not checked: the PDF of the 27B working paper (page only); full text of 2605.08513 (abstract only); whether "L36"
in 2606.13705 counts from 0; X and Reddit threads; anything behind a login.
