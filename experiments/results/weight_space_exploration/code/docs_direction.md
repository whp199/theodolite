# What the documents say: direction, caveats, house style, licence

Reader's area: `README.md`, `ROADMAP.md` (all 1,645 lines), `COMPUTE.md`, `REFERENCES.md`, `VOLUNTEERS.md`, `NOTICE`,
`LICENSE`, `docs/` in `(the repository root)`, read at `main` = `e6ac64f` (14 November 2026, clean tree, seven
commits ahead of `origin/main`). Read only; nothing in the repository was changed.

How to read this report:

- **Layer counts.** The README and the pages count layers 1–60 ("page"). `ROADMAP.md` §vi, `COMPUTE.md`,
  `VOLUNTEERS.md` and every script count 0–59 (page = layer + 1; ROADMAP.md:334-335, COMPUTE.md:10-11,
  VOLUNTEERS.md:31). Below I write "page N" for the 1–60 count and "layer N (from 0)" for the other. This is the
  single easiest thing to get wrong.
- **"Document says" against "code does".** Everything here is what a document says unless marked *checked in code*.
  I checked three things in code: the scientist's tool list (`backend/scientist_tools.py`), the Lexicon's entry for an
  edit (`frontend/src/screens/Lexicon.tsx`), and `.gitignore`.
- I also read, outside my file list and only where the README points to them, `ARCHITECTURE.md` §4.6, §4.8 and §6,
  and four of the owner's working notes, which are not in the repository (direction, design, validity review, review
  before push). They are cited as such.
- `docs/` holds only `docs/readme/`, fourteen PNG pictures for the README (frontispiece, atlas, vault, cartouche and
  so on). There is no prose document in it.

---

## 1. Purpose and direction, and what the documents already say next to this feature

### 1.1 Stated purpose

- The README's epigraph: an atlas of all 1,290,240 neurons of Gemma-4-31B-it, a census of what a layer's neurons do
  when pushed, "and the instrument to read them with" (README.md:7). The model line: 60 layers, 21,504 neurons a
  layer, 5,376 dimensions, a vocabulary of 262,144 word-pieces (README.md:11).
- "The repository holds two things": **the instrument** (`backend/`, `frontend/`: a FastAPI server and a React
  interface reading the QAT W4A16 checkpoint through the J++ lens and the logit lens) and **the study**
  (`experiments/`: seven experiments, eight adversarial verifiers, a neuron-label validation and a lens refit, "314
  GPU jobs through a shared lab daemon") (README.md:119-129).
- The owner's direction, recorded in the owner's working notes (22 October 2026, updated 1 November):
  the future is "a general-purpose interpretability tool and learning aid"; the gender study is no longer a focus;
  instrument features, accuracy and teaching value come before study write-ups; the atlas with the census is "our
  main selling point". The README was reordered on 2 November to lead with the atlas (README.md:31).
- The README still gives the study a whole section (README.md:297-358) and says "Start with experiments/REPORT.md"
  (README.md:129). The study's records now serve as the provenance of what the instrument can and cannot trust.

### 1.2 What ROADMAP.md is, and what it is not

- It is a list of what is still open, "gathered from the places that already name it" (the README's fault table, the
  report's errata, the pages' "not yet" marks, R1–R4's records). "Nothing here has a date or an owner"
  (ROADMAP.md:3-6), although nearly every entry is in fact dated.
- Seven sections: i the lens; ii neuron labels; iii smaller pieces, now built; iv the server's defaults; v loose
  ends; vi the atlas (about 1,200 of the 1,645 lines); vii the CPU and GGUF.
- **It has no entry about a workbench, an environment, a harness, weight-space work, or an agent that edits
  weights.** A search of README, ROADMAP, COMPUTE, REFERENCES, VOLUNTEERS and NOTICE for "workbench", "harness",
  "weight-space", "weight space", "LoRA", "adapter" (in the LoRA sense) and "peft" finds nothing. The feature would be
  a new section, not the completion of a planned one.
- **ROADMAP.md never mentions the exports at all** (no line contains "export" or "control vector"). The control
  vector is in README.md:684-690 and VOLUNTEERS.md:304-348; the LoRA adapter and the silenced model copy are only in
  ARCHITECTURE.md §6 (ARCHITECTURE.md:1558, 1752-1859).

### 1.3 Weight edits and the vault, as documented

- The README's plain description: "The catalogue of the model's 1,290,240 neurons: look one up by the words it
  writes, change one, and read back what was asked before" (README.md:189-191). In the literature's words: "a
  LARQL/VINDEX-style catalogue of neurons, queried in Python" (README.md:259).
- What an edit is: `backend/larql_vault.py` calls the `larql`/`vindex` binaries only for `--version`; every query
  runs in Python. The edit, named after MEMIT in the code (".patch MEMIT"), "writes one neuron's down-projection
  column (rank one, no covariance solve; DELETE sets it to zero): it is not MEMIT, and the pages call it an edit"
  (README.md:460-463). Statements: DELETE, UPDATE, INSERT (ARCHITECTURE.md:598-599).
- **Rollback is documented only in two places.** REFERENCES.md:51 says the vault's "`EDGES/POLES NEAREST`, `.patch`
  and `ROLLBACK` syntax is this project's own, not LQL", and the Lexicon says "An edit stays until you undo it; one
  command undoes them all: ROLLBACK" (*checked in code*, frontend/src/screens/Lexicon.tsx:1577-1582) and "ROLLBACK
  clears every overlay" (Lexicon.tsx:1602). README.md and ROADMAP.md do not contain the word. The documents describe
  only an all-at-once rollback; a rollback of one edit is not described in any document I read (not established).
- **The voice.** Each edit "can be heard before and after on a question of your choosing": the vault writes the
  answer the likeliest way, up to 12 word-pieces by default (1–64), just before the edit and again just after, under
  the one GPU lock (README.md:463-467; ARCHITECTURE.md:595-606). One default heard edit took 12.6 s on the GPU
  (ARCHITECTURE.md:606; the date and the unpacker are not stated). The reply carries the KL of the first word-piece
  and where the two answers part (ARCHITECTURE.md:617-626); each patch record carries `edit_kind`
  ("one_neuron_column") and `replaces_patch` (ARCHITECTURE.md:627).
- **Its stated limits.** "The voice is one question, so it shows what an edit did to that answer only; the study
  tested only DELETE by cause and effect (E6 §2.3)" (README.md:467-469). UPDATE and INSERT "are still tested by cause
  and effect only as far as that one answer goes" (ROADMAP.md:185-186).
- **The example edit** (29 October 2026): neuron 7830 of page 60 turns "A spider has eight legs." into "six legs.";
  on seven questions it also gave a dog six legs and left an insect's six legs, Paris, the sky's colour and the
  model's self-account alone. The note beside it says "such an edit is rare: the most active neurons at layers 41,
  46, 51, 56 and 59 were not heard on this question at strengths 3 or 30", and that at the last layer "the example
  shows an edit heard, not where a fact is kept" (ROADMAP.md:171-186). The earlier example (neuron 100 of page 36)
  changed one of seven answers, by a space.
- The only edit tested by cause and effect in the study: deleting the vault's top ten "she" neurons at layer 51 and
  rolling back moved the pull toward "she" by 1.80 log-odds, against at most 0.04 for ten random neurons
  (Lexicon.tsx:1584-1586, *checked in code*).
- **The vault reads through the base-fitted lens.** Fault 9 lists "the words, the gathered/recalled split, the
  landmark bearings, the neuron labels, the nudge and the vault" as inheriting the J++ lens (README.md:383;
  ROADMAP.md:28-29). ARCHITECTURE.md:1772-1773 says the vault's UPDATE and INSERT use the J++ lens. So a
  word-targeted weight edit made through the vault is a lens-derived edit.
- Field notes are returned by `DESCRIBE POLE` since 12 November 2026; "the vault reads them and never writes them"
  (ROADMAP.md:127-132).

### 1.4 The scientist and agents, as documented

- "A **scientist** in the side panel reads the atlas with the same tools, runs pushes and texts, names what it finds
  and points at it on the map; it runs on Claude Code or Codex where they are installed, or on any OpenAI-compatible
  endpoint" (README.md:220-222). In the literature's words: "an LLM agent with the atlas's tools (MCP, or
  OpenAI-style function calling)" (README.md:266).
- **Its reach is narrow by default.** Under Claude Code it "reads, writes and edits files only in its own work folder
  and has no shell"; under Codex its shell runs in a sandbox that writes only there and has no network.
  `THEODOLITE_SCIENTIST_REACH=home` gives a shell over the home folder, pre-approved, and "the page asks once before
  the first such chat". Over an endpoint it has the atlas's tools only, and the person's messages and tool replies
  are sent to that endpoint (README.md:692-703; ROADMAP.md:1502-1518).
- At either reach no settings file, hook or CLAUDE.md is loaded (`--setting-sources ""`). The rules were tried on
  Claude Code 2.1.312 against a stand-in API, "not a live model" (README.md:698-701).
- **What it writes is a proposal.** `set_name` and `write_note` propose; nothing reads a proposal before the person
  keeps it, "later chats included" (README.md:641-643; ROADMAP.md:278-281).
- **Its tools.** README.md:239 says 30; ARCHITECTURE.md:1132 says 32; the code holds 35 (*checked in code*: the
  `TOOLS` list of `backend/scientist_tools.py`, lines 191-493). None of the 35 edits a weight or queries the vault.
  Its one hand-over is `export_control_vector`. The vault's MCP tool, `query_larql_vault(lql_query, voice)`, belongs
  to the older, separate server `backend/mcp_scientist.py`, with `inspect_pole`, `trace_downstream_readers`,
  `test_motif_hypothesis` and `causal_spot_check` (README.md:600-604; ROADMAP.md:165-168).
- An endpoint chat's loop runs at most 40 rounds a turn, each tool result cut to 16,000 characters
  (ARCHITECTURE.md:1082-1084).
- `steer` batches: "several prompts, or several candidate neurons, into one batch, which costs about the time of
  one" (README.md:685-686).
- Still open, in the document's own words: "no live model has yet been watched to see whether what it reads would
  lead it to try what the CLI refuses"; Codex can still read outside its folder; a scientist with a home shell "could
  keep its own proposal through the loopback port, which the guard cannot tell from the page" (ROADMAP.md:1514-1518).

### 1.5 The nearest thing to an environment for a model to do research

- **The lab daemon.** `experiments/lab/labd.py` is "the GPU daemon holding the model and lenses"; `labclient.py`
  submits jobs; `labkit.py` gives "lens readouts, gender scores, the corrected decomposition, interventions, exact
  same-position VJPs through the 4-bit model" (README.md:762). Jobs "use the pre-bound `rt`, `lab` and `STORE` and
  top-level `await`" (README.md:739-741). It loads in about 15 s (README.md:735). It "executes arbitrary submitted
  Python, and binds to 127.0.0.1 only", and `/exec` wants a token written at each start (README.md:487-490).
- It is what Claude Code agents used: "The experiments, analyses and reports in `experiments/` were designed, run and
  written by Claude Code agents (Anthropic), directed by the repository owner: parallel experiment agents,
  independent adversarial verifiers, and a separate lens-refit session" (README.md:819-823), 314 GPU jobs in all
  (README.md:128), with a frozen job log (README.md:764).
- So the repository already has two agent surfaces of opposite kinds: a code-execution daemon with the model held
  (wide, unguarded beyond loopback and a token, used by the study) and a tool-only scientist (narrow, guarded, used
  by the atlas). The documents do not join them.
- Gradients exist: "the lab's gradient passes have a GGUF twin of their W4A16 autograd function" (ROADMAP.md:1555-1556;
  COMPUTE.md:247-248).

### 1.6 Exports, as documented

- **Control vector** (30 October 2026; from the server since 11 November): the push tab's "save for llama.cpp" and
  the scientist's `export_control_vector` write into `<atlas folder>/exports/`; loaded with `llama-server
  --control-vector-scaled FILE:1` (README.md:686-690; VOLUNTEERS.md:304-348). Checked on the 31B for layer 45's two
  top neurons at ±1; at ±2 llama.cpp's text collapses because it also pushes the start mark (VOLUNTEERS.md:344-348).
- **LoRA adapter and silenced copy** (2 November 2026, `scripts/export_lora.py`, ARCHITECTURE.md:1752-1859): a
  neuron's mul and a vault edit as a rank-one LoRA on `blk.<l>.ffn_down` (A = e_nᵀ, B = (f − 1) W_down[:, n], or
  B = f v − W_down[:, n]); a head's scale as rank 256 or 512; a whole block silenced is "not a LoRA (full rank): a
  copy of the GGUF" (ARCHITECTURE.md:1783-1786). `--vault STATEMENT` builds the runtime on the CPU ("minutes and some
  20 GB on the 31B"), runs the vault's own `execute_lql`, reads the column, "and leaves the vault as it was"
  (ARCHITECTURE.md:1770-1774). Its check: 77 checks in about 50 s on the CPU.
- Stated as not measured: "a GPU backend's arithmetic for B (A x), quantised GGUFs other than that Q4_0, `--vault`
  on the real runtime, a silenced copy of the 31B, and what any of these pushes does to the text"
  (ARCHITECTURE.md:1857-1859).
- A direction or adapter made from another GGUF differs a little: cosine 0.997 for the QAT Q4_0, 0.977 at worst for
  the Q4_K files tried (VOLUNTEERS.md:330-331). lmstudio-community's QAT Q4_0 is "a quantisation of its own": none of
  its 410 linears is the checkpoint's, 68–71% of codes equal (ROADMAP.md:1590-1592).

---

## 2. Validity caveats a new feature must carry, with their numbers

### 2.1 The lens (fault 9), "the largest accuracy fault left"

- J++ was fitted on the **base** model (`google/gemma-4-31B`, bf16) from 64 WikiText prompts of up to 128 tokens and
  is applied to the instruct model in 4 bits (README.md:300-303).
- It agrees with the model's own average gradients at only cos 0.15–0.33 for the pronoun direction, pages 31–59
  (README.md:383; ROADMAP.md:26-28).
- "Good for comparing and poor on its own": from about page 21 it orders conditions as behaviour does (r ≈ 0.9 or
  better), but through page 50 its top word matches the model's next word on prose at most about 5% of the time
  (README.md:303-305).
- **Trust by depth** (the bands every page draws; ROADMAP.md:148-151): pages 1–20 the least trusted; 21–30 orderings
  only; 31–50 to be confirmed with an instruct-fitted reading; 51–59 counting only where an instruct reading agrees
  (errata E49, E50); at page 60 the lens is the identity (README.md:411-412).
- No refit is served. R2, R3 and R4 each served nothing; the nearest misses were +0.046 over the naked eye where
  0.05 was required (R1 at page 59), and R1's lead proved to be in function words (ROADMAP.md:44-89).
- **Rules for any lens served later** (ROADMAP.md:90-97): confirm contrasts at pages 31–50 with an instruct reading;
  show pronoun readings at 51–59 only where both agree; keep the caution at 11–30; re-validate the label gate for the
  new lens; judge without function-word targets.
- **The base-lens caveat is shown once.** The head and foot of every page name the lens, a red line beside each
  layer's words says what the reading can bear there, and "the opening page says once, among its cautions, that the
  lens was fitted on the base model": the owner's choice of 28 October 2026 (README.md:383; ROADMAP.md:29-33). The
  owner's working notes add: do not re-add per-page base-lens warnings, but keep the layer bands and the
  label gate and still flag new validity problems. `/api/health`, every sweep and the scientist's tool replies still
  carry the full `lens` object.
- "Results reached without a lens (what the model says, activation geometry, weights, patching, and steering
  measured on the output) are untouched by this caution" (README.md:305-306). A weight-space harness judged on the
  model's own outputs is on the untouched side; one that picks or labels edits by lens words is not.

### 2.2 The quiet carry channel

- About 180 early-loud residual dimensions are read through RMSNorm "with up to 1700× the gain of the others"; every
  Jacobian or gradient covector puts 99–100% of its energy in the quietest tenth of dimensions at pages 36–59; "that
  is why lens-derived steering vectors break the model" (README.md:334-337).
- Masking the 181 quiet dimensions alone is not enough: a nudge masked that way broke the answer at 0.005 × ‖h‖;
  with each pushed layer's quietest tenth masked too, it read fluently at 0.005–0.01 and broke at 0.02
  (README.md:396-398).
- The old nudge defaults (scale 650 at five layers, 2.4–7.4× the stream's strength) collapse the model
  (README.md:375). A gradient- or lens-derived weight edit is exposed to the same channel; the documents do not test
  that for weights (not established).

### 2.3 The label gate

- Neuron output labels predicted real effects only at the sampled pages 41–58; at pages 11–31 no single neuron had a
  stable effect to label; pages 32–40 were not sampled; input-side labels are "near-useless"; labels through an
  instruct-fitted lens are untested (README.md:338-346).
- The rule: below page 41, and wherever the lens and the naked eye share none of their first ten word-pieces, no page
  sets a neuron's words in its main reading; the apparatus keeps them with the reason (README.md:399-402).
- Its strength: the overlap as a graded score had AUC 0.84 within pages 41–58; as the yes/no cut the instrument
  chose, AUC 0.64. Of 31 labels sharing none, 7 were sound; of 209 sharing at least one, 160 were; labels sharing
  exactly one were sound 16 times in 40, below the band's 167 in 240 (README.md:402-406).
- Every label the server sends carries `label_check` = {`shared_top10`, `held_back`, `reasons`, `band`,
  `sampled_layer`, `in_measured_band`, `lens_validated`, `edited`} and replies carry a `label_gate` card
  (README.md:408-409). **A neuron silenced in the vault "writes nothing; it has no count and is held back (reason
  `writes_nothing`)"** (README.md:406-407), and `label_check` has an `edited` field: the documents already expect
  labels to change under a weight edit.
- The gate applies only through base-fitted J++ (erratum E47); labels at page 59 are untested.

### 2.4 A push is not the neuron

- A census push of 0.8 of the stream "is 2.1× the most this neuron can ever write by itself, so it shows what a push
  that size does, not what the neuron does when it fires" (README.md:94-95). ±1 of the stream is about 1.8 to 2.5
  write ceilings at layers 40–50 from 0 (VOLUNTEERS.md:339-340; ROADMAP.md:362).
- 1 of the stream is 1.76–2.76 ceilings at layers 39–51, 3.14 at 38, 1.21 at 57, 3.9 at 35–37, 4.2–5.0 at 26–34,
  5–19 at 2–25 (all from 0; ROADMAP.md:740-742).
- The study's 60-fold push overdrove single neurons at pages 21–31 (errata E43, E46; README.md:310-311;
  ROADMAP.md:1498-1501).

### 2.5 Calibrated doses

`census.DOSES_CALIBRATED` lists layers a design at a time; it grows only on "a passed trial and then the owner's
word" (ROADMAP.md:1038-1039, 982-989).

| Design | Doses | Counted calibrated at (from 0) | Pages | Since |
|---|---|---|---|---|
| `break_rel` | ±1 and ±2 of the stream's median length | 39–51 | 40–52 | 40–50 on 29 Oct; 39 and 51 on 12 Nov |
| `soft_ceil` | ±1.6 and ±2.3 write ceilings | 38 and 52 | 39 and 53 | 13 Nov |
| `match_ceil` | two counts per layer, 2.14–2.79 ceilings (1.0 and 1.4 nats) | 32–37 | 33–38 | 13 Nov |

- Only for answers of 32 or 48 word-pieces (`break_rel`) or 48 alone (the two ceiling designs) (ROADMAP.md:546-547,
  983-985).
- "The other thirty-nine layers (1–32 and 54–60) have no design that passed there" (pages; README.md:73).
  `match_ceil` failed at layer 31 from 0; `soft_ceil` failed at 37; `break_rel` failed at 38 and 52
  (ROADMAP.md:971-981, 830-847, 1013-1035).
- `main_rel`'s doses come from Qwen3.5-4B and are not calibrated on Gemma (ROADMAP.md:449-451).
- Where doses are not counted calibrated, the atlas marks it "doses not calibrated at this layer" in red "and inks
  it without gilt" (README.md:73-74).
- Whole layers taken: 40–50 from 0 (eleven, all gates pass on each: README.md:54-59), and layer 36 from 0 on 14
  November.
- **Layer 36 is undecided.** Read whole, "the whole reading passes, and the lower dose alone fails one gate by 0.006,
  so by its trial's standard the layer fails" (0.564 against 0.57); whether it stays counted calibrated "is the
  owner's choice; nothing was changed" (ROADMAP.md:1125-1164). The README still says eleven layers.

### 2.6 What "passed" means

- **The bars**, fixed and hashed before the first trial on 29 October: lead (mass AUC against controls) 0.60; lead
  beyond text survival (six covariates fitted out) 0.58; words picked on one half of the questions and scored on the
  other 0.57 (both directions); the new questions alone 0.56 (ROADMAP.md:494-498, 999).
- **A trial passes** when three readings all clear those bars at 48 word-pieces: the whole reading; the lower dose's
  answers alone; and the neurons whose six text-survival covariates all lie inside the controls' range (at least half
  of the neurons must) (ROADMAP.md:997-1003). A trial is about 1,022 neurons against 1,024 controls, 49,110 answers,
  76–80 minutes.
- **What a pass licenses**: a whole layer there, and nothing more. "It does not say ±1 and ±2 are the best doses for
  that layer" (ROADMAP.md:1036-1042).
- **What a lead says**: "a high rank says that the neuron's strongest words stand further above it than nearly every
  random direction's own do, so its push differs from a random one; not that its words share a theme"
  (ROADMAP.md:574-577; README.md:49-52).
- A figure on the bar is a fail: 0.599 against 0.60 "sits on the bar; the rule fixed beforehand counts that a fail,
  and it is one" (ROADMAP.md:753-755).
- **Nulls.** 2,048 isotropic random directions a layer (about 10 of 21,504 neurons beat every control by chance), and
  a second null of mixes of the layer's own write columns, which "read as random directions" at layer 45 (lead 0.502)
  (ROADMAP.md:550-551, 609-630). Compare layers by q ≤ 0.05 (3,534 to 9,133 neurons), not by the count above every
  control (95 to 4,794), which "hangs on each layer's one largest control" (ROADMAP.md:683-685).
- **Collapse words** (same, own, thought and their joins) are left out of what a page shows but still counted in
  mass, p and q (ROADMAP.md:554-573).

### 2.7 The steer lens

- Built 13 November 2026 at a dose for each layer. "It is sound from layer 39 up and a weak guide below, and the page
  now says which" (from 0; ROADMAP.md:380-384).
- Agreement with the census: 85% at layer 40, 94–97% at 41–44, 98–99.5% at 45–50 (1.6–9.7% shuffled); 62% at 39,
  33% at 38, 20% at 37, 13–17% at 34–36, 7–10% at 32–33 (ROADMAP.md:385-395). At layer 36, whole: 0.185 against
  0.034 shuffled (ROADMAP.md:1154-1157).
- At layers 0–30, 4 to 7.5 of a neuron's twelve word-pieces are shared across the layer whichever neuron is pushed
  (ROADMAP.md:396-401). At thirty-nine layers (0–38) a neuron's line of words and the word search leave the steer
  lens out (ROADMAP.md:419-421).
- "The agreement was never a test of what a neuron does unpushed: it is two readings of the same push"
  (ROADMAP.md:412-413).

### 2.8 Other caveats a harness would meet

- **Wiring signs.** A link's strength and sign come from weights alone. Measured on the 31B (28–29 October), the sign
  from rest agreed with the measured change at 0.336 of links and the sign in use as served at 0.418, "below chance,
  links and null alike", because about nine readers in ten sat with the gate shut; scored at each reader's own
  operating point the sign in use agreed at 0.841 and size followed the first-order term at ρ 0.89; linked readers
  moved 5.5 times the null's. "Read a link's strength as a lead, and its sign as holding where its reader's gate is
  open" (README.md:417-428). This is the document's own evidence on how far weights alone predict behaviour.
- **What builds a neuron** (drive check, 29 October, 42 neurons): removing a named head or neuron moved the target
  the predicted way 94% and 95% of the time, by a median 0.89 and 0.93 of the predicted part; but only 0.57 (heads)
  and 0.37 (neurons) for leads written five or more layers back; untested where a neuron fires weakly
  (ROADMAP.md:237-277).
- **The writer's floor.** On the 5090, an unpushed answer written in a census batch of 64 first parted from itself
  written alone after 22 word-pieces; in a batch of 32 after 54; alone it held all 64 (28 October;
  ROADMAP.md:1427-1434). "The batch fixes the bits, the unpacker does not" (COMPUTE.md:89-92). A before-and-after
  comparison must be written in one batch shape.
- **Determinism.** On one card, across processes, a run is bit for bit itself (29 October; ROADMAP.md:1435-1438).
  Another card class differs: log-probabilities by a median 0.013–0.015, and greedy answers part at near-ties
  written at p = 0.50, 0.41, 0.50 (ROADMAP.md:672-679).
- **Passages** are "gathered on the 31B, and not yet validated as labels" (README.md:85-87, 381).
- **Pattern tests** (14 November, 21 faces): readers picked 177 of 196 pairs with a face's own pattern and 86 with
  another's; against near misses 158 of 196; 22 of 63 written texts fired and no near miss did; above layer 2 the
  phrase alone fired a neuron at 0 of 68 places. Readers were "fresh Claude conversations run by hand, because no
  endpoint is configured on this machine; the product's own path, an endpoint's model, has run only against a stub";
  patterns were written by the session that built the tests (ROADMAP.md:1233-1381).
- **The guess test.** `test_motif_hypothesis` once said "yes, clearly" at one sentence a side; it now gives Welch's
  t, a permutation p, no judgement below four sentences a side, and Benjamini–Hochberg across neurons named in one
  call. "Nothing corrects for tests run in separate calls or for sentences tried and changed"
  (ROADMAP.md:206-229). That is the document's own statement of the agent-fooling-itself gap.
- **Weights provenance.** A reading off the GPU, in float32 or through int4mm is recorded under its own model id;
  "every index also records the weights it was built from, and none built from other weights is read"
  (README.md:546-550); a census run is not continued, merged or joined across weights (ROADMAP.md:1422-1424).

---

## 3. House style

### 3.1 Document forms

- **Sections**: lower-case roman numeral, a middle dot, a lower-case title: `## i · the atlas`, `## v · how far to
  trust it`, `## vii · the shelves`, `## viii · the makers' marks` (README.md:31, 362, 753, 800). ROADMAP.md runs
  i–vii, COMPUTE.md i–viii, each divided by `<br>`. REFERENCES.md uses letters A–J; VOLUNTEERS.md plain headings;
  ARCHITECTURE.md arabic numbers (`### 4.7 The atlas`).
- **Cross-references**: "ROADMAP §vi", "COMPUTE §ii", "ARCHITECTURE §4.7", "README §v", "REPORT §3.3", "erratum E47".
- **An italic epigraph** under each title says what the file is and when it was gathered; ROADMAP.md's `<sub>` line
  records each time a section was gathered again and at which commit (ROADMAP.md:3-20; COMPUTE.md:3-11).
- **How a new section is added**: at the end with the next numeral, noted in the gathering line ("section vii, new",
  ROADMAP.md:13); the same heading can appear in two files (ROADMAP §vii and COMPUTE §vii are both "the CPU and
  GGUF"). The README's atlas section was put first and the others renumbered ii–viii, with the nav line of anchors
  updated (README.md:14-23). A weight-space section would be ROADMAP "viii", a new README section with the nav
  updated, an ARCHITECTURE "§7" (or §4.9 for its routes), and possibly a COMPUTE row in §i.
- **An entry's form in ROADMAP**: a bold lead phrase and a status word, then dated prose. Status words in use:
  "**Built.**", "**Built, and timed.**", "**Built, and gathered on the model; not yet validated as labels.**",
  "**Done**", "**Yes.**", "**In part.**", "**Mended in the instrument.**", "**Applied**", "**Open, and the largest
  accuracy fault left; now labelled.**", "Not built.", "Still open:", "*Not done:*", "*Not shown yet:*",
  "*Checked:*", "What it does not show.", "What this leaves open."
- Every claim says **where it was checked** ("on the CPU stand-in", "on the 31B", "driven in a headless browser")
  and **when**, and separates "**measured**" from "inferred", "arithmetic", "worked out", "estimated, not timed".
- **The owner's decisions are quoted and dated**: "the owner's word, 13 November: “Yes, design it”"; "the owner's
  choice of 28 October 2026".
- **Registrations**: "registered in `08fd50b`", "hashed at 02:18Z before any answer of these layers existed", then
  "Said beforehand and met" / "Said beforehand and missed", and every later change "filed as a deviation".
- **Commit messages** are whole sentences that state the result with its figures ("Layer 36 read whole: the whole
  reading passes, the lower dose alone fails one gate by 0.006").
- British spelling (licence, colour, quantisation, labelled, centre, mend). No emojis. Tables for comparisons.
  GitHub alert blocks (`> [!CAUTION]`, `> [!WARNING]`, `> [!IMPORTANT]`) for dangers.

### 3.2 The vocabulary

The README's own table, "The instrument's words, and what they name" (README.md:247-266):

| In the instrument | In the literature |
|---|---|
| a sighting | the stream at one word-piece, read through all 60 layers |
| the lens · the naked eye | the J++ readout · the logit lens |
| gathering · remembering | the attention write · the feed-forward write |
| a neuron's two faces | the positive and negative poles of a GeGLU neuron |
| threads | a later neuron whose gate reads an earlier neuron's write |
| landmarks | the survey frame |
| the nudge | additive activation steering on the residual stream |
| the vault | a catalogue of neurons, queried in Python |
| how it writes | decoding |
| the atlas | a neuron explorer |
| a push · the three voices | steering along one neuron's write direction · up, unpushed, down |
| the census | a steering sweep of every neuron of a layer, ranked against random controls |
| the steer lens | mean Δ log p of the next token under a push |
| the web | the neuron's connection cone |
| the scientist | an LLM agent with the atlas's tools |

Further words in steady use:

- **Things**: leaf (a layer's page), folio (a neuron's page), sheet (a layer on the map), the stream (residual
  stream), word-piece (never "token" in prose), the card (the GPU), the home card, the stand-in (a tiny random Gemma
  4 on the CPU), passages, a pattern, the spread, the cone, the field book, names and notes, proposals, the board,
  a kit, a trial, a walk, a canary, the writer's floor, a write ceiling, a dose, a design, gates and bars, a lead,
  an edit, an edit's voice, the shelves, the makers' marks, the frontispiece, the cartouche. Code names in the same
  register: `Incipit.tsx`, `Alidade.tsx`.
- **Verbs and phrases**: pushed up / left alone / pushed down; "heard" (an edit); "held back" (a label); "inked
  without gilt"; "counted calibrated"; "mended"; "taken" (a census, a canary); "unpack" (dequantise); "the same bits",
  "bit for bit"; "refused in plain words"; "read as a lead, never as a finding"; "a lead, not a proof".
- **The two readings of every page**: *rubrics*, red italic directions for a first reading, and *the apparatus*,
  the technical name, formula, raw values, sources and every control under a double gilt rule; "nothing is hidden
  from either reader" (README.md:136-139). A new page must have both.
- **The Lexicon** (`#/lexicon`) defines every word "in plain terms, with its formula, and the cautions on where it
  can mislead you" (README.md:195-196). Its index (*checked in code*, `frontend/src/screens/lexicon/entries.ts`)
  already has "edit, an", "push, a", "census, the", "scientist, the", "matched control", "test of a label, a",
  "pattern put to the test, a". A new feature gets Lexicon entries and a Lineage note.

### 3.3 Tone, from the owner's working notes

- The look: vellum, ink, rubric red, gilt, lapis (lapis = attention, gilt = feed-forward); EB Garamond with true
  small caps, IBM Plex Mono only for code; hairlines, not cards or shadows; words, not icons, for navigation.
- The era to evoke is "when scientific instruments were both powerful and beautiful": brass, engraved plates,
  instrument-makers' cartouches, printed treatises, critical editions. Surveying and printing metaphors.
- **No prayer-book, liturgical or devotional analogies.** Rubrics are explained as scribes' and early printers' red
  headings.
- "A learning tool and an experiment tool at once: never scare novices, never hide features from expert scholars."
- Every number shown in the interface is traceable to a results file.

### 3.4 Naming a new feature in keeping

- A plain noun from surveying, instrument-making, printing or the scholar's study, lower-case, with a definite
  article ("the vault", "the census", "the atlas"), paired with its literature name in the README table. The
  feature's parts want plain words too: an edit is already "an edit"; a hypothesis is "a guess" ("put a guess to the
  test": "The guess test's statistics", ROADMAP.md:206, and the comment at
  frontend/src/screens/lexicon/entries.ts:24); a result is "a lead" until tested; a control is "a matched control".
- Avoid in prose: token, steer, ablate, SOTA-style adjectives, and the name of a method the thing is not (the pages
  refuse to call the vault's edit MEMIT).

---

## 4. Hardware, time and money (COMPUTE.md), and the rented-card path

### 4.1 The machine

- One NVIDIA GPU with ≥ 32 GB for the instrument; the study's RTX 5090 has 32,607 MiB and "the instrument peaks near
  23 GB" (README.md:477). The home machine's CPU is an AMD Ryzen 9 7900X, 12 cores (COMPUTE.md:258).
- Since kit 4 (14 November) a headless 24 GB card can take a census layer: the 4-bit model rests at about 18 GiB and
  a batch takes up to about 4.2 GiB more (VOLUNTEERS.md:63-67). The six worst batches fit with 217 MiB to spare at
  23,900 MiB free (ROADMAP.md:1109-1112).
- Disk: about 22 GB for the model, 6.4 GB for the two lenses, up to about 27 GB more for cached activations
  (README.md:479-480). A census layer keeps 2.7 GB (ROADMAP.md:711).
- **"Run one GPU process at a time"**: the lab daemon, the server and a standalone `mcp_scientist` each load the 31B
  (README.md:484-485). The server holds one GPU lock for a whole sweep (README.md:599-600).
- **Every forward unpacks all 29.3 billion 4-bit weights afresh** (COMPUTE.md:28-32; README.md:648). 410 linears
  (ROADMAP.md:444).
- **Dense weights do not fit the card.** `resident` (weights kept unpacked) needs "about 60 GB free beside the 4-bit
  weights (an 80 GB card holds it, with little left for a batch)" (README.md:656-657). On the CPU, resident bf16
  takes about 62 GB of RAM; packed about 20 GB (README.md:531-533).

### 4.2 Time on the home 5090

| What | Time | Where, when |
|---|---|---|
| A decode step, `reference` unpacker | 0.706–0.743 s, whatever the batch (1–96 rows) | COMPUTE.md:52-57, 28 Oct 2026 |
| A decode step, `lut` | 0.275–0.303 s | same |
| A decode step, `compiled` | 0.091 s at 1 row, 0.103 at 32, 0.109 at 64, 0.126 at 96 | same |
| torch's peak above the model | 0.3–1.0 GiB at 1 row, 2.4–3.1 at 32, 4.8–5.3 at 64, 7.2–7.5 at 96 | same |
| `compiled`'s graphs at load | about 4 s | COMPUTE.md:59; README.md:653 |
| Three pushed answers of 80 word-pieces | 8.3 s with `compiled`; 50–54 s with `reference` | COMPUTE.md:61-64, 11 Nov |
| One heard vault edit (two answers of 12 word-pieces) | 12.6 s | ARCHITECTURE.md:606, date and unpacker not stated |
| Lab daemon load | about 15 s | README.md:735 |
| A GGUF loaded packed | about 21 s, into 19–21 GiB | README.md:540-542, 2 Nov |
| A written word-piece from a GGUF | 0.54–0.82 s reference, 0.121–0.126 s compiled | README.md:542-544 |
| A census trial (1,022 + 1,024 units, 49,110 answers) | 68–86 minutes | ROADMAP.md:997-998, 857, 915 |
| A whole census layer (565,254 answers) | 13.2 h at 11.88 answers a second; layer 36 took 15.5 h | ROADMAP.md:579-582, 1126-1127 |
| Controls alone (2,048, 49,158 answers) | about 73 minutes | COMPUTE.md:19 |
| Both calibration ladders on six layers | 65 and 49 minutes | COMPUTE.md:21, 28 Oct |
| The steer lens, 60 layers | 10.5 h (37,743 s); its dose pilot 38 minutes; a fine pilot of ten layers 10 minutes | COMPUTE.md:169-170; ROADMAP.md:874 |
| A corpus pass, 2,000,054 word-pieces | 58 minutes (stats, `reference`); 20 minutes (passages, `compiled`) | COMPUTE.md:23, 27–28 Oct |
| The wiring (weight products) | about 4 minutes; its measured check, 59 layers, about 31 minutes | COMPUTE.md:24 |
| The lens index | 144 s; the layout 9 minutes | ROADMAP.md:328-330, 27 Oct |
| The writer's floor | 2 minutes with `compiled` | COMPUTE.md:25 |
| Home canaries of 49 layers | seven minutes | ROADMAP.md:1066 |
| Other-card bars for ten layers | 20 minutes | ROADMAP.md:1172 |
| The drive check (42 neurons) | 10.5 minutes, reference | ROADMAP.md:243 |
| The pattern-test pilot (176 faces surveyed, 21 tested) | 34 minutes | ROADMAP.md:1233 |
| A label survey of pages 41–59 | about 2 × 10²⁰ multiply-adds: some 17 days here, 3.7 days of one H100 (inferred) | COMPUTE.md:26; ROADMAP.md:120 |

On the CPU: a prefill of the packed checkpoint took 31–37 s on the 12-core desktop; through `int4mm` the 31B wrote a
word-piece in 0.45 s from the QAT Q4_0 and took a 59-piece prefill in 3.5 s (2 November), but "its sums are the
kernel's, not the reference's" and its readings carry `+int4mm` in their model id (README.md:531-539, 546-548;
COMPUTE.md:255-272). "The CPU suits weight reads, single forwards, pilots and looking through int4mm now"
(COMPUTE.md:270-271). The exporters read two tensors of a layer on the CPU "in a few seconds"
(ARCHITECTURE.md:1572).

### 4.3 What is not allowed to change a reading

"Not for a census of record without more": a fused 4-bit matrix product, vLLM's steering fork, a static cache with
CUDA graphs, compiling the whole model, splitting a model across cards. "None gives the same bits." Each could serve
a run "marked as from another engine, after an equivalence check registered before it is read" (COMPUTE.md:97-102).

### 4.4 The rented-card path

- **Option zero** is the home 5090 headless: "It costs nothing, and it is the study's own card class"
  (COMPUTE.md:108-112).
- Rented, as read on 28 October 2026 through web search ("read the live page before asking for a budget"): RunPod
  Secure RTX 5090 $1.19/h for an interactive session; H200 141 GB $5.29/h for batch jobs (about $4.00 elsewhere);
  **RTX PRO 6000 96 GB $2.49/h "if `resident` is wanted on one card": "the cheapest card that holds the unpacked
  weights with room for a batch"** (COMPUTE.md:117-121).
- `scripts/pod/`: `setup.sh` (everything on one network volume), `pilot.sh` (a one-hour pilot ending in "proceed" or
  "stop"), `census_shard.sh`, `sync.sh`; `scripts/gpu_log.py` records approvals. "None has yet run on a rented card"
  (COMPUTE.md:186-195; README.md:705-728).
- The page is reached "through a tunnel, never a port"; no scientist on a batch pod; never
  `THEODOLITE_SCIENTIST_REACH=home` on a machine whose environment may hold cloud credentials (COMPUTE.md:205-212).
- "A new card class is a new instrument until checked" (COMPUTE.md:213-216).

### 4.5 The budget rule (COMPUTE.md:220-235)

1. The owner approves ahead, in GPU-hours on a named card and provider, with a dollar ceiling at that day's rate
   ("12 H200-hours on RunPod Secure at ≤ $5.29/h: ≤ $64"), plus storage, written beside the job before it runs.
2. A pilot on the same card class first; its projection × 1.25 must fit the approval, "else stop and ask".
3. Every pod logged; the write-up ends "used X of the Y GPU-hours ($a of $b) the owner approved".
4. Hard stops: prepaid credit no larger than the approval; each batch job under `timeout` at 1.25 × its projection.
5. A change of card or provider after the pilot is a deviation.

Precedent: R3 "used 5.54 of the 8 GPU-hours the user approved" (ROADMAP.md:71).

---

## 5. Licence and terms

- **The code**: Apache License 2.0 (`LICENSE`, the standard text, 201 lines; README.md:825). Copyright "2026 The
  Theodolite authors" (NOTICE:2).
- **What is not redistributed**: "This repository does NOT redistribute any model weights, lens tensors, third-party
  binaries, or third-party text corpora. It contains code that downloads and uses the third-party works listed
  below, plus research results (statistics and model generations) computed with them" (NOTICE:7-10; README.md:826-827).
- **Gemma 4**: NOTICE lists `google/gemma-4-31B-it-qat-w4a16-ct` as "Apache License 2.0 -
  https://ai.google.dev/gemma/docs/gemma_4_license" (NOTICE:15-16), and REFERENCES.md:9 says "License (Apache 2.0)".
  "Gemma is a trademark of Google LLC. This project is independent research and is not affiliated with or endorsed
  by Google" (NOTICE:19-20; README.md:827). Model output in `experiments/results/` is labelled as generated by
  Gemma 4.
- **A loose thread on Gemma's terms.** COMPUTE.md:299-300 lists "Gemma 4's licence and whether its repositories are
  gated" as read from a third-party tracker and "unsettled, since huggingface.co could not be reached to look";
  COMPUTE.md:201-204 says whether the checkpoint needs a token is unsettled. NOTICE and REFERENCES state Apache 2.0
  flatly. I did not open Google's licence page; the licence as the documents give it is not independently verified
  here.
- **MIT exception (neuron-explorer)**: parts of `backend/scientist_prompt.md`, `scientist_methodology.md`,
  `scientist_chat.py`, `scientist_tools.py`, `scientist_mcp.py`, `atlas*.py`, `census.py`, the census scripts and
  `frontend/src/screens/atlas/` adapt its code or prompts under MIT, whose notice is reproduced (NOTICE:117-168). A
  harness that extends the scientist's prompt or tools extends MIT-adapted files and must keep that notice.
- **CC BY-SA 4.0 exception**: E3's prose stimuli (NOTICE:110-115).
- **Lenses**: J++ and the Jacobian lens are Apache-2.0, downloaded at run time (NOTICE:22-27, 76-82). The Miru lens
  is "licence 'other', named Unlicense on its card", read from the local cache, only its sha256 committed
  (NOTICE:69-74).
- **Corpora**: the N1B corpus and verbatim contexts are not included (NOTICE:170-179). The atlas's stored passages
  "are the corpus's own text: they stay in the atlas folder and are never committed" (README.md:671-672).
- **Volunteers' data**: "what you send is published with the project (Apache-2.0)" (VOLUNTEERS.md:287); "Open data:
  every answer of every layer ... to be published with the project" (VOLUNTEERS.md:54).
- **Distributing edited weights or adapters: no document addresses it.** What the documents show:
  - Exports are written to `<atlas folder>/exports/`, and the atlas folder is git-ignored (README.md:635, 688).
    `.gitignore` (*checked*) also ignores `*.pt`, `*.npz`, `*.npy` under `experiments/results/`.
  - A control vector holds a direction computed from Gemma's weights; a LoRA adapter's B holds Gemma's own weight
    columns scaled ((f − 1) W_down[:, n]); a silenced copy is a whole GGUF of the model (ARCHITECTURE.md:1783-1786).
    Each is a derivative of the model's weights, and the last is the model.
  - Each export records its provenance under `theodolite.*`: the model, the weights read, each entry, the stats
    index's sha256, the commit and the time (ARCHITECTURE.md:1597-1600, 1795-1796).
  - VOLUNTEERS.md invites people to export control vectors for their own use (VOLUNTEERS.md:55-56, 304-311); it does
    not speak of publishing them.
  - Whether the project would publish an edited checkpoint, an adapter or a silenced copy, and under what notice, is
    not established. Under the documents' own reading of Gemma 4 as Apache 2.0, redistribution of derivatives would
    need the licence and notices carried and changes stated (Apache-2.0 §4, LICENSE:89-128), and the trademark is
    not granted (§6, LICENSE:138-141); that is my reading of the licence text, not a statement of any document. The
    repository's present stance is that it redistributes no weights.

---

## 6. Unbuilt or half-built items next to this feature, and what would conflict

### 6.1 Unbuilt or half-built

- **Running tests from the page**: "A person cannot yet run either test from the page; the scientist can, and so can
  a script" (ROADMAP.md:1389).
- **Endpoint judges**: the blind test's reader is meant to be an endpoint's model in a fresh conversation; "no
  endpoint is configured on this machine; the product's own path, an endpoint's model, has run only against a stub"
  (ROADMAP.md:1224-1227, 1268-1270).
- **Layers without calibration**: thirty-nine pages (1–32 and 54–60) have no design that passed (README.md:73);
  layer 36 from 0 is undecided (ROADMAP.md:1158-1164); layers 53–59 from 0 are named in the kit but none trialled
  (ROADMAP.md:1072-1074).
- **The steer lens below layer 39**: "The repair not yet tried is a lens read against its layer ... That is a
  rebuild (10.5 hours); a trial's 1,022 neurons at one layer would pilot it in minutes" (ROADMAP.md:424-426).
- **A retest design**: a screen's strongest neurons cannot be retested with `break_rel` (a neuron writes its picked
  answer again, 99.6% bit for bit); "A retest needs a design whose questions no screen asked. Whether to build one is
  still to choose" (ROADMAP.md:648-651).
- **Sparse mixes as a null**: "A mix of a few neurons would ask how many it takes to lose the lead. Not built"
  (ROADMAP.md:628-630).
- **A per-row null for the census search**: "needs more controls than 2,048 a layer, or a statistic pooled over
  layers; neither is built" (ROADMAP.md:1412-1413).
- **A label survey**: "nothing tests every label"; "no resumable driver is written yet" (ROADMAP.md:120;
  COMPUTE.md:180).
- **The vault**: UPDATE and INSERT tested only as far as one answer (ROADMAP.md:185-186); the example probe "did not
  record whether the neuron fires on the questions left alone" (ROADMAP.md:184-185).
- **The exports**: the LoRA export of a vault edit on the real runtime, a silenced copy of the 31B, and what any
  mul, head or vault export does to text are not measured (ARCHITECTURE.md:1857-1859).
- **The scientist**: no live model watched against the reach rules; rules tried on one CLI version only
  (ROADMAP.md:1514-1518). Nothing corrects for hypothesis tests run in separate calls (ROADMAP.md:226-227).
- **Base against instruct weights**: "watch a neuron form across training checkpoints: no intermediate checkpoints
  of Gemma 4 31B are known; the base model against the instruct one is the nearest substitute, and needs the base
  model and the GPU" (ROADMAP.md:133-134). This is the roadmap's one weight-space research item, and it is unbuilt.
- **A resting operating point for the wiring's signs**: "whether to build one is the owner's choice"
  (ROADMAP.md:1483-1486).
- **Modes and same-subject controls** for pattern tests: not built (ROADMAP.md:1383-1391).
- **Rented cards**: none of `scripts/pod/` has run on one (COMPUTE.md:194-195).
- **The CPU and GGUF**: int4mm's canary on a real file not run; a sighting on the CPU not run; splitting a model
  between card and CPU not built; "Beyond the 31B. Refused today" (ROADMAP.md:1597-1645).
- **The second reading** (an instruct lens beside J++) is built on the unmerged branch `it-second-reading` and
  serves nothing (ROADMAP.md:39-43, 99-102).

### 6.2 What a weight-space harness would conflict with, or supersede

- **One GPU process, one lock** (README.md:484-485). A harness cannot share the card with a census run (13–15 hours
  a layer), a volunteer intake's spot-check ("with the server stopped", VOLUNTEERS.md:360) or the lab daemon, and
  the lab daemon's default port is the server's (8765).
- **The scientist's reach rules** (README.md:692-703). The weight-space-programming idea rests on code execution;
  the scientist under Claude Code has no shell by default, and the documented way to give it one (`REACH=home`) is
  the one the README warns about. The lab daemon runs arbitrary Python but is the study's, not the scientist's.
- **Pinned files.** `engine_runtime.py` is pinned for R4's record ("None of it touches the R4-pinned
  `engine_runtime.py`", COMPUTE.md:46-47; ROADMAP.md:1609-1611); `theodolite_math.py` is pinned with the 31B's
  shapes (ROADMAP.md:1641-1642); R3's jobs refuse a `backend/` that differs from `d5405ee` (ROADMAP.md:103-104).
  New weight machinery has to be installed beside them, as `backend/w4a16_fast.py` was.
- **The volunteer kit.** "A change of any of them is a new kit" (VOLUNTEERS.md:369); a change to "what writes
  answers" is a new kit (ROADMAP.md:735-736). Two layers are running on volunteers' cards now. Anything that alters
  the census's writer or the loader must leave kit 4's answers bit for bit.
- **"The same bits."** Readings of record are made on the reference arithmetic or something tested equal to it;
  anything else is "another engine" under its own model id (COMPUTE.md:97-102; README.md:546-550). An edited model is
  other weights: by the documents' own rule its readings and indexes should not mix with the unedited model's.
- **The vault's naming.** The pages insist the edit "is not MEMIT" while the code keeps ".patch MEMIT" for old
  callers (README.md:461-463; ROADMAP.md:169-170). A harness that adds a real covariance-solved or multi-neuron edit
  would need a name that keeps the two apart, and would supersede the Lexicon's line that an edit "rewrites one
  neuron's output column and solves for nothing else".
- **Proposals.** The scientist's names are proposals until kept (ROADMAP.md:278-281). By the same rule an agent's
  weight edit would be a proposal until the person keeps it; an agent that could keep its own edit would break the
  pattern, and the document already notes that a home-shell scientist could keep its own proposal through loopback.
- **The validity note.** Per the owner's working notes, every multi-agent review gets a reader who asks whether the core
  measurement is valid, and "documented as known" may not downgrade a problem that default-visible text hides.
- **Review before push.** Per the owner's working notes, side sessions commit on their own branches and do not push;
  merges and pushes go through the coordinating chat after review, on the owner's word.

---

## 7. References closest to this feature, and how prior work is recorded

### 7.1 Closest entries in REFERENCES.md

- **[10] Hay, LARQL / VINDEX3** (REFERENCES.md:50-51): the vault's naming and "patch overlays"; the project's
  `.patch` and `ROLLBACK` syntax is its own. REPORT §3.3 quotes LARQL's warning "that a projected contribution is not
  a counterfactual".
- **[11] MEMIT** (REFERENCES.md:53-54): "named only"; the tool's edit "is **not** MEMIT".
- **[12] ROME** (REFERENCES.md:56-57): "background for rank-one weight edits and causal tracing".
- **[14] Elhage et al., A Mathematical Framework** (REFERENCES.md:62-63): the residual-stream view and the
  weight-based "virtual weight" coupling behind threads and the direct-path estimate.
- **[9] Cammarata, Neural Networks Are Already Clean** (REFERENCES.md:44-48) and **[54] Tigges et al., Neurons as an
  Interpretable Basis** (REFERENCES.md:179-181): neurons as the native units; [54] tests a written description by
  whether it predicts the neuron (the pattern tests), and "never pushes a neuron to describe it".
- **[47] neuron-explorer** (REFERENCES.md:171-174): the atlas, the census and "the scientist chat with its tools and
  report formats".
- **[24] Park et al.** (REFERENCES.md:92-93): the covector-against-vector distinction behind the "units error" rule
  for lens- or gradient-derived steering.
- **[21], [22]** activation steering; **[23]** activation patching; **[25]** Bills et al., LLMs explaining neurons.
- **[36] interp-engine**, **[37] FastMCP**, **[39] compressed-tensors**: the runtime, the MCP server and the W4A16
  loader.
- **[49]–[53], section J**: no published neuron atlas of Gemma-4-31B was found on 30 October 2026.

### 7.2 What is missing

- **The weight-space-programming paper (Takehi, Teoh, Zhang, Khattab, November 2026) is not in REFERENCES.md, nor
  anywhere in the repository** (a search for "Takehi", "Khattab", "weight-space programming" finds nothing).
- There is no entry for llama.cpp, for control vectors as a format, or for LoRA, although ARCHITECTURE §6 reads
  llama.cpp's source at commit `cd26896` and writes both formats.

### 7.3 How prior work is recorded

- Numbered entries whose bracketed numbers are stable across the reports, the README and the Lineage page; numbers
  follow the order of adding, not the section (so [54] stands in section I after [48], with [49]–[53] in J).
- Lettered sections by kind; each entry has a "*Use:*" line saying exactly what the repository takes from it and,
  where relevant, what it does **not** take ("Not taken: its simulation, its subcircuits and modes").
- Each section added later opens with a dated line saying when it was written and whether its links were opened
  ("These entries were written on 2026-10-27. The repository of [47] was read from a clone made that day; the link
  of [48] was not opened"). Code is cited at the commit read. A source's own figures are marked unchecked ("Its
  figures are its own and none is checked here").
- Two sections record work *not* relied on: C "Related work (acknowledged; not relied on)" and J "Prior work on
  Gemma-4-31B's neurons (searched; none relied on)", the latter with the date and the places searched.
- Adapted code and its licence go in `NOTICE`, file by file. The README's "makers' marks" names the pillars
  (README.md:807-817), and the interface's Lineage page "tells the same story".
- A new source that suggested a feature gets a paragraph in the ROADMAP entry that built it ("Goodfire's *Neurons as
  an Interpretable Basis* ... Two of its tests are built here on what the atlas already keeps", ROADMAP.md:1217-1220).

---

## Things I could not establish from these documents

- Whether a single vault edit can be rolled back alone (the documents describe ROLLBACK of all).
- How a vault edit is laid on the packed 4-bit weights at a forward, beyond the phrases "patch overlays" and "as its
  forward lays it on"; that is the code readers' to say.
- The unpacker and date of the 12.6 s heard edit.
- Whether Google's licence page still reads as NOTICE says; whether the checkpoint is gated.
- Any policy on publishing edited weights, adapters or silenced copies.
- Whether README.md's "30 tools" or ARCHITECTURE.md's "32" was ever right at the time of writing; the code has 35
  today.
