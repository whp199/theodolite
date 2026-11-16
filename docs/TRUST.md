# How far to trust it

*The faults the study’s audit found in its own instrument and where each stands, and the cautions that go with a reading: the interface’s defaults, the quiet mask, the label gate, the wiring’s signs, thinking, one word against another, and the vault.*

<sub>Moved out of the [README](../README.md) on 16 November 2026, word for word but for its links and headings, when the README was cut down. Layers count 1–60, as the pages count them; the census board and the commands count 0–59.</sub>

*This is research code, and its lens was fitted on the base model (fault 9 below): good for comparing, poor on its own.
Read a single sighting as a lead, never as a finding: the study reads lens outputs against a matched control
(REPORT §3.5), and the instrument can keep one for you. A later word-piece of an answer is read inside text the model wrote
itself, where the lens’s lead over the naked eye shrinks (E3); compare it across matched sightings, not with the other
word-pieces of its answer.*

The study audited its own instrument ([`V7_tool_bugs_NOTES.md`](../experiments/results/V7_tool_bugs_NOTES.md),
REPORT §3.4). Where each fault stands now:

| | The fault the audit found | Where it stands |
|---|---|---|
| **1** | The nudge’s defaults, scale 650 at five layers (2.4–7.4× the stream’s strength), with a clamped inverse direction, collapse the model | **Mended in the instrument.** `direction="fnw"`, `mask_quiet`, `spare_bos` and `strength` (a share of ‖h‖); the nudge page sends all four, at 0.01. The server’s own defaults are kept for old callers |
| **2** | The attention / feed-forward split ignored Gemma 4’s `layer_scalar` | **Applied**: [`V7_module1_fix.patch`](../experiments/results/V7_module1_fix.patch) |
| **3** | The survey frame depends on the order of the words, and uses raw unembedding rows | **The order mended in the instrument**: `frame_squaring="lowdin"` (order-free) on the raw rows, which the interface sends by default; the server’s own default stays Gram–Schmidt on raw rows for old callers, and a neuron-locked frame keeps Gram–Schmidt (its three axes have fixed roles, so their order is not a choice). Squared all at once on the raw rows, the frame agreed with the sign of the lens’s she−he score in 79–89% of readings by stretch of layers, against 72–88% squared in order (V7). `frame_fnw` (the rows the model weighs, fnw ⊙ W<sub>U</sub>), `frame_per_layer` and `centre_prompts` stay choices, off unless chosen; V7 did not measure a frame on weighted rows. The fraction of the state the frame catches is always sent; the study’s 2.4–3.9% was measured on raw rows. V7’s contrast axes are read beside the frame, not in it (`contrasts`, below) |
| **4** | The thread score C is a weak guide: ρ ≈ 0.41 with measured effects, against 0.87 for a direct-path estimate | **Mended in the instrument**: `thread_score="direct"` chooses and ranks threads by the direct-path estimate, and the interface sends it by default; the server’s own default stays C for old callers. Each binary row carries the one score it was chosen by; the JSON's `filaments_meta` (its first 60 threads, in source-layer order) carries both |
| **5** | No chat template: the instruct model ran as a raw language model | **Mended**: `chat`, `system`, `assistant_prefix`; on by default in the interface |
| **6** | A doubled start mark for chat prompts | **Applied**: [`V7_bos_fix.patch`](../experiments/results/V7_bos_fix.patch) |
| **7** | One input label for a neuron’s two faces | **Built, and gathered on the model; not yet validated as labels.** The atlas keeps each face’s own passages, with the word-pieces it reads and that follow (`scripts/atlas_build.py passages`), shown on the atlas’s passages tab and in the folio’s apparatus; gathered on Gemma-4-31B on 28 October 2026 over 2,000,054 word-pieces, they are counts over sixteen passages that no test has checked. Since 14 November 2026 a pattern written from them can be put to the test without a push (text written to it; a blind reader picking the passages kept back from whoever wrote it). Tried on the model that day on 21 faces drawn at random: readers picked 177 of 196 pairs with a face’s own pattern and 86 with another face’s, and when the quiet passage was a near miss (the same words around it) 158 with the own pattern and 106 with another’s, which is chance, with 9 of the 21 faces beyond chance where 15 had been; 22 of 63 written texts fired and no near miss did; six of the 21 faces need more of their own document than a short text holds (ROADMAP §vi) |
| **8** | Neurons ranked by \|a\|, not by the size of their write; the top fifteen explain 10–41% of the feed-forward write | **Mended in the instrument**: `pole_ranking="write"` lists each layer’s fifteen biggest writers, \|a\| · ‖w<sub>down</sub> ⊙ w<sub>post</sub>‖, and the interface sends it by default; the server’s own default stays \|a\| for old callers. Every layer reports each listed neuron’s write size and the share of the feed-forward write the fifteen explain. That share is a sample either way: in E3 (12 prompts, last word-piece; indicative only), the fifteen loudest explained 10–41% at layers 21, 31, 41, 51 and 59 and 70–74% at layers 3 and 11, the seven sampled, and so did the fifteen biggest writers, within half a percentage point. The threads join the listed neurons, so they follow the ranking; V7 measured thread scores only among neurons ranked by \|a\| |
| **9** | The J++ lens was fitted on the base model (`google/gemma-4-31B`, bf16) and is applied to the instruct model (4-bit QAT), where it agrees with the model’s own average gradients at only cos 0.15–0.33 for the pronoun direction (layers 31–59) | **Open, and the largest accuracy fault left; now labelled.** Everywhere the instrument uses the lens, it still reads through this base-fitted J++: the words, the gathered/recalled split, the landmark bearings, the neuron labels, the nudge and the vault. R1 refitted a lens on the instruct model ([`R1_IT_LENS_REPORT.md`](../experiments/results/R1_IT_LENS_REPORT.md)): its directional form is stable from layer 31; its full matrix exists at only 11 layers (11, 16, …, 56, 59) and is usable at only five of them, 41, 46, 51, 56 and 59, where it beats J++ (top-1 on the model’s own reply at layer 51: 25% → 45%); below layer 31 no averaged lens tested was reliable. The instrument does not load it. The head and the foot of every page name the lens plainly, a red line beside each layer's words says what the reading can bear there, and the opening page says once, among its cautions, that the lens was fitted on the base model: the repository owner’s choice of 28 October 2026, after no refit passed R2, R3 or R4. `/api/health`, every sweep reply and the replies of the scientist’s tools still carry the full `lens` object (`id`, `fitted_on`, `method`, `applied_to`, `caution`). R2’s pre-registered test served no second lens at any layer, and R3, J++’s recipe fitted on the instruct model, served none either. R4 tested R1’s lens at layers 56 and 59 on 640 fresh prompts and served it at neither. Its lead over the naked eye was in function words (“the”, “of”, which R2’s filter counts as content), and without them it beat neither baseline ([ROADMAP §i](../ROADMAP.md#i--the-lens-the-largest-fault-left)) |

- The E6/V7 audit numbers were measured on the backend before these fixes. The history keeps the original instrument as
  its first commit.
- **The interface’s defaults.** Unless you choose otherwise, the interface sends the mends for faults 1, 3 (the order), 4,
  5 and 8; the server’s own defaults stay as first built, for old callers. To read with the reading options as first
  built (a frame squared in order on raw rows, neurons by \|a\|, threads by C), use “read with the reading options as
  first built” in the instrument’s apparatus, or set each one: “the frame options as first built” on the landmarks page,
  the ranking and the thread score in the instrument’s apparatus, the ranking also on a layer’s leaf or a neuron’s folio.
  Those options are needed to set a reading beside the study’s. Beside V7’s frame figures they are enough on this
  count, for V7 took them with one start mark, as the instrument now sends; beside E6’s sightings and the recorded one
  they are not, for those were taken before the V7 fixes, their chat prompts with a doubled start mark (fault 6, which
  by itself moved bearings by a median of 0.4–1.9° in V7).
- **The quiet mask.** Masking only the 181 persistently quiet dimensions is not enough: in a check made while building the
  fix (not part of the study), a she/he nudge masked that way still broke the answer at 0.005 × ‖h‖. `mask_quiet` also masks
  each pushed layer’s quietest tenth of dimensions; with both, the nudge read fluently at 0.005–0.01 and broke at 0.02.
- **Neuron labels are held back as the study asked (REPORT §3.3, conditions 2 and 3).** Below layer 41, and wherever
  the two readings of a label share none of their first ten word-pieces, no page sets a neuron's words, or a gloss of
  them, in its main reading; the apparatus keeps them, with the reason. The cut at none of ten is the instrument's (the
  study set no threshold): re-counted from N1's own data
  ([`N1C_gate_cut.json`](../experiments/results/N1C_gate_cut.json), a re-analysis), 7 of the 31 labels within layers
  41–58 that shared none of ten were sound (causal P@10 at least 0.3), against 160 of the 209 that shared at least
  one (the cut's AUC as a yes/no sign is 0.64); labels sharing exactly one were sound only 16 times in 40, below the
  band's 167 in 240, so the pages quote each count's own record and set the gilt mark only where it beat the band. A
  neuron silenced in the vault (DELETE) writes nothing; it has no count and is held back (reason `writes_nothing`).
  Every label the server sends carries `label_check` = {`shared_top10`, `held_back`, `reasons`, `band`,
  `sampled_layer`, `in_measured_band`, `lens_validated`, `edited`}, and its replies carry a `label_gate` card just before `lens`; the word lists themselves
  are sent as before, and layers in these fields count from 0. The agreement cut applies only through the base-fitted
  J++ lens, the one it was validated for (erratum E47). Labels at layer 59 are untested; at layer 60 the lens is the
  identity, so the cut cannot fire there. The demonstration sighting's labels and the vault's recorded searches carry
  three and five words a side, too few for the count, so each was checked ten a side through the server's own
  DESCRIBE POLE (`experiments/scripts/N1G_01_recorded_label_checks.py`, on the GPU;
  [`N1G_recorded_label_checks.json`](../experiments/results/N1G_recorded_label_checks.json)): the pages hold back 46 of
  the sighting's 75 labelled faces and 98 of the vault's 240 rows.
- **The wiring’s signs.** A link’s strength and sign are read from the weights alone: from rest (g₀ = u₀ = 0, through
  GeGLU’s gate) or, with the passages built, where its reader fires hardest on a face; neither is where a reader sits at
  an ordinary word-piece. On Gemma-4-31B (the night of 28–29 October 2026: 512 writers a layer at the 59 layers a later
  one reads, each pushed both ways by a quarter of its own write ceiling, read at `main_rel`’s four census questions’ last word-piece,
  a mark of the chat template) about nine readers in ten sat with the gate shut, and under the push on the writer’s gate
  face the sign from rest agreed with the measured change at 0.336 of links (0.341 for as many readers not linked) and
  the sign in use as served at 0.418 (0.438), each the median over layers: below chance, links and null alike. Where the
  gate was open the sign from rest held at 0.82. Scored at each reader’s own gate and up values there, which the check
  now records (its rerun of 29 October), the sign in use agreed at 0.841 (0.853), and the change’s size followed the
  first-order term at ρ 0.89. The links are real by size: linked readers moved 5.5 times as much as the null’s. So
  read a link’s strength as a lead, and its sign as holding where its reader’s gate is open (ARCHITECTURE §4.7; the
  scripts and figures in [`atlas_wiring_check/`](../experiments/results/atlas_wiring_check/)).
- **Thinking turned on.** A chat's sighting with thinking turned on is switched on by the chat template's own switch
  (`enable_thinking`: `<|think|>` at the head of a system turn, the model's turn begun with no empty thought). Bare
  text has no turn to think in, so the interface leaves it out there (`think_bare="none"`; the pair says
  `think_left_out: "bare_text"`). The server's own default stays the old imitation for old callers, `<|think|>` set
  before the bare text (E6 §3, item 11), and the first pair names which was used (`think_protocol`: `"template"` |
  `"bare_prefix"`; null on later pairs, as the other `think_*` fields are, and without one).
- **One word against another.** A sweep can weigh one word against another at every layer, through the lens and by
  the naked eye, in nats of log-odds (`contrasts`, at most four; the words pair by position, as the study’s pronoun
  pairs, and each is read as the landmarks read theirs, but only as one word-piece). It can read them against the
  matched controls it is sent (`contrast_backgrounds: ["controls"]`: the sighting’s score less the controls’, a
  difference of two log-odds, as E3’s first recommendation asks) or against E3’s average state (`"study_mean"`: the
  score of the state less the average, as E3 centred it, which is not a log-odds the model gives;
  [`E3_background_mu.npy`](../experiments/results/E3_background_mu.npy), exported from E3’s own caches and pinned by its
  sha256). The server reads none unless asked (`contrasts: []`); the landmarks page reads up to four, in its section
  vii, against either background or both, when the reader turns that section on, and draws one at a time. Alone, the
  scores carry a lean that comes from the model’s average state, not from the question, and the lens and the naked eye
  read it differently. Through the J++ lens, on E3’s neutral prose at
  layers 41–50, the average state’s share of the six pronoun pairs’ score was +1.4 toward the female words (the score
  itself +0.8, and −0.7 once centred on the average state; E3 (c)), its size depending partly on how the lens was
  fitted (erratum E34); the average state read alone scores +2.3 there, which is not the lean any reading carries. By
  the naked eye the lean runs the other way: E3 read the average state at −4.3 at layers 1–20 and −0.9 at 41–50, and
  on neutral prose centring still left −4.8 and −5.7 at layers 11–30 (E3 (c), its naked-eye figures). Through the
  lens the scores inherit fault 9, and the page marks each stretch of layers with the bands every page draws for the
  lens’s words, the lowest the least trusted, and what errata E49 and E50 add for a contrast. E3’s average state is in
  the coordinates of the server’s `resid_post`: on one of E3’s neutral passages the two captures agree at every layer,
  and E3’s neutral-prose figures at layers 41–50 come back through the server’s own scoring (the end-to-end check, item
  12; its figures in
  [`E3_background_mu_check.json`](../experiments/results/E3_background_mu_check.json)). E3’s own per-layer lens
  figures for the average state reproduce today only as a band mean (+2.35 against its +2.3 at layers 41–50); layer
  by layer they miss by more than 0.04 at 33 of the 60 layers, layers 41 and 42 among them, by the study’s own code as
  by the server, though its naked-eye ones agree to within 0.03 at every layer; the cause is not established.
- **The vault (LARQL).** `backend/larql_vault.py` calls the `larql`/`vindex` binaries only for `--version`; every vault
  query runs in Python. Its edit, named after MEMIT in the code and its statements (".patch MEMIT"), writes one neuron’s
  down-projection column (rank one, no covariance solve; DELETE sets it to zero): it is not MEMIT, and the pages call it
  an edit. Each edit can be heard before and after on a question of your choosing (`voice` on the `larql` request,
  ARCHITECTURE §4.6): the vault writes the answer the likeliest way, up to 12 word-pieces by default (1–64), just
  before the edit and again just after, under the one GPU lock, and the vault page asks for it by default, on its
  edit tab and for an edit typed in its console. The MCP tool `query_larql_vault` takes the same `voice`; run on its
  own, the MCP server’s tools take turns under a lock of their own, so no other call lands between an edit’s two
  answers. The voice is one question, so it shows what an edit did to that answer only; the study tested only DELETE
  by cause and effect (E6 §2.3).
