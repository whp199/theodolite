# Roadmap

*What is still open in Theodolite, gathered from the places that already name it: the fault table (once the README’s, now `docs/TRUST.md`), the
study’s report and its errata, the pages’ own “not yet” marks, and the records of R1’s instruct refit, R2’s test of
what to serve, R3’s test of J++’s recipe and R4’s test of R1’s lens. Nothing here has a date or an owner. Layers are
counted 1–60, as on the pages; the study’s reports count from 0, so their L50 is layer 51 here.*

<sub>Gathered on 21 October 2026, at `3cbc4b5`; gathered again on 22 October 2026, at `75ed78a`, after the first round
of mends (the interface’s defaults, the label gate, the causal test and the smaller pieces) and R3’s registration;
section i brought up to date at `c15b2d3`, with R3’s result, and on 28 October 2026, at `9093dac`, with R4’s; sections
ii–vi gathered again on 28 October 2026, at `554f058`, after a session that guarded every route, readied the census for
its first calibration and for rented cards, built the passages, one field book, the wiring’s sign in use and faster
4-bit unpacking, and began the CPU and GGUF (section vii, new); sections i, ii, vi and vii brought up to date on the
night of 28–29 October 2026 (`f88dba5`, `88e832d`, `4e38b6d` and that night’s review), once the calibration, the
passages and the wiring’s check had been taken on the model; sections iii and vi brought up to date on 29 October
2026, after the drive, the sampled push, the vault’s example edit and a second steer-lens layer were checked on the
model; section vii carried on through 2 November 2026 (the GGUFs on the card) and section vi through 12 November 2026
(the eleven whole layers of the census, the census read by seeking and searched across layers, the trials beside the
band and the second volunteer kit), each entry dated where it stands; what each hard job costs is in
`COMPUTE.md`.</sub>

<br>

## i · the lens, the largest fault left

The J++ lens was fitted on the base model (`google/gemma-4-31B`, bf16) and is read here on the instruct model (4-bit
QAT), where it agrees with the model’s own average gradients for the pronoun direction at only cos 0.15–0.33 (layers
31–59; [docs/TRUST.md](docs/TRUST.md), fault 9). Everything that reads through the lens inherits it: the
words, the gathered/recalled split, the landmark bearings, the neuron labels, the nudge and the vault. The head and foot
of every page name the lens plainly, and a red line beside each layer’s words says what the reading can bear there; the
opening page says once, among its cautions, that the lens was fitted on the base model. That was the owner’s choice of
28 October 2026, after no refit passed R2, R3 or R4. `/api/health`, every sweep and the replies of the scientist’s tools
still carry the full `lens` object, its caution included; the reading itself is unchanged.

- **An instruct lens exists, and is not loaded.** R1 refitted a lens on this checkpoint
  ([`R1_IT_LENS_REPORT.md`](experiments/results/R1_IT_LENS_REPORT.md)). Its directional form is stable from layer 31; its
  full matrix exists at 11 layers and is usable at five of them (41, 46, 51, 56 and 59). Its files are gitignored, so they
  are not in the repository ([REPORT](experiments/REPORT.md), erratum E50).
- **The second reading is built, and not merged.** The local branch `it-second-reading` (17–18 October, not pushed) reads
  a second lens beside J++: R1’s at layers 51, 56 and 59, and the Miru community J-lens for gemma-4-31B-it at 41–50,
  52–55 and 57–58. Each lens file is checksum-pinned and loaded only at layers that `scripts/check_second_lenses.py` has
  passed, and the layer and instrument leaves mark the reading checked or provisional. Its own commit says it is not for
  merge as is.
- **R2 served nothing.** The pre-registered test of which layers to serve
  ([`R2_PREREG.md`](experiments/results/R2_PREREG.md) and
  [`R2_03_report_section.md`](experiments/results/R2_03_report_section.md), brought here with R3’s registration; the
  serving code and `scripts/check_second_lenses.py` stay on `it-second-reading`) passed no lens at any layer, so under
  its rules the second reading shows nothing anywhere. The nearest miss is R1’s lens at layer 59, which failed only the
  minimum-size rule (+0.046 over the naked eye, where 0.05 was required). Two of the pre-registered sensitivity checks,
  one without the prompts about the model itself and one without the prompts that share a topic with R1’s fit prompts,
  would serve R1’s lens at layer 56; the registered decision stands.
- **R3 served nothing either.** J++’s published recipe, as the code at jpp_lens `d7efc0de` runs it, was fitted on
  the 4-bit instruct checkpoint and judged by R2’s rules on 320 new prompts
  ([`R3_PREREG.md`](experiments/results/R3_PREREG.md),
  [`R3_06_report_section.md`](experiments/results/R3_06_report_section.md)). Neither refit, the full recipe or its
  pooled R-lens, passes at any of the seven bands from layer 31 to 59, so R3 recommends nothing to serve. The nearest
  miss is the pooled R-lens at layer 56: +0.066 over base J++, but only +0.031 [−0.009, +0.072] over the naked eye,
  short of the 0.05 minimum. At layers 57–59 both refits read worse than base J++ and the naked eye. That refutes H1
  for this recipe on J++’s 64 WikiText prompts: it does not show that no instruct lens could pass, and a refit on chat
  text or on more prompts is untested. Its secondary readings decide nothing:
  - J++’s weighted experts were not shown to beat the pooled R-lens at any band, and read worse at layers 57–58 (H5);
  - the recipe concentrated a lens no more than Miru’s plain instruct J-lens does (H3);
  - a plain J-lens fitted here on the 4-bit QAT checkpoint agrees with Miru’s bf16 instruct lens more closely than
    Miru agrees with the base JLens (H6, “smaller” in every band). Both comparisons also change the code and the
    prompts, and the first carries the QAT checkpoint’s further training too; Q1 ran on J++’s 64 prompts because
    Miru’s 425 could not be rebuilt. So H6 lays nothing on fine-tuning or quantisation alone (E50 still stands).

  To pass its own gate, R3 had to compute J++’s identity rule in fp32: in bf16 the rule as registered changed the
  model’s top word-piece at 90 of 2,048 positions (D33), and jpp_lens’s own patcher changes the same 90 (the context
  check, D36). Every change made after the first GPU output is filed as a deviation (D30, D31,
  D33–D36). The run used 5.54 of the 8 GPU-hours the user approved. Whether an instruct lens gives better neuron
  labels is still untested (E45).
- **R4 served nothing, and closed R1’s lead.** R1’s lens, read for reference in R3, had passed all five rules at layers
  56 and 59 on R3’s prompts: +0.071 and +0.058 over the naked eye.

  R4 tested that lead on its own
  ([`R4_PREREG.md`](experiments/results/R4_PREREG.md),
  [`R4_03_report_section.md`](experiments/results/R4_03_report_section.md)). It was registered before any data, after
  its code had re-read R3’s prompts and matched R3 exactly. It had 640 fresh prompts and R1 as the only candidate. It
  used R2’s rules, and the owner added a gate: R1 must also pass without the targets that are function words.
  - **What R2’s rule counts.** R2’s filter counts "the", "of" and "is" as content words. On R2’s and R3’s prompts, R1’s
    lead lay almost wholly in those targets.
  - **Layer 56.** R1 passed R2’s rule (+0.088 over the naked eye, +0.106 over J++) but failed the gate. Without
    function-word targets it was +0.015 over the naked eye and +0.025 over J++, and worse than J++ on short questions.
    It named 57% of function-word targets against the naked eye’s 32%, and 62% of the rest against 59%.
  - **Layer 59.** R1 failed R2’s rule itself, at +0.038 over the naked eye. On three prompt sets its margin there has
    been +0.046, +0.058 and +0.038.

  So R1’s lens predicts the model’s next function word better, and its next substantive word no better.
- **Rules for any lens served later**, from the errata:
  - confirm condition contrasts at layers 31–50 with an instruct reading (E49);
  - show pronoun readings at layers 51–59 only where J++ and the instruct reading agree (E49, E50);
  - keep the caution at layers 11–30 for every averaged lens: none tested was reliable there (E50);
  - re-validate the label gate of ii for the new lens: its AUC of 0.84 was measured for base-fitted J++ labels only
    (E47);
  - judge the lens without function-word targets as well (R4’s S6 gate): R2’s filter counts them as content, and a lens
    can win on them alone.

R3 and R4 serve nothing themselves: a pass would only have been a recommendation to later serving work on a branch from
`main` (R3_PREREG §7.2, R4_PREREG §2). With no lens passed by R2, R3 or R4, the serving work on `it-second-reading`
stays there as R2’s record. Its worktree also holds the only other copy of R1’s pinned lens file (gitignored); R4 copied
it to `~/.local/share/theodolite/lenses/` and reads it from there.
R3’s jobs refuse a `backend/` that differs from their registered runtime, `d5405ee` (R3_PREREG §0), so any rerun goes
from the branch `it-jpp-refit`, not from a `main` that carries this round’s mends. R3’s line cites to the README (D26
and A9, `README.md:407–408`) are to `d5405ee`’s README; on this tree that passage is the licence paragraph of README
§vii.

<br>

## ii · neuron labels

The study judged joining the pillars “neat as a question, oversold as built”, and named four conditions under which the
joined instrument earns its keep (REPORT §3.3, erratum E47). The Lineage page (iv) marks where each stands:

| | The condition | Where it stands |
|---|---|---|
| **1** | Labels grounded in what wakes a neuron: the passages that make each face fire, stored with it | **Built, and gathered on the model; not yet validated as labels.** `scripts/atlas_build.py passages` keeps each face’s sixteen strongest windows of a corpus, one event a window (ties to the lower window), with the gate and up values there and each word-piece’s count over the corpus (the passages index, ARCHITECTURE §4.7), checked by brute force on the CPU stand-in (`check_atlas_index.py`). The atlas’s passages tab, a neuron’s folio and the scientist’s `passages` show them, with the word-pieces each face reads and that follow and their lift over the corpus. They were gathered on Gemma-4-31B on 28 October 2026, over the 2,000,054 word-pieces the stats index read (§vi; the build’s log in `experiments/results/atlas_wiring_check/gpu_logs/`); those words are counts over sixteen passages that no test has checked, so they are not yet validated as labels, and fault 7 (one input label for both faces) stays open until a test has checked them |
| **2** | Output labels trusted only from layer 41 up | **Yes.** No page sets a label below layer 41 in its main reading; the apparatus keeps it, with the reason, and the server marks it (`label_check`, with the rule in `label_gate`). The vault's console shows the server's replies whole and says which labels the pages hold back |
| **3** | A label counts only when the lens and the naked eye agree | **Yes.** A label whose two readings share none of their first ten word-pieces is held back; the others show their count. The cut at none of ten is the instrument's (the study set none; in N1's data 7 of the 31 such labels within layers 41–58 were sound, against 160 of the 209 others, so as a yes/no sign the cut scores AUC 0.64, against the count's 0.84 as a graded score; labels sharing just one of ten were sound 16 times in 40, below the band's 167 in 240, `N1C_gate_cut.json`), and it applies only through the base-fitted J++ lens, within layers 41–59, the gate having been validated for base-fitted J++ labels on random neurons within 41–58 (E47). The demonstration sighting's labels and the vault's 24 recorded searches carry too few words for the count, so `experiments/scripts/N1G_01_recorded_label_checks.py` checked each ten a side on the GPU (`N1G_recorded_label_checks.json`): 46 of the sighting's 75 labelled faces and 98 of the vault's 240 rows are held back |
| **4** | Each label spot-checked by cause and effect | **In part.** A neuron’s folio puts its label to the test (“test it at this word-piece”, and “test it on the study’s 36 texts” beside it), as do the WebSocket action `causal_check`, `POST /api/scientist/causal_check` and the MCP tool `causal_spot_check` (`backend/causal_check.py`, ARCHITECTURE §4.5): N1A’s ±k · max\|a\| on the neuron at the read position, with the untouched row in the same forward, at k = 10 in the linear regime by default and a frozen-norm write as a choice, since 60× steering saturates the post-feed-forward norm at layers 21–51 (E47). Each test reports how straight its push stayed, how far it rose above the same test at k = 0.01, and its P@10 for the lens’s and the naked eye’s labels; only the test on the 36 texts compares with the study’s figures. `scripts/verify_causal_check.py` checks it against N1A’s and N1V’s numbers on the real model. It is still run one label at a time, when asked: nothing tests every label. Testing every label at pages 41–59 in N1A’s layout would be about 2 × 10²⁰ multiply-adds, some 3.7 days of one H100 (inferred, COMPUTE.md); sharing each text’s prefix would cut that about 45 times but changes N1A’s layout, and no resumable driver is written yet |

Cammarata’s practices for reading a neuron ([9] in [`REFERENCES.md`](REFERENCES.md)), as the Lineage page sets them
against the instrument, leave three more:

- read each face in long passages, not only its top word-pieces: the passages are built and gathered on the 31B
  (condition 1), not yet validated as labels;
- keep the field notes with the vault: **done** (12 November 2026). The server keeps them, the name as the atlas’s and
  each face’s notes in a field book beside it (§iii), and `DESCRIBE POLE` now returns them with the neuron’s record,
  under `field_notes`: the name and note with who wrote each and when, and both faces’ notes, or the one face
  `POLARITY` names. They are marked as kept by hand, not measured, and stand apart from the labels and their
  `label_check`; the vault reads them and never writes them, and a proposal of the scientist’s not yet kept is not
  among them (`check_vault_voice.py`). Not seen in a browser yet: the console’s lines were checked as text;
- watch a neuron form across training checkpoints: no intermediate checkpoints of Gemma 4 31B are known; the base model
  against the instruct one is the nearest substitute, and needs the base model and the GPU.

Looking upstream at what builds a neuron is in the folio now (§iii).

<br>

## iii · smaller pieces, now built

- **Landmarks: one word against another. Built.** A sweep can weigh one word against another at every layer, through
  the lens and by the naked eye, against the matched controls sent or against E3’s average state (`contrasts`,
  `contrast_backgrounds`; ARCHITECTURE §4.1–4.2): the contrast axes V7 suggests
  ([`V7_tool_bugs_NOTES.md`](experiments/results/V7_tool_bugs_NOTES.md)), read in log-odds beside the frame rather
  than as an axis of it. The landmarks page reads them in its section vii: up to four contrasts, one a row, against
  either background or both, drawn one at a time (a pill for each contrast the sighting read, each background its own
  line). Still open: through the lens the reading inherits the base-fitted J++ (i), so the page marks its layers with
  the bands every page draws for the lens’s words (1–20 the least trusted, where E3 (b) found the lens barely agreeing
  with the model’s own pronoun direction; 21–30 orderings only; 31–50 to be confirmed with an instruct-fitted reading;
  51–59 counting only where one agrees: errata E49 and E50), and the naked eye with its own background, which runs the
  other way (E3 (c)); the study measured the average state’s lean on its six pronoun pairs and on its noun pairs
  (woman–man and the like), where it leans toward the male words and is larger (E3 (c): at layers 41–50 a share of
  +1.4 on the pronouns and −3.1 on the nouns), not on one pair or on other words; E3’s own lens figures for the
  average state reproduce today only as a band mean at layers 41–50 (+2.35 against its +2.31, within 0.04): layer by
  layer they miss by 0.20 and 0.10 at layers 41 and 42, and by more than 0.04 at 33 of the 60 layers (median 0.18, at
  most 1.85 at layer 3), by the study’s own code as by the server, with the cause not established (by layer:
  [`E3_background_mu_check.json`](experiments/results/E3_background_mu_check.json)); four contrasts and two
  backgrounds were checked on the CPU only (`check_interface_defaults.py`), and the chart seen only with readings
  made up for it, since the stand-in cannot take a sighting (a sweep reads sixty layers and it has six). That E3’s
  average state is in the coordinates of the server’s `resid_post` is checked by the end-to-end check (item 12) on one
  of E3’s neutral passages.
- **The vault: an edit’s voice. Built.** An edit can be heard before and after on a question of the reader’s choosing,
  written the likeliest way under the one GPU lock (`voice`; ARCHITECTURE §4.6), and the vault page asks for it by
  default. It is offered through MCP too (`query_larql_vault(lql_query, voice)`, the same `VoiceRequest`; run on its
  own, the MCP server’s tools take turns under one lock of their own, so nothing lands between an edit’s two answers),
  and for an edit typed in the console, heard by default with the edit tab’s question (`check_vault_voice.py`, on the
  CPU stand-in; the end-to-end check compares the tool’s heard edit with the route’s, and did so on the 31B on 29
  October 2026). The pages now call the “.patch MEMIT” edit an edit, a one-neuron column edit and not MEMIT; the
  code’s names and the statements are kept for old callers and the study’s scripts. Every vault query runs in Python;
  the `larql`/`vindex` binaries are called only for `--version` (docs/TRUST.md). The edit tab now opens on an edit that
  is heard, with its own question (29 October 2026; `vault/lql.ts` `EXAMPLE_EDIT`). Its earlier example, neuron 100 of
  layer 36, changed only one of seven answers, and that by a space, so a first reader heard nothing. The new one is
  neuron 7830 of layer 60, the one firing hardest where the answer to “How many legs does a spider have?” begins:
  - it turns “A spider has eight legs.” into “six legs.”;
  - on seven questions it also gave a dog six legs and wrote a hexagon’s “6 sides” as “six sides”, each answer cut to
    those words;
  - it left an insect’s six legs, Paris, the sky’s colour and the model’s account of itself alone
    ([`vault_example_edit/probe.json`](experiments/results/vault_example_edit/probe.json)).

  A note beside it says so. It also says that such an edit is rare: the most active neurons at layers 41, 46, 51, 56
  and 59 were not heard on this question at strengths 3 or 30. And it says that at the last layer only the norms
  stand between the neuron’s write and the next word, so the example shows an edit heard, not where a fact is kept. It
  is heard as a chat with no system prompt, as it was tried; the tab opens so. The probe did not record whether the
  neuron fires on the questions left alone. Still open: it is one question, so UPDATE and INSERT are still tested
  by cause and effect only as far as that one answer goes (the study tested only DELETE, E6 §2.3).
- **Thinking on bare text. Built.** Sent as a chat, the sighting with thinking turned on uses the template’s own switch;
  sent as bare text, the interface leaves it out, since bare text has no turn to think in (`think_bare="none"`), and
  the first pair names how thinking was switched on (`think_protocol`; null on later pairs). Still open: the server’s
  own default stays the old imitation, `<|think|>` set before the text, for old callers, and the question’s apparatus
  keeps it as a marked choice; the study recorded no fix for the bare case (the Lexicon’s cautions; E6 audit, item 11).
  Even in a chat the thinking sighting is read at the end of its own prompt, where the silent one is read at the close
  of its empty thought, unless an answer is begun (`think_read_token` names the word-piece it was read at).
- **Who may ask, on every route. Built** (28 October 2026). Until then only the atlas, its scientist chat and the saved
  prompts refused a page from another site: CORS allowed every origin, and `/api/larql`, the four `/api/scientist/*`
  tools and `/ws/theodolite` checked neither Host nor Origin, so any page the reader visited (or one rebinding its name
  to 127.0.0.1, or any page while a pod’s server was tunnelled to 127.0.0.1:8765) could run sweeps and vault edits on
  the live model and read the replies. Every route, the pages and the socket now answer only this machine’s pages and
  scripts run on it (`backend/local_only.py`, ARCHITECTURE §4), CORS names only loopback origins, and on Linux a
  connection from another account on this machine is refused too (its owner read from `/proc/net/tcp`), except
  through `npm run dev`’s proxy while it runs; `scripts/check_local_only.py` checks every route and the socket. The
  lab daemon refuses web pages and any Host but this machine’s, and its `/exec` wants a token it writes at each start,
  readable by its user only (`labclient.py` sends it); its port, 8765, is still the server’s. The scientist’s CLIs are
  no longer given the Hugging Face or cloud credentials in the server’s environment, only their own logins, and a stop
  no longer waits on a chat’s open event stream (`--timeout-graceful-shutdown 3`, docs/SETUP.md).
- **The guess test’s statistics. Built.** A neuron’s folio, v (`test_motif_hypothesis`, `POST
  /api/scientist/test_hypothesis`) said “yes, clearly” whenever Cohen’s d over population variances reached 0.8, at any
  number of sentences, with nothing set against chance: one sentence a side gave d near 10⁴ (E6 §3, item 7). Its reply
  now adds `selectivity` on the last word-piece’s unrounded readings: sample variances, Welch’s t and the n each side; a
  two-sided p from splitting the sentences between the sides afresh (exact over every split up to 10,000 of them, else
  10,000 shuffles under seed 0); hedged words (“selective on its + face, p = 0.029, n = 4 a side”), and no judgement
  below four sentences a side. The folio shows that; the verdict as first built is still sent, for old callers
  (`scripts/check_hypothesis_test.py`). Since built (12 November 2026, each against a worked or enumerated reference
  in that check, and a seeded run of 2,000 neurons with no effect, of which the peak and the either test each find
  about one in twenty at 0.05, as they should, where the better of the two unpaid finds about one in ten): `paired` reads the
  two lists as matched pairs, as E6’s own test did (`selectivity_paired`: the mean of the pairs’ differences, its p by
  flipping their signs, exact over all 2ⁿ flippings up to 10,000, and no judgement below six pairs, since at five no
  flipping reaches 0.05); the peak is tested as the last word-piece is (`selectivity_peak`), and `selectivity_either`
  pays for quoting whichever of the two did better (the larger |t|, taken again under every relabelling); `also`
  names further neurons to test on the same sentences in one call, and their p’s come back corrected together
  (`correction`: Benjamini–Hochberg q’s, each with how many tests it is over; E6 §2.3). A neuron tested alone is told
  in words, in the reply and the folio, that its p is corrected for no other. The folio has a box for pairing (“the
  lines are matched pairs”), refuses unequal sides before anything is sent, and says so when a server from before the
  paired test answered unpaired; the panel was driven in a headless browser on the CPU stand-in (an unequal pair of
  lists refused, six pairs read as pairs, the same lines read unpaired with the box cleared), not yet on the 31B.
  Still open: the folio tests one neuron at a time (`also` is the request’s); nothing corrects for tests run in
  separate calls or for sentences tried and changed; the peak counts the start mark among a sentence’s word-pieces
  (not yet looked at on the 31B: where the start mark is the loudest the peak test can show nothing) and does not tell
  a longer sentence from a louder one; no place but the last word-piece and the peak is tested.
- **One field book. Built.** A neuron’s folio reads and writes through the server whenever it answers for the neuron:
  its name is the atlas’s name (`field_notes.json`, by the reader), and its notes, face by face (how many jobs, the
  pattern in words, the four ticks), go to `field_book.json` beside it (`/api/atlas/fieldbook`; ARCHITECTURE §4.7), a
  file of its own because a server from before moves aside any notes file but version 1 and keeps only the fields it
  knows. The atlas’s details show the book, with a link to the folio. With no server, or from the moment it does not
  keep a change, the folio keeps them in the browser and says so, and it offers once to carry a browser’s notes to
  the server, writing over nothing the server keeps. Lineage practice vi.
- **What builds a neuron, in the folio. Built.** Practice v, upstream: at the sighting’s own word-pieces, the heads,
  the earlier feed-forward blocks and the earlier neurons whose writes make up its activation there (`POST
  /api/atlas/attn_drive` with `ids`; each block’s part split over its neurons, one mat-vec a layer, checked on the
  stand-in to add up to the block’s part to float error and against a direct-path finite difference), and by the
  weights alone its strongest writers, each on the face that opens its gate. Both are leads: a neuron’s share is its
  direct write with the norms held, not what it does through the layers between. **Checked on the 31B** (29 October
  2026, through the server with the reference unpacker, 10.5 minutes;
  [`atlas_drive_check/`](experiments/results/atlas_drive_check/), `scripts/verify_atlas_drive.py`). The neurons were
  42, six at each of layers 11, 21, 31, 41, 46, 51 and 58, each read at its strongest stored passage, cut to end there.
  - The parts add up there: the drive’s `rest` is at most 0.5% of its linearised total, a block’s neurons add up to its
    part within 0.08% (median; at most 2%, the rounding of the block’s bf16 output), and the drive’s activation is the
    model’s own within 0.8%. The linearised total is about twice the activation (median 2.01, from 1.18 to 2.29), as
    gelu(g) u is near bilinear once the gate is open.
  - Each of the three strongest heads, and of the three strongest earlier neurons, was taken away in a real forward
    (a head scaled by 0, a neuron multiplied by 0, at every word-piece but the start mark), and the neuron read again.
    That removal moved the neuron the way its part said 117 times in 124 for heads and 118 in 124 for neurons (94% and
    95%; Wilson 95% intervals 89–97% and 90–98%, treating a neuron’s three leads as independent). It moved it by a
    median 0.89 and 0.93 of the part (Spearman ρ 0.89 and 0.85). The sign test is not only “taking an input away lowers
    the peak”: of the leads whose removal was predicted to raise the activation’s size, 34 of 36 heads and 16 of 16
    neurons did.
  - How far back a lead was written matters (read after the run, `read_after_the_run` in the record). Heads at the
    neuron’s own layer or the one below went the predicted way 81 times in 83, by a median 0.95 of the part. Heads
    further back went 36 in 41, by 0.57. Neurons up to four layers back went 109 in 111, by 0.92–1.0. Neurons five or
    more back went 9 in 13, by 0.37.
  - Against heads drawn at random from the layers at or below the neuron’s, and neurons drawn from the named ones’
    blocks, a lead moved the neuron more than a random one in 92% and 99% of pairs (ties counted half); the median
    change was 11% of the activation, against 1.3% for the random heads. Both nulls are easy. Only 4 of the 126 random
    heads sit at the neuron’s own layer, against 46 of the leads. And 89 of the 126 random neurons changed nothing the
    bf16 reading can show, mostly because they were silent there, so the 99% sets the leads against silence.
  - The leads are real; they are not all that matters. 18 of the 126 random heads moved the neuron more than the
    least-moving of its three named heads, through the layers between.
  - The weakest layer was 58: 16 of 18 the predicted way for each kind, and a median 0.61 of the part for heads, the
    lowest. The lowest shares were 15 of 17, for the heads at 21 and the neurons at 51.

  So the rows lead well where a neuron fires hardest, at least for parts written close by, in this sample. The
  folio’s note says so, and that a head it does not list can move the neuron as much. Still untested:
  - where a neuron fires weakly, as at most of a sighting’s word-pieces (41 of the 42 sat with the gate wide open,
    their linearised total 2.00–2.29 times the activation);
  - leads below the three strongest;
  - a null of neurons matched on activation;
  - how the named heads rank among every head of the layers just below by the whole effect of taking each away.
- **The scientist’s names as proposals. Built.** `set_name` and `write_note` propose (`field_proposals.json`,
  `/api/atlas/proposals`): the atlas page shows each with keep and discard, nothing reads one before it is kept (the
  map, lists, search, tooltips and later chats read only the names and notes kept), and a kept one is written as the
  scientist’s.

<br>

## iv · the server’s defaults, and the interface’s

The server keeps its old defaults for old callers. The interface sends the mended settings for all four; parts of fault
3’s mend, the weighted rows, the per-layer frame and centring on a control, stay choices:

| | The fault | The server’s default | What the interface sends unless you change it | The mend |
|---|---|---|---|---|
| **1** | The nudge collapses the model | five layers at scale 650 | `direction="fnw"`, `mask_quiet`, `spare_bos`, `strength` 0.01 × ‖h‖ | in use on the nudge page |
| **3** | The survey frame depends on the order of the words | Gram–Schmidt on raw rows | `frame_squaring="lowdin"` on raw rows (a neuron-locked frame keeps Gram–Schmidt) | also `frame_fnw`, `frame_per_layer`, `centre_prompts` (off unless chosen) |
| **4** | The thread score C is a weak guide | C | the direct-path estimate | `thread_score="direct"` |
| **8** | Neurons ranked by \|a\|, not by the size of their write | by \|a\| | `pole_ranking="write"` | `pole_ranking="write"` |

Still open here:
- **The weighted rows.** V7 put the order-free frame’s remaining disagreement with the lens (about one reading in five) down
  to the raw rows, but did not measure a frame on the rows the model weighs, fnw ⊙ W<sub>U</sub>; the interface leaves
  `frame_fnw` off until an instrument check measures it.
- **Threads among the biggest writers.** V7 measured the direct-path score’s 0.87 on threads among neurons ranked by
  \|a\|; the threads now join the biggest writers, where it was not measured.
- In E3 (12 prompts, last word-piece; indicative only), at the seven layers it sampled, the fifteen biggest writers
  explained the same share of the feed-forward write as the fifteen loudest, within half a percentage point (10–41% at
  layers 21, 31, 41, 51 and 59; 70–74% at layers 3 and 11), so
  fault 8’s mend changes which neurons are named, not how much of the write they cover.

<br>

## v · loose ends from the first round

- **Where the label gate cuts.** In N1’s own sample within layers 41–58 (`N1C_gate_cut.json`, `cuts`), holding back
  labels that share one of ten or fewer would have held back 71 instead of 31 and kept 144 sound labels of 169 (85%),
  against 160 of 209 (77%) at the cut chosen, none of ten. A cut chosen on that sample would be judged on the data it
  was chosen from.
- **The README’s pictures.** The views in docs/INSTRUMENT.md (once README §iii) were taken before this round, with the reading options as first
  built and before the label gate; their captions say so. They want retaking from a live sighting.
- **References. Opened.** Section H of `REFERENCES.md` ([45] and [46], R3’s sources) was written from local files, its
  links unopened. On 12 November 2026 each was opened: both resolve, [45] now carries its title, authors and date as
  arXiv lists them, and [46]’s revision, file size and sha256 are as Hugging Face lists that file.

<br>

## vi · the atlas

The atlas (`#/atlas`; docs/ATLAS.md and docs/INSTRUMENT.md, ARCHITECTURE §4.7–4.8) was built on 27 October 2026 without a GPU: every piece of it
was checked on a tiny random Gemma 4 on the CPU (`scripts/atlas_stand_in.py` and the atlas’s checks) and the page driven
in a headless browser against it. The same day its indexes were built on Gemma-4-31B on one RTX 5090 (the lens in 144 s;
the stats over 2,000,054 word-pieces, a document from each corpus in turn, in 58 minutes; the wiring in about 4 minutes;
the layout in 9) and the page driven end to end on them, a Codex scientist with it; the index files are kept outside the
repository. On 28 October the census, the wiring, the passages and the scientist gained the pieces below, each checked
on the CPU stand-in; each bullet says which have since run on the model (the calibration ladders and the writer’s floor
on 28 October; the passages, the wiring rebuilt and its measured check on the night of 28–29 October; two steer-lens
layers, the determinism run and the sampled push on 29 October). In this section layers count from 0, as the commands do
(page = layer + 1), unless pages are named. [`COMPUTE.md`](COMPUTE.md) says what each job costs and where to run it.

- **Take its censuses, and its steer lens, on the model.** Neither exists beyond the 8 + 8 `calib_rel` pilot of 27
  October (580 answers, about 16 minutes at `--max-new 64`), the two calibration ladders of 28 October (64 + 64 units at
  each of six layers, below), two steer-lens layers timed into a scratch folder (45 and 5), the 16 + 8 determinism run
  of 29 October (868 answers), the `break_rel` trial taken and read that day (both below) and the whole `break_rel`
  layers taken since 29 October, eleven by 11 November 2026 (40–50; layer 45 and the volunteers’ below), and five more
  whole layers by 16 November 2026 (39 and 51 with `break_rel`, 34–36 with `match_ceil`; below, under kit 4). The steer
  lens is built since 13 November 2026, all sixty layers (below). The pilot
  measured the cost: every decode step of the 31B takes about 0.71 s whatever the batch (1 to 64 rows), so a whole
  `main_rel` layer is about 3.6 days at `--batch 32 --max-new 64` and about two weeks at the default 256. The step is
  the 4-bit unpacking, not the model (COMPUTE §i). The steer lens was estimated at 15 to 20 hours for all 60 layers on
  a 5090. One layer timed on 29 October 2026 (layer 45, `compiled`, into a scratch folder, not served;
  [`atlas_steer_timing/`](experiments/results/atlas_steer_timing/)) took 287 s; a layer runs only the layers after it
  (the forward starts at l + 1), 14 for layer 45, so its cost grows with their number. A shallow layer timed the same
  day checks it: layer 5, which runs 54, took 1,181 s, 7% more than layer 45’s rate gives. So all 60 would be about
  10.1–10.8 hours (inferred, by either layer’s rate over 1,770 layer-forwards). Its dose, the census’s 0.8
  of the stream, is still to be chosen against the calibration before it is built. **What that dose does, read from
  the two layers already built** (12 November 2026, CPU; [`atlas_steer_dose/`](experiments/results/atlas_steer_dose/)):
  at layer 45 it moves the next word-piece by a median 0.34 nats (KL from the unpushed one), the readings differ
  neuron by neuron, and they agree with the layer’s census. For 98.3% of the 1,580 neurons above every control a
  steer-lens word-piece is one of the neuron’s five census words, or begins one (2.1% with the neurons shuffled);
  for 92.6% of the 6,307 with q ≤ 0.05; for 23.8% of the rest. At layer 5 the same dose moves it by a median 7.5 nats,
  and the reading is nearly one reading whichever neuron is pushed: “Could”, “Are”, “These” and “Based” are each
  listed by 40–49% of the layer’s neurons, on both signs. So 0.8 of the stream is sound where the census is and
  wrecks the early layers, and the lens needs a dose for each layer. Two rules were open: a push sized, layer by
  layer, to move the next word-piece as far as layer 45’s does, every neuron of a layer pushed alike as now; or a
  push of one and a half to two times each neuron’s own write ceiling, which is what 0.8 of the stream comes to at
  layers 40–50 (a dose of 1 there is 1.8 to 2.5 ceilings, ARCHITECTURE §6). **The owner chose the first** (12
  November), **and it is built, on the CPU stand-in** (ARCHITECTURE §4.7, “Its dose, a layer at a time”):
  `atlas_build.py steer_dose` is the pilot (128 neurons a layer at nine doses from 0.0125 to 3.2 of the stream; each
  layer’s dose is where its median KL meets layer 45’s at 0.8), `atlas_build.py steer` takes its doses, each layer’s
  file and the merged index record them, a layer built at another dose is refused, and the details tab, the
  scientist’s `steer_lens` and the lexicon say what a reading was pushed by and how that was chosen. Until the lens
  is built a neuron read live is pushed at the pilot’s dose for its layer once the pilot is taken, and before that
  at 0.8, which the page now says in red is not sized to the layer. **The pilot is taken** (13 November, 00:52Z to
  01:31Z, 38 minutes on the home card): all sixty layers measured, none off its ladder
  ([`pilot.json`](experiments/results/atlas_steer_dose/pilot.json), written by `pilot_in_ceilings.py` there). The
  dose runs from 0.11 to 0.19 of the stream at layers 9–21, through 0.3 to 0.5 at 26–38, to 0.68 to 0.91 at 39–51
  and 1.28 at 52; layers 0–8 jump between 0.21 and 0.90. Counted in write ceilings it hardly moves: between 1.3
  and 2.3 ceilings at every layer from 9 to 56, where as a share of the stream it changes elevenfold. So a count of
  ceilings is close to the right measure through the middle of the model, and not below layer 9 (1.8 to 6.9
  ceilings there, most of them 3 or more) or in the last three layers (0.5 to 1.0). **The build started at 01:31Z** on all sixty layers (10.1–10.8
  hours; its first two layers took 22.5 minutes each). It was paused at 02:18Z with layers 0 and 1 built, on the
  owner’s word, to let the census’s walk downward have the card first (step 3, below). It built layers 2 to 4
  between 06:45Z and 07:26Z, waited again behind the `match_ceil` walk, and went on from layer 5 at 14:56Z on 13
  November when that walk had stopped. **The lens is built** (13 November 2026, 23:43Z): all sixty layers, none
  skipped, in 10.5 hours of building over three sittings (2,702 s, 3,395 s and 31,646 s, against the 10.1–10.8
  estimated), and it is served. **It is sound at layers 39 to 52 and a weak guide at 32 to 38, and the page now says
  which** (corrected 14 November 2026: this said “sound from layer 39 up and a weak guide below”, which claimed more
  than was measured, since layers 53 to 59, like 0 to 31, have no census that checks it)
  ([`read_lens.py`](experiments/results/atlas_steer_dose/read_lens.py) and `lens.json` there, read on the CPU;
  `backend/steer_check.py`). Four readings:
  - *Beside the whole censuses, layers 40–50* (`against_census.py`, the build’s last step). For 85% of the neurons
    above every control at layer 40, 94–97% at 41–44 and 98–99.5% at 45–50, a steer-lens word-piece is one of the
    neuron’s five census words or begins one (1.6–9.7% with the neurons shuffled). Layer 45, built again, gives the
    98.3% of 1,580 it gave on 12 November.
  - *Beside the trials that passed, where no whole census is taken.* This entry said nothing could check the lens
    below 39; the trials taken since can, at 32–39, 51 and 52. Counted over the neurons above 95 in 100 controls,
    which a trial fills with 214–447 (layer 45’s own trial reads as its whole layer does, 86.6% against 86.1%): 84%
    at layer 51 and 75% at 52; 60–91% at the whole layers; 62% at 39; then 33% at 38, 20% at 37, 13–17% at 34–36 and
    7–10% at 32–33 (1–6% shuffled). At layer 38 the count leans on the census’s design too (59% of the 106 that
    stand out under `break_rel`, 33% of the 357 under `soft_ceil`); at 37 it does not (20% under both designs).
    Over the few neurons above every control it is higher and noisier: 7 of 8 at layer 34, 5 of 54 at 32.
  - *Whether a layer’s neurons read apart, all sixty layers.* Sizing the dose lessened what 0.8 did to layer 5 and
    did not cure it: there the most listed word-piece is now on 34% of the neurons’ lists and the next two on
    25–27%, where four were each on 40–49%. At layers 0–30, 4 to 7.5 of a neuron’s twelve word-pieces are listed by 2 in 100 of the
    layer’s neurons or more whichever is pushed, and one piece by 19–49% of the layer (stray pieces at 0–7; from 8
    up sentence openers: “Here”, “The”, “In”, “These”, “You”). It is 2 to 4 of twelve at 31–39, one at 40–41,
    under one at 42–50 and none at 51–59.
  - *Where the push landed.* The pilot’s 128 neurons a layer put the whole layer’s median KL within 0.8–1.08 of the
    target at 43 layers and 0.7–0.8 at eight more; at 9–11, 29 and 59 it is about half the target (0.47–0.55), and
    0.62–0.67 at 4, 6, 24 and 25. The sharing does not follow it: layers pushed on target (0, 2, 3, 21, 22, 26)
    share as many word-pieces as the ones pushed short, or more.

  What it means. The lens lists the word-pieces whose odds rose most, and nothing in it sets a neuron against other
  pushes as the census’s controls do, so below layer 40 a share of every reading is what any push at that layer
  does. It also reads the next word-piece where the census reads 48 word-pieces of answer, and a neuron in the
  middle of the model need not show in the first as it does in the second: the fall in agreement is not all the
  lens’s fault. It does not count against the trials at 32–38, which passed on their own words carrying between
  question halves; but nothing outside the census seconds them. And the agreement was never a test of what a neuron
  does unpushed: it is two readings of the same push.

  What the page does with it (chosen in building it; the owner’s to change). `scripts/atlas_steer_check.py` writes
  the check beside the index (`steer_check.json`; run it again when the lens is rebuilt, a census layer lands or a
  trial’s doses are counted calibrated). The details show all twelve word-pieces, the ones the layer shares faint,
  and say how the lens fared at that layer: in red where it shares a word with under half the neurons that stand
  out (layers 32–38), or where the layer has no census and its nearest checked layer is such a one (0–31). At those
  thirty-nine layers a neuron’s line of words and the word search leave the steer lens out, as they did before it
  was built; at 39–59 a line gives the neuron’s own steer words, the shared ones dropped. (Changed 14 November 2026:
  a layer with no census no longer takes its nearest checked layer’s standing. Layers 0–31 and 53–59 say they were
  not checked and are left out of lines, searches and pooled words as that; a line gives the neuron’s own steer
  words at 39–52, fourteen layers.) The scientist is told the
  same (`check`, `shared_with_layer`). A web’s pooled words and a head’s leave out the neurons at those layers, and
  say how many (a first look at the page showed why: a layer-46 neuron’s web read “Some”, “Sometimes”, “Depending”,
  “Perhaps”, “Meanwhile”, the openers its upstream layers share). The repair not yet tried is a
  lens read against its layer: each word-piece ranked by its rise over the layer’s mean rise, or over control
  pushes. That is a rebuild (10.5 hours); a trial’s 1,022 neurons at one layer would pilot it in minutes.

  The pilot’s ladders are also each layer’s dose curve, which a census
  design scaled to the layer needs (the plan’s step 3, below, where it is set beside every dose a trial has read). On the home card, headless, time a
  pilot first; on a rented one, `scripts/pod/pilot.sh` is that pilot (it times the step and the memory for each unpacker
  at several batches, takes a `calib_rel` run, times one steer-lens layer, sets a canary batch against the home card’s
  and projects each job × 1.25 against the owner’s approval, which `scripts/gpu_log.py` keeps with every pod against
  it); a layer’s census then runs in shards (`scripts/pod/census_shard.sh`, every shard at the batch the pilot measured
  on that card class) and `sync.sh --final` pulls and checks them before a pod goes. None of these has yet run on a
  rented card.
- **Faster unpacking, the same bits. Built, and timed.** Every forward unpacks the 4-bit weights afresh (five eager
  operations, three over int32 temporaries), which by the arithmetic is most of the 0.71 s step. `backend/w4a16_fast.py`
  installs, without touching the runtime’s file (which R4 pinned, and which the instrument now holds to its own
  recorded sha256, `scripts/runtime_files.py`), a lookup-table unpacker (`--unpacker lut`), the same arithmetic
  compiled alone (`compiled`, the model left eager with its hooks) or the weights kept unpacked (`resident`, about 60 GB
  beside the 4-bit ones). Each gives the reference’s bits: `scripts/check_dequant.py` shows it on the CPU for every
  (piece, bf16 scale) pair and for a 4-bit stand-in’s logits, vault edits, pushed answers and census rows, and `install`
  tests every linear again on the card, refusing in plain words on any difference. Each census and index meta, and
  `/api/health`, records which unpacker ran. On the home 5090 (28 October 2026, beside a desktop) `lut` and `compiled`
  passed that test on all 410 linears and took 0.28 s and 0.103 s a step at a batch of 32, against the reference’s
  0.71 s; `compiled` held 0.126 s at 96 rows (COMPUTE §ii). By that step a whole `main_rel` layer is about 5 hours at
  `--batch 96 --max-new 64`, not 3.6 days (arithmetic, not a run); the calibration below, taken with `compiled` at a
  batch of 64, took 65 and 49 minutes for its six layers. A GGUF’s packed layers have a `compiled` of their own since
  2 November 2026 (§vii).
- **`main_rel`’s doses come from Qwen3.5-4B; `break_rel`’s from this model’s ladders.** `main_rel` (−0.7, +0.5, +0.8
  and +3 of the stream’s median length) is
  neuron-explorer’s design, where its log found answers readable near 0.5–0.8 and wrecked by 3. Take `calib_rel` (±0.1
  to 4 of the stream) and `calib_ceil` (±0.5 to 8 × each neuron’s write ceiling) on a few layers of Gemma-4-31B first;
  one command takes several layers (`census_run.py --layers 20,30,40,45,50,57 --design calib_ceil`, layers from 0).
  Read each run with `census_tools.py curves`, which lays out its dose window: per dose, neurons against controls, the
  share of answers moved from the unpushed one and where they first part, the share cut off, distinct 4-grams, and the
  push as a multiple of each neuron’s ceiling, with a hedged verdict per sign. The 8 + 8 pilot already taken can be read
  the same way (`curves <layer> --weights`). Change the design if the window lies elsewhere.

  **Both ladders taken** (28 October 2026, layers 20, 30, 40, 45, 50 and 57, pages 21–58: 64 random neurons and 64
  controls a layer, `--batch 64 --max-new 64 --unpacker compiled`, 65 and 49 minutes on the home 5090; each layer’s
  dose window and a summary in `experiments/results/atlas_calibration/`, the runs in the atlas folder). One question
  set, one greedy answer each, so these are leads:
  - **Where text breaks** (distinct 4-grams more than 0.1 below the unpushed answers’): at ±0.5 of the stream at layer
    20, ±1 at layer 30 and ±1 to 2 from layer 40 to 57; in each neuron’s ceiling, at about 4 to 5 × at layers 20 and
    30, 2 to 4 × at 40–50 and 1.2 to 2 × at 57. Neither unit puts it at one dose across depth. `main_rel`’s +3 breaks
    text at every layer, and its +0.5, +0.8 and −0.7 at layer 20.
  - **Where effects begin lies below both ladders.** The smallest push of each (0.1 of the stream; 0.5 × the ceiling,
    0.05–0.41 of the stream) already changes 38–95% of the 64-word-piece answers somewhere, and a matched control
    changes as many: the neurons moved more answers than the controls at one dose of some 130 tested (layer 30, −0.5 ×
    the ceiling, 88% against 81%, uncorrected), about what chance gives. Whether an answer changes at all is too coarse
    a reading at this length, and `curves`’ verdict reads only that.
  - **The ranking tells neurons from controls clearly at layers 40 and 57; where the lead lives is not settled.** The
    census’s own ranking (the words each push brings) caps the push it reads at 1.6 of the stream (`compute_units`’
    `max_rel`), so as first read here it left out `calib_rel`’s ±2 and ±4 and every `calib_ceil` dose past 1.6 of the
    stream (at layer 57 all but ±0.5, ±1 and a little of ±2): those first figures (0.62–0.80 at 40–50, alike at 20, 30
    and 57) were over the doses under the cap, not every dose. Re-ranked on every answer (29 October 2026, CPU;
    `experiments/results/atlas_calibration/`, `subset_rank.py` with its `subset_ranking.json` and `subset_table.json`),
    neurons lead their controls clearly at layers 40 and 57 in both ladders (AUC of mass, the chance that a neuron’s
    mass beats a control’s, 0.62–0.70, p ≤ 0.01; in the ceiling at layer 40, 13 of 64 beat every control where about one
    would by chance, least q 0.08) and at 45 in the ceiling (0.63, p 0.006). At 45 in `calib_rel` and at 50 in both, the
    lead is no stronger than at 30 (0.55–0.58, p 0.06–0.15, against 0.58 and 0.59, p 0.06 and 0.05, at 30), and at 20
    there is none (0.52 and 0.50). Under the break alone (the doses short of where text breaks, as read above) it holds
    at 40 and 45 in `calib_rel` (0.62 and 0.60, p 0.01 and 0.03, and no less once the new words an answer keeps are
    regressed out: 0.64 and 0.62), weakly in the ceiling at 40–50 (about 0.58, p 0.05–0.07), and not at 20, 30 or 57,
    nor at 50 in `calib_rel`. At ±2 of the stream it is largest (Stouffer z 6.4 over the six layers), but the file’s
    covariate check credits most of that to the neurons’ text keeping more new words under the push than the controls’
    text does: with that regressed out the AUC is 0.52–0.57, at no layer p < 0.05.

  A design to replace `main_rel` was drafted on these leads: `break_rel` (±1 and ±2 of the stream on six questions read
  as two halves, every answer counted, each word against the controls alone), built on the branch `census-break-rel`
  and merged on 29 October once its trial was read; it has been the default design since that afternoon (the owner’s
  choice, below). Its trial was taken early on 29 October, its reader and gates fixed and hashed before it began (`SHA256SUMS` sha256
  `8d8d81de91dd03529db83f091c7ddeb26bbb4dd8037cac66f6dd91884c7c3b2d`), and read that morning with the gates as they
  were (the owner’s word). At layer 45 (1,022 fresh neurons against 1,024 controls) it **passes every gate** at 48
  word-pieces, and at 32 every gate the reader computes there (the check that the units file ranks as read is made at
  48 only). At 48: mass AUC 0.714, 0.724 with six measures of how the text survives fitted out, words picked on one half
  of the questions and scored on the other 0.688 and 0.674, the three new questions alone 0.669 (all AUCs), against
  bars of 0.60, 0.58, 0.57 and 0.56. The probe at layer 57 (507 against 512) passes at 48 and fails at 32 (its residual
  0.5608 against a bar of 0.56, then 0.5061). By the rules written before it, that licenses whole layers at 40–50 at 32
  word-pieces and at 51–57 at 48. The shorter reading at 45 lay within 0.02 on the two measures the rule checks, but it
  names 191 neurons where 48 names 262, and its residual falls 0.0215. A review the same morning
  ([`census_break_rel_trial/`](experiments/results/census_break_rel_trial/), `summary.json` and `review/`) used six agents: three
  readers, each checked by a second that re-derived its headline figures, the gates with code that shares nothing with
  the pre-registered reader, the by-dose and per-neuron figures with code built on its parts. It found the reading
  faithful; four small departures, none able to move a gate, are recorded there, one of them that the 32-piece reading
  is the 48-piece answers cut short. Then it asked what the pass measures:
  - **At layer 45 the lead is the ±1 answers’.** On them alone the AUC is 0.706 and clears every bar. On the ±2 answers
    alone it is 0.571 and clears none: at ±2 the text of neurons and controls alike collapses into loops on the same
    few words (own, thought, same; distinct-4 0.21 for both), so that half of the answers adds 0.008, though alone it
    still holds a weak lead (5.6 SE above chance). Among units whose twelve ±1 answers all stay fluent (distinct-4 at
    least 0.9; 705 neurons, 942 controls) the ±1 lead is 0.641, about two-thirds of the pooled lead above chance. A
    neuron picked on one half is found again on the other three times in four (120 of 161, against 7 of the 51 strongest
    controls), and its words nine times in ten (147 of 161, against 10 of 51). The two halves’ ranks agree at 0.46
    (controls 0.14), and about 7 of the 262 neurons named would be named by chance (95% upper bound about 14). The 893
    neurons whose text survival the controls do cover clear every gate on their own (0.692, 0.691, 0.657 and 0.643,
    0.651), the check layer 57 fails. The order lies in the layer’s tail: below the controls’ 95th percentile the halves
    agree barely more than the controls’ do (0.17 against 0.12), though those 767 neurons still lead the controls as a
    group (0.61 on the other half). The calibration’s “none named” was forced by its size (64 + 64 units cannot reach
    q ≤ 0.05 with fewer than 20 neurons above every control; it had 15).
  - **Part of that lead is text breaking early, and the words need filtering.** The named neurons’ ±1 answers loop
    (distinct-4 below 0.5) in 9.6% of cases (fresh neurons 4.1%, controls 0.4%): 135 of their 303 looping answers on
    their own words, 127 on the collapse words and 41 on neither. 194 of the 262 list “same”, “own” or “thought” among
    their words. The gates do not rest on these: with the 789 collapse words removed from every answer, every gate still
    passes. But a page should not show them as a neuron’s words, and since 29 October none does (below).
  - **At layer 57 the pass is fragile and is not words in fluent text.** ±1 alone gives 0.52, and only the two doses
    together clear the probe’s first bar. The named take 82% of their weight from broken answers (distinct-4 below 0.9;
    the controls 49%, nearly all at ±2, and all 507 neurons 60%); many loop on pieces such as “sem”, “pemb” and “ngày”,
    though about half their words are ordinary ones (vendor, volt, deck). Its residual falls below its bar in about half
    of the resamples. And 30% of its neurons write text whose survival no control matches, while the 354 the controls do
    cover fail every probe gate (0.551, 0.488, 0.522 and 0.537). This design cannot tell layer 57’s lead from how the
    text survives.
  - **Two sanity checks say less than their names.** The read gain is computed from the weights alone before any answer
    (at 57 the controls are read 1.18 times as loudly), and the relabelled-control split cannot name anything, so it
    shows only that the reading favours neither label.

  A whole layer is taken at batch 64 (the step-0 rule: chair parts at 20 pieces at batch 96), at the trial’s measured
  12.4 answers a second while writing: about 12.1 hours at 48 pieces (12.7 with 2,048 controls), about 8.0–8.4 at 32
  (estimated, not timed); the first whole layer, with 2,048 controls, took 13.2 hours at 11.88. The owner chose, on 29 October, to count `break_rel`’s doses calibrated at layers 40–50 alone
  (pages 41–51): `DOSES_CALIBRATED` gives each design its span of layers and the card tests the run’s own, so 51–57
  and every layer below 40 stay uncalibrated. Every page and tool that shows a census says its doses are uncalibrated
  (in red), or not calibrated at that layer for `break_rel` outside 40–50, save a `break_rel` census at 40–50.

  **The owner’s choices, and the first whole layer** (29 October 2026, afternoon):
  - **48 word-pieces.** The 32-piece reading is the 48-piece answers cut short, so a run at 48 gives both. A
    `break_rel` census now takes 48 pieces, a batch of 64 and 2,048 controls unless told otherwise
    (`census.run_defaults`), and its card counts the doses calibrated only for answers of 32 or 48 pieces
    (`DOSES_CALIBRATED_MAX_NEW`). The pod scripts still take `main_rel` unless told, as their pilot projects its cost.
  - **±2 stays.** The design is run as it was trialled, so its calibration covers the run; a design without ±2 would be
    a new one, its ±1 figures above read after the fact.
  - **2,048 controls.** The smallest p halves, and about 10 of a layer’s 21,504 neurons beat every control by chance
    alone, against about 21 with 1,024.
  - **`break_rel` is the default design** (`census.DEFAULT_DESIGN`). A census takes it when none is named, and a layer’s
    run of it is read before others.
  - **Collapse words are left out of the words a page shows** (`census.COLLAPSE_RULE`): for `break_rel`, a word at
    least 10 of every 1,024 controls (and at least two) hold in two or more of their ±2 answers, each answer’s own
    question’s unpushed words left out as the ranking leaves them, read from each run’s own controls; and a word made
    only of such words run together (thoughtthought, ownthought). This is the review’s rule with its hand-picked clause
    for words holding “thought” replaced by the joined words. The ranking file keeps every word and marks these
    (`collapse`, with `collapse_mass`, their share of the mass, and `mass_shown`), and the layer’s list is kept beside it
    (`<run>.collapse.json`). The mass, p and q still count them, as the design was calibrated. So:
    - a neuron is shown by those of its five strongest words that are not collapse words, never by words further down;
    - it is described beside links and on the map only where those words alone beat every control’s;
    - the census list gives every neuron’s share of its mass from words not shown;
    - the details tab names the words it left out;
    - the scientist’s tools give the rule and the share.

    A ranking made before the rule is marked in place when the atlas opens the layer, its masses unchanged; until then
    the map gives its neurons no census words. On the trial’s own answers (CPU,
    [`census_collapse_rule/`](experiments/results/census_collapse_rule/)) the rule holds 15 words: the review’s seven
    besides its “thought” clause (and, idea, own, owns, rest, same, sss) and the thought family itself. It joins 187
    loops of them (andthoughtthought and the like) and hides no question’s topic word. Of the 180 neurons above every
    control, 107 take some of their mass from hidden words (median 10%) and one takes half; 168 keep a description on
    the map.
  - **What a lead says** (`census.lead_says`, on every census card’s provenance line, the Lexicon and the scientist’s
    prompt): every unit, a control too, has words above the mean it is set against; a high rank says that the
    neuron’s strongest words stand further above it than nearly every random direction’s own do, so its push differs
    from a random one; not that its words share a theme.

  The first whole layer was started that afternoon: layer 45 (page 46), `break_rel`, all 21,504 neurons against 2,048
  controls drawn with seed 7 (the trial’s were 3, the ladders’ 0), `--batch 64 --max-new 48 --unpacker compiled`,
  565,254 answers (`atlas_data/census/runs/L45-break_rel.*`). It finished at 04:49 CDT on 30 October, 13.2 hours of
  writing at 11.88 answers a second. It began before the collapse rule was written, so it was ranked again with the
  rule (`census_tools.py rank 45`), then read on the CPU with the trial’s own reader, unchanged, its hashes checked
  ([`census_L45_whole/`](experiments/results/census_L45_whole/), `summary.json`, `read.sh`, `compare.py`):
  - **Every gate passes, within 0.01 of the trial.** At 48 word-pieces: lead 0.713, beyond text survival 0.724, words
    picked on one half and scored on the other 0.680 and 0.671, the new questions alone 0.662. The ±1 answers alone
    clear every bar too (0.714). At 32 every gate passes but the lead falls 0.023 and replication 0.022, past the
    0.02 the length rule allows: on the whole layer that rule would keep 48, as the owner did.
  - **What the layer names.** 1,580 of 21,504 neurons (7.3%) beat every control, about 10.5 of them by chance alone;
    6,308 have q ≤ 0.05, and 5,123 of those are named (their words replicate across the halves both ways). The halves’
    ranks agree at 0.49 among neurons, 0.12 among controls. At ±1 the neurons’ text loses 0.04 of its distinct-4, the
    controls’ 0.004; at ±2 both lose 0.80, loops for both kinds as in the trial.
  - **The trial’s neurons against fresh controls.** They wrote the same answers again: 99.6% of the 1,022 fresh
    neurons’ answers bit for bit (24,438 of 24,528), all of the 64 re-asked and all six unpushed. So the comparison
    re-draws the null alone, and the null held: the controls’ mass p50 and p95 moved from 9.81 and 13.62 (1,024, seed 3)
    to 9.83 and 13.74 (2,048, seed 7). Of the trial’s 262 named, 256 are named again. The six lost sat at the edge,
    q 0.042–0.048, four of them taking 37–52% of their mass from collapse words. Against twice as many controls, 76 of
    the 1,022 beat every one (7.4%), the layer’s own share, so the trial was a fair sample.
  - **The collapse rule here.** With 2,048 controls it needs 20 of them to hold a word: idea, own, owns, rest, same,
    sss, thought and their 395 joins. “And” is held by 17 here (the trial’s 11 of 1,024 cleared its 10), so a join
    such as andthought is shown. Of the 1,580 above every control, 879 take some mass from hidden words (median 8%),
    5 half or more, and 1,023 keep a description on the map. With the hidden words left out the lead is 0.699.
  - **What it does not show.** The agreement with the trial is mostly built in: same answers, new controls. The only
    test of the neurons themselves is still inside the run, the halves. And a lead says that a neuron’s push differs
    from one along an isotropic random direction. A down column is a learned direction, and a learned direction may
    beat an isotropic one for reasons that belong to the layer, such as its write subspace or its alignment with the
    unembedding, not to the neuron. So controls drawn from the layer’s own columns were built and taken (below).

  The same day the layer was read against a second null: 2,048 controls mixed from its own write directions
  (`--control-kind columns`, seed 11), each a random mix of all 21,504 neurons’ unit write directions, matched to their
  spread (share on the layer’s top 100 write directions 5.18%, the neurons’ 5.31%, an isotropic direction’s 2.22%).
  They were taken alone (`--neurons none`, 49,158 answers, about 73 minutes in two sittings: a GPU watchdog stopped the
  first during a kernel build, and the same command finished it) and joined to the layer run’s neurons, which were
  not asked again (`census_tools.py join`; the six unpushed answers the same bit for bit). Read with the trial’s reader,
  unchanged ([`census_L45_columns/`](experiments/results/census_L45_columns/), `summary.json`, `read.sh`):
  - **Every gate passes against the mixes, as against isotropic controls.** Lead 0.712 (0.713), beyond text survival
    0.734 (0.724), replication 0.668 and 0.665 (0.680 and 0.671), the new questions alone 0.651 (0.662). The ±1
    answers alone clear every bar (0.723).
  - **The mixes read as random directions.** Set where neurons stand against the isotropic controls, they lead at
    0.502 (beyond text survival 0.481, replication 0.517 and 0.515); none beats every control and none reaches
    q ≤ 0.05. So the neurons’ lead is not the layer’s write subspace or its spread: it is the single neuron’s own
    direction that stands out.
  - **What the count above every control rests on.** It moved from 1,580 to 2,357, at ±1 alone from 1,032 to 6,522,
    while q ≤ 0.05 moved from 6,308 to 6,746 and the named from 5,123 to 5,237 (5,052 in both). Below their largest
    the two nulls agree within 0.15 (at ±1, p50 8.88 and 8.78, p99 12.42 and 12.39). The isotropic null’s largest is
    one control whose ±1 answers turn to Indonesian, words no other control writes, standing 10.5 above that null’s
    next. Set two nulls side by side by the gates and the q’s, not by that count.
  - **What it does not show.** The mixes are dense (each weighs every neuron), so they share the neurons’ second
    moment and nothing beyond it, such as a neuron’s sparsity. A mix of a few neurons would ask how many it takes to
    lose the lead. Not built. The neurons’ answers are the layer run’s, so the halves are still their only test.
- **The push’s size.** On the 31B the norm after each block caps what a single neuron can write, and the census’s 0.8 of
  the stream is 1 to 10 times that cap at most layers (review of 27 October 2026). A census can now be taken in each
  neuron’s own ceiling (`--dose-unit ceiling`): a neuron is pushed by a multiple of the most it can write by itself, and
  a control by the ceiling of a neuron it is paired with, so one dose is the same share of each neuron’s most at every
  layer and the depth curve compares like with like. Every answer records its ceiling and its push’s real share of the
  stream. Which unit, and which doses, a whole layer is taken at waits on the calibration, taken (above) but read so far
  only as leads; `break_rel`’s doses, ±1 and ±2 of the stream, were counted calibrated at layers 40–50 alone (above; at
  39–51 since 12 November, on the two trials below),
  and since 29 October it keeps ±2 and is the default design. The steer lens’s and the census’s 0.8
  is to be chosen against the ceiling the same way before either is built on the model.
- **The census’s null.** Its controls are drawn as unit(w_post ⊙ r), matched to within 1.2× (`check_census.py`); with C
  controls about N / (C + 1) neurons beat every control by chance alone (84 of a whole layer with 256), so each neuron
  carries a p and the layer a false-discovery q, and with 256 controls no single neuron reaches q ≤ 0.05 unless some
  1,700 of the layer beat every control. A second stage now exists: `census_run.py --neurons-from <screen>.units.jsonl
  --top M --seed <another>` retests a screen’s M strongest neurons against fresh controls, its p and q taken over those
  M alone (`stage` in the meta and on the card; the layer’s census stays the screen). Its cost is mostly the controls:
  about (200 + 4,096) × 10 answers, some 17 hours on the 5090 at `--batch 32 --max-new 64` (worked out from the
  pilot’s step). The owner chose 2,048 controls for the first whole `break_rel` layer (above). Its strongest cannot be
  retested with `break_rel` itself: on a shared cell a neuron writes its picked answer again (99.6% bit for bit,
  measured on that layer), and `census_run.py` refuses that. A retest needs a design whose questions no screen asked.
  Whether to build one is still to choose. A second kind of control exists beside the isotropic one: `--control-kind
  columns`, a random mix of the layer’s own write directions, which can be taken alone (`--neurons none`) and joined
  to a layer’s run (`census_tools.py join`); on layer 45 it reads as the isotropic one does (above).
- **The other ten layers, opened to volunteers** (built 30 October 2026; `VOLUNTEERS.md`). The calibrated layers left,
  40–44 and 46–50, were opened to anyone with one card of 32 GB or more. By 11 November 2026 every one was done: six
  by volunteers and four on the home card. Each volunteer runs layer 45's command at another layer, from the kit's
  tag, with the kit's pinned versions and stats index (`volunteers/kit.env`, `scripts/volunteer/census.sh`).
  - **Claims.** A layer is claimed by a one-line `CENSUS` message carrying a code whose secret the volunteer keeps,
    and the board keeps one claim a layer, with holds that lapse (`scripts/census_board.py`; the pinned issue, the
    thread, `volunteers/board.png`).
  - **The canary.** Each layer's first batch, taken on the home card (`volunteers/home_canaries/`), says before a run
    begins whether the volunteer's card writes the home card's bits or is a class of its own.
  - **The intake** (`scripts/census_intake.py`) checks a bundle's claim, settings and every answer. It then writes 5
    chunks, drawn after the bundle arrives, again on the home card: they must match bit for bit when the canary did,
    and must otherwise meet rules fixed before any run was seen (`census_volunteer.ACCEPT_OTHER_CLASS`; since 14
    November 2026 its two log-probability bars are a design’s at its layers, `OTHER_CLASS_BARS`, below under kit 4).
  - **Another card class.** A layer from another card class is recorded so in its provenance file, and is read as its
    own instrument.

  `scripts/check_volunteer_kit.py` runs it all on the stand-in, with a tampered run, a made-up one, a wrong secret and
  a stray file refused, and another card class taken. By 11 November 2026 six runs had come home from three volunteers
  and been accepted. Layer 41, from an RTX 5090, wrote the home card's bits: its 5 spot chunks matched bit for bit.
  Layers 47, 42, 44, 48 and 50, each from an RTX PRO 6000 Blackwell (42, 48 and 50 from one volunteer), are a card
  class of their own and met its rules as amended on 31 October (`VOLUNTEERS.md`). On their spot chunks the first
  word-piece was the same in 0.981, 0.975, 0.979, 0.981 and 0.987 of answers (bar 0.90), and the log-probabilities
  differed by a median 0.0133, 0.0146, 0.0140, 0.0145 and 0.0131 (bar 0.028) and by 0.194, 0.168, 0.155, 0.215 and
  0.197 at the 99th percentile (bar 0.37). Each of those five runs' unpushed answers parted from ours at the same
  three near-ties (pieces 5, 21 and 27, which the home card writes at p = 0.50, 0.41 and 0.50; bar 0.6). All six pass
  layer 45's gates, read after the fact.
  - **What layers 40–50 read** (`experiments/results/census_L40_whole/` to `census_L50_whole/`; every layer but 45
    read against 45's gates after the fact). Every gate passes at every layer. The length rule would take 32
    word-pieces at layers 40–43 and keeps 48 at 44–50 (at 49 on the replication drop alone, 0.026; the lead's was
    0.0199); every figure here is at 48. Set layers side by side by q ≤ 0.05 (3,534 to 9,133 neurons) rather than by
    the count above every control (95 to 4,794), which hangs on each layer's one largest control, as on layer 45's two
    nulls (above): at layer 41 a control whose answers turn to Spanish stands 9.7 above the next, and 869 neurons beat
    that next one. At layers 48 and 49 the ±1 push already breaks text: the neurons' answers lose 0.15–0.17 of their
    distinct-4 and the controls' 0.09–0.13, against at most 0.07 and 0.02 at 40–47. Their strongest neurons also stand
    nearer the controls: 1,408 and 1,407 lie more than five of the controls' standard deviations above the controls'
    median, against 2,007 to 4,742 at 40–47. Their other counts stay in range (1,250 and 1,474 beat every control,
    against 1,259 at layer 40 and 1,580 at 45; 6,554 and 5,899 have q ≤ 0.05), and the lead holds (0.736 and 0.743),
    beyond text survival too (0.753 and 0.763). Layer 50, the last the doses were calibrated for, lies between 40–47
    and 48–49 on text: its ±1 push costs the neurons 0.10 of their distinct-4 and the controls 0.04. Fewer of its
    neurons stand far above the controls than at any other layer (1,353 beyond five standard deviations), and only 584
    beat every control. At ±2 its strongest controls loop on glued words, most often “thought”, the word of the empty
    thought channel the chat template sets before every answer: 19 of its 20 largest controls have a glued “thought”
    word among their top five, as 17 and 14 of 20 do at 49 and 48. What is its own is how far four of them stand out,
    23.5 to 19.9 against 17.2 for the fifth, and 3,510 neurons beat that fifth. On its ±1 answers alone 6,145 beat
    every control, fourth of the eleven. It has 5,914 with q ≤ 0.05, and the lead holds (0.736, and 0.744 beyond text
    survival).
- **The other forty-nine layers: whether, and in what order** (12 November 2026; a plan, none of it run yet).
  - **What a census at every layer would add.** Three things the page cannot do without one. (1) Below layer 40 the
    atlas’s main reading shows no words for a neuron: the label gate holds the lens’s labels back there (the apparatus
    keeps them, marked as held back) and the steer lens is not built, so only its passages speak for it. A census there
    would be the only reading of what a neuron writes that the page could show, for two thirds of the model. (2) The web runs off the census’s edge. Of the 24 strongest links each way of 150 neurons the
    census leads on (30 drawn at each of layers 40, 43, 45, 47 and 50, seed 1, from the wiring index), 68% of the
    upstream ends and 72% of the downstream ones lie in a layer with a census, so about three links in ten end where
    no neuron has census words to show. Nearly all of those lie just outside the band: 1,123 of the 1,143 upstream ones at
    layers 38 and 39, 879 of the 994 downstream ones at 51 and 52 (53 and 54 a further 102). With those four layers
    taken, 99% and 97% of these links would end in a census. (3) The search of every census at once (below) follows a
    word through eleven layers and no further.
  - **Why it cannot simply be run.** A whole layer is 565,254 answers: 13.2 hours on the home 5090 and 2.7 GB kept. The
    49 layers left are 27 days of that card, or as many volunteer-nights, and about 135 GB. And `break_rel`’s doses are
    calibrated at 40–50 alone (above). Upward, the ±1 push already breaks text at 48–50 (the neurons lose 0.10–0.17 of
    their distinct-4), and at 57 the trial’s probe passed by 0.0008, failed at 32 word-pieces, and its lead could not be
    told from how the text survives. Downward, the ladders put the break at ±1 of the stream at layer 30 and ±0.5 at
    layer 20, so ±1 and ±2 would be read at the break or past it; at layer 20 a neuron’s ceiling is 0.10 of the stream,
    so ±1 is ten times the most it can write; and on the ladders’ 64 + 64 units the ranking told neurons from controls
    not at all at 20 (0.52, 0.50) and weakly at 30 (0.58, 0.59). The ladders under-read, though: at layer 45 they gave
    0.55–0.63 where the trial’s 1,022 neurons gave 0.714. So layers 20 and 30 are not shown to fail. They are not shown.
  - **The plan: grow the band outward, a trial before every whole layer.**
    1. *The four neighbours first* (38, 39, 51, 52), where a census would add the most to what is already taken and
       the design is likeliest to hold. A trial at each as layer 45’s was: about 1,022 fresh neurons against 1,024
       controls, `break_rel` as it is, read with the trial’s reader and its gates as fixed on 29 October (0.60, 0.58,
       0.57, 0.56), with the review’s two checks made gates before any is run: the ±1 answers alone, and the units whose
       text survival the controls cover (the check layer 57 failed). About 49,000 answers and 70 minutes each, one night
       for the four. A whole layer follows only where its trial passes: 13.2 hours each, 53 for the four, or two nights
       on the volunteers’ cards with the kit as it is (the design is the kit’s; a canary a layer on the home card).
       `DOSES_CALIBRATED` is widened a layer at a time, on a passed trial and the owner’s word.
    2. *On while trials pass* (37, 36, … and 53, 54, …), each next layer a 70-minute trial. Where one fails, the band
       stops on that side under this design.
    3. *Past where `break_rel` holds, a design scaled to each layer’s break.* At layer 45 the lead lay in the answers
       just under the break (±1, where text breaks at ±2). A design that takes each layer at half its break and at its
       break (±1 and ±2 at 40–47, as now; ±0.5 and ±1 at 30; ±0.25 and ±0.5 at 20) needs each layer’s break measured:
       both ladders at the layers that have none (64 + 64 units, about 20 minutes a layer), then its own trial, its
       gates fixed first, at some of 5, 10, 15, 20, 25, 30, 35 and 54, 57, 59. Not built. It would be a new design and,
       since it changes what writes answers, a new kit. Whether to build it is the owner’s choice, best made once step
       2 shows where the band stops. **Designed, 12 November 2026, once layer 38 had failed**
       ([`census_scaled_design/`](experiments/results/census_scaled_design/), `DESIGN.md`; nothing of it run). Not
       the half-break rule above: what separates the layers that pass from the two that fail is the push counted in
       write ceilings. Read from the checkpoint for all sixty layers (`ceilings.py`), 1 of the stream is 1.76 to
       2.76 ceilings at every layer from 39 to 51, 3.14 at layer 38, 1.21 at layer 57, and it climbs below 38: 3.9 at
       35–37, 4.2 to 5.0 at 26–34, 5 to 19 at 2–25. So the design, `break_ceil`, is `break_rel` with its two doses
       counted in ceilings: ±2.3 and ±4.6 of each neuron’s own write ceiling (2.3 the middle of the passing band),
       the questions, lengths, controls, reading and gates unchanged. That is 0.73 and 1.46 of the stream at layer
       38, 0.48 and 0.96 at 30, 0.21 and 0.43 at 20; at 20 and 30 the ladders show the lower dose leaving text whole
       and the upper at or past the break’s edge. It may not mend layer 38: its fall from 0.694 to 0.497 in one layer
       is a cliff where 39–51 show no slope, and it lies one layer from where the lens’s labels stop being trusted,
       so the trouble may be depth and not dose (a neuron that does not yet write towards words gives a census
       nothing to read). One trial decides: layer 38 again with `break_ceil`, the same 1,022 neurons, 75 minutes.
       A prediction was committed before layer 52’s trial was read (`PREDICTION.md`, `c9c5a82`): at 1.06 ceilings,
       the least of any layer, it fails as 57 did.
       **Trialled at layer 38 that evening** ([`census_ceil_trials/`](experiments/results/census_ceil_trials/),
       registered in `e853e5c`; `summary.json`). **It fails, on one figure of one gate:** the whole reading’s lead
       is 0.599 against a bar of 0.60. Drawing the neurons and controls again puts it between 0.573 and 0.623, so it
       sits on the bar; the rule fixed beforehand counts that a fail, and it is one. Everything else passes. The
       lower dose alone reads 0.682, 0.727, 0.652 and 0.636, 0.649, which is layer 39’s lower dose over again (0.690,
       0.728, 0.673 and 0.680, 0.652, at 2.14 ceilings); the neurons the controls cover (946 of 1,022) read 0.608,
       0.622, 0.597 and 0.589, 0.630. Under `break_rel` the same neurons led by 0.497. **So the dose was the trouble,
       not the depth:** at 2.3 ceilings a layer-38 neuron is told from a random direction as well as a layer-39 one.
       What holds the whole reading down is the upper dose. Read alone (`by_dose.py`; looked at after the verdict,
       gating nothing), the ±4.6 answers lead by 0.506, which is chance, where layer 39’s upper dose (4.28 ceilings)
       still leads by 0.595; and `break_rel`’s lower dose at 38, 3.14 ceilings, was already at chance (0.496). At
       layer 38 the lead is gone somewhere between 2.3 and 3.1 ceilings; at 39 it thins by 4.3, and at 45 the whole
       reading holds with an upper dose of 5.1. Half of this design’s answers at layer 38 therefore carry nothing.
       By the registration this is its third case: the lead has risen clear of chance and falls short of the bars,
       no whole layer is taken at 38 or below with ±2.3 and ±4.6, and what to try next is the owner’s choice. The
       plain candidate is the same design with both doses at or under about 2.3 ceilings, registered anew and
       trialled on neurons this trial did not ask, since the choice would be made on these answers. (Reported as
       registered: a neuron’s mass under the two designs goes together, Spearman’s rho 0.54 over the 1,022.)
       **Layer 52 with the same doses** was registered beside it in `a09f1bf` (1,022 neurons its first trial did
       not ask, fresh controls) and started at 22:01Z.
       **The softer design, `soft_ceil`** (the owner’s word that evening; built in `b6f7b54`,
       [`SOFT.md`](experiments/results/census_scaled_design/SOFT.md)): `break_ceil` with its doses at
       ±1.6 and ±2.3 ceilings, the dose that read at layer 38 and one half a doubling under it. The ladders of 28
       October point the same way as far as 64 units can: at layer 30 a push of two ceilings leads (0.604) and one of
       one or of four does not. Its trial at layer 38 is registered
       ([`census_soft_trials/prereg/`](experiments/results/census_soft_trials/prereg/), hashed at 22:34Z before
       layer 52’s trial was read): 1,022 neurons neither earlier trial there asked, 1,024 fresh controls, the same
       reader and three gates. Said beforehand: the whole reading and the covered neurons pass; whether ±1.6 alone
       carries a lead is the open question. It ran after layer 52’s trial and before the steer lens.
       **Layer 52’s `break_ceil` trial, read at 23:30Z: it fails too, and by the same dose.** Its whole reading
       leads by 0.676, but that lead is not told from how the text survives: 0.553 beyond text survival against a
       bar of 0.58, on every neuron and on the 1,010 of 1,022 the controls cover. That is the doubt its registration
       named. The lower dose alone passes every gate on these new neurons (0.790, 0.679, 0.649 and 0.692, 0.692),
       repeating what the first trial’s ±2 of the stream (2.12 ceilings) showed on other neurons. Read alone
       afterwards (`by_dose.py`, gating nothing), the ±4.6 answers lead by 0.485, which is chance: their text has
       lost 0.83 of its distinct 4-grams, neurons and controls alike. So on both sides of the band a push of 2.3
       ceilings reads and one of 4.6 reads nothing, and the 4.6 answers are what fail each trial: at 38 they pull
       the lead down to its bar, at 52 the lead beyond text survival under its bar. By the registration no whole
       layer is taken at 52 with ±2.3 and ±4.6. One caution about layer 52: even 2.3 ceilings is 2.2 of the stream
       there, and at that dose the text has already lost 0.60 (neurons) to 0.73 (controls), so the lower dose’s
       lead rests on the covariate fit holding in broken text. Whether `soft_ceil` is trialled at 52 as well is
       the owner’s choice; it would want a third draw of neurons, both earlier draws having been read at 2.3.
       **Layer 38’s `soft_ceil` trial, read at 00:52Z on 13 November: it passes all three gates**
       ([`census_soft_trials/`](experiments/results/census_soft_trials/)). The whole reading leads by 0.675 (0.712
       beyond text survival; 0.636 and 0.630 across the question halves; 0.643 on the new questions). The lower
       dose alone, ±1.6 ceilings, which nothing had read before, leads by 0.654 (0.671; 0.594 and 0.605; 0.623).
       The 839 of 1,022 neurons the controls cover lead by 0.628 (0.673; 0.589 and 0.583; 0.598): the thinnest of
       the three, 0.013 over its bar on one replication figure. 110 neurons beat every control, where chance
       gives one. Said beforehand and met: the whole lead (0.64 to 0.70), the lower dose’s (0.60 to 0.66), and the
       ±2.3 answers alone on new neurons (0.655; said 0.64 to 0.72, and 0.682 on the first neurons). Said
       beforehand and missed: nine neurons in ten covered or more. It is eight in ten, because at these doses a
       random direction hardly moves the text (0.004 of its distinct 4-grams lost at ±2.3) and a neuron moves it
       more (0.03), so more neurons fall outside the controls’ range; the lead among the covered is lower than the
       whole lead for the same reason. By the registration this is the design for the walk downward: a trial
       before every whole layer (37, 36, …: 1,022 neurons a layer were drawn in the `break_ceil` registration and
       none has been asked), and a layer that passes may go to volunteers under a kit of its own.
       `DOSES_CALIBRATED` names no layer for `soft_ceil` yet: widening it to 38 is the owner’s word. What the pass
       does not show: that any layer below 38 reads with these doses, or that layer 52 does. The steer lens
       started behind this trial at 00:52Z.
       **What the steer lens’s pilot says of these doses** (taken that night, read afterwards; it gates nothing:
       `pilot_in_ceilings.py`, [`pilot.json`](experiments/results/atlas_steer_dose/pilot.json)). The pilot gives, at
       every layer, how far a push of a given size moves the next word-piece. Set beside the fourteen doses a trial
       has read alone, it sorts them cleanly where neither a share of the stream nor a count of ceilings did: the
       eight that move the next word-piece by 0.35 to 2.4 nats lead by 0.65 to 0.80, at five layers; the one that
       moves it by 0.18 leads by 0.60; the five that move it by more than 3 nats lead by 0.485 to 0.595. At layer 38,
       `soft_ceil`’s two doses are 0.35 and 1.34 nats. Going down, its upper dose stays inside that span at layers
       25–37 (0.64 to 1.31 nats) and leaves it at 23 and 24 (3.4 and 3.1). Its lower dose sits at the span’s low
       edge or under it at most layers below 38: 0.27 nats at 37 and 36, 0.21 at 34, 0.24 at 31. So the gate to
       watch on the walk is again the lower dose alone. The span was drawn after its fourteen readings were known,
       and the figures between the pilot’s rungs are read off a line; a trial is what tests it.
       **The walk downward, layers 37 to 30** (the owner’s word, 13 November: “Yes, walk first”; registered in
       [`census_soft_walk/prereg/`](experiments/results/census_soft_walk/prereg/), hashed at 02:18Z before any
       answer of these layers existed; started at 02:18Z). A `soft_ceil` trial at each layer in turn on the 1,022
       neurons drawn for it on 12 November, read as it is written; the next layer down is taken only if this one
       passed all three gates, and the walk stops at the first failure and at layer 30 at the latest (about 83
       minutes a layer). Said beforehand: the whole reading and the covered neurons pass wherever it reaches; the
       lower dose alone is the open gate and what it is expected to stop on, about even at 37 and 36 and more
       likely to fail than pass at 34; more likely than not it stops before layer 30.
       **Layer 37, read at 03:47Z: it fails, and the walk stops there**
       ([`census_soft_walk/`](experiments/results/census_soft_walk/)). The whole reading passes (lead 0.672; 0.695
       beyond text survival; 0.637 and 0.616 across the question halves; 0.648 on the new questions). The lower dose
       alone fails on one figure, the words picked on the second half of the questions and scored on the first:
       0.542 against 0.57 (its other figures pass: 0.633, 0.630, 0.583, 0.616). The 832 of 1,022 neurons the
       controls cover fail on the same figure by 0.002: 0.568 against 0.57 (0.633, 0.651, 0.602, 0.611). 28 neurons
       beat every control, where layer 38 had 110. Said beforehand and met: the whole reading passes; the lower dose
       alone is what stops the walk, about even at 37; its replication figures fall before its lead. Said beforehand
       and missed: that the covered neurons pass wherever the walk reaches. By the registration no whole layer is
       taken at 37 with ±1.6 and ±2.3 ceilings, and nothing is said of 36 and below, which were not trialled.
       Reported: the ±2.3 answers alone pass every gate here (0.657, 0.667, 0.604 and 0.605, 0.619). **The pilot’s
       forecast held on both new readings:** it put this layer’s lower dose at 0.27 nats, between the 0.18 that led
       by 0.603 and the 0.35 that led by 0.654, and it leads by 0.633; it put the upper dose at 0.95 nats, inside
       the span that reads, and it leads by 0.657. So `soft_ceil` is one layer wide below the band: 1.6 ceilings is
       a little too soft at 37. By the pilot the two pushes that pass at layer 38 (0.35 and 1.34 nats) come to
       1.73 and 2.53 ceilings at layer 37, and to between 1.6 and 1.9 and between 2.4 and 2.9 at layers 30–36. A
       design that takes each layer’s two doses from the pilot’s curve is the next candidate; it is the owner’s
       choice, is not built, and would be registered on neurons this walk did not ask.
       **Layer 52 with `soft_ceil`** (the owner’s word, 13 November; registered in
       [`census_soft_trials/prereg_L52/`](experiments/results/census_soft_trials/prereg_L52/), hashed at 02:50Z
       while layer 37 was being written): a third draw of 1,022 neurons that neither earlier trial there asked,
       1,024 fresh controls. It started at 03:48Z, behind the walk and ahead of the steer lens’s build.
       **Read at 05:14Z: it passes all three**
       ([`census_soft_trials/`](experiments/results/census_soft_trials/), `summary.json`). Whole: 0.804, 0.693,
       0.680 and 0.669, 0.758. The ±1.6 answers alone: 0.633, 0.657, 0.604 and 0.608, 0.577 (that last against a
       bar of 0.56). The 945 of 1,022 neurons the controls cover: 0.797, 0.684, 0.670 and 0.658, 0.750. The ±2.3
       answers alone, reported: 0.783, 0.682, 0.642 and 0.681, 0.675. 245 of the 1,022 beat every control (one by
       chance). 86 minutes at 9.6 answers a second. The lead beyond text survival, which failed here under
       `break_ceil` at 0.553, is 0.693 with the softer pair. Two of the four figures said beforehand missed: the
       whole lead (0.804; 0.72 to 0.80 was said) and the lower dose’s lead (0.633; 0.66 to 0.76 was said). The
       second one matters outside this layer. By the steer lens’s pilot 1.6 ceilings moves the next word-piece by
       0.58 nats here, inside the span (0.35 to 2.4 nats) where every dose read before it led by 0.65 or more. So
       how far a push moves the next word-piece sorts the leads less closely from layer to layer than those
       readings suggested, and `match_ceil`, below, rests on it. By its registration layer 52 may be offered to
       volunteers under a kit of its own for this design. `DOSES_CALIBRATED` is not widened to it, and layers 53
       and above each want a trial of their own; each of those is the owner’s word.
       **The design with each layer’s own doses, `match_ceil`** (the owner’s word, 13 November: “Yes, design it”;
       [`MATCH.md`](experiments/results/census_scaled_design/MATCH.md)): `soft_ceil` with its two counts of ceilings
       set a layer at a time, at the pushes that move the next word-piece by 1.0 and by 1.4 nats there. 1.0 nats is
       what `break_rel`’s lower dose makes at layers 39 and 45 and the push that passed alone at layer 37; 1.4 is
       the hardest push that has passed below the band. The counts come from a finer dose pilot
       (`scripts/census_dose_pilot.py`: the steer lens’s measurement on rungs 7% apart, where the first pilot’s
       rungs are a factor of two apart and every dose in question lies between the same two), by a rule fixed
       before that pilot is seen. Built and checked on the stand-in; a layer without measured doses is refused.
       **The fine pilot, taken at 05:14Z to 05:24Z** (layers 30–38 and the reference layer 45;
       [`fine_pilot.json`](experiments/results/census_scaled_design/fine_pilot.json)). The two counts of ceilings a
       layer, for 1.0 and for 1.4 nats, as they came out and as `MATCH_CEIL` now holds them: 2.42 and 2.67 at layer
       30, 2.58 and 2.76 at 31, 2.21 and 2.44 at 32, 2.31 and 2.51 at 33, 2.62 and 2.79 at 34, 2.27 and 2.41 at 35,
       2.27 and 2.46 at 36, 2.41 and 2.55 at 37, 2.14 and 2.30 at 38. The finer rungs correct the coarse pilot’s
       figures in ways that matter. Layer 37’s 2.3 ceilings, which passed alone, is 0.77 nats and not 0.95, so both
       of layer 37’s doses are harder than any push read there. Layer 38’s 1.6 ceilings is 0.26 nats and layer 37’s
       is 0.25, so at that dose the two layers were pushed alike by this measure, and the pilot does not explain why
       one passed and the other failed. Near the doses a push 7% larger moves the next word-piece 1.3 to 1.5 times
       as far, and the two doses are 6% apart.
       **Layer 37 with `match_ceil`** (registered in
       [`census_match_trials/prereg/`](experiments/results/census_match_trials/prereg/), hashed at 05:28Z on 13
       November before any answer of the design existed; the sha256 of its `SHA256SUMS` is `d4ebb22feb1a…`): ±2.41
       and ±2.55 ceilings, on 1,022 neurons the walk did not ask against 1,024 controls drawn afresh, the same
       reader and three gates. Said beforehand: about as likely to pass all three as not, the covered neurons the
       least sure. It ran ahead of the steer lens’s build.
       **Read at 06:45Z: it passes all three** ([`census_match_trials/`](experiments/results/census_match_trials/),
       `summary.json`). Whole: 0.653, 0.678, 0.609 and 0.627, 0.654. The ±2.41 answers alone: 0.647, 0.672, 0.599
       and 0.600, 0.628. The 857 of 1,022 neurons the controls cover: 0.629, 0.664, 0.585 and 0.606, 0.636. The
       ±2.55 answers alone, reported: 0.616, 0.635, 0.615 and 0.596, 0.618. 30 of the 1,022 beat every control (one
       by chance). 75 minutes at 10.9 answers a second. What was said beforehand held, but for the share covered
       (84 in 100; seven to eight in ten was said). Four cautions, each in `MATCH.md`: the two doses are 6% apart,
       so this is close to one dose read twice; it is layer 37’s second trial, and the second design was made
       knowing what the first showed; the harder dose leads by less (0.616 against 0.647), so 1.4 nats is nearer
       the edge here than at layer 38 and is the target to watch further down; and the rule the doses come from
       is weaker than when it was made. By its registration `match_ceil` is the design for the layers below 38, a
       trial before each, and layer 37 may be offered to volunteers under a kit of its own. `DOSES_CALIBRATED` is
       not widened, and the walk does not go on at 36, without the owner’s word. Three layers outside 39–51 now
       pass a trial, under two designs: 37 with `match_ceil`, 38 and 52 with `soft_ceil`.
       **The walk downward with `match_ceil`, layers 36 to 30** (the owner’s word, 13 November: “Walk first”;
       registered in [`census_match_walk/prereg/`](experiments/results/census_match_walk/prereg/), hashed at 07:25Z
       before any answer of these layers existed; the sha256 of its `SHA256SUMS` is `2d876b7c297a…`). A trial at
       each layer in turn at that layer’s two doses, on the 1,022 neurons drawn for it on 12 November, none asked
       before, against 1,024 controls; read as soon as it is written; the next layer down only on a pass; stopping
       at the first failure and at layer 30, where the fine pilot’s table ends. Said beforehand: layer 36 about as
       likely to pass as not; more likely than not the walk stops at layer 34 or above; all seven passing perhaps
       one chance in twenty. The steer lens’s build is paused behind it again, with layers 0 to 4 built.
       **Layer 36, read at 08:51Z: it passes all three, the second by 0.001**
       ([`census_match_walk/`](experiments/results/census_match_walk/), `summary.json`). ±2.27 and ±2.46 ceilings.
       Whole: 0.627, 0.636, 0.608 and 0.606, 0.612. The ±2.27 answers alone: 0.615, 0.622, 0.586 and 0.571, 0.600.
       The 980 of 1,022 neurons the controls cover: 0.622, 0.632, 0.600 and 0.600, 0.609. The ±2.46 answers alone,
       reported: 0.601, 0.619, 0.574 and 0.584, 0.595. 68 minutes at 12.1 answers a second. The pass rests on the
       lower dose’s words carrying across the question halves by 0.5709 against 0.57; another draw of neurons
       could fail as easily as pass. What was said beforehand held, but for the share covered (96 in 100; seven
       to nine in ten was said). The upper dose alone leads by 0.601, at the mark said beforehand for a target
       set too high. Fewer neurons stand out singly: 6 of the 1,022 beat every control (30 at layer 37, 110 at
       38) and 21 in 100 beat 95 in 100 of the controls (31 at 37, 35 at 38), so a whole layer taken here would
       tell less about each neuron. The whole lead has fallen at every layer going down: 0.694, 0.675, 0.653,
       0.627 from 39 to 36. By the registration layer 36 may be offered under a kit of its own, and the walk went
       on to layer 35 at 08:51Z. `DOSES_CALIBRATED` is not widened. Four layers outside 39–51 now pass a trial:
       36 and 37 with `match_ceil`, 38 and 52 with `soft_ceil`.
       **Layer 35, read at 10:01Z: it passes all three, with more room than 36.** ±2.27 and ±2.41 ceilings. Whole:
       0.664, 0.643, 0.639 and 0.623, 0.657. The ±2.27 answers alone: 0.667, 0.653, 0.582 and 0.595, 0.631. The
       978 of 1,022 neurons the controls cover: 0.658, 0.639, 0.632 and 0.615, 0.653. The ±2.41 answers alone,
       reported: 0.645, 0.641, 0.607 and 0.595, 0.616. The lead did not go on falling: it was said beforehand
       that it falls a little from layer to layer, and it rose from 0.627 to 0.664. The share covered missed
       again (96 in 100). More of the lead goes with how the text survives: beyond text survival is under the
       lead for the first time below the band, and the controls’ own masses agree between the question halves
       more (a correlation of 0.13, against 0.05 at 36 and 0.01 at 38); the gate made for that is met by 0.06.
       3 of the 1,022 beat every control. The card faulted once while the layer was written (Xid 8 at 09:17:31Z,
       after 18,240 answers); the walk’s script began it again as registered and every answer is there. The two
       chunks written around the fault (1,536 answers, 1,344 of them the last before it) were written again at
       14:53Z when the walk had freed the card: every one is the trial’s answer bit for bit, and both canaries
       too (`census_match_walk/spot_compare.py`). The answers of the two minutes before the fault are sound;
       the 16,896 written earlier were not asked again. By the registration
       layer 35 may be offered under a kit of its own, and the walk went on to layer 34 at 10:01Z. Five layers
       outside 39–51 now pass a trial: 35, 36 and 37 with `match_ceil`, 38 and 52 with `soft_ceil`.
       **Layer 34, read at 11:11Z: it passes all three, by more than any layer below the band.** ±2.62 and ±2.79
       ceilings, one session, no fault. Whole: 0.758, 0.747, 0.683 and 0.692, 0.709. The ±2.62 answers alone:
       0.745, 0.740, 0.640 and 0.650, 0.675. The 977 of 1,022 neurons the controls cover: 0.755, 0.742, 0.676 and
       0.687, 0.706. The ±2.79 answers alone, reported: 0.728, 0.723, 0.657 and 0.653, 0.659. Two things said
       beforehand were wrong, both on the low side: the whole lead was to lie between 0.61 and 0.68 at every
       layer reached, and the walk was more likely than not to stop at layer 34 or above. 42 neurons in 100 beat
       95 in 100 of the controls (24 at 35, 21 at 36); 8 of the 1,022 beat every control. From 36 to 34 the same
       two targets in nats gave leads of 0.627, 0.664 and 0.758: the dose rule gives doses that pass and does
       not say by how much. By the registration layer 34 may be offered under a kit of its own, and the walk
       went on to layer 33 at 11:11Z. Six layers outside 39–51 now pass a trial: 34 to 37 with `match_ceil`, 38
       and 52 with `soft_ceil`.
       **Layer 33, read at 12:26Z: it passes all three, a little under layer 34.** ±2.31 and ±2.51 ceilings, one
       session, no fault. Whole: 0.726, 0.726, 0.630 and 0.637, 0.691. The ±2.31 answers alone: 0.710, 0.706,
       0.601 and 0.617, 0.660. The 982 of 1,022 neurons the controls cover: 0.727, 0.721, 0.628 and 0.636, 0.691.
       The ±2.51 answers alone, reported: 0.702, 0.697, 0.605 and 0.591, 0.649. The whole lead is again above the
       0.61 to 0.68 said beforehand; it fell from layer 34’s, as said, after rising twice. 37 neurons in 100 beat
       95 in 100 of the controls; 29 of the 1,022 beat every control. From 36 to 33 the lead is 0.627, 0.664,
       0.758, 0.726: no fall towards the bar so far. By the registration layer 33 may be offered under a kit of
       its own, and the walk went on to layer 32 at 12:26Z. Seven layers outside 39–51 now pass a trial: 33 to
       37 with `match_ceil`, 38 and 52 with `soft_ceil`.
       **Layer 32, read at 13:40Z: it passes all three, with less room than 33 or 34.** ±2.21 and ±2.44 ceilings,
       one session, no fault. Whole: 0.699, 0.678, 0.615 and 0.631, 0.675. The ±2.21 answers alone: 0.671, 0.634,
       0.589 and 0.602, 0.640. The 942 of 1,022 neurons the controls cover: 0.686, 0.663, 0.604 and 0.619, 0.660.
       The ±2.44 answers alone, reported: 0.690, 0.667, 0.592 and 0.601, 0.631. The harder dose leads by more
       here, the first layer of the walk where it does, and more of the lower dose’s lead goes with how the text
       survives than at any layer of the walk before it. The whole lead is above the 0.61 to 0.68 said beforehand
       for the third layer running. 32 neurons in 100 beat 95 in 100 of the controls; 54 of the 1,022 beat every
       control. By the registration layer 32 may be offered under a kit of its own, and the walk went on to layer
       31 at 13:40Z. Eight layers outside 39–51 now pass a trial: 32 to 37 with `match_ceil`, 38 and 52 with
       `soft_ceil`.
       **Layer 31, read at 14:52Z: it fails, and the walk stops.** ±2.58 and ±2.76 ceilings, one session, no
       fault. Whole: 0.631, 0.634, 0.582 and 0.580, 0.618. The ±2.58 answers alone: 0.634, 0.632, 0.553 and
       0.548, 0.587. The 971 of 1,022 neurons the controls cover: 0.631, 0.628, 0.581 and 0.579, 0.616. The ±2.76
       answers alone, reported: 0.615, 0.620, 0.556 and 0.550, 0.601. It fails on one gate: the lower dose’s
       answers alone do not carry their words between the question halves (0.553 and 0.548 against 0.57), the
       failure said beforehand to come first. The lead holds at both doses. Neither dose alone carries and the
       two together do, so the trial points at no other push. Said beforehand: the walk would more likely than
       not stop at 34 or above; it stopped three layers further down. By the registration no whole layer is taken
       at 31 with this design and layer 30 is not asked. `match_ceil` passes at 32 to 37 and fails at 31. Eight
       layers outside 39–51 pass a trial: 32 to 37 with `match_ceil`, 38 and 52 with `soft_ceil`;
       `DOSES_CALIBRATED` was widened for none of them that afternoon.
       **The doses counted calibrated at the eight layers** (the owner’s word, 13 November: “Yes, prepare it”).
       `census.DOSES_CALIBRATED` names three designs now: `break_rel` at 39–51 as before, `soft_ceil` at 38 and
       52, `match_ceil` at 32–37, the two new ones for answers of 48 word-pieces alone, as their trials were
       gated. A design’s layers are listed, not spanned: `soft_ceil` was never tried between 38 and 52, and is
       not calibrated there. The card’s `doses_calibrated_layers` lists every layer; the page, the lexicon, the
       scientist’s prompt and the depth report say where each design is calibrated. No whole layer has been taken
       with either new design yet. Kit 3 offers seven of the eight to volunteers since 14 November 2026; layer 36
       is held for the home card (below). (Since then three `match_ceil` layers have been taken whole: 36 on the
       home card, 34 and 35 by a volunteer; below, under kit 4.)
       **The walk downward with `match_wide`, layers 31 and 30** (the owner’s word, 16 November: “One dose, same
       cost”; registered in [`census_wide_trials/prereg/`](experiments/results/census_wide_trials/prereg/), hashed at
       15:22Z before any answer of the design existed). Re-read on fewer questions, `match_ceil`’s failed trial at 31
       rose with each question a half (`census_question_count/NOTE.md`), so `match_wide` asks the lower of its two
       doses alone on twelve questions, two halves of six, at the same 24 answers a neuron. Gates N1 and N3 as
       before; with one dose there is no N2. **Layer 31, read at 16:59Z: it passes both.** ±2.58 ceilings, 79
       minutes in two sessions (paused three minutes for the fine pilot at 53–59). Whole: 0.630, 0.645, 0.583 and
       0.581, 0.608. The 986 of 1,022 neurons the controls cover: 0.636, 0.640, 0.586 and 0.587, 0.613. Read on
       break_rel’s six questions alone, as the `match_ceil` walk read them, its words carry by 0.549 and 0.539,
       where that walk’s lower dose was: the pass is the added questions’, not a better draw. Said beforehand and
       missed: a lead of 0.66 to 0.76 (it did not grow with questions; the projection assumed it would). 6 of the
       1,022 beat every control and none is named, though 31 and 39 are named on either six questions alone.
       **Layer 30, read at 18:23Z: it fails both, and the walk stops.** ±2.42 ceilings, 82 minutes. Whole: 0.579,
       0.611, 0.555 and 0.550, 0.584; the 974 covered: 0.582, 0.604, 0.557 and 0.551, 0.585. It fails on the lead
       and on words carrying between the halves, with the text whole. By the registration layer 31 may be offered
       under a kit of its own once its other card class’s bars are registered, `DOSES_CALIBRATED` is not widened
       without the owner’s word, and nothing is taken at 30 with this design. Nine layers outside 39–51 now pass a
       trial: 31 with `match_wide`, 32 to 37 with `match_ceil`, 38 and 52 with `soft_ceil`
       ([`census_wide_trials/RESULT.md`](experiments/results/census_wide_trials/RESULT.md)).
       **Two doses at layer 30, `match_both`** (the owner’s word, 16 November: “Two doses at 30 first”; registered in
       [`census_both_trial/prereg/`](experiments/results/census_both_trial/prereg/), hashed at 22:34Z before any
       answer of the design existed). Re-read on one to six questions a half, `match_wide`’s lead at 30 stopped
       rising at three, so this design asks its twelve questions at both of `match_ceil`’s doses instead, ±2.42 and
       ±2.67 ceilings, 48 answers a neuron. Gates N1 and N3; each dose alone is reported, not gated. **Read at
       02:04Z on 17 November: it fails both, on one figure.** 98,220 answers in 167 minutes. Whole: 0.618, 0.640,
       0.573 and 0.545, 0.608; the 956 covered: 0.622, 0.631, 0.577 and 0.546, 0.612. The lead passes, and the
       second dose carried more than a repeat (0.596 to 0.618; written twice, the same answers rise 0.007). But a
       neuron’s words picked on half B do not come back on half A (0.545 against 0.57). Said beforehand and missed:
       that a failure would come on the lead. The text loses under 0.005 of its distinct 4-grams. By the registration
       nothing is taken at 30 with this design, kit 5 is prepared for 31 with `match_wide` alone, and what to try at
       and below 30 is the owner’s choice
       ([`census_both_trial/RESULT.md`](experiments/results/census_both_trial/RESULT.md)).
       **Why the words do not carry at 30 and 53, read again from the trials** (17 November, exploratory, CPU, about
       two minutes; [`census_word_carry/NOTE.md`](experiments/results/census_word_carry/NOTE.md)). G3’s statistic was
       taken on matched sets of questions, picked on two and scored on one (or three and three) of the same half or
       of the other, on eighteen trials. It gives the registered figures back exactly when read whole. The halves
       are not what stops the words: on every trial they carry as well between the halves as within one (within
       minus between −0.014 to +0.017; at 30, −0.005 to +0.008). At 30 they carry faintly every way (0.52 to 0.55,
       where 32–36 read 0.56 to 0.62), and what differs is where they come back: half A’s questions, about the model
       and colours, take them up less than half B’s whichever half they were picked on (0.533 against 0.554 under
       `match_both`), so B→A fails. At 31 the halves take them up alike. The harder dose carried no more than the
       lower at 30. No neuron is named at 30 in either trial, as none is at 31 under `match_wide` or at 34–36, which
       passed: one control stands 12 SD out (it writes Spanish). Against the controls, 32 neurons are strong on both
       halves where about 4 would be by chance, and with the five strongest controls set aside 65 have q ≤ 0.05
       (35: 25, 36: 0). Under `match_both` five controls stand 7 to 12 SD out and the count stays 0. Nothing decided.
       **Layer 30 under `match_things`** (the owner’s word, 17 November: “You can go with design whose questions all
       take words up the way half B’s do”; registered in
       [`census_things_trial/prereg/`](experiments/results/census_things_trial/prereg/), hashed at 00:20:38Z on 18
       November and pushed at 02:06Z, before any answer of the design existed). It asks `match_both`’s two doses on
       twelve questions of the three genres that took words up at 30: how a thing works, three tips, an object. Half B
       is six questions no run had asked. Step 0 took two rounds: suitcase and both its spares parted, and money took
       its place by a rule hashed before its check. **Read at 05:04Z on 18 November: it passes both gates.** 98,220
       answers in 175 minutes. Whole: 0.618, 0.637, 0.582 and 0.587, 0.625; the 998 covered: 0.615, 0.633, 0.580 and
       0.586, 0.623. The words carry both ways, by 0.010 to 0.017 over the bar, under one standard error at the
       narrowest. Reported: the lower dose alone passes all four figure gates (0.616, 0.634, 0.588 and 0.581, 0.606),
       so the second dose added almost nothing here. Said beforehand and missed: that neither dose alone would. None
       is named; with the five strongest controls set aside (they write Portuguese, Spanish and Russian), 45 have
       q ≤ 0.05. It is the third design tried at 30, built on what the first two showed. By the registration layer 30
       may be offered under a kit carrying `match_things` (about 33 hours a layer on the 5090) once its other card
       classes’ bars are registered, and `DOSES_CALIBRATED` is not widened without the owner’s word. A layer so taken
       is asked none of who, orange or free. Ten layers outside 39–51 now pass a trial: 30 with `match_things`, 31
       with `match_wide`, 32 to 37 with `match_ceil`, 38 and 52 with `soft_ceil`
       ([`census_things_trial/RESULT.md`](experiments/results/census_things_trial/RESULT.md)).
       **The walk with `match_things`, up from 53 and down from 29** (the owner’s word, 18 November: “Yes, both ways”;
       registered in [`census_things_walk/prereg/`](experiments/results/census_things_walk/prereg/), hashed at 16:18Z
       before any answer of the design existed at these layers; the sha256 of its `SHA256SUMS` is `cf541b422de4…`).
       First the fine dose pilot was taken at 25 to 29 (15:43Z to 15:52Z;
       [`fine_pilot_low.json`](experiments/results/census_scaled_design/fine_pilot_low.json)), and its doses went into
       `MATCH_CEIL` (2989b14): 2.15 and 2.20 ceilings at 25, 2.05 and 2.12 at 26, 2.03 and 2.13 at 27, 2.04 and 2.16
       at 28, 2.10 and 2.32 at 29. 29 reads like 30 to 38. At 25 to 28 the doses sit on a steep step: near the lower
       one a push 7% larger moves the next word-piece 1.6 to 2.8 times as far (1.26 to 1.55 at 29 to 38), and the two
       are 2% to 6% apart. The walk takes 53, 29, 54, 28, 55, 27, 56, 26, 57, 25, 58 in turn. Each layer gets 1,022
       neurons no census run had asked there, against 1,024 controls (seed 11), and is read by the layer-30 trial’s
       reader byte for byte. Each arm stops at its first layer that fails N1 or N3. A third gate, N2 (the lower dose
       alone), counts only where a layer passes. It neither stops nor carries the walk, and a pass on it would let the
       layer be taken at one dose, half the answers, as a design of its own. Said beforehand: 53 and 29 each pass
       about 1 in 3, and either arm reaching its end under 1 in 20. Up to eleven trials, about 32 hours on the home
       card. Pushed at 16:23:53Z and begun at 16:24:00Z, on the owner’s “Push and start”. **It took four layers and
       stopped at 29 and 55** ([`census_things_walk/RESULT.md`](experiments/results/census_things_walk/RESULT.md)). 53
       passed (whole lead 0.751, words carrying 0.591 and 0.598), and so did 54 (0.712; 0.627 and 0.620), which passed
       N2 as well, narrowly (its lower dose’s words carry 0.575 and 0.583). 29 failed on the lead and the words carrying
       (0.574; 0.528 and 0.520), and 55 on the words carrying alone (0.518 and 0.539), so 25 to 28 and 56 to 58 were not
       trialled. The lead came in over its forecast at 53 (0.62 to 0.72) and just under it at 29 (0.58 to 0.64). The
       home card’s display watchdog (Xid 8) stopped trials three times; each was continued row for row, with no answer
       written twice. 53 and 54 may be offered under a kit that carries `match_things` there once their other card
       classes’ bars are registered; that kit, and widening `DOSES_CALIBRATED`, are the owner’s word (kit 7 carries
       them since 19 November, below). Twelve layers
       outside 39–51 now pass a trial: 30, 53 and 54 with `match_things`, 31 with `match_wide`, 32 to 37 with
       `match_ceil`, 38 and 52 with `soft_ceil`.
    4. *Cheaper than any of these: the steer lens, all sixty layers in a night* (10.1–10.8 hours, above). It gives every
       one of the 1,290,240 neurons a pushed reading that the page’s search already reads. It reads the next
       word-piece, not an answer, and has no controls of its own, so it is not a census; but it is the only push-based
       reading within a night’s reach of the layers below 40. Its dose is still to choose (above).
  - **The first trials** (12 November 2026; [`census_neighbour_trials/`](experiments/results/census_neighbour_trials/),
    `summary.json`). They were registered in `08fd50b` while the first was still being written: the reader (the first
    trial’s own, unchanged), three gates and each layer’s neurons, hashed before any answer was read (the sha256 of
    its `SHA256SUMS` is `ea5a984e6d80…`). Each trial is 1,022 neurons against 1,024 controls, 49,110 answers, 76 to
    80 minutes at 10.5 to 10.9 answers a second beside a desktop. A trial passes when three readings all clear layer
    45’s bars (0.60, 0.58, 0.57 and 0.57, 0.56) at 48 word-pieces: the whole reading; the ±1 answers alone; and the
    neurons whose six text-survival covariates all lie inside the controls’ range (at least half of them must). The
    last two are the checks the review of the first trial made after the fact, which layer 45 passed and layer 57
    failed; here they were gates from the start. On the first trial’s own answers the new reader gives the review’s
    figures, and it fails layer 57’s probe on all three.
    - **Layer 39 passes.** Whole: 0.694, 0.738, 0.678 and 0.679, 0.690. The ±1 answers alone: 0.690, 0.728, 0.673
      and 0.680, 0.652. The 930 of 1,022 neurons the controls cover: 0.707, 0.727, 0.679 and 0.682, 0.703. Layer
      45’s own trial read 0.714, 0.724, 0.688 and 0.674, 0.669. Fewer of its neurons stand apart: 28 of the 1,022 beat
      every control (about one by chance; 169 at layer 45 by the same count) and 80 are named. At ±1 its text loses
      0.04 of its distinct 4-grams (the controls’ 0.01), as at 45; at ±2 0.54 to 0.58, where 45 loses 0.79.
    - **Layer 51 passes.** Whole: 0.773, 0.730, 0.707 and 0.709, 0.728. The ±1 answers alone: 0.723, 0.710, 0.679
      and 0.666, 0.662. The 948 covered: 0.766, 0.718, 0.698 and 0.698, 0.721. 238 beat every control and 266 are
      named. At ±1 its text loses 0.01 (the controls’ none); at ±2 0.78 to 0.80. Cut to 32 word-pieces its lead falls
      0.027 and its replication 0.045, so the length rule keeps 48 here, where at 39 it would take 32.
    - **Layer 38 fails, and the band stops there downward.** Whole: 0.497, 0.644, 0.559 and 0.555, 0.494. The ±1
      answers alone: 0.496, 0.646, 0.575 and 0.575, 0.507. The 950 covered: 0.510, 0.636, 0.570 and 0.563, 0.502.
      Its lead is at chance; two of 1,022 beat every control and none is named. (Its trial was cut short at 35,776
      of 49,110 answers by a CUDA launch timeout on the home card, 18:43Z, and continued row for row by the walk;
      every answer is there and the sanity gate passes.) The reason is in the push, not in the neurons. The stream’s
      median length steps up below layer 39 (161 at layer 40, 174 at 39, 213 at 38, 240 at 37), so ±1 of it is 3.1
      times the most a layer-38 neuron can write, where it is 2.1 times at 39 and 1.75 at 51; and at ±1 the text
      already breaks at 38, the neurons losing 0.15 of their distinct 4-grams and the controls 0.10, against 0.04 and
      0.01 at 39. The one gate it clears on every reading is the lead beyond how the text survives (0.644, 0.646,
      0.636): the neurons still differ from random directions under the wreckage. So layer 38 is not shown to have
      nothing to read; it is shown to be pushed too hard by this design. By the rule fixed first, no whole layer is
      taken at 38 or below it with these doses: the board marks 0–38 *not taken*, and they wait for doses scaled to
      the layer (step 3 of the plan; the pilot the steer lens needs, above, measures just that).
    - **Layer 52 fails, as predicted before it was read** (`census_scaled_design/PREDICTION.md`, committed in
      `c9c5a82` while its trial was still being written). Its whole reading passes, and strongly: 0.802, 0.620, 0.598
      and 0.601, 0.777, with 229 of 1,022 above every control. The 819 covered pass too (0.780, 0.625, 0.587 and
      0.604, 0.750). But its ±1 answers alone do not: 0.603, 0.592, then 0.546 and 0.577 on replication (bar 0.57)
      and 0.554 on the new questions (bar 0.56). At layer 52 ±1 of the stream is 1.06 of a neuron’s write ceilings,
      the least of any layer, and those answers barely move (nothing lost of their distinct 4-grams); the lead is all
      in the ±2 answers, which are 2.1 ceilings, where the neurons’ text has lost 0.39 to 0.42 and the controls’ 0.58
      to 0.59. It is layer 57’s failure again, and the mirror of layer 38’s: there the lower dose was three ceilings
      and too hard, here it is one and too soft. By the rule fixed first the band stops at 51 upward under
      `break_rel`, and the board marks 52–59 *not taken*.
    - **What a pass licenses, and what it does not.** By the rules fixed first: a whole layer at 39 and at 51,
      offered to volunteers (kit 2, below). `DOSES_CALIBRATED` was widened to 39–51 the same day, on the owner’s word
      (it had been 40–50). It grows only so, a passed trial and then the owner’s word: a layer that passes later is
      marked “doses not calibrated at this layer”, and inked without gilt, until it is counted in. The gates are layer
      45’s, read at another layer: a pass says that there, as at 45, a neuron’s
      push is told from a random direction’s by the words it brings, in the ±1 answers and among neurons whose text
      survives as the controls’ does. It does not say ±1 and ±2 are the best doses for that layer.
    - **The walk** (the owner’s word, 12 November). After the four, `walk.sh` trials the next layer out on each side
      in turn, reads it, and goes on on that side only if it passed; a layer that passes is opened on the board as
      it is read, one that fails is marked not taken, and the band stops there on that side. It stopped at once on
      both sides, at 38 and at 52: under `break_rel` the census is layers 39–51, thirteen of the sixty. What takes it
      further is a dose counted in ceilings (`break_ceil`, step 3 above): its first trial, layer 38 again, was
      registered in `e853e5c` and read at 22:01Z that evening. It fails by a hair on the whole reading (0.599
      against 0.60) and passes on its lower dose alone (0.682): the gentler push gives layer 38 a lead, and the
      design’s upper dose takes half of it back (step 3).
      With both doses inside what reads (±1.6 and ±2.3 ceilings, `soft_ceil`) layer 38 passes its trial on new
      neurons (13 November, step 3). The walk downward goes on with that design, a trial before each layer;
      layer 38 stays *not taken* on kit 2’s board, since a new design is a new kit.
      The walk downward with it stopped at its first layer: 37 fails on the lower dose alone and, by 0.002, on
      the covered neurons (13 November, step 3). With these two doses the census reaches 38 and no further down.
      With doses set for the layer (±2.41 and ±2.55 ceilings, `match_ceil`) layer 37 passes its trial on new
      neurons the same night (step 3). Layer 36 passes with its own doses the next morning, one gate by a
      thousandth, layer 35 with more room, layers 34 and 33 widely, layer 32 with less; layer 31 fails that
      afternoon and the walk stops there (step 3).
      The doses are counted calibrated at the eight layers that passed since that evening (step 3), and kit 3
      was cut on 14 November 2026 (below).
  - **Kit 2: the other forty-nine, opened a layer at a time** (cut 12 November 2026, tag `census-kit-2`;
    `VOLUNTEERS.md`). The same command as kit 1 at every layer left; what changed is the layers, the tag and the
    addresses, nothing that writes an answer. The board gained two states, *trial first* and *not taken*, and only
    an open layer takes a claim; `census.sh reserve` asks the published board before it makes one. The home canaries
    of all 49 layers were taken at the kit’s commit `95deead` in seven minutes on the 5090, between two trials.
    Layer 45’s, taken again there, is kit 1’s canary bit for bit, so the code still writes kit 1’s answers; and the
    eleven old canaries against their whole runs are the same at the six layers of the home card’s bits (40, 41, 43,
    45, 46, 49) and not at the five taken on RTX PRO 6000 cards, as their class says. Opened that day: 39 and 51.
  - **Kit 3: a design for each layer** (cut 14 November 2026, tag `census-kit-3`; `VOLUNTEERS.md`). Kits 1 and 2 took every layer with one design. Kit 3’s file names one a layer (`volunteers/kit.env`,
    `KIT_DESIGNS`): `match_ceil` at 32–37, `soft_ceil` at 38 and 52, and `break_rel` still at 39 and 51, which
    stay open through the change and are taken by the command kit 2 took them with. Layers 53–59 are named with
    `soft_ceil` so that one whose trial passes later can be opened without a new kit; none has been trialled with
    it, and all stay shut on the board. The script picks the layer’s design, so a volunteer chooses nothing. The
    intake sets a bundle against the canary of its layer’s design, and the home canaries are taken a design at a
    time (`census_intake.py home-canaries`). The six unpushed answers a volunteer’s are set against are one set
    for the whole kit: on the home card they are the same bit for bit under all three designs (read from the
    trials’ own files at layers 32, 34, 37, 38, 39 and 52, against layer 45’s). One fault was found in preparing
    it: the canary step’s second half, the writer’s floor, refused `match_ceil` by name, so no volunteer could
    have taken a canary at 32–37; mended. `check_volunteer_kit` now runs a kit of two designs end to end on the
    stand-in, and checks that a kit of one design is still read as kits 1 and 2 were (61 checks, 53 before).
    *The cut* (14 November 2026; the owner’s word, “Yes, cut it”, with layer 36 held for the home card, since it
    passed one gate by a thousandth). It was prepared on a branch, because a volunteer’s `reserve` reads the board
    from main and kit 3’s board there would have turned kit 2’s volunteers away from layers 39 and 51; no one had
    claimed either when the kit was cut. The home canaries were taken on the home card at the kit’s merged commit,
    00:27Z to 00:37Z: fifteen new ones (`match_ceil` at 32–37, `soft_ceil` at 38 and 52–59), each written twice
    and the same both times; layers 39 and 51 taken again and the same as kit 2’s; layer 45 taken again and the
    same as kit 1’s. A second fault was found in the taking: the command stopped at layers 39 and 51, whose
    canaries kit 2 left in the folder and the writer will not write over. It now takes a kept canary again beside
    the one kept and compares them, and fails if they differ (`check_volunteer_kit`, 63 checks). Nine layers are
    open on the board: 32–35, 37–39, 51 and 52. Layer 36 is closed, held for the home card;
    `census_board.py layer 36 open` offers it. PR #2 (24 GB cards) is not in this kit. *Not done:* no whole layer
    had been taken with `match_ceil` or `soft_ceil` to its end, on any card, when this kit was cut (the home card
    has since taken layer 36, and a volunteer 34 and 35, below); and whether the intake carries a volunteer card’s own writer floor through to the page is
    not checked.
    *The reader for a whole layer taken with them* (written 14 November 2026):
    `experiments/results/census_scaled_design/read_whole.sh <layer>` reads a whole `match_ceil` or `soft_ceil`
    layer with the reader and gates that layer’s own trial was read by, then the upper dose’s answers alone,
    gating nothing, into `census_L<NN>_whole/` beside the other layers’. The registered reader’s three files are
    the same bytes in every trial’s folder, and the folder’s hashes are checked before anything is read. Run on the
    trials themselves at layers 36, 37, 38 and 52 it gives the verdicts recorded at the time, figure for figure,
    the upper dose’s among them; it refuses a `break_rel` run (whose reader is
    `census_neighbour_trials/read_whole.sh`), a run of another layer than the one named, and a layer with no trial
    it knows of unless that trial’s folder is given. Layer 36 was the first whole layer it read (below).
  - **Kit 4: kit 3 opened to 24 GB cards** (cut 14 November 2026, tag `census-kit-4`; the owner’s word that day:
    “Definitely need to incorporate the 24gb PR if it will work”). The pull request (whp199/theodolite#2, by
    bethanylong and her agent) makes two changes. `census.sh check` asks for 23,900 MiB free as well as a card of
    24,000 MiB. And the loader puts the vision tower on the CPU, where it is dropped, so it no longer sets the load’s
    peak on the card. It works here. On the home card with all but 23,900 MiB held aside, the six worst batches a
    layer can write (every row to 48 word-pieces) were written with 217 MiB to spare; with all but 23,600 they ran
    out of memory at the second; with as much free as a headless RTX 3090 has (24,124 MiB) the least to spare was
    721 MiB (`scripts/measure_batches.py --hold-mib`, 14 November; the author measured 768 MiB to spare on a 3090
    itself). The answers are the same: all seventeen of kit 3’s home canaries, taken again with the new loader, are
    the ones kept, and layer 45’s is kit 1’s. It is a new kit because the kit’s file changed. Its layers, designs,
    board and canaries are kit 3’s, which was live for about an hour and had no claim. A 24 GB layer is a card class
    of its own, read against its own controls and spot-checked as the RTX PRO 6000 layers were; a 3090 takes about
    43 hours a layer, by the author’s sixteenth of layer 45. *Not shown yet:* a whole layer from a 24 GB card
    through the intake, and a 4090. (Layer 51 from an RTX 3090 came through on 16 November, below; no 4090 has
    taken a layer yet.)
    *Layer 36 on the home card* (the owner’s word, 14 November: “we can offer layer 36 using your best judgement”;
    the judgement was that the home card take it, since the card was free, the layer passed one gate by a
    thousandth and a whole `match_ceil` layer is what the new reader most needs). It began at 01:07Z on 14 November
    with kit 4’s code and settings, about 13 to 17 hours. For its first twelve minutes a second process held the
    card down to 23,900 MiB free, the kit’s floor: a real run inside what a 24 GB card has. The board shows the
    layer closed, the home card taking it.
    *Layer 36 read whole (14 November, 17:33Z): the whole reading passes, and the lower dose alone fails one gate
    by 0.006, so by its trial’s standard the layer fails.* The run ended at 17:14Z with all 565,254 answers, 15.5
    hours of writing in two sessions (it was stopped for 34 minutes for the pattern tests), from kit 4’s commit,
    its first batch the home canary bit for bit. Read by its trial’s reader and gates
    ([`census_L36_whole/`](experiments/results/census_L36_whole/)); the figures are the lead, the lead beyond how
    the text survives, the two directions across the question halves, and the new questions alone, against 0.60,
    0.58, 0.57 and 0.57, 0.56:

    | | whole layer, 21,504 neurons | the trial, 1,022 neurons |
    |---|---|---|
    | every answer | 0.632, 0.650, 0.589 and 0.597, 0.617: passes | 0.627, 0.636, 0.608 and 0.606, 0.612 |
    | the ±2.27 answers alone | 0.616, 0.630, 0.577 and **0.564**, 0.611: fails | 0.615, 0.622, 0.586 and 0.571, 0.600 |
    | the neurons the controls cover | 97 in 100; 0.629, 0.650, 0.584 and 0.592, 0.614: passes | 96 in 100; 0.622, 0.632, 0.600 and 0.600, 0.609 |
    | the ±2.46 answers alone (reported) | 0.610, 0.632, 0.577 and 0.577, 0.600 | 0.601, 0.619, 0.574 and 0.584, 0.595 |

    - **What fails is one number.** The lower dose’s words, picked on the second half of the questions and scored
      on the first, lead by 0.564 where 0.57 is asked. The trial had it at 0.5709 and its write-up said another
      draw of neurons could fail as easily as pass; the whole layer is that other draw. Each dose alone sits at
      the bar on this check (0.564 to 0.577) and the two together clear it (0.589, 0.597). The doses are 8% apart,
      so the two are close to one dose read twice.
    - **How sure the 0.564 is, is not worked out.** It is 0.006 under the bar with 2,048 controls behind it. I
      have not computed its error; my guess is that 0.006 is about what the controls alone leave open, so “at the
      bar” is the honest reading, not “clearly under”.
    - **Per neuron it tells little, as the trial said it would.** 18 neurons beat every one of the 2,048 controls
      where chance gives 10.5, and none passes q ≤ 0.05. 23 in 100 beat 95 in 100 of the controls, and 1,877 are
      under p 0.05 on both halves of the questions where chance gives 54: the layer as a whole carries a lead,
      and few single neurons can be told from a control.
    - **Cut to 32 word-pieces the whole reading still passes** (0.619, 0.625, 0.584 and 0.587, 0.598), and by the
      length rule it would take 32; every figure here is at 48.
    - **The steer lens against it is weak, as at every layer checked below 39 (32 to 38)**
      (`scripts/atlas_steer_check.py`, run again with the whole layer in): of the 4,751 neurons above 95 in 100 controls, a steer-lens word-piece is one of
      the census’s words for 0.185, and for 0.034 with the neurons shuffled. Its neighbours’ trials read 0.127
      (35) and 0.197 (37).
    - **What this does not decide.** `match_ceil` is counted as calibrated at layers 32 to 37 on the trials
      (`DOSES_CALIBRATED`, the owner’s word of 13 November). Whether layer 36 stays counted, and what the board
      and the page say of it, is the owner’s choice; nothing was changed. Holding it back from volunteers was
      right. For the layers volunteers have, the trials’ lower dose carried across the halves by 0.589 and 0.602
      (32), 0.601 and 0.617 (33), 0.640 and 0.650 (34), 0.582 and 0.595 (35), 0.599 and 0.600 (37). Layer 36’s
      whole reading came out 0.009 and 0.007 under its trial on this check; one layer is not a rule, but if the
      others fall as far, 35 is the one that would come close.

    *The bars for another card class, measured for each of this kit’s designs* (14 November 2026; the owner’s
    word, “Bars first”; [`census_other_card_bars/kit4/`](experiments/results/census_other_card_bars/kit4/)). A run
    from a card that does not write the home card’s bits is taken if a few of its chunks, written again at home,
    stay within set distances of the home card’s. The two log-probability distances had been measured once, on 31
    October, at layers 40 to 50 under `break_rel`. Kit 4 takes its layers with three designs, two of which push by
    other lengths, and nothing had been measured for them; layer 34’s bundle, from a 96 GB card, was hours away.
    - **The measurement** (20 minutes on the home card, 17:16Z to 17:37Z, after layer 36): each of layers 32 to 39,
      51 and 52, with the kit’s design, its first 64 answers written thirteen ways: as the kit writes them, with
      only the arithmetic changed (three batch sizes, full-precision sums), and with the push 1, 2, 5 and 10%
      longer and shorter. The control was the home canaries bit for bit at all ten layers.
    - **The rule and what it would decide were fixed first** (`prereg/RULES.md`, hashed and committed at 16:50Z): the
      rule of 31 October unchanged, applied to each design’s layers together.

      | design, layers | arithmetic alone moves the median, the 99th percentile | a push 5% off, 10% off | the bars |
      |---|---|---|---|
      | `match_ceil`, 32–37 | 0.0050, 0.117 | 0.028, 0.53 | **0.012, 0.25** |
      | `soft_ceil`, 38 and 52 | 0.0065, 0.130 | 0.026, 0.63 | **0.013, 0.29** |
      | `break_rel`, 39 and 51 | 0.0143, 0.203 | 0.053, 0.81 | 0.028, 0.37 (those of 31 October, which stand) |
      | `break_rel`, 40–50 (31 October) | 0.0130, 0.169 | 0.059, 0.81 | 0.028, 0.37 |

    - **Under the two ceiling designs the home card’s own arithmetic moves a log-probability by less than half
      what it does under `break_rel`**, so their bars are tighter. The old bars would have been too loose there: a
      push 5% off moves the median to 0.028, which is the old bar itself, and one 2% off to 0.013, well inside it.
    - **At 39 and 51 the old bars stand, by the check fixed beforehand**: arithmetic alone stays under each and the
      push error each was built to catch stays over it. By the rule alone the median there would have had no bar
      (a push 5% off is 3.7 times the floor, and 4 is asked).
    - **The first word-piece bar stays 0.90**, and under `match_ceil` it catches little: a push 10% off still writes
      the same first word-piece in 98 answers of 100 or more. There the log-probability bars are the check that bites. At
      39 and 51, arithmetic alone parted 4 first word-pieces of 64 at one batch size (0.9375, where 0.93 was the
      mark for keeping the bar); on 31 October the least kept was 0.983.
    - **No layer needed bars of its own**: each layer’s own arithmetic is under its group’s bars.
    - **A fault found first, and a guard added.** The script made a push “a little off” by scaling the stream’s
      median length, which a ceiling design’s push does not use: every such run would have written the unchanged
      answers and both bars would have come out dropped, for the wrong reason. It now scales the write ceiling for
      those runs, and stops if any of them writes the control’s rows. And the intake now draws no spot chunks,
      and will not judge, a run of another card class at a design and layer no bars were measured for
      (`census_volunteer.accept_bars`): layers 53 to 59 are in that state.
    - **Two false starts.** The script refused twice at 17:14Z, writing nothing: it asked 26 GiB of the card free
      and the desktop held 5.1 GB. It now asks the kit’s own 23,900 MiB, and the registration is amended, dated,
      saying so.
    - **What this leaves open.** The variants are the home card’s own arithmetic; that another card class stays
      inside them rests on the five `break_rel` layers accepted so far (medians 0.013 to 0.015 against a floor of
      0.013), not on anything measured under a ceiling design. A run with one dose right and the other wrong is
      not modelled. And the bars at 32 to 37 were tightened after layer 34’s volunteer had begun under a guide
      that said 0.028 and 0.37; none of their answers had been seen, and the guide now says which bars hold
      where. `check_volunteer_kit` 74 checks then, 75 with the table held to the code.

    *Five more whole layers: four from volunteers, and layer 36 counted done* (15 and 16 November 2026; the owner
    approved the four downloads and, on 16 November, said how layer 36 is counted). The four bundles went through the
    intake on the home card on 16 November, 01:54Z to 02:39Z, each bundle’s sha256 the one its volunteer posted, and
    each layer was read whole by its trial’s registered reader by 03:29Z
    ([`census_L34_whole/`](experiments/results/census_L34_whole/), `census_L35_whole/`, `census_L39_whole/`,
    `census_L51_whole/`; a `read.sh` in each says how).
    - **What came home.** All four were taken with kit 4 at its tag (`81c13628`), from a clean tree.
      - Layer 34, `match_ceil`, by anon (claim eaef4eff), the volunteer who took 42, 48 and 50, on an RTX PRO 6000
        Blackwell Server Edition, a card class of its own. Begun 14 November at 05:26Z by the run’s record; DONE
        posted 15 November at 01:36Z. Its bars were tightened after it began, without sight of it (above); it is
        inside both the old and the new.
      - Layer 35, `match_ceil`, by the same volunteer (5d30aa4a) on the same card class: RESERVE and START at 01:47Z
        on 15 November, DONE at 20:31Z.
      - Layer 39, `break_rel`, by anon (1f217635), new to the census, on an RTX 5090 that writes the home card’s
        bits. Its RESERVE, START and DONE came in one comment at 18:50Z on 15 November, and by its record the run
        began at 05:42Z that day, so the board never held the layer for it; nothing else had claimed it. The rules
        ask for the claim first; nothing in the intake turns on it.
      - Layer 51, `break_rel`, by eigentactics (6d708804), who wrote this kit’s change for 24 GB cards, on an RTX
        3090 24 GB, a card class of its own: **the first layer taken on a 24 GB card.** DONE 15 November at 19:14Z.
        By the run’s record it began 14 November at 03:28Z, and the bundle keeps its answers file as last written at
        18:07Z on 15 November: about 39 hours (inferred from a file time set on the volunteer’s machine; the guide
        says about 43).
    - **The intake.** Every check passed: the claim’s secret, every setting and the commit, all 565,254 answers once,
      the canary. The spot chunks, 5 of 737 for each layer drawn when its bundle was checked, against the bars of the
      layer’s design:

      | layer | its bits | first word-piece the same (bar) | log-probabilities apart: median (bar) | 99th percentile (bar) |
      |---|---|---|---|---|
      | 34 | a class of its own | 0.990 (0.90) | 0.0051 (0.012) | 0.115 (0.25) |
      | 35 | a class of its own | 0.989 (0.90) | 0.0046 (0.012) | 0.110 (0.25) |
      | 39 | the home card’s | all 3,840 answers the same bit for bit | | |
      | 51 | a class of its own | 0.989 (0.90) | 0.0111 (0.028) | 0.150 (0.37) |

      The unpushed answers of 34 and 35 part from the home card’s at the three near-ties the RTX PRO 6000 layers
      parted at before (pieces 5, 21 and 27, which the home card writes at p = 0.50, 0.41 and 0.50), the 3090’s at
      two of them (5 and 27), and 39’s at none (bar 0.6). Under `match_ceil` the two medians, 0.0051 and 0.0046, sit
      at what the home card’s own arithmetic gave when the bars were set (0.0050, and 0.117 at the 99th percentile):
      what that entry expected of another card class, now seen on one.
    - **What they read**, by the figures and bars of layer 36’s table above (the lead, the lead beyond how the text
      survives, the two directions across the question halves, the new questions alone; 0.60, 0.58, 0.57 and 0.57,
      0.56), at 48 word-pieces, each with its trial beside it and layer 36 again for comparison:

      | layer, design, doses | every answer | the lower dose alone | the neurons the controls cover | its trial, every answer |
      |---|---|---|---|---|
      | 34, `match_ceil`, ±2.62 and ±2.79 ceilings | 0.761, 0.752, 0.693 and 0.698, 0.721 | 0.733, 0.723, 0.639 and 0.644, 0.679 | 98 in 100; 0.762, 0.750, 0.693 and 0.698, 0.722 | 0.758, 0.747, 0.683 and 0.692, 0.709 |
      | 35, `match_ceil`, ±2.27 and ±2.41 ceilings | 0.675, 0.650, 0.633 and 0.631, 0.650 | 0.670, 0.653, 0.605 and 0.606, 0.635 | 96 in 100; 0.672, 0.648, 0.627 and 0.625, 0.646 | 0.664, 0.643, 0.639 and 0.623, 0.657 |
      | 39, `break_rel`, ±1 and ±2 of the stream | 0.687, 0.733, 0.679 and 0.660, 0.684 | 0.689, 0.750, 0.675 and 0.660, 0.666 | 95 in 100; 0.696, 0.727, 0.680 and 0.662, 0.693 | 0.694, 0.738, 0.678 and 0.679, 0.690 |
      | 51, `break_rel`, ±1 and ±2 of the stream | 0.763, 0.734, 0.698 and 0.697, 0.713 | 0.714, 0.679, 0.657 and 0.649, 0.655 | 95 in 100; 0.756, 0.723, 0.688 and 0.688, 0.707 | 0.773, 0.730, 0.707 and 0.709, 0.728 |
      | 36, `match_ceil`, ±2.27 and ±2.46 ceilings, the home card | 0.632, 0.650, 0.589 and 0.597, 0.617 | 0.616, 0.630, 0.577 and **0.564**, 0.611 | 97 in 100; 0.629, 0.650, 0.584 and 0.592, 0.614 | 0.627, 0.636, 0.608 and 0.606, 0.612 |

      Reported beside them, gating nothing: the upper dose alone reads 0.733, 0.733, 0.654 and 0.647, 0.675 at 34,
      and 0.658, 0.646, 0.608 and 0.600, 0.620 at 35. By the length rule 34 and 51 keep 48 word-pieces and 39
      would take 32, as their trials did; 35 keeps 48 where its trial would have taken 32, and 36 would take 32
      where its trial kept 48 (above).
    - **Where they differ from their trials.** Each passes every gate its trial passed. The least margin over a bar,
      whole layer against trial: 0.069 against 0.070 at 34, 0.035 against 0.012 at 35, 0.087 against 0.090 at 39,
      0.079 against 0.096 at 51 (and at 36, −0.006 against 0.001).
      - Layer 34 reads at its trial or a little over on every answer (lead 0.761, against 0.758). Its lower dose
        alone leads by a little less (0.733, against 0.745) and carries across the halves as its trial did (0.639 and
        0.644, against 0.640 and 0.650).
      - Layer 35: the check layer 36 missed, the lower dose’s words carried across the question halves, reads 0.605
        and 0.606, over its trial’s 0.582 and 0.595. Layer 36’s entry said that if the others fell as far, 35 would
        come close; it rose. That check is still layer 35’s thinnest.
      - Layer 39 reads as its trial did but in one direction across the halves: 0.660, against 0.679 for every
        answer and 0.680 for the ±1 answers alone.
      - Layer 51 reads under its trial on 13 of the 15 figures, by 0.031 at most (the ±1 answers beyond how the
        text survives, 0.679 against 0.710). Its trial was taken on the home card and the whole layer on a 3090,
        each read against controls pushed on its own card.
      - The text: at 34 and 35 it hardly moves at either dose (the neurons’ answers lose 0.006 to 0.018 of their
        distinct 4-grams, the controls’ 0.004 to 0.011); at 39 and 51 it loses what the trials lost (at ±1, 0.04 and
        0.01 for the neurons, 0.01 and none for the controls; at ±2, about 0.55 and 0.78, neurons and controls
        alike).
    - **Per neuron.** The first three counts are those earlier entries give, from each reading at 48 word-pieces.
      The last two were computed for this entry from each run’s units file, by the step added at the end of each new
      `read.sh` (gating nothing); it gives layers 48–50’s 1,408, 1,407 and 1,353 again, and 2,007 to 4,742 at 40–47.

      | layer | above all 2,048 controls (10.5 by chance) | q ≤ 0.05 | above 95 in 100 controls | more than five control SDs over the controls’ median | the strongest control and the fifth, in those SDs |
      |---|---|---|---|---|---|
      | 34 | 110 | 0 | 43 in 100 | 2,445 | 11.5 and 8.3 |
      | 35 | 35 | 0 | 27 in 100 | 928 | 10.4 and 7.2 |
      | 36 | 18 | 0 | 23 in 100 | 930 | 13.3 and 8.9 |
      | 39 | 87 | 436 | 25 in 100 | 870 | 7.8 and 5.2 |
      | 51 | 2,123 | 7,145 | 43 in 100 | 2,485 | 5.3 and 3.6 |
      | 40–50 | 95 to 4,794 | 3,534 to 9,133 | 32 to 49 in 100 | 1,353 to 4,742 | 4.9 to 10.8, and 3.4 to 5.3 |

      - **At 34–36 no neuron has q ≤ 0.05, and the count hangs on a few controls that stand far out.** The count
        rests on how far the strongest controls stand out, as the count above every control rests on the one
        strongest (layers 40–50, above). Under `match_ceil` the strongest controls bring the words of another
        language with the text unbroken (read from their words: Portuguese, Spanish, French and German at 34;
        Indonesian, French, German and Portuguese at 35; Indonesian, Spanish, Portuguese, Vietnamese and French at
        36), as the strongest does at 39 and did at 41; what is new is how many stand out: even the fifth
        strongest stands 7.2 to 8.9 of the controls’ SDs above their median, where at every `break_rel` layer read
        whole it stands 3.4 to 5.3. At 34, with the five strongest controls set aside, 6,236 neurons would have q ≤
        0.05 (339 at 35; at 36 none until six are set aside, 241, and 945 with ten). That measures how much the
        count hangs on a few controls; it is not a reading, and the gates count every control. At 35 and 36 the neurons stand out less
        too: 928 and 930 lie more than five control SDs over the median and 27 and 23 in 100 beat 95 in 100 of the
        controls, fewer than at any of 40–50 (1,353 and 32 in 100 at the least). By the other counts layer 34 stands
        with layer 51: 43 in 100 of its neurons beat 95 in 100 of the controls.
      - **Layer 39 tells the fewest single neurons of any `break_rel` layer:** 436 with q ≤ 0.05 (3,534 to 9,133 at
        40–50, 7,145 at 51) and 87 above every control. Its trial had 28 of 1,022 above every control, and said then
        that fewer of its neurons stand apart.
      - **Layer 51 reads per neuron as 40–50 do**, from a 3090. Its strongest controls loop on glued “thought” words,
        as those of 48–50 do.
    - **Layer 36, as the owner decided** (16 November 2026, asked and answered): the board marks it done by the home
      card, with a note naming the miss, and the README counts it among the done layers. `census.DOSES_CALIBRATED`
      is unchanged (`match_ceil` 32–37, `soft_ceil` 38 and 52, `break_rel` 39–51): whether the miss changes what is
      counted calibrated at 36 is the owner’s to decide, and was not part of what was asked.
    - **The board** (16 November): sixteen layers done, 34–36 and 39–51, ten of them by volunteers; layer 37
      running, reserved and started at 22:13Z on 15 November (anon 042ec8f2, the volunteer who took 39, on an RTX
      5090 with the home card’s bits); 32, 33, 38 and 52 open; 0–31 and 53–59 not taken. Before the four were
      judged, the board’s sync was mended (`abd12ec`): it had lapsed layer 34’s claim by judging its DONE at the
      sync’s time.
    - **What a volunteer reported, not checked here.** eigentactics wrote on the board issue that they also took
      layer 51 whole on an AMD RX 7900 XTX under ROCm; that both runs pass the three checks by the kit’s
      `neighbour_read.py`; that of 7,874 neurons with q < 0.05 on either card, 16 in 100 are so on one card only;
      and that a layer split between the two cards passes too. They offer a pull request making 24 GB ROCm cards a
      card class. None of it has been read here, and any answer to them is the owner’s.
    - **The steer lens, checked again with the four in** (`scripts/atlas_steer_check.py`, 16 November, 03:58Z): of
      the neurons above 95 in 100 controls, a steer-lens word-piece is one of the census’s words for 0.157 at 34
      (9,087 neurons; 0.023 with the neurons shuffled), 0.154 at 35 (5,577; 0.030), 0.634 at 39 (5,260; 0.060) and
      0.850 at 51 (9,285; 0.012): weak at 34 and 35 and agrees at 39 and 51, as before. **Not done:** the page was
      not looked at with them.
    - **The walk upward with `soft_ceil` stopped at layer 53** (the owner’s words that day: “continue the census walk
      in hopes of opening more layers for our volunteers”; registered and committed before any answer, f935df9,
      pushed at 05:59Z, six minutes into the run and before any answer was read; run 05:52Z–07:07Z, 16 November;
      `experiments/results/census_soft_walk_up/RESULT.md`). Layer 53 failed all three
      gates. The whole reading missed on words carrying between the question halves (0.557 against 0.57), and its lead
      was 0.694, under the 0.72 to 0.82 said beforehand. The lower dose alone failed every figure (lead 0.551). The
      upper dose alone led by 0.689, but its words read at chance across the halves. Both doses moved the next
      word-piece inside the span where every earlier dose but one had led (0.37 and 1.53 nats), so by the
      registration that span is no guide above layer 52. Layers 54–59 were not trialled. The board marks 53 failed.
      What to try above 52, if anything, is the owner’s choice.
    *Two more whole layers from volunteers, 37 and 33* (17 November 2026; the owner approved each download: “Yes, take
    in 37” and “you can take it in”). Both were taken with kit 4 at its tag (`81c13628`), from a clean tree, and came
    home after kit 5 was prepared and before it was cut, so each was checked by kit 4’s own file, read from its tag
    (`census_volunteer.earlier_kit`). Each was read whole by its trial’s registered reader
    ([`census_L37_whole/`](experiments/results/census_L37_whole/), `census_L33_whole/`; a `read.sh` in each says how).
    - **What came home.**
      - Layer 37, `match_ceil`, by anon (042ec8f2), the volunteer who took 39, on an RTX 5090 with the home card’s
        bits. Begun 15 November at 22:14Z by the run’s record; DONE 16 November at 21:49Z. Its spot chunks (54, 183,
        271, 461 and 675 of 737) are the home card’s bit for bit, all 3,840 answers.
      - Layer 33, `match_ceil`, by anon (280c6dc2), the volunteer who took 34 and 35, on the same RTX PRO 6000 card
        class. Begun 16 November at 13:03Z, in three sessions; DONE 17 November at 05:50Z. Its spot chunks (140,
        249, 316, 447 and 657): the first word-piece the home card’s in 0.984 of answers (bar 0.90), log-probabilities
        apart by a median 0.0055 (0.012) and 0.116 at the 99th percentile (0.25), as 34’s and 35’s were.
    - **What they read**, by the figures and bars of the table above, at 48 word-pieces:

      | layer, design, doses | every answer | the lower dose alone | the neurons the controls cover | its trial, every answer |
      |---|---|---|---|---|
      | 37, `match_ceil`, ±2.41 and ±2.55 ceilings | 0.662, 0.700, 0.626 and 0.630, 0.651 | 0.650, 0.667, 0.602 and 0.607, 0.636 | 89 in 100; 0.646, 0.687, 0.607 and 0.609, 0.636 | 0.653, 0.678, 0.609 and 0.627, 0.654 |
      | 33, `match_ceil`, ±2.31 and ±2.51 ceilings | 0.745, 0.742, 0.653 and 0.662, 0.709 | 0.716, 0.707, 0.623 and 0.622, 0.668 | 97 in 100; 0.744, 0.738, 0.651 and 0.660, 0.707 | 0.726, 0.726, 0.630 and 0.637, 0.691 |

      Each passes every gate, with more room than its trial: the least margin over a bar is 0.032 against 0.015 at
      37, and 0.052 against 0.031 at 33, both on the lower dose’s words carried across the question halves (at 37’s
      trial, on the covered neurons’). The message of `71c7104` says layer 37’s trial passed one gate by 0.001; that
      was layer 36’s trial. Reported beside them, gating nothing: the upper dose alone reads 0.631, 0.665, 0.611 and
      0.597, 0.622 at 37, and 0.712, 0.706, 0.614 and 0.612, 0.659 at 33. By the length rule 33 keeps 48
      word-pieces, as its trial did, and 37 would take 32, as its trial would have. The text hardly moves: the
      neurons’ answers lose 0.022 to 0.030 of their distinct 4-grams at 37 and 0.003 to 0.004 at 33, the controls’
      0.005 to 0.009 and 0.002 to 0.004.
    - **Per neuron they tell more than 34–36**, by the counts of the table above:

      | layer | above all 2,048 controls (10.5 by chance) | q ≤ 0.05 | above 95 in 100 controls | more than five control SDs over the controls’ median | the strongest control and the fifth, in those SDs |
      |---|---|---|---|---|---|
      | 37 | 454 | 2,332 | 31 in 100 | 2,376 | 10.5 and 9.5 |
      | 33 | 1,041 | 5,576 | 40 in 100 | 2,891 | 7.9 and 6.0 |

      Their strongest controls bring another language’s words, as at 34–36 (French, Spanish, Portuguese and German
      at 37; Portuguese and Indonesian at 33), and at 37 they stand as far out as there or further (the fifth at 9.5
      SDs, against 7.2 to 8.9). Yet at 37 2,332 neurons have
      q ≤ 0.05, where 34–36 had none: more of its neurons stand further out still. With the five strongest controls
      set aside the count would be 4,017 at 37 and 6,197 at 33 (gating nothing).
    - **The board** (17 November, before the cut): eighteen layers done, 33–37 and 39–51, twelve of them by
      volunteers; 52 running (eigentactics, c40b4d3a, under kit 4); 32 and 38 open.
  - **Kit 5: kit 4 with layer 31 added** (cut 17 November 2026, tag `census-kit-5`; the owner’s words: “Prepare kit 5
    for layer 31”, 16 November, and “Cut kit 5”, 17 November). Layer 31 is taken with `match_wide`, `match_ceil`’s
    lower count alone (±2.58 ceilings) on twelve questions, whose trial passed there on 16 November (above); it is
    counted calibrated there alone, at 48 word-pieces (`a0e950e`). Every other layer is taken as under kit 4.
    - **The answers.** All seventeen of kit 4’s home canaries, taken again under kit 5, are the ones kept; layer
      31’s is new (`eebe4a6`). Nothing that writes an answer changed after that.
    - **Its own unpushed answers.** `match_wide` asks twelve questions, and a whole layer writes the second half’s
      unpushed answers in batches of 63, where chair’s log-probabilities differ from layer 45’s. So a design that
      asks other questions is checked against its own unpushed answers, from its trial on the home card
      (`census_volunteer.DESIGN_BASELINES`).
    - **Its bars for another card class**, registered and pushed before anything was measured
      ([`census_other_card_bars/kit5/prereg/RULES.md`](experiments/results/census_other_card_bars/kit5/prereg/RULES.md),
      `e6fa0cf`, pushed 02:37:48Z; measured 02:37Z–03:36Z, `22e73db`). The control’s twelve unpushed answers are
      the trial’s bit for bit. The near-tie bar stands at 0.6 (the home card’s own arithmetic parted only at pieces
      written at 0.52 or less). On the kit’s own chunk 0, 768 answers to all twelve questions, the home card’s
      arithmetic moves the median to 0.0053 and the 99th percentile to 0.115, as under `match_ceil`; the bars are
      0.013 and 0.26, and 0.90 of first word-pieces.
    - **A layer begun under kit 4** is finished on kit 4 and checked by kit 4’s own file, read from its tag
      (`census_volunteer.earlier_kit`), as 37 and 33 were; at the cut that is layer 52.
    - **The board** is kit 5’s with layer 31 open: 31, 32 and 38 open, 52 running. *Not shown yet:* a whole layer of
      `match_wide` from a volunteer. The thread post is the owner’s.
  - **Layer 52 read whole, the last layer begun under kit 4** (17 November 2026; the owner approved the download:
    “Yes, take in 52”). Taken with `soft_ceil` by eigentactics (c40b4d3a), the volunteer who took 51, on the same
    RTX 3090 24 GB card class, with kit 4 at its tag (`81c13628`), from a clean tree. Begun 16 November at 05:00Z by
    the run’s record; DONE 17 November at 17:44Z, on Hugging Face; the bundle’s sha256 matched its DONE line. Kit 5
    was cut while it ran, so it was checked by kit 4’s own file, read from its tag (`census_volunteer.earlier_kit`).
    Its spot chunks (291, 379, 642, 716 and 719 of 737): the first word-piece the home card’s in 0.995 of answers
    (bar 0.90), log-probabilities apart by a median 0.0077 (0.013) and 0.132 at the 99th percentile (0.29). Read
    whole by its trial’s registered reader ([`census_L52_whole/`](experiments/results/census_L52_whole/); the reader
    is `census/trials/soft/prereg_L52`, hashed 13 November at 02:50Z), by the figures and bars of the tables above:

    | layer, design, doses | every answer | the lower dose alone | the neurons the controls cover | its trial, every answer |
    |---|---|---|---|---|
    | 52, `soft_ceil`, ±1.6 and ±2.3 ceilings | 0.804, 0.703, 0.669 and 0.669, 0.765 | 0.628, 0.650, 0.598 and 0.617, 0.566 | 97 in 100; 0.805, 0.698, 0.665 and 0.665, 0.766 | 0.804, 0.693, 0.680 and 0.669, 0.758 |

    - It passes every gate. The least margin is 0.006, the lower dose’s lead on the new questions alone (0.566
      against 0.56); its trial’s was 0.017 on the same figure (0.577). The whole lead is the trial’s to the third
      place. Reported beside it, gating nothing: the upper dose alone reads 0.789, 0.683, 0.656 and 0.709, 0.692. By
      the length rule it would take 32 word-pieces, as its trial would have.
    - The text: the upper dose (±2.3) takes away 0.60 to 0.61 of the neurons’ distinct 4-grams and 0.72 of the
      controls’, as in the trial; the lower dose 0.024 to 0.027 and 0.010.
    - Per neuron it reads as 41–51 do: 3,944 above all 2,048 controls (10.5 by chance), 6,933 with q ≤ 0.05 (7,192
      with the five strongest controls set aside), 42 in 100 above 95 in 100 controls, 3,469 more than five control
      SDs over the controls’ median. Its strongest control stands 4.5 of those SDs out and the fifth 2.8, against
      6.0 to 9.5 for the fifth at 33–37; they loop on “thought”-glued words, as layer 50’s did.
    - The steer check, run again with it: layer 52 still agrees, now against the whole layer (a word shared with
      0.735 of the 9,077 neurons above 95 in 100 controls; 0.013 shuffled).
    - **The board**: nineteen layers done, 33–37 and 39–52, thirteen of them by volunteers; 32 running (anon,
      943783a3, kit 5); 31 and 38 open. Every layer begun under kit 4 is home.
  - **Layer 32 read whole, the first layer taken under kit 5** (18 November 2026; the owner approved the download:
    “Yes, take in 32”). Taken with `match_ceil` by anon (Asynonymous, 943783a3), who took 33 to 35, on the same RTX
    PRO 6000 Blackwell card class, with kit 5 at its tag (`3943703`), from a clean tree. Begun 17 November at 11:11Z by
    the run’s record; DONE 18 November at 03:35Z, as a GitHub release; the bundle’s sha256 matched its DONE line. Its
    spot chunks (91, 193, 432, 493 and 654 of 737): the first word-piece the home card’s in 0.986 of answers (bar
    0.90), log-probabilities apart by a median 0.0052 (0.012) and 0.117 at the 99th percentile (0.25). Read whole by
    its trial’s registered reader ([`census_L32_whole/`](experiments/results/census_L32_whole/); the reader is
    `census/trials/match_walk/prereg`, hashed 13 November at 07:25Z):

    | layer, design, doses | every answer | the lower dose alone | the neurons the controls cover | its trial, every answer |
    |---|---|---|---|---|
    | 32, `match_ceil`, ±2.21 and ±2.44 ceilings | 0.714, 0.701, 0.626 and 0.626, 0.681 | 0.689, 0.667, 0.600 and 0.588, 0.635 | 95 in 100; 0.708, 0.692, 0.620 and 0.619, 0.674 | 0.699, 0.678, 0.615 and 0.631, 0.675 |

    - It passes every gate. The least margin is 0.018, the lower dose’s words carrying from half B to half A (0.588).
      Reported beside it, gating nothing: the upper dose alone reads 0.683, 0.665, 0.596 and 0.586, 0.634. Cut to 32
      word-pieces it still passes, but the lead falls by 0.028, so the length rule keeps 48.
    - The text: neurons and controls alike lose 0.002 to 0.003 of their distinct 4-grams; 91 to 93 in 100 answers run
      to the 48-piece cut.
    - Per neuron it says much, unlike 35–37: 1,245 above all 2,048 controls (10.5 by chance), 2,946 with q ≤ 0.05
      (3,625 with the five strongest controls set aside), 1,896 more than five control SDs over the controls’
      median. Its strongest control stands 6.1 of those SDs out and the fifth 5.3, writing English (“however people
      most just knowledge standard”), not another language as at 35–37.
    - The steer check, run again with it: still weak at 32, now against the whole layer (a word shared with 0.064 of
      the 6,645 neurons above 95 in 100 controls; 0.030 shuffled).
    - **The board**: twenty layers done, 32–37 and 39–52, fourteen of them by volunteers; 31 (anon, eda6b586) and 38
      (Zlorthishen, 3853d507) running under kit 5; none open. Layer 30’s third trial passed (`match_things`), but no
      kit carries it yet.
  - **Kit 6: kit 5 with layer 30 added** (cut 18 November 2026, tag `census-kit-6`; the owner’s words: “Push it and
    register the bars for 30” and “Cut kit 6”, 18 November). Layer 30 is taken with `match_things`, `match_ceil`’s two
    counts there (±2.42 and ±2.67 ceilings) on twelve questions of three genres, whose trial passed on 18 November
    (above): 48 answers a unit, 1,130,508 a layer, about 33 hours on the home 5090. Its doses are not counted
    calibrated there: `DOSES_CALIBRATED` is not widened without the owner’s word. Every other layer is taken as under
    kit 5.
    - **The answers.** All eighteen of kit 5’s home canaries, taken again under kit 6, are the ones kept; layer 30’s
      is new (`76f9727`). Nothing that writes an answer changed after that. A 24 GB card still fits: at 30 the longest
      batch (walk’s) leaves 203 MiB of the 23,900 MiB `census.sh check` asks for.
    - **Its own unpushed answers**, from its trial on the home card (`census_volunteer.DESIGN_BASELINES`). A whole
      layer writes the first nine in batches of 64 and the last three in batches of 61, as its trial did.
    - **Its bars for another card class**, registered and pushed before anything was measured
      ([`census_other_card_bars/kit6/prereg/RULES.md`](experiments/results/census_other_card_bars/kit6/prereg/RULES.md),
      `ade1834`, pushed at 05:27Z; measured 05:28Z–06:43Z, `56f3021`). The control’s twelve unpushed answers are
      the trial’s bit for bit. The near-tie bar stands at 0.6: 24 of 132 arithmetic-only answers parted, the surest
      at a piece written at 0.52. On the kit’s own chunk 0, 768 answers to all twelve questions at both doses, the
      home card’s arithmetic moves the median to 0.0053 and the 99th percentile to 0.118; the bars are 0.013 and
      0.26, as `match_wide`’s at 31, and 0.90 of first word-pieces.
    - **Read whole** by its trial’s reader and two gates, the whole reading and the neurons the controls cover
      (`census_scaled_design/read_whole.sh`, which gives the trial’s verdict back gate for gate on the trial’s run).
    - **A layer begun under kit 5** is finished on kit 5 and checked by kit 5’s own file, read from its tag
      (`census_volunteer.earlier_kit`); at the cut those are 31 (anon, eda6b586) and 38 (Zlorthishen, 3853d507).
    - **The board** is kit 6’s with layer 30 open; 31 and 38 running under kit 5. *Not shown yet:* a whole layer of
      `match_things`. The thread post is the owner’s.
  - **Layers 31 and 38 read whole, the last two begun under kit 5** (19 November 2026; the owner approved the downloads:
    “Yes, take both in”). Each was taken with kit 5 at its tag (`3943703`), from a clean tree, and kit 6 was cut while
    it ran, so each was checked by kit 5’s own file, read from its tag (`census_volunteer.earlier_kit`). Both bundles’
    sha256 matched their DONE lines.
    - **Layer 38**, taken with `soft_ceil` by Zlorthishen (anon, 3853d507), who took 39 and 37, on an RTX 5090 with the
      home card’s bits. Begun 17 November at 22:36Z by the run’s record; DONE 18 November at 22:16Z, as a GitHub
      release. Its spot chunks (9, 22, 324, 382 and 423 of 737): 3,840 of 3,840 answers identical bit for bit.
    - **Layer 31**, taken with `match_wide` by anon (Asynonymous, eda6b586), who took 32 to 35, on the same RTX PRO 6000
      Blackwell card class. Begun 18 November at 03:41Z; DONE that day at 22:25Z, as a GitHub release. Its spot chunks
      (47, 121, 328, 340 and 541 of 737): the first word-piece the home card’s in 0.988 of answers (bar 0.90),
      log-probabilities apart by a median 0.0051 (0.013) and 0.117 at the 99th percentile (0.26).

    Each was read whole by its trial’s registered reader ([`census_L38_whole/`](experiments/results/census_L38_whole/),
    `census/trials/soft/prereg`, hashed 12 November at 22:34Z;
    [`census_L31_whole/`](experiments/results/census_L31_whole/), `census/trials/wide/prereg`, hashed 16 November at
    15:22Z):

    | layer, design, doses | every answer | the lower dose alone | the neurons the controls cover | its trial, every answer |
    |---|---|---|---|---|
    | 38, `soft_ceil`, ±1.6 and ±2.3 ceilings | 0.694, 0.727, 0.657 and 0.657, 0.670 | 0.653, 0.681, 0.612 and 0.589, 0.616 | 85 in 100; 0.661, 0.704, 0.618 and 0.619, 0.638 | 0.675, 0.712, 0.637 and 0.630, 0.643 |
    | 31, `match_wide`, ±2.58 ceilings | 0.641, 0.646, 0.584 and 0.586, 0.636 | one dose | 96 in 100; 0.642, 0.642, 0.584 and 0.585, 0.636 | 0.630, 0.645, 0.583 and 0.581, 0.608 |

    - Both pass every gate of their trials: 38 all three, 31 both of its two (`match_wide` has one dose, so no N2). 38’s
      least margin is 0.019, the lower dose’s words carrying from half B to half A (0.589). 31’s is 0.014, its words
      carrying from half A to half B (0.584); in its trial they carried 0.583 and 0.581. Each whole lead is over its
      trial’s, by 0.019 and 0.011.
    - Reported beside them, gating nothing: 38’s upper dose alone reads 0.664, 0.702, 0.654 and 0.630, 0.653. 31’s two
      readings on three questions a half lead by 0.623 and 0.620, and their words carry 0.546 to 0.556, under the bar.
      By the length rule 38 would take 32 word-pieces (its lead falls 0.004); 31 would not (its words carry 0.563 and
      0.568 at 32). Every kit has kept 48.
    - The text: at 38 the neurons lose 0.007 of their distinct 4-grams at the lower dose and 0.034 at the upper, the
      controls 0.001 and 0.004; at 31 both lose 0.002 to 0.003. 96.5 to 98.9 in 100 answers run to the 48-piece cut.
    - Per neuron 38 says much: 358 above all 2,048 controls (10.5 by chance), 7,003 with q ≤ 0.05 (7,047 with the five
      strongest controls set aside), 4,248 more than five control SDs over the controls’ median. Its strongest control
      stands 19.1 of those SDs out and the fifth 9.4; the five write Portuguese, Indonesian, Spanish and French. 31 says
      less: 250 above all controls, 1,893 with q ≤ 0.05 (2,151), 1,258 more than five SDs over; its strongest control
      stands 9.1 out, writing Spanish, and the fifth 4.2, writing English.
    - The steer check, run again with them: both still weak, now against whole layers (31: a word shared with 0.056 of
      the 4,007 neurons above 95 in 100 controls, 0.017 shuffled; 38: 0.328 of 8,104, 0.034 shuffled).
    - **The board**: twenty-two layers done, 31–52, sixteen of them by volunteers; 30 running under kit 6 (anon,
      6413bb68, an RTX PRO 6000); none open. 53 and 54 passed the `match_things` walk, but no kit carries them yet.
  - **Kit 7: kit 6 with layers 53 and 54 added** (cut 19 November 2026, tag `census-kit-7`; the owner’s words:
    “Register the bars for 53 and 54”, “Push it and run the bars measurement”, “You can start it up now” and “Cut kit
    7”, 19 November). Layers 53 and 54 are taken with `match_things`, on layer 30’s twelve questions at each layer’s
    own two counts (±2.24 and ±2.42 ceilings at 53, ±2.18 and ±2.34 at 54), whose walk passed there on 18 and 19
    November (above): 48 answers a unit, 1,130,508 a layer, about 28 and 30 hours on the home 5090 at the trials’
    pace. Their doses are not counted calibrated there: `DOSES_CALIBRATED` is not widened without the owner’s word.
    Every other layer is taken as under kit 6.
    - **The answers.** Kit 6’s seventeen home canaries still in the kit, taken again under kit 7, are the ones kept;
      53’s and 54’s are new (`dfb5ef2`). Nothing that writes an answer changed after that. A 24 GB card still fits:
      the longest batch (walk’s) leaves 321 MiB at 53 and 632 MiB at 54 of the 23,900 MiB `census.sh check` asks for.
    - **Their unpushed answers** are layer 30’s (`census_volunteer.DESIGN_BASELINES`): no push enters them, and the
      walk’s trials at 53 and 54 wrote the same twelve bit for bit.
    - **Their bars for another card class**, registered and pushed before anything was measured
      ([`census_other_card_bars/kit7/prereg/RULES.md`](experiments/results/census_other_card_bars/kit7/prereg/RULES.md),
      `00acf12`, pushed at 05:06Z). A first start refused at once, with nothing measured, as another job held the
      card; measured 06:15Z–08:45Z (`c28ea70`). At each layer the control’s twelve unpushed answers are the walk
      trial’s bit for bit, and the near-tie bar stands at 0.6: 24 of 132 arithmetic-only answers parted, the surest at
      a piece written at 0.52, as at 30. On the kit’s own chunk 0 of both layers, 768 answers a layer pooled, the home
      card’s arithmetic moves the median to 0.0037 and the 99th percentile to 0.114; the bars are 0.0091 (built to
      catch a push 5% off) and 0.26 (10% off), and 0.90 of first word-pieces. Neither layer’s own floor reaches the
      group’s bars. A push 10% off still keeps 0.98 of first word-pieces there (0.83 at 30), so at 53 and 54 the
      log-probability bars are what catch it.
    - **Read whole** by the walk’s reader and two gates, the whole reading and the neurons the controls cover
      (`census_scaled_design/read_whole.sh`, which gives each walk trial’s verdict back gate for gate on its run).
    - **A layer begun under kit 6** is finished on kit 6 and checked by kit 6’s own file, read from its tag
      (`census_volunteer.earlier_kit`); at the cut that is 30 (anon, 6413bb68, 36% on 19 November).
    - **The board** is kit 7’s with 53 and 54 open; 30 running under kit 6. *Not shown yet:* a whole layer of
      `match_things`. The thread post is the owner’s.
  - **Layers 30 and 53 read whole, the first whole layers of `match_things`** (21 November 2026; the owner’s words:
    “we have new layers to download and bring in + update the board”). Both bundles’ sha256 matched their DONE lines.
    - **The intake’s size cap refused both at first.** `census_intake.py check` capped what a manifest may say the
      answers unpack to at a flat 4 GB (since kit 1: 565,254 answers at about 7,080 bytes each), and a `match_things`
      layer writes 1,130,508 answers, 5.63 GB. Nothing on the volunteers’ side was wrong. The cap is now 7,000 bytes an
      answer times the answers the kit’s plan at the layer asks for, counted from the home canary’s meta, never the
      bundle’s (`MAX_ANSWER_BYTES`): 7.91 GB at 30, 53 and 54, 3.96 GB at the others. `check_volunteer_kit.py` checks
      the refusal (87 checks). The first checks had marked both claims rejected on the board; that was taken back
      before the board was published, and both were checked again under the new cap.
    - **Layer 53**, taken with kit 7 by Zlorthishen (anon, 7ce4c92d), who took 37 to 39, on an RTX 5090 with the home
      card’s bits. Begun 19 November at 21:35Z by the run’s record, in one session; DONE 20 November at 23:07Z, as a
      GitHub release. Its first PROGRESS line, posted at 21:39Z with its RESERVE and START, said 21%, which the run’s
      record does not bear out; the spot-check is what vouches for the answers. Its spot chunks (164, 174, 403, 1112
      and 1258 of 1,473): 3,840 of 3,840 answers identical bit for bit.
    - **Layer 30**, taken with kit 6 by anon (Asynonymous, 6413bb68), who took 31 to 35, on the RTX PRO 6000 Blackwell
      card class. Begun 18 November at 22:32Z, in four sessions; DONE 20 November at 23:22Z. Kit 7 was cut while it
      ran, so it was checked by kit 6’s own file. Its spot chunks (147, 336, 362, 552 and 577 of 1,473): the first
      word-piece the home card’s in 0.988 of answers (bar 0.90), log-probabilities apart by a median 0.0047 (0.013)
      and 0.112 at the 99th percentile (0.26).

    Each was read whole by its trial’s registered reader, two gates, N1 and N3
    ([`census_L53_whole/`](experiments/results/census_L53_whole/), `census/trials/things_walk/prereg`, hashed 18
    November at 16:18Z; [`census_L30_whole/`](experiments/results/census_L30_whole/), `census/trials/things/prereg`,
    hashed 18 November at 00:20Z). Each reading gives the lead, the lead past how the text survives, the words carried
    from half A to B and from B to A, and the new questions’ lead (bars 0.60, 0.58, 0.57, 0.56):

    | layer, doses | every answer | the neurons the controls cover | its trial, every answer |
    |---|---|---|---|
    | 53, ±2.24 and ±2.42 ceilings | 0.747, 0.714, 0.585 and 0.593, 0.757 | 99 in 100; 0.747, 0.713, 0.583 and 0.591, 0.757 | 0.751, 0.745, 0.591 and 0.598, 0.751 |
    | 30, ±2.42 and ±2.67 ceilings | 0.602, 0.613, **0.558 and 0.553**, 0.603 | 96 in 100; 0.604, 0.609, **0.559 and 0.554**, 0.605 | 0.618, 0.637, 0.582 and 0.587, 0.625 |

    - **53 passes both gates**, close to its trial. Reported, gating nothing: each dose alone leads (0.664 the lower,
      0.752 the upper) but neither carries its words between the halves (0.552 and 0.561; 0.544 and 0.559), nor do
      three questions a half (0.541 to 0.573), as in its trial. By the length rule it keeps 48 word-pieces (at 32 its
      words carry 0.534 and 0.543). The neurons lose 0.11 to 0.12 of their distinct 4-grams at the lower dose and 0.26
      to 0.27 at the upper, the controls 0.20 and 0.43; every answer runs to the cut. Per neuron it says much: 1,466
      above all 2,048 controls (10.5 by chance), 3,640 with q ≤ 0.05 (3,854 with the five strongest set aside), 3,054
      more than five SDs over the controls’ median. Its strongest controls stand 6.2 to 10.2 out writing “thought” run
      together, as the walk’s trials at 53 to 55 did. The steer lens, checked against a census above 52 for the first
      time, agrees: a word shared with 0.738 of the 5,807 neurons above 95 in 100 controls (0.009 shuffled).
    - **30 misses both gates on one figure**: how its words carry between the question halves, 0.558 and 0.553 against
      0.57 (0.559 and 0.554 on the neurons the controls cover). Its lead, 0.602, sits on its bar; nothing else is
      short. Reported: each dose alone and three questions a half lead by 0.57 to 0.60 and carry 0.537 to 0.556. Per
      neuron it says little: 49 above all controls (10.5 by chance), none with q ≤ 0.05 (655 with the five strongest
      set aside, which stand 8 to 11 SDs out writing German, Portuguese and Spanish), 475 more than five SDs over. The
      text barely moves (0.002 to 0.003 of distinct 4-grams). The steer lens stays weak there (0.042 of 2,627; 0.019
      shuffled).
    - **Why 30 fell, read after the verdict** ([`census_L30_whole/controls/`](experiments/results/census_L30_whole/controls/),
      unregistered and reported, the registered reader unchanged). The trial’s own 1,022 neurons, read from the
      volunteer’s run, fall short too (0.564 and 0.557). On those 49,056 answers the two cards agree on the first
      word-piece in 0.986 and on all 48 pieces in 0.678, as another card class does. Swapping one thing at a time:
      the volunteer’s answers against the trial’s 1,024 controls carry 0.579 and 0.582, a pass; the home card’s
      answers against the kit’s 2,048 carry 0.565 and 0.562, a miss; each half of the kit’s 2,048 alone gives 0.562
      and 0.557, and 0.560 and 0.561. So the card moves the figure by 0.001 to 0.005 (the volunteer’s lower each
      time) and the controls by about 0.02: three draws of 1,024 isotropic controls read the same neurons’ words
      carrying at 0.56, 0.56 and 0.58, and the trial’s was the kind one. Their push sizes are not the difference: both
      draws are paired with neurons from across the layer, and their write ceilings match the neurons’ (median 51.5
      and 51.6, the same tails). A margin of 0.012 at a trial is within what another draw of controls moves. Every other layer read whole
      against the kit’s controls has held its trial’s gates, save 36’s one figure.
    - **What it changes.** The runs are whole and honest and are kept. On the board 30 is done with the miss named, as
      36 is, and a reading of it is shown as any census whose doses are not counted calibrated. `DOSES_CALIBRATED` is
      not widened for `match_things` anywhere (the owner’s word).
    - **The board**: twenty-four layers done, 30–53, eighteen of them by volunteers; 54 running under kit 7 (anon,
      0471bd29, an RTX PRO 6000, 53% on 21 November); none open.
  - **Layer 54 read whole** (22 November 2026), taken with kit 7 by anon (Asynonymous, 0471bd29), who took 30 to 35, on
    the RTX PRO 6000 Blackwell card class. Begun 20 November at 23:29Z by the run’s record, in one session; DONE 22
    November at 01:03Z, as a GitHub release, its sha256 matching the DONE line. Its spot chunks (198, 526, 762, 990
    and 1267 of 1,473): the first word-piece the home card’s in 0.993 of answers (bar 0.90), log-probabilities apart
    by a median 0.0037 (0.0091) and 0.112 at the 99th percentile (0.26). Read by its walk’s registered reader, N1 and
    N3 ([`census_L54_whole/`](experiments/results/census_L54_whole/), `census/trials/things_walk/prereg`, hashed 18
    November at 16:18Z), each reading’s figures given as for 30 and 53 above (bars 0.60, 0.58, 0.57, 0.56):

    | layer, doses | every answer | the neurons the controls cover | its trial, every answer |
    |---|---|---|---|
    | 54, ±2.18 and ±2.34 ceilings | 0.727, 0.720, 0.621 and 0.625, 0.727 | 97 in 100; 0.724, 0.717, 0.614 and 0.619, 0.726 | 0.712, 0.714, 0.627 and 0.620, 0.709 |

    - **54 passes both gates**, a little above its trial on the lead, and with room: its least gated figure, 0.614,
      clears its bar by 0.044, twice what another draw of controls moved layer 30’s. Reported, gating nothing: the
      lower dose alone again passes what N2 asks (0.645, 0.667, 0.578 and 0.587, 0.598; its trial 0.645, 0.656, 0.575
      and 0.583, 0.597), so a design of that dose alone is still possible there; the upper dose alone leads (0.702)
      but its words carry 0.569 and 0.576 (0.593 and 0.591 in its trial). Three questions a half carry 0.589 and
      0.589, and 0.601 and 0.602. By the length rule it keeps 48 word-pieces (at 32 it passes too, lead 0.682, words
      0.580 and 0.588). The neurons lose 0.11 to 0.12 of their distinct 4-grams at the lower dose and 0.22 to 0.23 at
      the upper, the controls 0.12 and 0.28 to 0.29; all but a few controls’ answers run to the cut. Per neuron it says
      much: 4,254 above all 2,048 controls (10.5 by chance), 5,481 with q ≤ 0.05 (5,710 with the five strongest set
      aside), 4,731 more than five SDs over the controls’ median. Its strongest controls stand 5.7 to 6.3 SDs out
      writing “thought” run together, as at 53. The steer lens, unchecked at 54 until now, agrees: a word shared with
      0.798 of the 7,110 neurons above 95 in 100 controls (0.009 shuffled).
    - **The board**: twenty-five layers done, 30–54, nineteen of them by volunteers; none running, none open: every
      layer a kit offered is done. `DOSES_CALIBRATED` is not widened for `match_things` (the owner’s word).
  - **A volunteer’s two suggestions, tried** (22–23 November 2026; the owner’s words “Let’s try both of those
    recommendations”, then “Push, then start”). On Gemma 4 12B, eigentactics found (board issue #1, comment 7084903662)
    that the lead at their layer 24 came back once the upper dose sat at the layer’s text edge, the largest push keeping
    90% of the distinct 4-grams, and the lower half a doubling under it; and that a neuron’s write direction read
    through the final norm and the unembedding names its census words at their late layers.
    - **The upper dose at the text edge, at 29** ([`census_edge_trial/`](experiments/results/census_edge_trial/)). A
      fine ladder on `match_things`’s questions (`calib_edge`, 2.32 to 6.40 ceilings, 7% a rung; its rule hashed before
      it ran) broke the text first at 3.99 ceilings, so `edge_things` asked ±2.64 and ±3.73 (0.57 and 0.81 of the
      stream; `match_things` had ±2.10 and ±2.32). Registered and pushed before its first answer, its trial **fails
      every gate**: lead 0.473, words carrying 0.488 and 0.488, new questions 0.502. Across the two trials at 29 the
      lead on each dose alone falls with every step up, 0.569, 0.532, 0.502, 0.428 from ±2.10 to ±3.73: neurons and
      random directions alike switch to Indonesian, Portuguese, German and Spanish, and the neurons’ words are mostly
      those any push of that size brings out. The text barely moves even at ±3.73 (0.05 of its 4-grams). The 12B’s
      lesson does not carry here: nothing was past the edge, and the gentlest push tried read best. Nothing opens at 29.
    - **Each neuron’s weight-lens words, fixed before its answers are read**
      ([`census_weight_words/`](experiments/results/census_weight_words/), exploratory and unregistered, read on
      trials already seen). A neuron’s answers are scored on its naked-eye words (the atlas’s own labels) against all
      the controls’ on the same words, so nothing is picked and each question half is its own test. At 53 to 55 it
      leads on each half alone (0.68 to 0.76; about 0.51 shuffled), at 55 by 0.677 and 0.676 where the census’s picked
      words did not carry; at 29 and 30 it reads almost nothing (0.53 to 0.57, no neuron at q ≤ 0.05). At the top
      layers this is close to built in, a narrower claim than the census’s. Judging a layer by it would need a fresh
      trial and bars of its own, and whether a layer read only so counts as taken is the owner’s word.
- **A pattern put to the test, without a push** (built 14 November 2026 on the stand-in, and tried on the 31B the
  same day, below; the owner’s words that day, “Start on the first two ideas from the article”, then “Pause 36, try
  now”). The census
  and the steer lens say what a push along a neuron’s write direction does. No push strength has passed a trial
  below layer 32, and the steer lens is a weak guide at 32 to 38 and was not checked below 32, so thirty-two layers
  have no reading that has been checked. Goodfire’s *Neurons as an Interpretable Basis* (10 November 2026; REFERENCES [54]) reads neurons from
  where they fire by themselves and tests a written description by whether it predicts the neuron; its early-layer
  neurons read crisply that way. Two of its tests are built here on what the atlas already keeps
  (`backend/pattern_tests.py`, ARCHITECTURE §4.7):
  - *Written.* Texts written to fire a face, and near misses written not to, are each read once. The test says how
    many of the texts meant to fire reached half the face’s extreme, how many beat the strongest thing in the
    corpus, and how many near misses fired.
  - *Blind.* About half of each face’s stored passages (those in odd-numbered windows) are now kept back from the
    scientist. Each is paired with a passage where the face stays quiet, read live, and a reader given the pattern
    and nothing else says which of each pair fires. The reader is an endpoint’s model in a fresh conversation. With
    no endpoint the scientist answers the pairs itself, and the result says so.

  The scientist has both as tools (`test_by_writing`, `blind_test`) and is told to run them before it proposes a
  name from passages. Results are kept by face with the pattern they tested, and the details tab lists them, in red
  where a test went against the pattern.

  *Tried on the 31B* (14 November, 03:04Z to 03:38Z, with layer 36’s census stopped between batches and resumed
  after; `experiments/results/pattern_tests_pilot`, `pilot.py` and `read.py` there). Everything went through the
  server’s own routes, as the scientist’s tools do.
  - **The blind test can be made nearly everywhere.** Of 176 faces drawn at random, sixteen at each of layers 2, 8,
    14, 20, 26, 31, 36, 40, 45, 50 and 56, a test could be begun for 175. The other had three kept-back windows
    where four are needed. No face was refused for firing too widely, which is what the stand-in’s neurons do. A
    test has a median of 9 pairs (5 to 12): 8 of 9 right is 1 chance in 51 by guessing, 9 of 9 is 1 in 512.
  - **A pattern written from half a face’s passages picks out the other half.** For 21 of those faces, three at
    each of layers 2, 14, 26, 31, 36, 45 and 56 (the first three in the draw; none was chosen for being readable),
    a pattern was written from the passages not kept back. Each test was then read twice, by two readers in fresh
    conversations who saw only patterns and pairs: once with the face’s own pattern and once with another face’s,
    neither reader told which. With its own pattern: 177 of 196 pairs right (90 in 100), and 15 of the 21 faces
    better than 1 chance in 20 each. With another face’s pattern: 86 of 196 (44 in 100), and no face better than 1
    in 20. So the pairs give nothing away by themselves; the pattern does the work. Goodfire report 86 to 95 in 100
    on their model.
  - **A text written to the pattern fires the neuron about a third of the time, and a near miss never did.** Three
    texts and three near misses a face: 22 of 63 texts reached half the face’s extreme, 2 beat the strongest thing
    in the corpus, and 0 of 63 near misses fired. Twelve of the 21 faces had at least one text that fired.
  - **The two tests disagree on six faces, and the written one is right.** Six of the 21 do not fire on their own
    strongest passages when only the 60 word-pieces before the peak are read (the same passages read in their
    windows are what the index holds). None of the 18 texts written for those six fired; a reader still picked 49
    of their 54 pairs. Their patterns name the word and its phrase, which is enough to tell the passage from a
    random one, and not enough to fire the neuron. Read with more before the peak, they come back: one needs 128
    word-pieces, two need 256 and three need more than 256. Putting 256 or 420 word-pieces of an unrelated
    document in front does not stand in for their own (16 of their 18 passages stay silent behind it). One at
    layer 36 fires on “In conclusion” only when the essay it
    concludes is there to read. Over all 176 faces, 30 need more than 60 word-pieces (two at layer 8, the rest at
    layer 26 and later, where it is one face in four) and 8 need more than 256.
  - *What was changed on the strength of it.* A face’s stored passages often hold the same event twice, once among
    the strongest and once among those drawn at random, so a blind test would have counted one pair as two: it now
    takes one event a window (the figures above are after that fix; before it the same survey showed 11 pairs
    where there are 9). And a written test now reads the face’s own three strongest passages alone beside its
    texts (`own`), and says, on the page and to the scientist, when they no longer fire: then no short text can
    fire the face, and the pattern has yet to say what the earlier text must be.
  - *What the pilot does not show.* The patterns were written by the session that built the tests, from the fit
    passages alone, not by the scientist through its tools. The readers were Claude Fable 5.2 conversations run by
    hand, because no endpoint is configured on this machine; the product’s own path, an endpoint’s model, has run
    only against a stub. It is 21 faces and one reading of each. The stored passages are a face’s strongest
    events and a few above half its extreme, so both tests speak for where a neuron fires hard and say nothing of
    its many weaker firings. With another face’s pattern the readers were a
    little under a half (86 of 196; guessing does as badly 1 time in 20), as a reader steered by a wrong
    description would be.

  *Checked:* `check_atlas_api` 443 (424 before the tests were built), `check_scientist` 236 (226),
  `check_interface_defaults` 291, `check_local_only` 1213, `check_atlas_index` 178; the page type-checks and builds.
  *Built since, on the owner’s word (“Yes, build it”, 14 November):* **a written test behind a face’s own
  passage**, for the faces a short text cannot fire. One of the face’s stored passages is held up to its last few
  word-pieces (8 unless said, the peak among them) and each written text is read as its ending, in place of the
  passage’s own. Some endings are written to the pattern and should fire; near-miss endings share their words,
  leave out what the pattern says matters, and should not. Near misses are required: the held text is the
  neuron’s own strongest, so an ending that fires shows little unless a near miss stays quiet. The passage’s own
  ending is read beside them. The scientist’s `test_by_writing` takes `behind` (a passage’s window) and first shows
  where the passage is cut; a plain written test that finds a face needing more text points to it. What it shows
  is narrower than the plain test: with this earlier text given, what the pattern names in the ending is what
  fires the neuron. What the earlier text has to be is still not tested. *Checked on the stand-in only:*
  `check_atlas_api` 448, `check_scientist` 238, `check_interface_defaults` 291, `check_local_only` 1226 (a route
  more); the page type-checks and builds.

  *Run on the 31B* (14 November, 17:43Z, once layer 36 had left the card; `pilot.py behind`,
  `behind_results.json`), for the pilot’s six faces that no short text fires. The endings were written beforehand
  from where each passage is cut (`behind.json`), the first of each passage’s being its own words.
  - **Behind their own passage the six can be fired by new words: 6 of 17 new endings did, where none of their 18
    short texts had.** 2 of 24 near-miss endings fired too, where none of the pilot’s 63 short near misses had. Ten
    passages were read; the eleventh was refused, as it should be, because it reads under half the neuron’s extreme
    where it peaks. Every other passage’s own words fired behind its own earlier text (0.62 to 1.08 of the extreme).
  - **It sorts the six patterns, which the blind test could not** (readers had picked 49 of their 54 pairs):

    | face | new endings that fired | near misses that fired | what it shows |
    |---|---|---|---|
    | layer 26, neuron 4837, + | 4 of 4 | 1 of 4 | the pattern holds: a frequency or ordering word after the subject (“typically”, “often”, “Secondly”, “next”). The near miss that fired, “FPB is a hereditary”, fired on “is” at 0.51, so the word right after the subject carries some of it by itself |
    | layer 56, neuron 3342, + | 1 of 3 | 0 of 5 | partly: “and” between verbs of watching and controlling reads 0.52 and 0.36, between “build and repair” or “teach and train” under 0.08 |
    | layer 26, neuron 14541, − | 1 of 2 | 0 of 4 | not shown: the one that fired did so on “their”, not where the pattern said |
    | layer 45, neuron 16700, − | 0 of 3 | 0 of 5 | not shown, though it leans the pattern’s way: the “to” of purpose reads 0.46 and 0.33 against 0.04 for the near misses, and “how to” reads nothing |
    | layer 56, neuron 19940, − | 0 of 2 | 0 of 3 | not shown: other table rows of measurements read 0.03 and 0.02 where its own row reads 1.08 |
    | layer 36, neuron 12896, + | 0 of 3 | 1 of 3 | **wrong**: “To conclude”, “Overall” and “Ultimately” read 0.10 to 0.17, and “The third poem”, which concludes nothing, reads 0.98 on “The”. Here the neuron fires on “In” or “The” opening the paragraph, not on a conclusion |

  - **What it does not show.** One writer (me), who wrote the patterns too, and two to four new endings a passage:
    a pattern “not shown” may be right and badly served. An ending counts as fired wherever in it the neuron peaks,
    which is why the table says where. The earlier text is the neuron’s own strongest, so nothing here says what
    that text has to be.

  *And a harder quiet passage for the blind test* (the owner’s word, 14 November: “build the harder quiet passage
  for the blind test”). Against a passage drawn from anywhere, a pattern that names the word and its phrase is
  picked rightly whether or not the phrase is what fires the neuron; the six faces above passed that way. The quiet
  passage is now a **near miss** where the corpus has one: another place with the same marked word-piece between
  the same two neighbours, where the neuron stays quiet. Failing that, the same word-piece after the same one,
  then before the same one, then alone (six places tried of each), and only then a passage from anywhere. The
  phrase is then in both snippets, and only what the pattern says of the rest of the text tells them apart. The
  result gives the score over the pairs with a near miss, and by how near. With it:
  - **The places tried are a reading of their own.** Each is counted with whether the neuron fired there, so a test
    says how often the neuron fires where the same phrase stands elsewhere in the corpus. That is whether the
    phrase is enough, and it needs no reader and no written text.
  - **Both snippets of a pair show the same number of word-pieces**, so a short one gives nothing away. A reader
    can be shown up to 256 word-pieces before the marked one, for a neuron that needs more of its text.
  - **Weak firings are told apart from strong ones.** A stored passage that itself reads under the quiet bar is no
    longer used as a firing one, and the result gives the score over the pairs whose firing passage reaches half
    the neuron’s extreme. Looking back at the pilot with that: 50 of its 196 firing passages read under a half,
    and readers picked 136 of the 146 that reached it and 41 of the 50 that did not. *A correction:* this roadmap
    said of one pilot face (layer 14, neuron 11687) that its pattern “is true and is not all the neuron does”,
    because its three texts fired and a reader picked only 6 of its 10 pairs. All ten of those held-back passages
    were weak firings, under half its extreme, while the pattern was written from its strongest. The pattern was
    never tested on a strong held-back firing; what it shows is that the neuron’s weak firings are another matter.

  *Checked on the stand-in only:* `check_atlas_api` 454, `check_scientist` 240, `check_interface_defaults` 291,
  `check_local_only` 1226; the page type-checks and builds.

  *Run on the 31B* (14 November, 17:44Z to 17:58Z; `pilot.py start_near`, `answer_near`, `blind_near.json`): the
  pilot’s 21 faces again, the same patterns, the same 196 held-back passages, two fresh readers a test as before
  (one with the face’s own pattern, one with another face’s), each quiet passage now a near miss. A near miss was
  found for 189 of the 196 pairs.
  - **Readers picked 158 of 196, where they had picked 177 against a passage from anywhere.** With another
    face’s pattern they picked 106 (86 in the pilot), which is chance (guessing does as well 1 time in 7). 9 of
    the 21 faces are beyond chance one by one, where 15 were.
  - **The nearer the miss, the harder the pair**, as it was built to be:

    | the quiet passage has | pairs | picked with the own pattern | with another face’s |
    |---|---|---|---|
    | the same word between the same two neighbours | 64 | 45 (70 in 100) | 34 |
    | the same word after the same one | 60 | 48 (80) | 40 |
    | the same word before the same one | 37 | 34 (92) | 20 |
    | the same word alone | 28 | 25 (89) | 10 |
    | a passage from anywhere | 7 | 6 | 2 |

  - **What was said beforehand to look for did not happen.** The six faces that need their own document, whose
    patterns I thought named only the phrase, were to fall toward a half. They fell from 49 of 54 to 44, and the
    other fifteen from 128 of 142 to 114: about ten in a hundred each. The written test behind their own passage,
    above, is what told those six apart, not this.
  - **Two patterns the pilot passed whole are at chance against near misses**: layer 26’s neuron 4644 (9 of 9,
    now 4 of 9) and layer 31’s neuron 20806 (12 of 12, now 7 of 12). My reading: each names its phrase and not what
    else has to be there.
  - **The reading that needs no reader is the plainest.** For the 18 faces above layer 2, the neuron was read at
    68 other places in the corpus with the same word between the same two neighbours: it fired at none, and was
    quiet at 59. With only the word before the same, it fired at 1 of 50; with only the word after, at 1 of 28;
    the word alone, at 0 of 31. So above layer 2 the phrase a neuron fires on is not enough to fire it: the same
    words elsewhere leave it quiet. At layer 2 it can be: neuron 16560 fired at 44 of 45 such places and at 65 of
    72 with only the word before the same, a neuron of the word itself, and its test had to fall back to places
    where only the word after is the same.
  - **Strong firings are still read better than weak**: 123 of the 146 pairs whose firing passage reaches half the
    neuron’s extreme, 35 of the 50 that do not.
  - **One figure I cannot explain.** With another face’s pattern, readers picked 40 of the 60 pairs whose quiet
    passage has the same word after the same one. Alone that is 1 chance in 150; it is one of five rows looked
    at, it is spread over 17 faces and no one face makes it, and the other four rows are near or under a half. It may
    be chance. If it is not, something in those pairs gives the firing snippet away without the pattern, and the
    48 of 60 beside it would be worth less than it looks.
  - **What it does not show.** The same limits as the pilot: patterns written by one writer, 21 faces, one reading
    each, readers that are fresh Claude conversations and not an endpoint. A place counts as “fired” at half the
    neuron’s extreme and “quiet” under a quarter, and the search for a near miss stops at the first quiet place, so
    the counts of places are a rough reading, not a rate. Places inside the face’s own stored windows are left out.

  *Not done:*
  - **A passage on the same subject.** A near miss matches the word and its neighbours, not the topic. For a neuron
    that follows a subject and not a phrase, the near miss is often just the same common word elsewhere, which is
    little harder than a passage from anywhere. Goodfire’s control, a passage on the same subject where the neuron
    is quiet, is not built.
  - More pairs. A passages index built with more events a face (20 minutes on the card) would give a test more
    than 9 pairs, and give the scientist back what the keeping-back took.
  - A person cannot yet run either test from the page; the scientist can, and so can a script.
  - A second job the pattern leaves out is never looked for. Clustering a face’s firings by what drives them (their
    “modes”) is not built.
- **A census read by seeking, and every census at once** (built 12 November 2026; CPU checks on the stand-in, then
  driven on the eleven layers through the page). The server used to read a layer’s whole run, 2.7 GB, to show its list
  or one neuron’s answers, and kept two layers: 21 and 30 seconds to open a layer (timed on layers 43 and 46), again at
  every third layer a web of links led to. A run is now read by seeking (`census.RowsIndex`, `census.UnitsTable`;
  ARCHITECTURE §4.7): a layer’s list opens in 0.13 s and a neuron’s answers in 0.01 s, after 2.4–3.4 s once for each
  layer’s rows index and 6 s once for the eleven tables. That is also what a census at sixty layers would need of the
  page. With every table in memory:
  - **Every layer with a census is inked on the map at once** (`GET /census/levels`), where only the open one was, and a
    strip of bars over the census list shows each layer’s share of neurons with q ≤ 0.05; a bar opens its layer.
  - **A search of every census at once** (`GET /census/search`; the search box’s `census`; the scientist’s
    `search_census`): the neurons whose push brought a word, the strongest first. Each layer’s random controls are
    searched the same way and are its null. For “rain”, 135 neurons in the eleven layers against 3 of the 22,528
    controls, which among as many neurons would be about 32; layer 47’s N17966 comes first. A word is matched from its
    beginning because inside words the controls keep pace: “\*rain” (train, brain, grain) finds 1,138 neurons against
    77 controls, about 808 neurons’ worth. That count is the search’s null, and the list says it first; where no
    control brought the word (“king”: 74 neurons, none of 22,528 controls) it says “fewer than about 32”, not none.
    A hit no stronger than a control of its own layer is shown faded, as a weaker lead, but a bright row is not
    thereby beyond chance: the review of 12 November found “bridge” with 108 bright rows of 5,729 where about 6,437
    matches, and about 115 rows above every control, are what chance gives, and “ocean” with 53 bright of 66 where
    about 42 are chance. The page, the route and the scientist’s tool now say so, and none is bright in a layer
    without controls. A per-row null worth the name needs more controls than 2,048 a layer, or a statistic pooled
    over layers; neither is built. A census word in a neuron’s details searches for it.
  - **Smaller things.** A first visit opens on a layer with a census (it opened on layer 54, which has none). A layer
    without one says which layers have. The neurons and heads opened lately stay as a trail over the left panel. A
    neuron’s link can be copied. Where a run’s doses are calibrated at its layer, a neuron’s census answers come before
    the run’s card, which follows them; where they are not, the card stays first, in red.
- **The census on larger and rented cards.** A run’s batches are planned from its job list and the batch size, written
  whole, so a stopped run, a torn write and a run in shards each write the rows an unbroken run does (checked row for
  row on the stand-in). The batch and the plan are resume keys and fix a run’s determinism class: measure what a card
  holds once, then keep that `--batch`. Each meta records the runtime it was written on (the card, the versions, the
  commit, the stats index’s sha256) and warns when that changes; the unpacker is recorded beside it, and, since 12
  November 2026, the weights the answers were taken from (`weights`, the card the indexes carry: the checkpoint’s
  weight files, or a GGUF’s fingerprint), across which a run is not continued, merged or joined. `--shard i/n`
  writes whole chunks, each shard but the first with a canary batch, and the merge refuses a missing shard or answer
  (unless `--partial`) and compares the canaries; a shard is never served as a layer’s census. The census frees the J++
  lens after loading and reads each step back in one transfer. `scripts/verify_atlas_writer.py` measures the writer’s
  floor (how far an unpushed answer agrees with itself written in the batch shapes the census writes in), and the census
  card quotes it: a pushed answer that parts from the unpushed one no sooner than that floor may have parted by the
  batch’s shape alone. On the home 5090 (28 October 2026, 2 minutes with `compiled`;
  `experiments/results/atlas_writer_floor_rtx5090.json`) an unpushed answer written in a census batch of 64 first parted
  from itself written alone after 22 word-pieces (one question of four), in a batch of 32 after 54; alone, as `/steer`’s
  middle row and by the sweep’s writer it held all 64. Within one shape the unpushed row and a dose-0 row part alike, so
  a run, whose baseline is written in its own batches, compares like with like. Without the lens the 5090 holds a batch
  of 96 at `--max-new 64` beside a desktop (7.2 GiB above the model, COMPUTE §ii). On one card, across processes, a run
  is bit for bit itself (29 October 2026, the 5090, `compiled`, a `calib_rel` run of 16 neurons and 8 controls at layer
  45, 868 answers; [`census_determinism/`](experiments/results/census_determinism/)): the same run taken in two shards
  and merged equals the unbroken run in every field of every row, and two canaries taken in separate processes are
  identical. Still to measure: how far a run differs across cards.
- **The wiring.** Links are ranked by |P| f², with Gemma’s layer scalars and the norms’ usual scales (the review of 27
  October 2026), and every consumer reads their signs one way, from rest through GeGLU’s gate. That model is a
  simplification, and a second reading is now built beside it: to first order about a reader’s operating point (g₀, u₀),
  a writer moves it by f (α g + β u), α = gelu′(g₀) u₀, β = gelu(g₀), with the operating point served on each face taken
  as the median over its strongest stored passages of that face: where it fires hardest, not where it sits (below). It
  is odd in the writer, where the sign from rest is even. `model=use` reads it on `/neuron/wiring`, `/web/summary`,
  `/cone` and `/run/paths` (the default stays from rest), the details tab marks a link where the two disagree, and
  `atlas_build.py wiring_check` measures both: readers under ±d pushes of 512 writers a layer, d a quarter of each
  writer’s own ceiling, against as many readers they are not linked to.

  **Measured on the 31B** (the night of 28–29 October 2026: the passages, the wiring rebuilt with each link’s g and u,
  then all 59 layers of the check in about 31 minutes on the 5090 with `compiled`; per-layer medians, linked / null, at
  the last word-piece of the census’s four questions, the chat template’s closing `<channel|>`, where the answer begins,
  not a word of the question; ARCHITECTURE §4.7; every figure here is kept with the script that gives it in
  [`atlas_wiring_check/`](experiments/results/atlas_wiring_check/)). As built, the sign from rest held on the gate face
  at 0.336 / 0.341 and the sign in use at 0.418 / 0.438, below chance for links and null alike (changes of exactly 0
  left out, as the summary now counts them; the file’s stored summary, counting a 0 as a miss, gives 0.320 / 0.324 and
  0.417 / 0.427). No sign convention is wrong: at that word-piece about nine readers in ten sat with the gate shut,
  70% / 63% (pooled) below gelu’s minimum, where the up path carries about −sign(P) (the sign from rest holds where the
  gate is open, 0.82 / 0.69); and the sign in use is read where each reader fires hardest, its strongest stored
  passages, where the gate is open. Scored offline from the stored responses at the readers’ measured operating point
  (their g₀ and u₀ read afterwards on the CPU from the packed checkpoint, one prefill a question, by
  `scripts/z1_31b_g0u0_cpu.py` in the folder above), the same responses agree with the sign in use at 0.835 / 0.854
  (0.842 / 0.853 in float64, as the summary now computes it), and the change’s size follows f |α₀ g + β₀ u| d with
  ρ 0.89 / 0.87, against 0.31 for |P| f². The links are real by size: linked readers move 5.5 times as much as the
  null’s. Over the whole index the sign in use differs from the sign from rest on 28% of links on each face of the
  reader (28.2% on +, 28.3% on −; on both only 1.9%, so on about 55% of links on one face or the other; the sign in use
  there is read where the reader fires hardest). The two readings now state their conditions (`LINK_MODEL`: the sign
  from rest holds for an open gate, or a push big enough to open it; `LINK_MODEL_USE` and the details tab: read where
  the reader fires hardest, not where it sits), and the check records each reader’s own g₀ and u₀, scores the sign in
  use there, splits both signs by the reader’s regime (open, shut above gelu’s minimum, below it), judges sign shares
  against 0.5 (the null tests size) and ranks the size against the first-order term too.

  **Rerun with g₀ and u₀ recorded** (29 October 2026, 04:18 CDT; `wiring_check --fresh`, the night’s first check set
  aside; 59 layers in about 32 minutes; [`rerun_at_site.json`](experiments/results/atlas_wiring_check/rerun_at_site.json)):
  the check itself now gives what the offline scoring gave. Medians over layers, linked / null: the sign in use scored
  at each reader’s site holds at 0.841 / 0.853 (under the plain +d push 0.866 / 0.852); read where it fires hardest,
  as served, 0.418 / 0.438; the sign from rest 0.336 / 0.341; size against f |α₀ g + β₀ u| d at ρ 0.89 / 0.87, against
  0.31 for |P| f²; linked readers moved 5.5 times the null’s. By regime, for linked readers: the gate open at 7% of
  pairs (the sign from rest 0.82, at the site 0.86), shut above gelu’s minimum at 14% (0.47, 0.74), below it at 80%
  (0.19, 0.88); the null’s shares 12%, 21% and 68%. (The 70% / 63% above are pooled over every pair, these the median
  over layers.)

  Still open: (1) The sign in use the atlas serves is read where each reader fires hardest on a face, not where it sits
  at an ordinary word-piece. A resting operating point (the median g and u over a sample of ordinary word-pieces,
  beside the passages index) or one per GeGLU regime would read it there; whether to build one is the owner’s choice,
  and until then every page says where it is read. (2) The dose: at a quarter of the ceiling the pushes stay near
  first order; at the whole ceiling gates opened in 11–61% of linked pairs and the sign from rest rose (layer 35: 0.44
  to 0.77), while the first-order sign in use at the unpushed point fell to 0.42–0.53: each reading holds in its own
  regime of dose, measured for 24 writers on one question.
- **Stored passages for each face** (Cammarata’s practice iii, the study’s §3.5): **built on Gemma-4-31B** (28 October
  2026, 20 minutes on the 5090 with `compiled`, over the 2,000,054 word-pieces in 3,914 windows the stats index read;
  its log in `experiments/results/atlas_wiring_check/gpu_logs/`). One pass over the corpus the stats index was read over
  (`atlas_build.py passages`, the same windows) keeps each face’s sixteen strongest windows and six drawn at random
  above half its extreme, with the gate and up values at each event (the operating point the wiring’s sign in use
  reads). Still open: whether the words over sixteen passages make sound labels (condition 1 of §ii; no test has checked
  them). The stats pass now keeps its norms in float32 buffers (about 1.4 GB at two million word-pieces, where Python
  floats took about 11.5 GB, inferred), so a pass over ten million fits in memory, its quantiles unchanged.
- **Single neurons at pages 21–31.** The study's 60-fold push overdrove them (errata E43, E46). The calibration (above)
  moved their answers readily at layers 20 and 30, from 0.05 of the stream at 20 and 0.1 at 30, but no more than matched
  controls moved theirs; on every answer the census’s ranking told them from the controls not at all at layer 20 and
  weakly at 30 (AUC of mass 0.58 and 0.59, as weak as at 50).
- **The scientist’s reach.** Since the review of 27 October 2026 it reads, writes and edits files only in its work
  folder under Claude Code, with no shell, and runs Codex’s shell in its workspace-write sandbox;
  `THEODOLITE_SCIENTIST_REACH=home` gives neuron-explorer’s shell over the home folder, and the page asks first. On 28
  October the folder rules were tried on the real CLI, Claude Code 2.1.312, against a stand-in Messages API, not a live
  model (`scripts/verify_scientist_cli.py`): in the work folder Read, Write and Edit run; beside it, a path climbing out
  of it, the prompt files, the home folder and `/proc/self/environ` are refused; Bash and Glob are not there; the
  atlas’s tools run through the real MCP server. That run found that `--setting-sources project` let a CLAUDE.md (or a
  settings file’s hook) in the work folder, which the scientist may write, or in any folder above it, the home folder’s
  own `.claude/CLAUDE.md` included, reach every later chat; a chat now loads none (`--setting-sources ""`). A chat
  takes up changed instructions and tools: a Claude Code chat at the folder reach is started again at the next message,
  resuming its session; any other is marked stale and the page offers a restart, since ending its process group also
  ends any commands it left running. The names and notes a scientist writes are proposals, kept or discarded by the
  person (§iii). Still open: Codex in its sandbox can still read files outside its folder, credentials kept in files
  among them, and write the temporary folders (`/tmp`, `/dev/shm`: measured 14 November 2026 in its sandbox helper,
  not through a chat); torch.compile’s cache was moved out of `/tmp` that day (`backend/runtime_factory.py`), except
  in a volunteer’s kit and where a check script calls the checkpoint’s compiled unpacker with no install before it,
  and what a compile writes while it runs still
  passes through `/tmp`; no script tries the real Codex; another account is not told
  apart on macOS or elsewhere, nor through vite’s proxy (a per-launch key would cover both); a scientist with a shell
  over the home folder could keep its own proposal through the loopback port, which the guard cannot tell from the
  page; the rules hold for the CLI version tried, so a new one should be tried with the same script; and no live model
  has yet been watched to see whether what it reads would lead it to try what the CLI refuses.
- **The push, greedy or sampled** (neuron-explorer read again at 6a6f0e9, 28 October 2026). `/push`, `/steer`, the push
  tab and the scientist’s `steer` write greedily by default, as the census reads a push. On request they write with the
  instrument’s own sampler (temperature, top-k, top-p, a presence penalty only with a temperature, a seed drawn and
  echoed when none is given), each step one draw shared by the three voices (an exponential race seeded seed + s), so
  the voices part only where the push moves the odds; `/steer` writes up to eight samples in one batch, and every reply
  says how it was written. **Checked on the 31B** (29 October 2026, through the server with the reference unpacker, a
  question of a forest walk, 64 word-pieces; [`atlas_steer_sampling/`](experiments/results/atlas_steer_sampling/),
  `scripts/verify_steer_sampling.py`):
  - At dose 0 the three voices wrote the same words in bf16, greedy and sampled, in one sample and in each of eight.
    Eight samples written again gave the same eight texts, and a push of 0.8 parted the voices in all eight.
  - With the reference unpacker eight samples took 50 s and one took 49 s, so the rows barely cost: that unpacker’s
    step is the unpacking’s (COMPUTE §ii). With `compiled` rows cost more (0.103 s a step at 32 rows, 0.126 s at 96).
  - Sample s of eight is not always what one sample at seed + s writes alone. Three of eight were; the other five
    parted after 90 to 245 of their 285–304 characters. Each sample draws its noise from its own generator seeded
    seed + s, the same alone or in eight, and none parted at once as a shifted seed would. So the cause is the batch’s
    size moving a row’s numbers (the writer’s floor, above), by the code, not shown in the record. To write a sample
    again, ask for the same samples with the same seed.

  The census stays greedy.
- **Not carried over:** the explorer’s HF-against-vLLM steering check as such (the atlas runs on the instrument’s one
  runtime; what it measured is measured here as the writer’s floor, above); its steered prefill captures (their idea,
  a push at every prompt word-piece rebuilt by one prefill, gives the wiring its measured check); its steer’s min-p and
  repetition penalty (`backend/sampler_request.py` says why: the penalty counts the prompt, the end-of-turn mark with
  it, and can smooth away the loops a push is read by); its neuronary.org tools (recipes for Qwen3.5-4B’s neurons,
  published as the person by a model, which the scientist’s reach rules would not allow) and its Chinese interface.
- **Why the gap layers stay shut, and exploratory atlas views** (24–25 November 2026;
  [`ANSWER.md`](experiments/results/census_layer_gap/ANSWER.md)): after `weight_space_solution_brief` investigated the
  twenty-five taken layers (31–55, board 30–54) against 1–30 and 56–60, all three owner-authorized choices stand as
  follows: (1) **Open Decision 1 kept**: the census keeps `.units.jsonl`'s 29,265 page-described neurons at layers
  31–55, and `census_fdr_sweep` (`T_dec` / `fdr_rescue`) stays out of `/arch`, `/hover`, `/census/regime/*` and every
  page; (2) **Pilot 2 (`forced_odd_kill_pilot.py`) killed fixed-text `forced_odd` as a new census for 56–60**: on 18
  unseen prompts `A'/B'/C'` at layers 46, 55, 59 and 60 (0-based 45, 54, 58, 59; `N = 256`, `C_cols = 512`, `C_iso = 512`),
  layer 60 triggered Kill Criterion 1 (`iso->col` `0.3145`, `161/512`), layer 55 triggered Kill Criterion 2a
  (equal-budget `5+5` `forced_odd` +1.45% incremental decoding accuracy over `lens.npz`, `< +3%`; `75.4%` vs `lens`
  `85.5%`), and probability-weighted `Delta p` stayed in the zero-probability tail (`p_base` p50/p90 `0.0`, top-20 entry
  `<= 1.09%`), though layer 59 retained a genuine `12.5%` carried residual (`cos(tot, dir) = 0.875`); (3) **Pilots 3a
  and 3b kept 1–30 exploratory only**: full-context natural-firing ablation (`L2, L14, L20, L29`, `N = 64` vs random
  same-layer neurons, `bf16` no-op floor `0.00000`) confirmed small real participation (`Pk_AUC` `0.6824–0.8354`, `Sp_AUC`
  `0.6853–0.8154`, `p <= 1.2e-7`, median peak KL `0.00057–0.00112` nats), held-out `<eos>` document-half forward
  precision (`1,553` pure half-1 windows) cleared `>= 0.20 & 3x` for `11.25%` (`L2`), `5.28%` (`L14`), `6.37%` (`L20`)
  and `0.39%` (`L29`) concentrated in non-top-100 lexical triggers, and Direction-3 downstream activation
  (`downstream_pilot.py`) triggered `KILL` at `L25` (`T_auc = 0.5405`, `FDP05 = 0/256`) and numerical `SURVIVE` at
  `L29` (`T_auc = 0.5948`, `FDP05 = 25/256`) driven almost solely by immediate `L30–L32` multilingual/opener hubs. The
  atlas exposes read-only exploratory views (`/api/atlas/census/regime/*`, optional `/hover` `regime` and `/arch`
  `supplementary_census`) strictly with always-visible red caveats (`exploratory: true`, `not_taken: true`, never
  marking layers 1–30 or 56–60 taken and never exposing `fdr_rescue` or `circuit_mediated`).

<br>

## vii · the CPU and GGUF

The owner’s aim (28 October 2026): every part of the instrument on the CPU and from a GGUF, not only the W4A16
checkpoint on one GPU. Two layers were built that day without a GPU and checked then on the CPU stand-in only
(`scripts/check_gguf.py`, `scripts/check_gguf_cpu.py`, ARCHITECTURE §5). The first: torch unpackers for thirteen GGUF
types, equal to gguf-py’s bit for bit; a packed GGUF layer that hears the vault; a Gemma 4 built from a GGUF with its
provenance; `GGUFRuntime` beside the pinned runtime; `make_runtime()` choosing between them from the environment; and
`scripts/compare_gguf_ct.py`. The second: every entry point loads through the factory (the server, the census, the
builds, the scientist’s MCP server run alone, the lab daemon); the lab’s gradient passes have a GGUF twin of their W4A16
autograd function; every index names the weights it was built from, the server refuses one built from others, and a
build does not write over another model’s indexes unless asked; the CPU’s memory is estimated before a sweep, a census
batch, the vault’s voice, a causal check and a steer-lens batch and refused in plain words; and `int4mm`, torch’s int4
CPU kernel, holds a Q4_0 GGUF’s or the checkpoint’s linears on the CPU (1.7 to 8 ms for a 5,376 × 21,504 layer at one
row, against 0.5 to 1.5 s for the reference, measured on one 4-vCPU container), its readings kept apart by `+int4mm` in
their model id: on the stand-in 30 of its 64 canary answers matched the reference word for word, against 64 of 64
between two batches of the reference. A third followed on 2 November 2026: IQ4_NL and IQ4_XS unpacked here (bit for
bit, as the others), the other types gguf-py reads (the rest of the IQ types, TQ1_0, TQ2_0, MXFP4, NVFP4) unpacked by it
at load into dense layers, and split GGUF files read from any of their shards, each shard hashed in the provenance;
int4mm extended to Q4_1 and Q4_K (a scale and a zero per group of 32; on the stand-in the kernel’s weight is within
rounding of the file’s, as for Q4_0); and `THEODOLITE_GGUF_RESIDENT_GB`, which holds the layers the kernel cannot take
(Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS) resident within a budget, the packed bits, named in the weights card. The study’s
registered scripts stay on the checkpoint and the card they were registered on. On the CPU the wiring check’s offline
reads ran first (29 October 2026: eight prefills of the packed checkpoint on the reference unpacker, 31–51 s each on the
home machine’s 12-core CPU; §vi, ARCHITECTURE §4.7). Gemma-4-31B then ran from a GGUF on that CPU, on 2 November 2026,
two ways. Packed: both copies of Unsloth’s UD-Q4_K_XL file here, refused before over their 24 IQ4_XS tensors, load in
bf16 (26–45 s, no lens), their IQ4_XS tensors unpacked as gguf-py unpacks them bit for bit, and a forward of six pieces
(two to three minutes beside other sessions’ work, 38.8 s with the file in the page cache and nothing else heavy
running) puts “ Paris” first after “The capital of France is”. Through int4mm: the same UD-Q4_K_XL with its IQ4_XS,
Q6_K and Q5_K layers resident in 16 GB at 1.5 s a written word-piece (experiments/results/gguf_31b_cpu), a Q4_K_M
(its 55 Q6_K layers resident in 8 GB) at 0.56 s a written word-piece and the QAT Q4_0 at 0.45 s, a 59-piece prefill in 3.5 s each,
the Q4_K_M’s last-position top five the reference’s in order (ARCHITECTURE §5, “What works now” and “The 31B from a
GGUF”). A fourth layer followed that evening: `--unpacker compiled` (and `$THEODOLITE_UNPACKER=compiled`, which the lab
daemon and the scientist’s MCP server run alone now read too) serves a GGUF’s packed layers with the reference’s bits,
on the card and the CPU (`backend/gguf_fast.py`, `scripts/check_gguf_fast.py`), and the 31B ran from the QAT Q4_0, the
Q4_K_M and the UD-Q4_K_XL on the home 5090, packed: 0.54 to 0.82 s a written word-piece on the reference unpacker and
0.12 to 0.13 s compiled, the logits of two forwards through it the reference’s bits and the pieces written alone and
at a batch of 32 the reference’s pieces, on the Q4_0 one sighting too, and the optics check passed on the Q4_0
(experiments/results/gguf_31b_gpu; ARCHITECTURE §5, “The compiled unpacker” and “The 31B on the GPU”). What is left,
in order:

- **The real file, against the checkpoint.** Run `compare_gguf_ct.py` on a GGUF of the 31B: if Google’s QAT Q4_0 holds
  the checkpoint’s codes and scales exactly, the two are one model, and the owner can decide whether an index or census
  taken on one stands for the other (today the model ids and the indexes’ weights keep them apart). Answered for
  lmstudio-community’s QAT Q4_0 (2 November 2026, experiments/results/gguf_31b_gpu): its 421 F32 norms and layer scalars
  are the checkpoint’s, but none of its 410 Q4_0 linears is (68–71% of codes equal, in none the scales all equal), so it
  is a quantisation of its own and its readings do not stand for the checkpoint’s; whether Google publishes a Q4_0 GGUF
  that is the checkpoint stays open. Then load Google’s file and take the same tiny census both ways on the card (the
  GGUF and the checkpoint). Blocked by the file (which repository Google publishes it in is unverified, and
  huggingface.co was not reachable from the machine it was built on) and, for the tokenizer, by a cached Gemma 4
  tokenizer to compare the GGUF-built one with.
- **int4mm’s canary on a real file.** `scripts/compare_int4mm.py` on the CPU, whose rule, fixed before any run, says
  whether int4mm’s readings may stand for the reference’s on a given file; it needs no checkpoint, and can run now on
  lmstudio-community’s QAT Q4_0 or Q4_K_M on the machine (its verdict that file’s, the readings still under their own
  model id). Not yet run; waits on nothing but the CPU’s time.
- **The instrument’s sweeps.** A sighting on `GGUFRuntime` is the pinned code with the GGUF accessors underneath. It
  reads 60 layers and the J++ lens, so the stand-in cannot hold it, but it has run on the card from the QAT Q4_0
  (2 November 2026): `scripts/verify_optics.py` passed in full on the reference unpacker (the 60-layer sweep, the
  lens’s mid-layer bridge, the neuron poles, the steer and the vault run on that file and their answers have the shape
  the checks expect; peak 24.15 GiB, which the script prints as GB), and the optics check’s first sighting gave every
  number but its clock and memory gauge the same twice on the reference and once with `compiled`. Not yet run: a
  sighting on the CPU, one from a K-quant file, the Q4_0’s sighting against the checkpoint’s on the same prompt (how
  far a quantisation of its own moves the readings), and the server’s pages on a GGUF.
  The runtime’s two CUDA memory calls are sidestepped by hiding the GPUs on the CPU; mending them in the file itself
  no longer waits on R4’s pin, as this entry said it did until 14 November 2026: `engine_runtime.py` left R4’s sha256 with the loader
  change of 12 November 2026 (merged on the 13th, for 24 GB cards), and a rerun of R4 goes from its registration
  commit (`2bdf5ad`, in `main`’s history). The mending is not yet made; when it is, it is a change to a runtime file,
  written down in `scripts/runtime_files.py`. Those runs wait on nothing.
- **A fast path with the reference’s bits. Built, and timed once on each device** (2 November 2026). The reference
  unpacks every weight at every forward (3 to 16 ns a weight on the 4-vCPU container, minutes a forward there; a prefill
  of the 31B took 31 to 37 s on the home machine’s 12 cores). `--unpacker compiled` now serves a GGUF’s packed layers on
  the card and the CPU (`backend/gguf_fast.py`: the reference’s arithmetic compiled whole, one graph per type and layer
  shape, tested on every layer and every fp16 pattern at load); a lookup table was no faster on the card, and is
  refused. On the home 5090 it took a written word-piece from 0.54–0.82 s to 0.12–0.13 s (three Q4 files); on the
  12-core CPU one forward of the QAT Q4_0’s six-piece prompt from 37.2 to 11.9 s, after a 64 s install (a written
  word-piece through it not timed, but each unpacks every weight again), still far from int4mm’s 0.45 s a written
  word-piece, since the product stays dense and the weight is unpacked whole each time. Even compiled, a layer’s
  unpacking (0.24–0.46 ms on the card, a probe on random blocks, not kept) costs more than its one-row product
  (0.147 ms); the rest would need a fused unpack-and-multiply kernel, which would not give the reference’s product bits.
  A small further gain may be left: a probe that read the fp16 fields as int16 and viewed them as fp16, which Inductor
  fuses, gave the same bits and took Q6_K from 0.46 to 0.28 ms a layer, but made Q4_K and Q5_K slightly slower; not
  built. Census runs, atlas builds and the server’s pages through it on a GGUF are not yet timed. (int4mm itself now
  takes Q4_1 and Q4_K as well as Q4_0, and the layers it cannot take can be held resident within a budget: above.)
- **Splitting a model between the card and the CPU** (llama.cpp’s `-ngl`). Not built, and not needed for the files
  here: the QAT Q4_0 fits the 32 GB card whole with the lens and the optics check’s short sweep (peak 24.15 GiB); the
  K files, run without the lens and about 1 to 1.2 GiB larger on the card after loading, should fit too (inferred, not
  measured); a long sweep was not tried. A file larger than the card would need it, or a larger card.
- **The atlas’s builds and the census on the CPU.** Both take a GGUF and the CPU. The lens build and the wiring are
  products of the weights (hours on a CPU, inferred); the stats and passages passes are a forward per window (days
  even through int4mm); the steer lens stays the GPU’s; a whole census layer is weeks to months (about 21 s a step of 64
  rows through int4mm, inferred), so the CPU serves pilots. A census meta now names the weights its answers were
  taken from (`weights`, beside the runtime record, 12 November 2026; §vi), so a run begun on the checkpoint is not
  continued from a GGUF. Blocked by compute.
- **The CPU’s memory.** The estimates are arithmetic set against the card’s two measurements and lean high by about
  half: a pilot on a CPU should measure a sweep’s and a census batch’s peak and recalibrate them.
- **Beyond the 31B.** Refused today: Gemma 4’s mixture of experts (26B-A4B), its per-layer embeddings (E2B, E4B), its
  shared key-value layers, and the tensor types gguf-py does not read either (Q1_0, Q8_K, Q8_1, the integer types).
  Each is a reader and a check, but the sweeps’ shapes are the 31B’s, fixed in the pinned `theodolite_math.py`, so the
  instrument’s own pages wait on that pin too. Smaller, and the 31B’s own: torch unpackers for the IQ types other than
  IQ4_NL and IQ4_XS, TQ1_0, TQ2_0, MXFP4 and NVFP4, which gguf-py unpacks at load into dense layers today (two bytes a
  weight for those layers, where packed they would hold 1.6 to 4.5 bits a weight); each is an unpacker and a check
  against gguf-py’s, as IQ4_XS’s was, and worth it only for a file with many such layers.

<br>

## viii · a weight edit read against the base

- **1a, the first slice of the weight-edit kernel. Built** (16 November 2026, `96e4420`; ARCHITECTURE §7). One edit of
  the base weights (columns of a layer’s down projection scaled, silenced, set or added to, or a low-rank term there)
  is a hashed value pinned to its base, read in another row of the same forward as the untouched model, beside null
  rows and control edits, at two row counts, and every reading, refused and void ones too, goes into a chained ledger
  outside the checkout (`backend/weight_edits.py`, `backend/levelling.py`, `backend/level_book.py`,
  `scripts/levelling.py`). It is the owner’s tool at the terminal with the server stopped. It is not a page, a route,
  an agent’s tool, a claim or a verdict: it prints numbers, ranks against unmatched control edits, floors, and “void”
  and “unstable”. Built from `experiments/results/weight_space_exploration/BUILD_SPEC_1a.md` on the pilots of 14
  November 2026. The CPU check holds it to the spec on two stand-ins; the card check ran once on the 31B on 16
  November 2026 (`experiments/results/levelling/verify_1a.json`): 142 of 145 standing lines held. Rows do not leak, a
  silencing and a scale by 2 on the input side are the folded weight by bits, a restart under hooks is exact at layers
  5, 20 and 44, writing under the hooks keeps the base’s pieces, and pilot 6’s 0.72581 came back. It failed the cost
  model (refit since; the refit is in sample, so the next card run is its test) and “no null row above a nat” (every
  such row on raw text, most of it behind the start mark). It found that the base’s log-probability does not separate
  the items that misbehaved (3 of 51) from those that did not, so the item gate is now a printed caution, not a
  refusal, by the spec’s own rule; nearly every row count is its own arithmetic (at the three lengths mapped), so the
  two row counts read (12 and 32) are two of many; and on the two chat-formatted items read, a one-row capture stood
  off the same-shape reading by more than the floor at layers 40 and 44. The bars set from it, each by a rule written
  before the run, are in the spec’s section 12 (“The card check”); the run in full is the last section of
  `experiments/results/weight_space_exploration/pilot/RESULTS.md`.
- **1b, completing the Phase 1 weight-edit kernel. Built and card-checked** (18 November 2026; ARCHITECTURE §7). Adds row
  edits on `gate_proj` and `up_proj` (exact zero/power-of-two scaling, recomputed `F.linear` on edited coordinates for
  set/add/non-exact scales to avoid cancellation, folded through `_memit_row_overlays`), masked direction removal on
  `down_proj` (`quiet_mask`, `writers`, `capture_direction`, `remove_direction` projected off the quiet mask before
  dividing by `post_feedforward_layernorm`’s weight, reporting both raw and stream-side norms), per-arm `delivered`
  stream change, three new control families (`other_sites_matched` via 1-row base pre-pass with coverage-void on
  failure, `same_site` random masked directions matched in stream-side delta length and rank-1 coherence, and
  `sham_target` declared sham edits), additive delta composition (`stack`), structural diagnostics
  (`edit_diagnostics`), ledger `driver_view` redaction (`--driver-view`), and 1a review fixes (cumulative recipe file-size
  preflight checked unread, low-rank `-0.0` merge-order fix, atomic edit-folder staging, single-pass row-0 read-out,
  `_edit_words` without dose 1.0, and signed control sign-flip reporting). All 1,032 CPU checks pass
  (`scripts/check_levelling.py`), and `scripts/verify_levelling_1b.py` passed 68 of 68 checks on the 31B card on 18
  November 2026 (`experiments/results/levelling/verify_1b.json`, 86.2 s, peak 18.61 GiB, 0 unpacker fallbacks): 24/24
  exact row edits and 11/12 recomputed row-edit class readings matched folded `_memit_row_overlays` by bits (the 12th
  0.00086 nats), `remove_direction` hit 7.21e−10 stream orthogonality with zero quiet-mask delta, and all 10 completed
  paired readings fell within 0.908–1.005 of the refit cost model’s prediction.
- **Phase 1 completion (no-card CPU weights reader + long sampled/thinking-on written readings) and Phase 2 (the
  agent’s door, practice only). Built and card-checked** (19 November 2026; ARCHITECTURE §7).
  - `backend/weight_reader.py` reads W4A16 or dense safetensors checkpoints directly from disk on CPU without touching
    CUDA (opens the 23.26 GB 31B checkpoint in 3.9 ms; 9/9 `tensor_pin` and 9/9 `dense_cols`/`dense_rows` bit-identical
    to the live GPU model; `down_stream_norms` within `5.55e−17` in 0.435 s; `writers` exact top-16 order in 0.289 s).
  - `backend/levelling.py` adds long sampled and thinking-on written readings (`sampler`, `samples` up to 32 sharing one
    exponential race per sample across rows, `thinking=True`, `max_new` up to 2,048 under the 6 GiB cache guard,
    `detect_loop`, `split_thinking`, cut-off detection, and aggregate written `summary` preserved in `driver_view`). On
    the 31B card, a 2×192 thinking-on sampled reading ran in 14.5 s (peak 18.76 GiB, exact dose-0/untouched bit
    identity) and a 512-piece thinking-on pilot ran in 56.8 s (0.1105 s/step, peak 20.54 GiB, +2.57 GiB above rest).
  - `backend/level_sandbox.py`, `backend/level_session.py`, and `backend/level_mcp.py` deliver Phase 2’s practice-only
    Agent’s Door: `bwrap` off-card CPU cells (no network, no GPU, no `$HOME`, `--remount-ro` on code and resolved
    `/weights/*`, single writable `/work`, `RLIMIT_AS` + descendant `VmRSS` cap + wall-clock kill; 9/9 active
    containment probes pass with the 23.26 GB 31B checkpoint), reach/backend enforcement (refuses `home` and unprobed
    `codex`), GPU-second/forward/cell/wall budgets, gated `capture`/`read`/`write`/`book_view` returning redacted
    `driver_view` aggregates only (no word past a rank), transcript copy on `close()`, and blinded in-memory planted
    fault & canary sessions leaving `assert_base` clean after every call.
  - All 1,032 checks in `scripts/check_levelling.py` and all 137 CPU/sandbox checks in `scripts/check_level_door.py`
    pass; `scripts/verify_level_door.py` passed 37 of 37 checks on the 31B RTX 5090 card when Phase 2 was built
    (101.0 s, peak 20.54 GiB, 0 unpacker fallbacks); the record now in
    `experiments/results/levelling/verify_door.json` is a later run of 38 checks (20 November 2026, under Phase 5).
- **Phase 3 (The Claim Gate). Built and card-checked** (19 November 2026; ARCHITECTURE §7).
  - `backend/level_briefs.py` adds versioned, canonical-hashed briefs with tamper-checked `save_brief`/`load_brief`,
    policy refusal on safeguard-removal or hidden-trigger briefs, disjoint `practice` vs `kept_back` frame generators
    balanced across `half = 0` and `half = 1` (`>= 2` frames/half), fixed fresh-reading budgets `m` (`20m - 1`
    controls/family), chat-formatted retention batteries (`neutral_chat_generator`, `draw_labelled_sanity_panel`), and
    four starter briefs (`pronoun_pull`, `spider_legs_bleed`, `canary_null`, `enumeration_loop`).
  - `backend/level_claim.py` implements control-calibrated hashed `BarSet`s (`calibrate_bars`, `save_bars`/`load_bars`),
    ledger-persisted practice-draw blocking (`record_practice_draw`), post-freeze `os.urandom(32)` draws registered in
    the level book before sampling, read-once per `(brief_id, edit_id)` and fixed-`m` enforcement, dual batch classes
    (`small`/`large`, plus chunked written execution for `written_loop`), three control families (`other_sites_matched`,
    `same_site`, optional `sham_target`) gated by `p_fw <= alpha`, `house_bar` (`sqrt(F * M)`, dropped when `M < 4F`),
    frame-level 95% CIs and half-split replication, reversed-dose checks, gating retention batteries, author prompt arms,
    Row 21 `written_strata_summary`, `price_written_reading`, `compare_settings_audit`, and harness-authored Claim Sheets
    (`lead`, `lead, matched by a cheaper knob`, `no better than control edits`, `works but harms`, `did not replicate`,
    `kept the base`, `void`). Wired into `level_session.py`, `level_mcp.py` (`brief_info`, `draw_practice`, `claim`
    returning redacted `driver_record` only), and `scripts/levelling.py` (`brief`, `calibrate`, `claim`).
  - All 302 CPU checks in `scripts/check_level_claim.py` pass (including all planted-failure verdicts, 200 planted nulls
    calibration with 0/200 false leads at `m = 2, C = 39`, live paired and chunked `written_loop` stand-in claims, MCP
    & CLI tests, and zero kept-out words across all 10 Levelling files and rendered sheets). On the 31B RTX 5090 card,
    `scripts/verify_level_claim.py` passed 16 of 16 checks (`experiments/results/levelling/verify_claim.json`, 138.6 s,
    peak 18.71 GiB, 0 unpacker fallbacks, ledger whole with 20 entries): `pronoun_pull` calibration in 10.3 s; full
    `C = 39` fresh claim #1 on E6 layer-50 top-10 `'she'` neurons (44.8 s, `small = -1.647`, `large = -1.678`, `0/39`
    controls, `near_misses = 0.801`) returned **`works but harms`** (matching `arbiter.md` Row 20); full `C = 39`
    fresh claim #2 on the 7 `'she'`-only neurons (45.1 s, `small = -0.031`, `large = -0.011`, Row 24 negative control)
    returned **`kept the base`** with reading #1 printed on its Claim Sheet; read-once and `m = 2` refusals verified
    live; `compare_settings_audit` (`max_new=64`, 28.4 s) priced a full `C = 39` written claim at 26.8 min.
- **Phase 4 (The Person's Page). Built and card-checked** (19 November 2026; ARCHITECTURE §7).
  - `backend/level_routes.py` (bound into `backend/server.py` at `/api/level/*` under `local_pages_only` and `_GPU_LOCK`)
    exposes `/status`, `/estimate`, `/compare` (base, edit, and 1–19 control edits heard side by side in one forward
    plus optional two-batch-class paired next-piece table; refuses with HTTP 409 while any vault overlay is in force;
    asserts `assert_base` before and after), and `/book` + `/book/{seq}` for model-free ledger replay (live book and
    allowlisted verified 31B books).
  - `frontend/src/screens/vault/ComparePanel.tsx` (+ `compare.ts`, `compareRecorded.ts`) adds a `compare` tab to the
    Vault page (`#/vault`) with three side-by-side voices (untouched base, edit, selectable control edit), red-italic
    rubrics, a responsive two-batch-class paired metrics table, accessible form validation, and model-free ledger
    replay (with offline recorded 19 Nov 2026 31B verdicts). `Lexicon.tsx` addresses “An edit stays until you undo it”
    and adds one new Lexicon term (`a control edit`, `id="control-edit"`, beside the single new Vault tab word
    `compare`).
  - All 83 checks in `scripts/check_level_page.py` (esbuild/node + FastAPI `TestClient` + 409 guard + replay + zero
    kept-out words) and all 291 checks in `scripts/check_interface_defaults.py` pass, and `npm run build` succeeds
    cleanly. On the 31B RTX 5090 card, `scripts/verify_level_page.py` passed 11 of 11 checks
    (`experiments/results/levelling/verify_page.json`, 17.43 s, peak 18.37 GiB, 0 unpacker fallbacks, ledger whole):
    live `POST /api/level/compare` on E6 layer-51 `'she'` ten ran in 4.64 s (`small` odds `-1.853`, `large` odds
    `-1.817`, `0/3` matched controls, untouched rows identical, `base_after = True`), live Vault `DELETE` -> HTTP 409
    -> `ROLLBACK` isolation passed, and live + recorded replay verified.
  - **Mended in review, before the merge** (19 November 2026). (1) The compare tab’s offline cards carried times and
    hashes that were not their book lines’ (one card had another line’s hash), reworded gate notes and one wrong
    figure: `compareRecorded.ts` is now made from the committed `verify_claim_book.jsonl` by
    `scripts/level_recorded_cards.py`, and `scripts/check_level_page.py` checks every card against its line. (2) A
    control edit whose reading was missing or not finite stayed in the count of control edits and out of `at_least`,
    so it counted as one the edit topped and a family with every control missing passed at `p_fw = 0.05`: it now
    counts in `at_least`, a control’s mean needs a finite reading on every item read, and the gate note says how many
    gave none. (3) A claim run with its prompt arms switched off still printed “no author prompt arm matched it”:
    `cheaper_knobs["prompt_arms_state"]` now records whether they were compared, the sentence and the sheet say when
    they were not and why, and the agent’s door no longer has the switch (`level_mcp.py` refuses any argument a tool’s
    schema does not name; the owner’s `--no-prompt-arms` stays). The review’s other findings are not mended here.
  - **Mended after the merge: the review’s other seven findings, one it missed, and what two reviews of the mends
    then found** (19 November 2026).
    (1) Bars. A claim took any bar set made for its brief. It now takes only one whose hash is in a calibration note
    of its own book, made on the same weights; `load_bars` refuses a file with no hash; the calibration note
    records its settings (`mult` among them); the sheet names the calibration’s line and how many the book holds.
    (2) Design. The driver’s claim tool set the matching tolerance (unbounded), the item counts, the controls and
    its own sham edits, none of it registered. The tool now takes the edit, its dose, a forecast, the selection, its
    shams and the rule it declares for them; the rest is the gate’s or the session owner’s (`claim_design`), shown
    on the harness card and written whole into the registration. Shams are built first and must have the edit’s
    shape; in a paired reading one that does not move the stream within the matching tolerance of the edit’s counts
    against the edit (four shams a millionth of an edit’s size had passed the shape check and would have printed
    “0 of 4 moved as much”); the sheet prints how far the edit and each family’s controls moved the stream.
    (3) Kept-back frames. The four starter briefs’ are in this repository. A session’s practice gates now refuse an
    item that holds a kept-back frame’s wording, a near wording, or a retention battery’s question, and a claim is
    refused when a practice reading already in its book, entered by any door, holds one. What this is: a guard on
    wording, by words in order, with punctuation, case, spacing, invisible characters and the split between an
    item’s fields set aside. What it is not: a question put in other words passes, and so do look-alike letters;
    the sheet says so. (The review said a driver “at the folder reach” could read the frames; for Claude Code that
    was wrong, it reads only its own work folder there. They are open to Codex, to an attended session and to any
    reader of the repository.)
    (4) Budget. `m`, read-once and the memory of practised frames went by the brief’s hash and the book’s path.
    They now go by the brief’s name (case and end spaces aside); a brief changed after its first fresh reading is
    refused; a kept-back frame is read under one brief only, so a new name on the same frames is refused too;
    practised frames are known by template under any brief or id; and a reading taken outside the default level
    book is registered, printed and worded as a side-book reading. The harness still cannot see another book, and
    a new brief on new frames is a new budget: both are said, not solved.
    (5) Loops. The instrument’s loop rule was measured on written text: it calls a song’s refrain and a JSON
    schema’s boilerplate loops and misses a list collapsing to one answer under rising numbers (the “soft loop” of
    arXiv 2606.13705). That paper’s rule, written from the wording of its section 4, is now read beside it in every
    diagnosed row; a written brief names the rule it counts by, and the strata, the settings audit and the verdict
    all count by that one; the sheet prints both counts and the sentence of a written lead says a rule counted its
    loops. Neither rule has been checked against a person’s reading of this checkpoint’s answers; a loop in a form
    neither finds (“Item 4: X”, two answers taking turns) counts as none, so an edit that changes how a list is
    written can look like a cure; and the enumeration brief says its fault has not been seen here.
    (6) Cost. The first call of a session was never refused for its price, and a claim was not priced at all. Every
    call is now priced; a claim is priced before it is registered, and nothing certain to be refused (a dose the
    edit cannot be read at, an edit with no column or row term, a plan or a draw the reading cannot make) is
    registered at all, through the door or from the command line. A claim’s forwards are counted beside the
    practice forwards.
    (7) The card. The MCP server takes the card’s lock and refuses to start while the page server answers, with no
    card, or with too little of it free. The page server and the command line take no lock.
    And one the first review missed: the compare tab’s `scale` and `scale_rows` sent a recipe key the kernel
    refuses and answered HTTP 400; they work now.
    Left open after that pass, and taken up with pull request 4 (19 November 2026):
    (a) A blinded session. A plant’s layers no longer enter the price (`est_facts`), a blinded session runs every
    paired reading with `folded="never"` and `lower="whole"`, and `estimate` and a reading quote the same seconds, so
    a planted session and a canary are priced alike. That was not the whole of it: a planted reading still scaled
    its null rows at the plant’s layer when the plant sat below the edit, and its measured seconds, which the driver
    was shown, ran about 0.6% over a canary’s (on the 31B, 12 readings each taken in turn: medians 0.9443 s against
    0.9387 s with the plant at layer 40 and 0.9478 s against 0.9415 s at layer 55, the planted the slower in 12 of
    12, the ranges apart). Now the null rows are scaled at the layer the null program names, the edit’s; a blinded
    session is shown nothing measured of a reading’s cost (seconds, their ratio to the price, the peak of memory),
    is charged the price of a reading or a capture and not its measured seconds, and sees its own lines of the book
    and no other session’s (`scripts/check_level_door.py` 143). The book keeps the measured figures. Not answered: a
    driver that times its own calls.
    (b) A live replay shows its linears: `level_routes.py` takes them from the edit’s terms, the note or the book
    (and no longer fails on a book with one damaged line, which that scan first did), claim notes carry
    `edit_linears` and the edit’s recipe, and the compare route keeps the recipe in the plan.
    (c) A battery that was not run reads `not run`, on the compare tab (`batteryOutcome`) and on the claim sheet
    (`scripts/check_level_page.py` 89, `scripts/check_interface_defaults.py` 291, `npm run build` clean).
    `.gitignore` also ignores `*.safetensors` and `*.gguf` (`arbiter.md` row 32; the eleven tracked records stay
    tracked).
- **Phase 5 (of record, and out). Built, mended in review, and card-checked as a dry run** (19 and 20 November 2026;
  ARCHITECTURE §7). It came as pull request 4 from the agent that built phases 1b to 4; what follows is what it does
  after the review’s mends, after five rounds of two independent readers each going over those mends, and after
  the mends of what the fifth pair found, which the card has read and no reader has gone over yet. One finding of
  that pair is said and not mended, because it is a ruling: as the rule stands, no trial is expected to pass its
  matched control edits (“What waits on the owner”).
  - **The trial** (`backend/level_trial.py`, `scripts/levelling_trial.py`: `reworded`, `register`, `verify`, `run`).
    A trial is a fresh reading: asked what a claim is asked, when its folder is written and again when it is read,
    counted in the brief’s budget,
    one to an edit, listed on every later sheet. Where the book holds a claim reading of the edit, a trial follows
    it only if that claim read a lead. Its bars are the ones a calibration reading wrote into the book it is read
    in, on the same weights, and the calibration note must follow from that book’s own three readings
    (`level_claim.calibration_follows`). Its folder holds manifests and hashes; registration refuses what could not
    be read, and asks the kernel about every reading the trial will take, with no forward, before it writes
    anything (`preflight_trial`); the same is asked again before the trial is entered in the book, and a trial that
    is entered always gets a verdict note (an error after that is written `void` and raised again). It reads both
    batch classes against two control families (three for a declared rule), the folded form beside the per-row
    form on every item that read, a reworded set, the batteries and the prompt arms, with a checksum of every
    stored weight before and after. A matched or sham control edit counts for the edit only when, as read, it
    changed the stream at least as much as the edit did: over the prompt, over the positions that predict the
    answer and, where the brief’s metric is read at the last position, there too (a reading records the three:
    `delivered_prompt`, `delivered`, `delivered_end`). One that was quieter at any of them, or has no figure, is
    counted as moving as much; the matching factor has no part in it, and the same-site family is not asked it
    (what it delivered is printed). A trial’s matched control sets are drawn spread (one draw over all the edit’s
    terms, to share as few sites as the eligible sites allow; a site is a neuron of a layer, whatever projection
    changes it) and count as no more separate draws than the control edits found to share no site with one
    another. **As that rule stands no trial is expected to pass it**, and a registration says so in its result
    and its `RULES.md` ((al) below, and “What waits on the owner”). Registration takes a look, with the base
    alone, at how many separate draws that would be (`sizing_look`), records it in the folder and leaves a line
    in the book; the look sees the count alone. It is of record only when its
    process had asked nothing of the model before it, the checkout is clean (nothing changed, nothing untracked
    whatever git’s configuration says, no file git is told to overlook, no code where the runner imports from
    that no commit holds), its folder is, byte for byte, what the commit holds, the runner and reader are the
    files hashed at registration (every Python file of `backend/` and the trial’s two command lines), it is read
    in the level book and not a side book, the seed of its draw was drawn and not handed in, and that look did
    not say it could not pass; otherwise it is refused, or read as a dry run when asked. Verdicts `passed`, `passed, matched by a cheaper knob`, `failed`, `void`.
  - **The export** (`backend/level_export.py`, `scripts/levelling_export.py`: `lora`, `verify`, `show`, `smoke`): the
    owner’s command, on the CPU, to a folder outside every git checkout; compiled against the base served. Its
    keys say what was checked: a trial’s verdict, its kind and its brief are read from the trial’s book (the
    result file only says where to look), and with the registration the book’s readings are judged again before
    the verdict is written; the base it was compiled against is called the edit’s base for the whole of it only
    when the weights ids agree, and otherwise the key says the touched linears agree by their pins and the rest
    was not compared; a key says where it can be served (llama.cpp serves a GGUF: an adapter compiled against a
    checkpoint is exact to weights it will not be served over);
    an edit rebuilt off the model must meet the kernel’s caps that need no weights and be one a level book shows
    the kernel has read, and the key names that line. The file is shown to be the edit by arithmetic that does
    not go through its compiler, each figure compared with its own size, and a number that is not finite fails.
  - **The re-run** (`backend/level_rerun.py`, `scripts/levelling_rerun.py`: `pack`, `verify-kit`, `run`, `intake`): a
    kit asks a volunteer’s card for the home trial’s own items; the intake judges the returned book against the
    home book, both judged again by the home’s code, and sets every figure the judgement used beside the home’s
    (each item’s effect per-row and folded, each control edit’s mean as read, what the edit and each matched or
    sham control edit changed the stream by, how many separate draws the sets were counted as, the floors, the
    reversed dose, the batteries, the prompt arms), with the plan of every reading, the record of which item is which frame and half, and the
    checksum of the base. Which control edits each trial then set aside is said apart from the figures, beside
    the verdicts. A changed tracked file, or code no commit holds where its runner imports from, in the
    volunteer’s checkout, and a runner that is not the registered one, each need a leave, named in the verdict;
    other files git does not know there do not count. Whatever a bundle holds, an intake ends in words.
  - **Checks.** `scripts/check_level_trial.py` 322, `check_levelling.py` 1,056, `check_level_door.py` 150,
    `check_level_claim.py` 307, `check_level_page.py` 89, `check_lora_export.py` 76, `check_interface_defaults.py`
    291, `check_local_only.py` 1,310 (at commit `9d8c836`, the sixth round of mends). On the card, at that commit
    from a clean checkout (20 November 2026, 08:14Z to 08:26Z, one run of each script): `scripts/verify_level_trial.py`
    passed 37 of 37 (308.73 s, peak 18.542 GiB, 0 unpacker fallbacks, the book whole at 14 entries): a **dry run** of
    the E6 layer-50 ‘she’ ten (refused as it stood, for a process that had already read, a folder in no commit and a
    side book). The look at registration counted 39 of 39 and 19 of 19 control sets sharing no site (fewest eligible
    sites a neuron 42 on the sizing draw, 112 on the reworded set) and let it through. It read `failed`, its fresh
    reading `no better than control edits`, on the 15 of its 16 fresh items that read: fresh mean −1.27 (small) and
    −1.26 (large), floors 0.1173 and 0.1673, near-miss harm 0.84; **none of its 39 matched control edits was at
    least as loud as the edit at every place** (over the prompt the edit changed the stream 0.0104 and they a
    median of 0.0030, 5 of them as loud; over the answer 0.0120 and 0.0056, 0.0036 to 0.0177, 2 as loud; at the
    last position, where the metric is read, 0.0275 and 0.0032, none), so all 39 are counted as moving as much,
    where as read none moved the metric as much (the largest 0.15 against 1.27); the 39 sets shared no site (43
    or more eligible sites a neuron); 0 of 39 same-site control edits moved as much (the largest 0.37), at 0.98,
    0.99 and 0.98 of the edit’s loudness at the median; the author’s prompt arm moved the metric 7.97 with a
    near-miss of 1.31 and matches it; the folded form the same bits on 15 of 15 items in both classes; the
    reworded set −1.95 and −1.94 with none of its 19 control edits as loud as the edit at every place (at the last
    position 0.0246 against a median of 0.0032); all 15 fresh items that read and all 8 reworded ones under the
    kernel’s item gate (the base gives their forced answers −20.4 to −10.8 and −17.5 to −8.8 nats a piece); the
    checksum unchanged over 410 linears and 1,652 tensors; an adapter of 1,082,496 bytes that is the edit on the
    model and, off the model, on the cached checkpoint, its trial keys read from the book and judged again, and
    refused at a dose of 50; the same edit registered again refused at its read, and refused at registration when
    asked after the trial; and the same card re-running its own kit `accepted` with the same figures, 664 of
    them, none without a figure, no leave given. The other four: `verify_levelling_1b.py` 68 of 68 (94.3 s);
    `verify_level_door.py` 38 of 38 (103.4 s; its two written readings recorded peaks of 18.755 and 20.541 GiB;
    a planted and a canary reading replied 1.7002 s and 1.7001 s after they were asked, where the card took
    0.9093 s and 0.9041 s); `verify_level_claim.py` 16 of 16 (156.82 s; the known answer read `works but harms`
    there, −1.33 and −1.36 with a near-miss harm of 0.61); and the page’s 11 of 11 against the committed book
    (16.5 s).
    Of those four only the door’s record is kept from this run: the kernel’s, the claim’s and the page’s records
    are still those of their first runs (18 and 19 November). A first pass at `d14b487` (08:06Z to 08:11Z, records
    not kept) was stopped after the trial’s script: its dry run read `failed`, its fresh reading `kept the base`
    (the large class’s floor 0.327, too loud to set a bar by; 15 of 16 items; −1.36 and −1.38; near-miss harm
    0.77; none of 39 matched control edits as loud at every place), and three of the script’s 37 checks failed
    for expecting all 16 fresh items to read, which the script now counts. One round earlier, at `66d771a`
    (06:00Z to 06:12Z; the trial’s and the door’s records of that run were the ones kept until this round), the
    five passed (35, 68, 38, 16, 11) and the dry run read `failed`, its fresh reading `no better than control
    edits`, −1.49 and −1.51, with none of 39 as loud at every place. Before that, at `33057a7` (04:46Z to 04:58Z,
    records not kept), the same five passed (34, 68, 38, 16, 11) and the dry run read `failed` with 4 of its 39
    matched control edits within a factor of 2 of the edit’s over the answer and at the last position, the rule
    of that round (the edit 0.0242 at the last position, they a median of 0.0034). **No trial of record has been
    read on the 31B, and no kit has been run on another card.**
  - **Mended in review** (the review of pull request 4, 19 November 2026: ten findings, each shown by a probe, and
    what two independent readers added). As it arrived:
    (1) a trial took any bar set unless a flag was given: the committed trial book’s own readings, judged again,
    read `failed` under the recorded bars and `passed` under bars typed by hand. Bars now come from a calibration
    in the book the trial is registered against and read in, on the same weights, always.
    (2) Trials were in no count: after a claim spent a brief’s budget, trial after trial of one edit ran, each a
    fresh draw, and no sheet listed an earlier one (row 25 asks for that list); the claim gate’s refusals did not
    apply; a newline added to `SHA256SUMS` let one registration be read twice. A trial is now a fresh reading in
    every count, by the owner’s ruling, and shares the claim gate’s one function.
    (3) The intake compared what a bundle said of itself: one typed by hand, with a one-line book and no reading
    taken, was `accepted`, `same_bits`. It now judges the returned book.
    (4) A volunteer’s run drew its own items (the home seed was kept nowhere), so a re-run was compared with the
    home trial across two draws, and the recorded `accepted (same_bits)` was one process with one forced seed. By
    the owner’s ruling a re-run repeats the home items; the seed is kept in the home result and carried in the kit.
    (5) The sentence of a passed trial said “fresh-process”, “verified in folded weights” and “owner-written”
    whatever the run did, “no author prompt arm matched it” when none ran, and no dose. The conditions are now
    checked and a trial that does not meet them is not of record; the waiver of the folded form is gone; the
    sentence is the fresh reading’s own with the trial’s parts.
    (6) An adapter took the verdict of any trial result handed to it (the check itself gave one edit’s to
    another’s) and said `implant_flagged: false` when the check had not run, which was every export from the
    command line; the check looked at column terms alone.
    (7) Off the model the command line built edits by hand: a recipe the kernel refuses (a neuron named twice) was
    exported with that column turned to minus itself, a payload changed after saving was taken, and the plain form
    with no base named raised a `TypeError`.
    (8) A written brief’s trial was void after every reading, or, with the folded form waived, was judged with
    neither the folded form nor its registered reworded set, one of whose five uses was counted all the same. A
    written brief now has no trial, and registration says so.
    (9) `base_checksum` tested a kind its classifier never returns, hashed nothing on plain linears and did not see
    a weight written in place.
    (10) The blinded session, as under (a) above.
    Also mended: the folded tolerance (three floors, 0.46 against a house bar of 0.34 on the card’s figures) is one
    floor and never over half the bar; a reworded set’s uses are counted by each item’s wording (a note changed
    gave a set a new count); the intake takes a plain tar only and rejects odd JSON in words; the delta check of an
    adapter compared the compiler with itself; a registration folder is exactly its files; ARCHITECTURE named files
    and rules the code did not have.
    **What two reviews of those mends found, mended** (the same day; two independent readers of the mended tree,
    each finding shown by a probe).
    (a) A planted session still differed from a canary in what it refused: an edit of 16 terms that left the
    plant’s linear alone was 17 with the plant and refused, and an edit that set a column the plant had set was
    refused by that neuron’s name. The whole of a plant and an edit is now asked nothing the edit alone is not,
    and the later set stands (`weight_edits.stack(plant_first=True)`). A blinded call’s reply also waits for its
    price (1.25 times it, and 0.25 s), so what the plant costs is not in when the reply comes.
    (b) A trial certain to be refused was entered and spent a fresh reading: a dose past ±4 (which the earlier
    test let by), a reworded item the tokenizer cannot make, a result folder that cannot be written to; and an
    error after the entry left a registration with nothing after it. Asked first now, and a verdict note always.
    (c) One item that did not read made the folded form “not read” and the trial void, with that false reason.
    (d) “Of record” could be had from a checkout with untracked files that the user’s git configuration hides
    from `git status`, with files git was told to overlook (`--assume-unchanged`, `--skip-worktree`), or with a
    folder staged and not committed; six files stood for “the runner and reader”; a seed could be handed in; a
    side book counted as the book. And “read once” was said as if of the world, where it is a book’s.
    (e) A trial could follow a claim reading that had not read a lead: a second draw at one question. Refused.
    (f) A pass that one of the brief’s own prompt arms matched read `passed`. It has its own word now, in the
    verdict line, the sentence and an adapter’s keys, with the arm’s figure beside the edit’s.
    (g) A `brief_calibration` note typed into the book with bars of anyone’s choosing was a calibration.
    (h) A reworded item given another system turn, or another answer, started a new count of uses; a starter set
    under another brief’s name was “from a file”; practice on a set was found by exact wording alone.
    (i) The sheet and the sentence said things that were not so: that a claim had read the frames when none had;
    “the sentence says so” on a dry run, which has none; a driver’s prompt, in a trial, which has no driver; sham
    edits “supplied with the claim”; the large class’s figures alone where row 25 asks for both; a re-run’s
    calibration line as if it were its own book’s; “unpacker fallbacks 0” where there was no counter to read.
    (j) `RULES.md` left out the rule for one item’s gap and the matching tolerance, and could be changed by hand
    with its sums written again: it is rendered again from `trial.json` and compared.
    (k) The intake compared the items’ main-dose effects and nothing else, so control edits, the folded form and
    the batteries could read anything under “the same figures”; a returned book holding `1e999` made the ledger’s
    reader raise, and one typed by hand could raise in the judging code or ask for a list of a billion entries;
    one leave covered both a changed checkout and another runner, while the kit’s own untracked folder counted
    against an honest volunteer; reworded figures were compared within the fresh reading’s floor.
    (l) An adapter’s verdict and kind of trial came from `trial_result.json`, a file a hand can change; its
    read-back check took a file of NaN for the edit (`max(0, nan)` is 0) and a file of noughts for a small change
    (an absolute tolerance); “served base is the edit’s base: yes” was said from the pins of the touched linears;
    an edit rebuilt off the model was “kept to its id”, a hash anyone can work, and asked nothing else; the
    implant label was taken from any mapping handed in, at any dose.
    (m) `base_checksum` left out weights kept in buffers a module does not save (a GGUF linear’s blocks, an int4
    linear’s words); `implant_shape` copied the whole unembedding to the CPU (5.6 GB on the 31B) to look at one row.
    (n) Checks that could not fail: the default checkpoint (either branch passed), a trial read in another book
    (an `or` that the first refusal always met), the blinded view of a row that had no peak, and every intake,
    which was always given leave. Each now fails when the thing is wrong, and a trial of record is read whole by
    the check: a repository of its own holding the code as it stands, a level book of its own, the registration
    made by one process and committed, the trial read by another, then refused a second time. No stand-in reads
    `passed`, so that trial’s book is written again with figures that would pass, judged by the same path, and
    the sentence and sheet of record are made from it.
    **And what two reviews of that round found** (20 November 2026).
    (o) “Matched” other-site control edits are matched, neuron for neuron, on what a base pre-pass expects each
    to deliver, and the sheet, the sentence and the new `RULES.md` said “on delivered”. On the 31B’s trial as
    then recorded 13 and 13 of 39 of them (the small and the large class) delivered within a factor of 2 of the
    edit’s when read (their ratio to it 0.32 to 2.46 and 0.33 to 2.49, medians 0.47 and 0.48; this file first
    gave 13 and 14 and a median of 0.44, which that record does not hold). The words now say what is matched and how many delivered within the factor; and in a
    trial a control edit that did not is counted as moving as much, as a sham edit already was. A claim reading
    still counts them for the edit (as merged): see “What waits on the owner”.
    (p) One call still told a planted session from a canary: a scaling at the cap of 64 read at dose 1.01 was
    64.63 in the whole made of plant and edit and was refused there alone, and a scaling of a neuron the plant
    had scaled gave the plant’s own factor in the refusal (8 + 60 − 1 = 67). The cap on a factor is one more test
    the whole is not asked again. A blinded reply now waits for twice its price (readings with 39 control edits
    ran at up to 1.57 times theirs), and a blinded view does not show when its line was written.
    (q) “No author prompt arm matched it” was printed for an arm that moved the metric six times as far as the
    edit: an arm had to have a near-miss effect no larger than the edit’s in size, which a stronger arm never
    has. An arm now matches when its near-miss effect is no worse in size or as a share of its own effect, and
    the sentence gives the strongest arm’s figures whether it matched or not.
    (r) The sheet and `RULES.md` said each bar was “2.0 times the largest figure” its calibration gave. A bar is
    the larger of that and a registered least, and on the 31B’s pronoun brief all four are the least: the
    calibration moved none of them. Both now say so, with the bars.
    (s) An adapter compiled against the cached checkpoint said `served base … yes` and that its trial “applies
    to the served base”, of a base llama.cpp cannot serve. The keys are of the base it was *compiled against*,
    and a key and a warning say that over any GGUF its numbers are another quantisation’s.
    (t) Smaller: the implant label said `false` when part of an edit was out of its reach; a trial of more
    items than a reading takes (256) was entered and ended void; `RULES.md` named a check of “rows where nothing
    is edited” that is not made (the folded forward’s own repeated rows are compared); a trial followed a
    claim’s lead at any dose; a trial’s sheet still printed lines that are a claim’s alone; a reworded item with
    a word changed started a new count of uses; a battery’s line did not say how many of its items read; the
    prompt arms’ figures were not among those an intake compares; the book an off-model export takes its
    reading from was not verified.
    (u) From the reading for bugs: a control edit is gated a second time only under a plant, and was asked the
    tests on an edit’s size again there (a same-site control of an edit of more than 1,024 scaled columns was
    refused under a plant alone); an intake raised on a return whose words a file cannot hold (half of a
    surrogate pair in the volunteer’s name) and never looked at whether the home book verifies; a prompt arm too
    long to make an item of ended a trial void after every reading of it; off the model, a scaling by −64 and
    two low-rank terms that join past the cap on rank passed the caps “that need no weights”; a Python file git
    was told to ignore, at the checkout’s root or in `scripts/`, is code the runner imports and no status lists
    it (the conditions of record now look for Python no commit holds where the runner imports from); a
    registration folder under a path git quotes could never be found committed; the read-back check of an
    adapter took its columns 64 at a time, so a column’s small change was lost beside another’s large one; a
    question git could not be asked counted as a condition met; the implant label read the whole unembedding
    once for every direction of an edit. An intake also says now when a card drew other control edits from the
    same seed, which a base pre-pass on another card can do.
    **And what two reviews of the third round found** (20 November 2026: thirteen findings from the reading for
    validity and nine from the reading for bugs, some of them the same; mended at commit `69a87b6`).
    (v) What a control edit “delivered” was measured over the positions that predict the answer, and the three
    paired starter briefs read their metric one position further on, at the last. At the layer of a one-layer
    edit the stream at a position changes only by what the edited neurons do at that position, so the figure the
    third round’s rule compared said nothing of where the metric is read: on a stand-in built to show it, 8 of 8
    control edits were “within a factor of 2” while at the last position the edit’s change was 15 times their
    median. A reading now records the same figure at the last position (`delivered_end`), and where a brief’s
    metric is read there a control edit must be within the factor both over the answer and at that position to
    count for the edit. The base pre-pass that picks the stand-ins still looks only at the answer’s positions:
    see “What waits on the owner”.
    (w) Control sets were drawn each on its own, so a neuron’s stand-ins repeated across sets (39 sets of ten
    from 60 eligible sites gave a neuron 26 to 32 different ones), and the third round’s count, the smaller of
    the sets and the eligible sites, changed nothing where it mattered: with 60 eligible sites and none of 39
    sets moving as much the sheet printed 0.05, where the reader’s simulation of an effect resting on one neuron
    gave about 0.067. A plan may now ask for the spread
    draw (`controls.draw: "spread"`: a neuron takes each of its eligible sites once before any twice, and the
    sets’ order is shuffled, so they stay exchangeable); a trial’s plans ask for it; a reading’s coverage records
    the fewest different stand-ins a neuron had (`min_distinct`); and a trial counts its sets as no more separate
    draws than that. A plan that does not ask is drawn and written as it always was. The count is a bound for an
    effect that rests on one neuron, not an estimate.
    (x) Both of the third round’s rules let everything through when a figure was missing: with the edit’s own
    delivered figure absent or nought no control edit was set aside, and the count of draws was then not applied
    at all. A control edit, or the edit, with no figure is now counted as moving as much, in a claim’s sham rule
    too. And neither rule was shown by a check to change a verdict: the check now writes the book that would
    pass again with one thing changed at a time (control edits a hundredth as loud at the last position; a
    neuron with 18 different stand-ins; an edit with no figure, or one of nought; a reading that records nothing
    at the last position) and each fails, for the reason it should.
    (y) A trial that could not pass by the count of its control edits was read whole, and the edit’s one trial
    spent. The count rests on the items, which are drawn only after the trial is entered, so a look is taken at
    registration with the base alone, on a sizing draw of the same frames and on the reworded set
    (`sizing_look`, `levelling.matched_coverage`: the pre-pass and the draw of a matched reading, with no arm
    read and nothing entered). The folder and `RULES.md` record it, a trial it says could not pass is not read
    of record, and a failed family’s reason says when the count alone settled it.
    (z) A prompt arm’s match was wider in the code than in every document: within the registered bar or no worse
    than the edit’s, in size or as a share of its own effect, and an arm with no near-miss figure matched and
    printed “near-miss none”. With two arms the sentence gave the strongest arm’s figures after naming the one
    that matched. The rule is now said as it is, from one constant the sheet and `RULES.md` print
    (`level_claim.KNOB_RULE`); the arm that matched is said with its own figures, its near-miss effect beside
    the edit’s; and “not read” is said where it was not.
    (aa) An intake made a control edit’s mean `missing` when one card set it aside and the other did not, and on
    the card 67 of the 304 “figures compared” were pairs neither trial had a figure for. The means are now
    compared as read, a pair with no figure on either side is not counted as compared, and where two trials set
    aside other control edits the intake says so beside the verdicts. A return nested 3,000 deep raised
    `RecursionError`; a verdict of `NaN` was written into a file strict readers refuse; a checkout with code no
    commit holds was rejected by a check that showed only noughts, and one that could not say was accepted.
    (ab) A prompt arm too long only behind the brief’s own system turn, which is how it is read, still ended a
    trial void after every reading: the look before registration made the arm’s items without that turn. One
    helper now puts the turn together for both (`prompt_arm_system`), and the arms’ items are made again on the
    items as drawn, before any reading.
    (ac) Smaller: compiled Python and extension modules where the runner imports from (`numpy.pyc`, a `.so`)
    were not seen by the look for code no commit holds; an adapter could be written at a dose the kernel
    refuses (50); `static_caps` raised on a low-rank term that is not a pair; four places still said “matched
    on delivered”; a written claim’s sheet said its control edits’ delivered figures were “printed with each
    class”; under a plant that sets a column the pre-pass scored the planted neuron as silent (the per-row form
    multiplies its activation by nought and adds the new column after), so a matched reading that touched it
    ended for want of stand-ins where a canary read; a battery’s counts were on the sheet and not in the
    sentence; this file gave 13 and 14 of 39 for a record that reads 13 and 13; and the door’s card record had no
    date, no commit and no list of its checks.
    **And what two reviews of the fourth round found** (20 November 2026: seven findings from the reading for
    validity and ten from the reading for bugs; mended at commit `66d771a`, which no reader has gone over).
    (ad) The control edits that counted were quieter than the edit. A control edit counted when it had changed
    the stream within a band about the edit’s figure, and on the record those in the band sat at about two
    thirds of it; with such control edits the printed level is too small (in the reader’s simulation, where a
    set moves the metric by what it delivered times a random factor and the 39 are drawn again and again from
    the record’s control edits that were in the band, an edit that is nothing special topped all 39 in 8 to 9%
    of draws when that factor was normal and in 5.4 to 5.7% when it was a Student’s t of three degrees of
    freedom, where 2.5% was printed). And the band was the registered matching factor, which
    a registrant can widen, as the look’s own warning advised. A control edit now counts only when it was at
    least as loud as the edit; there is no upper side, so the factor has no part in it; and the warning no
    longer points at a wider factor.
    (ae) The count of separate draws was not a bound. It went by each neuron’s own stand-ins, and a site that
    stands in for one neuron in one set and for another in another was not seen: ten neurons sharing 60 sites
    give every neuron 39 different stand-ins, with each site in six or seven of the 39 sets, and there an edit
    that is nothing special topped all 39 in 3.2 to 11% of the reader’s simulated draws. The spread draw now
    avoids a site any kept set holds, a reading’s coverage counts the control edits that share no site with one
    another (`disjoint_sets`: 39 of 39 where the pools allow, at most 6 for those ten neurons), a trial counts
    its sets as no more draws than that, and a reading that records no count is counted as one.
    (af) The spread draw could lock at the end of a pass (two or three neurons, more sets than a pool has
    sites: 172 of 400 seeds gave 29 to 89 sets of 99 where 435 exist), which ends the reading void and spends
    the trial; the look, drawing with another seed, could not see it. After three attempts that give nothing
    new the draw takes from the whole of each pool for an attempt. The check that showed the draw now runs 120
    seeds, where it had passed on the one seed it used.
    (ag) What “over the answer and at the last position” left out: the prompt’s positions, which feed the
    read-out through the layers above (on a stand-in 12% of an edit’s move arrived from there). A reading
    records a third figure, `delivered_prompt`, and a control edit must be as loud as the edit there too. The
    figures are means over the items read; item by item they scatter more (of 16 control edits in the old band
    on the record, 3 were in it on every item).
    (ah) Every item behind the recorded verdicts is under the kernel’s item gate, and no claim’s or trial’s
    text said so: the starter briefs’ answers are forced text (a name, a verb and an object drawn into a
    frame), and on the 31B the base gives them 12 to 25 nats a piece less than certainty, 15 of the 16 fresh items of
    the dry run then on record (at `66d771a`) below everything the gate’s own caution was measured on. One look on
    the card at one item (it is in no committed book: these figures are this file’s word): the base wants to go
    on reciting the question (“Although Claire was …”) and gives the five forced pieces −26.9, −18.0, −0.6,
    −28.3 and −16.4 nats; at the last position, where the odds are read, its own next piece is “the”, “her” or
    “she”. A sheet and a sentence now say how many items were read under the gate, for the fresh items and the
    reworded set (for a battery only a trial’s sheet did then: see (ar)).
    (ai) The look at registration ran the base over fillings of kept-back frames and left no line, and the
    kernel function under it handed back the scores of the edit’s own neurons there. It now leaves one note in
    the book (`trial_look`) and gives counts alone. Its warning contradicted itself when the control sets could
    not all be drawn (it spoke of too few draws and advised more control edits); it now says that, and that
    fewer would do. It still sees the count alone: a trial of the known answer passes the look and fails at its
    control edits, and the texts say so.
    (aj) An intake’s “every figure the judgement used” compared no figure of what was delivered and no count of
    draws: a returned book whose control edits delivered three times as much was `accepted` with the same
    figures. Both are compared now (668 figures on the card, where 304 were). It also took `False` and `0.0` for
    a count of nought, and its verdict for a value nested past what Python walks was not passed through the
    writer that makes it safe to write.
    (ak) Smaller: a claim with a prompt arm too long to make an item of was registered, read five times and
    ended on an error with no verdict (a trial’s arms were asked beforehand, a claim’s were not); a prompt
    arm’s rule carried a clause, “or no worse than the edit’s”, that changes no verdict under bars that gate on
    near misses (see (at)); a package behind
    a folder that is a link under `backend/` was not seen by the look for code no commit holds; the trial’s
    rule preset was not raised when its rule changed, so an older folder was refused as “changed by hand” (it
    became `trial-2`, and the claim gate’s rule version 2); an adapter’s dose was asked in other arithmetic than
    the kernel’s; a plant’s column lengths unpacked its whole projection on every reading; three more places
    said “delivery-matched”; the door’s record carried no date, commit or checks until it was read again; and
    this file gave the door’s first run’s figures for the record now kept, and called a dry run “now on
    record” that no longer was.
    **And what two reviews of the fifth round found** (20 November 2026: seven findings from the reading for
    validity and eleven from the reading for bugs; mended at commits `d14b487` and `9d8c836`, which no reader
    has gone over. The first is said, not mended).
    (al) The fifth round’s rule is one no trial is expected to pass, and the texts named the wrong cause. A
    matched control edit counts only when it was at least as loud as the edit at every place compared. But a
    stand-in is drawn from a band on both sides of what its neuron is expected to deliver, so some of an edit’s
    control edits are quieter than it by the draw alone; each of those is counted as moving as much; and at the
    level a pass allows none of 39 (one of 99). On the record of that round 3 of the known answer’s 39 were as
    loud as it over the answer, where the matching does look (5 over the prompt, none at the last position).
    The texts had said it of one edit, blamed where the pre-pass looks, and named two remedies: a pre-pass
    where the metric is read, and control edits scaled to the edit. A pilot on the card read both (next
    bullet), and neither gives a rule that can be met within the kernel’s caps. The texts now say it of any
    trial; a registration says it in its result and its `RULES.md` (`level_trial.RULE_STANDS_SAY`), and the
    command line prints it at registration and before a read of record. The rule itself is not changed: it is
    the owner’s (“What waits on the owner”). The bold sentences had also left out that the known answer fails
    on its harm to near misses alone. And `no better than control edits`, the claim gate’s name for a family
    that did not reach the level, was carried bare by an adapter’s keys when, as read, no control edit had
    moved the metric a seventh as far as the edit: a failed family’s reason now says how many control edits
    that were read moved the metric as much, and an adapter carries the reasons beside the verdicts.
    (am) Two control edits that changed one neuron through different projections of a layer were counted as
    sharing no site: a site was a term’s own, and each term was drawn apart. This erred towards a pass (the
    reader’s figures: ten neurons changed in two projections, 39 sets counted as 39 where 14 to 24 shared no
    neuron). A site is a neuron of a layer, and the spread draw is one draw over all an edit’s terms.
    (an) The spread draw let a neuron with thousands of eligible sites use up the few another had (the reader’s
    probe: 42 among the others’ 3,000 gave fewer than 39 sets sharing no site on 59 of 200 seeds, where 42
    exist; the look, on another seed, could not see it). A position now takes first the sites no position
    with fewer could take. The count of sets sharing no site was taken one after another after the shuffle,
    so it fell as more sets were asked for (99 asked of three neurons sharing 125 sites: 27 to 34 counted
    where 41 exist; 99 is the default). The draw now says which sets it made of sites no earlier set had, and
    the coverage counts those when that is more, after checking that they share none. And a draw that had
    run out of sites looked through every pool at every pick (512 neurons sharing 8,000 sites: 65 s, now
    0.2 s).
    (ao) A trial counted a reading with no count of disjoint sets by the fewest different stand-ins it
    recorded, which is no bound; it counts it as one draw. The look’s warning said “fewer control edits would
    do” whenever enough sets were drawn, though what a pass needs is sets that share no site.
    (ap) A trial was asked what a fresh reading is asked only when it was read: a folder was written, and a
    look taken at fillings of the kept-back frames, for an edit that had already had its trial, and three
    texts said otherwise. A registration asks it too. The look’s line was entered before the folder was made,
    so a folder that could not be made left a look with no registration, and a book that could not be written
    to raised an error that was no refusal; the folder is made first, and both are refusals in words.
    (aq) Raising the claim gate’s rule version made a trial refuse every calibration taken before it, in
    words that blamed its note, while a claim took the same bars with nothing said: the note was judged
    against a bar set rebuilt under the new version. A note follows by the version its set names, and a
    sheet says which version its bars were calibrated under.
    (ar) The item gate’s count took in items that did not read and left out one that read with no finite
    figure; a claim’s batteries carried no counts, and no sentence gave a battery’s, where this file said
    “each battery”. The counts are over the items that read, and a sentence gives them for each battery.
    “Every item of the starter briefs is under the gate” was recorded for the pronoun brief alone, and the
    “one look on the card at one item” is in no committed book: both are now said as they are.
    (as) The same-site family is not asked to be as loud as the edit, and the README implied every control
    edit is. For a column edit it is built to the edit’s length in the stream (on the record it sat at 0.98
    to 0.99 of the edit’s loudness); for a row edit nothing ties it there (on a stand-in the reader found its
    median between 0.79 and 1.45 of the edit’s at the last position). A sentence now prints what it delivered
    beside the edit’s. And a claim’s sentence printed its level beside two counts that gate nothing: it now
    says that the level takes every matched control edit as a control and as a separate draw.
    (at) Smaller: a claim’s verdict note and driver record kept each control edit’s own figure over the
    prompt; an adapter’s dose was let through one part in a million past the cap, off the model, and a whole
    number too long to print raised; the reason given for dropping “or no worse than the edit’s” from a
    prompt arm’s rule was true only of bars that gate on near misses, as every calibrated set does (with a
    set that does not, which no claim or trial is read with, 79 of 8,000 inputs the reader made up turn from
    `lead, matched by a cheaper knob` to `lead`); a delivered figure near nought was compared between
    cards to within 1e-9; the trial’s rule preset is `trial-3`; and stale lines: this file still called “now
    on record” a dry run that was not, gave the door’s peak as 18.342 GiB (the allocator’s high mark at the
    script’s end, since a reading last reset it; its two written readings recorded 18.755 and 20.541 GiB),
    gave “8 to 9%” for a simulation that gave 5.4 to 5.7% under its other law, and “16 to 24 of 39” where a run
    here gives 14 to 26; ARCHITECTURE still gave the fourth round’s band for sham edits; and the message of
    commit `66d771a` says the base gives the starter items “12 to 20 nats” where its records say 11.6 to 25.2.
  - **A pilot of the two remedies** (20 November 2026, on the card, at the code of `cea5f2b`; run by another
    session of this review, in a worktree of its own. Its report, scripts and figures are **not in this
    repository**, and what follows is that report as read here, not computed again). On the known answer,
    with 16 practice items a run (15 read in the first), two runs, 39 control edits drawn spread at a factor
    of 2:
    matching where the metric is read gave 0 of 39 control edits as loud as the edit at the last position, in
    each of three forms, on both runs, and cut the sets sharing no site to between 2 and 22. The reason is one
    neuron: 5313 carries 95% and 99% of the edit’s score at the last position and is the loudest neuron of
    layer 50 there; 5 other sites come within a factor of 2 of it, and by the pre-pass scores at most 4 sets of
    ten other sites could reach the edit’s loudness there at all. Scaling each control edit’s dose up to the
    edit’s loudness gave 38 or 39 of 39 that count, but at doses with a median of 6.4 and 12.5 (up to 21),
    where the kernel’s cap is 4 and only 7 and 12 of the 39 stay within it; past dose 1 a silence turns its
    neurons round and amplifies them, and is no silence. In all 17 readings no control edit moved the log-odds
    as much as the edit (−1.97 to −2.19): at the edit’s own loudness the scaled control edits moved it a
    median of 0.07 and at most 0.40, and the 39 same-site control edits, at 0.97 to 0.99 of the edit’s
    loudness, at most 0.58 (one reading). Not read there: kept-back items, the reworded set, the folded form,
    any other edit, brief, layer or card.
  - **Left open, and known.**
    The fresh items of a trial are drawn from the brief’s kept-back frames (8 for the pronoun brief), the frames
    the claim that made the edit a lead has read: the frames are not new to the edit, and since the draw keeps out
    practice items only, some items may not be either (4 of the 15 that read in the second trial-4 trial had been
    read by its claim). The reworded set is what stands outside them. The sets written so far are the starter set
    in the source, which anyone can read, and the two that pull request 6’s builder wrote for its trial-4 trials,
    committed in their folders; a trial of record wants a set of the owner’s own, kept in a file, and a kit sends
    even that out in clear.
    The harness cannot see another book: a budget, a reworded item’s uses and “read once” are a book’s.
    A returned book is a volunteer’s word: a hash chain holds no secret. The intake shows it is whole, is this
    trial, and agrees or not with the home’s figures.
    The implant label looks only at writes in the last tenth of the layers, and has not been set against the
    ruling’s known positive (the vault’s own example, row 37).
    `ffn_gate` and `ffn_up` adapters are unread through llama.cpp, no adapter of this exporter has been served
    there, and compiling against a quantised GGUF was checked on a small F32 file only.
    Since pull request 6 the Lexicon has an entry for the page’s new words (trial of record, dry run, re-run,
    folded form, reworded set, not run): “a trial of record”.
    The trial’s command line takes the card’s lock, and since pull request 6 the page server takes it too on a
    CUDA card.
    **What “of record” does not show.** Each condition is looked at; none is a signature. A copy of the level
    book, or the book cut back to an earlier line, takes the same trial again, and so does a level book the
    environment points elsewhere (`$XDG_DATA_HOME`): the harness sees one book. Code that calls the model itself,
    before the trial, is not seen by the kernel’s count. These guard an honest owner against a mistake; they do
    not stand against one who sets out to get round them, and the sheet and `RULES.md` now say so of the book.
    Within one book, a renamed brief or a rescaled or negated copy of an edit no longer gets a second trial (the
    owner’s ruling of 22 November 2026, as the mends built it); what still does is under “What waits on the owner”.
    A trial after a claim that did not read a lead is refused. The known answer’s own claim has never read a
    lead (its harm to near misses is real), so in a book that holds its claim it has no trial, which is what a
    trial would say of it; but an edit that is a lead and whose claim drew `kept the base` or `did not replicate`
    has none there either. The rule is the builder’s, and with verdicts that turn on the draw it decides which
    edits can ever have a trial: the owner’s to change.
    **What a trial asks of its matched control edits, and why none is expected to pass it.** The base pre-pass
    that picks a neuron’s stand-ins looks at activations over the positions that predict the answer
    (`levelling._score_sites`) and takes sites within a factor, on both sides, of what the neuron is expected to
    deliver there. A trial then counts a control edit only when, as read, it was at least as loud as the edit
    over the prompt, over the answer and (for the three paired starter briefs) at the last position. Two things
    follow. By the draw alone some control edits are quieter than the edit even over the answer (on the record
    2 of the known answer’s 39 were as loud there, 5 over the prompt), and nothing matches them at the other two
    places; each quieter one is counted as moving as much, and a pass allows none of 39. And for the known
    answer the last position cannot be matched at all: there the edit changed the stream 0.0275 against 0.0120
    over the answer, its matched control edits a median of 0.0032, and by the pilot above its neuron 5313 is the
    loudest of layer 50 there. **So no trial is expected to pass, and none of the known answer can**; the two
    remedies an earlier text named (a pre-pass where the metric is read; control edits scaled to the edit) do
    not mend it within the kernel’s caps. An edit whose neurons were among the quietest of their layer would
    have louder stand-ins and might be an exception: none has been read. What the readings do show of the
    question behind the rule, whether the edit only does what a change of its size does, is under “What waits
    on the owner”.
    **The matched control edits of a claim reading** (as merged, Phase 3) are still counted for the edit
    whatever they delivered, are still drawn each set on its own, and are each counted as a draw: a `lead` there
    can rest on control edits that changed the stream a fifth as much as the edit where the metric is read, in
    sets that share sites (in a simulation of pools like the card’s, one neuron with 49 eligible sites and nine
    with 2,000 each, 14 to 26 of 39 sets drawn each on its own share none, over 400 seeds). The sheet and the
    sentence print how
    many were as loud as the edit and how many share no site, and the sentence says that its level counts
    them all. Judging a claim’s control edits as a trial’s are would, as things stand, let no claim reading
    read a lead: its matched family would fail as a trial’s does.
    “At least as loud” is a line at a ratio of 1 on figures that move with the batch class and the card by
    about a part in a hundred: a control edit that sits there counts on one card and not on another, and each
    one set aside is one more counted as moving as much (with 39 control edits at two fresh readings a brief, a
    pass needs none). An intake says when two cards set aside different ones. The three figures are sizes of a
    change, not its direction, and means over the items read.
    The count of separate draws is of control edits found to share no site (a neuron of a layer, whatever
    projection changes it): those the spread draw made of sites no earlier set had, or those found taking the
    sets one after another as they stand, whichever is more. It is never more than there are. It can be fewer
    than the most there are, where pools overlap in ways the draw’s one preference does not see (it knows which
    pools are narrower and nothing else); and it is exact only under a null in which sites carry separate
    effects.
    The look’s line in the book records, for each of its two readings, the fewest sites eligible for one neuron
    at the matching factor the registrant chose. Asked at several factors, that says where the edit’s neurons
    sit among their layer’s on fillings of the kept-back frames, with no reading of the edit. Each look leaves
    its line, so how often an edit was looked at, and at which factors, is in the book.
    The look taken at registration is taken on other fillings of the frames than the trial will draw, so it can
    err either way (on the card it found 42 eligible sites for the narrowest neuron and the trial’s own draw 43),
    and it sees the count alone: whether the control edits will be as loud as the edit is known only when the
    trial is read, which is why a trial of the known answer passes the look and fails. The card script’s check of
    the look sets the look’s own fields beside one another and does not take the look again (on the stand-in a
    check holds the look to figures worked out beforehand).
    **Every item of the pronoun brief on record is under the kernel’s item gate on the 31B.** The other three
    starter briefs’ items are built the same way and have not been recorded. Their answers are forced
    stems the base would not have written; the gate’s caution was measured on items the base gave −12.13 nats a
    piece or more, and the fresh items of the dry run on record sit at −10.8 to −20.4 (14 of the 15 that read
    below it). What that does to a reading of the odds
    after such a stem has not been measured. Whether the briefs’ items should be ones the base finds likely is
    the owner’s (on the one answer looked at, in a look that is in no committed book, beginning it with a space
    cost its first piece about 17 nats).
    No least number of items is asked of a trial or of a battery: the sheet and the sentence say how many of
    those given read. An item whose two row counts disagree is left out by the kernel: one of the 16 fresh items
    in each of the last two dry runs (a ratio of 3.08 and of 3.11 against a bar of 3.00), and the labelled
    panel’s first item in each of the three records looked at (3.01), so that panel’s figures on the card are
    of three items.
    Under a plant a matched reading’s coverage (how many sites were eligible) is the planted model’s and differs
    from a canary’s, as any figure of a planted model may; and a neuron the plant silenced has no stand-ins, so
    a matched reading of an edit on it ends void.
    `kept the base` is the claim gate’s name (Phase 3) for a reading that cleared no bar, one whose floor was too
    loud to set a bar by among them: a dry run that moved the metric by −1.29 read so. The name is the owner’s
    to change.
    The look for code no commit holds does not open `__pycache__`: a cache made to carry the present source’s
    time and size, with other code in it, would be run. It guards against a stale or stray file, not a planted
    one.
    **A planted session adds its plant’s change to the driver’s** (as since Phase 2): a driver that scales a
    neuron the plant silenced by 2 gets the base back (0 + 2 − 1 = 1), which on a checkpoint that had the plant’s
    weights it would not; a set is the one case made to stand (the driver’s). So on the plant’s own neurons a
    planted session is not “the edit on a model with those weights”, and a repair found there is a repair of the
    sum. Found in this review, not mended: it is how the door’s “exact repair” check is built.
    A blinded reply waits for twice its price, and only a call that took less than that is covered: one that
    took longer (a card under load, a plan the cost model prices low) replies when it is done, and what a plant
    costs grows with the number of arms (one composition an arm).
    A trial whose process is killed outright leaves its registration with nothing after it: later sheets list
    it as registered and never ended, and the brief’s reading is spent.
    An intake cannot tell a re-run from the home book’s own lines copied into a new book: the home book of a
    trial that is out for re-run stays unpublished until the returns are in. (The committed card record is a dry
    run’s book, and is public.)
    What an intake’s `accepted`, `same_figures` shows is chiefly that the items’ own effects agree. A control
    edit’s mean is compared within one home floor, and on the 31B those means are themselves of a floor’s size
    (on the record up to 0.15, against floors of 0.12 and 0.17), so that comparison passes almost whatever two
    cards read. Another card is also
    likely to draw other control sets altogether: a site is eligible when its score is within a band, and in
    the reader’s simulation a score that differed by a part in ten thousand left the sets the same in none of
    60 draws (at the median all 39 differed; by a part in a million, they were the same in 57 of 60). The
    control edits’ figures are then of other edits, and the intake would say the figures disagree. A delivered
    figure is compared within 5% of the home’s and never closer than 1e-5, which is a guess. No second card has
    run a kit.
    An edit rebuilt off the model is exported only when a level book holds a reading of it; that reading is the
    kernel’s own line, in a book with no signature. On the 31B the command line has no export on the model, so
    every export there is of this kind.
    The MCP door and the page’s routes do not reach the trial, the export or the re-run; a driver given a shell
    of its own could run their scripts. What keeps a driver from them is the door’s sandbox.
    A claim reading still takes a seed handed in without saying so (`run_fresh_claim`, as merged): only a trial
    marks it.
    No reading on a stand-in reads `passed`: the pass is checked by planted cases and by a real trial’s book
    written again to pass, not by a live pass.
  - **The known answer’s verdict depends on the draw** (measured on the card, 19 November 2026: the committed
    record and four further runs of `scripts/verify_level_claim.py` that day, each a fresh draw of 8 items; one
    reading each, their reports not kept). Four read the E6 ‘she’ ten as `works but harms`, its mean over frames in
    the small and the large class −1.65 and −1.68 (the committed record; its halves −1.82 and −1.47, −1.84 and
    −1.51), −1.56 and −1.54, −1.29 and −1.29, −1.27 and −1.24 (halves −1.10 and −1.44 in the small class). The
    fifth read `did not replicate`: half 0’s four frames moved −0.13 and −0.06 in the two classes against half
    1’s −2.17 and −2.13 (mean −1.15, frame interval −2.85 to +0.56), with the near-miss harm still there (0.95). So
    with four frames a half and one item a frame, the half-replication gate turns on which names and roles are
    drawn. Not looked into further: which frames or slot values were weak. It bears on how many items a claim
    should read (`claim_design`, the owner’s) and on reading any single verdict of this brief. It also means
    `scripts/verify_level_page.py`, which replays line 10 of the book that script rewrites and expects
    `works but harms`, fails one check when run straight after such a run (10 of 11); against the committed book
    it passes.
    Later the same day, in the review of pull request 4 and the mends that followed (one reading each, their
    reports not kept but for the trial’s): two more runs of that script read `kept the base` (the large class’s
    floor 0.225, over a quarter of the least effect 0.75, with a mean of −1.70 and 0 of 39 control edits moving as
    much) and `did not replicate` (half 0 at −0.12 and −0.16 against half 1 at +1.70 and +1.72 in directed terms);
    and of three readings of the same edit as a trial (16 items, two a frame), two read `works but harms` (means
    −1.41 and −1.42; −1.36 and −1.34) and the third, the dry run then on record, `kept the base` (the small class’s
    floor 0.2005). So this edit has read three ways in ten draws. The floor is the largest thing a null row did on
    any item, so it does not fall as items are added, and on this brief it sits near a quarter of the least effect:
    whether the brief’s least effect, the floor’s rule or the number of items should change is the owner’s.
    On 20 November 2026 (UTC), on the mended code, five more readings of it (one each; the card’s logs of the
    first four are not in the repository): two claim readings `works but harms` (means −1.46 and −1.46 with a
    near-miss harm of 0.67; −1.50 and −1.50 with 0.92); a dry-run trial `works but harms` (−1.49 and −1.46,
    0.84); a dry-run trial that read `no better than control edits`, once a trial counted the 20 of its 39
    matched control edits that did not deliver within a factor of 2 of the edit’s as moving as much; and the
    dry run then on record, `kept the base` (the small class’s floor 0.2414, where a quarter of the least effect
    is 0.1875; means −1.29 and −1.30). In fifteen readings: nine `works but harms`, two `did not replicate`,
    three `kept the base`, and one `no better than control edits` by the trial’s new count.
    Later on 20 November 2026, on the code of the fourth and of the fifth round (one run of each card script at
    each; the claim script’s books not kept): two more claim readings, `kept the base` (−1.37 and −1.40,
    near-miss harm 0.95) and `works but harms` (−1.01 and −0.98, 0.86), and two more dry-run trials, both `no
    better than control edits` by the count of their round (4 of 39 matched control edits within a factor of 2
    of the edit’s over the answer and at the last position, means −1.51 and −1.50; then none of 39 at least as
    loud as the edit, means −1.49 and −1.51). In nineteen readings: ten `works but harms`, four `kept the base`,
    three `no better than control edits`, two `did not replicate`.
    On the code of the sixth round (20 November 2026, 08:06Z to 08:26Z: one card pass stopped after its
    trial script, and one whole): two more dry-run trials, each judged on the 15 of its 16 fresh items that
    read, `kept the base` (the large class’s floor 0.327, too loud to set a bar by; means −1.36 and −1.38;
    near-miss harm 0.77) and `no better than control edits` (none of 39 as loud as the edit at every place;
    means −1.27 and −1.26; 0.84); and one more claim reading, `works but harms` (−1.33 and −1.36, near-miss
    harm 0.61). In twenty-two readings: eleven `works but harms`, five `kept the base`, four `no better than
    control edits`, two `did not replicate`.
- **Phase 6 (whether agents help). Built, reviewed and mended twice; no driver that is an agent has been run, so
  whether agents help is not yet measured** (20–21 November 2026; pull request whp199/theodolite#5).
  - **What it adds:**
    - an atlas reach;
    - cheaper knobs: a system prompt, decoding settings and transient steering, claimed through the weight path’s
      own gate;
    - drivers: two scripted baselines, a replay driver, an OpenAI-compatible driver, and Claude Code with the door’s
      tools only (PR 7 adds the Gemini CLI and the Antigravity CLI, below);
    - studies of arms of drivers;
    - three pieces that measure no driver yet: a blinded planted-fault benchmark, a rebuild description and a
      cold-driver page.

    `ARCHITECTURE.md` (Phase 6) says how each is built, and `docs/SCAFFOLD.md` says which parts a driver never loses.
  - **The first review.** Four readers (the knobs, the study, containment, and the words against the records; 20
    November 2026) found six blockers:
    - knob claims spent no fresh reading on the weight path;
    - three knob guards read keys that did not exist, so practice on kept-back wording and a changed brief were
      accepted;
    - a knob’s prompt skipped the kept-back guard;
    - the Claude driver could reach none of the door’s tools, so a paid run would have scored as an honest abstention;
    - the blinded benchmark’s figures were true by how it was built;
    - the card record named main’s commit.

    They also found some forty lesser defects, among them dollar caps checked only after the fact, a “local” endpoint
    found by a substring, the key following redirects, comparisons naming a winner at p = 1, a rebuild check that was
    a JSON round trip, and steer-lens words given the lens gate’s verdict. These were mended here on the owner’s word
    (`d5e6848`).
  - **The second review.** Three readers then went through those mends. They found one more blocker: sessions that
    never worked were counted as runs with no lead. They also found some twenty-five lesser defects, among them:
    - a knob and its weight-edit twin each read once;
    - a claim still reading when its driver was stopped, recorded as no submission;
    - a reply reporting zero tokens, believed;
    - a blinded case’s kind shown in its session id;
    - a cell able to learn the size and hash of any host file through a link left in `/work`, which was on `main`
      too.

    These were mended in `9f2dc6d`.
  - **Checks.** CPU at `9f2dc6d`: `check_level_eval.py` 214 of 214; levelling 1,056, door 150, claim 307, trial
    322, page 89, label gate 189, local-only 1,310, LoRA export 76, sampler mirror 6,691. Card at `9f2dc6d` (21
    November 2026): 19 of 19, with the figures in `ARCHITECTURE.md`. On `pronoun_pull` the scripted baselines found
    no lead at any reach in three sessions an arm. The knob twin of the known answer read “works but harms”, as the
    weight edit does in most draws.
- **Pull request 6 (whp199/theodolite#6), reviewed and mended** (built 21 November 2026, reviewed and mended 22
  November 2026). Built by another agent, it gave blinded sessions `propose_repair`, made the page server take the
  card’s lock, took the leading space off
  `pronoun_pull`’s answer stems and the starter reworded items, and built way (b) of “What waits on the owner”
  as rule `trial-4`, with a new Gram-preserving same-site draw; two trials of record of one edit were read under
  it (README). Its documents called the second the first passed trial of record. The review found that the pass
  exists only under trial-4 and that the documents outran the records. The mends:
  - put trial-3 back and take the Gram draw out, which had changed claim readings too, so same-site control edits
    are drawn as before;
  - make “an edit has one trial” hold by direction on any brief, the payload read with each entry at its own site
    and with a neuron’s scale one site whichever of its projections writes it (ARCHITECTURE §7, Phase 5);
  - refuse to read, verify or judge a trial-4 folder, or to export an adapter with its verdict (both folders stay
    committed as records), and name trial-4 where a sheet, a same-direction row, a sentence of record, a gate’s
    refusal, the door’s `brief_info` or `book_view`, or the Vault’s compare tab shows such a trial;
  - have a sheet and a sentence of record say when the folded tolerance was registered off the code’s defaults
    (both trial-4 trials were, at a floor multiplier of 1.5);
  - make the blinded benchmark score what the door took (ARCHITECTURE §7, Phase 6).

  The stems’ lost space moves `pronoun_pull`’s brief id from `d977dba69441` to `ea85a5b84c48` and its starter
  reworded set’s from `6e9f06d28bc9` to `6af18240e96b`: a bar set calibrated under the old id is refused, and a
  book that read `pronoun_pull` fresh under it refuses new claims of the brief (the review found neither on disk).
  The adapter the builder exported from the second trial, outside the repository, carries that trial’s verdict and
  says “earlier trials 0”; this code refuses such an export. CPU checks, at `43eb970`, and the card run, at `495bb8b`, are in
  ARCHITECTURE §7.
- **What waits on the owner.** Ruled on 19 November 2026, for Phase 5: a volunteer’s re-run repeats the home
  trial’s own items; a trial has a reworded set; a trial spends one of the brief’s fresh readings. Chosen by the
  builders there and the owner’s to change. **First, because nothing can pass until it is settled: what a trial
  asks of its control edits.** As built, a trial counts a matched or sham control edit for the edit only when it
  was at least as loud as the edit (over the prompt, over the answer and, for a metric read at the last
  position, there too; one with no figure does not count), draws its control sets spread, and counts them as no
  more separate draws than those found to share no site, where a claim reading (as merged) does none of these
  but for its sham edits. No trial is expected to pass that rule, and none of the known answer can (“Left open”,
  above). What the readings so far say of the question the rule was written to guard, whether the edit only
  does what any change of its size does: in the records looked at, no matched control edit moved the metric a
  fifth as far as the edit (on the record the largest moved it 0.15 against 1.27); none of the 39 same-site
  control edits, at 0.98 to 0.99 of the edit’s loudness at the median, moved it half as far (the largest 0.37
  on the record, 0.58 in the pass before it); and in the pilot, control edits scaled up to the edit’s loudness
  moved it at most 0.40 against 1.97 to 2.19. The ways to settle it, each with what it costs:
  (a) leave the rule: nothing passes, and the trial stays built and unread of record;
  (b) let the same-site family answer for size, and count the matched other-site family as read, with its
  loudness printed beside it. The same-site control edits are the edit’s own sites changed in random directions
  at the edit’s own length in the stream, so for a column edit they are as loud as it by how they are made. It
  would want a tolerance (they sit two or three parts in a hundred under the edit, and by the rule as written
  every one would be set aside) and a rule for row edits, where nothing ties them there. It is a change to the
  trial’s rule and none to the kernel. A `passed` would then say that other sites, quieter ones, did not do it,
  and that a change of the edit’s size at its own sites did not;
  (c) scale each other-site control edit up to the edit’s loudness (the pilot’s second way, and the one its
  report recommends of the two it read): control doses past the kernel’s cap of 4 (medians of 6.4 and 12.5
  there), each recorded, a calibration on the items read, and control edits that past dose 1 are reversed,
  amplified writes and no longer silences. It is a change to the kernel, which is on `main`;
  (d) match where the metric is read (the pilot’s first way): it was read, and does not work for this edit.
  Whichever is chosen, 39 control edits at two fresh readings a brief leave no slack: a pass needs all 39 sets
  to share no site and none to move as much. And until it is settled, whether a trial may be read of record at
  all is the owner’s too: as built it may, with a notice printed at registration and before the read.
  Asked on 20 November 2026 what a pass should ask of its control edits, the owner answered “Leave it for now”. Pull
  request 6’s builder then built way (b) as rule `trial-4` without the owner adopting it: same-site control edits
  (and sham edits) counted for the edit when at least 0.85 times as loud as it over the prompt, over the answer and
  at the last position, other-site and reworded control edits judged as read, and a Gram-preserving same-site draw
  for down-projection edits of 2 to 256 columns that are not rank 1 (which the trials’ rank-1 edit never reached).
  It registered and read two trials of record of one edit under it on 21 November 2026, with reworded sets it
  wrote and with no ruling from the owner on whether a trial may be read of record; the second read `passed` on its
  sheet and fails when the review judged it again by trial-3 (README). The design has only the owner count a trial
  in, from a reworded set of the owner’s own, and the review put to the owner whether that second trial counts as
  of record at all. On 22 November 2026 the owner ruled “trial 3”, and the mends took trial-4 and its same-site draw
  out: the rule in force is trial-3, and whether to change it is still the owner’s. The mends also kept trial-4’s
  two committed folders as records, which this code refuses to read, verify or judge, or to export an adapter with
  their verdicts. The coordinating review (a Claude Code session) had recommended (b) on 20 November, and withdrew the
  recommendation on 22 November in its review of pull
  request 6: for a rank-1 edit along a direction aligned with the metric, the same-site control edits pass by how
  they are made, and the other-site ones, judged as read, were about a sixth as loud as the edit where the metric is
  read, so a pass under (b) tests the direction, not the sites. That review names a pilot worth running before any
  rule changes: for an edit that adds to columns, as that one did, scaling each other-site control edit to the
  change the edit delivers where the metric is read is linear (unlike a silence past dose 1, under (c)).
  Then: whether a claim should judge its control
  edits as a trial does and draw its sets spread; whether the starter briefs’ items, of which every one of the
  pronoun brief’s on record is read under the kernel’s item gate on the 31B, should be ones the base finds
  likely; that a trial is not read
  of record when the look taken at its registration says it could not pass; that a trial whose fresh reading a
  cheaper knob matched reads `passed, matched by a cheaper knob` and not `failed`, and the rule for a match
  (0.95 of the effect, with a near-miss effect within the registered bar in size or as a share of its own
  effect, and an arm whose near-miss effect was not read matching on its effect alone: by it the author’s
  prompt arm matches the known answer); that a trial follows only a claim that read a lead, at its dose; that a
  trial of record is
  read in the level book; 99 control edits a family by default and 19 for the reworded set, which is judged
  alone at the level; 16 fresh items (the proposal spoke of 256); that a trial’s draw keeps out practice items
  only, so it may repeat an item an earlier fresh reading of the brief read (4 of the second trial-4 trial’s 15
  had been read by its claim); the folded tolerance, and that a registrant may set its floor multiplier, share and
  least tolerance through `register_trial` (both trial-4 trials were registered at a floor multiplier of 1.5, and
  at the default of 1 neither passes at any least effect); five uses of a reworded
  item; the tolerances of an intake (a figure within one home floor or 5% of the home figure, never under 0.02;
  a delivered figure within 5% of the home’s and never closer than 1e-5; a count of draws the same; the same
  figures to 1e-6) and its two leaves; that an adapter is exported off the model only for an edit a
  book shows the kernel has read, and with no trial behind it, saying so, and never past the kernel’s cap on a
  dose; a blinded reply’s wait of twice its
  price; that the compare tab replays a trial at all.
  Ruled on 22 November 2026 as well: “an edit has one trial” holds across briefs and up to scale. As built since
  (ARCHITECTURE §7): a trial is refused when the book holds, under any brief’s name, a trial of the edit’s
  direction or a claim of it that read no lead, a rescaled copy being one direction (as built, a negated one too, and
  a copy written with the factors another dose gives the edit, which each entry at its own site finds at the ratio
  of the doses).
  What still escapes one trial within one book: a scale written again as the add of its change, or the reverse (scaled
  entries are
  compared as factors and added ones as vectors, and no weights are read to make the two one); a copy whose payload
  is changed past a cosine of 0.9999 under both keyings (an extra site whose change counts is one way); and, as
  before, another book, a
  copy of the book, a level book `$XDG_DATA_HOME` points elsewhere, and code outside the harness. Where the gate
  cannot tell, it refuses: a set edit is told apart only with the model at hand, and a low-rank reading with no
  saved folder refuses every trial that shares its linear in that book, unless the trial edit’s own payload bounds
  the cosine below 0.9999. Left to the owner by the mends:
  - two edits are compared over every site either changes, so a copy with a negligible extra site is one
    direction: keep that widening, or compare only edits on the same sites?
  - should a scale and an add of its change be made one direction, by comparing the weight changes themselves?
  - the dose rule follows the direction across briefs, so two leads at different doses on two briefs would refuse
    every trial of it: keep that, or bind the dose rule only to claims on the trial’s own brief?
  - should read-once for claims follow the direction too (it is by edit id, on one brief)?
  - a negated copy, the opposite change, is one direction with the edit: keep that, or count only positive scales?
  - two edits are one direction when either keying finds them so. With a neuron’s scale one site, its column, its
    up-projection row and its gate row at 0 are one site, a neuron scaled at two projections is their product, and
    a silenced neuron’s other entries are dropped; past a single scale these are one model at dose 1 alone, and
    where only this keying finds two edits one direction the dose rule takes its scale, which holds at every dose
    only where a neuron's column and up-projection factors are moved between the two (a single scale moved, or two
    factors swapped), and else at dose 1 alone. So after a failed trial of a column silence at dose 0.5 a gate silence at 0.5 is refused, though at that
    dose it is another model, and after a lead of one it is let through as following it: keep that, or apply the
    neuron keying only at dose 1?
  - beside a low-rank term on a layer’s down projection the neuron keying still joins a column’s scale and an
    up-projection row’s, which there are two models; only an earlier edit can carry such a term, never a trial;
  - `RULES.md` changed in the mends’ second and third rounds without raising `TRIAL_VERSION`, so a folder
    registered by this pull request’s earlier code does not verify, and the refusal says to register it again: keep
    that, or raise the version?
  - a trial under any rule spends its direction’s one trial, a trial-4 one too though this code does not read it,
    so the owner’s book can take no further trial of edit `6789c6b18fce`’s direction: keep that, or let a trial-4
    trial spend nothing?
  For Phase 6, ruled on 20 November 2026: Claude Code gets the door’s tools and nothing else. Chosen by the builders
  in the mends, and the owner’s to change:
  - a fresh knob claim needs a steering part;
  - a knob that mixes `neuron_mul` with pushes, or puts `spare_bos` on `neuron_mul`, is refused;
  - a `neuron_mul` knob is read once with its weight-edit twin, and a trial of the edit follows the knob’s lead only
    when the knob was that arithmetic alone, at dose 1;
  - a session counts in its arm only when its driver worked the door and it ended by the driver’s choice or a budget;
  - a session whose claim the gate refused, but which worked the door, counts as a run with no lead;
  - a plan’s per-session dollar cap of 0 means what the study has left;
  - when a driver is stopped, a running reading is waited for however long it takes (a call that never ended would
    hang a study);
  - the rebuild’s 200 words count every word and number of its record.
  Chosen in the mends of pull request 6, and the owner’s to change:
  - the two trial-4 folders stay committed as records, which this code refuses to read, verify or judge; the other
    way is to take them out of the repository;
  - in the blinded benchmark, a case with no proposal recorded (the driver never called `propose_repair`, a cap
    stopped it first, or the door refused every call) is left out of both rates and tallied apart; the other way is
    to count it, a canary as not abstaining and a plant as a miss;
  - a blinded case whose driver is a stand-in (a transport that types fixed replies, a mock CLI) is scored as a
    measurement and marked `driver_stand_in`, not set apart as a plumbing check.

  Open:
  - whether a blinded session should have a way to propose a repair: pull request 6 added one (`propose_repair`)
    with no word from the owner, and none has been given; no driver that is an agent has used it in the benchmark;
  - the ruling’s two driver tests, a cold driver of two families and a rebuild in a fresh session: the sessions of
    21 November 2026 ran outside the benchmark with no committed record (ARCHITECTURE §7). The cold driver was of
    one family and did not reach the known answer, so that test has not been run as the ruling asks; a fresh
    session’s rebuild reproduced 416 figures in the builder’s scratch, and has no record in the repository;
  - how a dollar ceiling is approved (today it is the figure in the owner’s plan);
  - the first study with an agent driver, which at three sessions an arm could separate nothing.
  The ruling’s decisions (`experiments/results/weight_space_exploration/verify/arbiter.md`
  §9): who it is for first, the agent’s door or the page (0); what waits for the census and the volunteers (1); what
  each driver may do (2); the registered gate’s rule and what a stop stops (3); gradients, after pilot F (4); the public
  posture (5); budgets, and how many fresh readings a brief takes (6); the name and the words, “Levelling” and “level
  book” among them (7); the faults its section 3 found (8). The build’s own, in the spec’s section 12: whether an
  item whose two floors stand apart (on the card 11 item-and-layer pairs had them more than 3 times apart: 7 on chats
  and a passage given as an answer, 3.0 to 4.1 times, and 4 on raw text behind the start mark, 3.4 to 42.7 times; all
  but one of their 22 readings read “ok”) should read “unstable”; what to do with raw text behind a start mark, which the gate no longer refuses;
  whether 12 and 32 are the row counts to read, and whether a one-row capture should be offered on chat-formatted
  items at all; whether a control arm clear of the floor that reads far apart between row counts should still count
  (since 16 November 2026 the stability rule reads the edit’s arms alone and prints the control arms’ ratios beside
  them, as open point 5 said it would if control arms at the floor tripped it, which three did, all on items under
  the bar; on the raw recipe cut with no start mark two controls clear of the floor read 31.1 and 6.4 times apart,
  and that item now reads “ok”); and,
  from the build’s review, that a reading handed an edit object cannot be rebuilt from the ledger unless it was saved.

  For PR 7 (whp199/theodolite#7: the Gemini CLI and the Antigravity CLI, agy, as the scientist’s backends and as
  Levelling drivers), ruled on 22 November 2026: the scientist starts on Claude Code, “as before”, and agy is the
  default only within the Gemini family (“we just want agy to be default when selecting Gemini, not default for
  everything”); the agy Levelling driver is to stay (“Mend it and keep it”). Reviewed and mended on 22 November 2026;
  what a review of those mends found was mended in a second round the same day (`d2c6065`), and what a check of that
  round found in a third.
  Phase 6’s drivers above grow by two: a Gemini CLI driver, built to Claude Code’s shape with no ruling of its own,
  and an Antigravity CLI driver, refused until the owner accepts the three tools agy offers every run; both need
  prices in the plan under a dollar cap, and neither has worked the door on the 31B. CPU checks (no card run checks
  PR 7's changes), after the third round: `check_scientist.py` 341/341, `check_level_eval.py` 286/286 (22 November
  2026), and on the same tree, which holds the mended PR 6, every one of PR 6's CPU checks with the counts
  ARCHITECTURE §7 gives. On the real CLIs,
  22 November 2026: `verify_scientist_gemini_cli.py` 38 checks on Gemini CLI 0.70.0 against a stand-in API; seven
  live Antigravity chat turns at the folder reach on agy 1.5.14, none at the home reach; and four runs of
  `verify_level_agy.py` on agy 1.5.14 against a stand-in door with no model (each spends a little of the owner’s agy
  quota), by the sessions mending PR 7 and before those mends were committed (7fbe902), the last 11 of 11; it has
  not been run since. None of these live runs has a committed record. The owner’s to settle (ARCHITECTURE §4.8 and §7
  say what each means):
  - whether to accept agy’s three fixed tools (`list_resources`, `read_resource`, `manage_task`) for Levelling runs
    (`agy_fixed_tools_accepted`); until then the agy driver is refused;
  - whether to turn off the Antigravity account’s “Enable Telemetry” setting, which agy’s print runs (the driver’s)
    follow and the driver can neither set nor read; the chat’s runs pass `--disable-telemetry`, and whether the
    setting still applies to them was not checked;
  - whether to keep the pin to agy 1.5.14 (the chat and the driver refuse any other build), or to allow newer builds;
  - whether the Gemini CLI’s file tools should reach the home folder at the home reach (today they reach only the work
    folder and the chat’s own store, as a home folder among its workspace folders would make Gemini read every
    GEMINI.md under it; its shell reaches the whole machine either way);
  - whether to leave the scientist folder inside the checkout by default, where a GEMINI.md or AGENTS.md at the
    checkout’s root, or in `atlas_data`, stops every Antigravity chat (`THEODOLITE_SCIENTIST_DIR` moves it out);
  - whether a Gemini pick the server cannot run should fall back to the other Gemini backend first (today a new chat
    then starts on the first installed of Claude Code, Codex, Antigravity and the Gemini CLI);
  - whether to give `scripts/levelling_eval.py plan` price flags: without them a Gemini CLI or agy plan made there
    with a dollar cap is refused (and both drivers need one), so such a plan is written by hand;
  - whether Claude Code’s home-reach words on the page should say, as Codex’s do, that its shell has no sandbox and
    reaches the whole machine (today they say your home folder);
  - whether to keep Codex’s home listing no skill (`[skills] include_instructions = false`), which goes beyond the
    review’s finding on AGENTS.md: codex-cli 0.169.0 lists the skills of `work/.codex/skills`, `work/.agents/skills`
    and the checkout root’s `.agents/skills`, and whether a skill can still load another way (typing `$name`, say) was
    not checked (today the line is kept);
  - whether a study under a dollar cap should halt at the first Gemini CLI session whose usage is not known (today it
    charges that session the dollar cap it was handed and goes on; the Gemini CLI’s dollar cap cannot stop a run, so
    what it spent may be more);
  - whether to learn agy’s per-call MCP timeout for the door, which needs a live run with a slow stand-in tool, and then
    set one (`timeoutSeconds` in `door.md`; today none is set, and agy’s default is not known);
  - whether to remove or guard the AGENTS.md Codex still reads in its own home (`<scientist folder>/codex-home`), which
    at the folder reach no chat can write unless the scientist folder is in `/tmp` or `/dev/shm` (today it is read,
    and the documents say so);
  - whether the Claude Code and Codex chats’ readers should pass over a line longer than 16 MiB, as the Antigravity and
    Gemini CLI reader now does (today their reading stops at such a line);
  - whether the Claude Code, Codex and agy drivers should keep their own stop and failure notes whole after a long
    reply, as the Gemini CLI driver now does (`_notes_with_room`; today those notes come after the model’s text and can
    be cut at 2,000 characters; the Claude Code and agy drivers’ stop notes are also kept apart, in `stop_note`).
- **Not built** (the ruling’s §8 and the spec’s §8): attention, embedding and norm edits; a folded form for low-rank
  terms; fine-tuned LoRA adapters as a cheaper-knob arm; gradients.
