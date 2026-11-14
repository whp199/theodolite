# Roadmap

*What is still open in Theodolite, gathered from the places that already name it: the README’s fault table, the
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
31–59; [README §v](README.md#v--how-far-to-trust-it), fault 9). Everything that reads through the lens inherits it: the
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
  the `larql`/`vindex` binaries are called only for `--version` (README §v). The edit tab now opens on an edit that
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
  no longer waits on a chat’s open event stream (`--timeout-graceful-shutdown 3`, README §vi).
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
- **The README’s pictures.** The views in README §iii were taken before this round, with the reading options as first
  built and before the label gate; their captions say so. They want retaking from a live sighting.
- **References. Opened.** Section H of `REFERENCES.md` ([45] and [46], R3’s sources) was written from local files, its
  links unopened. On 12 November 2026 each was opened: both resolve, [45] now carries its title, authors and date as
  arXiv lists them, and [46]’s revision, file size and sha256 are as Hugging Face lists that file.

<br>

## vi · the atlas

The atlas (`#/atlas`; README §i and §iii, ARCHITECTURE §4.7–4.8) was built on 27 October 2026 without a GPU: every piece of it
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
  layers taken since 29 October, eleven by 11 November 2026 (40–50; layer 45 and the volunteers’ below). The steer
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
  estimated), and it is served. **It is sound from layer 39 up and a weak guide below, and the page now says which**
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
  was built; at 39–59 a line gives the neuron’s own steer words, the shared ones dropped. The scientist is told the
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
  installs, without touching the R4-pinned runtime, a lookup-table unpacker (`--unpacker lut`), the same arithmetic
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
    and must otherwise meet rules fixed before any run was seen (`census_volunteer.ACCEPT_OTHER_CLASS`).
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
       is held for the home card (below).
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
    has been taken with `match_ceil` or `soft_ceil` to its end, on any card (the home card began layer 36 on 14
    November, below); and whether the intake carries a volunteer card’s own writer floor through to the page is
    not checked.
    *The reader for a whole layer taken with them* (written 14 November 2026):
    `experiments/results/census_scaled_design/read_whole.sh <layer>` reads a whole `match_ceil` or `soft_ceil`
    layer with the reader and gates that layer’s own trial was read by, then the upper dose’s answers alone,
    gating nothing, into `census_L<NN>_whole/` beside the other layers’. The registered reader’s three files are
    the same bytes in every trial’s folder, and the folder’s hashes are checked before anything is read. Run on the
    trials themselves at layers 36, 37, 38 and 52 it gives the verdicts recorded at the time, figure for figure,
    the upper dose’s among them; it refuses a `break_rel` run (whose reader is
    `census_neighbour_trials/read_whole.sh`), a run of another layer than the one named, and a layer with no trial
    it knows of unless that trial’s folder is given. It has not yet read a whole layer; layer 36 will be the first.
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
    through the intake, and a 4090.
    *Layer 36 on the home card* (the owner’s word, 14 November: “we can offer layer 36 using your best judgement”;
    the judgement was that the home card take it, since the card was free, the layer passed one gate by a
    thousandth and a whole `match_ceil` layer is what the new reader most needs). It began at 01:07Z on 14 November
    with kit 4’s code and settings, about 13 to 17 hours. For its first twelve minutes a second process held the
    card down to 23,900 MiB free, the kit’s floor: a real run inside what a 24 GB card has. The board shows the
    layer closed, the home card taking it.
- **A pattern put to the test, without a push** (built 14 November 2026 on the stand-in, and tried on the 31B the
  same day, below; the owner’s words that day, “Start on the first two ideas from the article”, then “Pause 36, try
  now”). The census
  and the steer lens say what a push along a neuron’s write direction does. No push strength has passed a trial
  below layer 32, and the steer lens is a weak guide below 39, so thirty-two layers have no reading that has been
  checked. Goodfire’s *Neurons as an Interpretable Basis* (10 November 2026; REFERENCES [54]) reads neurons from
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
  *Not done:*
  - **A written test for a face that needs its own document.** The honest test there holds the earlier text fixed
    (the face’s own passage up to some word-pieces before the peak) and swaps the ending: one written to the
    pattern, one a near miss. Not built; today the test only says that short texts cannot judge such a face.
  - **A harder quiet passage.** The quiet passage is drawn from anywhere in the corpus, and the six faces above
    show what that lets through: a pattern can pass the blind test without being enough to fire the neuron.
    Goodfire’s control, a passage on the same subject (or one that has the pattern’s phrase and leaves the neuron
    quiet), is not built.
  - More pairs. A passages index built with more events a face (20 minutes on the card) would give a test more
    than 9 pairs, and give the scientist back what the keeping-back took.
  - A person cannot yet run either test from the page; the scientist can, and so can a script.
  - A second job the pattern leaves out is never looked for. Clustering a face’s firings by what drives them (their
    “modes”) is not built. One face in the pilot shows the gap: its three texts all fired, two of them harder than
    0.9 of its extreme, and a reader picked only 6 of its 10 pairs, so the pattern is true and is not all the
    neuron does.
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
  person (§iii). Still open: Codex in its sandbox can still read files outside its folder; another account is not told
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
  The pinned runtime’s two CUDA memory calls are sidestepped by hiding the GPUs on the CPU; mending them in the file
  itself waits until R4’s record no longer needs `engine_runtime.py` pinned (a rerun of R4 would then go from its
  registered commit). Those runs wait on nothing; the mending waits on R4’s pin.
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
