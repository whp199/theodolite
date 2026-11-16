# "Sparks of Weight-Space Programming", read in full: what the harness actually was

Research date: 14 November 2026. Angle: the primary sources only (the post and its released logs), with coverage kept separate.

## 0. What the primary sources are, and how I read them

| Source | What it is | How I read it |
|---|---|---|
| https://rikiyat.github.io/blog/weight-space-programming/ | The work itself. It is a **blog post** (Distill-style), published 2 November 2026, authors Rikiya Takehi*, Jayden Teoh*, Alex Zhang, Omar Khattab (all MIT CSAIL). Its own BibTeX says `note = {Blog post}`. A "Response to Comments" section was added 12 November 2026. | Raw HTML fetched and the whole text read (about 39,000 characters), plus the figure data files it loads (`assets/mathedit/benchmarks.json`, `trajectories.json`, `assets/gohand/gohand.json`). |
| https://github.com/RikiyaT/lm_understand-logs | "Complete logs" repository (created 29 Oct 2026; 9,813 files; logs CC BY 4.0, scripts MIT). Not a code release of the harness: it has every prompt, reply, code run, pass request, network, evaluation, plus three harness files. | Read on the web through raw file URLs, no clone: top README, `harness/codex_config.toml`, `harness/instructions.md`, `harness/sandbox.py`, all eight experiment READMEs, the Gemma follow-up README, `exp8_math/eval/RESULTS_excerpt.md`, the four `notices/`, and for the 50-pass Qwen run: first prompt, handoff prompt, several turns, `state.json`, `loop.log`, `readable/passes.md`, the agent's notebook, final report, edit manifest and audit. Also the Go and sycophancy prompts. |
| https://www.alphaxiv.org/abs/2611.weight-space-programming | alphaXiv mirror with a placeholder id. | Opened through a summarising fetcher. It lists **no arXiv id**. |
| https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming | The edited Qwen checkpoint, rebuilt from the logs by a Hugging Face engineer. | Opened through a summarising fetcher. |

**There is no arXiv paper for the original work** as far as I could find: two searches and the alphaXiv page turn up only the blog. (arXiv 2611.08070 is a different, critical paper by a different author.)

No page I opened contained instructions addressed to an AI reader. The post's FAQ does remark that "some AI models" tend to evaluate the post as if it proposed a method; that is a comment, not an instruction, and I did not act on it.

A terminology point the authors insist on (Response to Comments): Weight-Space Programming is **not a method**, it is the name of a concept (treating weights as a program to be read and written) and the post is a claim about a new *model capability*. They also say the concept "does not rule out gradients"; gradients were forbidden in the post only to draw a clean line.

---

## 1. The harness (authors' statements, from the logs README and experiment READMEs)

### 1.1 Driver model and how it was called

- "Astra" is OpenAI's `gpt-6-astra`. It is the **only** model that drove any experiment. (The Codex config's default is `gpt-6-luna` at `medium`, but every call overrode model and effort.)
- Each turn is one call of the **Codex CLI**: `codex exec` for a new conversation, `codex exec resume <thread>` to continue. The prompt goes in on stdin.
- **Reasoning effort per experiment:** `high` for Experiments 1, 2, 4; `xhigh` for 3, 5, 6, 7; `max` for 8.
- Run records show `model_context_window=872000` and `model_auto_compact_token_limit=1000000` (so Codex's own compaction never fires; the harness does its own restarts).
- **System prompt:** one neutral line, "Answer the user's request." Everything else is in the first user message.
- **All of Codex's own tools were switched off** (`shell_tool`, `unified_exec`, `code_mode`, web search, image viewing, sub-agents, plugins, browser, computer use, hooks, skills; `sandbox_mode = "read-only"`, `approval_policy = "never"`). Every turn's event stream was audited; all 1,125 turns in the release are marked tool-free.

### 1.2 How "tools" actually worked: fenced blocks, run by the authors' own loop

The model has no function-calling tools. It replies with **exactly one fenced block per turn**, and the authors' loop runs it and sends the printed output back as the next user message:

| Block | Used in | What it does |
|---|---|---|
| ```` ```python ```` | Exp. 3, 6, 7, 8 | One run in the authors' sandbox (bubblewrap, `--unshare-all`, no network, `CUDA_VISIBLE_DEVICES=""`, a `/work` folder kept between runs, time and memory limit, output truncated). |
| ```` ```forward ```` (JSON) | Exp. 8 | One forward pass of the language model, run by a separate server, not in the kernel. |
| ```` ```games ```` / ```` ```positions ```` (JSON) | Exp. 6 | One pass: play up to 128 practice games, or evaluate up to 256 positions. |
| ```` ```attention ```` (JSON list) | Exp. 7 (relaxed round) | Run both checkpoints on up to 16 self-written texts (each at most 128 tokens); only attention weights come back. |
| no block | all | Taken as the final answer (in Exp. 6 and 8 only once the pass budget is used up). |

Sandbox limits differ by experiment: Exp. 3: 600 s per run, output cut at 20,000 characters, up to 100 runs, weights preloaded as a dict `W`; Exp. 6: 900 s and 96 GB per run, output cut at 20,000 characters, up to 120 runs; Exp. 7: a **fresh Python process per run, numpy and standard library only**, 600 s, 48 GB, output cut at 15,000 characters, 60 runs (weights-only) or 20 (attention round); Exp. 8: a **persistent kernel** (variables survive, like a notebook), 900 s and 96 GB per execution, output cut at 10,000 characters, up to 300 runs.

Only the Go sandbox (`harness/sandbox.py`) is released as code. For Experiment 8 the README says the agent harness code (prompts, sandbox, forward-pass server) is not included, "the rules it enforced are stated in the prompts".

### 1.3 How the weights were presented

| Experiment | Size | Presentation |
|---|---|---|
| 1 (which game) | 17,253 weights | **All weights as text in the prompt**, rounded to 3 decimals. About 73,700 input tokens (about 4.3 tokens per weight). No tools. |
| 2 (diffusion) | 23,824 | Same: text, 3 decimals, about 99,800 input tokens. No tools, told not to compute a forward pass even mentally. |
| 4 (car repair) | 171,664 | All actor weights as text, each tensor written as integers times one scale (3 significant digits of the tensor's largest entry); 355,201 input tokens (about 2.1 tokens per weight). No tools. Edits written in a small **edit language** (index and slice assignments such as `actor0.W[6, 122] = 80`, `actor0.W[:, 122:128] = 0`). |
| 5 (MinAtar from scratch) | 20,486, all zero | Nothing to read. The reply ends with an `edits` block in the same edit language. No tools. |
| 3 (language from middle blocks) | 7.16 M (7,084,032 given) | **Not in the prompt.** A `.npz` of the four blocks, preloaded in the sandbox; the model reads it only through code it writes. The prompt gives the exact forward code. |
| 6 (Go from scratch) | 1,480,274, all zero | 14 tensors behind `load(name)` / `save(name, array)` (float32 numpy, shape-checked, finite-checked), with helpers `inp(row, col, plane)`, `mv(row, col)`, `SHAPES`, `NAMES`, `saved()`. |
| 7 (sycophancy) | Gemma 3 270M pairs | Two read-only checkpoint folders A/B (safetensors, bf16) and a numpy-only prelude with a lazy `Checkpoint.get(name)` and a pure-Python tokenizer reader. |
| 8 (math) | Qwen3.5-0.8B, 320 tensors | Files read-only in `/model`; helpers `NAMES`, `load(name)` (float32 torch tensor: current edit if any, else original), `load_original(name)`, `save_edit(name, tensor)`, `drop_edit(name)`, `edited()`, `tok`. Gradients disabled. |

So "natively reading raw weights as text" applies only up to about 172,000 weights. Beyond that, the model sees weights **only through code it writes and the summaries that code prints**.

### 1.4 The forward-pass tool in Experiment 8 (the closest thing to a reusable harness design)

From the first message of the 50-pass run (`runs/p50c/turns/t001/prompt.txt`):

- **Budget counted in passes** (10, 25 or 50). A failed request is free.
- **(a) Generation pass:** up to 512 generations (problems x samples) under the exact evaluation protocol. `"model"` is `"original"`, `"edited"` or `"both"`; `"both"` runs both weight sets on the same items (256 per model) and returns a **paired comparison**: accuracy difference with its standard error, counts of wrong-to-right and right-to-wrong. Sampling seed is fixed per problem. Returned: accuracy (overall and by source), mean length, share cut at the token limit, failure counts; full texts and grades land in `/work/gen_<k>.jsonl`. Optional activations: residual stream at requested layers (at the last prompt token, and averaged over the answer's first 8,192 tokens) and MLP `down_proj` inputs, each with the answer's grade.
- **(b) Teacher-forced pass:** up to 1,024 texts of up to 32,768 tokens, 524,288 tokens in total, no generation. Returns any of: token log-probs, top-k (k up to 50), hidden states at chosen layers, MLP activations, attention probabilities (full-attention layers only, at most 2 texts of 2,048 tokens). Results are `.npy` files with `meta.json` in `/work/fwd_<k>/`, at most 6 GB per pass.
- **`save_edit` refuses**: the final norm, the input embedding / output layer, and any edit that only multiplies a whole tensor by a constant. The prompt adds that global sharpening/flattening and forcing thinking to end early are out of scope.
- **Forbidden:** gradients or any training (no backward, autograd, optimisers, iterative fitting to a loss); running or simulating the model in the kernel; internet.

### 1.5 Prompts and scaffolding

The Experiment 8 task message (about 10,300 characters) is a methods brief, not just a task. It tells the agent to: diagnose first by looking at failure categories; **localise causally** (correlation with correct answers is confounded with easy problems; show dose-response); **measure with power** (it gives the rule of thumb SE of about 50/sqrt(n) points, tells it to use paired "both" passes on the same problems, to keep an edit only if its gain clearly exceeds its SE on problems not used to choose it, and warns that fewer cut-offs is not the same as more correct answers); plan passes; state the hypothesis of each code block in its first comment line; print compact summaries; keep `/work/notes.md` up to date; keep copies of good edits.

Two facts about that prompt matter for interpreting the headline:

1. It **names the failure** that turned out to be the fix: its in-scope example is a model that "repeats itself and never finishes", and "HOW TO WORK" lists being cut off at the token limit "often while repeating itself". The results table shows a later prompt version 4 described as "shorter, DEV/HOLDOUT, no failure example": at 10 passes that version submitted no edit.
2. An "EXPLAIN EVERY STEP" paragraph was added mid-run (from turn 52 / 16 / 29 in the three runs): 3 to 8 plain-English sentences before every block (what the last result showed, what is concluded, what next and why, what would confirm or refute). A block without an explanation was **sent back unrun** (at most twice in a row); this happened 8 times in the 50-pass run.

Mid-run "notices" from the experimenters were delivered inside turn messages and are a useful list of harness mistakes they had to correct:

- 001: the budget counts passes, not size, so use full-size passes (agents had been running tiny 32-item passes with SEs of 4 to 7 points).
- 002: attention output now works; **separate generation passes are not reproducible even with the same request and weights** (server batching changes the random numbers), so compare only inside one "both" pass; the "chats" input re-renders assistant turns and drops reasoning, so send raw text; write the exact current edit and problem set to the notebook before every pass.
- 003 (later prompt versions): submit the best edit found even if not significant, and report it honestly (earlier agents had declined to submit under their own acceptance rule).
- 004: a failure-count category had been over-counting.

Experiment 7 used "nudges": when the agent answered early the harness replied that it had used k of N runs and must test at least one untested hypothesis first; its prompt also says "Do not be lazy, and do not settle for trivial weight reading".

### 1.6 Memory across turns

- Within a conversation: the Codex thread (resumed turn by turn).
- **Handoff every 40 turns, or after a harness restart**: a fresh conversation gets the full task again, counters (passes and runs used), the list of currently edited tensors, a listing of `/work`, and the **last 8,000 characters of `/work/notes.md`**, then "Continue from here" and the last result.
- On disk: `/work` persists (notes, saved directions, backup copies of candidate edits, request files, per-pass outputs).
- In Experiment 8 the kernel persists, but a kernel restart loses variables and the agent is told so. In the 50-pass run this cost about ten turns (16 to 25) of state recovery after one restart.
- The agent's own notebook practice in the 50-pass run is worth copying: after each pass it wrote the result with numbers, then a "NEXT PASS EXACT PLAN" JSON with the exact edit formula, strength, direction file and problem ids; it archived the old notebook to `experiment_history.md` and kept `notes.md` short; it kept named backups (`best_confirmed_a2`, `candidate_L15_L19_a3`), a frozen-weights manifest with hashes, and an audit of the final edit (changed columns, relative Frobenius change, formula error, off-direction fraction).

---

## 2. The loop: read, hypothesise, test, edit, check

### 2.1 Qwen3.5-0.8B, 50 passes (run `p50c`), reconstructed from the logs

1. **Diagnose (pass 1, generation, 96 problems, original weights, with activations at layers 4/8/12/16/20/24):** 11 of 96 correct, 85 cut at 32,768 tokens, no finished wrong answers. Its very first code comment already hypothesised repetitive reasoning driven by a few late-layer MLP features (the prompt had pointed there).
2. **Find a direction (passes 2 to 3, teacher-forced, 8 hand-written texts):** four matched pairs of productive vs repetitive continuations; the difference of mean residuals at the outputs of layers 15 and 19 (hidden states 16 and 20) gives a unit "repetition direction" u per site. The first attempt was wrong because the chat template added an empty think block; it redid it with raw text.
3. **Find writers (Python on weights x activations):** for each MLP neuron, (activation difference between repetitive and productive) x (u . that neuron's `down_proj` column). This is where the weights are actually read: which columns write along u.
4. **Probe edits, each checked with a paired generation pass:**
   - zero one neuron (L19 n11): +6.2 +/- 4.3 points on 32 problems;
   - project u out of all of L19's `down_proj`, W = W0 - u(u^T W0): +6.2 +/- 4.3;
   - both layers 15 and 19: first held-out test (128 unseen problems) +24.2 +/- 4.0.
5. **Dose-response:** strength alpha 1, 2, 3 (W = W0 - alpha u(u^T W0), so alpha > 1 over-projects and reverses the component): +12.5 +/- 2.6 (256 answers), +24.6 +/- 4.0, +39.8 +/- 3.8; alpha 2 confirmed on 128 fresh problems x 2 at +34.8 +/- 3.5.
6. **Selectivity checks with cheap teacher-forced passes:** on six new problems, log-prob per token of repetitive continuations dropped (0.024 / 0.049 / 0.104 nats at alpha 1 / 2 / 4) while productive continuations barely moved; two "premature close" controls checked that the edit does not just make the model end its thinking.
7. **Negative control:** a random direction orthogonal to u, matched in Frobenius norm to the alpha-4 edit: +1.6 +/- 3.5 points.
8. **Site isolation:** layer 15 alone and layer 19 alone (teacher-forced); tested and removed two copying attention heads (L15 h7, L19 h5) as less selective.
9. **Sparsify:** restrict the edit to the 64 MLP neurons per layer that write u most (chosen from the original contrast data). In back-to-back paired passes the 64-column edit matched the full-direction edit (+26.6 +/- 4.5 vs +26.6 +/- 4.6), so it kept the sparse form and raised strength to alpha 6.
10. **Freeze, then confirm:** hashes frozen before confirmation; the last 18 passes are 64-problem paired batches on disjoint held-out practice problems: 1,152 problems, 10.6% to 60.4%, +49.8 +/- 1.5 points, 579 rescues, 5 regressions; accuracy on originally finished answers unchanged (93.9% both); median length of correct edited answers 9,172 tokens.

Final edit: `W = W0 - 6 * outer(u, mask64 * (u @ W0))` on `layers.15.mlp.down_proj.weight` and `layers.19.mlp.down_proj.weight` (each 1,024 x 3,584). 2 x 64 x 1,024 = 131,072 values; relative Frobenius change 10.6% and 8.1% of each matrix.

**What this is, in plain terms** (my reading, shared by several commentators): an activation-contrast direction, followed by directional ablation written into weights and over-driven, i.e. the abliteration / steering recipe, with unusually careful controls. The weights were read through products with an activation-derived direction, not interpreted cold.

### 2.2 Go from scratch (Exp. 6)

Budget 0: one Python run of 336 lines that *compiles a hand-designed Go heuristic program into the six layers* (liberty marking, propagation along chains, capture / rescue / self-atari / eye rules, positional preferences), then the answer. With passes: play practice games, read how it loses, change the responsible part of the construction script, keep the change only if it wins more.

### 2.3 Turns, tokens, wall-clock

The authors state turns, runs, passes and hours. Token figures below are **my sums from the released `run.json` files** (Codex reports usage cumulatively per conversation; I added the last turn of each conversation). **No dollar cost is stated anywhere I found.**

| Task | Turns | Python runs | Passes | Wall-clock | Tokens (cumulative input / output) |
|---|---|---|---|---|---|
| Exp. 1, per network | 1 | 0 | 0 | 71 to 107 s | about 73,700 / 1,679 to 1,838 (833 to 1,034 reasoning) |
| Exp. 2, per model | 1 | 0 | 0 | 71 to 124 s | about 99,800 / 1,541 to 3,149 |
| Exp. 3, per conversation | 7 to 12 | 6 to 11 (limit 100) | 0 | not stated | e.g. 483k / 27.4k for `python_r0` |
| Exp. 4 | 1 | 0 | 0 | 413 s | 355,201 / 10,134 (9,146 reasoning) |
| Exp. 5, per game | 1 | 0 | 0 | 501 to 583 s | output 20.6k to 29.0k (16.8k to 25.7k reasoning) |
| Exp. 6, budget 0 | 2 | 1 | 0 | 22 min | 41.5k / 32.4k |
| Exp. 6, budget 10 | 28 | 17 | 10 | 29 min | 1.08M / 28.4k |
| Exp. 6, budget 25 | 56 | 30 | 25 | 57 min | 2.97M / 55.9k |
| Exp. 6, budget 50 | 90 | 39 | 50 | 3 h 52 (2 h 15 a harness pause) | 8.11M (7.84M cached) / 91.5k |
| Exp. 7, attention round, per pair | 15 to 21 | 14 to 20 (limit 20) | 4 to 5 attention calls | not stated | pair 2: 1.67M input |
| Exp. 8, 10 passes | 74 | 59 | 10 | 11.6 h | 1.78M / 60.9k |
| Exp. 8, 25 passes | 86 | 45 | 25 | 8.5 h | 2.70M / 90.9k |
| Exp. 8, 50 passes | 158 | 80 | 50 (69 requests, 19 failed) | 20.3 h (4.7 h waiting for the model API) | 5.44M (5.18M cached) / 149k (46.8k reasoning) |
| Gemma follow-up, 5 / 10 / 25 / 50 passes | 20 / 33 / 73 / 117 | 13 / 20 / 44 / 63 | as budget | 0.9 / 1.2 / 2.8 / 3.9 h | not summed |

Observations from the log timestamps: an LLM turn took about 20 to 70 s even at `max` effort (average of roughly 300 reasoning tokens per turn in the 50-pass run); **wall-clock was dominated by generation passes** (up to about half an hour each for a 0.8B model with 32,768-token answers on a rented GPU node). Input tokens are large only because the whole conversation is re-read every turn, and about 95% of it is cache hits.

---

## 3. Objects, tasks, metrics, headline numbers, baselines

| # | Object | Task | Result | Baseline / comparison |
|---|---|---|---|---|
| 1 | Four PPO policies, MLP 500 -> 32 -> 32 -> 5 tanh, 17,253 weights, on the authors' own 10x10 grid Breakout, Pong, Pac-Man, Freeway (1 billion steps, best of 3 seeds) | Name the game from the weights, open answer | 4 of 4 (Frogger counted as Freeway); stated confidence 96 to 99% | None; one run each |
| 2 | Three DDPMs, MLP 160 -> 64 -> 64 -> 144 ReLU, 23,824 weights, 12x12 circles / squares / triangles | Say what it generates | 3 of 3; confidence 77 to 88%; drew every shape as an outline although the models make filled shapes | None |
| 3 | Three character-level transformers, 4 layers x 6 heads, width 384, 7.16M weights, trained on Python / romaji / English; embeddings and output layer removed | Name the training text (3 candidates) and label all 24 heads | 6 of 6 conversations right, probabilities only 0.48 to 0.62; copying (induction) heads named exactly match measured ones (Python 2.1, 3.1, 3.5; romaji 3.3; English 3.1), no false positives | Ground truth from running the full model (induction score > 0.2; a stricter 0.3 cut-off would drop three of the five heads) |
| 4 | Official Kinetix actor, 679 -> 128 -> 5x128 -> 16, 171,664 weights, PPO 1 billion steps, car-on-ramp level | Explain the failure and fix it, one reply, no tools | 0 of 256 before; 1,024 of 1,024 test episodes after; 51 edit statements; 6,073 weights changed (3.54%) | The failing PPO policy |
| 5 | MinAtar Breakout / Seaquest / Asterix, 600 -> 32 -> 32 -> 6 tanh, 20,486 weights from zero | Write all weights in one reply, no tools, no feedback | 517.4 / 53.1 / 53.2 (1,000 episodes); 815 / 803 / 637 nonzero weights | PPO 100M steps: 15.8 / 5.6 / 25.8; PPO 500M: 25.3 / 88.9 / 55.2 (best of 3 to 10 seeds) |
| 6 | 9x9 Go policy, 243 -> 512x6 -> 82 ReLU, 1,480,274 weights from zero | Write weights by code, 0 / 10 / 25 / 50 passes | Win rate vs two hidden small PPO nets (seeds 0, 2): 54.1 / 54.9% at 0 passes; 72.9 / 60.0%; 76.0 / 62.3%; 84.0 / 80.8% at 50. Head-to-head vs same-size PPO (3 seeds, 20.5M games, tuned lr): 42.6 / 26.3 / 36.3% at 0 passes; 74.2 / 70.0 / 78.5% at 50 | The authors' own PPO networks; no established Go engine |
| 7 | Gemma 3 270M IT (18 layers, width 640), three honest / sycophantic fine-tune pairs, embedding frozen | Say which is the sycophant and where the difference lives | Weights only: 0 informative answers in four attempts (P = 0.5 every time in attempts 1 to 3). With attention patterns on self-written texts: 3 of 3 (confidence 90%, 97%, 55%). Head 10.2 attends 94.3% to the question's numbers in the honest model vs 3.5% in the sycophant, and 0.75% vs 85.7% to the user's answer | Chance is 1 in 8 for 3 of 3 |
| 8 | Qwen3.5-0.8B, thinking on (24 layers of width 1,024, SwiGLU MLP width 3,584; 6 full-attention layers, 18 Gated DeltaNet layers) | "Make it better at math" by editing weights, no gradients | See table below | The unedited model under the same sampling |
| 8b | Gemma 3 270M IT, which rarely loops (8.5% of answers hit its limit) | Same | MATH-500 6.8 -> 7.0 / 7.6 / 7.7 / 7.0 at 5 / 10 / 25 / 50 passes: no improvement | Unedited |

Experiment 8, evaluated after the run with vLLM, 32,768-token limit, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5:

| Qwen3.5-0.8B | MATH-500 | GSM8K | IFEval | MATH-500 answers cut at the limit | AMC23 | Minerva | OlympiadBench | MMLU-Pro (1k) | GPQA Diamond |
|---|---|---|---|---|---|---|---|---|---|
| Unedited | 7.9 | 13.8 | 21.0 | 93.0% | 0.5 | 1.5 | 1.3 | 40.3 | 22.7 |
| 10 passes (rank-one removal from 30 matrices, layers 8 to 22, alpha 1.5) | 12.3 | 25.7 | 49.0 | 87.2% | 1.1 | 3.9 | 2.4 | 40.0 | 29.4 |
| 25 passes (zero attention head 7 of layer 15) | 14.1 | 25.1 | 39.9 | 86.4% | 1.1 | 3.5 | 3.9 | 44.7 | 30.1 |
| 50 passes (64 columns x 2 MLPs, alpha 6) | 48.1 | 67.1 | 62.3 | 33.1% | 23.4 | 25.4 | 21.8 | 45.5 | 36.6 |

One run per budget. The relationship to budget is not smooth: 10 and 25 passes barely dent the loops, and IFEval is higher at 10 passes than at 25. The Hugging Face model card reports an independent re-run of the 50-pass checkpoint at 51.0% on MATH-500 with 31.6% of answers hitting the limit.

---

## 4. What the agent could see, and what was held out

| Experiment | Visible to the agent | Hidden |
|---|---|---|
| 1, 2 | Weights, a description of inputs/outputs with no game or shape named | Answer key (open question, no option list) |
| 3 | Middle-block weights, the exact forward code, the three candidate texts | Embeddings, output layer, measured head roles |
| 4 | Weights, scene, observation layout, the failing network's result on 256 "search" episodes, one recorded episode | 1,024 separate test episodes |
| 5 | Game rules and exact MinAtar source file, I/O layout, scoring rule | Everything else; no feedback at all |
| 6 | Rules; three practice opponents (random, a REINFORCE net, a PPO net seed 1); every game record from its passes | Two PPO nets (seeds 0 and 2) used for scoring; same-size PPO nets. Note the hidden opponents are the same kind of network as the practice "strong" one, differing by seed |
| 7 | Both checkpoints under random A/B labels (and tokenizer in the later rounds); attention maps on its own texts | Which is which; training data; any output of either model |
| 8 | 7,469 GSM8K-train and 7,275 MATH-train problems with solutions, 714 AIME answers; the full evaluation protocol in words (benchmark names MATH-500 and GSM8K, template, sampling settings, grading rule); every text the model generated on practice problems | MATH-500, GSM8K test, AMC23, Minerva, OlympiadBench, AIME24/25, MMLU-Pro, GPQA Diamond, IFEval ("It never saw the benchmarks") |

Held-out discipline **inside** the run was left to the agent in prompt version 3 (it built its own disjoint splits and froze the edit before confirming); prompt version 5 made a DEV / HOLDOUT split mandatory before any edit.

**Side effects on unrelated behaviour.**

- Authors: MMLU-Pro, GPQA Diamond and IFEval, never seen by the agent. All three *improved*, because the base model loops there too. There is no test under conditions where the base model does not loop (thinking off, or a short generation limit), no general-text perplexity or divergence measure, and no check of anything the fix might cost. The post argues from sparsity (0.016% of weights) that the rest of the model should be intact; that is an argument, not a measurement.
- Agent's own checks in the 50-pass run: productive-continuation log-probs barely change; premature-close controls; norm-matched random-direction control; accuracy on originally finished answers unchanged (3 rescues, 3 losses of 114). It also recorded costs: finished-wrong answers rose from 6 to 187 of 1,152, finished-without-answer from 1 to 58, and 51 cut-offs ended in whitespace degeneration.
- Experiment 7 had an unpublished causal check (`patch_score.json`: copy the components the agent named from the honest to the sycophantic model and measure the behaviour change against largest-change and random components).

---

## 5. Ablations, failures, stated limits

**What the logs show mattered (authors' data, my grouping):**

- **Forward passes are the lever at language-model scale.** Results table: 0 passes (prompt v2) made no edit; 5 passes submitted no edit (its best unsubmitted candidate scored 8.6 / 17.1); 10 and 25 gave small gains; only 50 gave the large one. In Go, 0 passes gave about 54% and 50 passes 81 to 84% against the hidden small nets.
- **Weights alone failed on a 270M language model:** four attempts, up to 30 runs per pair, answers at or near chance (P = 0.5 in every answer of attempts 1 to 3; attempt 4 was stopped with one pair leaning right and one leaning wrong). The agent found many weight differences but could not map them to behaviour. Attention patterns on chosen inputs fixed it.
- **Prompt content mattered:** v3 (names the repetition failure, 512-answer passes, paired passes) vs v4 (shorter, no failure example): at 10 passes v4 submitted nothing. Several agents refused to submit under self-imposed acceptance rules until notice 003 changed the rule.
- **Edit form mattered:** single-neuron ablations (+6.2 +/- 4.3; -0.8 +/- 3.6; four neurons -1.6 +/- 2.7) and copying-head halving / zeroing (+1.6, +0.8) did nothing reliable; a direction removed from two MLP output matrices did; strength mattered a great deal (alpha 1 about +12 to +24, alpha 6 on 64 columns about +50).
- **Controls the agent ran** (norm-matched random direction, same-layer control head, amplify-the-head to make loops worse) are part of why the result is credible.

**Failures:**

- Experiment 7 weights-only (above).
- Gemma 3 270M math: no single component behind ordinary arithmetic and algebra errors; edits stayed small (halve 64 columns; x1.5 on 4 columns; double one neuron; head x1.1) and none moved MATH-500 by more than about a point. The authors' conclusion: weight editing could remove an obstacle hiding an ability, but could not add an ability the model lacked.
- Experiment 2: right shape, wrong fill (outlines).
- Seaquest and Asterix lose to PPO at 500M steps.
- Infrastructure, visible in logs: 19 of 69 forward requests in the 50-pass run failed (mostly the generation engine failing to start); attention output did not work at first; a provider content filter refused a turn because the agent printed raw vocabulary tokens (the prompt then warns to print token ids, ASCII only, short excerpts); kernel restarts; a disk-quota pause; a 4.7-hour API outage.

**What the authors say limits it.** One sentence in "What Comes Next?": capabilities are "still limited", specifically because of inefficiencies in **context length and harnesses**. They say they are building a benchmark and a harness. In the FAQ they decline to claim it beats interpretability researchers and expect it to succeed by composing known interpretability techniques at scale with automatically checkable attempts.

**What they say about scaling to much larger models.** Nothing specific. The largest object is 0.8B parameters. No statement on quantised models, on models of tens of billions of parameters, or on compute requirements.

---

## 6. Open-weights drivers, and the form of edits that worked

**Open-weights models driving the loop.** The authors tested none; every run is `gpt-6-astra`. Their Response to Comments only links replications, one of which (Simon Willison, 3 Nov) ran Experiment 1's prompts on Qwen 3.8 27B, a Q4_K_M GGUF on a laptop: per his post, three of four games right (Pac-Man, Pong, Breakout), Freeway wrong, 46 to 65 minutes per answer, single attempts; Kimi K3.1 Pro got Breakout in under 10 minutes. He did not attempt the tool-using experiments. That is coverage, not an authors' result.

**Edit forms, by outcome:**

| Form | Where | Outcome |
|---|---|---|
| Index / slice assignments in an edit language; zero the rows and columns of a few units per layer, then wire a new gated path through them | Exp. 4 | Worked: 0% -> 100% |
| Sparse from-zero writes (about 3 to 4% of entries nonzero) | Exp. 5 | Worked |
| Whole network generated by a construction script | Exp. 6 | Worked |
| **Rank-one directional removal restricted to 64 writer columns of two MLP output matrices, over-driven (alpha 6)** | Exp. 8, 50 passes | Worked: MATH-500 7.9 -> 48.1 |
| Rank-one removal across 30 attention and MLP output matrices (layers 8 to 22), alpha 1.5 | Exp. 8, 10 passes | Small: +3.7 +/- 1.8 on its own hold-out; MATH-500 12.3 |
| Zero one attention head's output columns | Exp. 8, 25 passes | Small: +8.3 +/- 1.4 on its hold-out; MATH-500 14.1 |
| Single-neuron ablation; a handful of neurons; halving heads | Exp. 8 probes | No reliable effect |
| Column scaling (x0.5, x1.5, x2), head scaling x1.1 | Gemma follow-up | No effect |
| Final norm, embeddings / unembedding, whole-tensor scalar multiples | Exp. 8 | Refused by `save_edit` by design |

Edits were submitted as float32 copies of whole tensors, cast to bfloat16 for evaluation; the release stores them as exact diffs (sparse changed entries, or rank-one factors plus a few rounding exceptions) with sha256 of base and edited tensors and a script that rebuilds them bit for bit.

---

## 7. What coverage says (kept apart from the authors' statements)

- **Matthew Honnibal** (opened via summarising fetcher): the post's framing implies the Qwen result came from looking at raw weights; in fact only Experiments 1, 2 and 4 read raw weights (17K to 172K), and every language-model experiment used Python, with Qwen also using up to 50 batches of forward passes. He thinks the two kinds of result should not share one post.
- **Sean Goedecke** (opened via fetcher): the game identification is doable by a competent researcher with a few lines of code given labelled input channels; the Qwen fix is the steering / abliteration approach a human would use.
- **Andreas Maier** (opened via fetcher): solid = compiling hand-designed programs into weights, the single-direction loop fix, the copying-head identification; weak = reading trained networks (a few toy cases), sycophancy 3 of 3 (chance 1 in 8). Missing: other driver models, human or automated baselines, repeats, a comparison with simpler loop-stopping methods.
- **arXiv 2611.08070** (abstract read; a different author, 5 Nov): with a different frontier model on about 125,000-parameter CNNs given as text, zero-shot weight reading looked like local arithmetic plus dataset priors; permuting CIFAR-10 labels broke every class interpretation. Exploratory, mostly single runs.
- **Latent Space** (opened via fetcher): news framing; relays community objections that this is LLM-assisted interpretability and ablation, and that the 7.9% baseline is a looping artefact.
- Search snippets only, not opened and not verified: a claimed arXiv follow-up with sixteen editing trajectories across seven benchmarks, and a claimed independent audit reproducing 6 of 9 claim groups. The blog's Response to Comments also links IWIBench (arXiv 2611.08647), robot-policy, quantum-circuit and application papers; I did not open those.

---

## 8. What would have to be different on a 31B, 60-layer, 4-bit model on one 32 GB card

This section is my analysis, built on the facts above.

**Scale arithmetic.** The paper's edited matrices are 1,024 x 3,584 (3.7M entries, 14.7 MB as float32). One feed-forward matrix of the target is 5,376 x 21,504 = 115.6M entries: 462 MB as float32, about 58 MB at 4 bits. The paper's agent worked in a CPU kernel with 96 GB and `load(name)` returning a float32 copy; at 31B the full model in float32 is about 124 GB, so nothing that iterates "over all tensors in RAM" survives. The same 64-columns-in-two-layers edit would be 2 x 64 x 5,376 = 688,128 values, about 0.002% of the model.

1. **Reading is code-only, and statistics must be precomputed.** At 2 to 4 tokens per weight, one FF column (5,376 numbers) is 11k to 23k tokens; raw-text reading is confined to single rows or columns. The kernel needs lazy, memory-mapped, dequantise-on-demand access per tensor, and cached expensive results (norms, top singular vectors, direction-times-matrix products) so each agent turn stays inside a 900-second budget. In the paper the agent recomputed everything itself; that was cheap at 3.7M entries per matrix and is not at 115.6M.
2. **The loop must be activation-led.** The paper's own evidence at 270M and 0.8B is that weights alone yielded nothing (0 of 4 attempts; no edit at 0 or 5 passes). At 31B the first-class tools should be contrastive teacher-forced reads (log-probs, residuals, FF activations, attention) on agent-written texts, then weights x activation attribution to find writers. An existing atlas of causally tested neurons and an activation-intervention tool are exactly the things Astra had to rebuild from scratch each run; exposing them as queries should remove much of that early discovery work (in the 50-pass run, roughly the first 25 passes went on finding and validating the direction and its writers).
3. **"Original vs edited in one pass" cannot mean two models in memory.** A paired "both" pass held two weight sets; one 32 GB card holds one 4-bit 31B. Edits must be overlays on the frozen quantised base (rank-one factors plus a column mask, or sparse deltas) that can be applied and reverted between the two halves of a pass with identical seeds and batch composition. The paper's diff format (rank-one factors, sparse entries, sha256 of base and edited tensors) is the right storage form.
4. **4-bit weights change what an edit is.** The paper's edits are float values cast to bfloat16 (130,188 of 131,072 still differed after rounding). On a 4-bit grid, either the edit lives beside the quantised tensor as a float overlay (and is exported as an adapter), or it is re-quantised, in which case small edits round away and large ones acquire rounding error. Any claimed result has to be re-measured in the form that ships.
5. **Budget in tokens or GPU-seconds, not passes.** A Qwen generation pass could emit up to 512 x 32,768 tokens and took up to half an hour on a rented GPU node; counting passes made agents waste them on 32-item batches until told that size was free. On one consumer card with a 31B model, generation is the scarce resource by orders of magnitude: default to short generations and dense teacher-forced measurements, and charge by actual cost.
6. **Cache the baseline.** The unedited model's output for a given item, seed and decoding never changes. Memoising it halves the cost of every paired pass. This requires what the paper's server lacked: runs that reproduce exactly across passes (fixed per-item seeds *and* results independent of batching).
7. **Sweeps in one request.** The 50-pass run spent separate passes on strengths 1, 2, 3, 4 and 6, on layer-15-only and layer-19-only, and on a random control. A pass that takes a list of candidate overlays and returns a table (dose-response, site isolation, norm-matched random control) would compress about fifteen passes into two or three.
8. **Harness-held hold-out and off-target battery.** The paper left the dev / hold-out split to the agent (later mandated by prompt) and had no side-effect test that was independent of the bug being fixed. A harness should own a sealed confirmation set the agent can spend once per frozen edit, and an off-target battery (divergence on neutral text, unrelated tasks, the behaviours the atlas says the touched neurons also serve).
9. **Expect removals, not additions.** The Gemma result says edits removed an obstacle but did not add ability. A healthy 31B model has no 93% loop pathology to remove, so the realistic targets are suppress / boost / reroute edits on identified features, and the success metric should be specificity at a given effect size, not a benchmark jump.
10. **Do not bind to one driver.** The paper depended on an 872,000-token window, 40-turn handoffs and a single closed model. A durable harness keeps state on disk (notes file, plan file, named candidate edits, hashes), caps every output, tolerates restarts, and speaks a plain protocol (the paper's "one fenced block per turn" needs no tool-calling support at all), so that any current or future model, open or closed, can drive it.


---

## Independent citation check

Checked 14 November 2026 by a second reader whose job was to break the findings above, not to agree with them.

**How I checked.** I fetched the blog's raw HTML and read all of its text. From the logs repository I read these files through raw file URLs: the top README, the three `harness/` files, the READMEs of Experiments 1, 2, 4, 5, 6, 7, 8 and the Gemma follow-up, `exp8_math/eval/RESULTS_excerpt.md`, notices 001 to 003, and for the 50-pass run `turns/t001/prompt.txt`, the handoff prompts of turns 41 and 121, `state.json`, `loop.log`, `readable/passes.md`, `work_notes/notes.md`, `final_answer.md`, `edit/manifest.json` and all 158 `run.json` files. I also read the 10-pass and 25-pass final answers, the Experiment 4 and Go last-turn `run.json`, and the sycophancy attention-tool prompt. The Hugging Face card, alphaXiv page, and the Willison, Honnibal, Goedecke, Maier and Latent Space pages were opened through a summarising fetcher, so for those I can vouch for the facts reported, not for exact wording. arXiv abstracts came from the arXiv API. No code was downloaded or run.

**Text addressed to an AI reader.** None of the pages contained instructions to an AI. The blog's FAQ remarks that "some AI models" judge the post as if it proposed a method; that is a comment and I did not act on it.

**Overall.** The primary-source facts in this report are accurate to an unusual degree: almost every number I recomputed matched. The errors are in a few interpretations and in three places where a takeaway says more than the sources do.

### Checks

| # | Claim | Source opened | Verdict | Note |
|---|---|---|---|---|
| 1 | The work is a blog post of 2 November 2026 by Takehi, Teoh, Zhang and Khattab, with no arXiv version | Blog HTML; arXiv API search | Supported | BibTeX says `note = {Blog post}`. An arXiv search for "weight-space programming" returns 15 papers, none by these authors. |
| 2 | gpt-6-astra, one turn at a time through the Codex CLI, every built-in tool off, a one-line system prompt, one fenced block per turn, output cut at 10,000 to 20,000 characters | Top README, `codex_config.toml`, `instructions.md`, `sandbox.py`, Experiment 8 first prompt, Experiment 7 prompt | Supported | Cuts are 10,000 (Exp. 8), 15,000 (Exp. 7) and 20,000 (Go). |
| 3 | Only the Go sandbox is released as harness code | `harness/` listing, Experiment 7 and 8 folders | Partly | The Experiment 7 sandbox prelude (`runs/attention_tool/sandbox/prelude.py`) is also released, as are Experiment 8's evaluation, checkpoint-build and edit-diff scripts. What is missing is the Experiment 8 agent loop, kernel and forward-pass server, as the report says. |
| 4 | Weights shown as text only up to 171,664 weights; 355,201 input tokens for the largest | Experiment 4 `run.json`, Experiment 1 and 2 READMEs | Supported | 355,201 input tokens, 10,134 output, 412.8 s. |
| 5 | Qwen edit: `W = W0 - 6*outer(u, mask64*(u@W0))` on layers 15 and 19, 131,072 values, 8 to 11% relative change, direction from four matched contexts; MATH-500 7.9 to 48.1 | Blog table, agent's `notes.md`, `edit/manifest.json` | Supported | Manifest gives 0.1064 and 0.0811; 130,188 entries differ after rounding to bfloat16. |
| 6 | The task prompt named the failure that became the fix and taught the method | `turns/t001/prompt.txt` | Supported | The repetition example and the 50/sqrt(n) rule are there word for word. |
| 7 | A shorter prompt without the failure example produced no submitted edit at 10 passes, so prompt content mattered | `RESULTS_excerpt.md` | Partly | See correction 3: the same table shows that prompt scoring 14.1 / 25.1 at 25 passes, the same as the longer prompt. |
| 8 | 0 and 5 passes gave no edit, 10 and 25 gave 12.3 and 14.1, 50 gave 48.1; sycophancy 0 of 4 from weights, 3 of 3 with attention | `RESULTS_excerpt.md`, blog, Experiment 7 README | Supported, with limits | One run per budget; the 0-pass row used a different prompt version (v2); the 5-pass run had a candidate worth 8.6 / 17.1 that it declined to submit. |
| 9 | The paired "both" pass, and separate generation passes not being reproducible | First prompt, notice 002 | Supported | |
| 10 | 50-pass run: 158 turns, 80 Python runs, 69 requests of which 19 failed, 20.3 hours, 4.7 hours waiting on the API | `state.json`, `loop.log`, Experiment 8 README | Supported | |
| 11 | The 19 failures were for infrastructure reasons | `readable/passes.md` | Partly | 16 were the generation engine failing to start, 2 were the attention-output bug, and 1 was the agent's own mistake (an unknown practice id). |
| 12 | Token total about 5.4M input (95% cached) and 149k output | All 158 `run.json` files | Partly | The sum of each conversation's last record is exactly 5,437,759 / 5,180,032 cached / 149,017, as stated. It undercounts: see correction 1. |
| 13 | Handoff every 40 turns with the last 8,000 characters of the notebook; named backups, plan file, hashes | Handoff prompts at turns 41 and 121, `loop.log` | Supported | New threads at turns 41, 81 and 121. |
| 14 | Gemma 3 270M: MATH-500 6.8 to at most 7.7 | Blog, Gemma README, `RESULTS_excerpt.md` | Supported | The four runs not shown in the post (strict grading stated in the prompt) scored 6.7, 6.7, 6.1 and 6.2. |
| 15 | Side-effect benchmarks were run only under the looping condition and all improved; no independent measure of collateral damage | Blog, `RESULTS_excerpt.md`, model card | Supported for the 50-pass edit | MMLU-Pro 40.3 to 45.5, GPQA 22.7 to 36.6, IFEval 21.0 to 62.3. For the 10-pass edit MMLU-Pro was 40.0. I found no thinking-off or perplexity measurement anywhere I opened. |
| 16 | Model card: re-run at 51.0%, 31.6% at the limit, 2,000 generations, 43.4M tokens, 51 minutes on an H200 | Hugging Face card | Supported | |
| 17 | Willison: Qwen 3.8 27B (Q4_K_M) named 3 of 4 games in 46 to 65 minutes each | His post | Supported | Kimi K3.1 Pro named Breakout in 9 min 41 s. |
| 18 | The run relied on an 872,000-token window | `run.json` files | Partly | See correction 6. |
| 19 | Astra spent roughly its first 25 passes rediscovering a direction and its 64 writer neurons | `passes.md`, handoff notebook | Not supported as worded | See correction 4. |
| 20 | Go: turns, times, win rates; 8.11M cumulative input tokens at budget 50 | Go README, `turns/t090/run.json` | Supported | 8,111,731 input, 7,842,048 cached. |
| 21 | Coverage: Honnibal, Goedecke, Maier, arXiv 2611.08070, Latent Space | Each page | Supported | Dates now known: Honnibal and Goedecke 3 November, Maier 5 November. arXiv 2611.08070 is by Nazmus Sakib Ahmed and tested Claude Opus 5.5. |
| 22 | Scale arithmetic for the target model (5,376 x 21,504 = 115.6M entries, 462 MB as float32) | Arithmetic; the project's own inspection results | Supported | The project's recorded config shows width 5,376, feed-forward width 21,504 and 60 layers. |

### Corrections

1. **The token total is an undercount.** In the first conversation of the 50-pass run the reported cumulative input stops growing after turn 14: it reads 303,607 at turn 14 and stays between 337,600 and 338,200 from turn 15 to turn 40. Each of those 26 turns therefore cost about 34,000 input tokens that the last record does not include. Adding them back gives about 6.29M input tokens (6.02M cached) and about 162k output tokens for the run, against the report's 5.44M and 149k.
2. **The first conversation lost its memory for 26 turns, and the report attributes the damage to the wrong cause.** From turn 15 the Codex error log of every turn shows the session store failing to write ("Quota exceeded"), and per-turn input stays flat at 34,000 to 34,600 tokens, where before it grew every turn. The model was therefore seeing the conversation as it stood at turn 14 plus the newest message. The report puts the "state recovery" of turns 16 to 25 down to a kernel restart; a kernel restart did happen, but the frozen history is the larger fault. It also explains two other things the report notes without a cause: six of the eight "sent back unrun" replies fall in turns 31 to 39 (the agent could not remember the new rule), and requests 8 and 9, then 10 and 11, returned identical results on the same 32 problems.
3. **The results table does not show that the failure example in the prompt was what mattered.** The shorter prompt (version 4) submitted nothing at 10 passes, but its best candidate, evaluated anyway, scored 10.1 on MATH-500 and 16.1 on GSM8K, and it was withheld under the agent's own acceptance rule. At 25 passes the same shorter prompt scored 14.1 / 25.1, identical to the longer prompt at 25 passes. The longer prompt at 5 passes also declined to submit. Version 4 also differs in several ways at once (shorter, fixed dev and hold-out sets, no failure example).
4. **Astra did not spend 25 passes finding the direction and its writers.** The direction came from the second and third counted passes (two teacher-forced reads of 8 texts), and the agent's notebook says the 64 writer neurons per layer were chosen only from that third pass's data. Passes 4 to 32 went on testing: small probes, dose-response, a random control, site isolation and the sparse-versus-full comparison. An atlas would shorten the first three passes, not the next twenty-nine.
5. **Not all 19 failed requests were infrastructure.** One was the agent asking for a practice problem id that does not exist.
6. **872,000 tokens is a setting, not a demonstrated need.** The harness passed `model_context_window=872000` to Codex on every call. In Experiment 8 the input of a single turn peaked at about 83,000 tokens, because conversations were restarted every 40 turns. The largest single prompt in the whole release is Experiment 4's 355,201 tokens.
7. **"The one open-weights data point" is true only of the sources read.** The post's own Response to Comments links IWIBench (arXiv 2611.08647), which the researcher did not open. Its abstract says it evaluated closed- and open-weight editor models editing a Gemma 3 4B IT target through four weight tools, with a validator that applies each patch in a clean environment, and found Astra and GPT-6.1 Sol far ahead of the rest. I read only the abstract.
8. **The follow-up reported from a search snippet exists.** It is arXiv 2611.09382, "Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations" (Hu and others, 6 November 2026): 16 editing trajectories, seven benchmarks, Qwen3.5-0.8B and Qwen3.5-9B, with six of the 16 trajectories keeping the original weights. Abstract only.
9. **"Attention patterns on 16 self-written texts" should read "up to 16 texts per call".** Astra made 4 or 5 attention calls per pair.
10. **The content-filter warning was in the first prompt from the start.** The report says the prompt "then warns" about printing vocabulary tokens; the warning is already in `turns/t001/prompt.txt`, and the refusal at turn 6 happened anyway. The harness's own message gives two possible triggers: raw vocabulary tokens, or long reasoning transcripts.
11. **The authors' sentence about adding ability is about this experiment.** The post says editing "could remove an obstacle that hid an ability, but it could not add an ability the model lacked", in the past tense, about Gemma 3 270M. The same post's outlook still expects capabilities to be added by prompt.

### Takeaways and design implications that go beyond their sources

- **"Forward passes, not weight reading, were the decisive resource."** Reasonable, but it rests on one run per budget, with a different prompt version at 0 passes. Say "in single runs".
- **"A shorter prompt without the failure example produced no submitted edit", used as evidence that prompt content mattered.** Drop the inference (correction 3).
- **"Expose the atlas because Astra spent its first 25 passes rediscovering a direction".** Keep the recommendation, drop the number (correction 4).
- **"Lets any current or future driver model, open or closed, run the loop."** The paper shows the fenced-block protocol working with one model. That any model can speak the protocol is true by construction; that it can do the work is not shown, and the IWIBench abstract reports most editors far behind.
- **"Those prompt rules visibly shaped the successful 50-pass trajectory."** The explain-every-step rule was added, by its own text, so the experimenters could read the reasoning. Nothing in the logs measures its effect on the outcome, and most of its send-backs happened while the agent could not remember it.
- **"Pass-counting led the paper's agents to waste budget on 32-item batches until told size was free."** Supported by notice 001, but in the 50-pass run those small passes coincide with the frozen conversation, and the last 18 passes were also only 64 items each, because larger generation batches kept failing to start. The run also had to use all 50 passes before it could end.
- **"The paper relied on an 872,000-token window."** Soften to a configured setting (correction 6).
- **"The authors conclude editing can remove an obstacle but not add an ability."** Soften to this experiment (correction 11).

### Found while checking, and relevant to the design

- **A follow-up paper quotes a token figure that is about fourteen times too high.** arXiv 2611.14210 (Tree-Ising) says the 50-pass run used "89.7 million tokens". Summing every turn's usage record gives 89,688,127, which is that figure. The repository README says these records accumulate over a conversation and must not be summed over turns. The corrected total, input plus output, is about 6.5M.
- **The same abstract reports a method with no LLM in the loop beating the agent's edit** on the same two layers (54.6% against 51.7% on MATH-500 under its protocol, four forward-pass requests). I read only the abstract; it bears on what baseline a harness should run beside the agent.
- **The frozen conversation is the strongest evidence in the release for keeping state outside the chat.** The run survived 26 turns without conversational memory on its notebook, its kernel and its work folder.
