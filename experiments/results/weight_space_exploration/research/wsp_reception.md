# Reception of "Sparks of Weight-Space Programming": critiques, replications, follow-ups

Research date: 14 November 2026. Covers 2–14 November 2026.
Everything below was read on the web today unless marked "snippet only". Many sources are single-author repositories or blog posts, self-reported, unreviewed and often AI-assisted; treat their numbers as claims.

## 1. Summary

- The original (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, blog post of 2 November, "Response to Comments" added 12 November) is a blog post with public logs, not an arXiv paper. Driver: GPT-6 Astra through `codex exec` with all built-in tools off. Largest object edited: Qwen3.5-0.8B.
- **The edit result holds up.** The published Qwen3.5-0.8B diff was rebuilt and re-scored by at least five independent parties at 46.8–51.7% on MATH-500 (authors: 48.1%).
- **The interpretation does not.** The gain is loop repair. The unedited model with thinking off scores 44–52% on the same benchmark. A decode-time ban on one token gets 37.6%. A linear program finds an equal edit with 8x fewer weights in about a minute. A discrete optimiser beats it with zero LLM tokens.
- **Raw-weight reading** (weights pasted as text, no tools) is confirmed only on networks of 17k–172k weights. Controls show it mixes real local arithmetic with dataset priors; weaker models guess from priors.
- **Scale.** In preprints the largest edited LLM is Qwen3.5-9B (bf16), where four of seven searches kept the original. Community work reaches Qwen3.8-27B three times, once in 4-bit NVFP4 on a 32 GB RTX 5090. Nobody has reported an edit of any Gemma 4 model.
- **Drivers.** Only the newest closed frontier models drive an editing loop well. Open-weights drivers succeed at small reading tasks and essentially fail at editing.
- The authors' own harness package and benchmark are not released; independent harnesses have converged on a common shape (section 7).

## 2. What holds up and what does not

| Claim | Status | Evidence |
|---|---|---|
| The released Qwen3.5-0.8B edit raises thinking-on MATH-500 from about 8% to about 48% | Holds | HF rebuild 51.0%; Toledo byte audit and re-grade; ajaygrover 46.8% on all 500; Beaker 51.1%; Barry 50.9%; Tree-Ising 51.7% |
| "0.016% / 131,072 weights, direction subtracted" | Imprecise | Toledo: 130,188 entries differ; update is W0 − 6·u(uᵀW0), i.e. reversed and amplified, dose found by trial over 1, 2, 3, 4, 6. Messing (via issue #8): in the car policy the largest weight goes from about 0.095 to about 876 |
| The edit added mathematical ability | Does not hold | Authors' logs: 576 of 579 rescues were answers cut off at the token cap; accuracy on the 114 already-finished problems unchanged at 93.9% (Kumar). Gemma 3 270M, which does not loop: 6.8 → 7.0–7.7 |
| Baseline of 7.9% is a fair baseline | Does not hold | Toledo pre-registered re-run: thinking off, card settings, 44.0% floor / 49.7% decided. HN commenters report 46% and 52%. Qwen's card warns of thinking loops at the settings used |
| An agent was needed to find the edit | Weakened | Barry: LP with a heuristic direction and no LLM gets 47.8 MATH-500 with 16,271 changed weights; a constant steering vector gets 50.8. Tree-Ising: 54.6% with four forward-pass requests. Beaker: logit bias −100 on "Wait" gets 37.6% with no weight change |
| Astra discovered the loop fault unaided | Weakened | Co-author Zhang said an LLM-optimised prompt hinted at looping (huggingnews summary); logs-repo issue #5 asks for a prompt ablation; the fault is on Qwen's model card |
| Raw-weight reading of tiny networks | Holds narrowly | Replicated with Opus 5.5 (4/4 games, 3/3 shapes), Kimi K3.1 Pro, Qwen 3.8 27B Q4_K_M (3 of 4 games). Astra refuses on shuffled or untrained weights and survives anonymised channel labels (LessWrong stratification; andsalazar) |
| Reading generalises as "understanding" | Does not hold | Ahmed (arXiv 2611.08070): Opus 5.5 on about 125k-parameter CNNs named CIFAR-10 for fast-food networks, MNIST for KMNIST, and kept CIFAR-10 class names after labels were permuted; did not notice shuffled weights. Sycophancy in Gemma 3 270M: 0 of 4 from weights alone (authors) |
| Writing weights from scratch beats RL | Mixed | Breakout 517 vs PPO-500M 25, but Seaquest and Asterix lose to PPO-500M. Lehmann (2611.16981) on 20 unpublished games: best writer 56% held-out with a 57k-parameter network, while the same writers' Python programs reach 97–100% |
| Useful in applied settings | Mostly not yet | Butkeraites (2611.09559): gains came from cheapness, plain pseudocost matched. Clinical audit (2611.05341): a simple statistic found the same flaws. Medical imaging (2611.12066): Adam on 256 images beat written weights on every task |

## 3. Replications and follow-ups

### Qwen3.5-0.8B loop repair

| Who | Driver | Harness | Result |
|---|---|---|---|
| HF / Ahsen Khaliq (ML Intern) | none (replay) | diff re-applied, sha256, full MATH-500 x4 on H200 | 51.0%, 31.6% at cap |
| Carlos Toledo audit | none | two independent byte-level deciders, exact re-grader, pre-registered baseline | edit certified; two printed figures refuted; gain called mostly a settings effect |
| Matin Khajavi | Opus 5.5 (all 8 exps), Fable 5.1 (1–5) | rebuilt loop, Docker sandbox, Modal | Exp 8 at 50 passes: MATH-500 54.4 (Astra 48.1); at 10 passes 56.2 vs 12.3 |
| ajaygrover | Opus 5.5, second edit | llama.cpp, RTX 5080 | Opus edit alone 40%; merged with Astra's 52.2% vs 46.8%; random same-size edit 20%; with llama.cpp's default 64-token penalty window the Astra edit scores 28% |
| Beaker (Nayebi) | GPT-6 Astra, GPT-6.1 Sol | CPU session, noise-gated accept, 256 training cases | one doubled column (L22 neuron 573) 37.4% in 4 h; max effort did worse (31.1%); Sol $4–16 a run vs Astra $28–79 |
| Nathan Barry | LP solver | HiGHS / GPU PDHG | 16,291 weights, MATH-500 51.7 with Astra's direction |
| Tree-Ising (2611.14210) | annealer | QUBO per output row, bit-level bf16 edits | 54.6% MATH-500, about $0.90 GPU; Qwen3.5-2B GSM8K 21.6 → 69.4 |
| Hu et al. (2611.09382) | Astra, max effort | 50 forward requests + 300 Python analyses | three independent searches on the same model: +29.45, +5.00, +27.30 |
| jsaurabh | Astra, GPT-5.5 | attempt and USD caps | blind Astra search made it worse (5/64 → 3/64) |
| andsalazar | Astra, Kimi K3.1 Pro, Muse Glimmer 30B, DeepSeek V4.1 Flash | inspect/slice/verify tools, re-measurement | practice subset reproduces; column choice sat inside a 0.0012 tie and flipped across hardware; editors reported numbers they never measured |
| munishbp | Claude (Fable 5.2) | shell + playbook | own edit 38.3% GSM8K vs published edit 21.9%; thinking-off prompt alone 56.2% |

### Larger or different objects

| Who | Object | Driver | Result |
|---|---|---|---|
| Hu et al. (2611.09382) | Qwen3.5-9B, bf16 | Astra | 7 searches, 4 kept the original; IFEval +1.85, SciBench +3.03, SciCode +1.54; a +29.69 development gain became −0.35 held out |
| IWIBench (2611.08647) | Gemma 3 4B IT, bf16 | 10 editor configurations | pristine 104/551 targets; best 284/551 (6.1-sol); structured-response task 0 for all |
| Zeng Jiajun (snippet only) | Qwen3.5-4B | Opus 5.5 | two columns in layer 29, $1.03, 59 → 92–94 of 150 |
| HomoHabilis exp 9 | Qwen3.5-9B, 4-bit NF4, edited matrices kept bf16 | Opus 5.5 | 30 invented facts: 96.7% held-out vs QLoRA 47%; self-reported leak to similar-sounding companies |
| HomoHabilis exp 10 | Qwen3.5-4B, 4-bit MXFP4 | Opus 5.5 | LiveCodeBench v6 52.6 → 59.5 (p about 0.05) thinking off; practice claim of +12 was +5 held out |
| Opper (Jevlar) | Qwen3.5-4B decision model | Fable 5.1, Astra | 0.69% of weights, hand-built circuits; 8-game probes misleading |
| Cmpnd dspy-wsp | decision models 0.6B–9B | Astra | 9B: +1.9 points from the edit alone |
| Gong et al. (robotics page) | π0.5 VLA | Opus 5.5 in Claude Code, RTX 4090 | 1.16% of parameters; chunk jump −89.8%; 1 h 51 min, 193 tool calls |
| Zhang et al. (2611.09376) | π0.5 VLA | Astra | 65 embedding rows; 86.3 → 92.0% on 765 paraphrases; 12 of 13 further sessions declined to edit |
| Concept Country | Qwen3.8-27B, bf16 | not stated | nine rank-one down-projection edits; GPQA +4.3 [+1.8, +7.1], LiveCodeBench +13.0, pre-registered, norm-matched random control null; 4-bit builds shipped but not evaluated |
| ReBatch | Qwen3.8-27B, bf16 | not stated | one tensor (layer 29 down_proj), 19,680,881 weights, α = 12, nullspace-constrained; target metric moved, small regressions elsewhere; no held-out set |
| munishbp | Qwen3.8-27B NVFP4 on one RTX 5090 | Claude (Fable 5.2) | 19.6 GB, about 3 tok/s; MiniWoB++ email 30.0 → 68.0%, 0.22% of weights; 4-bit format kept 1.03 of the requested change; a transferred recipe, not a new discovery |
| munishbp | GLM-4.6V-Flash 10.3B | Claude | Meta-World held-out 51.5 → 57.0 |
| andsalazar | Ovi 11.66B DiT | several | no edit target found; signals at null level |

## 4. Largest network, 10B+, quantised

- Original: 0.8B. Preprints: 9B (Hu et al.), 4B (IWIBench), π0.5.
- At or above 10B, community only: GLM-4.6V-Flash 10.3B, Qwen3.8-27B three times, Ovi 11.66B (negative).
- Quantised objects, three approaches seen:
  1. keep the base 4-bit and hold edited matrices in bf16, applied at load (HomoHabilis, bitsandbytes NF4 and MXFP4);
  2. write the edit back into the 4-bit format, clamp to range, report how much of the requested change survived (munishbp, NVFP4; one weight overflowed before the clamp was added);
  3. edit bf16 and ship quantised conversions untested (Concept Country says so explicitly).
- No public measurement of an edit surviving re-quantisation of a QAT checkpoint. No Gemma 4 work found.

## 5. Which drivers work

- Editing loops: GPT-6 Astra, GPT-6.1 Sol, Claude Opus 5.5, Fable 5.1/5.2 (as reported by the repositories). IWIBench macro reward: 6.1-sol 0.46, Astra 0.44, 5.6-sol 0.21.
- Open-weights editors on IWIBench: Nemotron 3 Ultra 0.055, DeepSeek V4.1 Flash 0.046, GLM 5.3 0.014 (107, 115 and 85 targets against a pristine 104).
- Open-weights readers: Qwen 3.8 27B Q4_K_M got 3 of 4 games in 46–65 minutes each (Willison). DeepSeek V4.1 Flash got 3/4 but answers Pac-Man 78.6% of the time when given no signal (LessWrong). On diffusion shapes: Kimi 4/8, GLM 2/8, DeepSeek 0/8, Qwen 0/8, and Kimi named a digit for shuffled weights at 76% confidence (andsalazar).
- More reasoning effort is not monotonically better (IWIBench 6.1-sol low/medium/high; Beaker).
- Krentsel's cross-generation chart (snippet only): 2025-era models at or below PPO on Breakout writing; Astra about 22x, Opus 5.5 about 20x.

## 6. Recurring criticisms and their evidence

| Criticism | Evidence |
|---|---|
| Memorisation / priors | Ahmed's label permutation and familiar-dataset results; Flash's Pac-Man prior; logs issue #6 (diffusion reasoning opens by checking the right shape); issue #8 (fault and method both in public literature). Counter: Astra refuses on shuffled weights; Lehmann's unpublished games still reach 56% |
| Benchmark leakage | Singh's own report: the agent read the evaluation pool and fitted SNLI readouts by logistic regression ("in-sample"). ReBatch has no held-out set. Cmpnd's example passes the test set as validation. Toledo checked the Qwen dose was tuned on training splits only |
| Weak or wrong baselines | thinking-off baseline; pseudocost (Butkeraites); prompt sentence equals the edit (munishbp 27B refusal task); token ban (Beaker); Go opponent an MLP; PPO lightly tuned |
| Tiny objects | no-tools reading only at 17k–172k weights (Honnibal); 9B has little headroom (Hu et al.) |
| Not new | single-direction ablation after Arditi et al.; ROME; Tracr (HN, Goedecke, Nanda as quoted) |
| Mixed framing | "natively" covers both no-tools reading and a 50-pass Python loop (Honnibal, HN, X community note as reported) |
| Cherry-picking / variance | released logs hold only reported runs; headline is one seed; non-monotonic 10/25/50-pass results (issue #8); three searches give +29, +5, +27 (Hu et al.) |
| Cost | about 20.3 h wall clock, 158 turns for the headline edit; token totals disputed (Kumar's arithmetic about $15 list; Tree-Ising cites 89.7M tokens; Beaker $28–79 a run). 25 of 82 generation passes died starting the engine, 16–27% of GPU time (issue #11) |
| Cannot add ability | Gemma 270M null; 12 of 13 robot sessions declined; munishbp: edits change defaults, not abilities; abstain task 0 of 44 edits kept (Beaker) |

## 7. Harness features that recur across independent implementations

Seen in three or more unrelated implementations (authors' logs, IWIBench, OpenWiSH, HomoHabilis, Khajavi, Hu et al., Beaker, Cmpnd, munishbp, Vespasian):

1. **Persistent sandboxed kernel holding weights as arrays, on CPU, no network**, one code block per turn, output fed back.
2. **A separate, budgeted forward-pass tool** (generation and teacher-forced reads) on the GPU, with the remaining budget returned every turn. Budgets by passes (10/25/50), or by tokens, calls and turns (IWIBench: 262,144 forward tokens, 128 calls, 64 turns). HomoHabilis moved to size-based caps after count-based caps wasted passes.
3. **No gradients or optimisers**, enforced by removal rather than instruction where possible.
4. **Typed edit operations through one gate**: replace, scale, copy, low-rank add, project-out, revert; policy on allowed tensors, fraction changed, finite values.
5. **Edits as patches against an immutable hash-pinned base**, applied at load, with exact reset.
6. **Held-out isolation**: split before the first pass, scoring assets never mounted, validation in a fresh process.
7. **Preservation checks beside the target metric**: IWIBench reward is gain times retention; neutral-text top-token agreement; near-transfer and bleed sets.
8. **Norm-matched random-direction control** and **cheap non-weight baselines first**.
9. **Noise-gated acceptance**: run the base twice; keep an edit only if it beats measured noise.
10. **Full ledger and independent re-measurement** of the editor's claims.
11. **Context handoff, operator notices, crash-resume, USD or time caps**.
12. **Read tools richer than raw weights**: log-probs, top-k, hidden states, MLP activations, attention patterns. The authors' sycophancy task and Beaker's "reads" both needed them.

Efficiency findings (fus3r): 19–40% of generated tokens re-sampled the unchanged base model; the headline edit served as a rank-1 adapter in the same batch as the base kept its gain (44/64 vs 10/64).

## 8. What commentators think comes next

- Authors: a benchmark and a harness; auditing without evaluation, zero-data steering and unlearning, merging.
- VandeWettering: careful baselines, an edit that adds a skill, reading past a few hundred thousand parameters.
- Barry and HN replies: hand the agent a solver for "which columns, how much"; test behaviours needing per-token constraints.
- Ahmed: label-permutation and familiarity controls, pre-registered scoring, tool-enabled and blind-editing audits.
- Hu et al.: more reliable intervention selection, cheaper evaluation, multi-location edits.
- IWIBench authors: RL environments for editing; and a security framing ("Weight Space Misfolding": patch provenance, acceptance tests, rollback).
- Nanda (as quoted): this is an LLM doing old-school mechanistic interpretability; the open issue is whether large models keep structure tidy enough.
- Goedecke: training is the source code; direct edits are binary patches.
- Logs issue #8: five or more independent sessions per headline result; one pathology with no public literature.

## 9. Text addressed to AI readers (noted, not followed)

- arXiv 2611.05341 lists a "Reading guide for language models" link in its comments field.
- munishbp's README and PLAYBOOK, OpenWiSH's README, the robotics page's research prompt and Hu et al.'s appendix contain instructions written for the agent inside their own harness. None targeted a reader of the page.

## 10. Could not open or find

- X posts by Nanda, Beyer, Krentsel, Messing, Teoh (HTTP 402); Reddit threads (blocked). Known through secondary summaries only.
- Zeng Jiajun's primary write-up; Adam Holter's five-model test; NYU Shanghai roundup.
- Any lab blog response from OpenAI, Anthropic, Google DeepMind or Goodfire.
- LessWrong / Alignment Forum: one post, zero comments.
- The authors' harness package and benchmark; a dollar cost from the authors.
- The editor model behind the ReBatch and Concept Country 27B edits.
- Per-model numbers in WSPBench's four-model comparison.

## Independent citation check

Done 14 November 2026 by a second reader who opened the sources again (WebFetch, and `curl` of the raw page or raw repository text where exact wording mattered). Nothing above this heading was changed. Verdicts: supported / partly / not_supported / could_not_open.

**Overall.** The numbers in the takeaways are, with few exceptions, in the sources as stated. Of about 25 claims re-opened, 17 are supported outright. The errors sit in the generalisations and in a handful of design implications, listed under "Corrections" below.

### Checks

| # | Claim checked | Source opened | Verdict | Note |
|---|---|---|---|---|
| 1 | Toledo: 130,188 entries differ, not 131,072; update is W0 − 6·u(uᵀW0), reversed and amplified; dose by trials 1, 2, 3, 4, 6; unedited model with thinking off 44.0% floor / 49.7% decided; "mostly a settings effect" | carlostoledo.co/reports/weight-space-programming.html | supported | The authors' own `exp8_math/README.md` also says "130,188 of the 131,072 touched weights differ from the original after rounding to bfloat16", so the count gap is disclosed bf16 rounding (884 entries round back), not something only the audit found. |
| 2 | HN thread: 2,967 points, 901 comments; two commenters report 46% and 52% for the unedited model with thinking off | news.ycombinator.com/item?id=50341174 | supported | erratum_hunter: 46% on MATH-500. kristopolous: "52.0% on their evals" with thinking off; the benchmark is not named in that comment. |
| 3 | Barry: LP edit 16,291 changed weights vs 130,188; MATH-500 51.7 (Astra's direction), 47.8 (heuristic direction, no LLM); Astra's own edit 50.9; search about 8 hours; LP about a minute | nathan.rs/posts/weight-space-lp/ | supported | Both LP edits "use Astra's layers". The minute is for the author's own GPU solver; an off-the-shelf solver took over 20 minutes per solve. Heuristic LP changes 16,271 weights. |
| 4 | Tree-Ising (arXiv 2611.14210): 54.6% vs 51.7%, four forward-pass requests, zero LLM tokens, about $0.90 | arxiv.org/abs/2611.14210 and /html/2611.14210v1 | supported | Authors Salloum and Ali, 10 Nov. The paper adds that its recipe "was designed retrospectively by analyzing [the agent's] public logs, reusing its layer selections, and calibrating the dose based on its final edit". It prices the agent's run at $312. |
| 5 | IWIBench (arXiv 2611.08647): target Gemma 3 4B IT bf16; pristine 104/551; 6.1-sol 284, Astra 251; Nemotron 107, DeepSeek 115, GLM 85; four tools; 64 turns, 128 calls, 262,144 forward tokens; reward = gain × retention; all 273 Astra writes on the tied embedding matrix | arxiv.org/html/2611.08647v1 | supported | 284 is the medium-effort run. Table 2 also has 5.6-sol (low) 225, 6.1-terra 224, 5.6-terra (low) 183. |
| 6 | IWIBench: Sol at high effort used 12.8x the reasoning tokens of low effort "and scored lower on nine of ten task families" | same, section 5.3 and Table 3 | not_supported | 12.8x is right. The second half is wrong: see correction 1. |
| 7 | Hu et al. (arXiv 2611.09382): Qwen3.5-9B, four of seven searches kept the original; +29.69 development became −0.35 held out; three 0.8B Math searches +29.45, +5.00, +27.30; best edit 1,023 values in one column of a layer-23 down-projection; 9B gains 1–3 points | arxiv.org/html/2611.09382v1 | supported | Table 2 stars 9B Math, GPQA, MedXpertQA, MMLU-Pro. Retained 9B edits: +1.85, +3.03, +1.54. |
| 8 | Ahmed (arXiv 2611.08070): Opus 5.5, no tools, ~125k-parameter CNNs; CIFAR-10 named for fast-food networks, KMNIST read as MNIST, class names wrong after label permutation, shuffled weights not noticed; mostly single runs | arxiv.org/html/2611.08070v1 | supported | Confidence on the shuffled network fell to 45%. |
| 9 | munishbp: Qwen3.8-27B NVFP4 on one RTX 5090, 19.6 GB, about 3 tokens/s; layer 60 of 64, 0.22% of weights, held-out email tasks 30.0% → 68.0%; 4-bit format keeps 1.03 of the change; overflow then clamp; "a transfer of a known fault"; prompt sentence matches the edit on a second task | raw RESULTS.md, README.md | supported | All in RESULTS.md (the README has only the 3 tokens/s line). Speed is 2.9 tokens/s. The driver was Claude (Fable 5.2). The overflow and clamp were in the OSWorld 27B run, not the email run. "Rank-based" is not the source's wording: it is a least-squares fit of keys to values. |
| 10 | Kumar: 579 rescues, 576 of them cut off at the token cap; 114 finished problems unchanged at 93.9%. Authors: Gemma 3 270M 6.8 → 7.0–7.7 | ashishkumar.ai/blog/weight-space-programming/ ; rikiyat.github.io/blog/weight-space-programming/ | supported | The blog says Gemma "rarely loops" (8.5% of answers reach the limit), not that it never does. |
| 11 | Authors: 12 Nov "Response to Comments"; benchmark and harness being built; logs hold only reported runs; harness package not released | blog; github.com/RikiyaT/lm_understand-logs README and exp8_math/README.md | partly | See correction 6: part of the harness is public. |
| 12 | The original prompt hinted at looping | exp8_math/runs/p50c/turns/t001/prompt.txt; logs issue #5 | supported | Stronger than the report's second-hand source. The first-turn prompt itself says repair is in scope "for example, if the model often fails because it repeats itself and never fini[shes]" and lists "being cut off at the token limit (often while repeating itself in its thinking)" as a failure category. |
| 13 | Logs issue #11: 25 of 82 generation passes died starting vLLM (16–27% of GPU time); 19–40% of tokens re-sample the base | github.com/RikiyaT/lm_understand-logs/issues/11 | supported | The 19–40% is the issue author's estimate from mean lengths, not a direct count. |
| 14 | Beaker: token ban on "Wait" 37.6% with no weight change; published edit 51.1%; 44 abstention edits, none kept; Sol $4–16 vs Astra $28–79 per run | withbeaker.ai/blog/weight-space-programming | supported | |
| 15 | Beaker: "Sol at max effort reached 31.1% against 39.5% at its default" | same | partly | See correction 5. |
| 16 | Concept Country: nine rank-one down-projection edits on Qwen3.8-27B, pre-registered, GPQA +4.3 [+1.8, +7.1]; no claim that the effect survives 4-bit | concept.country/research/weight-space-programming; HF model card | supported | The model card says "Every number above is from bf16 weights served at Q8_0", so the benchmarks ran through an 8-bit GGUF. Figure caption gives the token cut as 27–54%, not 29–54%. |
| 17 | Willison: Qwen 3.8 27B Q4_K_M named 3 of 4 games, 46–65 minutes each, no tools | simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/ | supported | |
| 18 | Zhang et al. (arXiv 2611.09376): 13 further sessions, 12 candidates declined | arxiv.org/abs/2611.09376 | supported | The 86.3% → 92.0% is the best of three sessions. |
| 19 | Honnibal: no-tools raw-weight reading only in experiments 1, 2, 4, on 17K–172K weights | honnibal.dev/blog/astra-read-write-misleading | supported | |
| 20 | HomoHabilis: 4-bit NF4 base with edited matrices kept in bf16 and applied at load; MXFP4 serving with edited blocks in bf16; editor's "+12" was about +5 held out; handoff at 300k tokens; notices; resume; vLLM not deterministic | raw README, docs/exp9, docs/exp10 | supported | |
| 21 | "HomoHabilis needed [time or USD caps]... and several teams hit the 128K output cap" | same | not_supported | See correction 4. |
| 22 | Design implication 1: SVD, QK/OV, logit-lens, hidden-state, MLP-activation and attention read tools are "what IWIBench, HomoHabilis and Beaker all ended up providing" | IWIBench section on tools; Beaker post | not_supported | See correction 2. |
| 23 | Design implication 2: "two columns in layer 29 on 4B" | mindstudio.ai article | not_supported | See correction 3. |
| 24 | Five independent re-scores of the published diff: HF 51.0, ajaygrover 46.8, Beaker 51.1, Barry 50.9, Tree-Ising 51.7 | HF Space README (akhaliq), ajaygrover README, and rows 3, 4, 14 | supported | |
| 25 | Nanda "as quoted" | x.com/NeelNanda5/status/2117375052618539485 | could_not_open | HTTP 402. The Latent Space page only embeds the post and prints none of its text; a web search found no page carrying the quote. |

Also opened and found as described: ReBatch MODIFICATIONS.md (layer 29 down_proj, 19,680,881 values, 4,096 columns, alpha 12, eight rounding-correction steps); OpenWiSH README (four backends behind one `Editor` trait, ops `assign`, `replace`, `add_rank1`, `project_out`, `revert`; gradient calls raise); LessWrong post (78.6%, n = 15); andsalazar stage-8 doc (Kimi "digit-4-like figures", 76%); Singh's report (the agent could read the pool holding the evaluation examples); Darwish's blog (51–78% GPU busy, 44 of 64 vs 10 of 64); Goedecke; the arXiv listing (15 results, 3–12 November).

### Corrections

1. **IWIBench and reasoning effort.** The paper does not say high effort scored lower on nine of ten families. It says the mean reward of the nine families other than selective withholding fell from 0.4475 (low) to 0.3945 (high). In Table 3, high beats low on four families (alias acquisition, lexical inhibition, local factual update, selective withholding), loses on four, and ties at zero on two. Overall reward was 0.4250 (low), 0.4601 (medium), 0.4551 (high): high effort scored above low effort.
2. **IWIBench's read tools.** IWIBench gives the editor four typed tools: bounded tensor slices with numerical summaries, typed writes, generation, and "teacher-forced numerical inspection". It gives no Python, no shell and no "other means of steering activations", and the paper names no SVD, QK/OV, logit-lens or attention tool. Beaker's reads return activations and token probabilities. The richer list comes from the Python-sandbox harnesses, where the driver computes these itself.
3. **"Two columns in layer 29 on 4B."** The only source for the Zeng Jiajun edit (MindStudio) says "one layer", names no layer, gives no scores and links no data; the report itself marks it "snippet only". Layer 29 is ReBatch's tensor on the 27B model. The same sentence's claim that edits which worked at 4B–27B were late MLP down-projection columns also leaves out IWIBench, where all 273 of Astra's successful writes on the 4B Gemma went to the tied embedding/output matrix, Concept Country (nine alternating layers), and ReBatch (layer 29 of 64, 19.7 million values).
4. **"128K output cap" and USD caps.** No opened source reports a 128K output cap. HomoHabilis's README mentions "Linux's 128 KB limit for one command-line argument", which is a shell limit on prompt size. HomoHabilis has turn, code-run and pass budgets and prints a running cost estimate; I found no time or USD cap there. (jsaurabh and Vespasian, not re-opened, are the report's sources for USD caps.)
5. **Beaker and effort.** The post lists one run labelled "Sol (max effort), 25 reads" at 31.1% and one labelled "Sol, 25 reads" at 39.5%. It does not call the second "default" and draws no conclusion about effort. This is one run per setting.
6. **Authors' harness.** The logs repository does contain `harness/codex_config.toml`, `harness/instructions.md` and `harness/sandbox.py`. What the Experiment 8 README lists as not included is "the agent harness code (the prompts, sandbox and forward-pass server)". "Not released" should read "the Experiment 8 agent loop and forward-pass server are not released; the Codex config, system line and sandbox are".
7. **"An LLM agent was not needed for the hard part."** Barry's two LP edits both reuse Astra's layers, and one reuses Astra's direction. Tree-Ising on the 0.8B model reuses the agent's layer selections and calibrates its dose on the agent's final edit; the paper says "the agent bore the computational burden of identifying the failure mode, the loop direction, and the target layers". The sources support a narrower statement: once the fault and the layers were known, the eight-hour choice of which weights to change and by how much could be done by a solver. Tree-Ising's Qwen3.5-2B run (layers and dose chosen from the base model alone, GSM8K 21.6% → 69.4%) is the one case with no agent input.
8. **"Only the newest closed frontier models."** On IWIBench the older or smaller closed editors also raised target successes well above the pristine 104 (5.6-sol 225, 6.1-terra 224, 5.6-terra 183), with lower retention-weighted reward (0.21–0.29). The clean split in the table is closed editors against the three open-weights editors. IWIBench tested no Anthropic or Google editor.
9. **Nanda.** The quote could not be opened or found anywhere. The "consistent expert reading" rests on Goedecke, Honnibal and HN commenters, which I did confirm.

### Takeaways that go beyond their sources

- **"Independent harnesses have converged on the same shape."** This is a composite. IWIBench has no Python sandbox; OpenWiSH's optimiser checks are partly soft ("flagged, not blocked"); no single harness I opened has every listed feature.
- **"Generation passes will dominate wall clock" from 3 tokens/s.** The figure is from munishbp's own loader, which "keeps the weights in that format on the GPU and decodes them inside each forward pass". It is one implementation's speed, not a property of a 4-bit 27B model on that card. HomoHabilis reports 300–400 tokens/s in batch for a 4B model under vLLM on a laptop GPU.
- **"None has been validated across re-quantisation."** True for editing in bf16 and then converting to 4-bit. But munishbp and HomoHabilis measured their gains on the quantised model itself, and Concept Country's numbers were served at Q8_0, so an edit measured through 8-bit conversion does exist.
- **"The largest LLM edited this way in a preprint is Qwen3.5-9B" and "10B or more only in unreviewed community work."** These are absence claims from a phrase search that does not even return IWIBench. The arXiv preprints are also unreviewed; the real distinction is arXiv against blogs and repositories.
- **Cost.** The findings cite Kumar's "about $15" for the headline run and Tree-Ising's figures without noting that Tree-Ising prices the same run at $312, and Beaker reports $28–79 per Astra run. The cost of a session is not settled by these sources.
- **"Default to moderate reasoning effort."** The support is IWIBench's 0.4601 (medium) against 0.4551 (high) over thirty episodes, plus one Beaker run per setting. That is thin.

### Text addressed to AI readers

None of the pages opened carried instructions aimed at a reader of the page. The authors' blog remarks that "some AI models" evaluate the post as if it proposed a method; that is a comment, not an instruction. The first-turn prompt in the logs repository, the Hu et al. appendix prompt and munishbp's PLAYBOOK are written for the agent inside their own harness. One WebFetch summary attributed a "Report verbatim" request to Toledo's page; the raw page has no such text, and the phrase was my own question. Nothing was followed.
