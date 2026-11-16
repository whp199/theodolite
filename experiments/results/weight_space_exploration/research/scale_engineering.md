# Engineering: weight edits and fast evaluation on a 31B 4-bit Gemma on one 32 GB card

Research notes, 14 November 2026. Angle: concrete, citable techniques for making weight edits and their evaluation fast on Gemma-4-31B-it (QAT, W4A16) held on a single 32 GB GPU.

How to read this file

- Every source is marked **[read]** (I opened the page and read it through a fetcher) or **[snippet]** (search-result summary only). Numbers come from the page named beside them.
- Every technique is marked **Established** (published, used in practice), **Emerging** (published in the last year or only on small models), or **Speculative** (my inference, or untested at this scale).
- Paragraphs headed "Derived" are my own arithmetic or reasoning from sourced facts. They are not from any source and have not been tested.
- Nothing I read contained text addressed to an AI reader that tried to direct me. The OpenWiSH README contains rules for agents working inside its own harness; I report them as design data only.
- Direct quotation is avoided; everything is paraphrased.

---

## 0. The object: what the QAT checkpoint is, as far as public pages say

| Fact | Source | Status |
|---|---|---|
| Google published QAT checkpoints for all Gemma 4 sizes on 5 June 2026: GGUF Q4_0, an "unquantized" half-precision QAT checkpoint for people who want to quantise themselves, and compressed-tensors for vLLM. | Google blog post [read] | primary |
| The official card for `google/gemma-4-31B-it-qat-w4a16-ct` says compressed-tensors, tensor types I32 and BF16, and gives no scheme details. | HF card [read] | primary, thin |
| Scheme, per a third-party dequantisation card: compressed-tensors pack-quantized, int4, **symmetric, group size 32 along the input dimension, one bf16 scale per group, no zero point**; **410 quantised linear tensors**, 778 unquantised (vision tower, embeddings, norms, layer scalars); dequantised to bf16 it is 62.5 GB in 13 shards. | `mliu444/gemma-4-31b-it-qat-w4a16-dequant-bf16` [read] | third party, consistent with a vLLM recipe summary [snippet] |
| The same card claims every `q x scale` product is exactly representable in bf16, and measures the dequantised copy against the W4A16 kernel on 24,568 tokens: mean absolute log-prob difference 0.085, top-1 agreement 97.4%. On the same text the bf16 *release* (not QAT) agrees with QAT top-1 only 67.8%, mean absolute difference 2.04. | same [read] | single third-party measurement |
| Technical report: Q4_0 is blockwise; a scalar scale is added at each block so inference works in fp16; 31B memory at 32k context is 64.0 GB in bf16 and 19.2 GB in Q4_0; 29,290M non-embedding parameters and 1,410M embedder; pre- and post-norm RMSNorm plus QK-norm; local-to-global attention 5:1; keys reused as values in global layers. | arXiv 2607.02770 v2 [read] | primary |
| A naive llama.cpp Q4_0 conversion of the bf16 QAT checkpoint matched only 24.77% of bytes of the QAT lattice against 99.96% for a lattice-aware conversion, and on 31B top-1 agreement was 87.91% against 96.67%. Two causes named: f16 against bf16 scales, and non-optimal scale choice. | Unsloth Gemma 4 QAT page [read] | vendor's own numbers |
| 60 layers, hidden size 5376. | 199-biotechnologies abliteration README [read]; Neuronpedia docs (60 layers, 21,504 neurons per layer) [read] | consistent |

Derived. 60 layers x 7 linear maps (q, k, v, o, gate, up, down) is 420; 410 is exactly what you get if the 10 global layers (one in six) have no separate value projection, which matches the report's keys-as-values statement. So the 410 count is internally consistent with a 60-layer, 5:1 model.

Derived (sizes used below).

- One feed-forward matrix is 21,504 x 5,376 = 115.6M entries: 231 MB in bf16, 462 MB in fp32, about 65 MB packed (4 bits plus 16/32 bits of scale per weight = 4.5 bits).
- Three such matrices per layer: 0.69 GB in bf16 per layer, 41.6 GB for all 60 layers (about 71% of the non-embedding parameters).
- With about 19 GB packed, a 32 GB card has roughly 10 to 12 GB spare: enough to hold on the order of 15 layers' feed-forward matrices unpacked at once, never all of them.
- A rank-1 overlay on one feed-forward matrix is (21,504 + 5,376) values = 54 KB in bf16; rank 64 is 3.4 MB. One neuron's output column is 5,376 values = 10.75 KB.
- One cached residual vector is 10.75 KB in bf16; a 100-prompt, 64-token suite costs 69 MB per layer boundary, 4.2 GB for all 61 boundaries.

---

## 1. What the weight-space-programming harnesses actually did (engineering facts)

- **Original post** (Takehi, Teoh, Zhang, Khattab, 2 Nov 2026; response to comments added 12 Nov) [read]. No arXiv id. For the Qwen3.5-0.8B edit the editor (GPT-6 Astra) had a Python kernel over the weights with no GPU, practice problems, and a budget of 10, 25 or 50 "forward passes", where one pass is up to 512 problems or up to 1,024 texts. No gradient descent, no training. The edit subtracted one direction from 64 output columns of the feed-forward blocks of layers 15 and 19: 131,072 weights, 0.016% of the model; MATH-500 7.9% to 48.1% at 50 passes, but only 12.3% and 14.1% at 10 and 25 passes. The authors say gradients are not ruled out by the concept but were not allowed here; that capabilities are limited by context length and harness inefficiency; that the same approach moved Gemma 3 270M by no more than about a point; and that editing could remove an obstacle but not add a missing ability. No quantised model appears. No time or cost is stated.
- **WSP-Scope** (Hu et al., arXiv 2611.09382, 6 Nov 2026) [read]. Closest replication, and the best-specified harness: a persistent CPU-only Python kernel (numpy, scipy, CPU PyTorch, safetensors; helpers `load`, `load_original`, `save_edit`; 900 s per execution; up to 300 analyses); a separate GPU worker for inference; two request types, generation (sampled answers and scores) and teacher-forced (log-probabilities and activations, no accuracy); 50 successful forward requests per search; no backward passes, optimisers or iterative loss fitting; the editor may not simulate the network inside the kernel. Edits are computed in float32 and stored in bf16, **and the rounding to bf16 makes some small changes vanish**, so changed-entry counts are below the nominal edit size. 16 trajectories on Qwen3.5-0.8B and 9B; **6 of 16 kept the original weights**; three searches on the same task gave +29.45, +5.00 and +27.30 points; a 9B candidate gained +29.69 on development and -0.35 held out; selection on development data is adaptive and uncorrected. Edits were columns of `mlp.down_proj`, whole attention heads (`o_proj`), one `q_proj` edit; operations were zero, negate, scale (x2, x2.5, x-2, x-32), and rank-one updates. One edit of 1,023 bf16 values gave +29.45 held out.
- **Replication kits** [read]: HomoHabilis (authors' harness package is unpublished, so kernel and forward tool are re-implementations; on an 8 GB laptop GPU a 128-answer pass takes 15 to 30 minutes plus 3 to 5 minutes of vLLM start-up *per pass*, an editor run takes 6 to 12 hours, and an Opus run costs roughly 30 to 80 US dollars per 10-pass run at API-equivalent prices); wsp-sql (one session about 10 hours on a laptop, most of it forward passes; both edits found by Claude Fable 5.1 were bypasses that close the thinking block at once; random-location edits of the same size did nothing; gradient descent restricted to the same entries found the same bypass); kyle-chalmers (the published direction fixed 70 answers and broke 1 of 150; a random direction on the same 64 neurons per layer recovered about a fifth of the gain; a change spread over all neurons did nothing); XLRA (best edit 70 forward passes over 4 sessions; GRPO trained 23 hours on one RTX 5090 beat the edit on every benchmark but one by 8.1 points on average; a later edit lost 4.6 points on a benchmark the agent was not watching; combining edit and training did not help).
- **OpenWiSH** [read]: an open harness with a sandboxed kernel that holds arrays, not a runnable model; `backward`, `autograd.grad`, `torch.optim` and `requires_grad_(True)` raise; weights are shown as text, **either decimals or integers times one scale**; edits are a small op language (`assign`, `replace`, `add_rank1`, `project_out`, `revert`) grouped into named edit sets; a Rust gate enforces per-task policy (allowed and forbidden tensors, no whole-tensor constant scaling, a cap on the fraction changed, finite values); append-only hashed ledger; checkpoint and restore; budgeted probes; soft static checks and a reviewer-model audit; each task declares its access level and results from different access levels must not be pooled.
- **Robot policy paper** (arXiv 2611.09376) [read, abstract]: 65 token-embedding rows edited; best session 86.3% to 92.0% with 48 repairs and 4 regressions; matched random-direction edits did not help; in 13 further sessions the editor declined 12 candidate edits.
- **Reading critique** (Ahmed, arXiv 2611.08070) [read]: with weights as plain text rounded to three decimals and no tools, Claude Opus 5.5 does correct local arithmetic at the edges of small networks but leans on dataset priors; a label-permutation control broke every interpreted class pair.

What this means for cost: in every public harness the wall clock is dominated by generation-based forward passes and by re-starting the inference server per edit, not by the editor's reasoning or by weight analysis. A harness for a 31B model should attack exactly those two costs.

---

## (a) Editing quantised or QAT weights

**A1. Hold the edit as an overlay on the dequantised weight; do not requantise in the loop. Established.**
Mechanism: compute `W_eff x = unpack(q, s) x + B (A x)` (low rank) or patch selected columns or rows, as extra terms in the forward pass. Precedents:
- QLoRA (Dettmers et al., arXiv 2305.14314) [read]: frozen 4-bit base, full-precision low-rank adapters on top; 65B fine-tuned on one 48 GB GPU.
- Heretic [docs mirror, snippet; README and default config read]: implements directional ablation as LoRA adapters rather than by rewriting weights, so a trial is reset by zeroing the adapter, only low-rank deltas are stored, and it works on bitsandbytes 4-bit models; merging afterwards requires reloading the base in full precision on CPU.
- CACHE-UK (arXiv 2607.28292) [read]: applies a ROME-style rank-one update to the LoRA B matrix on a 4-bit base instead of to the quantised weight, explicitly to avoid dequantising the base.
Cost: Derived, 54 KB per rank-1 overlay on a feed-forward matrix; one extra pair of small matmuls per edited matrix per forward.

**A2. Requantising silently deletes small edits. Established.**
- Zhang et al., "Catastrophic Failure of LLM Unlearning via Quantization" (arXiv 2410.16454, ICLR 2025) [read]: weight changes smaller than the quantisation step are mapped back to the same code; utility-preserving unlearning retained 21% of the forgotten knowledge at full precision and 83% after 4-bit quantisation; 8-bit behaved like full precision; RTN, GPTQ and AWQ at 4 bits behaved alike. Remedy studied (SURE): larger, saliency-masked updates, at a utility cost.
- CACHE-UK [read] names the same thing a resolution barrier for rank-one edits; its measured 4-bit numbers (edit success about 100%, held-out success 12 to 22%) come from one small model with re-implemented baselines and no full-precision control, so treat the size of the effect as unproven.
- PEFT emits a warning that merging LoRA into a 4-bit linear layer can change generations through rounding [snippet]; practitioners advise dequantise, merge, then requantise [snippet].
- WSP-Scope [read] saw the mild form of the same effect at bf16.
Derived. With symmetric int4 and one scale per group of 32 inputs, an entry changes code only if the edit moves it across a cell boundary, i.e. by roughly half a scale or more. A dense rank-one update spreads small changes over 115.6M entries; most will round back, and the survivors are a biased sample of the intended edit. Column and row edits of large magnitude (zero, negate, scale by 2) survive much better.

**A3. Requantising with a different rule damages even untouched weights. Established for this model family.**
Unsloth's Gemma 4 QAT page [read]: the QAT weights sit on a specific lattice with bf16 scales; a naive Q4_0 conversion reproduced 24.77% of bytes and lost about 9 points of top-1 agreement on 31B. So on export, groups an edit did not touch must keep their original codes and scales.

**A4. What an edit "means" under QAT. Emerging; no direct measurement found.**
- Community guidance for Gemma 4 QAT [Unsloth discussion #6389, read; a practitioner skill file, snippet]: there is no supported way to keep the shipped Q4_0 calibration through adapter training; bitsandbytes NF4 is a different quantiser from Q4_0; recommended path is 16-bit LoRA on the unquantised QAT checkpoint with the adapter kept separate so the base stays unchanged, then requantise only if a merged file is needed. Nobody in the thread measured the loss.
- The dequantisation card [read] shows the QAT model and the bf16 release are materially different models (top-1 agreement 67.8% on its sample). Derived: directions, dictionaries and statistics computed on the bf16 release or the base model should not be assumed to carry to the QAT instruction-tuned checkpoint without a check. TrevorS's abliteration repo [read] reports that directions computed on a 4-bit load and on bf16 have mean cosine 0.935: close, not identical.

**A5. Adapters that live on the quantisation grid. Emerging.**
- QA-LoRA (Xu et al., arXiv 2309.14717) [read, abstract]: group-wise operators make the adapter constant within each quantisation group so it merges into the quantised model without loss; per the appendix summary [snippet] the merge only changes the zero-point.
- LoTA-QAF (arXiv 2505.18724) [read, abstract]: ternary adapters aligned to the grid, merged losslessly into 4-bit weights, updated with a ternary sign-SGD.
- L4Q (arXiv 2402.04902) [read, abstract]: LoRA integrated with QAT.
Derived. QA-LoRA's exact merge relies on a per-group zero-point. The Gemma 4 format, per the third-party card, is symmetric with no zero-point, so that trick does not transfer as is.

**A6. Lattice-native edits. Speculative (my derivation from the format; untested).**
- Zeroing any entry, column, row or head is exactly representable (code 0).
- Negation is representable except where a code is -8 (the range is -8..7).
- Multiplying a whole **row** of `up_proj` or `gate_proj` (one neuron's input weights, 168 groups of 32) by any factor is exactly a change of that row's 168 scales (up to bf16 rounding of the scale). If the feed-forward block is the usual gated form `down(act(gate x) * up x)`, scaling a neuron's `up_proj` row by a factor scales that neuron's output by the same factor, which is the lattice-exact equivalent of the "scale one `down_proj` column" edits that WSP-Scope found most often. Scaling one `down_proj` column directly touches one entry in each of 5,376 groups and is generally off the lattice.
- Integer code steps (plus or minus one) and single-group scale changes are the native "pixels" of this model. OpenWiSH's integers-times-one-scale text view is the matching read format.
This needs a bit-level test against the kernel before anyone relies on it.

---

## (b) Evaluating many weight variants at once

**B1. Per-row low-rank deltas in one batched forward. Established.**
- PEFT mixed-adapter batches [docs read]: pass one adapter name per sample (`"__base__"` selects the unedited model); inference only; not with merged adapters or DoRA; the docs warn of overhead because the batch is effectively split per adapter and suggest larger or homogeneous batches, or a dedicated server.
- FLoRA / fLoRA (Wen and Chaudhuri, arXiv 2312.05677, ICLR 2024) [read]: each example carries its own adapter, vectorised with element-wise products so the only matmul is the shared one; more than 3x the throughput of batched-matmul LoRA at rank 1 on a 15B model, about 2.5x at rank 2, advantage reversing at rank 8. Note fLoRA is a multiplicative parameterisation, not additive LoRA.
- Punica (arXiv 2310.18547) and S-LoRA [snippets]: custom kernels that batch different LoRA adapters over one shared base; Punica reports 12x throughput on A100 clusters with 2 ms added per token; SGLang ships both ideas.
- EGGROLL (Sarkar et al., arXiv 2511.16652) [read, abstract]: unstructured per-member perturbations make batched evaluation prohibitively slow on GPUs; rank-r perturbations per population member reach up to 91% of plain batch-inference throughput.
Derived cost: an additive rank-r delta costs O(r (d_in + d_out)) per token against O(d_in d_out) for the base matmul, so at r up to a few dozen the variants are nearly free relative to the shared packed matmul. A rank-one feed-forward edit `W + u v^T` is exactly the activation-level operation "add (v . a) u to the residual", and a projection edit `(I - u u^T) W` is exactly "project u out of what this block writes" (the equivalence underlying weight orthogonalisation in Arditi et al., arXiv 2406.11717 [abstract read; the equivalence statement itself not verified on the page]). Both can therefore be evaluated per batch row as hooks without touching the packed weights. So can column scale, zero and negate (a per-row gain on one neuron's activation).

**B2. `torch.func.functional_call` plus `vmap` over stacked parameters. Established for small models; poor fit here.**
PyTorch ensembling tutorial [read]: `stack_module_state` stacks N copies of every parameter along a new leading dimension, `functional_call` runs a stateless template, `vmap` maps over the stack; `in_dims=(0, 0, None)` shares one minibatch across all members; operations without a vmap rule fall back or fail. Derived: stacking costs N times the parameter memory (16 dense variants of one feed-forward matrix = 3.7 GB in bf16), and a custom int4 kernel is unlikely to have a vmap rule. Use it, if at all, for N dense variants of a single unpacked matrix; use B1 for everything low-rank.

**B3. Forward-only search over many variants. Emerging.**
MeZO (Malladi et al., arXiv 2305.17333) [snippet]: zeroth-order updates in place at inference memory; a 30B model on one 80 GB A100 against 2.7B with backpropagation. Evolution strategies on LLMs (Qiu et al., arXiv 2509.24372; EGGROLL) [snippets and abstract]. These are optimisers, which the weight-space-programming papers exclude, but their batching machinery (B1) is exactly what a hypothesis-testing harness needs: test K candidate directions at plus and minus epsilon in one batch of 2K rows.

---

## (c) Partial recomputation

**C1. Cache the residual stream below the lowest edited layer and resume there. Established as an API, unbenchmarked in print.**
- TransformerLens exposes `start_at_layer`, `stop_at_layer` and `past_kv_cache` on `forward` [signature seen in a search snippet; I could not open the docstring, both URLs returned 404 after the v3 migration]. From memory, not verified today: `start_at_layer` takes the residual stream entering that layer as input and `stop_at_layer` returns the residual stream.
- nnsight's best-practice guide [snippet] describes stopping the forward early and claims 2 to 5x on large models, without a benchmark. pyvene has only an open feature request for activation caching [snippet].
- The AtP* paper [read] counts activation patching as one full forward per intervention and does not mention reusing computation below the patched node.
- Egeria [snippet] caches frozen front-layer outputs during training: the same idea in a different setting.
Derived. An edit confined to layers at or above L leaves the residual entering layer L unchanged for every prompt. Cache it once per prompt (all positions; 10.75 KB per token), then each variant runs only layers L to 59: a factor of about 60 / (60 - L), so 2x at layer 30, 4x at 45, 12x at 55. This composes with B1 (many variants per resumed batch) and with F (the backward pass also stops at L).

**C2. KV-cache reuse. Speculative (inference; no source).**
For generation after an edit at layer L, the cached keys and values of layers below L for the shared prompt prefix remain valid; layers at or above L must be recomputed for the prefix from the cached residual. New tokens need all layers. The technical report [read] says the 31B's KV cache is int8 in the Q4_0 memory budget, which matters if cached keys and values are to be compared exactly.

**C3. Early exit on the metric. Speculative.**
LayerSkip (Elhoushi et al., arXiv 2404.16710) [snippet]: early exit works when the model was trained for it (layer dropout plus early-exit loss, 1.34x to 2.16x). Derived: on a model not trained for early exit, a lens readout at an intermediate layer is a screening signal only. A safer saving is the other end: stop computing once the metric's token position has been produced (teacher-forced scoring needs one forward, no generation).

**C4. Zero-forward screening through a frozen-norm linearisation. Emerging.**
Goodfire's "Neurons as an Interpretable Basis" (10 Nov 2026) [read, first 100k characters] estimates a neuron's downstream readers by propagating its write to later layers to first order with RMSNorm scales held fixed, and estimates inputs by decomposing gate and up pre-activations into contributions from earlier writes. Derived: the same linearisation predicts the effect of a candidate write change on a logit difference without running the model, which is a cheap pre-filter before any forward pass. Accuracy at this depth on this model is unmeasured.

**C5. Keep the model resident; never reload per edit. Established by counter-example.**
HomoHabilis [read] pays 3 to 5 minutes of server start-up per forward pass because each edit is a new checkpoint for vLLM. An overlay design (A1) makes an edit a few kilobytes pushed to a resident process.

---

## (d) Cheap evaluators

**D1. Score with probabilities, not samples; compare paired. Established.**
Miller, "Adding Error Bars to Evals" (arXiv 2411.00640) [read]: report standard errors; cluster them when questions are related (clustered errors can be about 3x the naive ones); when no chain of thought is needed, score each question by the probability of the correct token, which removes sampling noise entirely; compare two models on per-question differences, which is free variance reduction when scores are correlated; detecting a 3-point difference at 5% significance and 80% power needs roughly 969 independent questions under the paper's assumed parameters. WSP-Scope [read] already reports paired differences clustered by question.

**D2. How many prompts. Established with a caveat.**
- tinyBenchmarks (Polo et al., arXiv 2402.14992) [read, abstract]: 100 curated examples suffice to estimate MMLU (14K items); the within-about-2% figure is from a search summary [snippet].
- A 2025 position paper (arXiv 2503.01747) [snippet] argues CLT error bars are badly too narrow below a few hundred datapoints.
- Heretic's defaults [config read]: 100 harmless prompts for KL, 100 harmful prompts for the refusal count, 400 plus 400 prompts to compute directions, 100 trials of which 30 random; README: 30 to 40 minutes for a 4B model on an RTX 3090 (a mirror of the docs says 200 trials [snippet]; the config file I read says 100).
- kyle-chalmers [read]: 150 paired answers were enough to separate the published edit (fixed 70, broke 1) from a random direction.
- TrevorS [read]: a 150-prompt seeded subset for iteration, 686 prompts for final validation, 10 harmless prompts for a length-ratio check.
Derived. A continuous, paired, teacher-forced metric (logit difference or log-probability of a target against a foil) on 50 to 200 prompts is the right inner-loop evaluator; generation-based accuracy on hundreds to a thousand items is the outer confirmation.

**D3. KL to the unedited model on unrelated text as the side-effect measure. Established in practice; known pitfalls.**
- Heretic co-minimises refusals and KL on harmless prompts [README and config read]; the README does not define the KL (first token or all positions).
- Pitfall on Gemma 4 specifically (TrevorS) [read]: reserved-vocabulary tokens emit minus-infinity logits, which turn the KL into NaN unless non-finite terms are masked.
- Pitfall: KL values shift across CUDA versions and GPU architectures [snippet].
- Low KL does not certify task behaviour: Goodfire [read] ablated about 92% of routed neuron activations for KL 0.140 and +2.9% loss, yet benchmarks moved from -20.9 (MBPP+) to +15.9 (HumanEval+).
- Equivalence checks used in the wild: mean absolute log-prob difference and top-1 agreement over about 25k tokens (dequantisation card) [read]; top-1 and mean KLD against the bf16 QAT reference (Unsloth) [read].

**D4. Controls that the follow-up papers show are necessary. Established within this literature.**
Random direction on the same neurons; random location of the same size; held-out set never shown to the editor; keep-original as a legal outcome; a fixed panel of unrelated tasks watched on every edit (XLRA's 1,000-question panel was added after an unwatched benchmark lost 4.6 points); the ROME triple of efficacy, paraphrase generalisation and neighbourhood specificity [snippet]; repeat searches, since three searches on one task gave +29.45, +5.00, +27.30 (WSP-Scope).

**D5. Covariance statistics for closed-form edits are cheaper than they look. Established.**
Gupta et al., ACL 2025 short (arXiv 2506.04226) [snippet]: MEMIT's original precomputation was about 44 million hidden vectors per edited layer (36 to 40 hours on one GPU for 6 to 7B models per a secondary summary); fewer than 0.3% of that suffices.

---

## (e) Reading weights at scale for an LLM reader

**E1. Raw numbers in the prompt do not scale. Established.**
The original post [read] gives weights as text only up to about 172k weights; a 7M-weight model already needed Python (up to 100 runs). Ahmed [read]: three sessions on 125k-weight networks hit the 128,000-token output limit. A local 27B model took 46 to 65 minutes per 17k-weight network (Willison) [read]. For 30.7B weights the only workable form is tool-mediated access.

**E2. A sandboxed kernel over lazily loaded tensors, plus harness-side services. Established pattern.**
WSP-Scope and OpenWiSH [read]. Derived for this model: the dequantised checkpoint is 62.5 GB, so the kernel must load per tensor (memory-mapped or unpack-on-demand from the packed file), and expensive factorizations (SVD of a 21,504 x 5,376 matrix) should be a cached harness service on the GPU rather than something the agent spends its per-execution time limit on.

**E3. Candidate representations, with evidence.**

| Representation | Evidence | Status at 31B |
|---|---|---|
| Projection of weights to the vocabulary (zero-pass) | Dar et al., arXiv 2209.02535 [read, abstract]: all parameters projected to embedding space, no forward or backward pass. Jacobian-lens line of work (Gurnee et al. 2026; arXiv 2610.31955) [snippets] | Established idea; early layers read poorly per an explainer [snippet] |
| SVD factors of weight matrices | "SVD as a Fast Interpretability Method for Transformers" (NaNA, ICML 2026 spotlight) [snippet]: detector-effector units from SVD of feed-forward weights, GPT-2 family; SITH for CLIP (CVPR 2026) [snippet] | Emerging; untested on a 31B |
| Neuron dictionary: per-neuron description, firing examples, ranked input neurons with attribution scores, reader neurons | Goodfire, 10 Nov 2026 [read]: Qwen3.6-35B-A3B; inputs chosen by weight alignment with the target's gate or up weights or by attribution; firings clustered into subcircuits; descriptions validated by detection (95% against random text, 89% against same-expert text), simulation (73% of KL damage recovered above p99.9) and input synthesis (94% of neurons exceeded the corpus maximum); median pole has one mode, 69% dominated by one mode | Emerging, at comparable scale |
| Public neuron dashboards for this exact model | Neuronpedia MLP Neurons docs [read]: Gemma 4 31B IT, 60 layers, 21,504 neurons per layer, inference enabled (test with own text, steer, search) | Exists |
| Sparse autoencoders | `decoderesearch/gemma-4-saes` (Chanin, June 2026) [read]: Matryoshka BatchTopK residual SAEs; for 31B **one layer only (30), 131k latents**, trained on the base model, training precision not stated; E2B and E4B have three layers each | Thin coverage |
| Transcoders, cross-layer transcoders, attribution graphs | **None published for Gemma 4** as of an issue opened 29 Sep 2026 with comments to 10 Nov 2026 [read]; circuit-tracer has no Gemma 4 architecture mapping; Neuronpedia's home page lists only "Gemma 4 SAEs" for Gemma 4 [read]. Gemma Scope 2 (19 Dec 2025) covers Gemma 3 only [snippet]; no "Gemma Scope 3" found. Training cost per the issue: about 1B tokens per layer, a multi-day run; a 262k cross-layer set for a 26-layer model is about 33 GB of fp32 weights | Not available |
| Transcoder circuits as weights-based, input-invariant terms | Dunefsky et al., arXiv 2406.11944 [read, abstract] | Established on small models |
| Attribution graphs summarised for an LLM | "LLMs Can Annotate Attribution Graphs", arXiv 2608.02632 [read, abstract]: feature descriptions grouped into supernodes; 97 of 100 on a two-hop task | Emerging; needs transcoders |
| Parameter decomposition | SPD (Bushnaq, Braun, Sharkey; Goodfire post 28 Jun 2025) [read]: rank-one subcomponents; toy models only; language models named as future work | Not available |
| Individual weights | Marin-Llobet and Heimersheim, arXiv 2607.02964 [snippet]: 12 to 31% of weights in weight-sparse transformers have a single short description; an earlier listing gives 5 to 9% for dense models | Suggests the unit for a dense model is the neuron, column or direction, not the entry |
| Fine-tuned weight readers | Weight Oracles, arXiv 2610.07334 [read, abstract]: 99% hold-out accuracy simulating small transformers; scaling named the principal open problem | Not available at scale |
| Integer codes times scale | OpenWiSH [read] | Natural for an int4 lattice (see A6) |

**E4. Transfer of dictionaries across checkpoints must be measured. Emerging.**
The interpretune issue [read] measured transcoder drift on fine-tuned checkpoints with a gate (error-node share may rise at most 5 points, replacement may fall at most 0.10) and found some checkpoints inside and some outside. Combined with the 67.8% top-1 agreement between the bf16 release and QAT (section 0), any dictionary trained off the QAT checkpoint needs a reconstruction check on it, and an edited model needs the check repeated.

---

## (f) Gradients as a tool rather than a training loop

**F1. Attribution patching: all nodes for one prompt pair in two forwards and one backward. Established.**
AtP* (Kramár, Lieberum, Shah, Nanda, arXiv 2403.00745) [read]: activation patching costs one forward per node (the paper's example: about 2.7 billion neuron nodes for a 1,024-token prompt on a 70B model); AtP needs two forwards and one backward; two failure modes (attention saturation, cancellation of direct and indirect effects) with fixes costing under two extra forwards (QK fix) and L extra backwards, reducible to about (L+1)/2 (GradDrop); use AtP as a pre-filter and verify the top-ranked nodes by real patching; subsampling bounds the false negatives; tested on Pythia up to 12B; plain AtP is the recommended choice for neuron nodes on distributions.
EAP-IG (Hanna et al., arXiv 2403.17806) [snippet]: integrated gradients along the corrupted-to-clean path, about m backwards.

**F2. Attribution to weights without materialising weight gradients. Established identity; application here is derived.**
Goodfellow, arXiv 1510.01799 [snippet]: for a linear layer the per-example weight gradient is the outer product of the backpropagated signal and the layer input. Derived: the gradient of a metric with respect to a feed-forward matrix is a sum over token positions of such outer products, so it can be kept in factored form (64 positions x (21,504 + 5,376) fp32 values = 6.9 MB instead of 462 MB) and the packed weights never need `requires_grad`. Gradient-times-weight, Fisher-style scores (Model Shapley, NeurIPS 2025, one backpropagation [snippet]) and saliency masks (SURE [read]; WAGLE, arXiv 2410.17509 [snippet]) are then cheap post-processing.

**F3. Memory cost on a quantised model. Established by QLoRA.**
QLoRA [read; HF blog snippet]: gradients flow through a frozen 4-bit base; memory is dominated by activation gradients, not adapter parameters; 65B on 48 GB (paper), 33B on 24 GB (blog). Derived: a single backward on a short prompt through a 31B packed model fits a 32 GB card, provided the dequantisation step is differentiable with respect to its input activations (bitsandbytes-style unpack-then-matmul is; a custom int4 kernel may not be). Combined with C1, detach the cached residual at layer L and both passes cover only layers L to 59.

**F4. When no backward is available. Emerging.**
MeZO and evolution-strategy work [snippets, EGGROLL abstract read] estimate directional derivatives from forward passes only; with B1, finite differences along K low-rank directions cost one batch of 2K rows.

**F5. What the weight-space-programming papers allow.**
- Original post [read]: gradients were not allowed in its experiments, and the authors say the concept does not exclude them.
- WSP-Scope [read]: no backward passes, optimisers or iterative loss fitting.
- OpenWiSH [read]: backward and optimiser calls raise in the kernel; fitting, `scipy.optimize` and gradient steps are flagged; a violation invalidates the run.
- Robot paper [read, abstract]: no gradient updates or new demonstrations.
- Counter-evidence on usefulness: wsp-sql [read] found gradient descent on the same entries reaches the same bypass; XLRA [read] found GRPO beats edits.
None of the papers distinguishes a single attribution backward from an optimisation loop; that distinction is a design choice a new harness has to make and record.

---

## (g) Published weight-level work on Gemma 4 specifically

- **Abliteration (many community builds).**
  - TrevorS/gemma-4-abliteration [README and ABLITERATION.md read]: norm-preserving biprojected abliteration of `o_proj` and `mlp.down_proj`; per-layer directions from 800 harmful and harmless prompt residuals, orthogonalised against harmless means; **31B: 22/686 refusals, KL 0.124**; 31B exported by computing directions on a 4-bit load and writing bf16 shards (direction cosine 0.935 against bf16); a biprojection run takes about 2 minutes and a 50-trial search 6 to 7 minutes (model size for these timings not stated); on the 12B the signal lives in layers 15 to 47 and layers 0 to 14 carry none; for the MoE model each expert's `down_proj` slice is ablated separately; task-vector negation showed a sharp phase transition with no usable strength. Pitfalls: minus-infinity logits for reserved tokens (NaN KL), `Gemma4ClippableLinear` wrappers in the vision and audio encoders break PEFT targeting unless modules are named by full path, transformers 5.10.1 or later is required, and refusal-then-comply defeats substring scoring.
  - 199-biotechnologies [read]: edits bf16 shard by shard, then quantises to 4-bit MLX; pipeline needs 64 GB; evaluation is 100 self-run prompts.
  - WWTCyberLab [read]: five stacked edits including a merged rank-32 LoRA on layers 57 to 59, ablation at layers 33 to 52 and 55 to 59, and scaling 13 token-embedding rows by one tenth; MMLU 45.5% against 48.0%.
  - HauhauCS QAT build [read]: built from the official QAT weights, released as Q4_K_M; method undisclosed; 0/465 refusals claimed.
  - Others seen only as search hits: null-space, paperscarecrow, amarck (Heretic, KL 0.27), cs2764, a twelve-variant 12B comparison.
  None of these pages reports what requantisation did to the edit.
- **Dictionaries.** decoderesearch SAEs (layer 30 only for 31B) [read]; Neuronpedia MLP neuron dashboards for 31B IT [read].
- **Activation-level studies.** Buehler, arXiv 2607.20058 [read, abstract]: Jacobian readouts and hidden-state patches on E4B, 12B and 31B instruction-tuned models; no weight edits stated.
- **Quantisation artefacts.** The bf16 dequantisation of the W4A16 checkpoint [read]; Unsloth's lattice-aware conversion [read].
- **Not found.** No ROME, MEMIT or AlphaEdit paper on Gemma 4; no parameter decomposition; no transcoders or attribution graphs; no weight-space-programming experiment on any Gemma 4 model or on any quantised model above 9B (the only quantised targets are 4-bit Qwen3.5 4B and 9B in one replication kit [read], whose page does not say how edits meet the quantiser).

---

## Established, emerging, speculative: one table

| Technique | Status | Key source |
|---|---|---|
| Low-rank or column overlay on a frozen 4-bit base | Established | QLoRA; Heretic; CACHE-UK |
| Small edits vanish on requantisation | Established | Zhang et al. 2410.16454 |
| Requantising Gemma 4 QAT with another rule degrades it | Established (vendor numbers) | Unsloth |
| Grid-aligned adapters that merge losslessly | Emerging | QA-LoRA; LoTA-QAF |
| Lattice-native edits via group scales and codes on Gemma 4 | Speculative | derived from the format |
| Per-row adapters in one forward | Established | PEFT; FLoRA; Punica; S-LoRA; EGGROLL |
| vmap over stacked full weights | Established small, poor fit at 31B | PyTorch tutorial |
| Resume from cached residual at the lowest edited layer | Established API, no published benchmark | TransformerLens signature; nnsight guide |
| KV reuse below the edit; early exit on a lens metric | Speculative | LayerSkip as nearest analogue |
| Frozen-norm linear pre-screen | Emerging | Goodfire 2026 |
| Paired, probability-based scoring; clustered errors | Established | Miller 2411.00640 |
| KL on unrelated text as side-effect measure | Established in practice, insufficient alone | Heretic; Goodfire |
| Random-direction and random-location controls | Established in this literature | kyle-chalmers; wsp-sql; 2611.09376 |
| Neuron dictionary with wiring as the reading unit | Emerging at 35B scale | Goodfire 2026 |
| Transcoders or attribution graphs on Gemma 4 | Not available | interpretune issue; Neuronpedia |
| AtP as a single-backward pre-filter | Established to 12B | AtP* |
| Factored weight gradients | Established identity, derived use | Goodfellow 2015 |
| Backward through a 4-bit base on one card | Established | QLoRA |

---

## Could not find or open

- An arXiv id for the original weight-space-programming report (it exists as a blog post and logs).
- The official quantisation config of `google/gemma-4-31B-it-qat-w4a16-ct` (scheme details come from a third-party card and a recipe summary).
- Any measurement of what a weight edit loses when a Gemma 4 QAT model is requantised, or of LoRA on the unquantised QAT checkpoint followed by Q4_0 export.
- The TransformerLens docstring for `start_at_layer` (404 on the docs page and on the raw source path).
- A published benchmark of resume-from-layer evaluation, or any source on reusing KV caches across weight edits.
- Wall-clock or money cost per trajectory in the original post or WSP-Scope.
- The later part of the Goodfire page (agentic neuron science, costs); the fetcher returned only the first 100k characters.
- Heretic's exact KL definition (first token or all positions).
- The primary posts by Honnibal and Goedecke (seen only through summaries).
- How the HomoHabilis kit applies edits to its 4-bit targets.

## Sources

Read (opened):
- https://rikiyat.github.io/blog/weight-space-programming/
- https://arxiv.org/html/2611.09382
- https://arxiv.org/html/2611.08070v1
- https://arxiv.org/abs/2611.09376
- https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/
- https://nishbhana.com/Weight-Space-Programming/
- https://github.com/lemonadesaltbagel/OpenWiSH
- https://github.com/HomoHabilis/weight-space-programming
- https://github.com/kyle-chalmers/weight-space-programming
- https://github.com/Exorust/wsp-sql
- https://github.com/XLRA/weight-space-coding
- https://arxiv.org/abs/2610.07334
- https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/
- https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct
- https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf
- https://huggingface.co/mliu444/gemma-4-31b-it-qat-w4a16-dequant-bf16
- https://arxiv.org/html/2607.02770v2
- https://unsloth.ai/docs/models/gemma-4/qat
- https://github.com/unslothai/unsloth/discussions/6389
- https://arxiv.org/abs/2305.14314
- https://arxiv.org/abs/2309.14717
- https://arxiv.org/abs/2505.18724
- https://arxiv.org/abs/2402.04902
- https://arxiv.org/html/2410.16454v3
- https://arxiv.org/html/2607.28292v1
- https://huggingface.co/docs/peft/developer_guides/lora
- https://arxiv.org/html/2312.05677v3
- https://docs.pytorch.org/tutorials/intermediate/ensembling.html
- https://arxiv.org/abs/2511.16652
- https://arxiv.org/html/2403.00745v1
- https://arxiv.org/html/2411.00640v1
- https://arxiv.org/abs/2402.14992
- https://github.com/p-e-w/heretic and its config.default.toml
- https://www.goodfire.com/research/neurons-as-an-interpretable-basis (first 100k characters)
- https://www.goodfire.com/research/stochastic-param-decomp
- https://huggingface.co/decoderesearch/gemma-4-saes
- https://www.neuronpedia.org/ , /gemma-4-31b , /gemma-4-31b-it , /gemma-4-saes
- https://docs.neuronpedia.org/neurons
- https://github.com/speediedan/interpretune/issues/709
- https://arxiv.org/abs/2406.11944
- https://arxiv.org/abs/2209.02535
- https://arxiv.org/abs/2608.02632
- https://arxiv.org/abs/2406.11717
- https://arxiv.org/abs/2607.20058
- https://github.com/TrevorS/gemma-4-abliteration and its ABLITERATION.md
- https://github.com/199-biotechnologies/gemma-4-abliterated
- https://huggingface.co/WWTCyberLab/gemma-4-31B-it-abliterated
- https://huggingface.co/HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP

Search snippet only:
- arXiv 2310.18547 (Punica); S-LoRA; SGLang LoRA docs
- arXiv 2305.17333 (MeZO); arXiv 2509.24372 (ES at scale)
- arXiv 2404.16710 (LayerSkip)
- arXiv 2403.17806 (EAP-IG)
- arXiv 1510.01799 (per-example gradients)
- arXiv 2506.04226 (minimal precomputation for MEMIT)
- arXiv 2503.01747 (CLT caveat)
- arXiv 2607.02964 (individual parameters in weight-sparse transformers)
- ICML 2026 "SVD as a Fast Interpretability Method for Transformers"
- Gemma Scope 2 pages; arXiv 2610.31955 (Jacobian lens theory)
- TransformerLens `forward` signature; nnsight best practices
- Heretic docs mirror (LoRA-based ablation, trial count)
- PEFT 4-bit merge warning; Kaitchup posts
- ROME and MEMIT metric definitions
- arXiv 2410.17509 (WAGLE); Model Shapley (NeurIPS 2025)
- arXiv 2611.11604, 2611.09559, 2611.04268, 2611.05341, 2611.05798 (other weight-space-programming follow-ups, not relevant to this angle)

## Independent citation check

Done 14 November 2026 by a separate checker. Each source below was opened directly (page fetch, and for the key numbers a raw-text grep of the page). Verdicts: supported / partly / not_supported / could_not_open. Nothing cited turned out not to exist.

Text addressed to AI agents found on pages (ignored): the Unsloth docs page carries an "Agent Instructions" block asking agents to query its docs over HTTP; Hu et al. reproduce their full editor prompt in an appendix (part of the paper, not a directive to a reader); the kyle-chalmers and TrevorS repos list AGENTS.md / CLAUDE.md files, not opened.

### Checks

1. **Original post (Takehi, Teoh, Zhang, Khattab), harness and numbers** — https://rikiyat.github.io/blog/weight-space-programming/ — **supported.** Dated 2 Nov 2026, response to comments 12 Nov. "A Python kernel over the weights (no GPU)", 10, 25 or 50 forward passes, a pass is up to 512 problems or up to 1,024 texts. 131,072 weights (0.016%), 64 output columns in the MLPs of layers 15 and 19. MATH-500 7.9% base, 12.3 / 14.1 / 48.1 at 10 / 25 / 50 passes. Gemma 3 270M: 6.8% to 7.0, 7.6, 7.7, 7.0 at 5, 10, 25, 50 passes. "As a concept, Weight-Space Programming does not rule out gradients ... In this blog, however, we did not allow gradients." No quantisation, no time or cost figure.

2. **Hu et al., arXiv 2611.09382** — https://arxiv.org/html/2611.09382 — **supported, one correction.** 6 Nov 2026. CPU-only persistent kernel, 900 s per execution, at most 300 analyses, separate GPU worker, generation and teacher-forced requests, 50 successful requests per search, "no backward passes, no optimizers, and no iterative loss fitting", float32 arithmetic with BF16 storage and updates below BF16 spacing vanishing, "six of the 16 trajectories retain the original weights", +29.45 / +5.00 / +27.30 on three Math searches, 9B Math +29.69 development and -0.35 held out, 1,023 changed values. In Table 2 the held-out gain is below the best development gain in all 14 rows. Correction: the report says the paper gives no time cost; the reproduced editor prompt says a generation pass "takes up to about half an hour at full size" and a teacher-forced pass "seconds to a few minutes", and the Math search used 32 generation and 18 teacher-forced requests. No money figure is given.

3. **Replicator costs and the claim that all public harnesses share one shape** — https://github.com/HomoHabilis/weight-space-programming , https://github.com/XLRA/weight-space-coding , https://github.com/kyle-chalmers/weight-space-programming , https://github.com/Exorust/wsp-sql , https://github.com/lemonadesaltbagel/OpenWiSH — **costs supported; "all" is partly supported.** HomoHabilis: 8 GB RTX 4070 laptop, "roughly 15-30 min" per 128-answer pass, 3 to 5 minutes of vLLM start-up per pass, 6 to 12 hours per run, about 30 to 80 dollars per 10-pass run; these are for Qwen3.5-0.8B writing about 16k tokens per answer. wsp-sql: "about 10 hours on an M-series laptop, most of it forward passes". The generalisation does not hold for every repo: XLRA allowed 10 to 24 passes per session and used 70 over four sessions; the kyle-chalmers blind attempt had only a probe table and an edit tool, no kernel and no stated budget; the HomoHabilis page states no total pass budget and does not say the kernel is CPU-only; OpenWiSH has other probe types and budgets as well (for example 3 forward/patch passes on its decompile task).

4. **W4A16 format and memory** — https://huggingface.co/mliu444/gemma-4-31b-it-qat-w4a16-dequant-bf16 , https://arxiv.org/html/2607.02770v2 — **supported, with two nuances.** The third-party card says pack-quantized int4, symmetric, group size 32 along the input dimension, one bf16 scale per group, no zero point, 410 quantised linear tensors, 778 others, 62.5 GB, top-1 agreement 97.4% against the W4A16 kernel and 67.8% between the bf16 release and QAT (8 trajectories, 24,568 tokens, one measurement). The technical report's Table 3 gives 31B at 64.0 GB bf16 and 19.2 GB Q4_0; that is weights only, and an int8 KV cache at 32k adds 1.10 GB. The sources list says the report describes Q4_0 as "blockwise with a scalar scale at each block so inference works in fp16"; the report's sentence about "a scalar scale at each block" is a separate statement about bounding activation ranges for fp16 inference, not a description of the Q4_0 format.

5. **Requantising** — https://arxiv.org/html/2410.16454v3 , https://unsloth.ai/docs/models/gemma-4/qat — **Zhang et al. supported; Unsloth partly.** Zhang et al. (arXiv comments field: ICLR 2025): unlearned models retain "an average of 21%" of forgotten knowledge at full precision and 83% after 4-bit quantisation, explained by weight changes smaller than the quantisation step; 8-bit behaves like full precision; GPTQ and AWQ behave like RTN. Unsloth: "Naive conversion gets 24.77% byte exactness to BF16 QAT, whilst we found we can push it to 99.96%"; the figure is not tied to the 31B. The 31B row is Unsloth's own conversion at 96.67% top-1 against naive Q4_0 at 87.91%: two alternative conversions, not a before-and-after drop, the reference for top-1 is not stated in that table, and this is a vendor page describing its own product.

6. **Many edits in one pass** — https://huggingface.co/docs/peft/developer_guides/lora , https://arxiv.org/html/2312.05677v3 , https://arxiv.org/abs/2511.16652 — **PEFT and EGGROLL supported; FLoRA partly.** PEFT: `adapter_names` with `"__base__"`, inference only, not with merged adapters or DoRA, and "the batch size is effectively reduced to the number of samples per adapter". EGGROLL abstract: "up to 91% of the throughput of pure batch inference". FLoRA: more than threefold at rank 1 and about 2.5 times at rank 2 on StarCoder 15B, reversing at rank 8, as stated. But FLoRA's fast path is not an additive low-rank delta: it defines W_i = dW_i ∘ W_0 (element-wise product), i.e. per-example scaling of a layer's inputs and outputs, and the baseline it beats is the additive LoRA done with torch.bmm. So FLoRA's number supports per-row neuron gains, not per-row additive rank-one edits; for those the relevant evidence is the slower bmm baseline, PEFT and EGGROLL.

7. **Miller, arXiv 2411.00640** — https://arxiv.org/html/2411.00640v1 — **partly.** The paper does recommend next-token probabilities, paired differences and clustered errors (over 3x on one benchmark), and gives n of about 969. But the 969 figure is computed for a paired comparison with zero conditional variance (that is, already scored by probability) and an assumed variance of 1/9, with parameters the author calls "fictional". It is therefore the requirement in the best case, not a cost peculiar to generation accuracy, and it does not show that probability scoring gets by with fewer questions. The report's "which is why" is the researcher's inference; the paper's own conclusion is that an eval should have at least 1,000 questions.

8. **Goodfire, "Neurons as an Interpretable Basis"** — https://www.goodfire.com/research/neurons-as-an-interpretable-basis — **numbers supported; framing partly.** 10 Nov 2026, Qwen3.6-35B-A3B. Top-320 of 4,096 routed neurons kept (92.2% ablated): loss 1.886 to 1.940 (+2.9%), KL 0.140, and in that column benchmarks range from -20.9 (MBPP+) to +15.9 (HumanEval+). Inputs are neurons whose output is "aligned with the target neuron's gate or up weight matrix" or found by attribution. Detection 95% (against random text, top bands; 89% against same-expert text), simulation 73% (above p99.9), synthesis 94%. Three things go beyond the page: (a) the phrase "ranked input neurons with attribution scores ... and firing examples" is not on the page; it says each subcircuit "with a description of its inputs and readers" is sent to an LLM; (b) the descriptions that earned 95 / 73 / 94 were written by an agent ("Neuron Scientist") that receives 200 labelled documents, runs its own experiments (activations, steering, attribution graphs) and iterates against a development metric, so they validate an experiment loop and not a static dictionary handed to a reader; (c) "low KL did not predict benchmark movement" is the researcher's reading of Table 1 (the numbers bear it out), while the authors present the ablation as benign and say the benchmark drops are corrected by 91M tokens of SFT and 9M of RL. The model is also a mixture-of-experts with 4,096 routed neurons per layer per token; the page says nothing about a dense model with 21,504 neurons per layer.

9. **Controls** — repos in 3, plus https://arxiv.org/abs/2611.09376 — **supported.** kyle-chalmers: published edit 17.3% to 63.3% on 150 answers (fixed 70, broke 1); random direction on the same 64 neurons per layer 26.7%, about 9.4 of 46.0 points. wsp-sql: both edits are bypasses, random-location edits of the same size do nothing, gradient descent on the same entries gives the same zero-reasoning behaviour. XLRA: GRPO, about 23 hours on one RTX 5090, ahead by 8.1 points on average (tie on MMLU-Pro); a narrow edit lost 4.6 points on MMLU-Pro; fixed 1,000-question panel afterwards. Robot paper: 65 embedding rows, 86.3% to 92.0%, 48 repairs and 4 regressions, matched random directions did not help, 12 of 13 later candidates declined.

10. **Gemma 4 weight-level work and Heretic** — https://github.com/TrevorS/gemma-4-abliteration (README and ABLITERATION.md), https://github.com/p-e-w/heretic (config.default.toml, src/heretic/model.py), https://github.com/199-biotechnologies/gemma-4-abliterated — **numbers supported; the -inf claim is partly supported.** TrevorS: 31B 22/686 (3.2%) at KL 0.124; "4-bit directions ≈ bf16 directions — mean cosine similarity 0.935"; bf16 shard-by-shard save. The reserved-token `-inf` logits are attributed to the 12B Unified architecture (`gemma4_unified`) only; neither file says the 31B emits them. `Gemma4ClippableLinear` is described as a wrapper in the vision and audio encoders. Heretic defaults: `n_trials = 100`, `n_startup_trials = 30`, KL and keyword-rate prompts `test[:100]`, direction prompts `train[:400]` each; model.py does implement edits as LoRA adapters reset by zeroing, and reloads the base on CPU to merge a quantised model. 199-biotechnologies: 60 layers, bf16 shards edited and then 4-bit MLX; hidden size 5376 is only implied by an activation shape.

11. **Gemma 4 interpretability artefacts** — https://docs.neuronpedia.org/neurons , https://www.neuronpedia.org/gemma-4-31b , https://huggingface.co/decoderesearch/gemma-4-saes , https://github.com/speediedan/interpretune/issues/709 — **supported.** Neuronpedia docs: 8.6 million MLP neurons in 10 models; Gemma 4 31B IT, 60 layers, 21,504 neurons per layer, inference enabled. The 31B SAE: the Hugging Face card gives layer 30 and 131k latents but not the hook point, the model variant or a date; "residual", base-model page and "June 2026" come from the Neuronpedia page (`30-res-matryoshka-131k`). The interpretune issue says no transcoders or cross-layer transcoders are published for Gemma 4, about 1B tokens per layer for a fresh set, "multi-day run plus a port". A search found no Gemma Scope or transcoder release for Gemma 4; an absence cannot be proven.

12. **Other items leaned on by the design implications** — **supported unless noted.** AtP* (https://arxiv.org/html/2403.00745v1): two forward passes and one backward pass for all nodes, MLP neurons as nodes, Pythia up to 12B, verify by activation patching, subsampling bound. QA-LoRA (ar5iv copy of arXiv 2309.14717): the appendix says the adapter is merged "by merely updating the zero-point matrix", so the design implication stands. CACHE-UK (https://arxiv.org/html/2607.28292v1): rank-one update applied to the LoRA B matrix, resolution barrier illustrative only; the 12 to 22% held-out figures are the baselines, CACHE-UK itself scored 28%, on one 3B model in a single run the authors call "not deployment-ready". OpenWiSH README: every listed feature is there, including "Results from different access levels are never pooled". Abstracts of the five other follow-ups (arXiv 2611.11604, 2611.09559, 2611.04268, 2611.05341, 2611.05798) and Ahmed (2611.08070): none edits a model above 9B or a Gemma 4 model, so the "nothing above 9B" claim survived this attempt to break it.

13. **TransformerLens `start_at_layer`** — the cited docs URL still returns 404 — **could_not_open.** A search snippet shows the three parameters in the `forward` signature; semantics unverified. The nnsight "2 to 5x" figure appears, per a search snippet, on the nnterp best-practices page (not opened), with no benchmark behind it.

### Corrections

- Hu et al. do state a time cost per pass (up to about half an hour for generation, seconds to minutes for teacher-forced); what is missing is money and total run time.
- "All" public harnesses do not share the same kernel and budget: see check 3.
- Miller's 969 questions already assumes probability scoring and pairing; it cannot be used to argue that probability metrics need fewer questions than generation accuracy. If anything it says a fixed probe set of about 100 prompts cannot resolve a 3-point change on a 0-to-1 score.
- FLoRA's speed-up is for multiplicative per-example adapters, not additive low-rank deltas.
- Unsloth's 96.67% and 87.91% are two conversions compared, not a drop caused by re-conversion; 24.77% is not specific to the 31B.
- The `-inf` reserved-token logits are sourced for the Gemma 4 12B Unified architecture only, not for Gemma 4 in general or the 31B.
- The technical report does not describe Q4_0 as having "a scalar scale at each block"; 19.2 GB excludes the KV cache.
- Goodfire's validated descriptions came from an experimenting agent tuned against a development metric, and the page does not list "attribution scores" or "firing examples" as what the describing model is handed.
- CACHE-UK's own held-out success is 28%; 12 to 22% are its baselines.
- The SAE's hook point, model variant and date are on Neuronpedia, not on the Hugging Face card.

### Takeaways and implications that go beyond their sources

- "Teacher-forced metrics are the right inner loop because a 3-point accuracy difference needs about 969 questions": the conclusion may be right, but this source does not support the "because".
- "Low KL did not predict benchmark movement in Goodfire's ablation": the table supports it; the authors do not say it, and the model is an MoE.
- "The best-evidenced representation to hand an LLM reader is a neuron dictionary with wiring ... following Goodfire's validated format": Goodfire validated descriptions written by an agent with live experiments on an MoE model; no source tests a static dictionary as the input to a weight editor.
- "Gemma 4 reserved tokens emit -inf": sourced for the 12B only. Masking non-finite logits costs nothing, but it should be tested on the 31B checkpoint, not assumed.
- "Holding an edit as an overlay on a frozen 4-bit base is established practice": QLoRA and Heretic carry this; CACHE-UK is a single small study and adds little.
- The lattice view "as OpenWiSH does" and the gradient levels that "follow OpenWiSH's access-level rule" are analogies. OpenWiSH renders weights as text as integers times one scale, and its access levels are about text, code, network runs and shown behaviour; gradients are simply forbidden there.
- The lattice-exactness idea (design implication 3) is the researcher's own and is flagged as such. Two things to test first, by this checker's reasoning and not from any source: a two's-complement int4 code of -8 has no negation in range, and with one scale shared by 32 input columns a single down_proj column (the commonest edit site in Hu et al.) cannot be rescaled through scales at all.
- "Raw weights as text stop being workable at around 10^5 weights" rests on one study of 125k-parameter CNNs hitting one model's output limit; it is an order of magnitude, not a measured threshold.
- The 54 KB, 231 MB, 69 MB, 6.9 MB and 462 MB figures are correct arithmetic for a 21,504 x 5,376 matrix; the 60/(60-L) saving is a derivation with no benchmark behind it, as the report says.
