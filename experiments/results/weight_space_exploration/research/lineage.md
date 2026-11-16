# Prior art: what "programming in weight space" has meant on real language models

Research note, 14 November 2026. Web sources only; every source is marked **[read]** (page opened and read through a fetcher) or **[snippet]** (search result only). Numbers are the pages' own. Where I add arithmetic of my own it is marked "my arithmetic".

## 0. The trigger and its first two weeks

**Takehi, Teoh, Zhang, Khattab, "Sparks of Weight-Space Programming" (blog/technical report, 2 Nov 2026) [read].** Weight-space programming (WSP) is defined as treating weights as a program that a general-purpose LLM reads and writes from a natural-language goal; it is offered as a concept, not a method. Eight experiments, all with one editor model (GPT-6 Astra):

- Reading as raw text: MLP policies of 17,253 weights, diffusion MLPs of 23,824 weights, a 7.16M-weight character transformer (with Python analysis but no forward passes). Largest network given as text: 171,664 weights.
- Writing from scratch: MinAtar policies and a 1,480,274-weight Go MLP.
- Repair: a 171,664-weight controller, 3.5% of weights changed, 0 of 256 episodes before, 1,024 of 1,024 after.
- The only real-LLM edit: Qwen3.5-0.8B (24 layers). A looping direction was traced to the MLPs of layers 15 and 19 and subtracted from 64 output columns in each: 131,072 weights, 0.016% of the model. With a CPU Python kernel and 50 forward passes on practice problems: MATH-500 7.9 to 48.1, GSM8K 13.8 to 67.1, IFEval 21.0 to 62.3, token-limit hits 93% to 33.1%. Gemma 3 270M as a negative control moved from 6.8 to at most 7.7.
- Sycophancy detection in Gemma 3 270M from weights alone failed in all four attempts; with an attention-pattern tool it succeeded on three of three pairs.
- The edits are stored as exact diffs against the public base weights with an apply script (logs repository [read]). A benchmark and harness are said to be in progress; I found neither.

The post does not cite ROME, MEMIT, Tracr, abliteration, SPD or task vectors by name; it positions compilers and hand-set weights as the simplest form of WSP.

**What followed within days**

| Paper | Date | What it adds | Status |
|---|---|---|---|
| Hu et al., "Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations" (arXiv 2611.09382) | 6 Nov 2026 | Same editor, Qwen3.5-0.8B **and 9B**, 16 search trajectories on 7 benchmarks; budget of 50 forward requests and up to 300 Python analyses per search. 0.8B held-out gains are large (Math 8.35 to 37.80, IFEval 21.40 to 58.12). At 9B most candidates were discarded; a Math candidate with +29.69 on development data scored -0.35 held-out. 6 of 16 trajectories ended by keeping the original weights. Three independent Math searches gave +29.45, +5.00, +27.30. Edits were single MLP down-projection columns (one column = 1,023 changed values at 0.8B, 4,096 at 9B), zeroed or scaled attention heads, doubled MLP columns, dense rank-one updates. No comparison with steering, abliteration or LoRA. | [read] |
| Zhang, Wang, Lin, "Can LLMs Debug Robots by Editing Their Weights?" (arXiv 2611.09376) | 6 Nov 2026 | 65 token-embedding rows of the pi-0.5 robot policy; 86.3% to 92.0% on 765 held-out paraphrases (48 repairs, 4 regressions); matched random-direction edits did not help; in 13 further sessions the editor declined 12 candidates. Conclusion: lexical repair, not new capability. | [read, abstract] |
| Gross, "Weight-Space Programming of Policies and Their Verification via Model Checking" (arXiv 2611.11604) | 8 Nov 2026 | LLM-written policy nets (16 to 2x32 hidden units) verified exactly with a probabilistic model checker. Commonest failure was a wrong tensor shape (32 of 67 answers from one model). Dataset repair raised accuracy to 98.4% and broke the formal requirement. | [read] |
| Ahmed, "Can an LLM Actually Reason and Program about Neural Network's Weights..." (arXiv 2611.08070) | 5 Nov 2026 | Audit of the reading side with a different LLM on ~125k-parameter CNNs: local arithmetic works; semantic claims follow dataset familiarity; a label permutation makes every interpreted class pair wrong. Editing not tested. | [read] |
| Mehta, Singh et al., "Interpretable weight-space programming builds transparent transformers and brain encoding models" (alphaXiv listing, Nov 2026; exact date not seen) | Nov 2026 | Agent writes 3-layer width-64 transformers with training forbidden; fixed non-editable evaluator, visible set of 200 and hidden set of 2,000, LLM audit of every saved model, and a "rebuild from a description of at most 200 words by a fresh session" test. All 26 Opus runs on algorithmic tasks reach 100% hidden accuracy; NLP tasks are crude. | [read, PDF pp. 1-6 and repository] |
| Maier, "What Is Measured and What Is Anecdote" (Substack) | Nov 2026 | Line-by-line review: the Qwen edit, controller repair and network writing are measured; reading claims are a handful of toy cases; no baseline against decoding changes or standard direction ablation; 25 passes did worse than 10. | [read] |
| "A Skeptical Look at Weight-Space Programming" (brainwagon blog) | 8 Nov 2026 | The Qwen edit is abliteration-like; the base model was run in a looping regime; an independent rerun of the edited model got 51.0% on MATH-500; a community steering vector along the same direction took 8 to 36 of 96 practice problems (random direction: 10). | [read] |
| Willison, local replication of the reading task | 3 Nov 2026 | A 27B open model in 4-bit named 3 of 4 grid games from weights, taking 46 to 65 minutes each. | [read] |

Read together: **the one real-LLM result is a single-direction subtraction on chosen MLP output columns, i.e. a neuron-localised relative of abliteration, found by an LLM with a Python kernel and a forward-pass budget; it removes an obstacle and does not add ability; it largely stops working at 9B.**

## 1. The five lines

### (a) Model editing (ROME, MEMIT, AlphaEdit and successors)

- **Unit of edit.** A rank-one (ROME) or low-rank least-squares (MEMIT, AlphaEdit) update to the down-projection of one or a few mid-layer MLPs, mapping a "key" (subject representation) to a "value" (new fact). AlphaEdit projects the update onto the null space of a covariance of preserved keys.
- **Verification.** Efficacy, generalisation (paraphrase), specificity (neighbour facts), fluency on CounterFact and zsRE; later ripple effects (RippleEdits, ~5K edits), multi-hop (MQuAKE), realistic free-generation QA (QAEdit/WILD), lifelong streams (WikiBigEdit, 500K+ pairs), downstream tasks after N edits.
- **Largest model.** GPT-NeoX 20B in MEMIT (2022) [read, abstract]. AlphaEdit and its 2026 reproduction top out at Llama3-8B [read]. I found no ROME/MEMIT/AlphaEdit evaluation above 20B.
- **What goes wrong (all measured):**
  - Reported success is inflated by teacher-forced evaluation: 96.8% falls to 38.5% under realistic QA, and methods fail with 1,000 sequential edits (Yang et al., "Mirage of Model Editing", ACL 2025) [read, abstract].
  - A single edit can collapse a model; perplexity tracks the collapse (Yang et al., "Butterfly Effect", ACL Findings 2024) [read, abstract].
  - Sequential editing degrades through over-optimisation and norm growth of the edited matrix; early stopping plus a Frobenius constraint reaches 10,000 edits (Gupta et al., arXiv 2502.01636) [read, abstract]. The ICML 2026 "Lifelong Normalization" paper explains the stable editors by running-statistics normalisation of the value gradient [read, abstract].
  - AlphaEdit reproduces at 2,000 edits on its original models, degrades around 5,000 (Llama3-8B CounterFact efficacy 99.25% to 49.02%), and **fails on Phi3-3.8B and Gemma2-2B** (CounterFact efficacy 48.85% and 45.25%); the authors' untested hypothesis for Gemma is the normalisation after the feed-forward block (Suresh and Hariharan, arXiv 2606.26783, July 2026) [read].
  - Edits do not propagate to implied facts; in-context editing beat weight editing on RippleEdits [read, abstract]; multi-hop questions fail (MQuAKE) [read, abstract].
  - Edited knowledge is forgotten faster than pretrained knowledge under later fine-tuning (Wen and Zhang, arXiv 2507.14198) [read, abstract].
  - Edits are detectable and selectively reversible from the weights because they sit in the dominant singular subspace of the edited matrix (Jiang et al., EMNLP 2026 Findings, arXiv 2609.02091) [read, abstract].
  - Quantisation: a finance paper states that 4-bit quantisation severely limits sequential editing and falls back to a rank-1 LoRA side adapter with a stability controller (CACHE-UK, arXiv 2607.28292) [read, abstract]. A search snippet attributes to MobiEdit a ROME success drop from 96% (FP32) to 41% (W8A16); the abstract I read does not contain that number [snippet].
- **Relevance.** This line supplies the best-developed vocabulary for a rank-one write (key, value, covariance, null-space projector) and the best-developed catalogue of how small edits fail. It also says the standard recipe may not transfer to Gemma's block layout unchanged.

### (b) Compiled and hand-written transformers

- **Unit of edit.** The whole network: a program in RASP (Weiss et al., ICML 2021) compiled by Tracr (Lindner et al., NeurIPS 2023), or ALTA (Shaw et al., TMLR 2025; adds loops and Universal Transformers) [all read, abstracts]; or weights set by hand (AdderBoard, 2026: hand-coded adders of 6 to 8 parameters against 36 for the smallest trained one, checked by a fixed script on 10 edge cases and 10,000 random pairs) [read].
- **Verification.** Exact: the program is the specification, so the network is tested against it exhaustively or formally (Gross's model checking).
- **Largest model.** Tiny. Percepta's WebAssembly interpreter in weights (11 Mar 2026) is 7 layers, width 36, 18 two-dimensional heads; the article is ambiguous about how much is compiled and how much learned, and embedding it in a trained LLM is proposed, not shown [read via a mirror; the primary page returned only its title]. WSP's largest written network is 1.48M weights.
- **Splicing into a trained model.** Tracr-Injection (Vergara-Browne and Soto, ACL Findings 2025) puts three RASP algorithms into GPT-2 large, but by gradient distillation into a residual subspace with a KL retention loss, not by writing weights; out-of-distribution results are mixed [read]. I found no demonstrated training-free transplant of a circuit between LLMs; WSP lists it as future work. The largest hand-written replacement of a mechanism in a trained LM is a single rank-one subcomponent (see (c)).
- **Relevance.** Gives the cleanest verification model (program as spec) and proves LLM agents can write small exact circuits, but nothing here has been composed with a trained 1B+ model without gradients. A harness should treat "write a circuit into Gemma" as unproven.

### (c) Parameter-space decomposition

- **Unit of edit.** A rank-one subcomponent u v^T of a weight matrix with a learned gate saying when it is causally needed; subcomponents cluster into components.
- **Verification.** Faithfulness (subcomponents sum to the weights), ablation loss under stochastic and adversarial masks, intruder-detection interpretability, edit-versus-random controls.
- **Largest model.**
  - APD (Jan 2025) and SPD (Jun 2025): toy models [read, abstracts].
  - VPD, "Interpreting Language Model Parameters" (Bushnaq et al., Goodfire; arXiv 2610.20748, stamped 14 Oct 2026, header says April 2026): one 4-layer, 67M-parameter transformer; 38,912 subcomponents, about 10,000 alive, about 205 active per position; CE 2.71 against 2.72 with all subcomponents on, 2.84 under stochastic masks; adversarial KL 0.83 at 20 attack steps and 25.3 at 160; 400,000 training steps. One subcomponent's write vector was replaced by hand to change an emoticon prediction; LoRA trained on 947 examples beat it on both off-target metrics; the authors call the example preliminary and cherry-picked [read].
  - Targeted PD (Vigouroux and Sharkey, arXiv 2607.13047, Jun 2026): decomposes only the mechanisms used on a chosen data subset, at roughly 7% of the FLOPs of the full decomposition; largest model 12 blocks, ~85M non-embedding parameters; ablating or swapping the recovered subcomponents changed "np"/"pd" completions with KL below 0.1 elsewhere [read].
  - ASPD (Cao et al., arXiv 2609.37731, Sep 2026): ties components to activation features; applied to **one matrix** each of Qwen-3-8B and Gemma-2-2B and to all 72 matrices of GPT-2 small; component-removal edits reach 6.7x and 11.1x a random baseline on the 8B model where VPD reaches 0.9x and 0.8x [read].
  - Sparse Weight Decomposition (Yan et al., arXiv 2608.03913, Aug 2026): W ~ AB with both factors sparse, fitted post hoc, with a zero-data variant; applied to **one matrix** of Qwen3.5-27B and to all of GPT-2 small (CE 3.90, 3.44 after fine-tuning) [read].
- **Relevance.** This is the only line whose native object is "a piece of weight with a meaning", which is what an agent wants to read and then write. It does not exist for a whole model above ~85M parameters; at 8B to 27B it exists one matrix at a time.

### (d) Weight arithmetic

- **Unit of edit.** A whole-model delta (task vector, merge), a rank-r delta (LoRA), or a projection P = I - r r^T applied to every matrix that writes the residual stream (abliteration).
- **Verification.** Task accuracy on source and control tasks; for abliteration, refusal rate plus KL divergence on harmless prompts plus a few capability benchmarks.
- **Largest model.** Merge-weight optimisation at the 100B scale is cited by the 2026 merging survey (Song and Zheng, arXiv 2603.09938) [read, first quarter]. Abliteration: 13 models up to 72B in Arditi et al. (2024) [read]; a 320B mixture-of-experts model in block-FP8 in Shi et al. (arXiv 2609.09793, Sep 2026) [read]. A community repository reports norm-preserving biprojected abliteration across Gemma 4 sizes, with only a single KL figure (0.124) and no method detail for the 31B [read].
- **What is known.**
  - Arditi et al. prove that orthogonalising the writers (W' = W - r r^T W on embedding, attention-output and MLP-output matrices and biases) equals ablating the direction at every layer and position; the direction is a difference of means selected under a KL < 0.1 constraint from a layer below 0.8 of depth; capability stays within confidence intervals except on TruthfulQA [read].
  - Tools differ widely in collateral damage: across three models GSM8K changed by -0.28 pp, -0.13 pp and -7.81 pp for three tools, with one -18.81 pp case; maths was the most sensitive benchmark (Young, arXiv 2512.13655) [read].
  - On the 320B model, matching writers by module name reached 0.066 of a 0.776 total effect because fused expert tensors were skipped silently; the authors enumerated writers from tensor structure and compared the output checkpoint tensor by tensor with the base. They also report that projecting and re-quantising reintroduces part of the removed direction [read; the fetcher withheld the procedure's operational detail, which I did not need].
  - The Gemma 4 repository reports that negating a task vector collapsed sharply near a coefficient of 0.4, that the edits go to attention o_proj and MLP down_proj, and that winsorising activations and preserving row norms mattered [read].
  - Contrast-set construction decides whether a direction exists at all (Petrov, arXiv 2603.22061) [read, abstract].
  - Steer2Edit (Sun et al., arXiv 2602.09870, 2026) converts a steering vector into closed-form rank-one edits of individual attention heads and MLP neurons, up to Llama-3-8B, and beats plain activation steering at matched utility [read].
  - LoRA "as a program": hypernetworks emit rank-8 adapters from a task description (Text-to-LoRA, Mistral-7B) or a document (Doc-to-LoRA, Gemma-2-2b) [read]; Diff Interpretation Tuning trains an adapter so a model can describe its own weight diff [read, abstract]. These are learned generators and readers, not reasoning over weights.
- **Relevance.** Abliteration is the only hand-made weight edit proven at and beyond 31B, and the WSP Qwen edit is a member of the same family. Its engineering lessons transfer directly.

### (e) Reading weights directly

- **Unit of reading.** A matrix or a product of matrices: projection of parameters to vocabulary space (Dar et al., 2022) [read, abstract]; singular vectors of OV and MLP matrices projected to tokens (Millidge and Black, 2022) [snippet; the post body did not load]; transcoder features whose connections split into input-invariant (weights) and input-dependent parts (Dunefsky et al., NeurIPS 2024; models to 1.4B) [read, abstract]; cross-layer transcoders and attribution graphs (Ameisen et al., Anthropic, Mar 2025) [read].
- **Verification.** Perturbation: steer or ablate the feature and compare with what the graph or weight predicts.
- **Largest model.** Anthropic's cross-layer transcoder on Claude 3.5 Haiku (up to 30M features, 21.7% normalised reconstruction error) is proprietary. Open: Gemma Scope 2 (22 Dec 2025) has per-layer transcoders for every layer of Gemma 3 up to 27B, cross-layer transcoders only for 270M and 1B, and warns that circuit work on big models is expensive [read]. **I found no SAE or transcoder release for Gemma 4.** Goodfire's "Neurons as an Interpretable Basis" (10 Nov 2026) works in the raw neuron basis of Qwen3.6-35B-A3B: keeping the top 320 of 4,096 routed neurons per token (92.2% ablated) raises loss from 1.886 to 1.940, though reasoning benchmarks drop and loops appear until fine-tuned; upstream neurons are found partly by aligning their output vectors with the target's gate/up weight rows; no weight edits [read, first part].
- **What is known about weights-only reading.**
  - Anthropic's prompt-independent "global weights" suffer from interference: large virtual weights between features that never co-fire; their fix weights by co-activation statistics and is called imperfect by its authors. Attention patterns are frozen, so QK circuits are not explained [read].
  - WeightLens (Golimblevskaia et al., ICLR 2026): weights-only descriptions of transcoder features, each candidate token confirmed by one single-token forward pass, cover 58.8% of GPT-2, 32.7% of Gemma-2-2B and 25.4% of Llama-3.2-1B features [read].
  - Weight Oracles (Kabra et al., Oct 2026): a fine-tuned LLM that simulates small transformers from weights and detects unseen backdoors at AUROC 0.93 / 0.81; scaling is the stated open problem [read, abstract].
  - LLMs reading raw weights as text: shown up to ~170k weights; fails a label-permutation control (Ahmed).
- **Relevance.** Weights-only readings are cheap hypotheses that hold for a minority of units and must be confirmed with a forward pass; the neuron basis is now defended at 35B scale, which matters because no learned dictionary exists for Gemma 4.

## 2. Summary table

| Line | Unit of edit | How an edit is verified | Largest model shown | Use to an agent editing a 31B |
|---|---|---|---|---|
| (a) Model editing | rank-one / low-rank update to MLP down-projection, key to value | efficacy, paraphrase, neighbours, ripple, multi-hop, sequential, downstream | 20B (MEMIT, 2022); recent work at most 8B | the algebra of a targeted write, and the failure catalogue |
| (b) Compiled / hand-written | whole small network from a program | exact, against the program; model checking | ~1.5M weights written; GPT-2 large by distillation | a model of exact verification; not yet a way into a trained LLM |
| (c) Parameter decomposition | rank-one subcomponent with a causal gate | masked-ablation loss, adversarial masks, edit vs random | 67M to 85M whole model; one matrix at 8B and 27B | the right object, available only per matrix at this scale |
| (d) Weight arithmetic | direction projected out of all writers; LoRA; task vector | refusal/behaviour rate, KL on neutral text, capability benchmarks | 72B dense, 320B MoE (abliteration); 100B (merging) | the proven large-scale hand edit and its engineering lessons |
| (e) Reading weights | vocabulary projection, SVD, virtual weights, transcoder features | perturbation agreeing with the reading | Gemma 3 27B per-layer transcoders (open); 35B neuron basis | cheap hypotheses that need a confirming forward pass |
| WSP (Nov 2026) | column delta, head scale, rank-one, embedding rows chosen by an LLM | dev vs held-out benchmarks, forward-pass budget, random-direction control | 0.8B works; 9B mostly null | the workflow itself; evidence that "no edit" is a common right answer |

## 3. What goes wrong, across lines

1. **The evaluation flatters the edit**: teacher forcing (Mirage), development-set selection (Hu et al.: +29.69 dev, -0.35 held-out), a pathological baseline (WSP's 7.9% with 93% of answers at the token limit), keyword refusal metrics (abliteration studies).
2. **Local success, global damage**: single-edit collapse, norm growth under sequences, maths degrading under refusal ablation, off-target KL.
3. **No propagation**: an edited fact does not update its consequences; a removed obstacle does not add an ability (Gemma 3 270M control; 9B results; robot paper).
4. **Architecture dependence**: AlphaEdit failing on Gemma2 and Phi3; name-matched writers missing 74% of the effect on an MoE; attention-only readings missing QK.
5. **Storage format**: quantisation rounds small updates away or reintroduces a removed direction; float checks pass and packed-weight checks fail (hobby lab on a 27B GGUF [read]).
6. **Impermanence and traceability**: edits are forgotten under later fine-tuning and can be located and reversed from the spectrum of the edited matrix.
7. **Weights-only reading over-claims**: interference in virtual weights; dataset priors in LLM reading; 25 to 59% validity for weights-only feature labels.
8. **High variance across searches**: three identical WSP searches gave +29.45, +5.00, +27.30.

## 4. The 2026 state of the art in one paragraph

By November 2026 there are three partly disconnected capabilities. First, hand-made directional edits (abliteration and its norm-preserving, multi-direction variants) are routine from 2B to 320B and are the only weight edits proven at 31B scale; they remove a behaviour and cost some capability, most visibly in maths. Second, key-value editing (ROME to AlphaEdit and the 2026 normalisation-based lifelong editors) can insert thousands of facts into models up to about 8B to 20B, but realistic evaluation cuts reported success by more than half, edits do not propagate, and the recipe is architecture-sensitive, with Gemma 2 a documented failure. Third, reading tools have moved toward the weights: parameter decomposition now covers a whole 67M to 85M model and single matrices of 8B and 27B models, transcoders cover every layer of Gemma 3 27B, and the raw neuron basis has been defended on a 35B model, while no learned dictionary exists for Gemma 4. Weight-space programming joins these by letting an LLM choose the edit: the measured results are an abliteration-like fix on a 0.8B model, mostly null results at 9B, exact small networks written from scratch, and reading claims that have not survived the first controlled audit. Nobody has yet shown an LLM adding a capability to a trained multi-billion-parameter model by reasoning over its weights.

## 5. Ideas most worth borrowing

1. **Typed, small, exactly-diffed edits with a structural check.** The edits LLM editors actually chose (Hu et al.) are few: subtract a direction from named down-projection columns, scale or negate a column, zero or scale a head's output block, a rank-one u v^T, overwrite embedding rows. Store each as an exact diff against the base (Takehi logs), enumerate residual writers from tensor shapes rather than names, and compare the result tensor by tensor with the base (Shi et al.).
2. **A verification ladder taken from the editing literature, with "keep the base" as a first-class outcome.** Free-generation scoring, not teacher forcing; development versus held-out sets; a budget of forward passes; a matched random-direction control for every edit; perplexity or KL on neutral text and matrix-norm growth after every edit as cheap collapse alarms; repeat the search, since one run is not evidence.
3. **Activation first, weights second, with the equivalence measured.** Arditi's proof and Steer2Edit's closed form let a behaviour be tested as a reversible activation intervention and then compiled into a weight edit. On Gemma's pre-and-post-norm blocks that equivalence should be checked per edit, not assumed.
4. **Decompose on demand, one matrix or one data slice.** Targeted PD (7% of the FLOPs), ASPD and zero-data SWD show that a rank-one, gated basis for a single matrix of an 8B to 27B model is feasible; this is the natural "disassembler" call for an agent with a hypothesis about one layer.
5. **The rebuild test.** From iWSP: an edit or reading counts as understood only if a fresh session, given a description of at most 200 words, reproduces it. Cheap to run and resistant to post-hoc storytelling.
6. **A covariance-aware write primitive.** AlphaEdit's null-space projection is one line once a key covariance exists; offering it per layer lets every rank-one write be constrained to leave typical inputs alone, with the stated limits (thousands of edits, not Gemma-verified).

## 6. Implications for a harness on a 31B Gemma in 4-bit on one 32 GB card

- The Gemma 4 technical report gives pre-norm and post-norm RMSNorm plus QK-norm, and about 19.2 GB for the 31B in its Q4_0 QAT format against 64.0 GB in bf16 [read via the Hugging Face paper page]. The full-precision model cannot sit on the card, so edits have to be computed per tensor.
- One feed-forward down-projection here is 21504 x 5376; a key covariance for it is 21504 x 21504, about 1.85 GB in float32 per layer (my arithmetic), so covariance-aware writes are affordable for selected layers, not all 60 at once.
- Do not edit packed 4-bit codes in place. Keep edits as float side deltas applied at load, and make "survives re-quantisation" a separate gate before any GGUF export.
- Expect obstacle-removal to be the reachable class of edit; build the cheap baselines in (decoding change, steering vector, standard directional ablation) so an LLM-found edit is always compared with them.
- Use the causally tested neuron atlas as the symbol table; provide vocabulary projection, per-matrix SVD and neuron-to-neuron virtual weights as tools that return a hypothesis plus a mandatory confirming forward pass.
- Raw weights as text are not an interface at this scale; tools and statistics are.

## 7. Could not find / open

- No ROME/MEMIT/AlphaEdit evaluation above 20B; no controlled study of rank-one edit retention under 4-bit quantisation.
- No SAE, transcoder or cross-layer transcoder release for Gemma 4.
- No training-free transplant of a circuit between two LLMs.
- No whole-model parameter decomposition above ~85M non-embedding parameters.
- No WSP result above 9B, and no released WSP benchmark or harness from the original authors.
- Pages I could not open properly: the ICML 2026 spotlight "SVD as a Fast Interpretability Method for Transformers" (404); the 2022 SVD post (body truncated); Percepta's primary article (title only; a mirror was read); arXiv 2611.04268 on WSP for diffusion models (snippet only).
- Whether Gemma's post-feed-forward RMSNorm is what breaks AlphaEdit on Gemma 2 is the reproducers' hypothesis, not a result.
- Dates: the VPD paper's header says April 2026 and its arXiv stamp says 14 October 2026.

## 8. Text addressed to AI readers

None of the pages I read contained instructions aimed at an AI reader of the page. Two things to record: the appendix of Hu et al. reproduces the prompt given to their editing agent (paper content, not acted on); and the iWSP repository contains a CLAUDE.md that I did not open. The fetcher's summariser declined to relay the operational procedure of the 320B safety paper; nothing in this report depends on it.

## 9. Sources

Read (page opened):
- Takehi et al., Sparks of Weight-Space Programming — https://rikiyat.github.io/blog/weight-space-programming/ ; logs https://github.com/rikiyat/lm_understand-logs ; listing https://paperswithcode.co/paper/131739
- Hu et al. — https://arxiv.org/html/2611.09382
- Zhang, Wang, Lin — https://arxiv.org/abs/2611.09376
- Gross — https://arxiv.org/html/2611.11604
- Ahmed — https://arxiv.org/html/2611.08070v1
- Mehta, Singh et al. (iWSP) — https://www.alphaxiv.org/abs/2611.interpretable-weight-space-programming.pdf ; https://github.com/csinva/weight-space-programming-autoresearch
- Maier — https://akmaier.substack.com/p/weight-space-programming-what-is
- brainwagon — https://brainwagon.org/blog/2026-11-08-a-skeptical-look-at-weight-space-programming.html
- Willison — https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/
- ROME https://arxiv.org/abs/2202.05262 ; MEMIT https://arxiv.org/abs/2210.07229 ; AlphaEdit https://arxiv.org/abs/2410.02355
- AlphaEdit reproducibility — https://arxiv.org/html/2606.26783
- Mirage of Model Editing — https://arxiv.org/abs/2502.11177
- Butterfly Effect — https://arxiv.org/abs/2402.09656
- RippleEdits https://arxiv.org/abs/2307.12976 ; MQuAKE https://arxiv.org/abs/2305.14795 ; WikiBigEdit https://arxiv.org/abs/2503.05683
- Lifelong editing regularisation — https://arxiv.org/abs/2502.01636 ; Lifelong Normalization (ICML 2026) — https://icml.cc/virtual/2026/poster/64339
- Retention after fine-tuning — https://arxiv.org/abs/2507.14198
- Selective edit reversal — https://arxiv.org/abs/2609.02091
- CACHE-UK — https://arxiv.org/abs/2607.28292 ; MobiEdit — https://arxiv.org/abs/2506.13772
- Steer2Edit — https://arxiv.org/html/2602.09870v2
- RASP https://arxiv.org/abs/2106.06981 ; Tracr https://arxiv.org/abs/2301.05062 ; ALTA https://arxiv.org/abs/2410.18077 ; Tracr-Injection https://arxiv.org/html/2505.10719
- AdderBoard — https://github.com/anadim/AdderBoard
- Percepta article (mirror) — https://code.nym.sh/kennethnym/percepta-computer-as-llm/raw/branch/main/article.md
- APD https://arxiv.org/abs/2501.14926 ; SPD https://arxiv.org/abs/2506.20790 ; VPD https://arxiv.org/html/2610.20748 ; tPD https://arxiv.org/html/2607.13047 ; ASPD https://arxiv.org/html/2609.37731v1 ; SWD https://arxiv.org/html/2608.03913
- Weight-sparse transformers https://arxiv.org/abs/2511.13653 ; individual parameters https://arxiv.org/abs/2607.02964
- Task arithmetic https://arxiv.org/abs/2212.04089 ; merging survey https://arxiv.org/html/2603.09938v2
- Arditi et al. https://arxiv.org/html/2406.11717v3 ; abliteration tools compared https://arxiv.org/html/2512.13655v1 ; Gemma 4 abliteration notes https://github.com/TrevorS/gemma-4-abliteration/blob/master/ABLITERATION.md ; 320B MoE https://arxiv.org/html/2609.09793 ; topic-matched baselines https://arxiv.org/abs/2603.22061
- Doc-to-LoRA / Text-to-LoRA https://pub.sakana.ai/doc-to-lora/ ; Diff Interpretation Tuning https://arxiv.org/abs/2510.05092 ; Weight Oracles https://arxiv.org/abs/2610.07334v1
- Dar et al. https://arxiv.org/abs/2209.02535 ; transcoders https://arxiv.org/abs/2406.11944 ; circuit tracing https://transformer-circuits.pub/2025/attribution-graphs/methods.html ; Gemma Scope 2 https://www.lesswrong.com/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2 ; WeightLens https://arxiv.org/html/2510.14936v2 ; Goodfire neurons https://www.goodfire.com/research/neurons-as-an-interpretable-basis
- Gemma 4 technical report (paper page) https://huggingface.co/papers/2607.02770.md
- Saved-weight editing lab on a 27B GGUF https://github.com/Teylersf/model-weight-editing-lab

Snippet only (not opened, or opened without usable content):
- Millidge and Black, SVD of transformer weight matrices — https://www.lesswrong.com/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight
- "SVD as a Fast Interpretability Method for Transformers", ICML 2026 spotlight — https://icml.cc/virtual/2026/spotlight/84904 (404)
- WSP for image-restoration diffusion models — https://arxiv.org/abs/2611.04268
- CLT-Forge — https://arxiv.org/abs/2603.21014
- Percepta primary — https://www.percepta.ai/blog/can-llms-be-computers (title only)
- EAMET (ICLR 2026), HiEdit (arXiv 2604.11214), MOSE (arXiv 2601.07873), Gupta et al. "Model Editing at Scale leads to Gradual and Catastrophic Forgetting" (ACL Findings 2024)

## Independent citation check

Done 14 November 2026 by a second reader who did not write the sections above. Every source listed here was opened again from its cited URL (through a fetcher; long arXiv pages were read past the first 100,000 characters where the claim sat further in; pages 1 to 6 of the iWSP PDF were read directly). Verdicts: **supported**, **partly**, **not supported**. Nothing above this heading was changed.

### Result in brief

No source was invented, and almost every number was copied correctly. Four statements are wrong or go beyond their source, and three of them feed design implications:

1. **"No SAE release exists for Gemma 4" is false.** Community SAEs for Gemma 4 31B exist (details in check 10). No transcoder, cross-layer transcoder or official Google release was found, so that half stands.
2. **"Name matching missed 74% of the effect" misreads Shi et al.** In that paper 74% is the share of the effect that appears only when both writer groups are edited together. By the paper's own figures name matching reached 0.066 of 0.776, so it missed about 91%.
3. **"Zero-data SWD on Qwen3.5-27B" is wrong.** The 27B matrix was fitted with calibration tokens (2,048 or 1.05M). The zero-data variant was evaluated only on one GPT-2 Small matrix.
4. **"The approach mostly stops working at 9B" overstates Hu et al.** The 9B base already scored 97.70% on held-out Math, so the -0.35 is a ceiling result. The paper's own explanation is headroom, and three of the seven 9B searches kept an edit with a positive held-out gain.

### Checks

| # | Claim checked | Source opened | Verdict | What the source says |
|---|---|---|---|---|
| 1 | Qwen3.5-0.8B edit: one direction, 64 MLP output columns in each of layers 15 and 19, 131,072 weights (0.016%), MATH-500 7.9 to 48.1, 50 forward passes, Python kernel without GPU; Gemma 3 270M control 6.8 to at most 7.7 | https://rikiyat.github.io/blog/weight-space-programming/ | supported | All numbers match (GSM8K 13.8 to 67.1, IFEval 21.0 to 62.3, token-limit hits 93.0% to 33.1%). Gemma control: 7.0, 7.6, 7.7, 7.0 at 5, 10, 25, 50 passes. Two things the takeaway leaves out: the headline is the best of three budgets (at 10 and 25 passes token-limit hits stayed at 87.2% and 86.4%; Maier's review quotes 14.1% on MATH-500 at 25 passes), and the page reports no baseline beyond the sentence that turning thinking off "raises the base score", with no number. A "Response to Comments" section dated 12 Nov 2026 adds no baseline. |
| 2 | Hu et al.: at 9B most candidates discarded; Math +29.69 on development data, -0.35 held-out; 6 of 16 searches kept the base; three 0.8B Math searches gave +29.45, +5.00, +27.30 | https://arxiv.org/html/2611.09382 | partly | Numbers are right. Reading is not. 9B held-out base to candidate: Math 97.70 to 97.35 (discarded), GPQA 80.05 to 81.06 (discarded), MedXpertQA 34.57 to 33.46 (discarded), MMLU-Pro 82.89 to 82.15 (discarded), IFEval 89.30 to 91.14 (kept), SciBench 72.46 to 75.49 (kept), SciCode 1.54 to 3.08 (kept). So 4 of 7 were discarded and 3 of 7 kept small gains. The paper attributes the 9B pattern to "initial model capability and available improvement headroom"; it does not say the method stops working. The +29.69 is a best-observed development figure (54.69 to 84.38), selected adaptively and reported without correction for multiple comparisons. The count of kept-base searches is inconsistent inside the paper (abstract and Table 2 say six; Section 3.4 and the appendix list differ). The three 0.8B Math searches changed 1,023, 1,037,572 and 1,981,083 values respectively. |
| 3 | Zhang et al.: 65 embedding rows, 86.3% to 92.0% on 765 held-out paraphrases, random-direction control, 12 of 13 further candidates declined | https://arxiv.org/abs/2611.09376 | supported | Abstract states each figure (48 repairs, four regressions) and concludes "precise lexical repair, not general acquisition of missing robot capabilities". Abstract only. |
| 4 | Mirage of Model Editing: 96.8% falls to 38.5% under realistic QA; teacher forcing; failure at 1,000 sequential edits; ACL 2025 | https://arxiv.org/abs/2502.11177 | supported | Abstract and comments field confirm all four. |
| 5 | AlphaEdit reproduction: holds at 2,000 edits, Llama3-8B efficacy 99.25% to 49.02% near 5,000, Gemma2-2B 45.25%, Gemma explanation is an untested hypothesis | https://arxiv.org/html/2606.26783 | supported | Figures match. Nuances: only 2,000, 5,000 and 10,000 edits were measured, and efficacy at 10,000 is 69.35%, so "near 5,000" is not an onset and the curve is not monotone. Gemma2-2B rose from 12.8% before editing to 45.25%; efficacy is a binary comparison with chance at 50%. The main text says the failure was "traced" to the post-feed-forward norm; the appendix says it "remains unclear" whether that is the sole cause and no ablation is reported, so "untested hypothesis" is the fair reading. |
| 6 | Arditi et al.: 13 models to 72B; orthogonalising residual writers equals directional ablation | https://arxiv.org/html/2406.11717v3 | supported | 13 chat models, Qwen 1.8B to Qwen 72B. Appendix E shows the identity for one matrix whose write is linear (x_post = x_pre + W_out t), assuming the direction has already been removed from everything written earlier. Post-block normalisation is not discussed, which backs the report's warning about Gemma. |
| 7 | Young: GSM8K change -0.28, -0.13, -7.81 points for three tools, one -18.81 case | https://arxiv.org/html/2512.13655v1 | supported | ErisForge -0.28, DECCP -0.13, Heretic -7.81, averaged over three models; worst case Yi-1.5-9B under Heretic. The study covers 16 models and four tools overall; GSM8K averages exist for three tools on three models. |
| 8 | Shi et al., 320B MoE: name-matched writers reach 0.066 of 0.776; fused tensors skipped without error; writers enumerated from tensor structure; tensor-by-tensor comparison with base; re-quantising reintroduces part of the direction | https://arxiv.org/html/2609.09793 | supported, except "74%" | 0.066 and 0.776 are absolute drops in a judged refusal rate over 256 held-out prompts (0.949 to 0.883 against 0.949 to 0.173). All four procedural statements are in the text. The figure 74% in the paper is "74% of the effect exists only under the joint intervention", not what name matching missed. First 100,000 of 118,034 characters read. |
| 9 | VPD: 67M-parameter 4-layer model; one subcomponent's write vector replaced by hand; LoRA on 947 examples beat it on both off-target metrics; authors call it preliminary | https://arxiv.org/html/2610.20748 | supported | 67M total (about 28M non-embedding), 38,912 subcomponents, 9,972 alive, about 205 active per position; CE 2.71 / 2.72 / 2.84; adversarial KL 0.828 and 25.256. Section 6: subcomponent 2.mlp.down:2359 rewritten; LoRA with n=947 beats the hand edit on both KL measures, LoRA with n=10 on one of two; "very preliminary", "cherry picked". Values only in a figure. The 400,000-step figure and the "67M is still small" remark were not in the 200,000 characters read. |
| 10 | No SAE, transcoder or cross-layer transcoder release exists for Gemma 4 | generic web search, then https://huggingface.co/decoderesearch/gemma-4-saes ; https://www.neuronpedia.org/gemma-4-saes ; https://huggingface.co/datasets/Adam1010/gemma-4-31b-sae-features ; https://huggingface.co/ddasd/gemma4-31b-fp8block-finaltoken-sae-200m | not supported for SAEs | Three community releases exist. (a) decoderesearch/gemma-4-saes (David Chanin, 2026, also browsable on Neuronpedia): Matryoshka BatchTopK SAEs for Gemma-4-E2B (layers 6, 17, 28), E4B (7, 21, 35) and **Gemma-4-31B layer 30, 131k latents**. (b) Adam1010/gemma-4-31b-sae-features: TopK-64 SAEs with 43,008 features per layer on all 60 layers of Gemma-4-31B-IT; a labelled subset of 3,000 features with a stated cross-validation confirmation of 35.8%; the card says full weights are separate, though the repository lists 111 GB. (c) ddasd: a 65,536-feature SAE on final-token embeddings of an FP8 31B-it, not an internal layer. I found no transcoder, no cross-layer transcoder and no official Gemma Scope release for Gemma 4. Quality of (a) to (c) is unverified. |
| 11 | Gemma Scope 2: Gemma 3 to 27B, cross-layer transcoders only for 270M and 1B | https://www.lesswrong.com/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2 | supported | 22 Dec 2025; SAEs and per-layer transcoders on every layer of all ten models; "CLTs on 270m & 1b"; Gemma 4 not mentioned. |
| 12 | Goodfire: 92.2% of routed neurons ablated on a 35B model, loss 1.886 to 1.940 | https://www.goodfire.com/research/neurons-as-an-interpretable-basis | supported | 10 Nov 2026. The model is a mixture of experts (256 experts of 512 neurons per layer, 8 routed per token); the page does not state parameter counts. Reasoning looped and some benchmarks dropped until the sparsified model was fine-tuned. Transfer to a dense 31B is the researcher's inference. |
| 13 | WeightLens validates 58.8% / 32.7% / 25.4% of features, each with a confirming forward pass | https://arxiv.org/html/2510.14936v2 | supported | Quoted sentence matches. The arXiv page names no venue; "ICLR 2026" was not confirmed. |
| 14 | Ahmed: LLM reading of raw weights failed a label-permutation control | https://arxiv.org/html/2611.08070v1 | supported, narrower than stated | "all nine class pairs it interpreted were wrong" on one permuted CIFAR-10 CNN. One reader model with no tools, about 125k-parameter CNNs, mostly single runs. The paper says "These findings do not refute Takehi et al." |
| 15 | Parameter decomposition at scale: tPD about 7% of full cost and about 85M parameters; ASPD one matrix of Qwen-3-8B; zero-data SWD one matrix of Qwen3.5-27B | https://arxiv.org/html/2607.13047 ; https://arxiv.org/html/2609.37731v1 ; https://arxiv.org/html/2608.03913 | partly | tPD: "≈7% of the FLOPs" is for extracting a CSS-only submodel of a 4-block transformer, one case; largest model 12 blocks, 85M non-embedding; off-target KL is "<< 10^-2" for the ablation and "<< 10^-1" for the swap, not "below 0.1" for both. ASPD: one matrix each of Qwen-3-8B and Gemma-2-2B, 72 matrices of GPT-2 small; 6.7x and 11.1x against 0.9x and 0.8x confirmed. SWD: one matrix of Qwen3.5-27B (layer 31 mlp.down_proj), fitted with calibration tokens; zero-data only on GPT-2 Small layer 8. |
| 16 | Edits sit in the dominant singular subspace and can be selectively reversed (Jiang et al., EMNLP 2026 Findings); edited knowledge forgotten faster under fine-tuning | https://arxiv.org/abs/2609.02091 ; https://arxiv.org/abs/2507.14198 | partly; supported | Jiang: the abstract frames sparse encoding in the dominant subspace as a hypothesis that its experiments support; venue confirmed. Abstract names no models or editors. Wen and Zhang: quoted claim confirmed. |
| 17 | Gemma 4 report: pre-norm and post-norm RMSNorm plus QK-norm; 31B is 64.0 GB in bf16, 19.2 GB in Q4_0 | https://huggingface.co/papers/2607.02770.md | supported | Both in the text and Table 3. The page does not give the 31B's width, feed-forward width or layer count; 5376, 21504 and 60 match this project's own code instead, and 21504 x 21504 x 4 bytes is 1.85 GB as stated. |
| 18 | iWSP rebuild test: a fresh session rebuilds from a description of at most 200 words | https://www.alphaxiv.org/abs/2611.interpretable-weight-space-programming.pdf (pp. 1-6) | supported | Mehta, Singh et al. (Microsoft Research, JHU, Columbia, NYU). "at most 200 words", visible set 200, hidden set 2,000, three layers of width 64, all 26 Opus algorithmic runs at 100%, rebuilt NLP models lose about 1.4 points. |
| 19 | WSP logs store the final edit as an exact diff with an apply script | https://github.com/rikiyat/lm_understand-logs | supported | README: "Experiment 8's as exact diffs against the public base weights, with a script that applies them." |
| 20 | Quantisation: 4-bit limits editing; small updates round away; float checks pass and packed checks fail "often" | https://arxiv.org/abs/2607.28292 ; https://github.com/Teylersf/model-weight-editing-lab | partly | CACHE-UK's abstract says 4-bit quantisation "severely limits the viability of sequential memory editing" (28% on a 4-bit 3B model); it says nothing about updates rounding away. The hobby repository records one explicit case (its V6 row) of float checks passing and the packed Q4 form failing, not a pattern. Rounding-away is in fact supported by Hu et al., who note float32 changes rounding back to the original bf16 value. |
| 21 | MEMIT reaches GPT-NeoX 20B; Steer2Edit is closed-form and training-free up to Llama-3-8B; Tracr-Injection uses distillation into GPT-2 large | https://arxiv.org/abs/2210.07229 ; https://arxiv.org/html/2602.09870v2 ; https://arxiv.org/html/2505.10719 | supported | All three as stated. |
| 22 | Commentary: Maier (no baseline; 25 passes worse than 10), brainwagon (abliteration-like; 51.0% rerun; steering vector 8 to 36 of 96, random 10), Willison (3 of 4 games, 46 to 65 minutes) | https://akmaier.substack.com/p/weight-space-programming-what-is ; https://brainwagon.org/blog/2026-11-08-a-skeptical-look-at-weight-space-programming.html ; https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/ | supported | Pages exist and say this. The brainwagon post calls the 51.0% rerun independent without naming who ran it, and its link points at the original authors' namespace, so independence is unconfirmed. |

### Corrections

- Takeaway 12 and design implication 9: replace "no SAE, transcoder or cross-layer transcoder release exists for Gemma 4" and "no learned dictionary exists for Gemma 4" with: community SAEs exist for Gemma 4 31B (one residual layer at 131k latents from decoderesearch; an all-layer TopK set with a labelled 3,000-feature subset from Adam1010), of unverified quality; no transcoders, cross-layer transcoders or official release were found. The neuron atlas can still be the default symbol table, but the reason is "causally tested here and covers every layer", not "nothing else exists".
- Design implication 3: replace "name matching silently missed 74% of the effect" with "name matching reached 0.066 of a 0.776 drop in refusal rate, about 9% of the effect". The 74% figure belongs to a different finding (the share of the effect that needs both writer groups edited together).
- Takeaway 8, design implication 13 and section 5 item 4: the Qwen3.5-27B decomposition used calibration tokens (as few as 2,048), not the zero-data variant. Zero-data SWD is shown on one GPT-2 Small matrix only.
- Takeaway 2, design implication 8, and the sentence "it largely stops working at 9B" in section 0: say instead that at 9B held-out gains were small (+1.5 to +3.0 points on the three kept edits, none on the other four) and that the paper attributes this to a strong base with little headroom; the 9B Math base was 97.70%. The -0.35 is a ceiling result plus development-set selection, not evidence that a large development gain evaporated on a model with room to improve.
- Takeaway 1: "the only weight-space-programming result on a real LLM" holds only inside the original post. Hu et al. edited the same 0.8B model and a 9B one. Add that the 48.1% is the best of three forward-pass budgets and a single run.
- Section 1(c), targeted PD: "KL below 0.1 elsewhere" should read "much less than 0.01 for the ablation, much less than 0.1 on most tokens for the swap".
- Design implication 2: cite Hu et al. for small updates rounding away; CACHE-UK does not say it. "A hobby lab's Q4 failures" is one recorded failure.
- Section 4 ("reading claims that have not survived the first controlled audit"): the audit used a different reader model with no tools on small CNNs, and its author says it does not refute the original claims.

### Claims that rest on not finding something

These could not be verified by opening a source and should be worded as "not found": no key-value editing result above 20B (one generic search found none); abliteration as the only hand-made weight edit at or above 31B; no training-free circuit transplant between LLMs; no capability-adding edit on a multi-billion-parameter model. The Gemma 4 SAE case above shows this kind of claim in the report can be wrong.

### Text addressed to AI readers

None of the pages opened for this check contained instructions aimed at an AI reader. Noted and not acted on: the editor-agent prompt reproduced in the appendix of Hu et al.; base64 blocks in Ahmed's page that the fetcher reports as prompts for the model under test; a "Co-authored by Claude Code" line in the brainwagon post. The fetcher saved the iWSP PDF to the session's tool-results folder as a side effect of opening the URL; it was read as a document and nothing was run.
