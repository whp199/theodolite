# What the study found

*The findings of the study Theodolite was first built for (is Gemma 4 “female-coded”?), with the caution on the lens they rest on, and every experiment’s write-up. The study itself is [experiments/REPORT.md](../experiments/REPORT.md).*

<sub>Moved out of the [README](../README.md) on 16 November 2026, word for word but for its links and headings, when the README was cut down. Layers count 1–60, as the pages count them; the census board and the commands count 0–59.</sub>

> [!CAUTION]
> **Where the findings below rest on J++ or the Jacobian lens, they rest on lenses fitted on the base model.** Most of
> the study’s lens readings, and every reading the instrument calls “the lens”, go through J++: a lens fitted on the
> *base* model, `google/gemma-4-31B` in bf16, from 64 WikiText prompts of up to 128 tokens, and applied here to the
> instruct model in 4 bits. On the instruct model it is good for comparing and poor on its own: from about layer 21 it
> puts conditions in the order the model’s behaviour does (r ≈ 0.9 or better), but through layer 50 its top word matches
> the model’s next word on prose at most about 5% of the time. Results reached without a lens (what the model says,
> activation geometry, weights, patching, and steering measured on the output) are untouched by this caution. Most stand
> as they are, the behaviour counts and the AI-versus-human offset in the raw stream among them; a few carry precision
> fixes of their own in the report’s [errata](../experiments/REPORT.md#errata-and-cautions-2026-10-17): where the
> self-gender the model predicts from is held (erratum E22), what share of a covector’s energy the ~180 quiet dimensions
> take (E7, E27), how far binding fades in the raw stream at layers 41–59 (E21), and what single neurons do at layers
> 21–31, which the study’s 60-fold push overdrove (E43, E46). Each lens finding below names its readout and its layers;
> the report’s caution on the lens, in [`experiments/REPORT.md`](../experiments/REPORT.md), gives the detail.

- **Gemma-4-31B is not female-coded by default.** With no system prompt it calls itself *it* (0/320 *she* across
  single- and multi-turn samples). Read at its own *I*/*me*/*my* tokens, its gender readout sits with a nameless AI’s:
  within about 1–3% of the woman–man persona span through the base-fitted J++ and Jacobian lenses at layers 21–50, and
  within about 5% through a lens refitted on the instruct model (R1, layers 31–59; 8–9% against neutral-named AIs,
  and 10%, toward *male*, on one full-matrix reading at 51–59). Where it sits on that span depends on the readout; only
  the comparison holds.
- **The "female" readings trace to other causes:**
  - the bare name **"Gemma" written as an antecedent**; "Gemma 4" stays *it*;
  - a forced-choice format that defaults to *woman*, plus an "assistant" stereotype;
  - late-layer (51–59) pronoun readings of the J++ lens, where 6 or 7 of 9 AI personas read at or above the woman
    persona. R1 found this mostly a quirk of J++’s fit: under the base-fitted Jacobian lens and under lenses refitted on
    the instruct model, none or one of the nine does. The default still leans toward the woman persona under each of them
    (0.54–0.80, man 0 to woman 1), just as the nameless AI does, which R1 reads as the offset in the next line rather than
    as the model’s own gender;
  - a raw persona axis contaminated by an AI/service-role offset.
- **At pronoun tokens the gender the stream carries is a role default, not the resolved referent**: at *I*, *me* and *my*
  in layers 21–40, and at *you* from about layer 31 (at 21–30 it is only partly there). Quoted speech dissociates the
  two, while the model’s actual predictions resolve the referent. The base-fitted J++ and Jacobian lenses, the plain
  logit lens, a probe trained on prose and a direction read without any lens agree in sign; it has not been re-read
  through a lens refitted on the instruct model.
- **A "quiet carry channel" in Gemma 4.** About 180 early-loud residual dimensions are read, through RMSNorm, with up to
  1700× the gain of the others. Every Jacobian or gradient covector, the instruct model’s own included, puts 99–100% of
  its energy in the quietest tenth of dimensions at layers 36–59, which contain these (the ~180 alone hold 56–98% at
  layers 31 and 41); that is why lens-derived steering vectors break the model.
- **Neuron-pole output labels predicted the neurons’ real effects only at the sampled layers 41–58** (the report's
  L40–57, counted from 0), read through the base-fitted J++ lens or the plain logit lens, which did as well except at
  41–46, where J++ was a little ahead. At the sampled layers 11–31 no single neuron had a stable effect to label; layers
  32–40 were not sampled. Input-side labels are near-useless. Labels read through a lens refitted on the instruct model
  are untested. The pages now hold such labels back from their main reading below layer 41, and wherever the lens and
  the naked eye share none of their first ten word-pieces (the study's agreement gate, REPORT §3.3: the overlap, used
  as a graded score, had AUC 0.84 within layers 41–58 for base-fitted J++ labels on random neurons; the cut at none of
  ten is the instrument's, since the study set none, and as a yes/no sign it scores AUC 0.64 on the same sample); the
  apparatus keeps them, and the server marks each label it sends.
- **The shipped J++ lens, and the Jacobian lens it was checked against, were fitted on the base model.** Most of the
  study’s lens readings went through J++, and the instrument still reads through it. A lens refitted on the instruct
  model predicts the next word better at the five late layers where its full matrix is usable, 41, 46, 51, 56 and 59
  (top-1 on the model’s own reply at layer 51: 25% → 45%; at layer 59: 69% → 85%), but not below layer 41, and on prose
  it stays poor, at most 17% top-1 through layer 56
  ([`R1_IT_LENS_REPORT.md`](../experiments/results/R1_IT_LENS_REPORT.md)).

Every headline claim was re-tested by an independent verifier with new controls, but the verifiers that read a lens read
through the same base-fitted ones. One of them (V6) measured J++ against the instruct model’s own gradients and put the
gap down, untested, to the quiet dimensions; the later refit (R1) found it is not confined there. R1 re-read only the
self-gender readouts and the lens’s own fidelity with a lens fitted on the instruct model. The claims that did not survive
are listed in §4 of the report.

## The experiments and their write-ups

- **Main experiments:** E1 self-token readout ([report](../experiments/results/E1_self_readout_REPORT.md)) · E2 behaviour
  ([report](../experiments/results/E2_behavior_REPORT.md)) · E3 lens validation
  ([report](../experiments/results/E3_lens_validation_REPORT.md)) · E4 causal ([report](../experiments/results/E4_causal_REPORT.md)) ·
  E5 name ([report](../experiments/results/E5_name_REPORT.md)) · E6 tool audit
  ([report](../experiments/results/E6_tool_audit_REPORT.md)) · E7 binding ([report](../experiments/results/E7_binding_REPORT.md))
- **Adversarial verifiers:** V1 name behaviour ([notes](../experiments/results/V1_name_behavior_NOTES.md)) · V2 forced choice
  ([notes](../experiments/results/V2_forced_choice_NOTES.md)) · V3 persona axis
  ([notes](../experiments/results/V3_persona_axis_traits_NOTES.md)) · V4 latent identity
  ([notes](../experiments/results/V4_latent_identity_NOTES.md)) · V5 binding dissociation
  ([notes](../experiments/results/V5_binding_dissociation_NOTES.md)) · V6 quiet dims
  ([notes](../experiments/results/V6_quiet_dims_NOTES.md)) · V7 tool bugs ([notes](../experiments/results/V7_tool_bugs_NOTES.md)) ·
  V8 statistics audit ([notes](../experiments/results/V8_stat_audit_NOTES.md))
- **Neuron-label study:** N1A (output, causal), N1B (activation labels) and N1C (integration) in the
  [N1 report](../experiments/results/N1_REPORT.md); the N1V verifiers’ [code audit](../experiments/results/N1V_code_audit.md) and
  [robustness check](../experiments/results/N1V_robustness.md).
- **Lens refit on the instruct model:** R1 ([report](../experiments/results/R1_IT_LENS_REPORT.md)) · R2, the pre-registered
  test of which layers to serve a second lens at, which served none
  ([registration](../experiments/results/R2_PREREG.md), [report](../experiments/results/R2_03_report_section.md)) · R3, J++’s
  recipe fitted on the instruct model, which served none either ([registration](../experiments/results/R3_PREREG.md),
  [report](../experiments/results/R3_06_report_section.md)) · R4, R1’s lens at layers 56 and 59 on fresh prompts, served
  at neither: its lead was in function words ([registration](../experiments/results/R4_PREREG.md),
  [report](../experiments/results/R4_03_report_section.md)).
- **Other scripts:** S1/S2, quick checks on the neuron and lens synthesis; `pilot_self_gender.py`, the first pilot;
  `patch_verify.py`, an end-to-end check of the applied patches.
