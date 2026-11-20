/**
 * Pure request builders, validators, layer conversion (page 1-60 <-> wire 0-59), and replay helpers for the Vault's
 * side-by-side compare tab (`frontend/src/screens/vault/compare.ts`).
 *
 * Kept React-free so `scripts/check_level_page.py` can bundle and test it under esbuild + node without a browser,
 * following the pattern of `screens/atlas/push.ts` and `screens/vault/lql.ts`.
 *
 * No kept-out words appear in code, labels, or generated messages.
 */

export type CompareOp = "silence" | "scale" | "silence_rows" | "scale_rows";
export type CompareLinear = "down_proj" | "gate_proj" | "up_proj";
export type ControlsFamily = "other_sites" | "other_sites_matched" | "same_site";

export interface CompareFormState {
  op: CompareOp;
  linear: CompareLinear;
  /** 1-based page layer ("1".."60"), converted to 0-based on the wire. */
  layer: string;
  /** Comma- or space-separated neuron indices (0..21503). */
  neurons: string;
  factor: string;
  dose: string;
  question: string;
  answer: string;
  system: string;
  controlsFamily: ControlsFamily;
  controlsCount: string;
  controlsSeed: string;
  maxNew: string;
  thinking: boolean;
  sampled: boolean;
  temperature: string;
  samplerSeed: string;
  includePaired: boolean;
  towardWord: string;
  awayWord: string;
  oddsAt: "end" | "span";
}

export interface ComparePayload {
  op: CompareOp;
  linear: CompareLinear;
  layer: number;
  neurons: number[];
  factor: number;
  dose: number;
  question: string;
  answer: string | null;
  system: string | null;
  controls_family: ControlsFamily;
  controls_count: number;
  controls_seed: number;
  max_new: number;
  thinking: boolean;
  sampled: boolean;
  temperature: number;
  sampler_seed: number;
  include_paired: boolean;
  toward_word: string | null;
  away_word: string | null;
  odds_at: "end" | "span";
}

export interface VoiceRow {
  key?: string;
  index?: number;
  dose?: number;
  text: string;
  pieces_count: number;
  ended: boolean;
  same_as_base?: boolean;
  parts_at?: number | null;
  logprob_gap?: { max_abs: number | null; mean_abs: number | null; n: number } | null;
  loop?: { looped: boolean; pattern_len?: number; repeats?: number } | null;
  cut_off?: boolean;
  stratum?: string;
  thinking?: { has_thought: boolean; thought_closed: boolean; thought_pieces: number; answer_pieces: number } | null;
}

export interface VoicesPayload {
  question?: string | null;
  answer?: string | null;
  system?: string | null;
  max_new?: number;
  thinking?: boolean;
  sampler?: Record<string, unknown> | null;
  untouched_rows_identical?: boolean;
  controls_family?: string | null;
  controls_count?: number;
  controls_seed?: number | null;
  base: VoiceRow;
  edit: VoiceRow;
  controls: VoiceRow[];
  summary?: Record<string, unknown> | null;
}

export interface PairedClassRow {
  rows?: number | null;
  metrics: Record<string, { mean_over_items?: number | null; max_over_items?: number | null; median_over_items?: number | null }>;
  delivered?: { mean?: number | null; max?: number | null; rel_mean?: number | null } | null;
  floor?: Record<string, { mean_abs?: number | null; max_abs?: number | null }> | null;
  folded?: { taken?: boolean; status?: string; max_abs?: number | null } | null;
  rank?: { at_least?: number | null; of?: number | null; rank_by?: string } | null;
  controls_count: number;
  at_least: number | null;
  control_abs_max: number | null;
  control_abs_median: number | null;
}

export interface PairedSummary {
  rank_by: string;
  main_dose: number;
  controls?: Record<string, unknown> | null;
  items_summary?: { asked?: number; kept?: number; unstable?: number; cautious?: number } | null;
  classes: Record<string, PairedClassRow>;
}

export interface CompareReply {
  v: number;
  status: string;
  why?: string | null;
  model_id: string;
  weights_id: string;
  edit: {
    id: string;
    short: string;
    linears: string[];
    layer: number;
    page_layer: number;
    op: CompareOp;
    neurons: number[];
    dose: number;
  };
  voices: VoicesPayload | null;
  paired: PairedSummary | null;
  written_lines: string[];
  paired_lines: string[];
  seconds: number;
  base_after: boolean;
  book: { written_seq?: number | null; paired_seq?: number | null; entries: number; head: string };
  note: string;
}

export interface ReplayCard {
  seq: number;
  at: string;
  hash: string;
  kind: "written" | "paired" | "note" | string;
  status?: string | null;
  why?: string | null;
  name?: string | null;
  edit_id?: string | null;
  edit_linears?: string[] | null;
  seconds?: number | null;
  base_after?: boolean | null;
  voices?: VoicesPayload | null;
  paired?: PairedSummary | null;
  lines?: string[];
  note_type?: string | null;
  brief_name?: string | null;
  brief_id?: string | null;
  verdict?: string | null;
  reasons?: string[];
  claim_sentence?: string | null;
  summary?: Record<string, unknown> | null;
  batteries?: Record<string, unknown> | null;
  cheaper_knobs?: Record<string, unknown> | null;
  bars?: Record<string, unknown> | null;
  bars_sha256?: string | null;
  book_seqs?: number[] | null;
  selection_mode?: string | null;
  controls_per_family?: number | null;
  folded_summary?: Record<string, unknown> | null;
  reworded_summary?: Record<string, unknown> | null;
  trial_sha256?: string | null;
  of_record?: boolean | null;
  not_of_record?: string[] | null;
  recorded_tag?: string | null;
}

export interface BookInfo {
  id: string;
  label: string;
  recorded: boolean;
  exists: boolean;
  entries: number;
  ok: boolean;
  head: string;
}

export interface LevelStatusReply {
  v: number;
  runtime_ready: boolean;
  model_id: string | null;
  n_layers: number | null;
  d_ff: number | null;
  is_base: boolean | null;
  active_vault_edits: number;
  live_book: {
    path: string;
    verify: { ok: boolean; entries: number; head: string; first_bad: number | null; why: string | null } | null;
  };
  books: BookInfo[];
  starter_briefs: Array<{
    name: string;
    brief_id: string;
    metric: string;
    direction: number;
    least_effect: number;
    max_fresh_readings: number;
    controls_per_family: number;
    behavior: string;
  }>;
}

export interface EstimateReply {
  written: { forwards: number; seconds: number; gib_above_model: number };
  paired: { forwards: number; seconds: number; gib_above_model: number } | null;
  total_seconds: number;
  total_forwards: number;
  peak_gib_above_model: number;
}

export interface ComparePreset {
  id: string;
  label: string;
  note: string;
  patch: Partial<CompareFormState>;
}

export const COMPARE_PRESETS: ComparePreset[] = [
  {
    id: "e6-she-10",
    label: "E6 layer-51 ‘she’ ten",
    note: "Top-10 ‘she’ neurons from the vault at layer 51 (index 50); moves female contexts strongly and bleeds into male near misses.",
    patch: {
      op: "silence",
      linear: "down_proj",
      layer: "51",
      neurons: "5313, 8801, 266, 121, 11503, 11724, 21151, 4165, 549, 2070",
      dose: "1",
      question: "Continue naturally: Claire was an experienced architect. Before leaving the gallery,",
      answer: " Claire picked up the notebook and",
      includePaired: true,
      towardWord: "she",
      awayWord: "he",
      oddsAt: "end",
    },
  },
  {
    id: "e6-she-only-7",
    label: "E6 layer-51 ‘she’-only seven",
    note: "The seven ‘she’ neurons not shared with ‘he’ (negative control: drops the shift to the noise floor on 31B).",
    patch: {
      op: "silence",
      linear: "down_proj",
      layer: "51",
      neurons: "266, 121, 11503, 11724, 21151, 4165, 2070",
      dose: "1",
      question: "Continue naturally: Claire was an experienced architect. Before leaving the gallery,",
      answer: " Claire picked up the notebook and",
      includePaired: true,
      towardWord: "she",
      awayWord: "he",
      oddsAt: "end",
    },
  },
  {
    id: "l21-two",
    label: "layer-21 pair (2 neurons)",
    note: "Two mid-depth neurons silenced on a neutral factual question.",
    patch: {
      op: "silence",
      linear: "down_proj",
      layer: "21",
      neurons: "4294, 9618",
      dose: "1",
      question: "In two or three sentences, why does the moon show phases?",
      answer: "The moon shows phases because we see different portions of its sunlit half as it orbits the Earth.",
      includePaired: true,
      towardWord: "",
      awayWord: "",
    },
  },
];

export const DEFAULT_COMPARE_FORM: CompareFormState = {
  op: "silence",
  linear: "down_proj",
  layer: "51",
  neurons: "5313, 8801, 266, 121, 11503, 11724, 21151, 4165, 549, 2070",
  factor: "0.5",
  dose: "1",
  question: "Continue naturally: Claire was an experienced architect. Before leaving the gallery,",
  answer: " Claire picked up the notebook and",
  system: "",
  controlsFamily: "other_sites_matched",
  controlsCount: "3",
  controlsSeed: "0",
  maxNew: "32",
  thinking: false,
  sampled: false,
  temperature: "0.8",
  samplerSeed: "1",
  includePaired: true,
  towardWord: "she",
  awayWord: "he",
  oddsAt: "end",
};

export interface ValidationErrors {
  layer?: string;
  neurons?: string;
  factor?: string;
  dose?: string;
  question?: string;
  controlsCount?: string;
  controlsSeed?: string;
  maxNew?: string;
  temperature?: string;
  odds?: string;
}

export function parseIntStrict(s: string): number | null {
  const t = s.trim();
  if (!/^-?\d+$/.test(t)) return null;
  const n = Number(t);
  return Number.isSafeInteger(n) ? n : null;
}

export function parseFloatStrict(s: string): number | null {
  const t = s.trim();
  if (t === "" || !/^-?(\d+\.?\d*|\.\d+)$/.test(t)) return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

export function parseNeuronList(raw: string, maxIdx = 21503): { ok: true; neurons: number[] } | { ok: false; error: string } {
  const parts = raw
    .split(/[\s,;]+/)
    .map((p) => p.trim())
    .filter(Boolean);
  if (parts.length === 0) return { ok: false, error: "Enter at least one neuron number (0 to " + maxIdx + ")." };
  if (parts.length > 64) return { ok: false, error: "At most 64 neurons may be entered in one side-by-side reading." };
  const out: number[] = [];
  const seen = new Set<number>();
  for (const p of parts) {
    const n = parseIntStrict(p);
    if (n === null || n < 0 || n > maxIdx) {
      return { ok: false, error: `“${p}” is not a neuron index between 0 and ${maxIdx}.` };
    }
    if (!seen.has(n)) {
      seen.add(n);
      out.push(n);
    }
  }
  return { ok: true, neurons: out };
}

export function validateCompareForm(form: CompareFormState, maxLayer = 60, maxNeuron = 21503): ValidationErrors {
  const errs: ValidationErrors = {};
  const L = parseIntStrict(form.layer);
  if (L === null || L < 1 || L > maxLayer) {
    errs.layer = `Layers run from 1 to ${maxLayer} on the page.`;
  }
  const nl = parseNeuronList(form.neurons, maxNeuron);
  if (!nl.ok) {
    errs.neurons = nl.error;
  }
  if (form.op === "scale" || form.op === "scale_rows") {
    const f = parseFloatStrict(form.factor);
    if (f === null || Math.abs(f) > 16) {
      errs.factor = "Scale factor must be a finite number between -16 and 16.";
    }
  }
  const dose = parseFloatStrict(form.dose);
  if (dose === null || dose === 0 || Math.abs(dose) > 8) {
    errs.dose = "Dose must be a non-zero number between -8 and 8.";
  }
  if (form.question.trim() === "") {
    errs.question = "Enter a question for the three voices to answer.";
  }
  const cc = parseIntStrict(form.controlsCount);
  if (cc === null || cc < 1 || cc > 19) {
    errs.controlsCount = "Choose between 1 and 19 control edits.";
  }
  const cs = parseIntStrict(form.controlsSeed);
  if (cs === null || cs < 0) {
    errs.controlsSeed = "Control seed must be a whole number from 0 upward.";
  }
  const mn = parseIntStrict(form.maxNew);
  if (mn === null || mn < 4 || mn > 256) {
    errs.maxNew = "Written length must be between 4 and 256 word-pieces.";
  }
  if (form.sampled) {
    const temp = parseFloatStrict(form.temperature);
    if (temp === null || temp <= 0 || temp > 2) {
      errs.temperature = "Sampled temperature must be above 0 and at most 2.";
    }
  }
  const tw = form.towardWord.trim();
  const aw = form.awayWord.trim();
  if (form.includePaired && ((tw !== "" && aw === "") || (tw === "" && aw !== ""))) {
    errs.odds = "Give both a toward word and an away word for contrast odds, or leave both blank.";
  }
  return errs;
}

export function buildComparePayload(form: CompareFormState, maxLayer = 60, maxNeuron = 21503): ComparePayload {
  const errs = validateCompareForm(form, maxLayer, maxNeuron);
  const first = Object.values(errs)[0];
  if (first) throw new Error(first);

  const pageLayer = parseIntStrict(form.layer)!;
  const parsed = parseNeuronList(form.neurons, maxNeuron);
  if (!parsed.ok) throw new Error(parsed.error);
  const isRow = form.op === "silence_rows" || form.op === "scale_rows";
  const linear: CompareLinear = isRow
    ? form.linear === "gate_proj" || form.linear === "up_proj"
      ? form.linear
      : "gate_proj"
    : "down_proj";
  const factor = form.op === "scale" || form.op === "scale_rows" ? parseFloatStrict(form.factor)! : 0;
  const ans = form.answer.trim();
  const sys = form.system.trim();
  const tw = form.towardWord.trim();
  const aw = form.awayWord.trim();

  return {
    op: form.op,
    linear,
    layer: pageLayer - 1, // page 1..60 -> wire 0..59
    neurons: parsed.neurons,
    factor,
    dose: parseFloatStrict(form.dose)!,
    question: form.question.trim(),
    answer: ans || null,
    system: sys || null,
    controls_family: form.controlsFamily,
    controls_count: parseIntStrict(form.controlsCount)!,
    controls_seed: parseIntStrict(form.controlsSeed)!,
    max_new: parseIntStrict(form.maxNew)!,
    thinking: Boolean(form.thinking),
    sampled: Boolean(form.sampled),
    temperature: form.sampled ? parseFloatStrict(form.temperature)! : 0,
    sampler_seed: parseIntStrict(form.samplerSeed) ?? 1,
    include_paired: Boolean(form.includePaired && ans !== ""),
    toward_word: tw && aw ? tw : null,
    away_word: tw && aw ? aw : null,
    odds_at: form.oddsAt,
  };
}

export function formatPlainEdit(form: CompareFormState): string {
  const parsed = parseNeuronList(form.neurons);
  const countTxt = parsed.ok
    ? parsed.neurons.length === 1
      ? `neuron ${parsed.neurons[0]}`
      : `${parsed.neurons.length} neurons (${parsed.neurons.slice(0, 4).join(", ")}${parsed.neurons.length > 4 ? ", …" : ""})`
    : "selected neurons";
  const L = parseIntStrict(form.layer);
  const layerTxt = L !== null ? `layer ${L} (index ${L - 1})` : "layer …";
  const actionTxt =
    form.op === "silence"
      ? `Silence ${countTxt} of down_proj in ${layerTxt}`
      : form.op === "scale"
        ? `Scale ${countTxt} of down_proj in ${layerTxt} by ×${form.factor || "…"}`
        : form.op === "silence_rows"
          ? `Silence ${countTxt} on ${form.linear} rows in ${layerTxt}`
          : `Scale ${countTxt} on ${form.linear} rows in ${layerTxt} by ×${form.factor || "…"}`;
  const ctlTxt =
    form.controlsFamily === "other_sites_matched"
      ? `${form.controlsCount || "?"} delivery-matched other-site control edits`
      : form.controlsFamily === "same_site"
        ? `${form.controlsCount || "?"} same-site random-direction control edits`
        : `${form.controlsCount || "?"} random other-site control edits`;
  return `${actionTxt}, heard in the same forward beside the untouched base and ${ctlTxt}. Nothing remains on the model once this reading finishes.`;
}

export function fmtMetric(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return (v < 0 ? "−" : "") + Math.abs(v).toFixed(digits);
}

export function fmtSigned(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return (v < 0 ? "−" : "+") + Math.abs(v).toFixed(digits);
}

export function controlsFamilyGloss(fam: string | null | undefined): string {
  if (fam === "other_sites_matched") return "other sites matched on delivered stream change";
  if (fam === "same_site") return "same sites with random masked directions of equal length";
  if (fam === "sham_target") return "declared sham-target control edits";
  return "other sites of the same layer and size";
}

/** Label for a ledger replay row's verdict / status badge. */
export function statusTone(status: string | null | undefined): "good" | "warn" | "bad" | "neutral" {
  const s = (status || "").toLowerCase();
  if (s === "lead" || s === "passed") return "good";
  if (s === "ok" || s === "calibrated" || s === "registered") return "neutral";
  if (
    s === "lead, matched by a cheaper knob" ||
    s === "passed, matched by a cheaper knob" ||
    s === "kept the base" ||
    s === "unstable"
  ) {
    return "warn";
  }
  if (
    s === "failed" ||
    s === "works but harms" ||
    s === "no better than control edits" ||
    s === "did not replicate" ||
    s === "void" ||
    s === "refused"
  ) {
    return "bad";
  }
  return "neutral";
}

/** Label and tone for a retention battery chip (never calls an unavailable battery "passed"). */
export function batteryOutcome(
  bv: Record<string, unknown> | null | undefined,
): { label: "passed" | "failed" | "not run"; tone: "good" | "bad" | "neutral" } {
  const available = bv?.available !== false;
  const gating = Boolean(bv?.gating);
  if (!available) {
    return { label: "not run", tone: gating ? "bad" : "neutral" };
  }
  return Boolean(bv?.passed) ? { label: "passed", tone: "good" } : { label: "failed", tone: "bad" };
}
