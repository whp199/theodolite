import React, { useEffect, useId, useRef, useState } from "react";
import { Apparatus, Rubric } from "../../components/kit";
import { useTheodoliteStore } from "../../store/useTheodoliteStore";
import { EmptyAnswer, MONO, chipStyle, sentenceInput, smallCaps } from "./bits";
import {
  COMPARE_PRESETS,
  type BookInfo,
  type CompareFormState,
  type CompareLinear,
  type CompareOp,
  type CompareReply,
  type ControlsFamily,
  type EstimateReply,
  type LevelStatusReply,
  type PairedSummary,
  type ReplayCard,
  type ValidationErrors,
  type VoiceRow,
  type VoicesPayload,
  batteryOutcome,
  buildComparePayload,
  controlsFamilyGloss,
  fmtMetric,
  fmtSigned,
  formatPlainEdit,
  statusTone,
  validateCompareForm,
} from "./compare";
import { RECORDED_REPLAY_CARDS } from "./compareRecorded";
import { useVaultUi } from "./ui";

const radioStyle: React.CSSProperties = { width: 18, height: 18, accentColor: "var(--rubric)", margin: 0 };
const radioLabel: React.CSSProperties = { display: "flex", alignItems: "center", gap: 8, minHeight: 44, fontSize: 18, cursor: "pointer" };
const fieldsetStyle: React.CSSProperties = {
  border: 0,
  margin: 0,
  padding: 0,
  display: "flex",
  flexWrap: "wrap",
  alignItems: "center",
  gap: "2px 24px",
};

function badgeColor(tone: "good" | "warn" | "bad" | "neutral"): string {
  if (tone === "good") return "var(--lapis)";
  if (tone === "bad") return "var(--rubric)";
  if (tone === "warn") return "var(--gilt-text)";
  return "var(--muted)";
}

const StatusPill: React.FC<{ text: string }> = ({ text }) => {
  const tone = statusTone(text);
  const col = badgeColor(tone);
  return (
    <span
      style={{
        display: "inline-block",
        border: `1px solid ${col}`,
        color: col,
        padding: "1px 8px",
        borderRadius: 2,
        fontSize: 13,
        fontVariantCaps: "all-small-caps",
        letterSpacing: "0.1em",
        lineHeight: 1.4,
      }}
    >
      {text}
    </span>
  );
};

/** Side-by-side three-voice display: Untouched base | Your edit | Control edit (with stepper across all C controls). */
const ThreeVoices: React.FC<{ voices: VoicesPayload; label?: string }> = ({ voices, label }) => {
  const [ctlIdx, setCtlIdx] = useState(0);
  const controls = voices.controls || [];
  const safeIdx = controls.length > 0 ? Math.min(ctlIdx, controls.length - 1) : 0;
  const ctl: VoiceRow | null = controls.length > 0 ? controls[safeIdx] : null;

  const voiceMeta = (v: VoiceRow, isBase: boolean) => {
    const bits: string[] = [`${v.pieces_count} pieces`, v.ended ? "finished" : "hit cap"];
    if (!isBase) {
      if (v.same_as_base) bits.push("same as base");
      else if (typeof v.parts_at === "number") bits.push(`parted at piece ${v.parts_at + 1}`);
      if (v.logprob_gap && typeof v.logprob_gap.max_abs === "number") {
        bits.push(`max Δlp ${v.logprob_gap.max_abs.toFixed(3)}`);
      }
    }
    if (v.loop?.looped) bits.push("loop detected");
    if (v.cut_off) bits.push("cut off");
    return bits.join(" · ");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {label ? <span style={smallCaps()}>{label}</span> : null}
      {voices.question ? (
        <p style={{ margin: 0, fontSize: 17, color: "var(--muted)" }}>
          <span style={smallCaps(13)}>question </span>
          <span style={{ color: "var(--ink)" }}>“{voices.question}”</span>
          {voices.thinking ? " · thinking on" : " · thinking off"}
          {voices.untouched_rows_identical === false ? " · untouched row mismatch" : ""}
        </p>
      ) : null}
      <div className="lvl-voices">
        <article className="lvl-voice-card">
          <div className="lvl-voice-head">
            <span style={smallCaps(15)}>untouched base</span>
            <span className="num" style={{ fontSize: 13, color: "var(--muted)" }}>
              row 0
            </span>
          </div>
          <Rubric style={{ fontSize: 14, margin: "4px 0 8px" }}>
            What the unedited checkpoint wrote in this very forward; the benchmark both other voices stand beside.
          </Rubric>
          <p className="lvl-voice-text">{voices.base.text || "(empty)"}</p>
          <p className="num" style={{ margin: "10px 0 0", fontSize: 13, color: "var(--muted)" }}>
            {voiceMeta(voices.base, true)}
          </p>
        </article>

        <article className="lvl-voice-card lvl-voice-edit">
          <div className="lvl-voice-head">
            <span style={{ ...smallCaps(15), color: "var(--gilt-text)" }}>your edit</span>
            <span className="num" style={{ fontSize: 13, color: "var(--muted)" }}>
              dose {voices.edit.dose ?? 1}
            </span>
          </div>
          <Rubric style={{ fontSize: 14, margin: "4px 0 8px" }}>
            Applied to this row alone for one reading; cleared before the reply returns so the base stays untouched.
          </Rubric>
          <p className="lvl-voice-text">{voices.edit.text || "(empty)"}</p>
          <p className="num" style={{ margin: "10px 0 0", fontSize: 13, color: "var(--muted)" }}>
            {voiceMeta(voices.edit, false)}
          </p>
        </article>

        <article className="lvl-voice-card">
          <div className="lvl-voice-head">
            <span style={smallCaps(15)}>control edit</span>
            {controls.length > 1 ? (
              <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                <button
                  type="button"
                  onClick={() => setCtlIdx((i) => (i - 1 + controls.length) % controls.length)}
                  aria-label="Previous control edit"
                  style={chipStyle(false, { minHeight: 28, minWidth: 28, padding: "0 6px", fontSize: 14 })}
                >
                  ‹
                </button>
                <span className="num" style={{ fontSize: 13, color: "var(--muted)" }}>
                  {safeIdx + 1} / {controls.length}
                </span>
                <button
                  type="button"
                  onClick={() => setCtlIdx((i) => (i + 1) % controls.length)}
                  aria-label="Next control edit"
                  style={chipStyle(false, { minHeight: 28, minWidth: 28, padding: "0 6px", fontSize: 14 })}
                >
                  ›
                </button>
              </span>
            ) : (
              <span className="num" style={{ fontSize: 13, color: "var(--muted)" }}>
                1 of {controls.length || 0}
              </span>
            )}
          </div>
          <Rubric style={{ fontSize: 14, margin: "4px 0 8px" }}>
            Same-size control ({controlsFamilyGloss(voices.controls_family)}): if this row shifts as far, your edit is not specific.
          </Rubric>
          <p className="lvl-voice-text">{ctl ? ctl.text || "(empty)" : "(no control row recorded)"}</p>
          <p className="num" style={{ margin: "10px 0 0", fontSize: 13, color: "var(--muted)" }}>
            {ctl ? voiceMeta(ctl, false) : "—"}
          </p>
        </article>
      </div>
    </div>
  );
};

/** Responsive two-batch-class paired metrics table (`responsive-table` guidance: sticky headers + @container stacked cards). */
const PairedMetricsTable: React.FC<{ paired: PairedSummary; captionId: string }> = ({ paired, captionId }) => {
  const classes = paired.classes || {};
  const hasOdds = Object.values(classes).some((c) => c.metrics?.odds?.mean_over_items !== undefined);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <Rubric style={{ maxWidth: 820 }}>
        Read in two batch classes (12 and 32 rows) beside null rows that measure floating-point rounding. A rank against control edits says
        only that this edit differed from those random edits on this text—never a claim of record.
      </Rubric>
      <div tabIndex={0} role="region" aria-labelledby={captionId} className="lvl-table-wrap">
        <table
          className="lvl-table"
          style={
            {
              "--label-1": '"Batch class"',
              "--label-2": '"Floor (null)"',
              "--label-3": '"KL (nats)"',
              "--label-4": '"Own Δlp"',
              "--label-5": hasOdds ? '"Odds Δ"' : '"Delivered"',
              "--label-6": hasOdds ? '"Delivered"' : '"Rank vs controls"',
              "--label-7": hasOdds ? '"Rank vs controls"' : '"Folded check"',
              "--label-8": '"Folded check"',
            } as React.CSSProperties
          }
        >
          <caption id={captionId} style={{ textAlign: "left", padding: "0 0 8px", ...smallCaps(14) }}>
            paired next-piece read-out (dose {paired.main_dose}, ranked by {paired.rank_by})
          </caption>
          <thead>
            <tr>
              <th scope="col">Batch class</th>
              <th scope="col">Floor (null)</th>
              <th scope="col">KL (nats)</th>
              <th scope="col">Own Δlp</th>
              {hasOdds ? <th scope="col">Odds Δ</th> : null}
              <th scope="col">Delivered</th>
              <th scope="col">Rank vs controls</th>
              <th scope="col">Folded check</th>
            </tr>
          </thead>
          <tbody>
            {(["small", "large"] as const).map((cls) => {
              const row = classes[cls];
              if (!row) return null;
              const flMetric = row.floor?.[paired.rank_by]?.mean_abs ?? row.floor?.kl?.mean_abs ?? null;
              const klVal = row.metrics?.kl?.mean_over_items ?? null;
              const ownVal = row.metrics?.own?.mean_over_items ?? null;
              const oddsVal = row.metrics?.odds?.mean_over_items ?? null;
              const deliv = row.delivered?.mean ?? null;
              const rankTxt =
                row.at_least !== null && row.controls_count > 0
                  ? `${row.at_least} / ${row.controls_count} as large`
                  : row.controls_count > 0
                    ? `0 / ${row.controls_count}`
                    : "—";
              const foldedTxt = row.folded?.taken
                ? `${row.folded.status ?? "ok"}${typeof row.folded.max_abs === "number" ? ` (${row.folded.max_abs.toExponential(1)})` : ""}`
                : "not taken";
              return (
                <tr key={cls}>
                  <th scope="row" className="num">
                    {cls} ({row.rows ?? "?"} rows)
                  </th>
                  <td className="num">{fmtMetric(flMetric, 4)}</td>
                  <td className="num">{fmtMetric(klVal, 4)}</td>
                  <td className="num">{fmtSigned(ownVal, 4)}</td>
                  {hasOdds ? <td className="num">{fmtSigned(oddsVal, 4)}</td> : null}
                  <td className="num">{fmtMetric(deliv, 4)}</td>
                  <td className="num">{rankTxt}</td>
                  <td className="num">{foldedTxt}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

/** Detailed replay view for one selected level-book entry (written, paired, or Phase 3 claim note). */
const ReplayInspector: React.FC<{ card: ReplayCard }> = ({ card }) => {
  const capId = useId();
  const sum = card.summary as Record<string, Record<string, unknown>> | null | undefined;
  const bats = card.batteries as Record<string, Record<string, unknown>> | null | undefined;
  return (
    <div
      style={{
        borderTop: "1px solid var(--gilt-rule)",
        borderBottom: "1px solid var(--hair)",
        padding: "20px 0 24px",
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "8px 16px" }}>
        <span style={smallCaps(16)}>
          replay · book line {card.seq} ({card.kind}
          {card.note_type ? ` / ${card.note_type}` : ""})
        </span>
        {card.status ? <StatusPill text={card.status} /> : null}
        <span className="num" style={{ fontSize: 14, color: "var(--muted)" }}>
          {card.at} · hash <code style={{ fontFamily: MONO, fontSize: 13 }}>{card.hash}</code>
        </span>
        {card.recorded_tag ? (
          <span style={{ fontSize: 14, fontStyle: "italic", color: "var(--gilt-text)" }}>[{card.recorded_tag}]</span>
        ) : null}
      </div>

      {card.name ? (
        <p style={{ margin: 0, fontSize: 20, fontWeight: 500, color: "var(--ink)" }}>{card.name}</p>
      ) : null}

      {card.edit_id ? (
        <p className="num" style={{ margin: 0, fontSize: 15, color: "var(--muted)" }}>
          edit <code style={{ fontFamily: MONO, fontSize: 14 }}>{card.edit_id.slice(0, 16)}</code>
          {card.edit_linears && card.edit_linears.length > 0 ? ` · ${card.edit_linears.join(", ")}` : ""}
          {typeof card.seconds === "number" ? ` · ${card.seconds.toFixed(2)} s` : ""}
          {card.bars_sha256 ? ` · bars ${card.bars_sha256.slice(0, 12)}` : ""}
        </p>
      ) : null}

      {card.voices ? <ThreeVoices voices={card.voices} label="recorded three voices (replayed without a model)" /> : null}
      {card.paired ? <PairedMetricsTable paired={card.paired} captionId={capId} /> : null}

      {card.note_type === "claim_verdict" || card.note_type === "trial_of_record" ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {card.claim_sentence ? (
            <blockquote
              style={{
                margin: 0,
                padding: "10px 16px",
                borderLeft: "2px solid var(--gilt)",
                fontSize: 17,
                lineHeight: 1.55,
              }}
            >
              {card.claim_sentence}
            </blockquote>
          ) : null}
          {card.reasons && card.reasons.length > 0 ? (
            <div>
              <span style={smallCaps(14)}>gate notes</span>
              <ul style={{ margin: "6px 0 0", paddingLeft: 22, fontSize: 16, color: "var(--rubric-deep)" }}>
                {card.reasons.map((r, idx) => (
                  <li key={idx}>{r}</li>
                ))}
              </ul>
            </div>
          ) : null}
          {sum ? (
            <div tabIndex={0} role="region" aria-labelledby={capId} className="lvl-table-wrap">
              <table
                className="lvl-table"
                style={
                  {
                    "--label-1": '"Class"',
                    "--label-2": '"Frames"',
                    "--label-3": '"Mean (95% CI)"',
                    "--label-4": '"Half 0 / Half 1"',
                    "--label-5": '"Floor / Bar"',
                    "--label-6": '"Matched controls"',
                    "--label-7": '"Same-site controls"',
                  } as React.CSSProperties
                }
              >
                <caption id={capId} style={{ textAlign: "left", padding: "0 0 8px", ...smallCaps(14) }}>
                  kept-back frame summary
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Class</th>
                    <th scope="col">Frames</th>
                    <th scope="col">Mean (95% CI)</th>
                    <th scope="col">Half 0 / Half 1</th>
                    <th scope="col">Floor / Bar</th>
                    <th scope="col">Matched controls</th>
                    <th scope="col">Same-site controls</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(sum).map(([cls, d]) => {
                    const ci = (d.ci95 as [number | null, number | null] | undefined) ?? [null, null];
                    const h0 = (d.half_0 as { mean?: number | null } | undefined)?.mean;
                    const h1 = (d.half_1 as { mean?: number | null } | undefined)?.mean;
                    const hb = (d.house_bar as { floor?: number | null; bar?: number | null } | undefined) ?? {};
                    const ctls = (d.controls as Record<string, { at_least?: number | null; count?: number; p_fw?: number | null }> | undefined) ?? {};
                    const osm = ctls.other_sites_matched;
                    const ss = ctls.same_site;
                    return (
                      <tr key={cls}>
                        <th scope="row" className="num">
                          {cls}
                        </th>
                        <td className="num">{String(d.n_frames ?? "—")}</td>
                        <td className="num">
                          {fmtSigned(d.mean as number | null, 4)} [{fmtSigned(ci[0], 3)}, {fmtSigned(ci[1], 3)}]
                        </td>
                        <td className="num">
                          {fmtSigned(h0, 3)} / {fmtSigned(h1, 3)}
                        </td>
                        <td className="num">
                          {fmtMetric(hb.floor, 4)} / {fmtMetric(hb.bar, 4)}
                        </td>
                        <td className="num">
                          {osm?.count ? `${osm.at_least}/${osm.count} (p_fw ${fmtMetric(osm.p_fw, 3)})` : "—"}
                        </td>
                        <td className="num">
                          {ss?.count ? `${ss.at_least}/${ss.count} (p_fw ${fmtMetric(ss.p_fw, 3)})` : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : null}
          {bats ? (
            <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 22px", fontSize: 15 }}>
              {Object.entries(bats).map(([bname, bv]) => {
                const outcome = batteryOutcome(bv);
                const gating = Boolean(bv?.gating);
                return (
                  <span key={bname} className="num">
                    <span style={smallCaps(13)}>{bname}</span>{" "}
                    <span style={{ color: badgeColor(outcome.tone) }}>
                      {outcome.label}
                    </span>
                    {gating ? " (gating)" : " (info)"}
                    {typeof bv?.mean_kl === "number" ? ` · KL ${bv.mean_kl.toExponential(1)}` : ""}
                    {typeof bv?.abs_effect === "number" ? ` · |Δ| ${bv.abs_effect.toFixed(3)}` : ""}
                  </span>
                );
              })}
            </div>
          ) : null}
          {card.note_type === "trial_of_record" && (card.folded_summary || card.reworded_summary) ? (
            <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 22px", fontSize: 15 }}>
              {card.folded_summary ? (
                <span className="num">
                  <span style={smallCaps(13)}>folded form</span>{" "}
                  <span style={{ color: badgeColor((card.folded_summary as { ok?: boolean }).ok ? "good" : "bad") }}>
                    {(card.folded_summary as { status?: string }).status ?? ((card.folded_summary as { ok?: boolean }).ok ? "ok" : "failed")}
                  </span>
                </span>
              ) : null}
              {card.reworded_summary ? (
                <span className="num">
                  <span style={smallCaps(13)}>reworded set</span>{" "}
                  <span style={{ color: badgeColor((card.reworded_summary as { ok?: boolean }).ok ? "good" : "bad") }}>
                    {(card.reworded_summary as { status?: string }).status ?? ((card.reworded_summary as { ok?: boolean }).ok ? "ok" : "failed")}
                  </span>
                </span>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : null}

      {card.lines && card.lines.length > 0 ? (
        <Apparatus title="raw kernel lines" label="Raw kernel lines for this book entry">
          <pre style={{ margin: 0, fontFamily: MONO, fontSize: 13, lineHeight: 1.5, whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
            {card.lines.join("\n")}
          </pre>
        </Apparatus>
      ) : null}
    </div>
  );
};

export const ComparePanel: React.FC = () => {
  const ui = useVaultUi();
  const activeEdits = useTheodoliteStore((s) => s.activeEdits);
  const form = ui.compareForm;
  const setForm = (patch: Partial<CompareFormState>) => ui.set({ compareForm: { ...form, ...patch } });

  const [status, setStatus] = useState<LevelStatusReply | null>(null);
  const [statusErr, setStatusErr] = useState<string | null>(null);
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [submitted, setSubmitted] = useState(false);
  const [announcement, setAnnouncement] = useState("");
  const [estimate, setEstimate] = useState<EstimateReply | null>(null);
  const [estimating, setEstimating] = useState(false);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [reply, setReply] = useState<CompareReply | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Replay state (works with no model loaded, falling back to recorded 31B cards offline)
  const [bookId, setBookId] = useState<string>("live");
  const [bookVerify, setBookVerify] = useState<{ ok: boolean; entries: number; head: string; why?: string | null } | null>(null);
  const [bookEntries, setBookEntries] = useState<ReplayCard[]>([]);
  const [bookOffline, setBookOffline] = useState(false);
  const [selectedCard, setSelectedCard] = useState<ReplayCard | null>(null);
  const [bookLoading, setBookLoading] = useState(false);

  const layerRef = useRef<HTMLInputElement | null>(null);
  const neuronsRef = useRef<HTMLInputElement | null>(null);
  const factorRef = useRef<HTMLInputElement | null>(null);
  const doseRef = useRef<HTMLInputElement | null>(null);
  const questionRef = useRef<HTMLTextAreaElement | null>(null);
  const ctlCountRef = useRef<HTMLInputElement | null>(null);
  const maxNewRef = useRef<HTMLInputElement | null>(null);

  const liveMetricsCapId = useId();
  const replayTableCapId = useId();

  const maxLayer = status?.n_layers ?? 60;
  const maxNeuron = status?.d_ff ? status.d_ff - 1 : 21503;
  const errors: ValidationErrors = validateCompareForm(form, maxLayer, maxNeuron);
  const showError = (field: keyof ValidationErrors) => Boolean(errors[field] && (submitted || touched[field]));

  const announceLater = (msg: string) => {
    setAnnouncement("");
    window.setTimeout(() => setAnnouncement(msg), 60);
  };

  const markBlur = (field: keyof ValidationErrors) => {
    setTouched((prev) => ({ ...prev, [field]: true }));
    const err = errors[field];
    if (err) announceLater(err);
  };

  // Load status and default replay book on mount
  const loadStatus = async () => {
    try {
      const r = await fetch("/api/level/status");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = (await r.json()) as LevelStatusReply;
      setStatus(data);
      setStatusErr(null);
      // If live book is empty and a verified 31B book exists, stay on live if user selected it, otherwise show verify_claim when empty
      if (data.live_book?.verify?.entries === 0) {
        const rec = data.books.find((b) => b.id === "verify_claim" && b.exists);
        if (rec) setBookId("verify_claim");
      }
    } catch {
      setStatus(null);
      setStatusErr("No live server answered; side-by-side readings wait on the server, while recorded 31B replays below work offline.");
      setBookOffline(true);
      setBookEntries(RECORDED_REPLAY_CARDS);
      setSelectedCard(RECORDED_REPLAY_CARDS[0] ?? null);
    }
  };

  const loadBook = async (id: string) => {
    setBookLoading(true);
    try {
      const r = await fetch(`/api/level/book?book=${encodeURIComponent(id)}&tail=40`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = (await r.json()) as {
        verify: { ok: boolean; entries: number; head: string; why?: string | null };
        entries: ReplayCard[];
      };
      setBookOffline(false);
      setBookVerify(data.verify);
      setBookEntries(data.entries);
      if (data.entries.length > 0) {
        setSelectedCard((prev) => (prev && data.entries.some((e) => e.seq === prev.seq) ? prev : data.entries[0]));
      } else {
        setSelectedCard(null);
      }
    } catch {
      setBookOffline(true);
      setBookVerify(null);
      setBookEntries(RECORDED_REPLAY_CARDS);
      setSelectedCard(RECORDED_REPLAY_CARDS[0] ?? null);
    } finally {
      setBookLoading(false);
    }
  };

  useEffect(() => {
    void loadStatus();
  }, []);

  useEffect(() => {
    if (!bookOffline) void loadBook(bookId);
  }, [bookId, bookOffline]);

  const focusFirstError = (errs: ValidationErrors) => {
    const order: [keyof ValidationErrors, React.RefObject<HTMLElement | null>][] = [
      ["neurons", neuronsRef],
      ["layer", layerRef],
      ["factor", factorRef],
      ["dose", doseRef],
      ["question", questionRef],
      ["controlsCount", ctlCountRef],
      ["maxNew", maxNewRef],
    ];
    for (const [k, ref] of order) {
      if (errs[k] && ref.current) {
        ref.current.focus();
        announceLater(errs[k]!);
        return;
      }
    }
  };

  const runEstimate = async () => {
    setSubmitted(true);
    if (Object.keys(errors).length > 0) {
      focusFirstError(errors);
      return;
    }
    setEstimating(true);
    setRunError(null);
    try {
      const payload = buildComparePayload(form, maxLayer, maxNeuron);
      const r = await fetch("/api/level/estimate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body?.detail || `HTTP ${r.status}`);
      setEstimate(body as EstimateReply);
      announceLater(`Estimated ${body.total_forwards} forwards, about ${body.total_seconds} seconds.`);
    } catch (exc) {
      const msg = (exc as Error).message || String(exc);
      setRunError(msg);
      announceLater(msg);
    } finally {
      setEstimating(false);
    }
  };

  const runCompare = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    if (Object.keys(errors).length > 0) {
      focusFirstError(errors);
      return;
    }
    const ctl = new AbortController();
    abortRef.current = ctl;
    setRunning(true);
    setRunError(null);
    announceLater("Running base, edit, and control edit side by side.");
    try {
      const payload = buildComparePayload(form, maxLayer, maxNeuron);
      const r = await fetch("/api/level/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: ctl.signal,
        body: JSON.stringify(payload),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body?.detail || `HTTP ${r.status}`);
      const res = body as CompareReply;
      setReply(res);
      announceLater(`Reading finished in ${res.seconds} seconds; model is clean at base.`);
      setBookId("live");
      setBookOffline(false);
      void loadStatus();
      void loadBook("live");
    } catch (exc) {
      if ((exc as Error).name !== "AbortError") {
        const msg = (exc as Error).message || String(exc);
        setRunError(msg);
        announceLater(msg);
      }
    } finally {
      setRunning(false);
      if (abortRef.current === ctl) abortRef.current = null;
    }
  };

  const inspectEntry = async (seq: number) => {
    if (bookOffline) {
      const hit = RECORDED_REPLAY_CARDS.find((c) => c.seq === seq) ?? null;
      setSelectedCard(hit);
      return;
    }
    try {
      const r = await fetch(`/api/level/book/${seq}?book=${encodeURIComponent(bookId)}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = (await r.json()) as { card: ReplayCard };
      setSelectedCard(data.card);
      announceLater(`Replaying book line ${seq}.`);
    } catch (exc) {
      setRunError((exc as Error).message || String(exc));
    }
  };

  const isRowOp = form.op === "silence_rows" || form.op === "scale_rows";
  const isScaleOp = form.op === "scale" || form.op === "scale_rows";
  const vaultDirty = activeEdits.length > 0 || (status?.active_vault_edits ?? 0) > 0;
  const availableBooks: BookInfo[] = status?.books ?? [
    { id: "recorded", label: "recorded 31B claim check (19 Nov 2026)", recorded: true, exists: true, entries: RECORDED_REPLAY_CARDS.length, ok: true, head: "6cf3621dbee8c2a3" },
  ];

  return (
    <section
      role="tabpanel"
      id="panel-compare"
      aria-labelledby="tab-compare"
      style={{ paddingTop: 34, display: "flex", flexDirection: "column", gap: 40 }}
    >
      {/* Single polite live region for accessible error and status announcements */}
      <div className="visually-hidden" aria-live="polite" role="status">
        {announcement}
      </div>

      {/* Section ii: side-by-side audition */}
      <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
        <h2 className="h-sec">ii · hear base, edit and a control edit side by side</h2>
        <Rubric style={{ maxWidth: 820 }}>
          Unlike the vault’s edit tab, nothing here stays on the model. One forward runs the untouched base, your edit, and same-size control
          edits in separate rows of the same batch, enters the reading in the level book, and returns the model untouched.
        </Rubric>

        {statusErr ? (
          <p style={{ margin: 0, fontSize: 16, fontStyle: "italic", color: "var(--muted)" }}>{statusErr}</p>
        ) : null}

        {vaultDirty ? (
          <div style={{ border: "1px solid var(--rubric)", padding: "10px 14px", fontSize: 16, color: "var(--rubric-deep)" }}>
            The vault currently has active edits in force ({Math.max(activeEdits.length, status?.active_vault_edits ?? 0)}). Undo them on the
            edit tab (<code>ROLLBACK</code>) before running a side-by-side reading: a reading never runs on an already-edited model.
          </div>
        ) : null}

        {/* Preset chips */}
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "8px 12px" }}>
          <span style={smallCaps(14)}>presets</span>
          {COMPARE_PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              title={p.note}
              onClick={() => {
                setForm(p.patch);
                setEstimate(null);
              }}
              style={chipStyle(false, { padding: "0 12px", fontSize: 16 })}
            >
              {p.label}
            </button>
          ))}
        </div>

        <form onSubmit={runCompare} noValidate style={{ display: "flex", flexDirection: "column", gap: 20, maxWidth: 920 }}>
          <fieldset style={fieldsetStyle}>
            <legend style={{ float: "left", padding: 0, marginRight: 10, ...smallCaps() }}>what to try</legend>
            {(
              [
                ["silence", "silence down_proj columns"],
                ["scale", "scale down_proj columns"],
                ["silence_rows", "silence gate/up rows"],
                ["scale_rows", "scale gate/up rows"],
              ] as [CompareOp, string][]
            ).map(([k, lbl]) => (
              <label key={k} style={radioLabel}>
                <input
                  type="radio"
                  name="cmp-op"
                  checked={form.op === k}
                  onChange={() =>
                    setForm({
                      op: k,
                      linear: k === "silence_rows" || k === "scale_rows" ? (form.linear === "up_proj" ? "up_proj" : "gate_proj") : "down_proj",
                    })
                  }
                  style={radioStyle}
                />
                {lbl}
              </label>
            ))}
          </fieldset>

          {isRowOp ? (
            <fieldset style={fieldsetStyle}>
              <legend style={{ float: "left", padding: 0, marginRight: 10, ...smallCaps() }}>input projection</legend>
              {(["gate_proj", "up_proj"] as CompareLinear[]).map((lin) => (
                <label key={lin} style={radioLabel}>
                  <input
                    type="radio"
                    name="cmp-lin"
                    checked={form.linear === lin}
                    onChange={() => setForm({ linear: lin })}
                    style={radioStyle}
                  />
                  <span style={{ fontFamily: MONO, fontSize: 16 }}>{lin}</span>
                </label>
              ))}
            </fieldset>
          ) : null}

          {/* Sentence-form edit specification */}
          <div style={{ fontSize: 25, lineHeight: 1.85 }}>
            <label htmlFor="cmp-neurons">Neurons</label>{" "}
            <input
              ref={neuronsRef}
              id="cmp-neurons"
              type="text"
              value={form.neurons}
              onChange={(e) => setForm({ neurons: e.target.value })}
              onBlur={() => markBlur("neurons")}
              aria-invalid={showError("neurons") ? "true" : "false"}
              aria-describedby={showError("neurons") ? "cmp-neurons-err" : undefined}
              autoComplete="off"
              spellCheck={false}
              style={sentenceInput(22, "18em", { numeric: true })}
            />{" "}
            <label htmlFor="cmp-layer">in layer</label>{" "}
            <input
              ref={layerRef}
              id="cmp-layer"
              type="number"
              min={1}
              max={maxLayer}
              value={form.layer}
              onChange={(e) => setForm({ layer: e.target.value })}
              onBlur={() => markBlur("layer")}
              aria-invalid={showError("layer") ? "true" : "false"}
              aria-describedby={showError("layer") ? "cmp-layer-err" : undefined}
              style={sentenceInput(25, "2.8em", { numeric: true })}
            />
            {isScaleOp ? (
              <>
                {" "}
                <label htmlFor="cmp-factor">scaled by ×</label>{" "}
                <input
                  ref={factorRef}
                  id="cmp-factor"
                  type="text"
                  value={form.factor}
                  onChange={(e) => setForm({ factor: e.target.value })}
                  onBlur={() => markBlur("factor")}
                  aria-invalid={showError("factor") ? "true" : "false"}
                  aria-describedby={showError("factor") ? "cmp-factor-err" : undefined}
                  style={sentenceInput(25, "3.2em", { numeric: true })}
                />
              </>
            ) : null}
            {" "}
            <label htmlFor="cmp-dose">at dose</label>{" "}
            <input
              ref={doseRef}
              id="cmp-dose"
              type="text"
              value={form.dose}
              onChange={(e) => setForm({ dose: e.target.value })}
              onBlur={() => markBlur("dose")}
              aria-invalid={showError("dose") ? "true" : "false"}
              aria-describedby={showError("dose") ? "cmp-dose-err" : undefined}
              style={sentenceInput(25, "2.8em", { numeric: true })}
            />
          </div>

          {showError("neurons") ? <p id="cmp-neurons-err" className="lvl-field-err">{errors.neurons}</p> : null}
          {showError("layer") ? <p id="cmp-layer-err" className="lvl-field-err">{errors.layer}</p> : null}
          {showError("factor") ? <p id="cmp-factor-err" className="lvl-field-err">{errors.factor}</p> : null}
          {showError("dose") ? <p id="cmp-dose-err" className="lvl-field-err">{errors.dose}</p> : null}

          {/* Control family & generation settings */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: "12px 28px", alignItems: "center", fontSize: 17 }}>
            <label style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              <span style={smallCaps(14)}>control edits</span>
              <select
                aria-label="Control edit family"
                value={form.controlsFamily}
                onChange={(e) => setForm({ controlsFamily: e.target.value as ControlsFamily })}
                style={{
                  background: "transparent",
                  border: "1px solid var(--control)",
                  borderRadius: 2,
                  padding: "6px 10px",
                  minHeight: 40,
                  fontSize: 16,
                }}
              >
                <option value="other_sites_matched">matched other sites</option>
                <option value="other_sites">random other sites</option>
                <option value="same_site">same site (random direction)</option>
              </select>
            </label>

            <label style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              <span style={smallCaps(14)}>count</span>
              <input
                ref={ctlCountRef}
                type="number"
                min={1}
                max={19}
                value={form.controlsCount}
                onChange={(e) => setForm({ controlsCount: e.target.value })}
                onBlur={() => markBlur("controlsCount")}
                aria-invalid={showError("controlsCount") ? "true" : "false"}
                aria-describedby={showError("controlsCount") ? "cmp-ctl-err" : undefined}
                style={sentenceInput(18, "3em", { numeric: true })}
              />
            </label>

            <label style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              <span style={smallCaps(14)}>pieces</span>
              <input
                ref={maxNewRef}
                type="number"
                min={4}
                max={256}
                value={form.maxNew}
                onChange={(e) => setForm({ maxNew: e.target.value })}
                onBlur={() => markBlur("maxNew")}
                aria-invalid={showError("maxNew") ? "true" : "false"}
                aria-describedby={showError("maxNew") ? "cmp-maxnew-err" : undefined}
                style={sentenceInput(18, "3.4em", { numeric: true })}
              />
            </label>

            <label style={radioLabel}>
              <input
                type="checkbox"
                checked={form.thinking}
                onChange={(e) => setForm({ thinking: e.target.checked })}
                style={radioStyle}
              />
              thinking on
            </label>

            <label style={radioLabel}>
              <input
                type="checkbox"
                checked={form.sampled}
                onChange={(e) => setForm({ sampled: e.target.checked })}
                style={radioStyle}
              />
              sampled
            </label>
          </div>
          {showError("controlsCount") ? <p id="cmp-ctl-err" className="lvl-field-err">{errors.controlsCount}</p> : null}
          {showError("maxNew") ? <p id="cmp-maxnew-err" className="lvl-field-err">{errors.maxNew}</p> : null}

          {/* Question & optional paired answer / contrast */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <label htmlFor="cmp-question" style={smallCaps(14)}>
              question to hear
            </label>
            <textarea
              ref={questionRef}
              id="cmp-question"
              rows={2}
              value={form.question}
              onChange={(e) => setForm({ question: e.target.value })}
              onBlur={() => markBlur("question")}
              aria-invalid={showError("question") ? "true" : "false"}
              aria-describedby={showError("question") ? "cmp-q-err" : undefined}
              style={{
                width: "100%",
                padding: "8px 10px",
                background: "transparent",
                border: "1px solid var(--control)",
                borderRadius: 2,
                fontSize: 18,
                lineHeight: 1.5,
                resize: "vertical",
              }}
            />
            {showError("question") ? <p id="cmp-q-err" className="lvl-field-err">{errors.question}</p> : null}

            <div style={{ display: "flex", flexWrap: "wrap", gap: "10px 24px", alignItems: "center" }}>
              <label style={radioLabel}>
                <input
                  type="checkbox"
                  checked={form.includePaired}
                  onChange={(e) => setForm({ includePaired: e.target.checked })}
                  style={radioStyle}
                />
                also measure paired next-piece table (two batch classes)
              </label>
            </div>

            {form.includePaired ? (
              <div style={{ display: "flex", flexWrap: "wrap", gap: "12px 20px", alignItems: "center" }}>
                <label style={{ flex: "1 1 320px", display: "flex", flexDirection: "column", gap: 4 }}>
                  <span style={smallCaps(13)}>answer prefix to score</span>
                  <input
                    type="text"
                    value={form.answer}
                    onChange={(e) => setForm({ answer: e.target.value })}
                    style={{
                      padding: "6px 10px",
                      background: "transparent",
                      border: "1px solid var(--control)",
                      borderRadius: 2,
                      fontSize: 17,
                    }}
                  />
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <span style={smallCaps(13)}>odds toward (optional)</span>
                  <input
                    type="text"
                    value={form.towardWord}
                    onChange={(e) => setForm({ towardWord: e.target.value })}
                    onBlur={() => markBlur("odds")}
                    aria-invalid={showError("odds") ? "true" : "false"}
                    aria-describedby={showError("odds") ? "cmp-odds-err" : undefined}
                    style={{
                      width: "8em",
                      padding: "6px 10px",
                      background: "transparent",
                      border: "1px solid var(--control)",
                      borderRadius: 2,
                      fontSize: 17,
                    }}
                  />
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <span style={smallCaps(13)}>odds away (optional)</span>
                  <input
                    type="text"
                    value={form.awayWord}
                    onChange={(e) => setForm({ awayWord: e.target.value })}
                    onBlur={() => markBlur("odds")}
                    aria-invalid={showError("odds") ? "true" : "false"}
                    aria-describedby={showError("odds") ? "cmp-odds-err" : undefined}
                    style={{
                      width: "8em",
                      padding: "6px 10px",
                      background: "transparent",
                      border: "1px solid var(--control)",
                      borderRadius: 2,
                      fontSize: 17,
                    }}
                  />
                </label>
              </div>
            ) : null}
            {showError("odds") ? <p id="cmp-odds-err" className="lvl-field-err">{errors.odds}</p> : null}
          </div>

          <p style={{ margin: 0, fontSize: 17, lineHeight: 1.55, color: "var(--muted)", borderTop: "1px solid var(--hair-soft)", paddingTop: 12 }}>
            {formatPlainEdit(form)}
          </p>

          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 14 }}>
            <button
              type="submit"
              disabled={running || vaultDirty || !status?.runtime_ready}
              className="btn-primary"
              style={{ minHeight: 44, padding: "0 20px", fontSize: 18, cursor: running ? "wait" : "pointer" }}
            >
              {running ? "hearing three voices…" : "hear side by side"}
            </button>
            {running ? (
              <button
                type="button"
                onClick={() => abortRef.current?.abort()}
                style={chipStyle(false, { padding: "0 14px", fontSize: 16 })}
              >
                stop
              </button>
            ) : (
              <button
                type="button"
                disabled={estimating || !status?.runtime_ready}
                onClick={() => void runEstimate()}
                style={chipStyle(false, { padding: "0 14px", fontSize: 16 })}
              >
                {estimating ? "pricing…" : "estimate cost"}
              </button>
            )}
            {estimate ? (
              <span className="num" style={{ fontSize: 15, color: "var(--muted)" }}>
                estimated: {estimate.total_forwards} forwards · ~{estimate.total_seconds.toFixed(1)} s · ~
                {estimate.peak_gib_above_model.toFixed(2)} GiB above model
              </span>
            ) : null}
          </div>
          {runError ? (
            <p className="lvl-field-err" role="alert">
              {runError}
            </p>
          ) : null}
        </form>

        {/* Live audition output */}
        {reply ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 22, paddingTop: 16, borderTop: "1px solid var(--gilt-rule)" }}>
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "8px 16px" }}>
              <span style={smallCaps(16)}>this reading</span>
              <StatusPill text={reply.status} />
              <span className="num" style={{ fontSize: 15, color: "var(--muted)" }}>
                edit <code style={{ fontFamily: MONO, fontSize: 14 }}>{reply.edit.short}</code> · layer {reply.edit.page_layer} (index{" "}
                {reply.edit.layer}) · {reply.seconds.toFixed(2)} s · book line{" "}
                {reply.book.written_seq ?? "—"}
                {reply.book.paired_seq != null ? ` & ${reply.book.paired_seq}` : ""} · base asserted clean
              </span>
            </div>
            {reply.voices ? <ThreeVoices voices={reply.voices} /> : null}
            {reply.paired ? <PairedMetricsTable paired={reply.paired} captionId={liveMetricsCapId} /> : null}
            <p style={{ margin: 0, fontSize: 15, fontStyle: "italic", color: "var(--muted)" }}>{reply.note}</p>
            <Apparatus title="ledger record and raw lines" label="Apparatus for this side-by-side reading">
              <pre style={{ margin: 0, fontFamily: MONO, fontSize: 13, lineHeight: 1.5, whiteSpace: "pre-wrap" }}>
                {[...reply.written_lines, "", ...reply.paired_lines].join("\n").trim()}
              </pre>
            </Apparatus>
          </div>
        ) : null}
      </div>

      {/* Section iii: Model-free ledger replay */}
      <div style={{ display: "flex", flexDirection: "column", gap: 18, paddingTop: 22, borderTop: "1px solid var(--hair)" }}>
        <h2 className="h-sec">iii · replay from the level book</h2>
        <Rubric style={{ maxWidth: 820 }}>
          Every paired reading, written reading, calibration, and Phase 3 claim verdict is hash-chained into the level book. Replaying any
          line below reads only the ledger and never touches the GPU.
        </Rubric>

        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "8px 12px" }}>
          <span style={smallCaps(14)}>book</span>
          {availableBooks.map((b) => (
            <button
              key={b.id}
              type="button"
              onClick={() => {
                setBookOffline(false);
                setBookId(b.id);
              }}
              style={chipStyle(bookId === b.id || (bookOffline && b.id === "recorded"), { padding: "0 12px", fontSize: 15 })}
            >
              {b.label} ({b.entries})
            </button>
          ))}
        </div>

        {bookVerify ? (
          <p className="num" style={{ margin: 0, fontSize: 15, color: bookVerify.ok ? "var(--muted)" : "var(--rubric)" }}>
            {bookVerify.ok
              ? `the book is whole: ${bookVerify.entries} entries, head ${bookVerify.head.slice(0, 12)}`
              : `chain broken: ${bookVerify.why}`}
          </p>
        ) : bookOffline ? (
          <p style={{ margin: 0, fontSize: 15, fontStyle: "italic", color: "var(--gilt-text)" }}>
            Showing recorded 31B card Phase 3 verdicts (19 November 2026) offline—no server required.
          </p>
        ) : null}

        {bookLoading ? (
          <p style={{ margin: 0, fontSize: 16, fontStyle: "italic", color: "var(--muted)" }}>Reading the level book…</p>
        ) : bookEntries.length === 0 ? (
          <EmptyAnswer head="[no readings in this book yet]">
            Run a side-by-side reading above, or choose one of the recorded 31B verification books to inspect its entries.
          </EmptyAnswer>
        ) : (
          <div tabIndex={0} role="region" aria-labelledby={replayTableCapId} className="lvl-table-wrap">
            <table
              className="lvl-table"
              style={
                {
                  "--label-1": '"Line"',
                  "--label-2": '"Kind"',
                  "--label-3": '"Reading / Brief"',
                  "--label-4": '"Outcome"',
                  "--label-5": '"Edit"',
                  "--label-6": '"Action"',
                } as React.CSSProperties
              }
            >
              <caption id={replayTableCapId} style={{ textAlign: "left", padding: "0 0 8px", ...smallCaps(14) }}>
                recent level-book entries (newest first)
              </caption>
              <thead>
                <tr>
                  <th scope="col">Line</th>
                  <th scope="col">Kind</th>
                  <th scope="col">Reading / Brief</th>
                  <th scope="col">Outcome</th>
                  <th scope="col">Edit</th>
                  <th scope="col">Action</th>
                </tr>
              </thead>
              <tbody>
                {bookEntries.map((ent) => {
                  const active = selectedCard?.seq === ent.seq;
                  return (
                    <tr key={ent.seq} style={{ background: active ? "var(--pressed)" : undefined }}>
                      <th scope="row" className="num">
                        #{ent.seq}
                      </th>
                      <td style={{ fontFamily: MONO, fontSize: 13 }}>{ent.note_type || ent.kind}</td>
                      <td>{ent.name || "—"}</td>
                      <td>{ent.status ? <StatusPill text={ent.status} /> : "—"}</td>
                      <td style={{ fontFamily: MONO, fontSize: 13 }}>{ent.edit_id ? ent.edit_id.slice(0, 12) : "—"}</td>
                      <td>
                        <button
                          type="button"
                          onClick={() => void inspectEntry(ent.seq)}
                          style={chipStyle(active, { minHeight: 32, minWidth: 72, padding: "0 10px", fontSize: 14 })}
                        >
                          {active ? "showing" : "replay"}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {selectedCard ? <ReplayInspector card={selectedCard} /> : null}
      </div>
    </section>
  );
};
