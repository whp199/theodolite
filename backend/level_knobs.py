"""Levelling Phase 6: cheaper-knob readings and fresh knob claims (`backend/level_knobs.py`).

Reads non-weight interventions (a prompt, decoding settings: the sampler, the thinking toggle, a logit bias, and
transient steering pushes) on the untouched base model, under the same brief, kept-back frames, retention batteries,
calibrated bar set and verdict engine as a weight edit's claim (`backend/level_claim.py`). Zero weights are modified;
`weight_edits.assert_base` is asserted before and after every call.

What a fresh knob claim shares with a weight claim, and how:
- One budget of fresh readings a brief. A knob claim is asked `level_claim.fresh_reading_gate` (`what="knob"`) before
  it is registered, the gate a weight claim and a trial are asked: practice on kept-back wording, a brief changed
  under its name, frames seen in practice or read under another brief, bars not calibrated in the book, a knob read
  before, and the brief's `m` spent. Its registration is a note `level_claim.brief_history` counts, so weight
  claims, trials and knob claims spend the same `m` in one book, and the `m` in p_fw = m x p_emp is that one budget.
- The same draw: kept-back items drawn after registration from `os.urandom(32)`, refusing items and frames seen in
  practice.
- The same verdict engine: `level_claim.judge_claim`, given the knob's own families of controls, with both batch
  classes read for the target and the batteries (the larger battery figure kept), the reversed read (the knob with
  its steering reversed and its logit bias negated), the near-miss size and its ratio to the effect, and the gating
  and habitat batteries the bars name. Four choices of this gate are stricter than a weight claim's and are kept
  (`STRICTER_SAY`): the noise floor is the larger class's for both classes; a near miss's size is the mean of each
  item's size (`knob_batteries`); and, each where the engine's order puts its kind (`_stricter`), a frame CI that
  does not clear zero is "did not replicate" and a labelled-panel KL that was not read counts as harm.
- Its controls differ from the knob only in the steering: the knob's prompt, thinking setting, sampler and logit
  bias apply to every control row as to the knob's. `neuron_mul` is the arithmetic of a weight edit that scales its
  neurons' down_proj columns, and its controls are drawn as that edit's are (`weight_edits.matched_control_edits`,
  `weight_edits.same_site_controls`) and read by the weight path's own per-row hooks. `direction` and `neuron_add`
  are pushes added to the stream; their controls are pushes of the same length at the same layer along random
  directions (`CONTROL_FAMILY_SAY`).

Refused before a fresh reading is spent: a knob with no steering part (a prompt or a decoding setting has no family
of controls, and a weight claim with none is refused), a knob that mixes `neuron_mul` with pushes, a `neuron_mul`
that spares the start mark (no weight edit does that arithmetic), fewer than one control a family, and a part that
cannot act on the brief's reading (`parts_that_cannot_act`).

Every knob reading is read against the brief's chat settings: its system turn (or an item's own) and its thinking
setting. The knob's prompt is added to that system turn and its thinking setting, when it has one, replaces the
brief's; nothing else differs between the knob's row and the base row.

Layers count from 0 in code and files; human-facing strings give both ("layer 50 (page layer 51)").
Zero kept-out words appear in source, docstrings, or rendered sheets.
"""
from __future__ import annotations

import contextlib
import dataclasses
import datetime
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from backend import atlas_ops as ao
from backend import level_book as lb
from backend import level_briefs as lbr
from backend import level_claim as lcl
from backend import levelling as lv
from backend import sampler_request as sr
from backend import weight_edits as we
from backend.causal_check import logits_fp32
from backend.engine_runtime import SamplerSettings
from backend.theodolite_math import FINAL_LOGIT_SOFTCAP

KNOB_VERSION = 1  # the knob specification's version (it is hashed into a knob's id)
KNOB_CLAIM_VERSION = 2  # 2 (20 November 2026): one budget with weight claims, judged by level_claim.judge_claim
MAX_PROMPT_CHARS = 4000
MAX_LOGIT_BIAS_TOKENS = 16
MAX_LOGIT_BIAS = 10.0
MAX_STEER_TERMS = 16
MAX_STEER_DOSE = 8.0
MAX_STEER_SCALE = 1000.0
MAX_MUL_NEURONS = 1024
MAX_ADD_NEURONS = 64
MAX_MUL_FACTOR = 64.0
DEFAULT_KNOB_CONTROLS = 19
NULL_ES = (-0.125, -0.03125, 0.03125, 0.125)  # the null rows: the layer's MLP output times 1 + e

KNOB_VERDICTS = (
    "lead",
    "no better than control edits",
    "works but harms",
    "did not replicate",
    "kept the base",
    "void",
)

# What each kind of steering term does, in the words a sheet and a driver read (`steering_words`).
STEERING_SAY = {
    "direction": ("a fixed push added to the residual stream at the output of layer {L} (page layer {P}): the given "
                  "vector, projected off the layer's quiet mask, at length |dose x scale| = {mag:g}, the same at every "
                  "position{bos}"),
    "neuron_mul": ("the activations of {k} neuron(s) of layer {L} (page layer {P}) multiplied by {f:g} where they enter "
                   "the down projection, at every position{bos}: the arithmetic of a weight edit that scales those "
                   "neurons' down_proj columns by {f:g}"),
    "neuron_add": ("a fixed push added to the residual stream at the output of layer {L} (page layer {P}), along the "
                   "summed write direction of {k} neuron(s) (their down_proj columns times the post-feedforward norm's "
                   "gain, summed and projected off the quiet mask), at length |dose x scale| = {mag:g} and the same at "
                   "every position whatever the neurons do{bos}. It is not an add to the neurons' activations"),
}
# Each family of controls a knob claim is judged against: what its controls are, and what they are and are not
# matched on. Printed on every knob sheet; the rule they are counted by is the weight claim's (`CONTROL_RULE_SAY`).
CONTROL_FAMILY_SAY = {
    "other_sites_matched": (
        "other neurons of the same layers, drawn by the weight path's own matched draw "
        "(weight_edits.matched_control_edits) for the weight edit whose arithmetic the knob is (its neurons' down_proj "
        "columns scaled by its factor), and read as that edit's controls are read: each stand-in is within a factor of "
        "{tol:g} of its neuron on what a base pre-pass over the items read, as the knob reads them, expects it to "
        "deliver over the answer. Not matched on what they delivered as read, over the prompt, or at the last "
        "position"),
    "same_site": (
        "the knob's own neurons, their down_proj columns changed along random directions off the layer's quiet mask, "
        "each column's change as long in the stream as the knob's (weight_edits.same_site_controls), read as that "
        "edit's controls are read. Matched on the size of each column's change, not on its direction"),
    "random_directions": (
        "pushes at the same layers, of the same length (|dose x scale|) and the same start-mark rule, along random "
        "directions off each layer's quiet mask. Matched on the size of the push at the layer where it is added; not "
        "on what it does further up, nor on how hard any neuron fires: a random direction is no neuron's write "
        "direction, and a knob whose direction was chosen from readings is set beside directions that were not"),
}
CONTROL_RULE_SAY = (
    "every family of controls is counted by the weight claim's rule as it stands, which the owner has left open: "
    "p_fw = m x (1 + at_least) / (C + 1), at most alpha, in both batch classes (a control that gave no finite reading "
    "counts as moving as much)")
CONTROLS_SHARE_SAY = ("the controls share the knob's prompt, thinking setting, sampler and logit bias, and differ from "
                      "it only in the steering")
REVERSED_SAY = ("the reversed read is the knob with its steering reversed (a push's dose negated, a neuron_mul's factor "
                "f read at 2 - f, as a weight edit at dose -1) and its logit bias negated; its prompt, thinking setting "
                "and sampler are as in the knob")
# What a weight claim reads and a knob claim does not: listed on every knob sheet.
NOT_READ_SAY = (
    "the brief's author prompt arms are not read beside a knob, so this sheet cannot say whether one of them does as "
    "well",
    "what each control changed in the stream as read (a weight claim prints it, and gates nothing on it)",
    "the kernel's item gate (how many fresh items' forced answers sit under its bar)",
)
NOT_READ_WRITTEN_SAY = ("the written strata and the base settings audit a written weight claim prints (neither gates "
                        "its verdict)")
NOT_READ_SAMPLER_SAY = ("the harm of the knob's sampler: the retention batteries are read teacher-forced, where a "
                        "sampler does nothing")
STRICTER_SAY = (
    "where this gate is stricter than a weight claim's: the noise floor is the larger of the two classes' for both; "
    "a near miss's size is the mean of each item's size, never less than the size of the mean a weight claim reads; "
    "a frame CI (large class) that does not clear zero reads \"did not replicate\"; and a labelled-panel KL that was "
    "not read counts as harm")
NO_CONTROLS_SAY = (
    "A fresh knob claim needs a steering part: a prompt or a decoding setting alone has no family of controls to be "
    "read against, and a claim with none is refused (a weight edit with no control family is refused the same way). "
    "Practice readings of such a knob are open (knob_read, knob_write).")
PROMPT_ARMS_SAY = "not compared with a knob"
PRACTICE_CONTROLS_SAY = (
    "Practice steering controls: for neuron_mul, other neurons of the same layer drawn at random, NOT matched on what "
    "they deliver; for direction and neuron_add, random directions of the same length at the same layer. They share "
    "the knob's prompt, thinking setting and logit bias. A fresh claim draws neuron_mul's controls as a weight edit's "
    "are drawn.")


class KnobRefused(ValueError):
    """A malformed, over-cap, leaking, or budget-refused cheaper-knob specification or reading."""


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean_float(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _resolve_npy(path_str: str, base_dir: Optional[Path], expected_sha: Optional[str]) -> Tuple[np.ndarray, str]:
    p = Path(str(path_str))
    if p.is_absolute() or ".." in p.parts:
        raise KnobRefused("Steering vector path must be a relative path inside the session's work folder.")
    if base_dir is None:
        raise KnobRefused("Steering vector file path requires a base_dir.")
    root = Path(base_dir).resolve()
    target = (root / p).resolve()
    if not str(target).startswith(str(root) + os.sep):
        raise KnobRefused("Steering vector path escapes base_dir.")
    if not target.is_file():
        raise KnobRefused(f"Steering vector file {path_str!r} does not exist.")
    raw = target.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None and str(expected_sha).lower() != actual_sha:
        raise KnobRefused(
            f"Steering vector {path_str!r} sha256 mismatch (expected {expected_sha}, got {actual_sha})."
        )
    try:
        arr = np.load(target, allow_pickle=False)
    except Exception as exc:
        raise KnobRefused(f"Could not read steering vector {path_str!r}: {exc}") from None
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    return arr, actual_sha


@dataclass(frozen=True)
class ResolvedSteerTerm:
    """One validated steering push ready for per-row hook installation."""

    kind: str  # "direction" | "neuron_mul" | "neuron_add"
    layer: int
    spare_bos: bool
    # For "direction" and "neuron_add": unit float32 vector [D] projected off quiet_mask, plus signed magnitude dose*scale
    unit_vec: Optional[np.ndarray] = None
    magnitude: float = 0.0
    quiet_share: float = 0.0
    # For "neuron_mul": neuron indices and scalar factor
    neurons: Tuple[int, ...] = ()
    factor: float = 1.0
    # Canonical serializable metadata
    canonical: Mapping[str, Any] = None  # type: ignore[assignment]


@dataclass(frozen=True)
class KnobSpec:
    """Canonical, validated cheaper-knob specification pinned to a base model."""

    prompt: Optional[str]
    decoding: Dict[str, Any]
    steering_raw: Tuple[Dict[str, Any], ...]
    steering_resolved: Tuple[ResolvedSteerTerm, ...]
    knob_classes: Tuple[str, ...]
    model_id: str
    weights_id: str
    id: str

    def short(self) -> str:
        return self.id[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "v": KNOB_VERSION,
            "id": self.id,
            "model_id": self.model_id,
            "weights_id": self.weights_id,
            "knob_classes": list(self.knob_classes),
            "prompt": self.prompt,
            "decoding": dict(self.decoding),
            "steering": [dict(t) for t in self.steering_raw],
        }


def _validate_decoding(raw: Any, vocab_size: int) -> Dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise KnobRefused("decoding must be a JSON object.")
    allowed = {"thinking", "sampler", "logit_bias"}
    extra = sorted(k for k in raw if k not in allowed)
    if extra:
        for bad in extra:
            if bad in sr.NOT_OFFERED:
                raise KnobRefused(sr.NOT_OFFERED[bad])
        raise KnobRefused(f"Unknown decoding key(s): {', '.join(extra)} (allowed: {', '.join(sorted(allowed))}).")
    out: Dict[str, Any] = {}
    if "thinking" in raw and raw["thinking"] is not None:
        if not isinstance(raw["thinking"], bool):
            raise KnobRefused("decoding.thinking must be a boolean.")
        out["thinking"] = bool(raw["thinking"])
    if "sampler" in raw and raw["sampler"] is not None:
        sraw = raw["sampler"]
        if not isinstance(sraw, Mapping):
            raise KnobRefused("decoding.sampler must be a JSON object.")
        for bad in ("min_p", "repetition_penalty"):
            if bad in sraw:
                raise KnobRefused(sr.NOT_OFFERED[bad])
        s_allowed = {"temperature", "top_k", "top_p", "presence_penalty"}
        s_extra = sorted(k for k in sraw if k not in s_allowed)
        if s_extra:
            raise KnobRefused(f"Unknown sampler key(s): {', '.join(s_extra)}.")
        temp = _clean_float(sraw.get("temperature", 0.0))
        if temp is None or not 0.0 <= temp <= 2.0:
            raise KnobRefused("decoding.sampler.temperature must be finite and in [0, 2].")
        top_k = sraw.get("top_k", 0)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 0 <= top_k <= vocab_size:
            raise KnobRefused(f"decoding.sampler.top_k must be an integer in 0..{vocab_size}.")
        top_p = _clean_float(sraw.get("top_p", 1.0))
        if top_p is None or not (0.0 < top_p <= 1.0):
            raise KnobRefused("decoding.sampler.top_p must be finite and in (0, 1].")
        pres = _clean_float(sraw.get("presence_penalty", 0.0))
        if pres is None or not -2.0 <= pres <= 2.0:
            raise KnobRefused("decoding.sampler.presence_penalty must be finite and in [-2, 2].")
        if temp == 0.0 and pres != 0.0:
            raise KnobRefused("decoding.sampler.presence_penalty requires temperature > 0 (backend/sampler_request.py).")
        out["sampler"] = {
            "temperature": round(temp, 6),
            "top_k": int(top_k),
            "top_p": round(top_p, 6),
            "presence_penalty": round(pres, 6),
        }
    if "logit_bias" in raw and raw["logit_bias"] is not None:
        lb_raw = raw["logit_bias"]
        if not isinstance(lb_raw, Mapping):
            raise KnobRefused("decoding.logit_bias must be a mapping of token_id -> bias.")
        if len(lb_raw) > MAX_LOGIT_BIAS_TOKENS:
            raise KnobRefused(f"decoding.logit_bias caps at {MAX_LOGIT_BIAS_TOKENS} tokens; got {len(lb_raw)}.")
        clean_lb: Dict[str, float] = {}
        for k, v in lb_raw.items():
            try:
                if isinstance(k, bool):
                    raise ValueError("bool")
                tid = int(k)
            except (TypeError, ValueError):
                raise KnobRefused(f"decoding.logit_bias token key {k!r} is not an integer token id.") from None
            if not 0 <= tid < vocab_size:
                raise KnobRefused(f"decoding.logit_bias token id {tid} is outside 0..{vocab_size - 1}.")
            fv = _clean_float(v)
            if fv is None or abs(fv) > MAX_LOGIT_BIAS:
                raise KnobRefused(f"decoding.logit_bias[{tid}] must be finite and in [-{MAX_LOGIT_BIAS:g}, {MAX_LOGIT_BIAS:g}].")
            if fv != 0.0:
                clean_lb[str(tid)] = round(fv, 6)
        if clean_lb:
            out["logit_bias"] = clean_lb
    return out


def build_knob(
    h: lv.Handle,
    spec: Mapping[str, Any],
    *,
    base_dir: Optional[Path] = None,
    allowed_layers: Optional[Sequence[int]] = None,
) -> KnobSpec:
    """Validate and canonicalize a cheaper-knob spec against handle `h` (read-only; touches zero weights)."""
    if not isinstance(spec, Mapping):
        raise KnobRefused("Knob specification must be a JSON object.")
    allowed_top = {"prompt", "decoding", "steering", "note"}
    extra = sorted(k for k in spec if k not in allowed_top)
    if extra:
        raise KnobRefused(f"Unknown knob key(s): {', '.join(extra)} (allowed: prompt, decoding, steering, note).")

    prompt_val: Optional[str] = None
    if spec.get("prompt") is not None:
        if not isinstance(spec["prompt"], str):
            raise KnobRefused("knob.prompt must be a string.")
        ps = spec["prompt"].strip()
        if len(ps) > MAX_PROMPT_CHARS:
            raise KnobRefused(f"knob.prompt exceeds {MAX_PROMPT_CHARS} characters ({len(ps)}).")
        if ps:
            prompt_val = ps

    vocab_n = int(h.W_U.shape[0])
    decoding = _validate_decoding(spec.get("decoding"), vocab_n)

    hidden_d = ao.hidden_size(h.lm)
    ffn_w = ao.ffn_width(h.lm, 0)
    allowed_set = set(int(x) for x in allowed_layers) if allowed_layers is not None else None

    steer_raw_in = spec.get("steering")
    steer_canonical: List[Dict[str, Any]] = []
    steer_resolved: List[ResolvedSteerTerm] = []
    if steer_raw_in is not None:
        if not isinstance(steer_raw_in, Sequence) or isinstance(steer_raw_in, (str, bytes)):
            raise KnobRefused("knob.steering must be a list of steering term objects.")
        if len(steer_raw_in) > MAX_STEER_TERMS:
            raise KnobRefused(f"knob.steering caps at {MAX_STEER_TERMS} terms; got {len(steer_raw_in)}.")
        for idx, raw_t in enumerate(steer_raw_in):
            if not isinstance(raw_t, Mapping):
                raise KnobRefused(f"steering[{idx}] must be a JSON object.")
            kind = str(raw_t.get("kind") or "")
            if kind not in ("direction", "neuron_mul", "neuron_add"):
                raise KnobRefused(
                    f'steering[{idx}].kind must be "direction", "neuron_mul", or "neuron_add"; got {kind!r}.'
                )
            layer_raw = raw_t.get("layer")
            if isinstance(layer_raw, bool) or not isinstance(layer_raw, int) or not 0 <= layer_raw < h.n_layers:
                raise KnobRefused(f"steering[{idx}].layer must be an integer in 0..{h.n_layers - 1}.")
            layer_i = int(layer_raw)
            if allowed_set is not None and layer_i not in allowed_set:
                raise KnobRefused(
                    f"steering[{idx}] targets layer {layer_i} (page layer {layer_i + 1}), outside brief allowed_layers."
                )
            spare_bos = bool(raw_t.get("spare_bos", False))

            if kind == "direction":
                dose = _clean_float(raw_t.get("dose", 1.0))
                scale = _clean_float(raw_t.get("scale", 1.0))
                if dose is None or abs(dose) > MAX_STEER_DOSE or dose == 0.0:
                    raise KnobRefused(f"steering[{idx}].dose must be finite, non-zero, and in [-{MAX_STEER_DOSE:g}, {MAX_STEER_DOSE:g}].")
                if scale is None or not (0.0 < scale <= MAX_STEER_SCALE):
                    raise KnobRefused(f"steering[{idx}].scale must be finite and in (0, {MAX_STEER_SCALE:g}].")
                if raw_t.get("path") is not None:
                    vec, sha = _resolve_npy(str(raw_t["path"]), base_dir, raw_t.get("sha256"))
                elif raw_t.get("vector") is not None:
                    v_in = raw_t["vector"]
                    if not isinstance(v_in, Sequence) or isinstance(v_in, (str, bytes)):
                        raise KnobRefused(f"steering[{idx}].vector must be a list of numbers.")
                    vec = np.asarray([float(x) for x in v_in], dtype=np.float32).reshape(-1)
                    sha = hashlib.sha256(vec.tobytes()).hexdigest()
                else:
                    raise KnobRefused(f'steering[{idx}] (kind="direction") requires either "path" (.npy) or "vector".')
                if vec.shape[0] != hidden_d:
                    raise KnobRefused(f"steering[{idx}] vector has length {vec.shape[0]}, expected hidden size {hidden_d}.")
                if not np.all(np.isfinite(vec)):
                    raise KnobRefused(f"steering[{idx}] vector contains non-finite values.")
                v64 = vec.astype(np.float64)
                n2 = float(np.dot(v64, v64))
                if n2 <= 1e-24:
                    raise KnobRefused(f"steering[{idx}] direction vector has zero norm.")
                qm = we.quiet_mask(h.lm, layer_i)
                q_share = float(np.dot(v64[qm], v64[qm]) / n2)
                if q_share > we.MAX_QUIET_SHARE:
                    raise KnobRefused(
                        f"steering[{idx}] direction has {q_share:.3f} energy in layer {layer_i}'s quiet mask "
                        f"(cap {we.MAX_QUIET_SHARE:g})."
                    )
                v64[qm] = 0.0
                p_norm = float(np.sqrt(np.dot(v64, v64)))
                if p_norm <= 1e-12:
                    raise KnobRefused(f"steering[{idx}] direction vanishes after quiet-mask projection.")
                unit = (v64 / p_norm).astype(np.float32)
                can = {
                    "kind": "direction",
                    "layer": layer_i,
                    "dose": round(dose, 6),
                    "scale": round(scale, 6),
                    "spare_bos": spare_bos,
                    "vec_sha256": sha,
                    "quiet_share": round(q_share, 6),
                }
                steer_canonical.append(can)
                steer_resolved.append(
                    ResolvedSteerTerm(
                        kind="direction",
                        layer=layer_i,
                        spare_bos=spare_bos,
                        unit_vec=unit,
                        magnitude=float(dose * scale),
                        quiet_share=q_share,
                        canonical=can,
                    )
                )

            elif kind == "neuron_mul":
                ns_in = raw_t.get("neurons")
                if not isinstance(ns_in, Sequence) or isinstance(ns_in, (str, bytes)) or not ns_in:
                    raise KnobRefused(f"steering[{idx}].neurons must be a non-empty list of neuron indices.")
                if len(ns_in) > MAX_MUL_NEURONS:
                    raise KnobRefused(f"steering[{idx}].neurons caps at {MAX_MUL_NEURONS}; got {len(ns_in)}.")
                ns: List[int] = []
                for x in ns_in:
                    if isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < ffn_w:
                        raise KnobRefused(f"steering[{idx}] neuron {x!r} outside 0..{ffn_w - 1}.")
                    ns.append(int(x))
                if len(set(ns)) != len(ns):
                    raise KnobRefused(f"steering[{idx}].neurons contains duplicate indices.")
                ns_sorted = tuple(sorted(ns))
                factor = _clean_float(raw_t.get("factor", 0.0))
                if factor is None or abs(factor) > MAX_MUL_FACTOR:
                    raise KnobRefused(f"steering[{idx}].factor must be finite and in [-{MAX_MUL_FACTOR:g}, {MAX_MUL_FACTOR:g}].")
                if factor == 1.0:
                    raise KnobRefused(f"steering[{idx}].factor is 1.0 (no-op).")
                can = {
                    "kind": "neuron_mul",
                    "layer": layer_i,
                    "neurons": list(ns_sorted),
                    "factor": round(factor, 6),
                    "spare_bos": spare_bos,
                }
                steer_canonical.append(can)
                steer_resolved.append(
                    ResolvedSteerTerm(
                        kind="neuron_mul",
                        layer=layer_i,
                        spare_bos=spare_bos,
                        neurons=ns_sorted,
                        factor=float(factor),
                        canonical=can,
                    )
                )

            elif kind == "neuron_add":
                ns_in = raw_t.get("neurons")
                if not isinstance(ns_in, Sequence) or isinstance(ns_in, (str, bytes)) or not ns_in:
                    raise KnobRefused(f"steering[{idx}].neurons must be a non-empty list of neuron indices.")
                if len(ns_in) > MAX_ADD_NEURONS:
                    raise KnobRefused(f"steering[{idx}].neurons caps at {MAX_ADD_NEURONS}; got {len(ns_in)}.")
                ns = []
                for x in ns_in:
                    if isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < ffn_w:
                        raise KnobRefused(f"steering[{idx}] neuron {x!r} outside 0..{ffn_w - 1}.")
                    ns.append(int(x))
                if len(set(ns)) != len(ns):
                    raise KnobRefused(f"steering[{idx}].neurons contains duplicate indices.")
                ns_sorted = tuple(sorted(ns))
                dose = _clean_float(raw_t.get("dose", 1.0))
                scale = _clean_float(raw_t.get("scale", 1.0))
                if dose is None or abs(dose) > MAX_STEER_DOSE or dose == 0.0:
                    raise KnobRefused(f"steering[{idx}].dose must be finite, non-zero, and in [-{MAX_STEER_DOSE:g}, {MAX_STEER_DOSE:g}].")
                if scale is None or not (0.0 < scale <= MAX_STEER_SCALE):
                    raise KnobRefused(f"steering[{idx}].scale must be finite and in (0, {MAX_STEER_SCALE:g}].")
                # Sum the stream-side write directions of the requested neurons and project off quiet_mask
                from backend.gguf_weights import dense_cols

                layer_m = h.lm.layers[layer_i]
                idx_t = torch.as_tensor(list(ns_sorted), dtype=torch.long)
                cols = dense_cols(layer_m.mlp.down_proj, idx_t).T.detach().to("cpu", dtype=torch.float64)  # [D, K]
                w_post = layer_m.post_feedforward_layernorm.weight.detach().to("cpu", dtype=torch.float64)  # [D]
                eff = (w_post[:, None] * cols).sum(dim=1).numpy()
                n2 = float(np.dot(eff, eff))
                if n2 <= 1e-24:
                    raise KnobRefused(f"steering[{idx}] selected neuron(s) have zero stream-side write norm.")
                qm = we.quiet_mask(h.lm, layer_i)
                q_share = float(np.dot(eff[qm], eff[qm]) / n2)
                eff[qm] = 0.0
                p_norm = float(np.sqrt(np.dot(eff, eff)))
                if p_norm <= 1e-12:
                    raise KnobRefused(f"steering[{idx}] selected neuron write vanishes after quiet-mask projection.")
                unit = (eff / p_norm).astype(np.float32)
                can = {
                    "kind": "neuron_add",
                    "layer": layer_i,
                    "neurons": list(ns_sorted),
                    "dose": round(dose, 6),
                    "scale": round(scale, 6),
                    "spare_bos": spare_bos,
                    "quiet_share": round(q_share, 6),
                }
                steer_canonical.append(can)
                steer_resolved.append(
                    ResolvedSteerTerm(
                        kind="neuron_add",
                        layer=layer_i,
                        spare_bos=spare_bos,
                        unit_vec=unit,
                        magnitude=float(dose * scale),
                        quiet_share=q_share,
                        neurons=ns_sorted,
                        canonical=can,
                    )
                )

    classes: List[str] = []
    if prompt_val is not None:
        classes.append("prompt")
    if decoding:
        classes.append("decoding")
    if steer_resolved:
        classes.append("steering")
    if not classes:
        raise KnobRefused("A cheaper knob must specify at least one of prompt, decoding, or steering.")

    payload = {
        "v": KNOB_VERSION,
        "model_id": h.model_id,
        "weights_id": h.weights_id,
        "prompt": prompt_val,
        "decoding": decoding,
        "steering": steer_canonical,
    }
    kid = hashlib.sha256(lb.canonical(payload)).hexdigest()
    return KnobSpec(
        prompt=prompt_val,
        decoding=decoding,
        steering_raw=tuple(steer_canonical),
        steering_resolved=tuple(steer_resolved),
        knob_classes=tuple(classes),
        model_id=h.model_id,
        weights_id=h.weights_id,
        id=kid,
    )



def steering_words(t: ResolvedSteerTerm) -> str:
    """What one steering term does, in plain words (`STEERING_SAY`)."""
    bos = " but the start mark" if t.spare_bos else ""
    return STEERING_SAY[t.kind].format(L=t.layer, P=t.layer + 1, mag=abs(float(t.magnitude)), k=len(t.neurons),
                                       f=float(t.factor), bos=bos)


# ------------------------------------------------------------------------------------- what a knob reads against


def _settings_of(brief: Optional[lbr.Brief]) -> Tuple[Optional[str], bool]:
    """(the system turn, thinking on) of the brief's chat settings, which every knob reading is read against."""
    cs = dict((brief.chat_settings if brief is not None else None) or {})
    return cs.get("system"), bool(cs.get("thinking", False))


def knob_system_turn(knob: KnobSpec, system: Optional[str]) -> Optional[str]:
    """The system turn an item is read with under `knob`: `system` (the item's own, else the brief's) with the knob's
    prompt on a new line after it, as `level_claim.prompt_arm_system` puts an author prompt arm."""
    if knob.prompt is None:
        return system
    return f"{system}\n{knob.prompt}" if system else knob.prompt


def _retokenised(h: lv.Handle, item: lv.Item, system: Optional[str], thinking: bool, what: str) -> lv.Item:
    if item.question is None or item.answer is None:
        raise KnobRefused(f'Item "{item.name}" has no question and answer text to read {what} on.')
    try:
        return lv._item(lv.chat_ids(h.tokenizer, item.question, system=system, thinking=thinking), item.answer_ids,
                        item.name, item.question, item.answer, system)
    except lv.Refused as exc:
        raise KnobRefused(f'Item "{item.name}" under {what}: {exc}') from None


def ref_and_knob_items(h: lv.Handle, item: lv.Item, knob: KnobSpec,
                       brief: Optional[lbr.Brief]) -> Tuple[lv.Item, lv.Item]:
    """(the base row's item, the knob's item) for a teacher-forced reading of `item`. Both are read under the brief's
    chat settings: the item's own system turn (the brief's when it has none) and the brief's thinking setting. The
    knob's item adds its prompt to that system turn and takes its thinking setting when it has one; nothing else
    differs. (An item is made with thinking off; under a brief that reads with thinking on, both rows are made again
    with it on, so a prompt knob is not read against a base with thinking off, nor a thinking-off knob against one.)"""
    b_sys, b_think = _settings_of(brief)
    ref_sys = item.system if item.system is not None else b_sys
    ref = item if (ref_sys == item.system and not b_think) else _retokenised(h, item, ref_sys, b_think,
                                                                             "the brief's chat settings")
    k_sys = knob_system_turn(knob, ref_sys)
    k_think = bool(knob.decoding.get("thinking", b_think))
    kit = ref if (k_sys == ref_sys and k_think == b_think) else _retokenised(h, item, k_sys, k_think, "the knob")
    return ref, kit


def knob_texts(knob: KnobSpec, item: Any, brief: Optional[lbr.Brief]) -> Tuple[Optional[str], Optional[str],
                                                                               Optional[str]]:
    """The texts the model reads of `item` under `knob`, in order (system turn, question, answer): what a kept-back
    guard (`level_briefs.KeptBack.hit`) is asked of, so a kept-back frame's wording is found in a knob's prompt as it
    is in an item."""
    b_sys, _think = _settings_of(brief)
    own = getattr(item, "system", None)
    return (knob_system_turn(knob, own if own is not None else b_sys), getattr(item, "question", None),
            getattr(item, "answer", None))


def brief_sampler(brief: lbr.Brief) -> Dict[str, Any]:
    """The sampler a written brief's base is read with: `level_claim`'s written claims' own (temperature 1, top-k
    64, top-p 0.95, no presence penalty, unless the brief's written_config says otherwise), so a knob's written
    reading has the base row a weight claim's has."""
    wcfg = dict(brief.written_config or {})
    return {"temperature": round(float(wcfg.get("temperature", 1.0)), 6), "top_k": int(wcfg.get("top_k", 64)),
            "top_p": round(float(wcfg.get("top_p", 0.95)), 6),
            "presence_penalty": round(float(wcfg.get("presence_penalty", 0.0)), 6)}


def parts_that_cannot_act(knob: KnobSpec, brief: Optional[lbr.Brief], *, written: bool) -> List[str]:
    """The parts of `knob` that cannot act on a reading (teacher-forced, or written with `written`) under `brief`,
    in plain words: a sampler where nothing is sampled, or the brief's own sampler; a thinking setting the brief
    already reads with. Such a part reads exactly as the knob without it: it would be registered and printed and
    never read, and a knob of such parts alone would spend a reading on nothing."""
    _sys, b_think = _settings_of(brief)
    out: List[str] = []
    dec = knob.decoding
    if "sampler" in dec:
        if not written:
            out.append("its sampler: a teacher-forced reading samples nothing")
        elif brief is not None and dict(dec["sampler"]) == brief_sampler(brief):
            out.append("its sampler: it is the brief's own")
    if "thinking" in dec and bool(dec["thinking"]) == b_think:
        out.append(f"its thinking setting: the brief already reads with thinking {'on' if b_think else 'off'}")
    return out


def _refuse_parts_that_cannot_act(knob: KnobSpec, brief: Optional[lbr.Brief], *, written: bool) -> None:
    inert = parts_that_cannot_act(knob, brief, written=written)
    if inert:
        raise KnobRefused(
            f"This knob has a part that cannot act on {'a written' if written else 'a teacher-forced'} reading, so it "
            f"would read exactly as the knob without it: {'; '.join(inert)}. Leave that part out."
        )


def _anchor_layer(h: lv.Handle, knob: KnobSpec) -> int:
    """Where the null rows sit: the lowest steered layer (as a weight reading puts them at its edit's lowest layer),
    else the last layer (a prompt or a decoding setting enters at the input or the output, not at layer 0)."""
    return min((t.layer for t in knob.steering_resolved), default=max(0, h.n_layers - 1))


def _mirror_of(knob: KnobSpec) -> Tuple[Tuple[ResolvedSteerTerm, ...], Optional[Dict[str, float]]]:
    """The reversed knob's steering and logit bias (`REVERSED_SAY`): a push's dose negated, a neuron_mul's factor f
    at 2 - f (a weight edit at dose -1), the bias negated. Refused when the reverse is past the knob's caps."""
    terms: List[ResolvedSteerTerm] = []
    for t in knob.steering_resolved:
        if t.kind == "neuron_mul":
            f = 2.0 - float(t.factor)
            if abs(f) > MAX_MUL_FACTOR:
                raise KnobRefused(
                    f"The reverse of a neuron_mul by {t.factor:g} is a factor of {f:g}, past the cap of "
                    f"{MAX_MUL_FACTOR:g}: a fresh claim reads every knob's reverse on a signed brief.")
            terms.append(dataclasses.replace(t, factor=f, canonical={**dict(t.canonical), "factor": round(f, 6),
                                                                      "reversed": True}))
        else:
            terms.append(dataclasses.replace(t, magnitude=-float(t.magnitude),
                                             canonical={**dict(t.canonical), "reversed": True}))
    bias = knob.decoding.get("logit_bias")
    return tuple(terms), ({str(k): -float(v) for k, v in bias.items()} if bias else None)


def _draw_steering_controls(
    h: lv.Handle,
    terms: Sequence[ResolvedSteerTerm],
    count: int,
    seed: int,
) -> List[Tuple[ResolvedSteerTerm, ...]]:
    """Build `count` random steering controls, a tuple of terms each:
    - For `direction` / `neuron_add`: random Gaussian directions in the same layer, projected off `quiet_mask`,
      unit-normalized, with the exact same magnitude and `spare_bos`: matched on the size of the push at its layer,
      nothing else. A fresh claim's `random_directions` family is these.
    - For `neuron_mul`: random other neurons of the same count, layer, factor, and `spare_bos`, NOT matched on what
      they deliver. Practice readings only: a fresh claim draws a neuron_mul knob's controls as the weight edit's
      whose arithmetic it is (`_draw_claim_controls`)."""
    if not terms or count <= 0:
        return []
    rng = np.random.default_rng(int(seed) & ((1 << 63) - 1))
    hidden_d = ao.hidden_size(h.lm)
    ffn_w = ao.ffn_width(h.lm, 0)
    qms: Dict[int, np.ndarray] = {t.layer: we.quiet_mask(h.lm, t.layer) for t in terms}
    controls: List[Tuple[ResolvedSteerTerm, ...]] = []
    for _c in range(int(count)):
        c_terms: List[ResolvedSteerTerm] = []
        for t in terms:
            if t.kind in ("direction", "neuron_add"):
                qm = qms[t.layer]
                v = rng.standard_normal(hidden_d).astype(np.float64)
                v[qm] = 0.0
                vn = float(np.sqrt(np.dot(v, v)))
                if vn <= 1e-12:
                    v[0] = 1.0
                    vn = 1.0
                unit = (v / vn).astype(np.float32)
                c_terms.append(
                    ResolvedSteerTerm(
                        kind=t.kind,
                        layer=t.layer,
                        spare_bos=t.spare_bos,
                        unit_vec=unit,
                        magnitude=t.magnitude,
                        quiet_share=0.0,
                        canonical={"kind": f"control_{t.kind}", "layer": t.layer, "magnitude": round(t.magnitude, 6)},
                    )
                )
            elif t.kind == "neuron_mul":
                excl = set(t.neurons)
                pool = [n for n in range(ffn_w) if n not in excl]
                if len(pool) < len(t.neurons):
                    raise KnobRefused(
                        f"Layer {t.layer} has only {len(pool)} other neurons for {len(t.neurons)} neuron_mul control sites."
                    )
                picked = tuple(sorted(int(x) for x in rng.choice(pool, size=len(t.neurons), replace=False).tolist()))
                c_terms.append(
                    ResolvedSteerTerm(
                        kind="neuron_mul",
                        layer=t.layer,
                        spare_bos=t.spare_bos,
                        neurons=picked,
                        factor=t.factor,
                        canonical={"kind": "control_neuron_mul", "layer": t.layer, "count": len(picked), "factor": t.factor},
                    )
                )
        controls.append(tuple(c_terms))
    return controls


@contextlib.contextmanager
def _steering_hooks(
    lm: Any,
    row_programs: Sequence[Optional[Sequence[ResolvedSteerTerm]]],
    null_rows: Mapping[int, Tuple[int, float]],
    bos_mask: Optional[torch.Tensor] = None,
) -> Iterator[None]:
    """Install temporary per-row activation steering hooks and null-row scaling hooks.

    - `row_programs[r]` is None (untouched base row) or a sequence of `ResolvedSteerTerm`.
    - `null_rows` maps `row_idx -> (layer_idx, e)` scaling that layer's MLP output by `1 + e` (identical to
      `weight_edits.per_row` null rows).
    - Untouched rows (`row_programs[r] is None` and `r not in null_rows`) receive ZERO arithmetic.
    - Every hook is always removed in `finally`.
    """
    hooks: List[Any] = []
    # Group steering terms by layer and hook point
    add_by_layer: Dict[int, List[Tuple[int, ResolvedSteerTerm]]] = {}
    mul_by_layer: Dict[int, List[Tuple[int, ResolvedSteerTerm]]] = {}
    null_by_layer: Dict[int, List[Tuple[int, float]]] = {}

    for r, prog in enumerate(row_programs):
        if not prog:
            continue
        for t in prog:
            if t.kind in ("direction", "neuron_add"):
                add_by_layer.setdefault(t.layer, []).append((r, t))
            elif t.kind == "neuron_mul":
                mul_by_layer.setdefault(t.layer, []).append((r, t))

    for r, (l_i, eps) in null_rows.items():
        null_by_layer.setdefault(l_i, []).append((r, float(eps)))

    try:
        # 1. Null-row scaling on lm.layers[l].mlp output (exact match to weight_edits.per_row null rows)
        for l_i, entries in null_by_layer.items():
            def _make_null(ents: List[Tuple[int, float]]) -> Any:
                def post(_mod: Any, _args: Tuple[Any, ...], out: Any) -> Any:
                    h_out = out[0] if isinstance(out, tuple) else out
                    y = h_out.clone()
                    for r_idx, eps in ents:
                        s = torch.tensor(1.0 + eps, dtype=y.dtype, device=y.device)
                        y[r_idx] = y[r_idx] * s
                    return (y, *out[1:]) if isinstance(out, tuple) else y
                return post

            hooks.append(lm.layers[l_i].mlp.register_forward_hook(_make_null(entries)))

        # 2. Pre-hooks on down_proj for neuron_mul
        for l_i, entries in mul_by_layer.items():
            def _make_mul(ents: List[Tuple[int, ResolvedSteerTerm]]) -> Any:
                def pre(_mod: Any, args: Tuple[Any, ...]) -> Tuple[Any, ...]:
                    x = args[0]
                    y = x.clone()
                    for r_idx, t in ents:
                        idx_t = torch.as_tensor(t.neurons, dtype=torch.long, device=y.device)
                        g = torch.tensor(t.factor, dtype=y.dtype, device=y.device)
                        if t.spare_bos and bos_mask is not None and y.shape[1] == bos_mask.shape[1]:
                            m = (~bos_mask[r_idx]).to(y.dtype).unsqueeze(-1)  # [T, 1]
                            cur = y[r_idx, :, idx_t]
                            y[r_idx, :, idx_t] = cur * (1.0 + (g - 1.0) * m)
                        else:
                            y[r_idx, :, idx_t] = y[r_idx, :, idx_t] * g
                    return (y, *args[1:])
                return pre

            hooks.append(lm.layers[l_i].mlp.down_proj.register_forward_pre_hook(_make_mul(entries)))

        # 3. Post-hooks on decoder layer output for residual-stream direction / neuron_add
        for l_i, entries in add_by_layer.items():
            def _make_add(ents: List[Tuple[int, ResolvedSteerTerm]]) -> Any:
                cached: Dict[Tuple[int, Any, Any], torch.Tensor] = {}

                def post(_mod: Any, _args: Tuple[Any, ...], out: Any) -> Any:
                    h_out = out[0] if isinstance(out, tuple) else out
                    y = h_out.clone()
                    for j, (r_idx, t) in enumerate(ents):
                        key = (j, y.device, y.dtype)
                        dv = cached.get(key)
                        if dv is None:
                            dv = (torch.as_tensor(t.unit_vec, dtype=torch.float32, device=y.device) * float(t.magnitude)).to(y.dtype)
                            cached[key] = dv
                        if t.spare_bos and bos_mask is not None and y.shape[1] == bos_mask.shape[1]:
                            m = (~bos_mask[r_idx]).to(y.dtype).unsqueeze(-1)
                            y[r_idx] = y[r_idx] + m * dv.unsqueeze(0)
                        else:
                            y[r_idx] = y[r_idx] + dv.unsqueeze(0)
                    return (y, *out[1:]) if isinstance(out, tuple) else y
                return post

            hooks.append(lm.layers[l_i].register_forward_hook(_make_add(entries)))

        yield
    finally:
        for hk in reversed(hooks):
            hk.remove()


def _apply_logit_bias(logits: torch.Tensor, logit_bias: Optional[Mapping[str, float]]) -> torch.Tensor:
    if not logit_bias:
        return logits
    out = logits.clone()
    for k, v in logit_bias.items():
        out[..., int(k)] = out[..., int(k)] + float(v)
    return out



def _score_states_pair(
    h: lv.Handle,
    H_base: torch.Tensor,
    sp_base: lv._Span,
    H_arm: torch.Tensor,
    sp_arm: lv._Span,
    readouts: Mapping[str, Any],
    *,
    logit_bias: Optional[Mapping[str, float]] = None,
) -> Dict[str, Optional[float]]:
    """Compute paired KL, own log-prob diff, and odds diff over aligned answer positions between base and arm."""
    out: Dict[str, Optional[float]] = {"kl": None, "own": None, "odds": None}
    with torch.no_grad():
        x_b = H_base[sp_base.lo:sp_base.T - 1]  # [L_ans, D]
        x_a = H_arm[sp_arm.lo:sp_arm.T - 1]      # [L_ans, D]
        if readouts.get("kl") or readouts.get("own"):
            lb_raw = logits_fp32(x_b, h.W_U)
            la_raw = _apply_logit_bias(logits_fp32(x_a, h.W_U), logit_bias)
            L0 = F.log_softmax(lb_raw, dim=-1)
            La = F.log_softmax(la_raw, dim=-1)
            if readouts.get("kl"):
                kl_pos = (L0.exp() * (L0 - La)).sum(dim=-1)
                out["kl"] = _clean_float(kl_pos.mean().item())
            if readouts.get("own"):
                own0 = L0.gather(1, sp_base.y.unsqueeze(1)).squeeze(1)
                owna = La.gather(1, sp_arm.y.unsqueeze(1)).squeeze(1)
                out["own"] = _clean_float((owna - own0).mean().item())
        odds_cfg = readouts.get("odds")
        if isinstance(odds_cfg, Mapping):
            t_ids = [int(i) for i in odds_cfg["toward"]]
            a_ids = [int(i) for i in odds_cfg["away"]]
            Wt = h.W_U[torch.as_tensor(t_ids, dtype=torch.long, device=h.W_U.device)].to(h.device).float()
            Wa = h.W_U[torch.as_tensor(a_ids, dtype=torch.long, device=h.W_U.device)].to(h.device).float()
            at_end = odds_cfg.get("at", "end") == "end"
            qb = H_base[sp_base.T - 1:sp_base.T].float() if at_end else x_b.float()
            qa = H_arm[sp_arm.T - 1:sp_arm.T].float() if at_end else x_a.float()
            zt_b = torch.tanh(qb @ Wt.T / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP
            za_b = torch.tanh(qb @ Wa.T / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP
            zt_a = torch.tanh(qa @ Wt.T / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP
            za_a = torch.tanh(qa @ Wa.T / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP
            if logit_bias:
                for pos_i, tid in enumerate(t_ids):
                    bv = logit_bias.get(str(tid))
                    if bv:
                        zt_a = zt_a.clone()
                        zt_a[..., pos_i] = zt_a[..., pos_i] + float(bv)
                for pos_i, tid in enumerate(a_ids):
                    bv = logit_bias.get(str(tid))
                    if bv:
                        za_a = za_a.clone()
                        za_a[..., pos_i] = za_a[..., pos_i] + float(bv)
            ob = (torch.logsumexp(zt_b, -1) - torch.logsumexp(za_b, -1)).mean().item()
            oa = (torch.logsumexp(zt_a, -1) - torch.logsumexp(za_a, -1)).mean().item()
            out["odds"] = _clean_float(oa - ob)
    return out



# A row's program: None (untouched), a tuple of steering terms (this module's hooks), or a `weight_edits.Arm` (a
# weight edit's control, read by the weight path's own per-row hooks).
Program = Any


@contextlib.contextmanager
def _row_hooks(
    lm: Any,
    programs: Sequence[Program],
    null_rows: Mapping[int, Tuple[int, float]],
    bos_mask: Optional[torch.Tensor] = None,
) -> Iterator[None]:
    """Row r of every forward in the block runs `programs[r]`: steering terms through `_steering_hooks`, a weight
    edit through `weight_edits.per_row` (the hooks the weight path reads its own control edits with). Rows are
    disjoint: each hook touches its own rows only. Every hook comes off when the block ends."""
    steer = [p if isinstance(p, tuple) and p else None for p in programs]
    arms = [p if isinstance(p, we.Arm) else None for p in programs]
    with contextlib.ExitStack() as stack:
        stack.enter_context(_steering_hooks(lm, steer, null_rows, bos_mask=bos_mask))
        if any(a is not None for a in arms):
            stack.enter_context(we.per_row(lm, arms))
        yield


def _eval_paired_arms(
    h: lv.Handle,
    ref_item: lv.Item,
    knob_item: lv.Item,
    arms: Sequence[Tuple[str, Program, Optional[Mapping[str, float]]]],
    readouts: Mapping[str, Any],
    metric: str,
    rows: int,
    anchor_layer: int,
) -> Dict[str, Any]:
    """Read `arms` ([(name, program, logit bias)]) on one item at one batch class of `rows` rows. The base row is
    `ref_item` (the item under the brief's chat settings), with null rows at `anchor_layer` and an untouched twin
    that must match it to the bit; every arm is read on `knob_item` (the item as the knob reads it) and scored against
    the base row with its own logit bias added. Returns {"void", "forwards", "floor", "scores": {name: {"kl", "own",
    "odds"}}}: `floor` is the largest |metric| of the null rows."""
    if rows < len(NULL_ES) + 3:
        raise KnobRefused(f"rows must be at least {len(NULL_ES) + 3} to hold base, twin, null rows, and knob; got {rows}.")
    sp_b = lv._Span.of(ref_item, h.device)
    sp_k = lv._Span.of(knob_item, h.device)
    same = ref_item.ids == knob_item.ids
    bos_id = getattr(h.tokenizer, "bos_token_id", None)
    forwards = 0

    def run(ids: Sequence[int], progs: List[Program], nulls: Dict[int, Tuple[int, float]]) -> torch.Tensor:
        nonlocal forwards
        bos_m = None
        if bos_id is not None:
            row_bos = torch.tensor([int(t) == int(bos_id) for t in ids], dtype=torch.bool, device=h.device)
            bos_m = row_bos.unsqueeze(0).expand(len(progs), -1)
        with _row_hooks(h.lm, progs, nulls, bos_mask=bos_m):
            Hf = lv._whole(h, ids, len(progs))
        forwards += 1
        return Hf

    pending = list(arms)
    progs0: List[Program] = [None] * rows
    nulls0 = {1 + i: (anchor_layer, e) for i, e in enumerate(NULL_ES)}
    placed0: List[Tuple[int, str, Optional[Mapping[str, float]]]] = []
    if same:  # the arms share forward 0 with the base, null and twin rows
        for r in range(1 + len(NULL_ES), rows - 1):
            if not pending:
                break
            name, prog, bias = pending.pop(0)
            progs0[r] = prog
            placed0.append((r, name, bias))
    Hf0 = run(ref_item.ids, progs0, nulls0)
    if not torch.isfinite(Hf0[0]).all():
        return {"void": "non-finite base state", "forwards": forwards}
    if not torch.equal(Hf0[0], Hf0[rows - 1]):
        return {"void": "untouched twin row parted from base row", "forwards": forwards}
    H0 = Hf0[0].detach()
    null_vals = []
    for i in range(len(NULL_ES)):
        mv = _score_states_pair(h, H0, sp_b, Hf0[1 + i], sp_b, readouts).get(metric)
        if mv is not None:
            null_vals.append(abs(float(mv)))
    floor_val = max(null_vals) if null_vals else 0.0
    scores: Dict[str, Dict[str, Optional[float]]] = {}
    for r, name, bias in placed0:
        scores[name] = _score_states_pair(h, H0, sp_b, Hf0[r], sp_b, readouts, logit_bias=bias)
    k_row0 = None
    while pending:
        progs: List[Program] = [None] * rows
        placed: List[Tuple[int, str, Optional[Mapping[str, float]]]] = []
        for r in range(1, rows - 1):
            if not pending:
                break
            name, prog, bias = pending.pop(0)
            progs[r] = prog
            placed.append((r, name, bias))
        Hf = run(ref_item.ids if same else knob_item.ids, progs, {})
        if not torch.isfinite(Hf[0]).all():
            return {"void": "non-finite knob base state", "forwards": forwards}
        if same:
            if not torch.equal(Hf[0], H0) or not torch.equal(Hf[0], Hf[rows - 1]):
                return {"void": "later forward base or twin row parted from forward 0", "forwards": forwards}
        else:
            if not torch.equal(Hf[0], Hf[rows - 1]):
                return {"void": "knob forward twin row parted from row 0", "forwards": forwards}
            if k_row0 is None:
                k_row0 = Hf[0].detach()
            elif not torch.equal(Hf[0], k_row0):
                return {"void": "later knob forward row 0 parted from first knob forward", "forwards": forwards}
        for r, name, bias in placed:
            scores[name] = _score_states_pair(h, H0, sp_b, Hf[r], sp_b if same else sp_k, readouts, logit_bias=bias)
    return {"void": None, "forwards": forwards, "floor": float(floor_val), "scores": scores}


def _write_rows_with_bias(
    h: lv.Handle,
    rows_ids: Sequence[Sequence[int]],
    progs: Sequence[Program],
    *,
    max_new: int,
    sampler: Optional[SamplerSettings],
    samples_of: Optional[Sequence[int]],
    row_biases: Sequence[Optional[Mapping[str, float]]],
) -> List[Dict[str, Any]]:
    """Batched decode on the untouched base model with per-row programs (`_row_hooks`) and per-row logit bias."""
    lm = h.lm
    W_U = h.W_U
    dev = h.device
    B = len(rows_ids)
    settings = ao.sampling_of(sampler, int(W_U.shape[0]))
    race = None
    if settings is not None:
        of = [0] * B if samples_of is None else [int(s) for s in samples_of]
        race = ao._Race(settings, int(sampler.seed), of, dev)
    written: List[List[int]] = [[] for _ in range(B)]
    T = max(len(r) for r in rows_ids)
    ids = torch.tensor([[0] * (T - len(r)) + [int(t) for t in r] for r in rows_ids], dtype=torch.long, device=dev)
    mask = torch.tensor([[0] * (T - len(r)) + [1] * len(r) for r in rows_ids], dtype=torch.long, device=dev)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    bos_id = getattr(h.tokenizer, "bos_token_id", None)
    bos_m = (ids == int(bos_id)) if bos_id is not None else None
    stop_ids = set(int(x) for x in h.rt.end_of_turn_ids())
    done = [False] * B
    ended = [False] * B

    with torch.no_grad(), _row_hooks(lm, progs, {}, bos_mask=bos_m):
        out = lm(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
        cache = out.past_key_values
        last_h = out.last_hidden_state[:, -1]
        del out
        cur_pos = pos[:, -1:] + 1
        cur_mask = mask
        for _step in range(int(max_new)):
            logits = ao._model_logits(lm, W_U, last_h, FINAL_LOGIT_SOFTCAP)
            for r_i, bmap in enumerate(row_biases):
                if bmap:
                    logits = logits.clone()
                    for k, v in bmap.items():
                        logits[r_i, int(k)] = logits[r_i, int(k)] + float(v)
            if race is None:
                nxt = logits.argmax(-1)
            else:
                nxt, _p = race.draw(logits, written)
            nxt_list = nxt.tolist()
            all_done = True
            for r_i in range(B):
                if done[r_i]:
                    continue
                tok = int(nxt_list[r_i])
                written[r_i].append(tok)
                if tok in stop_ids:
                    done[r_i] = True
                    ended[r_i] = True
                else:
                    all_done = False
            if all_done:
                break
            cur_mask = torch.cat([cur_mask, torch.ones((B, 1), dtype=torch.long, device=dev)], dim=1)
            out = lm(input_ids=nxt[:, None], attention_mask=cur_mask, position_ids=cur_pos,
                     past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            last_h = out.last_hidden_state[:, -1]
            del out
            cur_pos = cur_pos + 1

    rows_out: List[Dict[str, Any]] = []
    for r_i in range(B):
        pieces = written[r_i]
        text = h.tokenizer.decode(pieces, skip_special_tokens=False)
        rows_out.append({
            "pieces": pieces,
            "text": text,
            "ended": ended[r_i],
            "cut_off": (not ended[r_i]) and len(pieces) >= int(max_new),
            "loop": lv.detect_loop(pieces),
            "loop_published": lv.published_loop(pieces, text, ended=ended[r_i]),
        })
    return rows_out


def _sampler_for(cfg: Mapping[str, Any], seed_val: int) -> Optional[SamplerSettings]:
    t = float(cfg.get("temperature", 0.0))
    if t <= 0.0:
        return None
    return SamplerSettings(
        temperature=t,
        top_k=int(cfg.get("top_k", 0)),
        top_p=float(cfg.get("top_p", 1.0)),
        presence_penalty=float(cfg.get("presence_penalty", 0.0)),
        seed=int(seed_val),
    )


def _written_ids(h: lv.Handle, it: lv.Item, knob: KnobSpec, brief: lbr.Brief) -> Tuple[List[int], List[int]]:
    """(the base row's prompt, the knob's prompt) for a written reading of `it`, under the brief's chat settings."""
    if it.question is None:
        raise KnobRefused(f'Item "{it.name}" has no question text to write from.')
    b_sys, b_think = _settings_of(brief)
    ref_sys = it.system if it.system is not None else b_sys
    k_think = bool(knob.decoding.get("thinking", b_think))
    return (lv.chat_ids(h.tokenizer, it.question, system=ref_sys, thinking=b_think),
            lv.chat_ids(h.tokenizer, it.question, system=knob_system_turn(knob, ref_sys), thinking=k_think))


def _eval_written_arms(
    h: lv.Handle,
    items: Sequence[lv.Item],
    knob: KnobSpec,
    brief: lbr.Brief,
    arms: Sequence[Tuple[str, Program, Optional[Mapping[str, float]]]],
    *,
    max_new: Optional[int] = None,
    seeds: Optional[Sequence[int]] = None,
    chunk_rows: int = lv.WRITTEN_ROWS_MAX,
) -> Dict[str, Any]:
    """Write each item under the brief's chat settings and sampler (the base row, with an untouched twin that must
    write the same pieces) and under each of `arms` ([(name, program, logit bias)]: the knob's prompt, thinking
    setting and sampler, each arm's own steering and bias), every seed of the brief's. Returns {"void", "forwards",
    "deltas": {name: [an item's mean over seeds of (arm looped - base looped)]}, "base_loop_rate"}, loops counted
    by the brief's own loop rule (`level_briefs.loop_rule_of`)."""
    wcfg = dict(brief.written_config or {})
    m_new = int(max_new if max_new is not None else wcfg.get("max_new_tokens", 128))
    seed_list = [int(s) for s in (seeds if seeds is not None else (wcfg.get("seeds") or [1, 2]))]
    rule = lbr.loop_rule_of(brief)
    base_s = brief_sampler(brief)
    knob_s = dict(knob.decoding.get("sampler") or base_s)
    chunk = max(1, int(chunk_rows))
    deltas: Dict[str, List[float]] = {name: [] for name, _p, _b in arms}
    base_rates: List[float] = []
    forwards = 0
    for it in items:
        b_ids, k_ids = _written_ids(h, it, knob, brief)
        per: Dict[str, List[float]] = {name: [] for name, _p, _b in arms}
        b_loops: List[float] = []
        for sd in seed_list:
            b_rows = _write_rows_with_bias(
                h, [b_ids, b_ids], [None, None], max_new=m_new, sampler=_sampler_for(base_s, sd), samples_of=[0, 0],
                row_biases=[None, None],
            )
            forwards += 1
            if b_rows[0]["pieces"] != b_rows[1]["pieces"]:
                return {"void": "written base twin row parted from base row", "forwards": forwards}
            b_lp = 1.0 if lv.loop_flag(b_rows[0], rule) else 0.0
            b_loops.append(b_lp)
            for start in range(0, len(arms), chunk):
                part = list(arms[start:start + chunk])
                a_rows = _write_rows_with_bias(
                    h, [k_ids] * len(part), [p for _n, p, _b in part], max_new=m_new,
                    sampler=_sampler_for(knob_s, sd), samples_of=[0] * len(part),
                    row_biases=[b for _n, _p, b in part],
                )
                forwards += 1
                for (name, _p, _b), row in zip(part, a_rows):
                    per[name].append((1.0 if lv.loop_flag(row, rule) else 0.0) - b_lp)
        for name in deltas:
            deltas[name].append(float(np.mean(per[name])))
        base_rates.append(float(np.mean(b_loops)) if b_loops else 0.0)
    return {"void": None, "forwards": forwards, "deltas": deltas,
            "base_loop_rate": float(np.mean(base_rates)) if base_rates else 0.0}


def _paired_forwards(rows: int, n_arms: int, same: bool) -> int:
    """How many forwards `_eval_paired_arms` takes for `n_arms` arms at `rows` rows."""
    inner = max(1, rows - 2)
    if same:
        extra = max(0, n_arms - max(0, rows - 2 - len(NULL_ES)))
        return 1 + (math.ceil(extra / inner) if extra else 0)
    return 1 + (math.ceil(n_arms / inner) if n_arms else 0)


def _same_tokens(knob: KnobSpec, brief: Optional[lbr.Brief]) -> bool:
    _sys, b_think = _settings_of(brief)
    return knob.prompt is None and bool(knob.decoding.get("thinking", b_think)) == b_think


def estimate_knob_reading(
    h: lv.Handle,
    spec: Mapping[str, Any],
    items: Mapping[str, lv.Item],
    *,
    kind: str = "paired",
    controls: int = 3,
    rows: Optional[Mapping[str, int]] = None,
    max_new: int = 48,
    samples: int = 1,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Dry cost estimate (forwards and GPU-seconds) for a cheaper-knob practice reading without touching the GPU."""
    knob = build_knob(h, spec, base_dir=base_dir)
    if not items:
        raise KnobRefused("estimate_knob_reading requires at least one item.")
    n_ctl = int(controls) if knob.steering_resolved else 0
    if kind == "written":
        n_arm_batches = 1 + math.ceil((1 + n_ctl) / lv.WRITTEN_ROWS_MAX)
        fwds = len(items) * max(1, int(samples)) * n_arm_batches
        secs = round(fwds * int(max_new) * 0.095, 3)
        return {"kind": "written", "forwards": fwds, "seconds": secs, "knob_classes": list(knob.knob_classes)}

    r_sm = int((rows or {}).get("small", lv.ROWS_SMALL))
    r_lg = int((rows or {}).get("large", lv.ROWS_LARGE))
    same_tok = (knob.prompt is None) and ("thinking" not in knob.decoding)
    total_fwds = 0
    total_s = 0.0
    for it in items.values():
        T = len(it.ids)
        for R in (r_sm, r_lg):
            n_f = _paired_forwards(R, 1 + n_ctl, same_tok)
            total_fwds += n_f
            total_s += n_f * lv.whole(R, T)
    return {
        "kind": "paired",
        "forwards": int(total_fwds),
        "seconds": round(total_s, 3),
        "knob_classes": list(knob.knob_classes),
        "steering_controls": n_ctl,
    }


def _items_as_read(knob: KnobSpec, items: Sequence[lv.Item], brief: Optional[lbr.Brief]) -> List[Dict[str, Any]]:
    """What a practice note records of the items it read: each one's texts as the knob read them (its system turn
    with the knob's prompt), in the form a reading's entry gives an item, so a later fresh claim's look through the
    book for kept-back wording read in practice (`level_claim.brief_history`) finds it in a knob's prompt too."""
    out = []
    for it in items:
        system, question, answer = knob_texts(knob, it, brief)
        out.append({"item": {"name": it.name, "system": system, "question": question, "answer": answer}})
    return out


def read_knob_practice(
    h: lv.Handle,
    spec: Mapping[str, Any],
    items: Mapping[str, lv.Item],
    *,
    readouts: Optional[Mapping[str, Any]] = None,
    metric: str = "odds",
    direction: int = 0,
    controls: int = 3,
    rows: Optional[Mapping[str, int]] = None,
    seed: int = 0,
    brief: Optional[lbr.Brief] = None,
    base_dir: Optional[Path] = None,
    book_path: Optional[Path] = None,
    name: str = "knob_practice",
) -> Dict[str, Any]:
    """Run a gated paired practice reading of a cheaper knob (no word past a rank; logged in the level book, with the
    items' texts as the knob read them). Refused for a knob with a part that cannot act on a teacher-forced reading.
    The steering controls are `PRACTICE_CONTROLS_SAY`'s."""
    we.assert_base(h.lm, h.hooks_at_load)
    if not items:
        raise KnobRefused("read_knob_practice requires at least one item.")
    allowed_l = brief.allowed_layers if brief is not None else None
    knob = build_knob(h, spec, base_dir=base_dir, allowed_layers=allowed_l)
    _refuse_parts_that_cannot_act(knob, brief, written=False)
    ro = dict(readouts or (brief.readouts if brief is not None else {"kl": True, "own": True, "odds": None}))
    met = str(brief.metric if brief is not None else metric)
    sgn = 1.0 if (brief.direction if brief is not None else direction) == 0 else float(
        brief.direction if brief is not None else direction
    )
    r_sm = int((rows or {}).get("small", lv.ROWS_SMALL))
    r_lg = int((rows or {}).get("large", lv.ROWS_LARGE))
    n_ctl = int(controls) if knob.steering_resolved else 0
    ctl_list = _draw_steering_controls(h, knob.steering_resolved, n_ctl, seed=int(seed)) if n_ctl > 0 else []
    bias = knob.decoding.get("logit_bias")
    arms = [("knob", knob.steering_resolved or None, bias)] + [(f"ctl:{i}", c, bias) for i, c in enumerate(ctl_list)]
    anchor = _anchor_layer(h, knob)

    t0 = time.perf_counter()
    total_fwds = 0
    small_vals: List[float] = []
    large_vals: List[float] = []
    small_floors: List[float] = []
    large_floors: List[float] = []
    ctl_sm_acc: List[List[float]] = [[] for _ in ctl_list]
    unstable_count = 0
    void_why: Optional[str] = None
    recorded = _items_as_read(knob, list(items.values()), brief)

    try:
        for _nm, it in items.items():
            ref, kit = ref_and_knob_items(h, it, knob, brief)
            res_sm = _eval_paired_arms(h, ref, kit, arms, ro, met, r_sm, anchor)
            total_fwds += int(res_sm.get("forwards", 0))
            if res_sm.get("void"):
                void_why = str(res_sm["void"])
                break
            res_lg = _eval_paired_arms(h, ref, kit, arms, ro, met, r_lg, anchor)
            total_fwds += int(res_lg.get("forwards", 0))
            if res_lg.get("void"):
                void_why = str(res_lg["void"])
                break
            v_sm = _clean_float(res_sm["scores"]["knob"].get(met))
            v_lg = _clean_float(res_lg["scores"]["knob"].get(met))
            f_sm = float(res_sm.get("floor", 0.0))
            f_lg = float(res_lg.get("floor", 0.0))
            small_floors.append(f_sm)
            large_floors.append(f_lg)
            if v_sm is None or v_lg is None:
                continue
            ok_agree, _ratio = lv.agree(v_sm, v_lg, max(f_sm, f_lg), met in ("odds", "own"))
            if not ok_agree:
                unstable_count += 1
                continue
            small_vals.append(v_sm)
            large_vals.append(v_lg)
            for ci in range(len(ctl_list)):
                cs = _clean_float(res_sm["scores"][f"ctl:{ci}"].get(met))
                if cs is not None:
                    ctl_sm_acc[ci].append(cs)
    finally:
        we.assert_base(h.lm, h.hooks_at_load)

    elapsed = round(time.perf_counter() - t0, 3)
    sm_mean = float(np.mean(small_vals)) if small_vals else None
    lg_mean = float(np.mean(large_vals)) if large_vals else None
    sm_dir = (sgn * sm_mean) if sm_mean is not None else None
    lg_dir = (sgn * lg_mean) if lg_mean is not None else None
    fl_sm = max(small_floors) if small_floors else None
    fl_lg = max(large_floors) if large_floors else None

    ctl_rank = None
    if ctl_list and sm_dir is not None:
        c_means = [
            (sgn * float(np.mean(xs))) if len(xs) == len(small_vals) and xs else None
            for xs in ctl_sm_acc
        ]
        at_least = sum(1 for cm in c_means if cm is None or cm >= sm_dir)
        ctl_rank = {"at_least": at_least, "of": len(c_means)}

    status = "void" if void_why else ("ok" if small_vals else "void")
    body = {
        "note_type": "knob_practice",
        "name": str(name),
        "status": status,
        "why": void_why,
        "knob": knob.to_dict(),
        "metric": met,
        "n_asked": len(items),
        "n_read": len(small_vals),
        "unstable": unstable_count,
        "small": {"rows": r_sm, "mean": sm_mean, "directed_mean": sm_dir, "floor": fl_sm},
        "large": {"rows": r_lg, "mean": lg_mean, "directed_mean": lg_dir, "floor": fl_lg},
        "steering_controls_rank": ctl_rank,
        "forwards": total_fwds,
        "seconds": elapsed,
        "items": recorded,
    }
    entry = lb.append("note", body, path=book_path)
    return {
        "book": {"seq": entry["seq"], "hash": entry["hash"]},
        "status": status,
        "why": void_why,
        "knob_id": knob.id,
        "knob_classes": list(knob.knob_classes),
        "metric": met,
        "n_asked": len(items),
        "n_read": len(small_vals),
        "unstable": unstable_count,
        "small": body["small"],
        "large": body["large"],
        "steering_controls_rank": ctl_rank,
        "forwards": total_fwds,
        "seconds": elapsed,
        "note": "Practice knob reading only: returns aggregates and ranks, no word past a rank. " + PRACTICE_CONTROLS_SAY,
        "steering_said": [steering_words(t) for t in knob.steering_resolved],  # what each term does, in words
    }


def write_knob_practice(
    h: lv.Handle,
    spec: Mapping[str, Any],
    items: Mapping[str, lv.Item],
    brief: lbr.Brief,
    *,
    max_new: int = 48,
    seeds: Sequence[int] = (1,),
    controls: int = 1,
    seed: int = 0,
    base_dir: Optional[Path] = None,
    book_path: Optional[Path] = None,
    name: str = "knob_write_practice",
) -> Dict[str, Any]:
    """Run a gated written practice reading of a cheaper knob on practice items (logged in the level book, with the
    items' texts as the knob read them). Refused for a knob with a part that cannot act on a written reading."""
    we.assert_base(h.lm, h.hooks_at_load)
    if not items:
        raise KnobRefused("write_knob_practice requires at least one item.")
    knob = build_knob(h, spec, base_dir=base_dir, allowed_layers=brief.allowed_layers)
    _refuse_parts_that_cannot_act(knob, brief, written=True)
    n_ctl = int(controls) if knob.steering_resolved else 0
    ctl_list = _draw_steering_controls(h, knob.steering_resolved, n_ctl, seed=int(seed)) if n_ctl > 0 else []
    bias = knob.decoding.get("logit_bias")
    arms = [("knob", knob.steering_resolved or None, bias)] + [(f"ctl:{i}", c, bias) for i, c in enumerate(ctl_list)]
    recorded = _items_as_read(knob, list(items.values()), brief)
    t0 = time.perf_counter()
    try:
        wres = _eval_written_arms(h, list(items.values()), knob, brief, arms, max_new=int(max_new), seeds=list(seeds))
    finally:
        we.assert_base(h.lm, h.hooks_at_load)
    elapsed = round(time.perf_counter() - t0, 3)
    status = "void" if wres.get("void") else "ok"
    effs = (wres.get("deltas") or {}).get("knob") or []
    sgn = 1.0 if brief.direction == 0 else float(brief.direction)
    mean_eff = float(np.mean(effs)) if effs else None
    dir_eff = (sgn * mean_eff) if mean_eff is not None else None
    base_rate = wres.get("base_loop_rate")
    ctl_rank = None
    if ctl_list and dir_eff is not None:
        dl = wres.get("deltas") or {}
        at_least = 0
        for i in range(len(ctl_list)):
            xs = dl.get(f"ctl:{i}") or []
            at_least += 1 if (not xs or sgn * float(np.mean(xs)) >= dir_eff) else 0  # none read counts as as much
        ctl_rank = {"at_least": at_least, "of": len(ctl_list)}
    body = {
        "note_type": "knob_write_practice",
        "name": str(name),
        "status": status,
        "why": wres.get("void"),
        "knob": knob.to_dict(),
        "n_asked": len(items),
        "base_loop_rate": base_rate,
        "knob_loop_rate": (base_rate + mean_eff) if (base_rate is not None and mean_eff is not None) else None,
        "mean_effect": mean_eff,
        "directed_effect": dir_eff,
        "steering_controls_rank": ctl_rank,
        "forwards": int(wres.get("forwards", 0)),
        "seconds": elapsed,
        "items": recorded,
    }
    entry = lb.append("note", body, path=book_path)
    return {
        "book": {"seq": entry["seq"], "hash": entry["hash"]},
        **{k: v for k, v in body.items() if k not in ("knob", "items")},
        "knob_id": knob.id,
        "knob_classes": list(knob.knob_classes),
        "note": "Practice knob reading only: returns aggregates and ranks, no word past a rank. " + PRACTICE_CONTROLS_SAY,
        "steering_said": [steering_words(t) for t in knob.steering_resolved],  # what each term does, in words
    }


# -------------------------------------------------------------------------------------------- fresh knob claims


@dataclass(frozen=True)
class _ClaimPlan:
    """What a fresh knob claim will read, settled before anything is registered (`_claim_plan`)."""

    knob: KnobSpec
    written: bool
    families: Tuple[str, ...]
    C: int
    eq_edit: Optional[we.Edit]  # neuron_mul: the weight edit whose arithmetic the knob is
    mirror: Optional[Tuple[Tuple[ResolvedSteerTerm, ...], Optional[Dict[str, float]]]]
    tol: float
    rows: Dict[str, int]
    written_rows: int
    design: Dict[str, Any]


def _claim_plan(
    h: lv.Handle,
    brief: lbr.Brief,
    spec: Mapping[str, Any],
    *,
    base_dir: Optional[Path],
    n_target: Any,
    n_neutral: Any,
    n_near: Any,
    n_panel: Any,
    rows: Optional[Mapping[str, int]],
    written_rows: Any,
    controls_override: Any,
    match_tolerance: Any,
    min_frames_per_half: int,
) -> _ClaimPlan:
    """Every refusal a fresh knob claim can be given without its book, in plain words (`KnobRefused`), and what it
    will read: the knob, its families of controls and their count, the weight edit a neuron_mul knob's arithmetic
    is, its reverse, and the design registered before the draw."""
    knob = build_knob(h, spec, base_dir=base_dir, allowed_layers=brief.allowed_layers)
    written = brief.metric == "written_loop"
    terms = knob.steering_resolved
    if not terms:
        raise KnobRefused(NO_CONTROLS_SAY)
    kinds = {t.kind for t in terms}
    if "neuron_mul" in kinds and kinds != {"neuron_mul"}:
        raise KnobRefused(
            "A fresh knob claim reads a neuron_mul knob against a weight edit's controls and a push against pushes "
            "along random directions: a knob that mixes neuron_mul with direction or neuron_add terms has no one "
            "family of controls, so it is refused. Claim the parts as knobs of their own."
        )
    _refuse_parts_that_cannot_act(knob, brief, written=written)
    C = controls_override if controls_override is not None else brief.default_controls
    for what, n, least in (("n_target", n_target, 2 * int(min_frames_per_half)), ("n_neutral", n_neutral, 1),
                           ("n_near", n_near, 1), ("n_panel", n_panel, 1), ("controls a family", C, 1)):
        if isinstance(n, bool) or not isinstance(n, int) or n < least:
            raise KnobRefused(f"A fresh reading takes {what} as a whole number of {least} or more; {n!r} was given.")
    rows_cfg = {"small": (rows or {}).get("small", lv.ROWS_SMALL), "large": (rows or {}).get("large", lv.ROWS_LARGE)}
    for cls, r in rows_cfg.items():
        if isinstance(r, bool) or not isinstance(r, int) or r < len(NULL_ES) + 3:
            raise KnobRefused(f"A knob claim's {cls} class takes a whole number of rows of {len(NULL_ES) + 3} or more "
                              f"(the base, its twin, {len(NULL_ES)} null rows and an arm); {r!r} was given.")
    w_rows = lcl.DESIGN_DEFAULTS["written_rows"] if written_rows is None else written_rows
    if isinstance(w_rows, bool) or not isinstance(w_rows, int) or not 1 <= w_rows <= lv.WRITTEN_ROWS_MAX:
        raise KnobRefused(f"written_rows is a whole number from 1 to {lv.WRITTEN_ROWS_MAX}; {w_rows!r} was given.")
    tol = lv.DEFAULT_MATCH_TOL if match_tolerance is None else match_tolerance
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not math.isfinite(float(tol)) or float(tol) < 1.0:
        raise KnobRefused(f"The matched controls' tolerance is a finite factor of 1 or more; {match_tolerance!r} was given.")
    eq_edit: Optional[we.Edit] = None
    if kinds == {"neuron_mul"}:
        if any(t.spare_bos for t in terms):
            raise KnobRefused(
                "A neuron_mul that spares the start mark is the arithmetic of no weight edit, so its controls cannot "
                "be drawn and read as a weight edit's are: claim it without spare_bos."
            )
        try:
            eq_edit = we.build(h.lm, [we.scale(t.layer, list(t.neurons), float(t.factor)) for t in terms],
                               model_id=h.model_id, weights_id=h.weights_id,
                               note="the weight edit whose arithmetic a neuron_mul knob is")
            lcl._check_edit_allowed_by_brief(eq_edit, brief)
        except (we.Refused, lcl.ClaimRefused) as exc:
            raise KnobRefused(f"The weight edit whose arithmetic this neuron_mul knob is cannot be claimed: {exc}") from None
        for d in ([1.0, -1.0] if brief.direction != 0 else [1.0]):
            words = we.dose_refusal(eq_edit, d, h.dtype)
            if words:
                raise KnobRefused(f"The weight edit whose arithmetic this knob is cannot be read at dose {d:g}: {words}")
        families: Tuple[str, ...] = ("other_sites_matched", "same_site")
    else:
        families = ("random_directions",)
    mirror = _mirror_of(knob) if brief.direction != 0 else None
    if knob.prompt is not None:
        hit = lbr.KeptBack([brief], batteries=True).hit(knob.prompt)
        if hit is not None:
            whose = "of the retention batteries" if hit["brief"] == lbr.BATTERIES else f'of brief "{hit["brief"]}"'
            raise KnobRefused(
                f"The knob's prompt holds the {hit['how']} of a kept-back frame {whose} ({hit['frame']}): it would "
                "stand in the system turn of every fresh item, so the reading would not be on fresh frames."
            )
    # a prompt or a thinking setting that cannot be made into an item is refused here, before anything is registered
    sample = brief.generator.draw("practice", 1, seed_bytes=b"preflight_knob", tokenizer=h.tokenizer,
                                  system_override=(brief.chat_settings or {}).get("system"))
    if sample:
        if written:
            _written_ids(h, sample[0].item, knob, brief)
        ref_and_knob_items(h, sample[0].item, knob, brief)
    design = {
        "n_target": int(n_target), "n_neutral": int(n_neutral), "n_near": int(n_near), "n_panel": int(n_panel),
        "rows": {k: int(v) for k, v in rows_cfg.items()}, "written_rows": int(w_rows),
        "controls_per_family": int(C), "control_families": list(families), "match_tolerance": float(tol),
        "reversed": mirror is not None, "settings": {"system": _settings_of(brief)[0], "thinking": _settings_of(brief)[1]},
    }
    if written:
        design["loop_rule"] = lbr.loop_rule_of(brief)
    return _ClaimPlan(knob=knob, written=written, families=families, C=int(C), eq_edit=eq_edit, mirror=mirror,
                      tol=float(tol), rows={k: int(v) for k, v in rows_cfg.items()}, written_rows=int(w_rows),
                      design=design)


def _price(h: lv.Handle, brief: lbr.Brief, plan: _ClaimPlan) -> Dict[str, Any]:
    """Forwards and GPU-seconds of the claim `plan` reads, with no forward (lengths from a sizing draw)."""
    d = plan.design
    prac = brief.generator.draw("practice", max(1, min(int(d["n_target"]), 4)), seed_bytes=b"est_knob",
                                tokenizer=h.tokenizer, system_override=(brief.chat_settings or {}).get("system"))
    sample_t = max((len(d_.item.ids) for d_ in prac), default=64) + (len(lv.chat_ids(h.tokenizer, "", plan.knob.prompt))
                                                                       if plan.knob.prompt else 0)
    same = _same_tokens(plan.knob, brief)
    n_arms = 1 + (1 if plan.mirror is not None else 0) + plan.C * len(plan.families)
    n_bat = int(d["n_neutral"]) + int(d["n_near"]) + int(d["n_panel"])
    fwds = 0
    secs = 0.0
    if plan.eq_edit is not None:  # the matched draw's base pre-pass: one 1-row forward an item
        fwds += int(d["n_target"])
        secs += int(d["n_target"]) * lv.whole(1, sample_t)
    for R in (plan.rows["small"], plan.rows["large"]):
        bf = _paired_forwards(R, 1, same)
        fwds += n_bat * bf
        secs += n_bat * bf * lv.whole(R, sample_t)
    if plan.written:
        wcfg = dict(brief.written_config or {})
        m_new = int(wcfg.get("max_new_tokens", 128))
        n_seeds = max(1, len(wcfg.get("seeds") or [1, 2]))
        dec = int(d["n_target"]) * n_seeds * (1 + math.ceil(n_arms / plan.written_rows))
        fwds += dec
        secs += dec * m_new * 0.095
    else:
        for R in (plan.rows["small"], plan.rows["large"]):
            tf = _paired_forwards(R, n_arms, same)
            fwds += int(d["n_target"]) * tf
            secs += int(d["n_target"]) * tf * lv.whole(R, sample_t)
    return {"forwards": int(fwds), "seconds": round(float(secs), 2), "knob_classes": list(plan.knob.knob_classes),
            "steering_controls": plan.C, "control_families": list(plan.families), "arms": n_arms}


def estimate_knob_claim(
    h: lv.Handle,
    brief: lbr.Brief,
    spec: Mapping[str, Any],
    *,
    base_dir: Optional[Path] = None,
    n_target: int = 8,
    n_neutral: int = 4,
    n_near: int = 4,
    n_panel: int = 4,
    rows: Optional[Mapping[str, int]] = None,
    written_rows: Optional[int] = None,
    controls_override: Optional[int] = None,
    match_tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    """Price one fresh cheaper-knob claim reading before registering it (zero GPU work). Refuses, in the claim's own
    words, every knob the claim would refuse without its book (`_claim_plan`)."""
    plan = _claim_plan(h, brief, spec, base_dir=base_dir, n_target=n_target, n_neutral=n_neutral, n_near=n_near,
                       n_panel=n_panel, rows=rows, written_rows=written_rows, controls_override=controls_override,
                       match_tolerance=match_tolerance, min_frames_per_half=lcl.MIN_FRAMES_PER_HALF)
    return _price(h, brief, plan)


def _draw_claim_controls(
    h: lv.Handle,
    plan: _ClaimPlan,
    score_items: Sequence[lv.Item],
    seed_u32: int,
) -> Tuple[Optional[Dict[str, List[Program]]], Optional[Dict[str, Any]], Optional[str], int]:
    """The claim's controls, a family each ({family: [program, ...]}), the matched draw's coverage, why they could not
    be drawn (or None), and the forwards the draw took. A neuron_mul knob's are its weight edit's: other sites drawn
    by the weight path's matched draw on a base pre-pass over `score_items` (the items as the knob reads them), and
    same-site random directions, each read as a `weight_edits.Arm`. A push's are pushes along random directions."""
    if plan.eq_edit is not None:
        kind = "written" if plan.written else "paired"
        scores = lv._score_sites(h, plan.eq_edit, list(score_items), kind)
        matched, cov = we.matched_control_edits(plan.eq_edit, h.lm, scores, int(seed_u32), plan.C, tolerance=plan.tol)
        cov_clean = lv._clean(cov)
        for term in (cov_clean.get("terms") or ()) if isinstance(cov_clean, dict) else ():
            if isinstance(term, dict):
                term.pop("edit_scores", None)
        if matched is None:
            return None, cov_clean, (
                f"Matched controls requested {cov.get('requested')} distinct control sets within factor "
                f"{cov.get('tolerance')} of expected delivered magnitude, but only {cov.get('found')} were available "
                "on these items."), len(score_items)
        same = we.same_site_controls(plan.eq_edit, h.lm, int(seed_u32) + 1, plan.C)
        return ({"other_sites_matched": [we.Arm(e, 1.0) for e in matched],
                 "same_site": [we.Arm(e, 1.0) for e in same]}, cov_clean, None, len(score_items))
    pushes = _draw_steering_controls(h, plan.knob.steering_resolved, plan.C, seed=int(seed_u32))
    return {"random_directions": list(pushes)}, None, None, 0


def _battery_classes(
    h: lv.Handle,
    items: Sequence[lv.Item],
    knob: KnobSpec,
    brief: lbr.Brief,
    readouts: Mapping[str, Any],
    metric: str,
    rows: Mapping[str, int],
    anchor: int,
) -> Dict[str, Any]:
    """One retention battery read under `knob` in both batch classes, as a weight claim reads its batteries:
    {"void", "forwards", "given", "classes": {cls: {"mean", "abs_mean", "own_mean", "kl_mean", "n_read"}}}. Each item
    is read against the base row under the brief's chat settings (`ref_and_knob_items`)."""
    arm = [("knob", knob.steering_resolved or None, knob.decoding.get("logit_bias"))]
    forwards = 0
    per: Dict[str, Dict[str, List[float]]] = {c: {"m": [], "own": [], "kl": []} for c in rows}
    for it in items:
        ref, kit = ref_and_knob_items(h, it, knob, brief)
        for cls, R in rows.items():
            res = _eval_paired_arms(h, ref, kit, arm, readouts, metric, int(R), anchor)
            forwards += int(res.get("forwards", 0))
            if res.get("void"):
                return {"void": f"{cls} class: {res['void']}", "forwards": forwards}
            sc = res["scores"]["knob"]
            for key, src in (("m", metric), ("own", "own"), ("kl", "kl")):
                v = _clean_float(sc.get(src))
                if v is not None:
                    per[cls][key].append(v)
    out: Dict[str, Any] = {}
    for cls, acc in per.items():
        out[cls] = {
            "mean": float(np.mean(acc["m"])) if acc["m"] else None,
            "abs_mean": float(np.mean([abs(v) for v in acc["m"]])) if acc["m"] else None,
            "own_mean": float(np.mean(acc["own"])) if acc["own"] else None,
            "kl_mean": float(np.mean(acc["kl"])) if acc["kl"] else None,
            "n_read": len(acc["m"]),
        }
    return {"void": None, "forwards": forwards, "given": len(items), "classes": out}


def _largest(vals: Sequence[Optional[float]]) -> Optional[float]:
    got = [v for v in vals if v is not None]
    return max(got) if got else None


def _stricter(judged: Mapping[str, Any], ci_class: str, panel: Optional[Mapping[str, Any]]) -> Tuple[str, List[str]]:
    """The verdict after the knob gate's two rules that are stricter than `level_claim.judge_claim`, each where the
    engine's order puts its kind: a frame CI (in `ci_class`) whose lower end does not clear zero is "did not
    replicate", which comes before the controls; a labelled-panel KL that was not read is harm, which comes after
    them. Returns (verdict, the reasons added)."""
    verdict = str(judged["verdict"])
    added: List[str] = []
    if verdict in ("no better than control edits", "works but harms", "lead"):
        fr = ((judged.get("classes") or {}).get(ci_class) or {}).get("frames") or {}
        lo = _clean_float((fr.get("directed_ci95") or [None, None])[0])
        if lo is None or lo <= 0.0:
            verdict = "did not replicate"
            added.append(f"{ci_class} class frame CI does not clear zero (its lower end is {lo}): the knob gate "
                         "reads that as not replicated, which a weight claim does not ask")
    if verdict == "lead" and isinstance(panel, Mapping) and panel.get("mean_kl") is None:
        verdict = "works but harms"
        added.append("labelled_panel KL was not read: the knob gate counts that as harm")
    return verdict, added


def knob_batteries(bat: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The retention batteries as `level_claim.judge_claim` takes them, from `_battery_classes`' readings of each
    (`{name: {"given", "classes": {cls: ...}}}`), as a weight claim makes them: the larger figure of the two batch
    classes, for the neutral battery's KL, the near misses' size, and the panel's KL and own-probability drop. A
    near miss's size is the mean of each item's size (never less than the size of the mean, which a weight claim
    takes), and a panel figure missing in either class is missing. The habitat battery is not run, as it is not for
    a weight claim: no atlas passages index is mounted."""
    out: Dict[str, Dict[str, Any]] = {}
    for name in ("neutral_text", "near_misses", "labelled_panel"):
        b = bat.get(name)
        if not isinstance(b, Mapping):
            continue
        cl = dict(b.get("classes") or {})
        items = {"read": min((int(c.get("n_read", 0)) for c in cl.values()), default=0), "given": b.get("given")}
        if name == "neutral_text":
            out[name] = {"mean_kl": _largest([c.get("mean") for c in cl.values()]), "items": items}
        elif name == "near_misses":
            out[name] = {"mean_effect": (cl.get("large") or {}).get("mean"),
                         "abs_effect": _largest([c.get("abs_mean") for c in cl.values()]), "items": items}
        else:
            drops = [max(0.0, -c["own_mean"]) if c.get("own_mean") is not None else None for c in cl.values()]
            kls = [c.get("kl_mean") for c in cl.values()]
            out[name] = {"own_drop": None if (not drops or None in drops) else max(drops),
                         "mean_kl": None if (not kls or None in kls) else max(kls), "items": items}
    out["habitat"] = {"available": False, "note": "no atlas passages index mounted"}
    return out


def judge_knob(
    *,
    brief: lbr.Brief,
    bars: lcl.BarSet,
    classes: Mapping[str, Mapping[str, Any]],
    batteries: Mapping[str, Mapping[str, Any]],
    families: Sequence[str],
    practice_count: int = 0,
    practice_best: Optional[float] = None,
    void_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """The verdict of a fresh knob claim: `level_claim.judge_claim` (the weight claim's engine: frames a half, the
    house bar, half replication, the reversed read, every family of controls in every class, every gating battery
    the bars name, the near-miss size and its ratio to the effect), given the knob's own `families`, and then the knob
    gate's stricter rules (`_stricter`). The brief's author prompt arms are not read beside a knob. Pure: no reading.
    Returns {"verdict", "reasons", "added" (the reasons `_stricter` gave), "judged" (the engine's whole result)}."""
    judged = lcl.judge_claim(
        brief=brief, bars=bars, classes=classes, batteries=batteries,
        cheaper_knobs={"prompt_arms": [], "prompt_arms_state": PROMPT_ARMS_SAY}, selection_mode="free_pick",
        practice_count=int(practice_count), practice_best=practice_best, void_reason=void_reason,
        control_families=tuple(families),
    )
    ci_class = "written" if brief.metric == "written_loop" else "large"
    verdict, added = _stricter(judged, ci_class, batteries.get("labelled_panel") if not void_reason else None)
    if verdict not in KNOB_VERDICTS:
        raise RuntimeError(f"judge_claim gave a knob claim the verdict {verdict!r}, which no knob claim can have.")
    return {"verdict": verdict, "reasons": list(judged.get("reasons") or []) + added, "added": added,
            "judged": judged}


def _fmt(x: Any, digits: int = 4) -> str:
    v = _clean_float(x)
    return "n/a" if v is None else f"{v:.{digits}f}"


def render_knob_sheet(
    brief: lbr.Brief,
    bars: lcl.BarSet,
    knob: KnobSpec,
    verdict: str,
    summary: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> str:
    """Render a bounded, plain-spoken Knob Claim Sheet (zero kept-out words): what was read, what its controls are
    and are not matched on, what the verdict came from, and what a weight claim reads that this one does not."""
    st = summary.get("settings") or {}
    lines = [
        f"# Cheaper-Knob Claim Sheet — {brief.name}",
        f"- Verdict: **{verdict}**",
        f"- Knob id: `{knob.short()}` (classes: {', '.join(knob.knob_classes)}); zero weights modified",
        f"- Base: `{knob.model_id}` (weights `{str(knob.weights_id or 'none')[:12]}`), reading #{meta.get('reading_index')} "
        f"of {brief.max_fresh_readings}: the brief's fresh readings are one budget, spent by weight claims, knob "
        "claims and trials in this book alike",
        f"- Brief: `{brief.short()}` (`{brief.metric}`, least effect M = {brief.least_effect:g}), bars `{bars.short()}` "
        f"(alpha = {bars.alpha:g}); judged by {lcl.RULE_PRESET} v{lcl.RULE_VERSION} (level_claim.judge_claim)",
        f"- Book: `{meta.get('book_name')}` ({'default level book' if meta.get('is_default_book') else 'side book'})",
        f"- Read under the brief's chat settings: system turn {'set' if st.get('system') else 'none'}, thinking "
        f"{'on' if st.get('thinking') else 'off'}; the knob's prompt is added to that system turn and its thinking "
        "setting, when it has one, replaces the brief's",
    ]
    if knob.prompt is not None:
        lines.append(f"- Prompt part: {len(knob.prompt)} chars (sha256 `{hashlib.sha256(knob.prompt.encode('utf-8')).hexdigest()[:12]}`)")
    if knob.decoding:
        lines.append(f"- Decoding part: `{json.dumps(knob.decoding, sort_keys=True)}`")
    for i, t in enumerate(knob.steering_resolved):
        lines.append(f"- Steering term {i}: {steering_words(t)}")
    lines.append(
        f"- Target effect (directed): small = {summary.get('small_directed')}, large = {summary.get('large_directed')}, "
        f"house bar = {summary.get('house_bar')} (floor {summary.get('floor_worst')}), replicated = {summary.get('replicated')}"
    )
    rv = summary.get("reversed") or {}
    if rv:
        lines.append("- Reversed read: " + ", ".join(f"{c} = {_fmt(v.get('directed'))}" for c, v in rv.items())
                     + f" (directed; it fails when it reaches the house bar). {REVERSED_SAY}")
    lines.append(f"- Controls ({summary.get('controls_per_family')} a family; {CONTROLS_SHARE_SAY}). "
                 f"{CONTROL_RULE_SAY}:")
    for fam in summary.get("control_families") or ():
        lines.append(f"  - {fam}: {CONTROL_FAMILY_SAY[fam].format(tol=float(summary.get('match_tolerance') or lv.DEFAULT_MATCH_TOL))}")
        for cls, by_fam in (summary.get("control_standing") or {}).items():
            cs = by_fam.get(fam) or {}
            lines.append(f"    {cls}: {cs.get('at_least')} of {cs.get('count')} moved the metric as much "
                         f"(p_fw = {_fmt(cs.get('p_fw'))}, passes = {cs.get('passes_alpha')})")
    bats = summary.get("batteries") or {}
    lines.append(
        f"- Retention batteries (both batch classes, the larger figure kept): neutral KL = {bats.get('neutral_kl')} "
        f"(bar <= {bars.neutral_kl_max:g}); near-miss size = {bats.get('near_miss_abs')} (bar <= "
        f"{bars.near_miss_abs_max:g}; ratio to the effect = {bats.get('near_miss_ratio')}, bar <= "
        f"{bars.near_miss_ratio_max:g}); panel KL = {bats.get('panel_kl')} (bar <= {bars.panel_kl_max:g}), panel own "
        f"drop = {bats.get('panel_own_drop')} (bar <= {bars.panel_own_drop_max:g}); habitat: not run (no atlas passages "
        f"index mounted); gating: {', '.join(bars.gating_batteries)}"
    )
    lines.append(f"- {STRICTER_SAY}")
    lines.append("- Not read for a knob claim: " + "; ".join(summary.get("not_read") or NOT_READ_SAY))
    if summary.get("reasons"):
        lines.append(f"- Reasons: {'; '.join(str(r) for r in summary['reasons'])}")
    lines.append(f"- Cost: {meta.get('forwards')} forward(s), {meta.get('seconds')} s; base verified untouched before and after.")
    return "\n".join(lines)


def run_fresh_knob_claim(
    h: lv.Handle,
    brief: lbr.Brief,
    bars: lcl.BarSet,
    spec: Mapping[str, Any],
    *,
    base_dir: Optional[Path] = None,
    forecast: Optional[Mapping[str, Any]] = None,
    practice_count: int = 0,
    practice_best: Optional[float] = None,
    practice_item_ids: Optional[Sequence[str]] = None,
    practice_frame_ids: Optional[Sequence[str]] = None,
    n_target: int = 8,
    n_neutral: int = 4,
    n_near: int = 4,
    n_panel: int = 4,
    rows: Optional[Mapping[str, int]] = None,
    written_rows: Optional[int] = None,
    controls_override: Optional[int] = None,
    match_tolerance: Optional[float] = None,
    book_path: Optional[Path] = None,
    alpha: Optional[float] = None,
    _seed_override: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Run one registered fresh cheaper-knob claim on `brief`'s kept-back frames and calibrated `bars`.

    1. Refuse, before anything is registered, what `_claim_plan` refuses, and what the weight path's own gate
       refuses (`level_claim.fresh_reading_gate`, what="knob": one budget with weight claims and trials).
    2. Register (`knob_claim_registration`), then draw from `os.urandom(32)`: kept-back items refusing those seen in
       practice, near misses, neutral and panel items.
    3. Read the knob, its reverse and its controls on the target items in both batch classes (or written), and the
       batteries in both classes; judge with `level_claim.judge_claim` and the knob gate's stricter rules
       (`_stricter`); write the verdict (`knob_claim_verdict`) and the sheet."""
    we.assert_base(h.lm, h.hooks_at_load)
    if bars.brief_id != brief.id:
        raise KnobRefused(f"BarSet {bars.short()} is for brief {bars.brief_id[:12]}, not {brief.short()}.")
    if alpha is not None and _clean_float(alpha) != float(bars.alpha):
        raise KnobRefused(f"A knob claim is judged at its bars' level (alpha = {bars.alpha:g}), as a weight claim is; "
                          f"{alpha!r} was given.")
    bk = Path(book_path).resolve() if book_path is not None else lb.book_path().resolve()
    plan = _claim_plan(h, brief, spec, base_dir=base_dir, n_target=n_target, n_neutral=n_neutral, n_near=n_near,
                       n_panel=n_panel, rows=rows, written_rows=written_rows, controls_override=controls_override,
                       match_tolerance=match_tolerance, min_frames_per_half=bars.min_frames_per_half)
    knob = plan.knob
    try:
        gate = lcl.fresh_reading_gate(h, brief, bars, None, bk, what="knob", knob_id=knob.id,
                                      practice_item_ids=practice_item_ids, practice_frame_ids=practice_frame_ids)
    except lcl.ClaimRefused as exc:
        raise KnobRefused(str(exc)) from None
    hist, cal_note, reading_index = gate["hist"], gate["cal_note"], int(gate["reading_index"])
    practice_total = max(int(practice_count), int(hist["practice_readings"]))
    priced = _price(h, brief, plan)
    kb_frames = brief.generator.frames_for("kept_back")
    published = lbr.published_template_ids(h.tokenizer)
    kept_back_note = {"frames": len(kb_frames), "published": sum(1 for fr in kb_frames if fr.template_id in published),
                      "template_ids": list(gate["kb_tids"])}
    d = plan.design
    r_sm, r_lg = plan.rows["small"], plan.rows["large"]

    reg_body = {
        "note_type": "knob_claim_registration",
        "brief_id": brief.id,
        "brief_name": brief.name,
        "bars_sha256": bars.id,
        "bars_calibration_seq": int(cal_note["seq"]),
        "knob_id": knob.id,
        "knob": knob.to_dict(),
        "knob_classes": list(knob.knob_classes),
        "reading_index": reading_index,
        "max_fresh_readings": int(brief.max_fresh_readings),
        "rule_preset": lcl.RULE_PRESET,
        "rule_version": lcl.RULE_VERSION,
        "knob_claim_version": KNOB_CLAIM_VERSION,
        "practice_count": practice_total,
        "practice_count_given": int(practice_count),
        "practice_readings_in_book": int(hist["practice_readings"]),
        "practice_best": _clean_float(practice_best),
        "forecast": dict(forecast) if isinstance(forecast, Mapping) else None,
        "design": d,
        "priced": {"forwards": int(priced["forwards"]), "seconds": float(priced["seconds"])},
        "kept_back": kept_back_note,
        "book": gate["book"],
        "book_fresh_readings": dict(hist["book_fresh_readings"]),
        "prior_verdicts": [
            {"seq": v["seq"], "edit_id": v.get("edit_id"), "knob_id": v.get("knob_id"), "verdict": v.get("verdict"),
             "reading_index": v.get("reading_index"), "brief_id": v.get("brief_id")}
            for v in hist["verdicts"]
        ],
        "prior_trials": lcl.prior_trials(hist),
        "model_id": h.model_id,
        "weights_id": h.weights_id,
    }
    reg_entry = lb.append("note", reg_body, path=bk)

    t0 = time.perf_counter()
    total_fwds = 0
    void_reason: Optional[str] = None
    sgn = 1.0 if brief.direction == 0 else float(brief.direction)
    bias = knob.decoding.get("logit_bias")
    anchor = _anchor_layer(h, knob)
    cls_names: Tuple[str, ...] = ("written",) if plan.written else ("small", "large")
    kept_drawn: List[lbr.DrawnItem] = []
    effs: Dict[str, List[float]] = {c: [] for c in cls_names}
    revs: Dict[str, List[float]] = {c: [] for c in cls_names}
    ctl_acc: Dict[str, Dict[str, List[List[float]]]] = {}
    floors: Dict[str, List[float]] = {c: [] for c in cls_names}
    unstable_n = 0
    coverage = None
    bat: Dict[str, Any] = {}
    seed_sha = None

    try:
        seed_bytes = bytes(_seed_override) if _seed_override is not None else os.urandom(32)
        seed_sha = hashlib.sha256(seed_bytes).hexdigest()
        seed_u32 = int.from_bytes(seed_bytes[:4], "little") & 0x7FFFFFFF
        sys_ovr = (brief.chat_settings or {}).get("system")
        try:
            target_drawn = brief.generator.draw(
                "kept_back", int(d["n_target"]), seed_bytes=seed_bytes + b":target", tokenizer=h.tokenizer,
                system_override=sys_ovr, forbid_item_ids=gate["forbid_items"], forbid_frame_ids=gate["forbid_frames"],
                name_prefix=f"kcl_{reading_index}_tgt",
            )
            near_drawn = brief.near_miss_generator.draw(
                "kept_back", int(d["n_near"]), seed_bytes=seed_bytes + b":near", tokenizer=h.tokenizer,
                system_override=sys_ovr, name_prefix=f"kcl_{reading_index}_nm",
            )
            neu_drawn = lbr.neutral_chat_generator().draw(
                "kept_back", int(d["n_neutral"]), seed_bytes=seed_bytes + b":neu", tokenizer=h.tokenizer,
                name_prefix=f"kcl_{reading_index}_neu",
            )
            panel_items = [row["item"] for row in lbr.draw_labelled_sanity_panel(
                int(d["n_panel"]), seed_bytes=seed_bytes + b":panel", tokenizer=h.tokenizer)]
        except lbr.BriefError as exc:
            # a fresh item seen in practice is found only by the draw: the reading is void and says why
            void_reason = f"The fresh draw was refused: {exc}"
            target_drawn, near_drawn, neu_drawn, panel_items = [], [], [], []

        controls: Dict[str, List[Program]] = {}
        pairs: Dict[str, Tuple[lv.Item, lv.Item]] = {}
        if void_reason is None:
            try:
                if plan.written:
                    score_items = [lv.item_from_ids(_written_ids(h, dd.item, knob, brief)[1], dd.item.answer_ids,
                                                    name=dd.item.name) for dd in target_drawn]
                else:
                    pairs = {dd.item.name: ref_and_knob_items(h, dd.item, knob, brief) for dd in target_drawn}
                    score_items = [pairs[dd.item.name][1] for dd in target_drawn]
                got, coverage, why_not, pre_f = _draw_claim_controls(h, plan, score_items, seed_u32)
            except (KnobRefused, lv.Refused, we.Refused) as exc:
                # what only the items as drawn can tell (an item too long under the knob, say) ends it void
                got, why_not, pre_f = None, f"The items as drawn cannot be read under this knob: {exc}", 0
            total_fwds += pre_f
            if got is None:
                void_reason = why_not
            else:
                controls = got
        ctl_acc = {c: {fam: [[] for _ in progs] for fam, progs in controls.items()} for c in cls_names}
        arms: List[Tuple[str, Program, Optional[Mapping[str, float]]]] = [("knob", knob.steering_resolved, bias)]
        if plan.mirror is not None:
            arms.append(("reversed", plan.mirror[0], plan.mirror[1]))
        for fam, progs in controls.items():
            arms += [(f"{fam}:{c}", p, bias) for c, p in enumerate(progs)]

        if void_reason is None and plan.written:
            wres = _eval_written_arms(h, [dd.item for dd in target_drawn], knob, brief, arms,
                                      chunk_rows=plan.written_rows)
            total_fwds += int(wres.get("forwards", 0))
            if wres.get("void"):
                void_reason = str(wres["void"])
            else:
                kept_drawn = list(target_drawn)
                dl = wres["deltas"]
                effs["written"] = [float(x) for x in dl["knob"]]
                revs["written"] = [float(x) for x in dl.get("reversed", [])]
                for fam, progs in controls.items():
                    for c in range(len(progs)):
                        ctl_acc["written"][fam][c] = [float(x) for x in dl[f"{fam}:{c}"]]
        elif void_reason is None:
            metric = brief.metric
            for dd in target_drawn:
                ref, kit = pairs[dd.item.name]
                got_cls = {}
                for cls, R in (("small", r_sm), ("large", r_lg)):
                    res = _eval_paired_arms(h, ref, kit, arms, brief.readouts, metric, R, anchor)
                    total_fwds += int(res.get("forwards", 0))
                    if res.get("void"):
                        void_reason = f"{cls} class: {res['void']}"
                        break
                    got_cls[cls] = res
                if void_reason is not None:
                    break
                v_s = _clean_float(got_cls["small"]["scores"]["knob"].get(metric))
                v_l = _clean_float(got_cls["large"]["scores"]["knob"].get(metric))
                f_s, f_l = float(got_cls["small"]["floor"]), float(got_cls["large"]["floor"])
                floors["small"].append(f_s)
                floors["large"].append(f_l)
                if v_s is None or v_l is None:
                    continue
                ok_agree, _ratio = lv.agree(v_s, v_l, max(f_s, f_l), metric in ("odds", "own"))
                if not ok_agree:
                    unstable_n += 1
                    continue
                kept_drawn.append(dd)
                for cls in ("small", "large"):
                    sc = got_cls[cls]["scores"]
                    effs[cls].append(_clean_float(sc["knob"].get(metric)))
                    if "reversed" in sc:
                        rv = _clean_float(sc["reversed"].get(metric))
                        if rv is not None:
                            revs[cls].append(rv)
                    for fam, progs in controls.items():
                        for c in range(len(progs)):
                            cv = _clean_float(sc[f"{fam}:{c}"].get(metric))
                            if cv is not None:
                                ctl_acc[cls][fam][c].append(cv)

        # the retention batteries: teacher-forced, both batch classes, as a weight claim reads them
        nm_metric = brief.metric if brief.metric != "written_loop" else "kl"
        nm_ro = dict(brief.readouts) if nm_metric == "odds" else {**dict(brief.readouts), "odds": None, nm_metric: True}
        kl_ro = {"kl": True, "own": True, "odds": None}
        for bname, items_b, ro, met in (("neutral_text", [dd.item for dd in neu_drawn], kl_ro, "kl"),
                                        ("near_misses", [dd.item for dd in near_drawn], nm_ro, nm_metric),
                                        ("labelled_panel", panel_items, kl_ro, "kl")):
            if void_reason is not None:
                break
            try:
                res_b = _battery_classes(h, items_b, knob, brief, ro, met, plan.rows, anchor)
            except (KnobRefused, lv.Refused) as exc:
                void_reason = f"{bname} battery: its items cannot be read under this knob: {exc}"
                break
            total_fwds += int(res_b.get("forwards", 0))
            if res_b.get("void"):
                void_reason = f"{bname} battery: {res_b['void']}"
                break
            bat[bname] = res_b
    except BaseException as exc:
        we.assert_base(h.lm, h.hooks_at_load)
        lb.append(
            "note",
            {
                "note_type": "knob_claim_verdict",
                "brief_id": brief.id,
                "brief_name": brief.name,
                "bars_sha256": bars.id,
                "knob_id": knob.id,
                "knob_classes": list(knob.knob_classes),
                "reading_index": reading_index,
                "max_fresh_readings": int(brief.max_fresh_readings),
                "registration_seq": int(reg_entry["seq"]),
                "verdict": "void",
                "reasons": [f"error during knob claim: {type(exc).__name__}: {exc}"],
                "summary": {"verdict": "void", "reasons": [f"error during knob claim: {type(exc).__name__}: {exc}"]},
                "cost": {"forwards": total_fwds, "seconds": round(time.perf_counter() - t0, 3)},
            },
            path=bk,
        )
        raise
    finally:
        we.assert_base(h.lm, h.hooks_at_load)

    elapsed = round(time.perf_counter() - t0, 3)

    # judge: the weight claim's engine, with the knob's families of controls
    floor_worst = _largest([max(v) if v else None for v in floors.values()])
    n_kept = len(kept_drawn)
    classes: Dict[str, Dict[str, Any]] = {}
    for cls in cls_names:
        classes[cls] = {
            "item_effects": [{"frame_id": dd.frame_id, "half": dd.half, "item_id": dd.item.id, "value": v}
                             for dd, v in zip(kept_drawn, effs[cls])],
            "floor": None if plan.written else floor_worst,  # the larger class's floor for both: stricter
            "reversed_mean": float(np.mean(revs[cls])) if revs[cls] else None,
            # a control's mean is None unless it read on every item kept (`control_standing` counts None against)
            "controls": {fam: [float(np.mean(xs)) if (xs and len(xs) == n_kept) else None for xs in lists]
                         for fam, lists in (ctl_acc.get(cls) or {}).items()},
        }
    batteries = knob_batteries(bat)
    if void_reason is None and not kept_drawn:
        void_reason = "no target items remained stable across batch classes"
    jk = judge_knob(brief=brief, bars=bars, classes=classes, batteries=batteries, families=plan.families,
                    practice_count=practice_total, practice_best=practice_best, void_reason=void_reason)
    judged, verdict, reasons, added = jk["judged"], jk["verdict"], jk["reasons"], jk["added"]

    jcls = judged.get("classes") or {}
    jbat = judged.get("batteries") or {}
    dirs = {c: _clean_float(((jcls.get(c) or {}).get("frames") or {}).get("directed_mean")) for c in cls_names}
    hb = (jcls.get(cls_names[-1]) or {}).get("house_bar") or {}
    harm_reasons = [f"{b}: {v}" for b, v in jbat.items() if v.get("gating") and not v.get("passed")]
    harm_reasons += [a for a in added if "harm" in a]
    not_read = list(NOT_READ_SAY) + ([NOT_READ_WRITTEN_SAY] if plan.written else []) \
        + ([NOT_READ_SAMPLER_SAY] if "sampler" in knob.decoding else [])
    summary = {
        "verdict": verdict,
        "reasons": reasons,
        "n_target_asked": int(d["n_target"]),
        "n_target_read": n_kept,
        "unstable_target": unstable_n,
        "small_directed": round(dirs[cls_names[0]], 5) if dirs[cls_names[0]] is not None else None,
        "large_directed": round(dirs[cls_names[-1]], 5) if dirs[cls_names[-1]] is not None else None,
        "floor_small": round(max(floors["small"]), 6) if floors.get("small") else None,
        "floor_large": round(max(floors["large"]), 6) if floors.get("large") else None,
        "floor_worst": round(floor_worst, 6) if floor_worst is not None else None,
        "house_bar": round(hb["bar"], 5) if _clean_float(hb.get("bar")) is not None else None,
        "house_bar_dropped": bool(hb.get("floor_too_loud")),
        "replicated": bool(jcls) and all(bool(v.get("halves_ok")) and bool(v.get("reversed_ok", True))
                                         for v in jcls.values())
        and verdict not in ("did not replicate", "void", "kept the base"),
        "frame_stats": (jcls.get(cls_names[-1]) or {}).get("frames"),
        "controls_applicable": True,
        "controls_per_family": plan.C,
        "control_families": list(plan.families),
        "match_tolerance": plan.tol,
        "control_standing": {c: dict((jcls.get(c) or {}).get("controls") or {}) for c in cls_names} if jcls else None,
        # each control's mean over the items kept, as judged (None: it did not read on every item, and is counted
        # as moving as much)
        "control_means": {c: dict(classes[c]["controls"]) for c in cls_names},
        "controls_coverage": coverage,
        # a weight edit's controls (a neuron_mul knob's) by their edit ids, as a weight reading's entry names them
        "control_ids": {fam: [p.edit.id for p in progs] for fam, progs in controls.items()
                        if progs and all(isinstance(p, we.Arm) for p in progs)},
        "reversed": ({c: {"mean": (jcls.get(c) or {}).get("reversed_mean"),
                          "directed": (sgn * jcls[c]["reversed_mean"]) if (jcls.get(c) or {}).get("reversed_mean")
                          is not None else None,
                          "ok": (jcls.get(c) or {}).get("reversed_ok")} for c in cls_names}
                     if (plan.mirror is not None and jcls) else None),
        "settings": d["settings"],
        "batteries": {
            "neutral_kl": _clean_float((batteries.get("neutral_text") or {}).get("mean_kl")),
            "near_miss_abs": _clean_float((batteries.get("near_misses") or {}).get("abs_effect")),
            "near_miss_ratio": _clean_float((jbat.get("near_misses") or {}).get("ratio_to_target")),
            "panel_kl": _clean_float((batteries.get("labelled_panel") or {}).get("mean_kl")),
            "panel_own_drop": _clean_float((batteries.get("labelled_panel") or {}).get("own_drop")),
            "harm_passed": not harm_reasons,
            "harm_reasons": harm_reasons,
            "judged": jbat,
        },
        "not_read": not_read,
    }
    is_default_book = (bk == lb.default_book_path().resolve())
    meta = {
        "reading_index": reading_index,
        "book_name": bk.name,
        "is_default_book": is_default_book,
        "forwards": total_fwds,
        "seconds": elapsed,
    }
    sheet = render_knob_sheet(brief, bars, knob, verdict, summary, meta)

    verdict_body = {
        "note_type": "knob_claim_verdict",
        "brief_id": brief.id,
        "brief_name": brief.name,
        "bars_sha256": bars.id,
        "knob_id": knob.id,
        "knob_classes": list(knob.knob_classes),
        "reading_index": reading_index,
        "max_fresh_readings": int(brief.max_fresh_readings),
        "registration_seq": int(reg_entry["seq"]),
        "verdict": verdict,
        "reasons": reasons,
        "draw_seed_sha256": seed_sha,
        "summary": lv._clean(summary),
        "cost": {"forwards": total_fwds, "seconds": elapsed},
    }
    verdict_entry = lb.append("note", verdict_body, path=bk)

    driver_record = {
        "brief_id": brief.id,
        "brief_name": brief.name,
        "knob_id": knob.id,
        "knob_classes": list(knob.knob_classes),
        "reading_index": reading_index,
        "max_fresh_readings": int(brief.max_fresh_readings),
        "verdict": verdict,
        "summary": summary,
        "sheet": sheet,
        "book": {
            "registration_seq": int(reg_entry["seq"]),
            "verdict_seq": int(verdict_entry["seq"]),
            "all_seqs": [int(reg_entry["seq"]), int(verdict_entry["seq"])],
        },
    }
    return {
        "verdict": verdict,
        "knob": knob.to_dict(),
        "summary": summary,
        "judged": judged,
        "sheet": sheet,
        "driver_record": driver_record,
        "cost": {"forwards": total_fwds, "seconds": elapsed},
        "book": driver_record["book"],
    }
