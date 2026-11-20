"""Levelling Phase 6: cheaper-knob readings and fresh knob claims (`backend/level_knobs.py`).

Evaluates non-weight interventions — prompts, decoding settings (sampler, thinking toggle, logit bias), and transient
activation steering pushes — on the untouched base model, under the same brief, kept-back frames, retention batteries,
calibrated bar set, and statistical gates as weight-edit claims (`backend/level_claim.py`).

Why this exists (`verify/arbiter.md` §8, `design/synthesis.md` §ix):
- Phase 6 compares weight-edit sessions against the same driver and the same budget allowed ONLY prompts, decoding, and
  steering (`mode="knobs_only"`), replacing "with no search over cheaper knobs" with a matched-budget measurement.
- Zero weights are ever modified. `weight_edits.assert_base` is asserted before and after every call.
- Every reading logs to the chained level book (`kind="note"` with `note` in `"knob_practice"`,
  `"knob_claim_registration"`, `"knob_claim_verdict"`), checks untouched-row bit-identity and null-row noise floors,
  and evaluates both batch classes (`small` and `large`) for paired metrics.

Layers count from 0 in code and files; human-facing strings give both ("layer 50 (page layer 51)").
Zero kept-out words appear in source, docstrings, or rendered sheets.
"""
from __future__ import annotations

import contextlib
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

KNOB_VERSION = 1
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

KNOB_VERDICTS = (
    "lead",
    "no better than control edits",
    "works but harms",
    "did not replicate",
    "kept the base",
    "void",
)


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


def _draw_steering_controls(
    h: lv.Handle,
    terms: Sequence[ResolvedSteerTerm],
    count: int,
    seed: int,
) -> List[Tuple[ResolvedSteerTerm, ...]]:
    """Build `count` matched random steering controls:
    - For `direction` / `neuron_add`: random Gaussian directions in the same layer, projected off `quiet_mask`,
      unit-normalized, with the exact same magnitude and `spare_bos`.
    - For `neuron_mul`: random disjoint other-site neurons of the same count, layer, factor, and `spare_bos`."""
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


def _make_knob_item(
    h: lv.Handle,
    base_item: lv.Item,
    knob: KnobSpec,
    brief: Optional[lbr.Brief] = None,
) -> lv.Item:
    """Build the tokenized item seen by `knob` (applying prompt and/or thinking override if set)."""
    base_sys = base_item.system
    if brief is not None and base_sys is None:
        base_sys = (brief.chat_settings or {}).get("system")
    new_sys = base_sys
    if knob.prompt is not None:
        new_sys = f"{base_sys}\n{knob.prompt}" if base_sys else knob.prompt
    base_think = bool((brief.chat_settings or {}).get("thinking", False)) if brief is not None else False
    new_think = bool(knob.decoding.get("thinking", base_think))
    if new_sys == base_item.system and new_think == base_think:
        return base_item
    if base_item.question is None or base_item.answer is None:
        raise KnobRefused(f'Item "{base_item.name}" has no question/answer text to apply a prompt/thinking knob.')
    p_ids = lv.chat_ids(h.tokenizer, base_item.question, system=new_sys, thinking=new_think)
    a_ids = list(base_item.answer_ids)
    try:
        return lv._item(p_ids, a_ids, base_item.name, base_item.question, base_item.answer, new_sys)
    except lv.Refused as exc:
        raise KnobRefused(f'Prompt/thinking knob item "{base_item.name}": {exc}') from None


def _eval_paired_item_class(
    h: lv.Handle,
    base_item: lv.Item,
    knob_item: lv.Item,
    knob: KnobSpec,
    controls: Sequence[Tuple[ResolvedSteerTerm, ...]],
    readouts: Mapping[str, Any],
    metric: str,
    rows: int,
) -> Dict[str, Any]:
    """Evaluate one item at one batch class (`rows`), checking untouched-row bit-identity and null-row floor."""
    null_es = (-0.125, -0.03125, 0.03125, 0.125)
    # Place null rows at the lowest steered layer (matching levelling.py), or at the final layer when no
    # hidden-layer steering is active (prompt / decoding knobs enter at the input/output boundary, not layer 0).
    anchor_layer = min((t.layer for t in knob.steering_resolved), default=max(0, h.n_layers - 1))
    sp_b = lv._Span.of(base_item, h.device)
    sp_k = lv._Span.of(knob_item, h.device)
    same_tokens = (base_item.ids == knob_item.ids)
    lb_map = knob.decoding.get("logit_bias")
    forwards = 0

    def _run_batch(ids: Sequence[int], progs: List[Optional[Sequence[ResolvedSteerTerm]]], nulls: Dict[int, Tuple[int, float]]) -> torch.Tensor:
        nonlocal forwards
        bos_id = getattr(h.tokenizer, "bos_token_id", None)
        bos_m = None
        if bos_id is not None:
            row_bos = torch.tensor([int(t) == int(bos_id) for t in ids], dtype=torch.bool, device=h.device)
            bos_m = row_bos.unsqueeze(0).expand(len(progs), -1)
        with _steering_hooks(h.lm, progs, nulls, bos_mask=bos_m):
            Hf = lv._whole(h, ids, len(progs))
        forwards += 1
        return Hf

    # Available slots per forward: row 0 = base, last row = twin, rows 1..len(null_es) on forward 0 = null rows
    if rows < len(null_es) + 3:
        raise KnobRefused(f"rows must be at least {len(null_es) + 3} to hold base, twin, null rows, and knob; got {rows}.")

    # Forward 0 on base_item: base (0), null rows (1..4), optional knob + controls when same_tokens, twin (rows-1)
    progs0: List[Optional[Sequence[ResolvedSteerTerm]]] = [None] * rows
    nulls0: Dict[int, Tuple[int, float]] = {1 + i: (anchor_layer, e) for i, e in enumerate(null_es)}
    slot_start = 1 + len(null_es)
    slot_end = rows - 1

    knob_scores: Optional[Dict[str, Optional[float]]] = None
    ctl_scores: List[Optional[float]] = []

    if same_tokens:
        # Pack knob + steering controls right into the same forward(s) as base and null rows!
        arm_list: List[Tuple[str, Optional[Sequence[ResolvedSteerTerm]]]] = [("knob", knob.steering_resolved or None)]
        for ci, c_terms in enumerate(controls):
            arm_list.append((f"ctl_{ci}", c_terms))

        idx = 0
        # Forward 0
        f0_arms: List[Tuple[int, str]] = []
        for r in range(slot_start, slot_end):
            if idx >= len(arm_list):
                break
            name_a, prog_a = arm_list[idx]
            progs0[r] = prog_a
            f0_arms.append((r, name_a))
            idx += 1
        Hf0 = _run_batch(base_item.ids, progs0, nulls0)
        if not torch.isfinite(Hf0[0]).all():
            return {"void": "non-finite base state", "forwards": forwards}
        if not torch.equal(Hf0[0], Hf0[rows - 1]):
            return {"void": "untouched twin row parted from base row", "forwards": forwards}
        H0 = Hf0[0].detach()
        # Measure null-row floor
        null_vals: List[float] = []
        for i in range(len(null_es)):
            ns = _score_states_pair(h, H0, sp_b, Hf0[1 + i], sp_b, readouts)
            mv = ns.get(metric)
            if mv is not None:
                null_vals.append(abs(float(mv)))
        floor_val = max(null_vals) if null_vals else 0.0

        for r, name_a in f0_arms:
            sc = _score_states_pair(
                h, H0, sp_b, Hf0[r], sp_b, readouts,
                logit_bias=(lb_map if name_a == "knob" else None),
            )
            if name_a == "knob":
                knob_scores = sc
            else:
                ctl_scores.append(sc.get(metric))

        # Extra forwards for remaining controls
        while idx < len(arm_list):
            progs_n: List[Optional[Sequence[ResolvedSteerTerm]]] = [None] * rows
            fn_arms: List[Tuple[int, str]] = []
            for r in range(1, rows - 1):
                if idx >= len(arm_list):
                    break
                name_a, prog_a = arm_list[idx]
                progs_n[r] = prog_a
                fn_arms.append((r, name_a))
                idx += 1
            Hfn = _run_batch(base_item.ids, progs_n, {})
            if not torch.equal(Hfn[0], H0) or not torch.equal(Hfn[0], Hfn[rows - 1]):
                return {"void": "later forward base or twin row parted from forward 0", "forwards": forwards}
            for r, _name_a in fn_arms:
                sc = _score_states_pair(h, H0, sp_b, Hfn[r], sp_b, readouts)
                ctl_scores.append(sc.get(metric))
    else:
        # Different input tokens (prompt / thinking knob): run base forward for H0 + floor, then knob forward(s).
        Hf_b = _run_batch(base_item.ids, progs0, nulls0)
        if not torch.isfinite(Hf_b[0]).all():
            return {"void": "non-finite base state", "forwards": forwards}
        if not torch.equal(Hf_b[0], Hf_b[rows - 1]):
            return {"void": "untouched base twin row parted from base row", "forwards": forwards}
        H0 = Hf_b[0].detach()
        null_vals = []
        for i in range(len(null_es)):
            ns = _score_states_pair(h, H0, sp_b, Hf_b[1 + i], sp_b, readouts)
            mv = ns.get(metric)
            if mv is not None:
                null_vals.append(abs(float(mv)))
        floor_val = max(null_vals) if null_vals else 0.0

        # Knob forward(s) on knob_item: row 0 = prompted base (no steering), last row = twin, inner rows = knob + controls
        arm_list = [("knob", knob.steering_resolved or None)]
        for ci, c_terms in enumerate(controls):
            arm_list.append((f"ctl_{ci}", c_terms))
        idx = 0
        Hk0_ref = None
        while idx < len(arm_list):
            progs_k: List[Optional[Sequence[ResolvedSteerTerm]]] = [None] * rows
            fk_arms: List[Tuple[int, str]] = []
            for r in range(1, rows - 1):
                if idx >= len(arm_list):
                    break
                name_a, prog_a = arm_list[idx]
                progs_k[r] = prog_a
                fk_arms.append((r, name_a))
                idx += 1
            Hfk = _run_batch(knob_item.ids, progs_k, {})
            if not torch.isfinite(Hfk[0]).all():
                return {"void": "non-finite knob base state", "forwards": forwards}
            if not torch.equal(Hfk[0], Hfk[rows - 1]):
                return {"void": "knob forward twin row parted from row 0", "forwards": forwards}
            if Hk0_ref is None:
                Hk0_ref = Hfk[0].detach()
            elif not torch.equal(Hfk[0], Hk0_ref):
                return {"void": "later knob forward row 0 parted from first knob forward", "forwards": forwards}
            for r, name_a in fk_arms:
                sc = _score_states_pair(
                    h, H0, sp_b, Hfk[r], sp_k, readouts,
                    logit_bias=(lb_map if name_a == "knob" else None),
                )
                if name_a == "knob":
                    knob_scores = sc
                else:
                    ctl_scores.append(sc.get(metric))

    return {
        "void": None,
        "forwards": forwards,
        "floor": float(floor_val),
        "scores": knob_scores or {},
        "control_scores": ctl_scores,
    }


def _write_rows_with_bias(
    h: lv.Handle,
    rows_ids: Sequence[Sequence[int]],
    progs: Sequence[Optional[Sequence[ResolvedSteerTerm]]],
    *,
    max_new: int,
    sampler: Optional[SamplerSettings],
    samples_of: Optional[Sequence[int]],
    row_biases: Sequence[Optional[Mapping[str, float]]],
) -> List[Dict[str, Any]]:
    """Batched decode on the untouched base model with optional per-row steering hooks and per-row logit bias."""
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

    with torch.no_grad(), _steering_hooks(lm, progs, {}, bos_mask=bos_m):
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


def _eval_written_items(
    h: lv.Handle,
    items: Sequence[lv.Item],
    knob: KnobSpec,
    brief: lbr.Brief,
    controls: Sequence[Tuple[ResolvedSteerTerm, ...]],
    *,
    max_new: Optional[int] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Evaluate a written_loop brief under `knob` vs base (and optional steering controls)."""
    wcfg = dict(brief.written_config or {})
    m_new = int(max_new if max_new is not None else wcfg.get("max_new_tokens", 128))
    seed_list = list(seeds if seeds is not None else wcfg.get("seeds", [1, 2]))
    loop_rule = str(wcfg.get("loop_rule", "either"))
    base_think = bool((brief.chat_settings or {}).get("thinking", False))
    base_sys = (brief.chat_settings or {}).get("system")
    knob_think = bool(knob.decoding.get("thinking", base_think))
    knob_sys = lcl.prompt_arm_system(brief, knob.prompt) if knob.prompt is not None else base_sys

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

    base_scfg = {
        "temperature": float(wcfg.get("temperature", 0.0)),
        "top_k": int(wcfg.get("top_k", 0)),
        "top_p": float(wcfg.get("top_p", 1.0)),
        "presence_penalty": float(wcfg.get("presence_penalty", 0.0)),
    }
    knob_scfg = dict(knob.decoding.get("sampler") or base_scfg)
    lb_map = knob.decoding.get("logit_bias")

    per_item_base: List[float] = []
    per_item_knob: List[float] = []
    per_item_ctl: List[List[float]] = [[] for _ in controls]
    strata_rows: List[Tuple[str, bool, bool]] = []
    forwards = 0

    for it in items:
        q = it.question or ""
        b_p_ids = lv.chat_ids(h.tokenizer, q, system=base_sys, thinking=base_think)
        k_p_ids = lv.chat_ids(h.tokenizer, q, system=knob_sys, thinking=knob_think)
        b_loops = 0
        k_loops = 0
        c_loops = [0] * len(controls)
        for sd in seed_list:
            # Base decode (with twin row to verify untouched determinism)
            b_rows = _write_rows_with_bias(
                h, [b_p_ids, b_p_ids], [None, None], max_new=m_new,
                sampler=_sampler_for(base_scfg, sd), samples_of=[0, 0], row_biases=[None, None],
            )
            forwards += 1
            if b_rows[0]["pieces"] != b_rows[1]["pieces"]:
                return {"void": "written base twin row parted from base row", "forwards": forwards}
            b_lp = lv.loop_flag(b_rows[0], loop_rule)
            b_loops += int(b_lp)

            # Knob decode
            k_rows = _write_rows_with_bias(
                h, [k_p_ids], [knob.steering_resolved or None], max_new=m_new,
                sampler=_sampler_for(knob_scfg, sd), samples_of=[0], row_biases=[lb_map],
            )
            forwards += 1
            k_lp = lv.loop_flag(k_rows[0], loop_rule)
            k_loops += int(k_lp)
            st_b = lv.written_stratum(b_rows[0]["ended"], b_lp, b_rows[0]["cut_off"])
            strata_rows.append((st_b, b_lp, k_lp))

            # Steering controls (if any)
            if controls:
                c_ids = [k_p_ids] * len(controls)
                c_progs = [c for c in controls]
                c_res = _write_rows_with_bias(
                    h, c_ids, c_progs, max_new=m_new,
                    sampler=_sampler_for(base_scfg, sd), samples_of=[0] * len(controls),
                    row_biases=[None] * len(controls),
                )
                forwards += 1
                for ci, cr in enumerate(c_res):
                    c_loops[ci] += int(lv.loop_flag(cr, loop_rule))

        n_s = max(1, len(seed_list))
        per_item_base.append(b_loops / n_s)
        per_item_knob.append(k_loops / n_s)
        for ci in range(len(controls)):
            per_item_ctl[ci].append(c_loops[ci] / n_s)

    item_diffs = [k - b for b, k in zip(per_item_base, per_item_knob)]
    ctl_means = [
        float(np.mean([c - b for b, c in zip(per_item_base, per_item_ctl[ci])]))
        for ci in range(len(controls))
    ]
    return {
        "void": None,
        "forwards": forwards,
        "item_effects": item_diffs,
        "base_loop_rate": float(np.mean(per_item_base)) if per_item_base else 0.0,
        "knob_loop_rate": float(np.mean(per_item_knob)) if per_item_knob else 0.0,
        "control_means": ctl_means,
        "strata_rows": strata_rows,
    }


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
    """Dry cost estimate (forwards and GPU-seconds) for a cheaper-knob reading without touching the GPU."""
    knob = build_knob(h, spec, base_dir=base_dir)
    if not items:
        raise KnobRefused("estimate_knob_reading requires at least one item.")
    n_ctl = int(controls) if knob.steering_resolved else 0
    if kind == "written":
        n_arm_batches = 2 + (1 if n_ctl > 0 else 0)
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
            if same_tok:
                cap0 = max(1, R - 6)
                extra = max(0, (1 + n_ctl) - cap0)
                cap_n = max(1, R - 2)
                n_f = 1 + (math.ceil(extra / cap_n) if extra > 0 else 0)
            else:
                cap_n = max(1, R - 2)
                n_f = 1 + math.ceil((1 + n_ctl) / cap_n)
            total_fwds += n_f
            total_s += n_f * lv.whole(R, T)
    return {
        "kind": "paired",
        "forwards": int(total_fwds),
        "seconds": round(total_s, 3),
        "knob_classes": list(knob.knob_classes),
        "steering_controls": n_ctl,
    }


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
    """Run a gated paired practice reading of a cheaper knob (no word past a rank; logged in the level book)."""
    we.assert_base(h.lm, h.hooks_at_load)
    if not items:
        raise KnobRefused("read_knob_practice requires at least one item.")
    allowed_l = brief.allowed_layers if brief is not None else None
    knob = build_knob(h, spec, base_dir=base_dir, allowed_layers=allowed_l)
    ro = dict(readouts or (brief.readouts if brief is not None else {"kl": True, "own": True, "odds": None}))
    met = str(brief.metric if brief is not None else metric)
    sgn = 1.0 if (brief.direction if brief is not None else direction) == 0 else float(
        brief.direction if brief is not None else direction
    )
    r_sm = int((rows or {}).get("small", lv.ROWS_SMALL))
    r_lg = int((rows or {}).get("large", lv.ROWS_LARGE))
    n_ctl = int(controls) if knob.steering_resolved else 0
    ctl_list = _draw_steering_controls(h, knob.steering_resolved, n_ctl, seed=int(seed)) if n_ctl > 0 else []

    t0 = time.perf_counter()
    total_fwds = 0
    small_vals: List[float] = []
    large_vals: List[float] = []
    small_floors: List[float] = []
    large_floors: List[float] = []
    ctl_sm_acc: List[List[float]] = [[] for _ in ctl_list]
    ctl_lg_acc: List[List[float]] = [[] for _ in ctl_list]
    unstable_count = 0
    void_why: Optional[str] = None

    try:
        for _nm, it in items.items():
            k_it = _make_knob_item(h, it, knob, brief=brief)
            res_sm = _eval_paired_item_class(h, it, k_it, knob, ctl_list, ro, met, r_sm)
            total_fwds += int(res_sm.get("forwards", 0))
            if res_sm.get("void"):
                void_why = str(res_sm["void"])
                break
            res_lg = _eval_paired_item_class(h, it, k_it, knob, ctl_list, ro, met, r_lg)
            total_fwds += int(res_lg.get("forwards", 0))
            if res_lg.get("void"):
                void_why = str(res_lg["void"])
                break

            v_sm = _clean_float((res_sm.get("scores") or {}).get(met))
            v_lg = _clean_float((res_lg.get("scores") or {}).get(met))
            f_sm = float(res_sm.get("floor", 0.0))
            f_lg = float(res_lg.get("floor", 0.0))
            small_floors.append(f_sm)
            large_floors.append(f_lg)
            if v_sm is None or v_lg is None:
                continue
            signed_m = met in ("odds", "own")
            ok_agree, _ratio = lv.agree(v_sm, v_lg, max(f_sm, f_lg), signed_m)
            if not ok_agree:
                unstable_count += 1
                continue
            small_vals.append(v_sm)
            large_vals.append(v_lg)
            for ci in range(len(ctl_list)):
                cs = _clean_float(res_sm["control_scores"][ci]) if ci < len(res_sm["control_scores"]) else None
                cl = _clean_float(res_lg["control_scores"][ci]) if ci < len(res_lg["control_scores"]) else None
                if cs is not None:
                    ctl_sm_acc[ci].append(cs)
                if cl is not None:
                    ctl_lg_acc[ci].append(cl)
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
        "note": "knob_practice",
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
        "note": "Practice knob reading only: returns aggregates and ranks, no word past a rank.",
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
    """Run a gated written practice reading of a cheaper knob on practice items."""
    we.assert_base(h.lm, h.hooks_at_load)
    if not items:
        raise KnobRefused("write_knob_practice requires at least one item.")
    knob = build_knob(h, spec, base_dir=base_dir, allowed_layers=brief.allowed_layers)
    n_ctl = int(controls) if knob.steering_resolved else 0
    ctl_list = _draw_steering_controls(h, knob.steering_resolved, n_ctl, seed=int(seed)) if n_ctl > 0 else []
    t0 = time.perf_counter()
    try:
        wres = _eval_written_items(
            h, list(items.values()), knob, brief, ctl_list, max_new=int(max_new), seeds=list(seeds),
        )
    finally:
        we.assert_base(h.lm, h.hooks_at_load)
    elapsed = round(time.perf_counter() - t0, 3)
    status = "void" if wres.get("void") else "ok"
    effs = wres.get("item_effects") or []
    sgn = 1.0 if brief.direction == 0 else float(brief.direction)
    mean_eff = float(np.mean(effs)) if effs else None
    dir_eff = (sgn * mean_eff) if mean_eff is not None else None
    body = {
        "note": "knob_write_practice",
        "name": str(name),
        "status": status,
        "why": wres.get("void"),
        "knob": knob.to_dict(),
        "n_asked": len(items),
        "base_loop_rate": wres.get("base_loop_rate"),
        "knob_loop_rate": wres.get("knob_loop_rate"),
        "mean_effect": mean_eff,
        "directed_effect": dir_eff,
        "forwards": int(wres.get("forwards", 0)),
        "seconds": elapsed,
    }
    entry = lb.append("note", body, path=book_path)
    return {
        "book": {"seq": entry["seq"], "hash": entry["hash"]},
        **{k: v for k, v in body.items() if k != "knob"},
        "knob_id": knob.id,
        "knob_classes": list(knob.knob_classes),
    }


def _knob_book_history(brief: lbr.Brief, book_path: Optional[Path]) -> Dict[str, Any]:
    """Count prior fresh readings (weight claims, trials, and knob claims) and prior knob ids on `brief` in `book_path`."""
    hist = lcl.brief_history(brief.id, book_path, name=brief.name)
    knob_regs: List[Dict[str, Any]] = []
    knob_verdicts: List[Dict[str, Any]] = []
    knob_ids: set[str] = set()
    p = Path(book_path) if book_path is not None else lb.book_path()
    bkey = lcl._name_key(brief.name)
    if p.exists():
        for entry in lb.entries(p):
            if entry.get("kind") != "note":
                continue
            body = entry.get("body") or {}
            nt = body.get("note")
            if nt == "knob_claim_registration":
                same = (body.get("brief_id") == brief.id) or (
                    body.get("brief_name") and lcl._name_key(body.get("brief_name")) == bkey
                )
                if same:
                    knob_regs.append(body)
                    if body.get("knob_id"):
                        knob_ids.add(str(body["knob_id"]))
            elif nt == "knob_claim_verdict":
                same = (body.get("brief_id") == brief.id) or (
                    body.get("brief_name") and lcl._name_key(body.get("brief_name")) == bkey
                )
                if same:
                    knob_verdicts.append(body)
    return {
        "weight_hist": hist,
        "knob_regs": knob_regs,
        "knob_verdicts": knob_verdicts,
        "knob_ids": knob_ids,
        "total_fresh": int(hist["fresh_count"]) + len(knob_regs),
    }


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
    controls_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Price one fresh cheaper-knob claim reading before registering it (zero GPU work)."""
    knob = build_knob(h, spec, base_dir=base_dir, allowed_layers=brief.allowed_layers)
    n_ctl = int(controls_override if controls_override is not None else brief.default_controls) if knob.steering_resolved else 0
    # Draw synthetic dummy items from practice split to size token lengths without touching kept-back frames
    prac = brief.generator.draw("practice", max(1, min(int(n_target), 4)), seed_bytes=b"est_knob", tokenizer=h.tokenizer)
    sample_t = max((len(d.item.ids) for d in prac), default=64)
    if brief.metric == "written_loop":
        wcfg = dict(brief.written_config or {})
        m_new = int(wcfg.get("max_new_tokens", 128))
        n_seeds = max(1, len(wcfg.get("seeds") or [1, 2]))
        n_batches = 2 + (1 if n_ctl > 0 else 0)
        fwds = int(n_target) * n_seeds * n_batches + (int(n_neutral) + int(n_near) + int(n_panel)) * 4
        secs = round(int(n_target) * n_seeds * n_batches * m_new * 0.095 + (int(n_neutral) + int(n_near) + int(n_panel)) * 1.0, 2)
        return {"forwards": fwds, "seconds": secs, "knob_classes": list(knob.knob_classes), "steering_controls": n_ctl}

    r_sm = int((rows or {}).get("small", lv.ROWS_SMALL))
    r_lg = int((rows or {}).get("large", lv.ROWS_LARGE))
    same_tok = (knob.prompt is None) and ("thinking" not in knob.decoding)
    fwds_per_tgt = 0
    secs_per_tgt = 0.0
    for R in (r_sm, r_lg):
        if same_tok:
            cap0 = max(1, R - 6)
            extra = max(0, (1 + n_ctl) - cap0)
            cap_n = max(1, R - 2)
            nf = 1 + (math.ceil(extra / cap_n) if extra > 0 else 0)
        else:
            cap_n = max(1, R - 2)
            nf = 1 + math.ceil((1 + n_ctl) / cap_n)
        fwds_per_tgt += nf
        secs_per_tgt += nf * lv.whole(R, sample_t)
    # Retention batteries (no steering controls needed on batteries, just base + knob)
    n_bat = int(n_neutral) + int(n_near) + int(n_panel)
    fwds_bat = n_bat * (2 if same_tok else 4)
    secs_bat = n_bat * (lv.whole(r_sm, sample_t) * (2 if same_tok else 4))
    return {
        "forwards": int(int(n_target) * fwds_per_tgt + fwds_bat),
        "seconds": round(float(int(n_target) * secs_per_tgt + secs_bat), 2),
        "knob_classes": list(knob.knob_classes),
        "steering_controls": n_ctl,
    }


def _eval_knob_battery(
    h: lv.Handle,
    items: Sequence[lv.Item],
    knob: KnobSpec,
    brief: lbr.Brief,
    readouts: Mapping[str, Any],
    metric: str,
    rows: int,
) -> Dict[str, Any]:
    """Evaluate one retention battery (`neutral_text`, `near_misses`, or `labelled_panel`) under `knob`."""
    vals: List[float] = []
    own_vals: List[float] = []
    forwards = 0
    void_why = None
    for it in items:
        k_it = _make_knob_item(h, it, knob, brief=brief)
        res = _eval_paired_item_class(h, it, k_it, knob, [], readouts, metric, rows)
        forwards += int(res.get("forwards", 0))
        if res.get("void"):
            void_why = str(res["void"])
            break
        sc = res.get("scores") or {}
        mv = _clean_float(sc.get(metric))
        if mv is not None:
            vals.append(mv)
        ov = _clean_float(sc.get("own"))
        if ov is not None:
            own_vals.append(ov)
    mean_v = float(np.mean(vals)) if vals else None
    mean_abs = float(np.mean([abs(v) for v in vals])) if vals else None
    mean_own = float(np.mean(own_vals)) if own_vals else None
    return {
        "void": void_why,
        "forwards": forwards,
        "n_asked": len(items),
        "n_read": len(vals),
        "mean": mean_v,
        "abs_mean": mean_abs,
        "own_mean": mean_own,
    }


def render_knob_sheet(
    brief: lbr.Brief,
    bars: lcl.BarSet,
    knob: KnobSpec,
    verdict: str,
    summary: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> str:
    """Render a bounded, plain-spoken Knob Claim Sheet (zero kept-out words)."""
    lines = [
        f"# Cheaper-Knob Claim Sheet — {brief.name}",
        f"- Verdict: **{verdict}**",
        f"- Knob id: `{knob.short()}` (classes: {', '.join(knob.knob_classes)})",
        f"- Base: `{knob.model_id}` (weights `{str(knob.weights_id or 'none')[:12]}`), reading #{meta.get('reading_index')} of {brief.max_fresh_readings}",
        f"- Brief: `{brief.short()}` (`{brief.metric}`, least effect M = {brief.least_effect:g}), bars `{bars.short()}`",
        f"- Book: `{meta.get('book_name')}` ({'default level book' if meta.get('is_default_book') else 'side book'})",
    ]
    if knob.prompt is not None:
        lines.append(f"- Prompt knob: {len(knob.prompt)} chars (sha256 `{hashlib.sha256(knob.prompt.encode('utf-8')).hexdigest()[:12]}`)")
    if knob.decoding:
        lines.append(f"- Decoding knob: `{json.dumps(knob.decoding, sort_keys=True)}`")
    if knob.steering_raw:
        lines.append(f"- Steering knob: {len(knob.steering_raw)} term(s), zero weights modified")
    lines.append(
        f"- Target effect (directed): small = {summary.get('small_directed')}, large = {summary.get('large_directed')}, "
        f"house bar = {summary.get('house_bar')}, replicated = {summary.get('replicated')}"
    )
    if summary.get("controls_applicable"):
        cs = summary.get("control_standing") or {}
        lines.append(
            f"- Steering controls: {cs.get('at_least')} of {cs.get('count')} matched random steering controls moved as much "
            f"(p_fw = {cs.get('p_fw')}, passes = {cs.get('passes_alpha')})"
        )
    else:
        lines.append("- Steering controls: not applicable (prompt/decoding-only knob; judged on house bar, frame CI, half replication, and harm bars)")
    bats = summary.get("batteries") or {}
    lines.append(
        f"- Retention batteries: neutral KL = {bats.get('neutral_kl')} (bar <= {bars.neutral_kl_max:g}); "
        f"near-miss |delta| = {bats.get('near_miss_abs')} (bar <= {bars.near_miss_abs_max:g}); "
        f"panel KL = {bats.get('panel_kl')} (bar <= {bars.panel_kl_max:g}), "
        f"panel own drop = {bats.get('panel_own_drop')} (bar <= {bars.panel_own_drop_max:g})"
    )
    if summary.get("reasons"):
        lines.append(f"- Reasons: {'; '.join(str(r) for r in summary['reasons'])}")
    lines.append(
        f"- Cost: {meta.get('forwards')} forward(s), {meta.get('seconds')} s; base verified untouched before and after."
    )
    lines.append(
        "- Note: This reading evaluated a non-weight intervention (prompts, decoding, and/or transient steering) under "
        "the same kept-back frames and calibrated harm bars used for weight edits."
    )
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
    n_target: int = 8,
    n_neutral: int = 4,
    n_near: int = 4,
    n_panel: int = 4,
    rows: Optional[Mapping[str, int]] = None,
    controls_override: Optional[int] = None,
    book_path: Optional[Path] = None,
    alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """Run one registered Phase 6 fresh cheaper-knob claim on `brief`'s kept-back frames and calibrated `bars`."""
    we.assert_base(h.lm, h.hooks_at_load)
    if bars.brief_id != brief.id:
        raise KnobRefused(f"BarSet {bars.short()} is for brief {bars.brief_id[:12]}, not {brief.short()}.")
    bk = Path(book_path).resolve() if book_path is not None else lb.book_path().resolve()
    cal = lcl.calibration_note(bars, bk)
    if cal is None:
        raise KnobRefused(
            f"Bar set {bars.short()} is not from a calibration reading recorded in {bk.name} for brief {brief.name!r}."
        )
    if cal.get("weights_id") and cal.get("weights_id") != h.weights_id:
        raise KnobRefused(
            f"Bar set {bars.short()} was calibrated on weights {str(cal.get('weights_id') or 'none')[:12]}, not {str(h.weights_id or 'none')[:12]}."
        )

    knob = build_knob(h, spec, base_dir=base_dir, allowed_layers=brief.allowed_layers)
    khist = _knob_book_history(brief, bk)
    whist = khist["weight_hist"]
    if whist.get("other_brief_ids"):
        raise KnobRefused(
            f'Brief "{brief.name}" was already read in this book under another hash; refusing changed brief.'
        )
    if knob.id in khist["knob_ids"]:
        raise KnobRefused(
            f"Knob {knob.short()} has already had a fresh claim reading on brief {brief.name!r} in this book (read once)."
        )
    if khist["total_fresh"] >= int(brief.max_fresh_readings):
        raise KnobRefused(
            f'Brief "{brief.name}" has spent all {brief.max_fresh_readings} fresh reading(s) in this book.'
        )
    # Check kept-back practice leakage in the book
    kb_guard = lbr.KeptBack([brief], batteries=True)
    if bk.exists():
        for entry in lb.entries(bk):
            if entry.get("kind") in ("paired", "written"):
                body = entry.get("body") or {}
                nm = str((body.get("plan") or {}).get("name") or "")
                if not nm.startswith(lcl.RESERVED_NAMES):
                    for it_row in (body.get("items") or []):
                        if isinstance(it_row, Mapping):
                            hit = kb_guard.hit(it_row.get("system"), it_row.get("question"), it_row.get("answer"))
                            if hit is not None:
                                raise KnobRefused(
                                    f"Book line {entry.get('seq')} holds a practice reading of kept-back wording "
                                    f"({hit['frame']}); refusing fresh knob claim in this book."
                                )

    reading_index = khist["total_fresh"] + 1
    r_sm = int((rows or {}).get("small", lv.ROWS_SMALL))
    r_lg = int((rows or {}).get("large", lv.ROWS_LARGE))
    eff_alpha = float(bars.alpha if alpha is None else alpha)
    n_ctl = int(controls_override if controls_override is not None else brief.default_controls) if knob.steering_resolved else 0

    # Pre-check that the prompt/thinking knob can tokenize on a sample practice item before spending a fresh reading
    if knob.prompt is not None or "thinking" in knob.decoding:
        sample_prac = brief.generator.draw("practice", 1, seed_bytes=b"preflight_knob", tokenizer=h.tokenizer)
        if sample_prac:
            _make_knob_item(h, sample_prac[0].item, knob, brief=brief)

    # 1. Register BEFORE drawing fresh kept-back items from os.urandom(32)!
    reg_body = {
        "note": "knob_claim_registration",
        "brief_id": brief.id,
        "brief_name": brief.name,
        "bars_sha256": bars.id,
        "knob_id": knob.id,
        "knob": knob.to_dict(),
        "reading_index": reading_index,
        "max_fresh_readings": int(brief.max_fresh_readings),
        "practice_count": int(practice_count),
        "practice_best": _clean_float(practice_best),
        "forecast": dict(forecast) if isinstance(forecast, Mapping) else None,
        "design": {
            "n_target": int(n_target),
            "n_neutral": int(n_neutral),
            "n_near": int(n_near),
            "n_panel": int(n_panel),
            "rows": {"small": r_sm, "large": r_lg},
            "steering_controls": n_ctl,
            "alpha": eff_alpha,
        },
        "model_id": h.model_id,
        "weights_id": h.weights_id,
    }
    reg_entry = lb.append("note", reg_body, path=bk)

    t0 = time.perf_counter()
    total_fwds = 0
    void_reasons: List[str] = []
    reasons: List[str] = []
    sgn = 1.0 if brief.direction == 0 else float(brief.direction)

    try:
        # 2. Post-freeze OS-random draw of kept-back items and batteries
        seed_bytes = os.urandom(32)
        ctl_seed = int.from_bytes(seed_bytes[:8], "little") & ((1 << 31) - 1)
        sys_ovr = (brief.chat_settings or {}).get("system")
        target_drawn = brief.generator.draw(
            "kept_back", int(n_target), seed_bytes=seed_bytes + b":target",
            tokenizer=h.tokenizer, system_override=sys_ovr, name_prefix=f"kcl_{reading_index}_tgt",
        )
        near_drawn = (
            brief.near_miss_generator.draw(
                "kept_back", int(n_near), seed_bytes=seed_bytes + b":near",
                tokenizer=h.tokenizer, system_override=sys_ovr, name_prefix=f"kcl_{reading_index}_nm",
            )
            if brief.near_miss_generator is not None else []
        )
        neu_drawn = lbr.neutral_chat_generator().draw(
            "kept_back", int(n_neutral), seed_bytes=seed_bytes + b":neu",
            tokenizer=h.tokenizer, name_prefix=f"kcl_{reading_index}_neu",
        )
        panel_rows = lbr.draw_labelled_sanity_panel(
            int(n_panel), seed_bytes=seed_bytes + b":panel", tokenizer=h.tokenizer,
        )
        panel_items = [row["item"] for row in panel_rows]

        # Verify zero overlap with any practice-drawn frame templates in this book
        prac_tids = set(whist.get("practice_template_ids") or ())
        for d in target_drawn + near_drawn:
            if d.template_id in prac_tids:
                void_reasons.append(f"drawn frame template {d.template_id[:12]} was seen in practice")

        ctl_list = _draw_steering_controls(h, knob.steering_resolved, n_ctl, seed=ctl_seed) if n_ctl > 0 else []

        # 3. Evaluate target kept-back items
        small_effs: List[float] = []
        large_effs: List[float] = []
        kept_drawn: List[lbr.DrawnItem] = []
        small_floors: List[float] = []
        large_floors: List[float] = []
        ctl_sm_acc: List[List[float]] = [[] for _ in ctl_list]
        ctl_lg_acc: List[List[float]] = [[] for _ in ctl_list]
        unstable_n = 0

        if not void_reasons and brief.metric == "written_loop":
            wres = _eval_written_items(h, [d.item for d in target_drawn], knob, brief, ctl_list)
            total_fwds += int(wres.get("forwards", 0))
            if wres.get("void"):
                void_reasons.append(str(wres["void"]))
            else:
                small_effs = [float(x) for x in wres["item_effects"]]
                large_effs = list(small_effs)
                kept_drawn = list(target_drawn)
                small_floors = [0.0]
                large_floors = [0.0]
                for ci, cm in enumerate(wres.get("control_means") or []):
                    ctl_sm_acc[ci] = [float(cm)]
                    ctl_lg_acc[ci] = [float(cm)]
        elif not void_reasons:
            for d in target_drawn:
                k_it = _make_knob_item(h, d.item, knob, brief=brief)
                r_s = _eval_paired_item_class(h, d.item, k_it, knob, ctl_list, brief.readouts, brief.metric, r_sm)
                total_fwds += int(r_s.get("forwards", 0))
                if r_s.get("void"):
                    void_reasons.append(f"small class: {r_s['void']}")
                    break
                r_l = _eval_paired_item_class(h, d.item, k_it, knob, ctl_list, brief.readouts, brief.metric, r_lg)
                total_fwds += int(r_l.get("forwards", 0))
                if r_l.get("void"):
                    void_reasons.append(f"large class: {r_l['void']}")
                    break
                v_s = _clean_float((r_s.get("scores") or {}).get(brief.metric))
                v_l = _clean_float((r_l.get("scores") or {}).get(brief.metric))
                f_s = float(r_s.get("floor", 0.0))
                f_l = float(r_l.get("floor", 0.0))
                small_floors.append(f_s)
                large_floors.append(f_l)
                if v_s is None or v_l is None:
                    continue
                ok_agree, _ratio = lv.agree(v_s, v_l, max(f_s, f_l), brief.metric in ("odds", "own"))
                if not ok_agree:
                    unstable_n += 1
                    continue
                small_effs.append(v_s)
                large_effs.append(v_l)
                kept_drawn.append(d)
                for ci in range(len(ctl_list)):
                    cs = _clean_float(r_s["control_scores"][ci]) if ci < len(r_s["control_scores"]) else None
                    cl = _clean_float(r_l["control_scores"][ci]) if ci < len(r_l["control_scores"]) else None
                    if cs is not None:
                        ctl_sm_acc[ci].append(cs)
                    if cl is not None:
                        ctl_lg_acc[ci].append(cl)

        # 4. Evaluate retention batteries (exact same three gating batteries as Phase 3!)
        bat_neutral = {"mean": None, "abs_mean": None, "own_mean": None, "n_read": 0}
        bat_near = {"mean": None, "abs_mean": None, "own_mean": None, "n_read": 0}
        bat_panel = {"mean": None, "abs_mean": None, "own_mean": None, "n_read": 0}
        if not void_reasons:
            bat_neutral = _eval_knob_battery(
                h, [d.item for d in neu_drawn], knob, brief, {"kl": True, "own": True, "odds": None}, "kl", r_sm
            )
            total_fwds += int(bat_neutral.get("forwards", 0))
            if bat_neutral.get("void"):
                void_reasons.append(f"neutral battery: {bat_neutral['void']}")
        if not void_reasons and near_drawn:
            nm_metric = brief.metric if brief.metric in ("odds", "own") else "kl"
            nm_ro = brief.readouts if nm_metric in ("odds", "own") else {"kl": True, "own": True, "odds": None}
            bat_near = _eval_knob_battery(
                h, [d.item for d in near_drawn], knob, brief, nm_ro, nm_metric, r_sm
            )
            total_fwds += int(bat_near.get("forwards", 0))
            if bat_near.get("void"):
                void_reasons.append(f"near-miss battery: {bat_near['void']}")
        if not void_reasons:
            bat_panel = _eval_knob_battery(
                h, panel_items, knob, brief, {"kl": True, "own": True, "odds": None}, "kl", r_sm
            )
            total_fwds += int(bat_panel.get("forwards", 0))
            if bat_panel.get("void"):
                void_reasons.append(f"labelled panel battery: {bat_panel['void']}")

    except BaseException as exc:
        we.assert_base(h.lm, h.hooks_at_load)
        lb.append(
            "note",
            {
                "note": "knob_claim_verdict",
                "brief_id": brief.id,
                "brief_name": brief.name,
                "bars_sha256": bars.id,
                "knob_id": knob.id,
                "knob_classes": list(knob.knob_classes),
                "reading_index": reading_index,
                "registration_seq": int(reg_entry["seq"]),
                "verdict": "void",
                "summary": {"verdict": "void", "reasons": [f"error during knob claim: {type(exc).__name__}: {exc}"]},
                "cost": {"forwards": total_fwds, "seconds": round(time.perf_counter() - t0, 3)},
            },
            path=bk,
        )
        raise
    finally:
        we.assert_base(h.lm, h.hooks_at_load)

    elapsed = round(time.perf_counter() - t0, 3)

    # 5. Judge under the exact Phase 3 gate rules
    floor_sm = max(small_floors) if small_floors else None
    floor_lg = max(large_floors) if large_floors else None
    floor_worst = (
        max(f for f in (floor_sm, floor_lg) if f is not None)
        if (floor_sm is not None or floor_lg is not None) else None
    )
    hb = lcl.house_bar(floor_worst, brief.least_effect)

    sm_rows = [{"frame_id": d.frame_id, "half": d.half, "value": v} for d, v in zip(kept_drawn, small_effs)]
    lg_rows = [{"frame_id": d.frame_id, "half": d.half, "value": v} for d, v in zip(kept_drawn, large_effs)]
    f_sm = lcl.frame_statistics(sm_rows, direction=brief.direction)
    f_lg = lcl.frame_statistics(lg_rows, direction=brief.direction)
    sm_mean = _clean_float(f_sm.get("directed_mean"))
    lg_mean = _clean_float(f_lg.get("directed_mean"))
    sm_raw = _clean_float(f_sm.get("mean"))
    lg_raw = _clean_float(f_lg.get("mean"))

    ctl_standing = None
    controls_pass = True
    if ctl_list and sm_raw is not None and lg_raw is not None:
        c_sm = [
            float(np.mean(xs)) if len(xs) == len(kept_drawn) and xs else None
            for xs in ctl_sm_acc
        ]
        c_lg = [
            float(np.mean(xs)) if len(xs) == len(kept_drawn) and xs else None
            for xs in ctl_lg_acc
        ]
        cs_sm = lcl.control_standing(sm_raw, c_sm, max_fresh_readings=brief.max_fresh_readings, alpha=eff_alpha)
        cs_lg = lcl.control_standing(lg_raw, c_lg, max_fresh_readings=brief.max_fresh_readings, alpha=eff_alpha)
        ctl_standing = cs_lg if not cs_lg["passes_alpha"] else cs_sm
        controls_pass = bool(cs_sm["passes_alpha"] and cs_lg["passes_alpha"])

    neu_kl = _clean_float(bat_neutral.get("mean"))
    nm_abs = _clean_float(bat_near.get("abs_mean")) if near_drawn else 0.0
    pan_kl = _clean_float(bat_panel.get("mean"))
    pan_own = _clean_float(bat_panel.get("own_mean"))
    pan_drop = max(0.0, -pan_own) if pan_own is not None else None

    harm_reasons: List[str] = []
    if neu_kl is None or neu_kl > bars.neutral_kl_max:
        harm_reasons.append(f"neutral_kl {neu_kl} > bar {bars.neutral_kl_max:g}")
    if near_drawn and (nm_abs is None or nm_abs > bars.near_miss_abs_max):
        harm_reasons.append(f"near_miss_abs {nm_abs} > bar {bars.near_miss_abs_max:g}")
    if pan_kl is None or pan_kl > bars.panel_kl_max:
        harm_reasons.append(f"panel_kl {pan_kl} > bar {bars.panel_kl_max:g}")
    if pan_drop is None or pan_drop > bars.panel_own_drop_max:
        harm_reasons.append(f"panel_own_drop {pan_drop} > bar {bars.panel_own_drop_max:g}")

    bar_val = hb.get("bar")
    h0_n = int((f_lg.get("half_0") or {}).get("n_frames", 0))
    h1_n = int((f_lg.get("half_1") or {}).get("n_frames", 0))
    too_few_frames = (h0_n < bars.min_frames_per_half or h1_n < bars.min_frames_per_half)

    def _halves_pass(fs: Mapping[str, Any]) -> bool:
        if bar_val is None:
            return False
        h0 = _clean_float((fs.get("half_0") or {}).get("directed_mean"))
        h1 = _clean_float((fs.get("half_1") or {}).get("directed_mean"))
        req = float(bars.half_bar_fraction) * float(bar_val)
        return bool(h0 is not None and h1 is not None and h0 > 0.0 and h1 > 0.0 and h0 >= req and h1 >= req)

    ci_lg = f_lg.get("directed_ci95") or [None, None]
    ci_ok = bool(ci_lg[0] is not None and float(ci_lg[0]) > 0.0)
    replicated = bool(_halves_pass(f_sm) and _halves_pass(f_lg) and ci_ok)

    if void_reasons or not kept_drawn or too_few_frames:
        verdict = "void"
        reasons = void_reasons or (
            [f"too few kept-back frames per half ({h0_n}, {h1_n}; need >= {bars.min_frames_per_half})"]
            if too_few_frames else ["no target items remained stable across batch classes"]
        )
    elif hb.get("floor_too_loud") or bar_val is None:
        verdict = "kept the base"
        reasons = [f"floor {floor_worst} too loud to set house bar (M={brief.least_effect:g})"]
    elif sm_mean is None or lg_mean is None or sm_mean < bar_val or lg_mean < bar_val:
        verdict = "kept the base"
        reasons = [f"directed effect (small={sm_mean}, large={lg_mean}) did not clear house bar {bar_val:.4f}"]
    elif not controls_pass:
        verdict = "no better than control edits"
        reasons = [f"matched random steering controls not cleared (p_fw={ctl_standing.get('p_fw')})"]
    elif not replicated:
        verdict = "did not replicate"
        h0_m = (f_lg.get("half_0") or {}).get("directed_mean")
        h1_m = (f_lg.get("half_1") or {}).get("directed_mean")
        reasons = [f"half-split replication or frame CI failed (half0={h0_m}, half1={h1_m}, ci_ok={ci_ok})"]
    elif harm_reasons:
        verdict = "works but harms"
        reasons = harm_reasons
    else:
        verdict = "lead"
        reasons = []

    is_default_book = (bk == lb.default_book_path().resolve())
    summary = {
        "verdict": verdict,
        "reasons": reasons,
        "n_target_asked": int(n_target),
        "n_target_read": len(kept_drawn),
        "unstable_target": unstable_n,
        "small_directed": round(sm_mean, 5) if sm_mean is not None else None,
        "large_directed": round(lg_mean, 5) if lg_mean is not None else None,
        "floor_small": round(floor_sm, 6) if floor_sm is not None else None,
        "floor_large": round(floor_lg, 6) if floor_lg is not None else None,
        "house_bar": round(bar_val, 5) if bar_val is not None else None,
        "house_bar_dropped": bool(hb.get("floor_too_loud")),
        "replicated": replicated,
        "frame_stats": f_lg,
        "controls_applicable": bool(ctl_list),
        "control_standing": ctl_standing,
        "batteries": {
            "neutral_kl": round(neu_kl, 5) if neu_kl is not None else None,
            "near_miss_abs": round(nm_abs, 5) if nm_abs is not None else None,
            "panel_kl": round(pan_kl, 5) if pan_kl is not None else None,
            "panel_own_drop": round(pan_drop, 5) if pan_drop is not None else None,
            "harm_passed": not harm_reasons,
            "harm_reasons": harm_reasons,
        },
    }
    meta = {
        "reading_index": reading_index,
        "book_name": bk.name,
        "is_default_book": is_default_book,
        "forwards": total_fwds,
        "seconds": elapsed,
    }
    sheet = render_knob_sheet(brief, bars, knob, verdict, summary, meta)

    verdict_body = {
        "note": "knob_claim_verdict",
        "brief_id": brief.id,
        "brief_name": brief.name,
        "bars_sha256": bars.id,
        "knob_id": knob.id,
        "knob_classes": list(knob.knob_classes),
        "reading_index": reading_index,
        "registration_seq": int(reg_entry["seq"]),
        "verdict": verdict,
        "summary": summary,
        "cost": {"forwards": total_fwds, "seconds": elapsed},
    }
    verdict_entry = lb.append("note", verdict_body, path=bk)

    driver_record = {
        "brief_id": brief.id,
        "brief_name": brief.name,
        "knob_id": knob.id,
        "knob_classes": list(knob.knob_classes),
        "reading_index": reading_index,
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
        "sheet": sheet,
        "driver_record": driver_record,
        "cost": {"forwards": total_fwds, "seconds": elapsed},
        "book": driver_record["book"],
    }
