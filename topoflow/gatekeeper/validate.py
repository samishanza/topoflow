"""The gatekeeper is synchronous and deterministic. Nothing reaches the record without passing it.

It does not judge whether a value is *right*; that is the reconciler's and the cross-checks' job.
It judges whether a value is *admissible*: sourced, in scope, in units, the right shape, and physically possible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import grid
from ..catalog import FieldSpec, is_agent_allowed, spec_for
from ..contract import Proposal, Rejection, RunRequest
from ..units import area_weighted_mean, convert, mass_to_mole_fraction_ppb, normalise_unit

# For point fields: the property that carries the value checked against the catalog bounds.
POINT_VALUE = {
    "emitters.assets": "t_ch4_per_year",
    "emitters.facilities": "t_co2e_ch4_per_year",
    "plumes.detections": "kg_per_hour",
}
# How far outside the run box a point may sit (degrees) before it is refused as belonging elsewhere.
REGION_MARGIN_DEG = 1.0
# Regional mean surface CH4 against the global marine background. Basins and wetlands run well above it,
# but not threefold over a whole region; a doctored field usually does.
CH4_REGIONAL_RATIO = (0.85, 1.6)


@dataclass
class Accepted:
    proposal: Proposal
    value: Any  # after unit conversion
    unit: str
    converted: bool


@dataclass
class GateResult:
    accepted: list[Accepted]
    rejected: list[Rejection]


def _reject(p: Proposal, codes: list[str], reasons: list[str]) -> Rejection:
    return Rejection(agent_id=p.agent_id, field=p.field, source_id=p.evidence.source_id if p.evidence else None,
                     codes=codes, reasons=reasons)


def _check_grid(p: Proposal, spec: FieldSpec, value: np.ndarray, req: RunRequest, codes, reasons):
    lats, lons = grid.axes(req)
    want = (len(lats), len(lons)) if spec.static else (len(grid.times(req)), len(lats), len(lons))
    if not isinstance(value, np.ndarray) or value.shape != want:
        codes.append("shape_mismatch")
        reasons.append(f"expected an array of shape {want}, got {getattr(value, 'shape', type(value).__name__)}")
        return
    finite = np.isfinite(value)
    nan_frac = 1 - finite.mean()
    if nan_frac > spec.max_nan_fraction:
        codes.append("too_patchy")
        reasons.append(f"{nan_frac:.1%} of cells are missing; at most {spec.max_nan_fraction:.0%} allowed for {spec.key}")
    if spec.min is not None and finite.any():
        v = value[finite]
        bad = ((v < spec.min) | (v > spec.max)).mean()
        if bad > spec.max_out_of_bounds_fraction:
            codes.append("out_of_bounds")
            reasons.append(f"{bad:.2%} of values fall outside the physical range {spec.min}..{spec.max} {spec.unit} "
                           f"(observed {float(v.min()):.4g}..{float(v.max()):.4g})")


def _check_points(p: Proposal, spec: FieldSpec, value: Any, req: RunRequest, codes, reasons):
    if not isinstance(value, list) or not all(isinstance(f, dict) and "lon" in f and "lat" in f for f in value):
        codes.append("shape_mismatch")
        reasons.append("expected a list of features with lon and lat")
        return
    if not value:
        codes.append("empty")
        reasons.append("no features")
        return
    w, s, e, n = req.bbox
    m = REGION_MARGIN_DEG
    if spec.key != "static.coastline":
        outside = [f for f in value if not (w - m <= f["lon"] <= e + m and s - m <= f["lat"] <= n + m)]
        if len(outside) > 0.1 * len(value):
            codes.append("outside_region")
            reasons.append(f"{len(outside)} of {len(value)} features lie outside the run box")
    prop = POINT_VALUE.get(spec.key)
    if prop and spec.min is not None:
        vals = [f["properties"].get(prop) for f in value]
        bad = [x for x in vals if x is None or not np.isfinite(x) or x < spec.min or x > spec.max]
        if len(bad) > 0.01 * len(vals):
            codes.append("out_of_bounds")
            reasons.append(f"{len(bad)} of {len(vals)} features have {prop} missing or outside {spec.min}..{spec.max}")
    if spec.key == "wind.power_check":
        allv = [x for f in value for x in f["properties"].get("daily_ws10m", {}).values()]
        if not allv or min(allv) < 0 or max(allv) > spec.max:
            codes.append("out_of_bounds")
            reasons.append("daily wind speeds missing or outside 0..75 m s-1")


def _check_scalar(p: Proposal, spec: FieldSpec, value: Any, codes, reasons):
    x = value.get("ppb") if isinstance(value, dict) else value
    if not isinstance(x, (int, float)) or not (spec.min <= x <= spec.max):
        codes.append("out_of_bounds")
        reasons.append(f"{x!r} is outside {spec.min}..{spec.max} {spec.unit}")


def cross_field(p: Proposal, value: Any, req: RunRequest, known: dict[str, Any]) -> list[str]:
    """Checks that need more than one field. A number can be in range on its own and still impossible in context."""
    reasons: list[str] = []
    bg = known.get("ch4.global_mean_ppb")
    bg_ppb = bg.get("ppb") if isinstance(bg, dict) else None
    lats, _ = grid.axes(req)
    if bg_ppb and p.field in ("ch4.surface_ppb", "ch4.surface_mass") and isinstance(value, np.ndarray):
        field = value
        if p.field == "ch4.surface_mass":
            sp, t2m = known.get("met.sp"), known.get("met.t2m")
            if isinstance(sp, np.ndarray) and isinstance(t2m, np.ndarray) and sp.shape == value.shape:
                field = mass_to_mole_fraction_ppb(value, sp, t2m)
            else:
                field = mass_to_mole_fraction_ppb(value, 101_325.0, 288.15)
        mean = area_weighted_mean(np.nanmean(field, axis=0) if field.ndim == 3 else field, lats)
        lo, hi = CH4_REGIONAL_RATIO
        if np.isfinite(mean) and not (lo * bg_ppb <= mean <= hi * bg_ppb):
            reasons.append(f"regional mean {mean:.0f} ppb is {mean / bg_ppb:.2f}x the NOAA global background "
                           f"{bg_ppb:.0f} ppb; allowed {lo}-{hi}x")
    if p.field in ("wind.u10", "wind.v10") and isinstance(value, np.ndarray):
        other = known.get("wind.v10" if p.field == "wind.u10" else "wind.u10")
        if isinstance(other, np.ndarray) and other.shape == value.shape:
            speed = np.hypot(value, other)
            if np.nanmean(speed) > 25:
                reasons.append(f"mean 10 m wind speed {np.nanmean(speed):.1f} m s-1 over the whole region and window "
                               "is beyond any observed climatology")
    return reasons


def gate(proposals: list[Proposal], req: RunRequest, known: dict[str, Any]) -> GateResult:
    accepted, rejected = [], []
    # Reference values (background CH4, pressure, temperature) are judged first, so the fields checked against them
    # see them however the agent ordered its proposals.
    order = {"scalar": 0, "grid": 1, "points": 2}
    reference = {"met.sp", "met.t2m", "wind.u10"}
    proposals = sorted(proposals, key=lambda q: (order.get(q.shape, 3), q.field not in reference))
    for p in proposals:
        codes: list[str] = []
        reasons: list[str] = []
        spec = spec_for(p.field)
        if spec is None:
            rejected.append(_reject(p, ["unknown_field"], [f"{p.field} is not in the catalog"]))
            continue
        if not is_agent_allowed(p.agent_id, p.field):
            codes.append("agent_not_permitted")
            reasons.append(f"{p.agent_id} may not propose {p.field}")
        if p.derived:
            if not p.derived_from or not all(k in known for k in p.derived_from):
                codes.append("derived_requires_lineage")
                reasons.append(f"derived from {p.derived_from or 'nothing named'}; every input must already be admitted")
        elif p.evidence is None or not p.evidence.refs:
            codes.append("missing_evidence")
            reasons.append("no source document")
        if not 0 <= p.confidence <= 1:
            codes.append("bad_confidence")
            reasons.append(f"confidence {p.confidence} is outside 0..1")
        if len(p.rationale.strip()) < 10:
            codes.append("missing_rationale")
            reasons.append("say why this value is what it is")

        value, unit, converted = p.value, normalise_unit(p.unit), False
        if unit != spec.unit:
            try:
                value = convert(value, unit or "", spec.unit)
                if isinstance(value, np.ndarray):
                    value = value.astype("float32")
                unit, converted = spec.unit, True
            except ValueError as exc:
                codes.append("unit_error")
                reasons.append(str(exc) if unit else f"no unit given; the catalog needs {spec.unit}")

        if "unit_error" not in codes:
            if spec.shape == "grid":
                _check_grid(p, spec, value, req, codes, reasons)
            elif spec.shape == "points":
                _check_points(p, spec, value, req, codes, reasons)
            else:
                _check_scalar(p, spec, value, codes, reasons)
        if not codes:
            xf = cross_field(p, value, req, known)
            if xf:
                codes.append("cross_field_implausible")
                reasons.extend(xf)

        if codes:
            rejected.append(_reject(p, codes, reasons))
        else:
            accepted.append(Accepted(p, value, unit or spec.unit, converted))
            # Later proposals in the same batch are checked against what was just admitted (v against u, say).
            # First admitted wins here only for cross-checking; the reconciler makes the real choice.
            known.setdefault(p.field, value)
    return GateResult(accepted, rejected)
