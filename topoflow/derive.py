"""Deterministic derivations and reconciliation. Code decides here; agents only proposed.

Reconcile: among admitted candidates for a field, pick by the catalog's source preference, then confidence.
Two candidates that disagree are both kept, the field is marked conflicting, and the numbers are recorded.
Derive: compute what no source serves (CH4 mole fraction, divergence, critical points) from admitted fields only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import grid, topology
from .agents.base import Emit, AgentContext
from .catalog import FIELDS
from .contract import Candidate, FieldRecord, RunRequest, grid_summary
from .gatekeeper import Accepted
from .units import mass_to_mole_fraction_ppb

# A second grid candidate is "conflicting" when its RMS difference from the chosen one exceeds this share of the
# chosen field's spread. Below it, the two are the same field within sampling and conversion noise.
CONFLICT_RELATIVE_RMS = 0.5


def _source(a: Accepted) -> str:
    return a.proposal.evidence.source_id if a.proposal.evidence else "derived"


def _summary(a: Accepted) -> dict:
    v = a.value
    if isinstance(v, np.ndarray):
        return grid_summary(v)
    if isinstance(v, list):
        return {"features": len(v)}
    if isinstance(v, dict):
        return {k: x for k, x in v.items() if isinstance(x, (int, float, str)) or x is None}
    return {}


def reconcile(accepted: list[Accepted]) -> tuple[dict[str, Any], dict[str, FieldRecord], dict[str, list[Accepted]]]:
    by_field: dict[str, list[Accepted]] = {}
    for a in accepted:
        by_field.setdefault(a.proposal.field, []).append(a)

    selected: dict[str, Any] = {}
    records: dict[str, FieldRecord] = {}
    for key, cands in by_field.items():
        spec = FIELDS[key]
        pref = list(spec.preference)

        def rank(a: Accepted):
            s = _source(a)
            return (pref.index(s) if s in pref else len(pref), -a.proposal.confidence)

        cands.sort(key=rank)
        chosen = cands[0]
        selected[key] = chosen.value
        state = "derived" if chosen.proposal.derived else "observed"
        conflict = None
        if len(cands) > 1 and isinstance(chosen.value, np.ndarray):
            for other in cands[1:]:
                if not isinstance(other.value, np.ndarray) or other.value.shape != chosen.value.shape:
                    continue
                both = np.isfinite(chosen.value) & np.isfinite(other.value)
                if both.sum() < 10:
                    continue
                d = (other.value - chosen.value)[both]
                rms = float(np.sqrt(np.mean(d**2)))
                spread = float(np.std(chosen.value[both])) or 1.0
                c = {"against": _source(other), "rms_difference": round(rms, 4), "mean_bias": round(float(d.mean()), 4),
                     "relative_rms": round(rms / spread, 3), "cells_compared": int(both.sum())}
                if rms / spread > CONFLICT_RELATIVE_RMS:
                    state, conflict = "conflicting", c
                    break
                conflict = conflict or {**c, "verdict": "agree"}
        records[key] = FieldRecord(
            field=key, state=state, unit=spec.unit, selected=_source(chosen),
            candidates=[Candidate(source_id=_source(a), agent_id=a.proposal.agent_id, confidence=a.proposal.confidence,
                                  data_origin=a.proposal.evidence.data_origin if a.proposal.evidence else "real",
                                  summary=_summary(a)) for a in cands],
            conflict=conflict,
            note=chosen.proposal.rationale,
        )
    return selected, records, by_field


def derive(req: RunRequest, known: dict[str, Any], ctx: AgentContext) -> Emit:
    """Proposals from the 'derive' identity. They still go through the gate like anything else."""
    e = Emit("derive", ctx)
    lats, lons = grid.axes(req)
    times = grid.times(req)

    mass, sp, t2m = known.get("ch4.surface_mass"), known.get("met.sp"), known.get("met.t2m")
    if all(isinstance(x, np.ndarray) for x in (mass, sp, t2m)):
        ppb = mass_to_mole_fraction_ppb(mass, sp, t2m).astype("float32")
        e.propose("ch4.surface_ppb", ppb, None, "ppb", confidence=0.7, derived=True,
                  derived_from=["ch4.surface_mass", "met.sp", "met.t2m"],
                  rationale="x = (c/M) / (p/RT) with ERA5 surface pressure and 2 m temperature at each cell and time. "
                            "Ignores water vapour, so it reads up to a few percent low in humid air.")

    u, v = known.get("wind.u10"), known.get("wind.v10")
    if isinstance(u, np.ndarray) and isinstance(v, np.ndarray):
        div, vort = topology.div_vort(u, v, lats, lons)
        e.propose("flow.divergence", div, None, "s-1", confidence=0.8, derived=True, derived_from=["wind.u10", "wind.v10"],
                  rationale="Centred differences on the sphere at the run resolution. Coarse grids smooth out small convergence lines.")
        e.propose("flow.vorticity", vort, None, "s-1", confidence=0.8, derived=True, derived_from=["wind.u10", "wind.v10"],
                  rationale="Centred differences on the sphere at the run resolution.")

        ch4 = known.get("ch4.surface_ppb")
        if not isinstance(ch4, np.ndarray) and all(isinstance(x, np.ndarray) for x in (mass, sp, t2m)):
            ch4 = mass_to_mole_fraction_ppb(mass, sp, t2m)
        blh = known.get("met.blh")
        feats = []
        for ti, t in enumerate(times):
            ch4_t = ch4[ti] if isinstance(ch4, np.ndarray) else None
            regional = float(np.nanmean(ch4_t)) if ch4_t is not None and np.isfinite(ch4_t).any() else None
            for cp in topology.critical_points(u[ti], v[ti], lats, lons):
                i = int(np.abs(lats - cp["lat"]).argmin())
                j = int(np.abs(lons - cp["lon"]).argmin())
                props = {**cp, "family": topology.family(cp["kind"]), "time": str(pd.Timestamp(t)), "t_index": ti}
                if ch4_t is not None and np.isfinite(ch4_t[i, j]):
                    props["ch4_ppb"] = round(float(ch4_t[i, j]), 1)
                    props["ch4_anomaly_ppb"] = round(float(ch4_t[i, j]) - regional, 1) if regional else None
                if isinstance(blh, np.ndarray) and np.isfinite(blh[ti, i, j]):
                    props["blh_m"] = round(float(blh[ti, i, j]), 0)
                feats.append({"lon": cp["lon"], "lat": cp["lat"], "properties": props})
        e.propose("topology.critical_points", feats or None, None, "1", confidence=0.6, derived=True,
                  derived_from=["wind.u10", "wind.v10"],
                  rationale="Zeros of the piecewise-linear 10 m wind on each grid triangle, classified by the Jacobian. "
                            "Inferred from a sampled grid: a point can move or vanish at a finer resolution.")
        if not feats:
            e.note("no critical points in this region and window")
    return e
