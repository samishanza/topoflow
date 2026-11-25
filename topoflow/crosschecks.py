"""Deterministic cross-checks between independent sources. Each passes, fails, or says it could not run, with numbers."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import grid
from .contract import CrossCheck, RunRequest
from .units import area_weighted_mean


def _nearest(lats, lons, lat, lon):
    return int(np.abs(lats - lat).argmin()), int(np.abs(lons - lon).argmin())


def era5_vs_power(req: RunRequest, known: dict[str, Any]) -> CrossCheck:
    name = "ERA5 wind speed vs NASA POWER (MERRA-2), daily means at sample cells"
    u, v, pts = known.get("wind.u10"), known.get("wind.v10"), known.get("wind.power_check")
    if not (isinstance(u, np.ndarray) and isinstance(v, np.ndarray) and pts):
        return CrossCheck(name=name, status="unknown", detail="ERA5 wind or POWER points missing")
    lats, lons = grid.axes(req)
    times = pd.DatetimeIndex(grid.times(req))
    speed = np.hypot(u, v)
    pairs = []
    for f in pts:
        i, j = _nearest(lats, lons, f["lat"], f["lon"])
        for day, ws in f["properties"]["daily_ws10m"].items():
            d = pd.Timestamp(day)
            sel = times.normalize() == d
            if sel.any() and np.isfinite(speed[sel, i, j]).all():
                pairs.append((float(speed[sel, i, j].mean()), float(ws)))
    if len(pairs) < 5:
        return CrossCheck(name=name, status="unknown", detail=f"only {len(pairs)} matched day-points")
    a, b = np.array(pairs).T
    mad = float(np.median(np.abs(a - b)))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    ok = mad <= 1.5 and (np.isnan(corr) or corr >= 0.5)
    return CrossCheck(
        name=name, status="pass" if ok else "fail",
        detail=(f"median |ERA5 - POWER| = {mad:.2f} m/s, r = {corr:.2f} over {len(pairs)} day-points. "
                "ERA5 daily mean is built from the run's sampled hours only."),
        numbers={"median_abs_diff_ms": round(mad, 3), "correlation": round(corr, 3), "n": len(pairs),
                 "era5_mean": round(float(a.mean()), 2), "power_mean": round(float(b.mean()), 2)},
    )


def ch4_vs_noaa(req: RunRequest, known: dict[str, Any]) -> CrossCheck:
    name = "Surface CH4 regional mean vs NOAA global marine background"
    ch4, bg = known.get("ch4.surface_ppb"), known.get("ch4.global_mean_ppb")
    if not (isinstance(ch4, np.ndarray) and isinstance(bg, dict)):
        return CrossCheck(name=name, status="unknown", detail="surface CH4 or the NOAA background is missing")
    lats, _ = grid.axes(req)
    mean = area_weighted_mean(np.nanmean(ch4, axis=0), lats)
    ratio = mean / bg["ppb"]
    ok = 0.95 <= ratio <= 1.35
    return CrossCheck(
        name=name, status="pass" if ok else "fail",
        detail=(f"regional mean {mean:.0f} ppb vs background {bg['ppb']:.0f} ppb ({ratio:.3f}x). "
                "Continental and source regions sit above the marine background; far below it suggests a unit or conversion fault."),
        numbers={"regional_mean_ppb": round(mean, 1), "background_ppb": bg["ppb"], "ratio": round(ratio, 4)},
    )


def plume_wind_vs_era5(req: RunRequest, known: dict[str, Any]) -> CrossCheck:
    name = "Carbon Mapper plume wind vs ERA5 at the plume"
    u, v, plumes = known.get("wind.u10"), known.get("wind.v10"), known.get("plumes.detections")
    if not (isinstance(u, np.ndarray) and plumes):
        return CrossCheck(name=name, status="unknown", detail="no plumes or no ERA5 wind")
    lats, lons = grid.axes(req)
    times = pd.DatetimeIndex(grid.times(req))
    speed = np.hypot(u, v)
    pairs = []
    for f in plumes:
        p = f["properties"]
        if p.get("plume_wind_speed") is None or not p.get("time"):
            continue
        t = pd.Timestamp(p["time"]).tz_localize(None) if pd.Timestamp(p["time"]).tzinfo else pd.Timestamp(p["time"])
        k = int(np.abs((times - t).total_seconds()).argmin())
        if abs((times[k] - t).total_seconds()) > 3 * 3600:
            continue
        i, j = _nearest(lats, lons, f["lat"], f["lon"])
        if np.isfinite(speed[k, i, j]):
            pairs.append((float(speed[k, i, j]), float(p["plume_wind_speed"])))
    if len(pairs) < 3:
        return CrossCheck(name=name, status="unknown",
                          detail=f"{len(pairs)} plumes fall within 3 h of a run time; add hours with --hours to match more")
    a, b = np.array(pairs).T
    mad = float(np.median(np.abs(a - b)))
    caveat = (" Agreement this close suggests Carbon Mapper's plume wind is itself drawn from a weather model, so this"
              " confirms the sampling and timing, not the wind; check Carbon Mapper's methods before counting it as independent."
              if mad < 0.5 else "")
    return CrossCheck(name=name, status="pass" if mad <= 2.0 else "fail",
                      detail=f"median |ERA5 - plume wind| = {mad:.2f} m/s over {len(pairs)} plumes.{caveat}",
                      numbers={"median_abs_diff_ms": round(mad, 3), "n": len(pairs)})


def emitters_vs_edgar(req: RunRequest, known: dict[str, Any]) -> CrossCheck:
    name = "Named emitters vs EDGAR hotspots (co-location)"
    flux = known.get("ch4.emission_flux")
    pts = (known.get("emitters.assets") or []) + (known.get("emitters.facilities") or [])
    if not (isinstance(flux, np.ndarray) and pts):
        return CrossCheck(name=name, status="unknown", detail="EDGAR grid or named emitters missing")
    lats, lons = grid.axes(req)
    thresh = float(np.nanquantile(flux, 0.75))
    hits = sum(1 for f in pts if flux[_nearest(lats, lons, f["lat"], f["lon"])] >= thresh)
    share = hits / len(pts)
    return CrossCheck(name=name, status="pass" if share >= 0.5 else "fail",
                      detail=(f"{share:.0%} of {len(pts)} named emitters sit in the top quarter of EDGAR cells"
                              + (". On a grid this coarse the top quarter is most land, so co-location is easy to pass"
                                 if req.resolution_deg >= 2 else "")),
                      numbers={"share_in_top_quartile": round(share, 3), "n": len(pts)})


def topology_ch4_association(req: RunRequest, known: dict[str, Any]) -> CrossCheck:
    """Exploratory, never pass or fail: does surface CH4 run higher at convergent critical points?"""
    name = "CH4 anomaly at critical points, by kind (exploratory)"
    cps = known.get("topology.critical_points")
    if not cps:
        return CrossCheck(name=name, status="unknown", detail="no critical points")
    by: dict[str, list[float]] = {}
    for f in cps:
        a = f["properties"].get("ch4_anomaly_ppb")
        if a is not None:
            by.setdefault(f["properties"]["kind"], []).append(a)
    if not by:
        return CrossCheck(name=name, status="unknown", detail="no CH4 at the critical points")
    numbers = {f"{k}_mean_anomaly_ppb": round(float(np.mean(v)), 2) for k, v in by.items()}
    numbers.update({f"{k}_n": len(v) for k, v in by.items()})
    detail = "; ".join(f"{k}: {np.mean(v):+.1f} ppb (n={len(v)})" for k, v in sorted(by.items()))
    return CrossCheck(name=name, status="unknown",
                      detail=f"{detail}. Descriptive only: a short window on a coarse grid tests nothing.", numbers=numbers)


ALL = (era5_vs_power, ch4_vs_noaa, plume_wind_vs_era5, emitters_vs_edgar, topology_ch4_association)


def run_all(req: RunRequest, known: dict[str, Any]) -> list[CrossCheck]:
    out = []
    for fn in ALL:
        try:
            out.append(fn(req, known))
        except Exception as exc:  # noqa: BLE001 - a broken check reports itself, it never sinks the run
            out.append(CrossCheck(name=fn.__name__, status="unknown", detail=f"check failed: {type(exc).__name__}: {exc}"))
    return out
