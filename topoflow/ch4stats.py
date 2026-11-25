"""Is methane higher where surface flow stagnates? Statistics designed before looking at the CH4 data (decision D12).

Inputs (all daily, on the same 1-degree grid):
  stagnation  bool (day, lat, lon): a persistent stagnation zone, from the climatology (see persistent_zones)
  ch4         float (day, lat, lon): XCH4 (TROPOMI) or surface/column CH4 (CAMS), NaN where not retrieved

Design:
  1. Anomaly. CH4 minus the same day's median over its 10-degree latitude band (land and ocean separately when a land
     mask is given). This removes the seasonal cycle, the latitudinal gradient and the long-term trend, which would
     otherwise make any spatial pattern look significant.
  2. Composite. Mean anomaly inside stagnation zones minus mean anomaly outside them, same days, same bands.
  3. Significance. A moving-block bootstrap over days (blocks of 7 days keep the autocorrelation of weather regimes)
     under the null of no association: zone masks are re-paired with CH4 from a randomly shifted block of days.
     Regional p-values are corrected with Benjamini-Hochberg.
  4. Lag. Correlation between the daily share of a region in stagnation and the region's mean anomaly k days later,
     k = 0..3, with the same block bootstrap for confidence intervals.

Nothing here assumes the answer. If the effect is absent the numbers say so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def persistent_zones(occupancy_steps: np.ndarray, min_steps: int = 4) -> np.ndarray:
    """Daily mask of cells where stagnation lasted at least min_steps consecutive 6-hourly steps (4 = 24 h).

    occupancy_steps: bool (step, lat, lon) at 6-hourly resolution, a cell marked when a point lies within one cell.
    Returns bool (day, lat, lon): True if a qualifying run overlaps that day.
    """
    n, ny, nx = occupancy_steps.shape
    run = np.zeros((ny, nx), dtype=np.int32)
    persistent = np.zeros_like(occupancy_steps)
    for k in range(n):
        run = np.where(occupancy_steps[k], run + 1, 0)
        hit = run >= min_steps
        if hit.any():
            # Mark the whole run back to its start.
            for back in range(min_steps):
                persistent[k - back][hit] = True
    days = n // 4
    return persistent[: days * 4].reshape(days, 4, ny, nx).any(1)


def band_anomaly(ch4: np.ndarray, lats: np.ndarray, band_deg: float = 10.0,
                 land: np.ndarray | None = None) -> np.ndarray:
    out = np.full_like(ch4, np.nan, dtype="float64")
    edges = np.arange(-90, 90 + band_deg, band_deg)
    surfaces = [np.ones(ch4.shape[1:], bool)] if land is None else [land > 0.5, land <= 0.5]
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = (lats >= lo) & (lats < hi)
        for surf in surfaces:
            m = rows[:, None] & surf
            if not m.any():
                continue
            vals = ch4[:, m]
            med = np.nanmedian(vals, axis=1, keepdims=True)
            out[:, m] = vals - med
    return out


@dataclass
class Composite:
    inside_mean: float
    outside_mean: float
    difference: float
    n_inside: int
    n_outside: int
    p_value: float
    ci95: tuple[float, float]


def _diff(anom: np.ndarray, zones: np.ndarray, weights: np.ndarray) -> tuple[float, float, int, int]:
    ok = np.isfinite(anom)
    w = np.broadcast_to(weights, anom.shape)
    zin, zout = ok & zones, ok & ~zones
    if zin.sum() == 0 or zout.sum() == 0:
        return np.nan, np.nan, int(zin.sum()), int(zout.sum())
    a = float((anom[zin] * w[zin]).sum() / w[zin].sum())
    b = float((anom[zout] * w[zout]).sum() / w[zout].sum())
    return a, b, int(zin.sum()), int(zout.sum())


def composite(anom: np.ndarray, zones: np.ndarray, lats: np.ndarray, n_boot: int = 1000, block: int = 7,
              seed: int = 0) -> Composite:
    """Inside-minus-outside anomaly, with a block-bootstrap null and a bootstrap CI."""
    weights = np.cos(np.deg2rad(lats))[None, :, None]
    a, b, nin, nout = _diff(anom, zones, weights)
    obs = a - b
    rng = np.random.default_rng(seed)
    days = anom.shape[0]
    null, boot = [], []
    nblocks = max(1, days // block)
    for _ in range(n_boot):
        # Null: pair each block of zone masks with CH4 from a random other block (breaks the association,
        # keeps both fields' own autocorrelation and seasonality within blocks).
        starts = rng.integers(0, days - block + 1, nblocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:days]
        shuffled = anom[idx[: len(idx)]]
        z = zones[: len(idx)]
        x, y, _, _ = _diff(shuffled, z, weights)
        null.append(x - y)
        # CI: resample blocks of paired days.
        bi = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, days - block + 1, nblocks)])[:days]
        x, y, _, _ = _diff(anom[bi], zones[bi], weights)
        boot.append(x - y)
    null = np.array([v for v in null if np.isfinite(v)])
    boot = np.array([v for v in boot if np.isfinite(v)])
    p = float((np.abs(null) >= abs(obs)).mean()) if len(null) and np.isfinite(obs) else np.nan
    p = max(p, 1 / (len(null) + 1)) if np.isfinite(p) else p
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if len(boot) else (np.nan, np.nan)
    return Composite(a, b, obs, nin, nout, p, ci)


def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Boolean mask of discoveries at false discovery rate q."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.zeros_like(p, dtype=bool)
    ps = np.sort(p[ok])
    m = len(ps)
    if m == 0:
        return out
    thresh = q * np.arange(1, m + 1) / m
    below = ps <= thresh
    if not below.any():
        return out
    cut = ps[np.nonzero(below)[0].max()]
    out[ok] = p[ok] <= cut
    return out


def lag_correlation(zone_share: np.ndarray, anom_mean: np.ndarray, max_lag: int = 3, n_boot: int = 1000,
                    block: int = 7, seed: int = 0) -> list[dict]:
    """Correlation of a region's daily stagnation share with its CH4 anomaly k days later, with block-bootstrap CIs."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(max_lag + 1):
        x = zone_share[: len(zone_share) - k] if k else zone_share
        y = anom_mean[k:]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 3 * block or x.std() == 0 or y.std() == 0:
            out.append({"lag_days": k, "r": np.nan, "ci95": [np.nan, np.nan], "n": int(len(x))})
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        boots = []
        nb = len(x) // block
        for _ in range(n_boot):
            idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, len(x) - block + 1, nb)])
            if x[idx].std() > 0 and y[idx].std() > 0:
                boots.append(np.corrcoef(x[idx], y[idx])[0, 1])
        out.append({"lag_days": k, "r": round(r, 4), "n": int(len(x)),
                    "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]})
    return out
