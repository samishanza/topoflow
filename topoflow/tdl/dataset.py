"""Cochains on the icosphere from the processed climatology, one 6-hourly step at a time.

Per step:
  edge features (E, 2):    wind flowing along each edge (m/s, signed by edge orientation), and wind speed
  vertex features (V, 9):  orography (km), sub-grid orography std (km), land fraction, sin(lat),
                           log boundary-layer height, and four "now" flags within the radius:
                           any stagnation point, attracting point, saddle, point persisting >= 24 h
  target (V,):             a stagnation point persisting >= 24 h within the radius, 24 h later

Steps whose target time falls outside the period are dropped. Cached as float16 under data/tdl/cache/<period>.npz.
"""
from __future__ import annotations

from pathlib import Path

import numcodecs
import numpy as np
import pandas as pd
import requests
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from .complex import Complex, edge_frames

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "tdl" / "cache"
ARCO = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/"
VERTEX_FEATURES = ["orography_km", "orography_std_km", "land", "sin_lat", "log_blh",
                   "any_now", "attracting_now", "saddle_now", "persistent_now"]
EDGE_FEATURES = ["flow_along_edge", "speed"]
HORIZON_STEPS = 4      # 24 h ahead at 6-hourly steps
PERSIST_H = 24.0


def _arco_static(var: str) -> np.ndarray:
    idx = int((pd.Timestamp("2019-01-01") - pd.Timestamp("1900-01-01")) / pd.Timedelta(hours=1))
    r = requests.get(f"{ARCO}{var}/{idx}.0.0", timeout=120)
    r.raise_for_status()
    return np.frombuffer(numcodecs.Blosc().decode(r.content), dtype="<f4").reshape(721, 1440)


def static_vertex_features(cx: Complex) -> np.ndarray:
    """Orography, its sub-grid std, and land fraction at each vertex (ERA5 invariants via ARCO), plus sin(lat)."""
    path = CACHE / f"static_V{len(cx.xyz)}.npy"
    if path.exists():
        return np.load(path)
    lat = np.arange(90, -90.0001, -0.25)[::-1]
    lon = np.arange(0, 360, 0.25)
    cols = []
    for var, scale in (("geopotential_at_surface", 1 / 9.80665 / 1000), ("standard_deviation_of_orography", 1 / 1000),
                       ("land_sea_mask", 1.0)):
        a = _arco_static(var)[::-1] * scale
        a = np.concatenate([a, a[:, :1]], axis=1)
        f = RegularGridInterpolator((lat, np.append(lon, 360.0)), a)
        cols.append(f(np.c_[cx.lat, cx.lon % 360]))
    cols.append(np.sin(np.deg2rad(cx.lat)))
    out = np.stack(cols, 1).astype("float32")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


class StepBuilder:
    def __init__(self, cx: Complex, radius_km: float = 200.0, planet_radius_km: float = 6371.0,
                 static: np.ndarray | None = None):
        self.cx = cx
        self.tree = cKDTree(cx.xyz)
        self.chord = 2 * np.sin(radius_km / planet_radius_km / 2)
        self.elat, self.elon, self.etan = edge_frames(cx)
        self.static = static

    def _interp(self, field: np.ndarray, lat: np.ndarray, lon: np.ndarray, qlat, qlon):
        order = np.argsort(lat)
        lat, field = lat[order], field[order]
        lon_w = np.append(lon, lon[0] + 360)
        field = np.concatenate([field, field[:, :1]], axis=1)
        f = RegularGridInterpolator((lat, lon_w), field, bounds_error=False, fill_value=None)
        q = ((qlon - lon[0]) % 360) + lon[0]
        return f(np.c_[np.clip(qlat, lat[0], lat[-1]), q])

    def flags(self, pts: pd.DataFrame) -> np.ndarray:
        """(V, 4): any, attracting, saddle, persistent (>= 24 h) stagnation point within the radius."""
        out = np.zeros((len(self.cx.xyz), 4), bool)
        if pts.empty:
            return out
        la, lo = np.deg2rad(pts["lat"].values), np.deg2rad(pts["lon"].values)
        xyz = np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
        masks = [np.ones(len(pts), bool), pts["kind"].str.startswith("attracting").values,
                 (pts["kind"] == "saddle").values, (pts["age_h"] >= PERSIST_H).values]
        hits = self.tree.query_ball_point(xyz, self.chord)
        for p, near in enumerate(hits):
            for c, m in enumerate(masks):
                if m[p]:
                    out[near, c] = True
        return out

    def period(self, period_dir: Path) -> dict[str, np.ndarray]:
        ds = xr.open_dataset(period_dir / "display.nc")
        tdim = "valid_time" if "valid_time" in ds.dims else "time"
        times = ds[tdim].values
        lat, lon = ds["latitude"].values, ds["longitude"].values
        pts = pd.read_parquet(period_dir / "points.parquet")
        by_t = {t: g for t, g in pts.groupby("time")}
        empty = pts.iloc[:0]
        flags = [self.flags(by_t.get(t, empty)) for t in times]
        X_e, X_v, Y, T = [], [], [], []
        for k in range(len(times) - HORIZON_STEPS):
            u = self._interp(ds["u10"].isel({tdim: k}).values.astype("float32"), lat, lon, self.elat, self.elon)
            v = self._interp(ds["v10"].isel({tdim: k}).values.astype("float32"), lat, lon, self.elat, self.elon)
            flow = u * self.etan[:, 0] + v * self.etan[:, 1]
            X_e.append(np.c_[flow, np.hypot(u, v)])
            blh = self._interp(ds["blh"].isel({tdim: k}).values.astype("float32"), lat, lon, self.cx.lat, self.cx.lon)
            dyn = np.c_[np.log1p(np.clip(blh, 0, None)) / 8.0, flags[k].astype("float32")]
            X_v.append(np.c_[self.static, dyn])
            Y.append(flags[k + HORIZON_STEPS][:, 3])
            T.append(times[k])
        ds.close()
        return {"X_e": np.array(X_e, "float16"), "X_v": np.array(X_v, "float16"), "y": np.array(Y, bool),
                "t": np.array(T)}


def build_cache(cx: Complex, months_dir: Path, periods: list[str] | None = None) -> list[Path]:
    CACHE.mkdir(parents=True, exist_ok=True)
    static = static_vertex_features(cx)
    sb = StepBuilder(cx, static=static)
    out = []
    for p in sorted(periods or [d.name for d in months_dir.glob("*") if (d / "summary.json").exists()]):
        f = CACHE / f"{p}.npz"
        if not f.exists():
            np.savez_compressed(f, **sb.period(months_dir / p))
        out.append(f)
    return out
