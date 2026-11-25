"""Multi-year stagnation climatology from monthly ERA5 files.

    python -m topoflow.climatology process [--watch]      # consume data/era5/raw/*.nc as they arrive
    python -m topoflow.climatology summary                 # closure and persistence across processed months

Each month goes through the same discipline as a run: the raw fields are gated against the catalog (units, range,
missing share) before anything is derived, the source and request are recorded, and the raw file is deleted only after
everything derived from it is written. Output per month, under data/era5/months/<res>deg/<YYYY-MM>/:

  points.parquet     every located stagnation point: time, lat, lon, kind, index, strength, divergence, vorticity,
                     track id and age (h) from 6-hourly linking
  closure.parquet    per time step: winding sum, edge error, caps, point count. winding_sum == 2 + edge_error always.
  occupancy.nc       per 1-degree cell: how many time steps held a point of each kind
  display.nc         u10, v10, blh on a 2-degree grid every 6 h (float16), for the globe
  summary.json       provenance, gate results, counts
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .catalog import FIELDS
from .contract import now_iso
from .sphere import analyse
from .sources.registry import SOURCES

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "era5" / "raw"
OUT = ROOT / "data" / "era5" / "months"
KINDS = ("saddle", "attracting_node", "repelling_node", "attracting_focus", "repelling_focus", "center")
# Occupancy layers: each kind, plus unions. A step counts once per cell however many points are near it.
LAYERS = KINDS + ("node", "focus", "attracting", "all")
LAYER_MEMBERS = {**{k: (k,) for k in KINDS}, "node": ("attracting_node", "repelling_node"),
                 "focus": ("attracting_focus", "repelling_focus", "center"),
                 "attracting": ("attracting_node", "attracting_focus"), "all": KINDS}
LINK_KM = 400.0  # a point 6 h later within this distance, same family, continues the same track


def _family(kind: str) -> str:
    return "saddle" if kind == "saddle" else ("node" if kind.endswith("node") else "focus")


def gate_month(ds: xr.Dataset) -> list[str]:
    """Deterministic admissibility for a raw month. Returns reasons to refuse; empty means admitted."""
    problems = []
    wind_units = {"m s**-1", "m s-1", "m/s"}
    for var, key, unit_ok in (("u10", "wind.u10", wind_units), ("v10", "wind.v10", wind_units),
                              ("blh", "met.blh", {"m"})):
        if var not in ds:
            if var != "blh":  # boundary-layer height is optional: not every reanalysis serves it
                problems.append(f"{var} missing")
            continue
        spec = FIELDS[key]
        da = ds[var]
        if da.attrs.get("units") not in unit_ok:
            problems.append(f"{var} units {da.attrs.get('units')!r}, expected {spec.unit}")
        vals = da.values
        finite = np.isfinite(vals)
        if 1 - finite.mean() > spec.max_nan_fraction:
            problems.append(f"{var}: {1 - finite.mean():.1%} missing")
        bad = ((vals[finite] < spec.min) | (vals[finite] > spec.max)).mean()
        if bad > spec.max_out_of_bounds_fraction:
            problems.append(f"{var}: {bad:.2%} outside {spec.min}..{spec.max}")
    return problems


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class Tracker:
    """Greedy 6-hourly linking of points of the same family, nearest first, within LINK_KM."""

    def __init__(self):
        self.prev_lat = np.empty(0)
        self.prev_lon = np.empty(0)
        self.prev_fam = np.empty(0, dtype=object)
        self.prev_track = np.empty(0, dtype=np.int64)
        self.prev_age = np.empty(0)
        self.next_id = 0

    def link(self, pts: list[dict], hours: float) -> None:
        used = np.zeros(len(self.prev_lat), dtype=bool)
        for p in pts:
            fam = _family(p["kind"])
            p["family"] = fam
            cand = (~used) & (self.prev_fam == fam)
            best = None
            if cand.any():
                d = np.where(cand, _haversine_km(p["lat"], p["lon"], self.prev_lat, self.prev_lon), np.inf)
                k = int(d.argmin())
                if d[k] < LINK_KM:
                    best = k
            if best is None:
                p["track"], p["age_h"] = self.next_id, 0.0
                self.next_id += 1
            else:
                used[best] = True
                p["track"], p["age_h"] = int(self.prev_track[best]), float(self.prev_age[best]) + hours
        self.prev_lat = np.array([p["lat"] for p in pts])
        self.prev_lon = np.array([p["lon"] for p in pts])
        self.prev_fam = np.array([p["family"] for p in pts], dtype=object)
        self.prev_track = np.array([p["track"] for p in pts], dtype=np.int64)
        self.prev_age = np.array([p["age_h"] for p in pts])


def _dilate(mask: np.ndarray) -> np.ndarray:
    """A cell and its 8 neighbours (longitude wraps, latitude does not): a point wanders a cell between steps."""
    out = mask.copy()
    for di in (-1, 0, 1):
        shifted = np.zeros_like(mask)
        if di < 0:
            shifted[..., :di, :] = mask[..., -di:, :]
        elif di > 0:
            shifted[..., di:, :] = mask[..., :-di, :]
        else:
            shifted = mask
        for dj in (-1, 0, 1):
            out |= np.roll(shifted, dj, axis=-1)
    return out


def _occupancy_step(hit: np.ndarray) -> np.ndarray:
    """1 where a point of the layer's kinds lies within one cell this step, per layer."""
    near = _dilate(hit)
    return np.stack([near[[KINDS.index(k) for k in LAYER_MEMBERS[layer]]].any(0) for layer in LAYERS]).astype("int32")


def process_month(raw: Path, res_tag: str, tracker: Tracker | None = None, out_root: Path | None = None,
                  source_id: str | None = None, display_stride: int = 2) -> dict:
    month = raw.name.split("_")[1]
    out = (out_root or OUT) / res_tag / month
    out.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(raw)
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    problems = gate_month(ds)
    if problems:
        ds.close()
        (out / "refused.json").write_text(json.dumps({"month": month, "refused": problems, "at": now_iso()}, indent=2))
        raise RuntimeError(f"{month} refused by the gate: {problems}")

    lats = ds["latitude"].values
    lons = ds["longitude"].values
    times = pd.DatetimeIndex(ds[tdim].values)
    tracker = tracker or Tracker()
    rows, closure = [], []
    occ_lat = np.sort(lats)
    occ_lon = np.sort(((lons + 180) % 360) - 180)
    occ = np.zeros((len(LAYERS), len(occ_lat), len(occ_lon)), dtype="int32")
    step_h = 6.0
    u_all, v_all = ds["u10"].values, ds["v10"].values
    for k, t in enumerate(times):
        r = analyse(u_all[k], v_all[k], lats, lons)
        pts = r["points"]
        tracker.link(pts, step_h)
        hit = np.zeros((len(KINDS), len(occ_lat), len(occ_lon)), dtype=bool)
        for p in pts:
            rows.append((t, p["lat"], p["lon"], p["kind"], p["index"], p["strength"], p["divergence"],
                         p["vorticity"], p["track"], p["age_h"]))
            i = int(np.abs(occ_lat - p["lat"]).argmin())
            j = int(np.abs(occ_lon - p["lon"]).argmin())
            hit[KINDS.index(p["kind"]), i, j] = True
        occ += _occupancy_step(hit)
        closure.append((t, r["winding_sum"], r["edge_error"], r["edge_flips"], r["cap_north"], r["cap_south"],
                        r["n_points"]))

    pd.DataFrame(rows, columns=["time", "lat", "lon", "kind", "index", "strength", "divergence", "vorticity",
                                "track", "age_h"]).astype({"index": "int8", "track": "int64"}).to_parquet(
        out / "points.parquet", index=False)
    cl = pd.DataFrame(closure, columns=["time", "winding_sum", "edge_error", "edge_flips", "cap_north", "cap_south",
                                        "n_points"])
    cl.to_parquet(out / "closure.parquet", index=False)
    xr.Dataset({"occupancy": (("layer", "lat", "lon"), occ)},
               coords={"layer": list(LAYERS), "lat": occ_lat, "lon": occ_lon},
               attrs={"steps": len(times), "month": month}).to_netcdf(
        out / "occupancy.nc", encoding={"occupancy": {"zlib": True}})

    # Display grid: every second degree, poles included, longitudes -180..178.
    have = [v for v in ("u10", "v10", "blh") if v in ds]
    disp = ds[have].sortby("latitude").assign_coords(
        longitude=((ds["longitude"] + 180) % 360) - 180).sortby("longitude")
    s = display_stride
    disp = disp.isel(latitude=slice(None, None, s), longitude=slice(None, None, s)).astype("float32")
    # NetCDF has no float16: pack as int16 (0.01 m/s for wind, 0.25 m for BLH), compressed.
    pack = {k: v for k, v in {"u10": 0.01, "v10": 0.01, "blh": 0.25}.items() if k in have}
    disp.to_netcdf(out / "display.nc", encoding={v: {"zlib": True, "complevel": 5, "dtype": "int16",
                                                     "scale_factor": s, "_FillValue": -32768}
                                                 for v, s in pack.items()})

    entry = SOURCES[source_id or ("arco_era5" if "ARCO" in ds.attrs.get("source", "") else "cds_era5")]
    summary = {
        "month": month, "processed_at": now_iso(), "steps": len(times),
        "source": {"id": entry.id, "name": entry.name, "license": entry.license, "attribution": entry.attribution,
                   "file": raw.name, "variables": have, "resolution": res_tag},
        "gate": "admitted",
        "points": len(rows),
        "closure_exact_2": int((cl.winding_sum == 2).sum()),
        "identity_holds": bool((cl.winding_sum == 2 + cl.edge_error).all()),
        "mean_points_per_step": round(len(rows) / max(len(times), 1), 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    ds.close()
    raw.unlink()
    return summary


def process_all(watch: bool = False, res_tag: str = "1deg"):
    tracker = Tracker()
    idle = 0
    while True:
        raws = sorted(RAW.glob("era5_*_*deg.nc"))
        if not raws:
            if not watch or idle > 240:  # stop after 4 h with nothing new
                return
            idle += 1
            time.sleep(60)
            continue
        idle = 0
        for raw in raws:
            t0 = time.monotonic()
            try:
                s = process_month(raw, res_tag, tracker)
                print(f"{s['month']}: {s['points']} points, closure exactly 2 in {s['closure_exact_2']}/{s['steps']} "
                      f"steps, identity {'holds' if s['identity_holds'] else 'BROKEN'}, {time.monotonic() - t0:.0f}s",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 - one bad month is reported, not fatal
                print(f"{raw.name}: {type(exc).__name__}: {exc}", flush=True)


def load(kind: str, res_tag: str = "1deg") -> pd.DataFrame:
    files = sorted((OUT / res_tag).glob(f"*/{kind}.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def summary(res_tag: str = "1deg") -> dict:
    cl = load("closure", res_tag)
    pts = load("points", res_tag)
    if cl.empty:
        return {"months": 0}
    return {
        "months": len(list((OUT / res_tag).glob("*/summary.json"))),
        "steps": len(cl),
        "mean_winding_sum": round(float(cl.winding_sum.mean()), 3),
        "share_exactly_2": round(float((cl.winding_sum == 2).mean()), 3),
        "identity_holds_all": bool((cl.winding_sum == 2 + cl.edge_error).all()),
        "mean_points_per_step": round(float(cl.n_points.mean()), 1),
        "points": len(pts),
        "by_kind": pts.kind.value_counts().to_dict() if not pts.empty else {},
    }


def main():
    ap = argparse.ArgumentParser(prog="topoflow.climatology")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("process")
    p.add_argument("--watch", action="store_true")
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "process":
        process_all(a.watch)
    else:
        print(json.dumps(summary(), indent=2))


if __name__ == "__main__":
    main()
