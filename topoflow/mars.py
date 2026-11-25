"""Stagnation climatology on Mars, from the OpenMARS reanalysis (Holmes, Lewis et al., Open University).

    python -m topoflow.mars --year 34          # stream every OpenMARS file of Mars Year 34, process, delete
    python -m topoflow.mars --year 34 --limit 2

OpenMARS MY28-35 (v3, CC BY 4.0, https://ordo.open.ac.uk/articles/dataset/OpenMARS_continuous_MY28-35_standard_database/24573205)
is a 5 x 5 degree, 35 sigma-level reanalysis with output every 2 Mars hours, 30 sols per 407 MB file. There is no
10 m wind: the lowest sigma level (0.9995, a few metres above the surface) stands in for it.

Each file goes through the same discipline as an ERA5 month: gate the raw fields, analyse every step on the sphere
(Mars radius), track points, count occupancy, write, and delete the raw file. Output per file, under
data/mars/periods/5deg/<file stem>/: points.parquet, closure.parquet, occupancy.nc, display.nc, summary.json.
The `time` column is the sol number (OpenMARS `time`), and `label` gives Mars Year and solar longitude Ls.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .climatology import KINDS, LAYERS, Tracker, _occupancy_step
from .contract import now_iso
from .sphere import analyse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "mars" / "raw"
OUT = ROOT / "data" / "mars" / "periods" / "5deg"
ARTICLE = "https://api.figshare.com/v2/articles/24573205/files?page_size=500"
MARS_RADIUS_M = 3_389_500.0
SOL_HOURS = 24.6597  # one sol in Earth hours
LICENSE = "CC BY 4.0"
ATTRIBUTION = ("OpenMARS MY28-35 standard database v3 (Holmes, J.A., Lewis, S.R., Patel, M.R. et al.), "
               "The Open University, CC BY 4.0")
# Near-surface winds on Mars stay well inside this; anything beyond is a unit or file fault.
WIND_LIMIT = 120.0


def list_files(year: int) -> list[dict]:
    files = requests.get(ARTICLE, timeout=60).json()
    out = [f for f in files if re.match(rf"openmars_my{year}_ls\d+_my\d+_ls\d+\.nc$", f["name"])]
    return sorted(out, key=lambda f: int(re.search(r"_ls(\d+)_", f["name"]).group(1)))


def download(f: dict) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / f["name"]
    if target.exists() and target.stat().st_size == f["size"]:
        return target
    part = target.with_suffix(".part")
    with requests.get(f"https://ndownloader.figshare.com/files/{f['id']}", stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(part, "wb") as fh:
            for block in r.iter_content(1 << 20):
                fh.write(block)
    part.rename(target)
    return target


def gate(u: np.ndarray, v: np.ndarray) -> list[str]:
    problems = []
    for name, a in (("u", u), ("v", v)):
        finite = np.isfinite(a)
        if finite.mean() < 0.98:
            problems.append(f"{name}: {1 - finite.mean():.1%} missing")
        if (np.abs(a[finite]) > WIND_LIMIT).mean() > 0.001:
            problems.append(f"{name}: values beyond +-{WIND_LIMIT} m/s")
    return problems


def process_file(path: Path) -> dict:
    stem = path.stem
    out = OUT / stem
    out.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(path, decode_times=False)
    u = ds["u"].isel(lev=0).values.astype("float32")
    v = ds["v"].isel(lev=0).values.astype("float32")
    problems = gate(u, v)
    if problems:
        (out / "refused.json").write_text(json.dumps({"file": path.name, "refused": problems}, indent=2))
        raise RuntimeError(f"{stem} refused: {problems}")
    lats, lons = ds["lat"].values, ds["lon"].values
    sols = ds["time"].values.astype(float)
    ls = ds["Ls"].values.astype(float)
    my = ds["MY"].values.astype(int)
    labels = [f"MY{m} · Ls {l:5.1f}° · sol {s:.2f}" for m, l, s in zip(my, ls, sols)]

    tracker = Tracker()
    step_h = float(np.median(np.diff(sols))) * SOL_HOURS if len(sols) > 1 else 2 * SOL_HOURS / 24
    occ_lat, occ_lon = np.sort(lats), np.sort(lons)
    occ = np.zeros((len(LAYERS), len(occ_lat), len(occ_lon)), dtype="int32")
    rows, closure = [], []
    for k in range(len(sols)):
        r = analyse(u[k], v[k], lats, lons, radius_m=MARS_RADIUS_M)
        pts = r["points"]
        tracker.link(pts, step_h)
        hit = np.zeros((len(KINDS), len(occ_lat), len(occ_lon)), dtype=bool)
        for p in pts:
            rows.append((sols[k], p["lat"], p["lon"], p["kind"], p["index"], p["strength"], p["divergence"],
                         p["vorticity"], p["track"], p["age_h"]))
            hit[KINDS.index(p["kind"]), int(np.abs(occ_lat - p["lat"]).argmin()), int(np.abs(occ_lon - p["lon"]).argmin())] = True
        occ += _occupancy_step(hit)
        closure.append((sols[k], labels[k], float(ls[k]), int(my[k]), r["winding_sum"], r["edge_error"],
                        r["edge_flips"], r["cap_north"], r["cap_south"], r["n_points"]))

    pd.DataFrame(rows, columns=["time", "lat", "lon", "kind", "index", "strength", "divergence", "vorticity",
                                "track", "age_h"]).to_parquet(out / "points.parquet", index=False)
    cl = pd.DataFrame(closure, columns=["time", "label", "ls", "my", "winding_sum", "edge_error", "edge_flips",
                                        "cap_north", "cap_south", "n_points"])
    cl.to_parquet(out / "closure.parquet", index=False)
    xr.Dataset({"occupancy": (("layer", "lat", "lon"), occ)},
               coords={"layer": list(LAYERS), "lat": occ_lat, "lon": occ_lon},
               attrs={"steps": len(sols), "period": stem}).to_netcdf(out / "occupancy.nc")
    disp = xr.Dataset(
        {"u10": (("valid_time", "latitude", "longitude"), u), "v10": (("valid_time", "latitude", "longitude"), v),
         "blh": (("valid_time", "latitude", "longitude"), ds["dustcol"].values.astype("float32"))},
        coords={"valid_time": sols, "latitude": lats, "longitude": lons},
        attrs={"note": "u10/v10 are the lowest sigma level (0.9995); blh holds the dust column optical depth"},
    ).sortby("latitude")
    disp.to_netcdf(out / "display.nc", encoding={k: {"zlib": True} for k in ("u10", "v10", "blh")})
    summary = {
        "month": stem, "processed_at": now_iso(), "steps": len(sols),
        # First and last Ls, not min/max: a file that crosses into the next Mars year wraps through 360.
        "mars_year": int(my[0]), "ls_range": [round(float(ls[0]), 1), round(float(ls[-1]), 1)],
        "source": {"id": "openmars", "name": "OpenMARS MY28-35 standard database (lowest sigma level)",
                   "license": LICENSE, "attribution": ATTRIBUTION, "file": path.name, "resolution": "5deg"},
        "gate": "admitted", "points": len(rows),
        "closure_exact_2": int((cl.winding_sum == 2).sum()),
        "identity_holds": bool((cl.winding_sum == 2 + cl.edge_error).all()),
        "mean_points_per_step": round(len(rows) / max(len(sols), 1), 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    ds.close()
    path.unlink()
    return summary


def main():
    ap = argparse.ArgumentParser(prog="topoflow.mars")
    ap.add_argument("--year", type=int, default=34)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = list_files(a.year)
    if a.limit:
        files = files[: a.limit]
    todo = [f for f in files if not (OUT / Path(f["name"]).stem / "summary.json").exists()]
    print(f"Mars Year {a.year}: {len(files)} files, {len(todo)} to process", flush=True)
    for f in todo:
        t0 = time.monotonic()
        try:
            s = process_file(download(f))
            print(f"{s['month']}: Ls {s['ls_range']}, {s['points']} points, closure exactly 2 in "
                  f"{s['closure_exact_2']}/{s['steps']}, identity {'holds' if s['identity_holds'] else 'BROKEN'}, "
                  f"{time.monotonic() - t0:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{f['name']}: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
