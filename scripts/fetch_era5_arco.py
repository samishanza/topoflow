"""Stream ERA5 (u10, v10, boundary layer height) from ARCO-ERA5, Google's public cloud copy of ERA5, one month at a time.

ARCO-ERA5 stores one global hour per chunk at 0.25 degrees, so a 6-hourly month is 124 x 3 chunk reads (about 1.2 GB
transferred, nothing kept at 0.25 deg). Each month is reduced to a 1-degree grid by taking every fourth point (the
0.25 deg grid is itself an interpolation of ERA5's native grid) and written to data/era5/raw/ in the same layout the
CDS route produced, so `python -m topoflow.climatology process` consumes it unchanged.

    python scripts/fetch_era5_arco.py --start 2010-01 --end 2024-12 --threads 12

Resumable: months already processed or waiting in raw/ are skipped. The downloader holds at most --buffer raw months.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np
import pandas as pd
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "era5" / "raw"
DONE = ROOT / "data" / "era5" / "months"
BASE = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/"
VARS = {"u10": ("10m_u_component_of_wind", "m s**-1"), "v10": ("10m_v_component_of_wind", "m s**-1"),
        "blh": ("boundary_layer_height", "m")}
EPOCH = pd.Timestamp("1900-01-01")
CODEC = numcodecs.Blosc()
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "topoflow/0.3 (research prototype)"


def month_order(start: str, end: str) -> list[str]:
    months = [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]
    first = [m for m in months if m.startswith("2019")]
    after = sorted(m for m in months if m >= "2020")
    before = sorted((m for m in months if m < "2019"), reverse=True)
    return first + after + before


def chunk(var: str, t: pd.Timestamp, stride: int) -> np.ndarray:
    idx = int((t - EPOCH) / pd.Timedelta(hours=1))
    for attempt in range(5):
        try:
            r = SESSION.get(f"{BASE}{var}/{idx}.0.0", timeout=120)
            r.raise_for_status()
            a = np.frombuffer(CODEC.decode(r.content), dtype="<f4").reshape(721, 1440)
            return a[::stride, ::stride].copy()
        except Exception:  # noqa: BLE001 - transient network errors retry with backoff
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def verify_time_axis():
    """The chunk index must be hours since 1900-01-01. Check it once against the stored time coordinate."""
    probe = pd.Timestamp("2019-01-01")
    idx = int((probe - EPOCH) / pd.Timedelta(hours=1))
    per_chunk = 67706
    r = SESSION.get(f"{BASE}time/{idx // per_chunk}", timeout=120)
    r.raise_for_status()
    t = np.frombuffer(CODEC.decode(r.content), dtype="<i8")
    if int(t[idx % per_chunk]) != idx:
        raise SystemExit(f"ARCO time axis is not contiguous hours from 1900: {t[idx % per_chunk]} != {idx}")


def fetch_month(month: str, stride: int, threads: int) -> Path:
    start = pd.Timestamp(month + "-01")
    times = pd.date_range(start, start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=18), freq="6h")
    lat = np.arange(90, -90.0001, -0.25)[::stride]
    lon = np.arange(0, 360, 0.25)[::stride]
    out = {}
    with ThreadPoolExecutor(threads) as pool:
        for name, (var, _) in VARS.items():
            out[name] = np.stack(list(pool.map(lambda t, v=var: chunk(v, t, stride), times)))
    ds = xr.Dataset(
        {n: (("valid_time", "latitude", "longitude"), out[n], {"units": VARS[n][1]}) for n in VARS},
        coords={"valid_time": times.values, "latitude": lat, "longitude": lon},
        attrs={"source": "ARCO-ERA5 (gs://gcp-public-data-arco-era5), ERA5 reanalysis by ECMWF/Copernicus C3S",
               "grid": f"every {stride}th point of the 0.25 deg grid", "url": BASE},
    )
    res = 0.25 * stride
    target = RAW / f"era5_{month}_{res:g}deg.nc"
    part = target.with_suffix(".part")
    ds.to_netcdf(part, encoding={n: {"zlib": True, "complevel": 1} for n in VARS})
    part.rename(target)
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--res", type=float, default=1.0, help="0.25, 0.5, 1.0 or 2.0")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--buffer", type=int, default=4)
    a = ap.parse_args()
    stride = int(round(a.res / 0.25))
    RAW.mkdir(parents=True, exist_ok=True)
    verify_time_axis()
    tag = f"{a.res:g}deg"
    todo = [m for m in month_order(a.start, a.end)
            if not (DONE / ("1deg" if tag == "1deg" else tag) / m / "summary.json").exists()
            and not (RAW / f"era5_{m}_{tag}.nc").exists()]
    print(f"{len(todo)} months to stream at {a.res} deg", flush=True)
    for m in todo:
        while len(list(RAW.glob("era5_*.nc"))) >= a.buffer or shutil.disk_usage(RAW).free < 2e9:
            time.sleep(10)
        t0 = time.monotonic()
        try:
            p = fetch_month(m, stride, a.threads)
            print(f"{m}: {p.stat().st_size / 1e6:.0f} MB in {time.monotonic() - t0:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{m}: FAILED {type(exc).__name__}: {exc}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
