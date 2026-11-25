"""Download global ERA5 (u10, v10, boundary layer height) one month at a time from the Copernicus CDS.

Raw months land in data/era5/raw/ and are consumed and deleted by `python -m topoflow.climatology process`. The
downloader never holds more than --buffer raw months, so the disk stays bounded however far behind processing is.

    python scripts/fetch_era5_monthly.py --start 2010-01 --end 2024-12 --res 1.0 --workers 3

Order: 2019 first (a full year inside the TROPOMI era, so the globe has a year to show early), then 2020-2024,
2018, and back to 2010. Already processed or downloaded months are skipped, so the script resumes after any stop.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "era5" / "raw"
DONE = ROOT / "data" / "era5" / "months"
MIN_FREE_GB = 2.0


def month_order(start: str, end: str) -> list[str]:
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    months = [f"{y}-{m:02d}" for y in range(y0, y1 + 1) for m in range(1, 13)
              if (y, m) >= (y0, m0) and (y, m) <= (y1, m1)]
    first = [m for m in months if m.startswith("2019")]
    after = sorted(m for m in months if m >= "2020")
    before = sorted((m for m in months if m < "2019"), reverse=True)
    return first + after + before


def raw_path(month: str, res: float) -> Path:
    return RAW / f"era5_{month}_{res:g}deg.nc"


def done(month: str, res: float) -> bool:
    return (DONE / f"{res:g}deg" / month / "summary.json").exists() or raw_path(month, res).exists()


def fetch(month: str, res: float) -> str:
    import calendar

    import cdsapi

    y, m = map(int, month.split("-"))
    days = [f"{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1)]
    target = raw_path(month, res)
    part = target.with_suffix(".part")
    cdsapi.Client(quiet=True, progress=False).retrieve("reanalysis-era5-single-levels", {
        "product_type": ["reanalysis"],
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind", "boundary_layer_height"],
        "year": [f"{y}"], "month": [f"{m:02d}"], "day": days,
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "grid": [res, res],
        "data_format": "netcdf", "download_format": "unarchived",
    }, str(part))
    part.rename(target)
    return f"{month}: {target.stat().st_size / 1e6:.0f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--res", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--buffer", type=int, default=12, help="most raw months on disk at once")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    todo = [m for m in month_order(a.start, a.end) if not done(m, a.res)]
    print(f"{len(todo)} months to fetch at {a.res} deg", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        pending = {}
        while todo or pending:
            while todo and len(pending) < a.workers:
                raw_now = len(list(RAW.glob(f"era5_*_{a.res:g}deg.nc"))) + len(pending)
                free_gb = shutil.disk_usage(RAW).free / 1e9
                if raw_now >= a.buffer or free_gb < MIN_FREE_GB:
                    break
                m = todo.pop(0)
                pending[pool.submit(fetch, m, a.res)] = m
            if not pending:
                time.sleep(30)  # waiting for the processor to drain the buffer, or for disk
                continue
            for fut in as_completed(list(pending), timeout=None):
                m = pending.pop(fut)
                try:
                    print(fut.result(), flush=True)
                except Exception as exc:  # noqa: BLE001 - report and requeue at the end
                    print(f"{m}: FAILED {type(exc).__name__}: {exc}", flush=True)
                    todo.append(m)
                    time.sleep(60)
                break
    print("all months fetched", flush=True)


if __name__ == "__main__":
    sys.exit(main())
