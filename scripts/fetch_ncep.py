"""Independent reanalyses for cross-product agreement: NCEP/NCAR R1 and NCEP-DOE R2 10 m wind from NOAA PSL.

    python scripts/fetch_ncep.py --product r1 --years 2019 2020
    python scripts/fetch_ncep.py --product r2 --years 2019

Each year's u and v files (about 50-70 MB each) are downloaded, split into months in the layout the climatology
processor reads, processed on their native T62 Gaussian grid (about 1.9 deg; nothing is regridded), and deleted.
Output: data/ncep_<product>/months/T62/<YYYY-MM>/.

Data provided by the NOAA PSL, Boulder, Colorado, USA, from their website at https://psl.noaa.gov.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topoflow.climatology import Tracker, process_month  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
URLS = {
    "r1": "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/{var}.10m.gauss.{year}.nc",
    "r2": "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis2/gaussian_grid/{var}.10m.gauss.{year}.nc",
}


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    part = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=600, headers={"User-Agent": "topoflow/0.3 (research prototype)"}) as r:
        r.raise_for_status()
        with open(part, "wb") as fh:
            for block in r.iter_content(1 << 20):
                fh.write(block)
    part.rename(dest)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", choices=list(URLS), default="r1")
    ap.add_argument("--years", type=int, nargs="+", default=[2019])
    a = ap.parse_args()
    root = ROOT / "data" / f"ncep_{a.product}"
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tracker = Tracker()
    for year in a.years:
        files = {v: download(URLS[a.product].format(var=v, year=year), raw / f"{v}.{year}.nc") for v in ("uwnd", "vwnd")}
        u = xr.open_dataset(files["uwnd"])
        v = xr.open_dataset(files["vwnd"])
        uu = u["uwnd"].squeeze(drop=True)
        vv = v["vwnd"].squeeze(drop=True)
        for month, idx in uu.groupby("time.month").groups.items():
            tag = f"{year}-{month:02d}"
            if (root / "months" / "T62" / tag / "summary.json").exists():
                continue
            ds = xr.Dataset({"u10": uu.isel(time=idx).astype("float32"), "v10": vv.isel(time=idx).astype("float32")})
            ds = ds.rename({"time": "valid_time", "lat": "latitude", "lon": "longitude"})
            for name in ("u10", "v10"):
                ds[name].attrs = {"units": "m/s"}
            ds.attrs["source"] = f"NCEP {a.product.upper()} via NOAA PSL"
            f = raw / f"ncep{a.product}_{tag}_T62.nc"
            ds.to_netcdf(f)
            s = process_month(f, "T62", tracker, out_root=root / "months", source_id=f"ncep_{a.product}",
                              display_stride=1)
            print(f"{a.product} {tag}: {s['points']} points, closure exactly 2 in {s['closure_exact_2']}/{s['steps']}, "
                  f"identity {'holds' if s['identity_holds'] else 'BROKEN'}", flush=True)
        u.close()
        v.close()
        for f in files.values():
            f.unlink()


if __name__ == "__main__":
    main()
