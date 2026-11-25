"""Sources that need a key or a file from a person: Copernicus CDS, Google Earth Engine, and local NetCDF.

Each checks for its credentials first and, when they are absent, reports itself barred with the route a person
takes to open it. None of these have been run against live credentials in this repo yet; see
docs/research/2026-11-25-ch4-wind-datasets.md for their test status.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .. import grid
from ..contract import GridPayload, RunRequest, now_iso
from ..http import Fetcher
from .base import SourceUnavailable, doc_fields
from .registry import SOURCES


def _attrs_text(obj) -> str:
    """Free text from file and variable attributes. Kept so the review agent can see it, and never obeyed."""
    parts = []
    for k, v in getattr(obj, "attrs", {}).items():
        if isinstance(v, str):
            parts.append(f"{k}: {v}")
    return "\n".join(parts)[:4000]


def _to_grid(da: xr.DataArray, req: RunRequest, time_dim: str) -> np.ndarray:
    """Nearest-cell sample of a (time, lat, lon) array onto the run grid, over the run's times."""
    lat_name = next(n for n in ("latitude", "lat") if n in da.dims)
    lon_name = next(n for n in ("longitude", "lon") if n in da.dims)
    lats, lons = grid.axes(req)
    if float(da[lon_name].max()) > 180:
        da = da.assign_coords({lon_name: ((da[lon_name] + 180) % 360) - 180}).sortby(lon_name)
    da = da.sortby(lat_name)
    want = grid.times(req)
    da = da.sel({time_dim: want}, method="nearest", tolerance=np.timedelta64(90, "m"))
    da = da.sel({lat_name: lats, lon_name: lons}, method="nearest")
    return da.transpose(time_dim, lat_name, lon_name).values.astype("float32")


# ----------------------------------------------------------------------------------------- Copernicus CDS


def _cds_ready() -> bool:
    return Path.home().joinpath(".cdsapirc").exists() or bool(os.environ.get("CDSAPI_KEY"))


def cds_era5_wind(fx: Fetcher, req: RunRequest) -> GridPayload:
    entry = SOURCES["cds_era5"]
    if not _cds_ready():
        raise SourceUnavailable("barred", "no Copernicus CDS credentials", entry.person_route)
    days = pd.date_range(req.start, req.end, freq="D")
    w, s, e, n = req.bbox
    request = {
        "product_type": ["reanalysis"],
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "year": sorted({f"{d:%Y}" for d in days}),
        "month": sorted({f"{d:%m}" for d in days}),
        "day": sorted({f"{d:%d}" for d in days}),
        "time": [f"{h:02d}:00" for h in req.hours_utc],
        "area": [n, w, s, e],
        "grid": [req.resolution_deg, req.resolution_deg],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
    target = fx.cache_dir / "cds" / f"era5-{key}.nc"
    log = fx.log("cds_era5")
    log.urls.append(f"{entry.url} request {key}")
    if not target.exists():
        if fx.mode == "replay":
            raise SourceUnavailable("missing", "CDS result not cached and replay mode cannot request it")
        try:
            import cdsapi  # optional dependency
        except ImportError as exc:
            raise SourceUnavailable("barred", "the cdsapi package is not installed (pip install cdsapi)") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        cdsapi.Client(quiet=True).retrieve("reanalysis-era5-single-levels", request, str(target))
        log.calls += 1
        log.bytes += target.stat().st_size
    else:
        log.replayed = fx.mode == "replay"
    ds = xr.open_dataset(target)
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    payload = GridPayload(**doc_fields(fx, "cds_era5", f"{req.resolution_deg} deg"),
                          times=grid.times(req), lats=grid.axes(req)[0], lons=grid.axes(req)[1],
                          variables={"u10": _to_grid(ds["u10"], req, tdim), "v10": _to_grid(ds["v10"], req, tdim)},
                          units={"u10": ds["u10"].attrs.get("units", "m s**-1"), "v10": ds["v10"].attrs.get("units", "m s**-1")},
                          raw_text=_attrs_text(ds))
    ds.close()
    return payload


# ----------------------------------------------------------------------------------------- Earth Engine TROPOMI


def gee_tropomi_xch4(fx: Fetcher, req: RunRequest) -> GridPayload:
    entry = SOURCES["gee_tropomi"]
    project = os.environ.get("EARTHENGINE_PROJECT")
    if not project:
        raise SourceUnavailable("barred", "EARTHENGINE_PROJECT is not set", entry.person_route)
    if fx.mode == "replay":
        raise SourceUnavailable("missing", "Earth Engine results are not recorded; run live or record")
    try:
        import ee  # optional dependency
        ee.Initialize(project=project)
    except Exception as exc:  # noqa: BLE001 - any auth failure means a person has to act
        raise SourceUnavailable("barred", f"Earth Engine did not initialise: {exc}", entry.person_route) from exc

    lats, lons = grid.axes(req)
    w, s, e, n = req.bbox
    r = req.resolution_deg
    days = pd.date_range(req.start, req.end, freq="D")
    band = "CH4_column_volume_mixing_ratio_dry_air"
    out = np.full((len(days), len(lats), len(lons)), np.nan, dtype="float32")
    log = fx.log("gee_tropomi")
    for k, d in enumerate(days):
        img = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CH4").select(band)
               .filterDate(d.strftime("%Y-%m-%d"), (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")).mean())
        arr = ee.data.computePixels({
            "expression": img,
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {
                "dimensions": {"width": len(lons), "height": len(lats)},
                "affineTransform": {"scaleX": r, "shearX": 0, "translateX": w - r / 2,
                                    "shearY": 0, "scaleY": -r, "translateY": n + r / 2},
                "crsCode": "EPSG:4326",
            },
        })
        log.calls += 1
        vals = np.asarray(arr[band], dtype="float32")[::-1]  # north-up rows to south-first
        vals[vals == 0] = np.nan  # masked pixels come back as zero
        out[k] = vals
    log.urls.append(f"ee://COPERNICUS/S5P/OFFL/L3_CH4/{band} {req.start}..{req.end}")
    return GridPayload(**doc_fields(fx, "gee_tropomi", "daily mean"), times=days.values.astype("datetime64[ns]"),
                       lats=lats, lons=lons, variables={"xch4": out}, units={"xch4": "ppb"})


# ----------------------------------------------------------------------------------------- local files


def local_era5_wind(fx: Fetcher, req: RunRequest) -> GridPayload:
    entry = SOURCES["local_era5"]
    u_file, v_file = os.environ.get("TOPOFLOW_U_FILE"), os.environ.get("TOPOFLOW_V_FILE")
    if not (u_file and v_file):
        raise SourceUnavailable("skipped", "TOPOFLOW_U_FILE / TOPOFLOW_V_FILE not set", entry.person_route)
    for f in (u_file, v_file):
        if not Path(f).exists():
            raise SourceUnavailable("missing", f"{f} does not exist", entry.person_route)
    u_ds, v_ds = xr.open_dataset(u_file), xr.open_dataset(v_file)
    tdim = "valid_time" if "valid_time" in u_ds.dims else "time"
    fx.log("local_era5").urls.extend([f"file://{u_file}", f"file://{v_file}"])
    u, v = u_ds["u10"], v_ds["v10"]
    lats, lons = grid.axes(req)
    payload = GridPayload(source_id="local_era5", refs=[f"file://{u_file}", f"file://{v_file}"],
                          label="Local ERA5 10 m wind files", retrieved_at=now_iso(),
                          times=grid.times(req), lats=lats, lons=lons,
                          variables={"u10": _to_grid(u, req, tdim), "v10": _to_grid(v, req, tdim)},
                          units={"u10": u.attrs.get("units", ""), "v10": v.attrs.get("units", "")},
                          raw_text=_attrs_text(u_ds) + "\n" + _attrs_text(u))
    u_ds.close()
    v_ds.close()
    return payload


def local_ch4(fx: Fetcher, req: RunRequest) -> GridPayload:
    entry = SOURCES["local_ch4"]
    folder = os.environ.get("TOPOFLOW_METHANE_DIR")
    if not folder:
        raise SourceUnavailable("skipped", "TOPOFLOW_METHANE_DIR not set", entry.person_route)
    if not Path(folder).is_dir():
        raise SourceUnavailable("missing", f"{folder} is not a folder", entry.person_route)
    var_env = os.environ.get("TOPOFLOW_METHANE_VAR")
    unit_env = os.environ.get("TOPOFLOW_METHANE_UNIT")
    lats, lons = grid.axes(req)
    days = pd.date_range(req.start, req.end, freq="D")
    out = np.full((len(days), len(lats), len(lons)), np.nan, dtype="float32")
    unit, text, refs = None, [], []
    for k, d in enumerate(days):
        f = Path(folder) / f"{d:%Y%m%d}.nc"
        if not f.exists():
            continue
        with xr.open_dataset(f) as ds:
            name = var_env or (list(ds.data_vars)[0] if len(ds.data_vars) == 1 else None)
            if not name or name not in ds:
                raise SourceUnavailable("error", f"{f.name}: set TOPOFLOW_METHANE_VAR to one of {list(ds.data_vars)}")
            da = ds[name]
            if "time" in da.dims or "valid_time" in da.dims:
                da = da.isel({next(t for t in ("time", "valid_time") if t in da.dims): 0})
            lat_name = next(n for n in ("latitude", "lat") if n in da.dims)
            lon_name = next(n for n in ("longitude", "lon") if n in da.dims)
            if float(da[lon_name].max()) > 180:
                da = da.assign_coords({lon_name: ((da[lon_name] + 180) % 360) - 180}).sortby(lon_name)
            out[k] = da.sortby(lat_name).sel({lat_name: lats, lon_name: lons}, method="nearest").values
            unit = unit or unit_env or da.attrs.get("units")
            text.append(_attrs_text(ds) + "\n" + _attrs_text(da))
            refs.append(f"file://{f}")
    if not refs:
        raise SourceUnavailable("missing", f"no YYYYMMDD.nc files in {folder} for {req.start}..{req.end}", entry.person_route)
    fx.log("local_ch4").urls.extend(refs)
    return GridPayload(source_id="local_ch4", refs=refs, label="Local daily CH4 files", retrieved_at=now_iso(),
                       times=days.values.astype("datetime64[ns]"), lats=lats, lons=lons,
                       variables={"ch4": out}, units={"ch4": unit or ""}, raw_text="\n".join(text)[:4000])
