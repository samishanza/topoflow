"""Bulk downloads: the EDGAR emission grid and Natural Earth land polygons.

Both are cached under data/cache once and reduced to the run's grid. Raw EDGAR files are deleted after regridding,
because this project runs on a machine with little free disk.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import shapely
import xarray as xr

from .. import grid
from ..contract import GridPayload, PointsPayload, RunRequest
from ..http import Fetcher
from ..units import SECONDS_PER_YEAR, cell_areas_m2
from .base import SourceUnavailable, doc_fields

EDGAR_DIR = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/v80_FT2022_GHG/CH4/TOTALS/emi_nc/"
EDGAR_LATEST_YEAR = 2022
NE_LAND_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson"


# ------------------------------------------------------------------------------------------- EDGAR


def _edgar_native(fx: Fetcher, year: int) -> Path:
    """The 0.1 degree EDGAR grid for one year, as a compact float32 NetCDF in the cache."""
    cache = fx.cache_dir / "edgar"
    native = cache / f"edgar_v8_ch4_{year}_0p1.nc"
    if native.exists():
        fx.log("edgar").replayed = fx.mode == "replay"
        fx.log("edgar").urls.append(f"{EDGAR_DIR}v8.0_FT2022_GHG_CH4_{year}_TOTALS_emi_nc.zip")
        return native
    name = f"v8.0_FT2022_GHG_CH4_{year}_TOTALS_emi_nc.zip"
    zpath = fx.download("edgar", EDGAR_DIR + name, cache / name)
    tmp = cache / f"unzip-{year}"
    try:
        with zipfile.ZipFile(zpath) as z:
            nc_names = [n for n in z.namelist() if n.endswith(".nc")]
            if not nc_names:
                raise SourceUnavailable("error", f"{name} holds no NetCDF file")
            z.extract(nc_names[0], tmp)
            ds = xr.open_dataset(tmp / nc_names[0])
            var = "emissions" if "emissions" in ds else list(ds.data_vars)[0]
            da = ds[var].astype("float32")
            unit = da.attrs.get("units", "")
            da.attrs["source_units"] = unit
            da.to_dataset(name="emissions").to_netcdf(native, encoding={"emissions": {"zlib": True, "complevel": 4}})
            ds.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        zpath.unlink(missing_ok=True)
    return native


def edgar_flux(fx: Fetcher, req: RunRequest, year: int = EDGAR_LATEST_YEAR) -> GridPayload:
    native = _edgar_native(fx, year)
    ds = xr.open_dataset(native)
    da = ds["emissions"]
    src_unit = str(da.attrs.get("source_units", "")).strip()
    # EDGAR v8 emi_nc ships annual totals per cell in tonnes. Anything else must be looked at by a person first.
    if src_unit.lower() not in {"tonnes", "ton", "t", "tonnes/year", "t/year"}:
        raise SourceUnavailable("error", f"EDGAR emissions unit is {src_unit!r}; expected tonnes per cell per year")
    lat_name = "lat" if "lat" in da.dims else "latitude"
    lon_name = "lon" if "lon" in da.dims else "longitude"
    lat_n = da[lat_name].values
    lon_n = da[lon_name].values
    lon_n = np.where(lon_n > 180, lon_n - 360, lon_n)

    lats, lons = grid.axes(req)
    r = req.resolution_deg
    tonnes = np.nan_to_num(da.values.astype("float64"))
    # Sum mass of every native cell whose centre falls inside each target cell. Mass is conserved; nothing is interpolated.
    ii = np.floor((lat_n - (lats[0] - r / 2)) / r).astype(int)
    jj = np.floor((lon_n - (lons[0] - r / 2)) / r).astype(int)
    out = np.zeros((len(lats), len(lons)))
    vi = (ii >= 0) & (ii < len(lats))
    vj = (jj >= 0) & (jj < len(lons))
    sub = tonnes[np.ix_(vi, vj)]
    rows = ii[vi]
    cols = jj[vj]
    col_sum = np.zeros((sub.shape[0], len(lons)))
    np.add.at(col_sum.T, cols, sub.T)
    np.add.at(out, rows, col_sum)
    ds.close()

    flux = out * 1000.0 / SECONDS_PER_YEAR / cell_areas_m2(lats, r, r)[:, None]  # kg m-2 s-1
    return GridPayload(**doc_fields(fx, "edgar", f"year {year}, regridded to {r} deg by mass sum"),
                       times=np.array([np.datetime64(f"{year}-07-01")]), lats=lats, lons=lons,
                       variables={"emission_flux": flux[None].astype("float32")},
                       units={"emission_flux": "kg m-2 s-1"})


# ------------------------------------------------------------------------------------------- Natural Earth


def natural_earth_land(fx: Fetcher, req: RunRequest, subsamples: int = 4) -> tuple[GridPayload, PointsPayload]:
    gj = fx.get_json("natural_earth", NE_LAND_URL)
    polys = [shapely.geometry.shape(f["geometry"]) for f in gj.get("features", [])]
    if not polys:
        raise SourceUnavailable("error", "Natural Earth land file had no polygons")
    land = shapely.union_all(polys)
    lats, lons = grid.axes(req)
    r = req.resolution_deg
    # Land fraction by sampling a subsamples x subsamples lattice inside each cell.
    offs = (np.arange(subsamples) + 0.5) / subsamples - 0.5
    la = (lats[:, None] + offs[None, :] * r).ravel()
    lo = (lons[:, None] + offs[None, :] * r).ravel()
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    inside = shapely.contains_xy(land, LO, LA).astype("float32")
    frac = inside.reshape(len(lats), subsamples, len(lons), subsamples).mean(axis=(1, 3))

    w, s, e, n = req.bbox
    box = shapely.box(w - r, s - r, e + r, n + r)
    lines = []
    for p in polys:
        clipped = p.exterior.intersection(box)
        for g in getattr(clipped, "geoms", [clipped]):
            if g.is_empty or g.geom_type not in ("LineString", "LinearRing"):
                continue
            xs, ys = g.xy
            lines.append({"lon": float(xs[0]), "lat": float(ys[0]),
                          "properties": {"x": [round(v, 3) for v in xs], "y": [round(v, 3) for v in ys]}})
    doc = doc_fields(fx, "natural_earth", "1:110m")
    return (
        GridPayload(**doc, times=np.array([np.datetime64("2000-01-01")]), lats=lats, lons=lons,
                    variables={"land_fraction": frac[None]}, units={"land_fraction": "1"}),
        PointsPayload(**doc, features=lines),
    )

