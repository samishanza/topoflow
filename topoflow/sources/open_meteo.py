"""ERA5 and CAMS on a regular grid, by asking Open-Meteo for every cell centre in batches.

Open-Meteo snaps each requested point to its nearest model cell (0.25 deg ERA5, about 0.4 deg CAMS). Asking for a
coarser grid therefore samples, rather than averages, the model. That is stated on every value from here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import grid
from ..contract import GridPayload, RunRequest
from ..http import Fetcher
from .base import SourceUnavailable, doc_fields

ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"
CAMS_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
BATCH = 100  # points per request. Each point costs one call against the 600/min budget.

ERA5_VARS = {
    # Open-Meteo name -> (unit, record name)
    "wind_speed_10m": ("m s-1", "wind_speed_10m"),
    "wind_direction_10m": ("deg", "wind_direction_10m"),
    "boundary_layer_height": ("m", "blh"),
    "surface_pressure": ("hPa", "sp"),
    "temperature_2m": ("degC", "t2m"),
}


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _fetch_grid(fx: Fetcher, source_id: str, url: str, req: RunRequest, hourly: list[str], extra: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lats, lons = grid.axes(req)
    pts = grid.points(req)
    want = pd.DatetimeIndex(grid.times(req))
    out = {v: np.full((len(want), len(lats), len(lons)), np.nan, dtype="float32") for v in hourly}

    for chunk_i, chunk in enumerate(_batched(pts, BATCH)):
        params = {
            "latitude": ",".join(f"{la:g}" for la, _ in chunk),
            "longitude": ",".join(f"{lo:g}" for _, lo in chunk),
            "start_date": req.start,
            "end_date": req.end,
            "hourly": ",".join(hourly),
            "timezone": "GMT",
            **extra,
        }
        body = fx.get_json(source_id, url, params, cost=len(chunk))
        locs = body if isinstance(body, list) else [body]
        if len(locs) != len(chunk):
            raise SourceUnavailable("error", f"asked for {len(chunk)} points, got {len(locs)}")
        for k, loc in enumerate(locs):
            flat = chunk_i * BATCH + k
            i, j = divmod(flat, len(lons))
            h = loc.get("hourly") or {}
            t = pd.DatetimeIndex(pd.to_datetime(h.get("time", [])))
            pos = t.get_indexer(want)
            for v in hourly:
                series = np.array([np.nan if x is None else x for x in h.get(v, [])], dtype="float32")
                if series.size == 0:
                    continue
                ok = pos >= 0
                out[v][ok, i, j] = series[pos[ok]]
    return want.values, out


def era5_grid(fx: Fetcher, req: RunRequest) -> GridPayload:
    times, raw = _fetch_grid(fx, "open_meteo_era5", ERA5_URL, req, list(ERA5_VARS),
                             {"models": "era5", "wind_speed_unit": "ms"})
    lats, lons = grid.axes(req)
    variables = {ERA5_VARS[k][1]: v for k, v in raw.items()}
    units = {ERA5_VARS[k][1]: ERA5_VARS[k][0] for k in raw}
    return GridPayload(**doc_fields(fx, "open_meteo_era5", f"{len(lats)}x{len(lons)} points, sampled"),
                       times=times, lats=lats, lons=lons, variables=variables, units=units)


def cams_methane_grid(fx: Fetcher, req: RunRequest) -> GridPayload:
    if pd.Timestamp(req.start) < pd.Timestamp("2025-01-01"):
        # Tested 2026-11-25: requests before early 2025 return all nulls. Say so instead of returning an empty field.
        raise SourceUnavailable("missing", "CAMS methane on Open-Meteo has no history before early 2025",
                                "Use the ADS (CAMS reanalysis EAC4) with a key, or pick a window from 2025 on.")
    times, raw = _fetch_grid(fx, "open_meteo_cams", CAMS_URL, req, ["methane"], {"domains": "cams_global"})
    lats, lons = grid.axes(req)
    return GridPayload(**doc_fields(fx, "open_meteo_cams", f"{len(lats)}x{len(lons)} points, sampled"),
                       times=times, lats=lats, lons=lons, variables={"methane": raw["methane"]},
                       units={"methane": "ug m-3"})
