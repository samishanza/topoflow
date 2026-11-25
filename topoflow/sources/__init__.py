"""The source seam. Agents reach the outside world only through SourceClient.call(tool, ...).

The same agent code runs against live APIs, recorded fixtures, or the labelled synthetic client in tests.
Every call leaves a SourceRun behind, whether it answered, was missing, was barred, or failed.
"""
from __future__ import annotations

from typing import Any, Callable

from ..contract import RunRequest, SourceRun
from ..http import DeadlineExceeded, Fetcher, NoRecording
from . import bulk, keyed, open_meteo, points
from .base import SourceUnavailable
from .registry import SOURCES

# tool name -> (source id, adapter)
TOOLS: dict[str, tuple[str, Callable[..., Any]]] = {
    "open_meteo.era5": ("open_meteo_era5", open_meteo.era5_grid),
    "open_meteo.cams": ("open_meteo_cams", open_meteo.cams_methane_grid),
    "nasa_power.points": ("nasa_power", points.power_wind_points),
    "noaa_gml.monthly": ("noaa_gml", points.noaa_global_ch4),
    "climate_trace.assets": ("climate_trace", points.climate_trace_assets),
    "epa_ghgrp.facilities": ("epa_ghgrp", points.epa_ghgrp_facilities),
    "carbon_mapper.plumes": ("carbon_mapper", points.carbon_mapper_plumes),
    "edgar.flux": ("edgar", bulk.edgar_flux),
    "natural_earth.land": ("natural_earth", bulk.natural_earth_land),
    "cds.era5": ("cds_era5", keyed.cds_era5_wind),
    "gee.tropomi": ("gee_tropomi", keyed.gee_tropomi_xch4),
    "local.era5": ("local_era5", keyed.local_era5_wind),
    "local.ch4": ("local_ch4", keyed.local_ch4),
}


class SourceClient:
    def __init__(self, fetcher: Fetcher, request: RunRequest, tools: dict[str, tuple[str, Callable[..., Any]]] | None = None,
                 skip: set[str] | None = None):
        self.fx = fetcher
        self.req = request
        self.tools = tools or TOOLS
        self.skip = skip or set()
        self.runs: dict[str, SourceRun] = {}

    def call(self, tool: str, **kwargs) -> Any | None:
        if tool not in self.tools:
            raise KeyError(f"unknown tool {tool!r}")
        source_id, fn = self.tools[tool]
        entry = SOURCES.get(source_id)
        if source_id in self.skip:
            self._record(source_id, "skipped", "skipped by --skip", entry.person_route if entry else "")
            return None
        try:
            result = fn(self.fx, self.req, **kwargs)
        except SourceUnavailable as exc:
            self._record(source_id, exc.status, exc.reason, exc.person_route or (entry.person_route if entry else ""))
            return None
        except NoRecording as exc:
            self._record(source_id, "missing", str(exc))
            return None
        except DeadlineExceeded as exc:
            self._record(source_id, "error", str(exc))
            return None
        except Exception as exc:  # noqa: BLE001 - a source failing must never stop the run; it is reported instead
            self._record(source_id, "error", f"{type(exc).__name__}: {exc}"[:400])
            return None
        self._record(source_id, "ok", "")
        return result

    def _record(self, source_id: str, status, reason: str, person_route: str = ""):
        log = self.fx.log(source_id)
        prev = self.runs.get(source_id)
        # A source called twice (e.g. land grid and coastline) keeps its worst status.
        order = {"ok": 0, "skipped": 1, "missing": 2, "barred": 3, "error": 4}
        if prev and order[prev.status] > order[status]:
            status, reason, person_route = prev.status, prev.reason, prev.person_route
        self.runs[source_id] = SourceRun(source_id=source_id, status=status, reason=reason, calls=log.calls,
                                         bytes=log.bytes, ms=log.ms, person_route=person_route)
