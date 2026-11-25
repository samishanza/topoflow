"""Point and scalar sources: cross-check winds, background CH4, emitters and plumes."""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .. import grid
from ..contract import PointsPayload, RunRequest, ScalarPayload
from ..http import Fetcher
from .base import SourceUnavailable, doc_fields

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
GML_URL = "https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.txt"
CTRACE_URL = "https://api.climatetrace.org/v6/assets"
EPA_BASE = "https://data.epa.gov/efservice"
CMAPPER_URL = "https://api.carbonmapper.org/api/v1/catalog/plumes/annotated"

# Climate TRACE asks for low volume. These caps are per run.
CTRACE_PAGE = 100
CTRACE_MAX_PAGES = 6
CTRACE_SECTORS = ["fossil-fuel-operations", "waste", "agriculture"]
CMAPPER_PAGE = 500
CMAPPER_MAX_PAGES = 4
GHGRP_LATEST_YEAR = 2023  # 2024 had no sector rows when tested on 2026-11-25.

# ISO-3 countries whose land touches each named region. Climate TRACE has no area filter, so we ask by country.
REGION_COUNTRIES = {
    "permian": ["USA"], "gulf_coast": ["USA", "MEX"], "turkmenistan": ["TKM", "IRN", "UZB", "KAZ", "AFG"],
    "sudd": ["SSD", "SDN", "ETH", "UGA", "COD", "CAF"],
}


def _in_bbox(lon: float, lat: float, bbox) -> bool:
    w, s, e, n = bbox
    return w <= lon <= e and s <= lat <= n


# ----------------------------------------------------------------------------------------- NASA POWER


def power_wind_points(fx: Fetcher, req: RunRequest, n: int = 12) -> PointsPayload:
    feats = []
    start = pd.Timestamp(req.start).strftime("%Y%m%d")
    end = pd.Timestamp(req.end).strftime("%Y%m%d")
    for lat, lon in grid.sample_points(req, n=n):
        body = fx.get_json("nasa_power", POWER_URL, {
            "parameters": "WS10M", "community": "RE", "longitude": f"{lon:g}", "latitude": f"{lat:g}",
            "start": start, "end": end, "format": "JSON",
        })
        series = body.get("properties", {}).get("parameter", {}).get("WS10M", {})
        fill = body.get("header", {}).get("fill_value", -999.0)
        daily = {d: v for d, v in series.items() if v is not None and v != fill}
        if daily:
            feats.append({"lon": lon, "lat": lat, "properties": {"daily_ws10m": daily, "unit": "m s-1"}})
    if not feats:
        raise SourceUnavailable("missing", "NASA POWER returned no wind for any sample point")
    return PointsPayload(**doc_fields(fx, "nasa_power", f"{len(feats)} sample points"), features=feats)


# ----------------------------------------------------------------------------------------- NOAA GML


def parse_gml_monthly(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        year, month, _decimal, average, unc = parts[:5]
        rows.append({"year": int(year), "month": int(month), "ppb": float(average),
                     "uncertainty": None if float(unc) < 0 else float(unc), "preliminary": float(unc) < 0})
    return rows


def noaa_global_ch4(fx: Fetcher, req: RunRequest) -> ScalarPayload:
    rows = parse_gml_monthly(fx.get_text("noaa_gml", GML_URL))
    t = pd.Timestamp(req.start)
    match = [r for r in rows if r["year"] == t.year and r["month"] == t.month]
    if not match:
        raise SourceUnavailable("missing", f"NOAA GML has no global mean for {t:%Y-%m} yet")
    r = match[0]
    return ScalarPayload(**doc_fields(fx, "noaa_gml", f"{t:%Y-%m}"),
                         values={"ppb": r["ppb"], "uncertainty": r["uncertainty"], "preliminary": r["preliminary"],
                                 "month": f"{t:%Y-%m}"})


# ----------------------------------------------------------------------------------------- Climate TRACE


def climate_trace_assets(fx: Fetcher, req: RunRequest, year: int = 2024) -> PointsPayload:
    countries = REGION_COUNTRIES.get(req.region)
    feats, seen = [], set()
    for sector in CTRACE_SECTORS:
        for page in range(CTRACE_MAX_PAGES):
            params = {"limit": CTRACE_PAGE, "offset": page * CTRACE_PAGE, "gas": "ch4", "year": year, "sectors": sector}
            if countries:
                params["countries"] = ",".join(countries)
            body = fx.get_json("climate_trace", CTRACE_URL, params)
            assets = body.get("assets") or []
            for a in assets:
                geom = (a.get("Centroid") or {}).get("Geometry")
                if not geom or a.get("Id") in seen:
                    continue
                lon, lat = float(geom[0]), float(geom[1])
                if not _in_bbox(lon, lat, req.bbox):
                    continue
                ch4 = [e.get("EmissionsQuantity") for e in a.get("EmissionsSummary", []) if e.get("Gas") == "ch4"]
                if not ch4 or ch4[0] is None:
                    continue
                seen.add(a.get("Id"))
                feats.append({"lon": lon, "lat": lat, "properties": {
                    "name": a.get("Name"), "sector": a.get("Sector"), "country": a.get("Country"),
                    "t_ch4_per_year": float(ch4[0]), "year": year, "asset_id": a.get("Id"),
                    "confidence": ((a.get("Confidence") or [{}])[0].get(str(year)) or [{}])[0].get("ch4_emissions"),
                }})
            if len(assets) < CTRACE_PAGE:
                break
    if not feats:
        raise SourceUnavailable("missing", "Climate TRACE has no CH4 assets with coordinates in this region")
    return PointsPayload(**doc_fields(fx, "climate_trace", f"{len(feats)} assets, year {year}"), features=feats)


# ----------------------------------------------------------------------------------------- EPA GHGRP


def epa_ghgrp_facilities(fx: Fetcher, req: RunRequest, year: int = GHGRP_LATEST_YEAR) -> PointsPayload:
    w, s, e, n = req.bbox
    # Rough US extent incl. Alaska and Hawaii. Outside it the source has nothing to say.
    if e < -180 or w > -60 or n < 18 or s > 72:
        raise SourceUnavailable("skipped", "GHGRP covers US facilities only; this region is outside the US")
    path = (f"{EPA_BASE}/PUB_DIM_FACILITY/YEAR/=/{year}/LATITUDE/>/{s}/LATITUDE/</{n}/LONGITUDE/>/{w}/LONGITUDE/</{e}"
            f"/PUB_FACTS_SECTOR_GHG_EMISSION/GAS_ID/=/2/JSON")
    rows = fx.get_json("epa_ghgrp", path, timeout=300)
    by_fac: dict[int, dict] = {}
    totals: dict[int, float] = defaultdict(float)
    for r in rows:
        if r.get("latitude") is None or r.get("longitude") is None:
            continue
        fid = r["facility_id"]
        totals[fid] += float(r.get("co2e_emission") or 0)
        by_fac.setdefault(fid, r)
    feats = [{"lon": float(r["longitude"]), "lat": float(r["latitude"]), "properties": {
        "name": r.get("facility_name"), "state": r.get("state"), "facility_id": fid, "year": year,
        "t_co2e_ch4_per_year": round(totals[fid], 2), "industry": r.get("reported_industry_types"),
    }} for fid, r in by_fac.items()]
    if not feats:
        raise SourceUnavailable("missing", f"no GHGRP facility reported CH4 in this box for {year}")
    return PointsPayload(**doc_fields(fx, "epa_ghgrp", f"{len(feats)} facilities, {year}"), features=feats)


# ----------------------------------------------------------------------------------------- Carbon Mapper


def carbon_mapper_plumes(fx: Fetcher, req: RunRequest, pad_days: int = 0) -> PointsPayload:
    w, s, e, n = req.bbox
    start = (pd.Timestamp(req.start) - pd.Timedelta(days=pad_days)).strftime("%Y-%m-%dT00:00:00Z")
    end = (pd.Timestamp(req.end) + pd.Timedelta(days=1 + pad_days)).strftime("%Y-%m-%dT00:00:00Z")
    feats = []
    for page in range(CMAPPER_MAX_PAGES):
        # The API takes bbox as four repeated parameters; requests encodes a list that way.
        params = {"bbox": [w, s, e, n], "datetime": f"{start}/{end}", "limit": CMAPPER_PAGE,
                  "offset": page * CMAPPER_PAGE}
        body = fx.get_json("carbon_mapper", CMAPPER_URL, params)
        items = body.get("items") or []
        for it in items:
            if it.get("gas") != "CH4" or it.get("hide_emission"):
                continue
            coords = (it.get("geometry_json") or {}).get("coordinates")
            if not coords:
                continue
            feats.append({"lon": float(coords[0]), "lat": float(coords[1]), "properties": {
                "plume_id": it.get("plume_id"), "time": it.get("scene_timestamp"),
                "kg_per_hour": it.get("emission_auto"), "kg_per_hour_unc": it.get("emission_uncertainty_auto"),
                "plume_wind_speed": it.get("wind_speed_avg_auto"), "plume_wind_dir": it.get("wind_direction_avg_auto"),
                "sector": it.get("sector"), "platform": it.get("platform"), "status": it.get("status"),
            }})
        if len(items) < CMAPPER_PAGE:
            break
    if not feats:
        raise SourceUnavailable("missing", "Carbon Mapper published no CH4 plumes in this box and window")
    return PointsPayload(**doc_fields(fx, "carbon_mapper", f"{len(feats)} plumes"), features=feats)
