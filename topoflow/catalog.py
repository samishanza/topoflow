"""The field catalog: every value the record can hold, its unit, its plausible range, and who may propose it.

Rule: agents PROPOSE, the gatekeeper validates, deterministic code DECIDES and DERIVES.
No agent writes a field directly. Every proposal carries evidence or it is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Cheapest access first. The orchestrator spends in this order.
#   0  open API, no key        (Open-Meteo, NOAA GML, NASA POWER, Climate TRACE, EPA, Carbon Mapper)
#   1  keyed API               (Copernicus CDS/ADS, Google Earth Engine)
#   2  bulk download           (EDGAR gridded inventory, Natural Earth)
#   3  a person has to act     (request access, accept a licence, send a file)
Tier = Literal[0, 1, 2, 3]

# What sort of evidence a source is. Kept apart because they fail differently:
# a reanalysis is smooth and complete but modelled; a satellite retrieval is observed but patchy;
# an inventory is an annual estimate, not a measurement.
DataKind = Literal[
    "reanalysis",       # ERA5, MERRA-2 (via NASA POWER)
    "model_analysis",   # CAMS global composition analysis/forecast
    "satellite",        # TROPOMI
    "in_situ",          # NOAA GML flask/in-situ network
    "inventory",        # EDGAR, Climate TRACE asset estimates
    "facility_report",  # EPA GHGRP
    "plume_detection",  # Carbon Mapper
    "basemap",          # Natural Earth
    "local_file",       # files the user supplies
    "derived",          # computed here, from accepted fields
    "synthetic",        # fabricated for tests and the oversight demo; always labelled
]

AgentId = Literal["context", "wind", "methane", "emissions", "derive", "review"]
Shape = Literal["grid", "points", "scalar"]


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    unit: str
    shape: Shape
    allowed_agents: tuple[AgentId, ...]
    # Physical plausibility. Values outside are impossible, not merely unusual.
    min: float | None = None
    max: float | None = None
    # Share of grid cells that may be NaN before the proposal is refused as too patchy.
    max_nan_fraction: float = 0.02
    # Share of finite cells that may fall outside min..max before the proposal is refused.
    # Zero tolerance would let one corrupt cell sink a global field; a few percent would let a doctored one in.
    max_out_of_bounds_fraction: float = 0.001
    description: str = ""
    # Sources in the order the reconciler prefers them when more than one is admitted.
    preference: tuple[str, ...] = field(default_factory=tuple)
    # A grid with no time axis (lat, lon): basemaps and annual inventories.
    static: bool = False


FIELDS: dict[str, FieldSpec] = {
    f.key: f
    for f in [
        # ---------------------------------------------------------------- wind and boundary layer
        FieldSpec("wind.u10", "10 m eastward wind", "m s-1", "grid", ("wind",), -75, 75,
                  description="Eastward component at 10 m. Open-Meteo serves speed and direction; u is converted from them.",
                  preference=("local_era5", "cds_era5", "open_meteo_era5")),
        FieldSpec("wind.v10", "10 m northward wind", "m s-1", "grid", ("wind",), -75, 75,
                  preference=("local_era5", "cds_era5", "open_meteo_era5")),
        FieldSpec("met.blh", "Boundary layer height", "m", "grid", ("wind",), 0, 6500,
                  description="Depth of the mixed layer. A shallow layer under a convergent flow traps surface CH4."),
        FieldSpec("met.sp", "Surface pressure", "Pa", "grid", ("wind",), 45_000, 110_000,
                  description="Needed to turn a CH4 mass concentration into a mole fraction."),
        FieldSpec("met.t2m", "2 m temperature", "K", "grid", ("wind",), 180, 340),
        FieldSpec("wind.power_check", "Daily 10 m wind speed at sample points (NASA POWER, MERRA-2)", "m s-1",
                  "points", ("context",), 0, 75,
                  description="An independent reanalysis, used only to cross-check the ERA5 wind. Never drawn as the wind field."),
        # ---------------------------------------------------------------- methane
        FieldSpec("ch4.surface_mass", "Surface CH4 mass concentration (CAMS)", "ug m-3", "grid", ("methane",), 300, 8000,
                  max_nan_fraction=0.05,
                  description="CAMS global analysis at the lowest model level, as served by Open-Meteo."),
        FieldSpec("ch4.surface_ppb", "Surface CH4 mole fraction", "ppb", "grid", ("methane", "derive"), 1500, 5000,
                  max_nan_fraction=0.05,
                  description="From a local file, or derived from the CAMS mass concentration using ERA5 surface pressure and temperature.",
                  preference=("local_ch4", "derived")),
        FieldSpec("ch4.column_ppb", "Column-average dry-air CH4 (XCH4, TROPOMI)", "ppb", "grid", ("methane",), 1500, 2300,
                  max_nan_fraction=1.0,  # Clouds and glint leave most of a daily L3 grid empty. That is the data, not a fault.
                  description="Satellite column, not a surface value. Shown as its own layer and never mixed into surface CH4."),
        FieldSpec("ch4.global_mean_ppb", "Global monthly mean CH4 (NOAA GML marine surface)", "ppb", "scalar", ("methane",), 1500, 2500,
                  description="Background reference for the plausibility check on any surface CH4 field."),
        # ---------------------------------------------------------------- emissions
        FieldSpec("ch4.emission_flux", "CH4 emission flux (EDGAR, annual)", "kg m-2 s-1", "grid", ("emissions",), 0, 1e-5,
                  max_nan_fraction=0.0, static=True,
                  description="Annual inventory, regridded by summing mass. Its year usually differs from the wind window; the record says so."),
        FieldSpec("emitters.assets", "Methane-emitting assets (Climate TRACE)", "t CH4 yr-1", "points", ("emissions",), 0, 5e7),
        FieldSpec("emitters.facilities", "Facilities reporting CH4 (EPA GHGRP, US only)", "t CO2e yr-1", "points", ("emissions",), 0, 5e7),
        FieldSpec("plumes.detections", "Detected CH4 plumes (Carbon Mapper)", "kg h-1", "points", ("emissions",), 0, 1e6,
                  description="Point-source plume detections with the plume's own wind estimate, used to cross-check ERA5."),
        # ---------------------------------------------------------------- context
        FieldSpec("static.land_fraction", "Land fraction (Natural Earth 1:110m)", "1", "grid", ("context",), 0, 1, max_nan_fraction=0.0, static=True),
        FieldSpec("static.coastline", "Coastline polylines (Natural Earth 1:110m)", "1", "points", ("context",)),
        # ---------------------------------------------------------------- derived (deterministic code only)
        FieldSpec("flow.divergence", "Horizontal divergence of 10 m wind", "s-1", "grid", ("derive",), -1e-3, 1e-3,
                  max_nan_fraction=0.05),
        FieldSpec("flow.vorticity", "Relative vorticity of 10 m wind", "s-1", "grid", ("derive",), -1e-3, 1e-3,
                  max_nan_fraction=0.05),
        FieldSpec("topology.critical_points", "Critical points of the 10 m wind (nodes, saddles, foci)", "1", "points", ("derive",),
                  description="Where the piecewise-linear wind vanishes, classified by its Jacobian. Inferred, never observed."),
    ]
}


def spec_for(key: str) -> FieldSpec | None:
    return FIELDS.get(key)


def is_agent_allowed(agent_id: str, key: str) -> bool:
    spec = FIELDS.get(key)
    return bool(spec) and agent_id in spec.allowed_agents
