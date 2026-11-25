"""The four gathering agents. Each owns a slice of the catalog and a scoped tool list."""
from __future__ import annotations

from ..units import wind_components
from .base import AgentContext, AgentResult, Emit, align_to_run


class ContextAgent:
    id = "context"
    purpose = "Land and coastline for the map, and an independent wind sample to cross-check ERA5."
    tools = ("natural_earth.land", "nasa_power.points")

    def run(self, ctx: AgentContext) -> AgentResult:
        e = Emit(self.id, ctx)
        land = e.call("natural_earth.land")
        if land:
            g, coast = land
            e.propose("static.land_fraction", g.variables["land_fraction"][0], g, "1", confidence=0.95,
                      rationale="Share of each cell inside Natural Earth 1:110m land polygons, from a 4x4 sample lattice.")
            if coast.features:
                e.propose("static.coastline", coast.features, coast, "1", confidence=0.95,
                          rationale="Coastline segments clipped to the run's box, for drawing only.")
            else:
                e.note("no coastline inside this box (inland region)")
        else:
            e.miss("static.land_fraction", "static.coastline")
        power = e.call("nasa_power.points")
        if power:
            e.propose("wind.power_check", power.features, power, "m s-1", confidence=0.7,
                      rationale="Daily mean 10 m wind speed from MERRA-2 at fixed sample cells. A second reanalysis, used only to check ERA5.")
        else:
            e.miss("wind.power_check")
        return e.result()


class WindAgent:
    id = "wind"
    purpose = "10 m wind, boundary layer height, surface pressure and temperature on the run grid."
    tools = ("local.era5", "cds.era5", "open_meteo.era5")

    def run(self, ctx: AgentContext) -> AgentResult:
        e = Emit(self.id, ctx)
        # Native u/v first, where a person has supplied them or a key opens them.
        for tool, conf, why in (
            ("local.era5", 0.95, "ERA5 u10/v10 read directly from the user's files."),
            ("cds.era5", 0.95, "ERA5 u10/v10 from the Copernicus CDS at the run resolution."),
        ):
            g = e.call(tool)
            if g:
                e.propose("wind.u10", g.variables["u10"], g, g.units["u10"], confidence=conf, rationale=why)
                e.propose("wind.v10", g.variables["v10"], g, g.units["v10"], confidence=conf, rationale=why)

        # Open-Meteo always runs: it is the only source of BLH, pressure and temperature here, and a second wind.
        g = e.call("open_meteo.era5")
        if g:
            u, v = wind_components(g.variables["wind_speed_10m"], g.variables["wind_direction_10m"])
            why = ("ERA5 via Open-Meteo, sampled at the nearest 0.25 deg cell. Converted from speed and direction "
                   "(direction is where the wind comes from), so u and v carry Open-Meteo's 1-degree rounding of direction.")
            e.propose("wind.u10", u.astype("float32"), g, "m s-1", confidence=0.85, rationale=why)
            e.propose("wind.v10", v.astype("float32"), g, "m s-1", confidence=0.85, rationale=why)
            e.propose("met.blh", g.variables["blh"], g, g.units["blh"], confidence=0.8,
                      rationale="ERA5 boundary layer height, sampled at the nearest cell.")
            e.propose("met.sp", g.variables["sp"], g, g.units["sp"], confidence=0.9,
                      rationale="ERA5 surface pressure; needed to turn CH4 mass into a mole fraction.")
            e.propose("met.t2m", g.variables["t2m"], g, g.units["t2m"], confidence=0.9,
                      rationale="ERA5 2 m temperature; needed for the same conversion.")
        else:
            e.miss("met.blh", "met.sp", "met.t2m")
        if not any(p.field == "wind.u10" for p in e.proposals):
            e.miss("wind.u10", "wind.v10")
        return e.result()


class MethaneAgent:
    id = "methane"
    purpose = "Surface and column CH4 on the run grid, plus the global background to judge them by."
    tools = ("local.ch4", "open_meteo.cams", "gee.tropomi", "noaa_gml.monthly")

    def run(self, ctx: AgentContext) -> AgentResult:
        e = Emit(self.id, ctx)
        req = ctx.request
        g = e.call("local.ch4")
        if g:
            e.propose("ch4.surface_ppb", align_to_run(g, g.variables["ch4"], req), g, g.units["ch4"], confidence=0.8,
                      rationale="The user's daily CH4 files, one field per day, applied to every run time on that day.")
        g = e.call("open_meteo.cams")
        if g:
            e.propose("ch4.surface_mass", g.variables["methane"], g, g.units["methane"], confidence=0.75,
                      rationale="CAMS global composition at the lowest model level, via Open-Meteo, nearest cell.")
        elif "ch4.surface_ppb" not in [p.field for p in e.proposals]:
            e.miss("ch4.surface_mass", "ch4.surface_ppb")
        g = e.call("gee.tropomi")
        if g:
            e.propose("ch4.column_ppb", align_to_run(g, g.variables["xch4"], req), g, g.units["xch4"], confidence=0.8,
                      rationale="TROPOMI XCH4 daily L3 mean. A column average, not a surface value.")
        else:
            e.miss("ch4.column_ppb")
        s = e.call("noaa_gml.monthly")
        if s:
            e.propose("ch4.global_mean_ppb", s.values, s, "ppb", confidence=0.95,
                      rationale=f"NOAA marine surface global mean for {s.values['month']}"
                                + (" (preliminary)" if s.values.get("preliminary") else "") + ".")
        else:
            e.miss("ch4.global_mean_ppb")
        return e.result()


class EmissionsAgent:
    id = "emissions"
    purpose = "Where CH4 is emitted: a gridded inventory, named assets and facilities, and observed plumes."
    tools = ("edgar.flux", "climate_trace.assets", "epa_ghgrp.facilities", "carbon_mapper.plumes")

    def run(self, ctx: AgentContext) -> AgentResult:
        e = Emit(self.id, ctx)
        g = e.call("edgar.flux")
        if g:
            year = str(g.times[0])[:4]
            e.propose("ch4.emission_flux", g.variables["emission_flux"][0], g, g.units["emission_flux"], confidence=0.6,
                      rationale=f"EDGAR v8.0 annual total for {year}, mass-summed onto the run grid. "
                                f"Inventory year differs from the wind window; treat as a prior, not a measurement.")
        else:
            e.miss("ch4.emission_flux")
        for tool, key, unit, conf, why in (
            ("climate_trace.assets", "emitters.assets", "t CH4 yr-1", 0.5,
             "Climate TRACE modelled asset-level CH4 for 2024. Each asset carries its own confidence grade."),
            ("epa_ghgrp.facilities", "emitters.facilities", "t CO2e yr-1", 0.8,
             "Facility-reported CH4 under the EPA GHGRP, as CO2e; summed over each facility's sectors."),
            ("carbon_mapper.plumes", "plumes.detections", "kg h-1", 0.7,
             "Airborne/satellite plume detections in the run window, with emission rate and the plume's own wind estimate."),
        ):
            p = e.call(tool)
            if p:
                e.propose(key, p.features, p, unit, confidence=conf, rationale=why)
            else:
                e.miss(key)
        return e.result()


DISCOVERY_AGENTS = (ContextAgent, WindAgent, MethaneAgent, EmissionsAgent)
