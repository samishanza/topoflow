# Status and data flow

## One run

```
 request (region, bbox, grid, window, hours)
        │
        ▼  preflight: disk guard, budgets scaled to the grid
 ┌──────────────── discovery agents, cheapest first ───────────────┐
 │ context   natural_earth.land, nasa_power.points                  │
 │ wind      local.era5, cds.era5, open_meteo.era5                  │
 │ methane   local.ch4, open_meteo.cams, gee.tropomi, noaa_gml      │
 │ emissions edgar.flux, climate_trace, epa_ghgrp, carbon_mapper    │
 └──────────────────────────────────────────────────────────────────┘
        │ proposals, each with Evidence (source, URLs, time, licence, origin)
        ▼
 gate(): scope · evidence · unit (convert or refuse) · shape · NaN share · physical range · cross-field physics
        │ admitted                          │ refused (codes + reasons, kept in the manifest)
        ▼
 reconcile(): choose by source preference, keep alternatives, measure disagreement (conflicting if rel. RMS > 0.5)
        ▼
 derive(): CH4 ppb from ug m-3 with p and T · divergence · vorticity · critical points (+ CH4 anomaly, BLH at each)
        ▼  (derived proposals go through gate() too)
 cross-checks: ERA5 vs NASA POWER · CH4 vs NOAA background · plume wind vs ERA5 · emitters vs EDGAR · CH4 at critical points
        ▼
 data/runs/<run_id>/  fields.nc · points.geojson · manifest.json        ──►  app/dashboard.py
        ▼
 optional: topoflow review (Claude proposes flags) → topoflow decide (a person accepts or rejects, with a reason)
```

## Field states

`observed` (read from a source) · `derived` (computed here from admitted fields) · `conflicting` (two admitted sources
disagree beyond sampling noise; the chosen one is drawn, the other is kept) · `missing` (no source answered; the
manifest says who tried and why).

## Source states

`ok` · `missing` (no data for this place or time, or not recorded in replay) · `barred` (needs a key or a person: the
manifest gives the route) · `skipped` (not configured, or outside its coverage) · `error` (failed; the reason is kept).

## First runs (2026-11-25)

**Permian, 0.5 degrees, 2025-07-01..07, 4 times a day, recorded.** 11 of 13 sources answered (TROPOMI barred without
Earth Engine; local files not configured). CDS ERA5 and Open-Meteo ERA5 agreed (relative RMS 0.008 / 0.009).
Cross-checks: ERA5 vs NASA POWER median |diff| 0.69 m/s, r = 0.65 (pass); CH4 regional mean 2,355 ppb vs NOAA 1,927 ppb,
1.22x (pass); plume wind vs ERA5 0.15 m/s over 13 plumes (pass, likely not independent); 79% of 417 named emitters in
EDGAR's top quartile (pass). 81 critical points (41 saddles, 15 attracting nodes, 14 attracting foci, 6 repelling
nodes, 5 repelling foci), found in 20 of the 28 time steps.

**Global, 5 degrees, same window, live.** Winds, NASA POWER, EDGAR, Climate TRACE (9,163 emitters with EPA), and 134
Carbon Mapper plumes admitted; ERA5 vs POWER 0.74 m/s, r = 0.52 (pass). CAMS methane failed on Open-Meteo's
per-minute limit because the fetcher tracked the two Open-Meteo hosts separately. Fixed (budgets are per provider,
per-minute 429s wait and retry, record mode resumes); see the rerun in the plan's Handoffs.
