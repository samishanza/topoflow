# Decisions

Open decisions, assumptions, and team choices. Context is in [context.md](context.md).

**Current phase:** Build, under [the multi-source collection plan](../plans/2026-11-25-feat-multi-source-collection-plan.md).

| ID | Decision | Status | Owner |
|---|---|---|---|
| D1 | Purpose: relate wind-field topology to CH4, as a research view; no facility emission estimates, compliance or health claims | Decided 2026-11-25 | Lead |
| D2 | Stack: Python 3.10+, numpy/xarray/netCDF4, pydantic for the contract, requests, shapely, Dash/Plotly. No database: a run is a folder | Decided 2026-11-25 | Lead |
| D3 | Architecture copied from the reference orchestration (agents propose with evidence, a deterministic gate admits, code reconciles and derives, a Claude review agent proposes, a person decides) | Decided 2026-11-25 | Lead |
| D4 | Default demo: Permian Basin, 0.5 degrees, 2025-07-01..07, 00/06/12/18 UTC. CAMS methane on Open-Meteo has no history before early 2025 (tested), so the window sits after that | Decided 2026-11-25 | Lead |
| D5 | Owner of the shared contract (`catalog.py`, `contract.py`) and of dependency additions: the session on `main` | Decided | Lead |
| D6 | Writing-task isolation: named worktrees under `.worktrees/` | Decided | Lead |
| D7 | Wind source preference: user's local ERA5, then CDS ERA5 (native u/v), then Open-Meteo ERA5 (converted from speed/direction). Open-Meteo always runs, because it is the only source of BLH, pressure and temperature | Decided | Lead |
| D8 | Recording policy: replayable fixtures for the Permian demo are committed; Carbon Mapper responses and bulk/keyed results (EDGAR, CDS) stay in the gitignored cache; global recordings go to `data/cache/fixtures-global` | Decided, follows the terms check | Lead |
| D9 | Licence posture: research, private repo. Two sources are non-commercial (Carbon Mapper, Open-Meteo free tier) and Earth Engine is free only for non-commercial use. Revisit before making the repo public or any commercial use | Decided, revisit | Lead |
| D10 | Review agent model: `claude-opus-5-5`, effort `high`, server-side refusal fallback on (`fallbacks: "default"`). It proposes only; decisions are attributed and final | Decided | Lead |
| D11 | Disk guard: refuse to start with less than 400 MB spare after the run's estimate | Decided | Lead |
| D12 | Which hypothesis test to run on CH4 at critical points (window length, resolution, null model) | **Open**: design before looking at more data | Team |
| D13 | Add TROPOMI via Earth Engine (needs a registered noncommercial Cloud project) | **Open**: needs a person to register | Owner of the Google account |
| D14 | Add CAMS EAC4 reanalysis via the ADS for CH4 before 2025 (needs ADS account and licence acceptance) | **Open** | Lead |

## Assumptions

- **A1** Open-Meteo returns ERA5 at the nearest 0.25 degree cell to each requested point. Agreement with CDS ERA5 at
  0.5 degrees (relative RMS 0.008 for u, 0.009 for v, Permian run) supports this.
- **A2** NOAA GML's marine global mean is a fair background for the plausibility gate. Regions above 1.6x it are
  refused; that bound is generous on purpose and has not been tuned on real outliers.
- **A3** Carbon Mapper's `wind_speed_avg_auto` comes from a weather model, not an independent measurement. Unverified;
  the Permian run agreed with ERA5 to 0.15 m/s, the global run to 1.20 m/s.
