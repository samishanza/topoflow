# Wind and methane datasets: coverage, access, terms

Date: 2026-11-25. Research input for the registry (`topoflow/sources/registry.py`).

**Labels:** **Tested** means queried from this machine and admitted in a run. **Read** means the provider's pages were
read but no run used it. **Not tested** means named but not yet tried. Terms were read on the provider's own pages;
where a page did not state something, it says **unverified**.

## In the pipeline

| Source | Gives TopoFlow | Access | Status | Notes from testing |
|---|---|---|---|---|
| Open-Meteo Historical (ERA5) | 10 m speed + direction, BLH, surface pressure, 2 m temperature | API, no key; many points per request | **Tested** | 100 points per request; each point costs one call. Nearest 0.25 deg cell. Agrees with CDS ERA5 at relative RMS 0.008 (u), 0.009 (v) |
| Open-Meteo Air Quality (CAMS global) | Surface CH4, ug m-3 | API, no key | **Tested** | All nulls for 2022-09, 2023-06, 2024-10; values from 2025-03 on. Shares the rate budget with the weather API |
| Copernicus CDS ERA5 single levels | Native u10/v10 | `cdsapi`, account + token, licence accepted once | **Tested** (credentials already on this Mac) | A 13 x 13 x 28 request returned in seconds |
| NASA POWER daily point | WS10M (MERRA-2) at sample cells | API, no key | **Tested** | Used only to cross-check ERA5: 0.69 m/s median diff (Permian), 0.74 (global) |
| NOAA GML global monthly CH4 | Marine background, ppb | Static text file | **Tested** | Latest months preliminary (uncertainty -9.99); July 2025 = 1,927 ppb at the time of the run |
| Climate TRACE v6 assets | Asset-level CH4, t/yr, with confidence | API, no key, beta | **Tested** | No area filter (`bbox` returns no `assets`); filtered by country, then box. Capped at 6 pages x 3 sectors |
| EPA GHGRP (Envirofacts) | Facility CH4 as CO2e, US | API, no key | **Tested** | Range filters and a join work in one path. 2024 had 0 sector rows; 2023 is used. 284 rows in the Permian box |
| Carbon Mapper plumes | Plume points, kg/h, plume wind | API, no key | **Tested** | `bbox` as four repeated params and `datetime` interval work. 47,102 plumes in the catalogue |
| EDGAR v8.0 CH4 totals | 0.1 deg annual emissions | Bulk zip, about 15 MB per year | **Tested** | Latest year in v8.0 is 2022. Reduced to a 14 MB float32 NetCDF in the cache; raw files deleted |
| Natural Earth 1:110m land | Land fraction, coastline | Static GeoJSON | **Tested** | 138 KB |
| Sentinel-5P TROPOMI L3 CH4 (Earth Engine) | XCH4 column, daily | `earthengine-api` + registered Cloud project | **Read**, adapter written, **not tested** | Barred until `EARTHENGINE_PROJECT` is set |
| Local ERA5 / CH4 NetCDF | User's own files | Environment variables | **Not tested** on real files | The original files were no longer in Downloads |

## Terms (read 2026-11-25)

| Source | Licence | Automated access | Restrictions | Attribution | Verified |
|---|---|---|---|---|---|
| Open-Meteo (both APIs) | CC BY 4.0 | Yes | Free API non-commercial; 600/min, 5,000/h, 10,000/day, 300,000/month | "Weather data by Open-Meteo.com" + licence link + changes | Yes |
| NASA POWER | Not named on the pages read | Yes | 429 when overused; no number published | NASA LaRC POWER Project sentence, with version and access date | Partly (licence unverified) |
| NOAA GML | Freely available | Yes | None stated | Cite in publications | Yes |
| Climate TRACE | CC BY 4.0 (some embedded datasets differ) | Yes, beta API | "Keep volume low" | Credit Climate TRACE, note changes | Partly (API docs 404) |
| Carbon Mapper | Own licence: revocable, **non-commercial, share-alike** | Yes, no bots that impair service | Redistribution must carry the same terms | "Data by Carbon Mapper®" or similar | Yes (terms effective 2026-01-13) |
| EPA Envirofacts | Not stated for this API; EPA geospatial data is public domain by default | Yes | Requests under 15 minutes | None stated | Partly |
| EDGAR v8.0 | CC BY 4.0, (c) European Union | Yes | None | Credit, note changes | Yes (`copyright.txt`) |
| Natural Earth | Public domain | Yes | None | Optional | Yes |
| Copernicus CDS / ADS | CC BY 4.0 | Yes, with token | Account; accept each dataset's licence once | Per CC BY | Yes |
| TROPOMI via Earth Engine | Copernicus Sentinel terms (open) | Yes, with a Cloud project | Earth Engine free for non-commercial and research only | "Contains modified Copernicus Sentinel data [year]" | Yes |

## Worth adding next

| Dataset | Why | Access, as understood (**not tested**) |
|---|---|---|
| CAMS global reanalysis EAC4 (ADS) | Surface and column CH4 before 2025, consistent with ERA5 | ADS account; same `cdsapi` client pointed at the ADS |
| NOAA GML in-situ tower and flask sites | Point observations to check CAMS surface CH4 directly | ObsPack files (registration) |
| GOSAT / GOSAT-2 XCH4 | Longer column record than TROPOMI | NIES GDAS (registration) |
| WAD2M wetland extent | Natural CH4 source context (Sudd, Gulf coast) | Zenodo download |
| IMEO MARS / Methane Alert notifications | Plumes from UNEP's system, cross-check for Carbon Mapper | Public data portal; terms unread |
| ERA5 100 m wind | Flow above the surface layer, where plumes travel | Same CDS request, extra variables |

Use the `source-scout` subagent to research each one before writing an adapter.
