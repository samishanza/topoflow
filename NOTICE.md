# Third-party data and attribution

TopoFlow's code fetches data from the providers below at run time. A few small files from some of them are bundled
in this repository so the tests and the offline demo run without a network. Each bundled file keeps its provider's
licence; nothing here relicenses it.

## Bundled in this repository

| Path | Provider and licence | Changes made |
|---|---|---|
| `fixtures/open_meteo_era5/`, `fixtures/open_meteo_cams/` | Weather and air-quality data by [Open-Meteo.com](https://open-meteo.com/), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Underlying data: ERA5 (ECMWF / Copernicus Climate Change Service) and CAMS (Copernicus Atmosphere Monitoring Service). | API responses recorded as returned, wrapped with URL and retrieval time, gzipped. |
| `fixtures/climate_trace/` | [Climate TRACE](https://climatetrace.org), [CC BY 4.0](https://climatetrace.org/terms). | Responses recorded as returned (filtered by the request to CH4 and the demo region). |
| `fixtures/nasa_power/` | The data was obtained from the National Aeronautics and Space Administration (NASA) Langley Research Center (LaRC) Prediction of Worldwide Energy Resource (POWER) Project funded through the NASA Earth Science Directorate Applied Science Program. | Responses recorded as returned. |
| `fixtures/noaa_gml/` | Lan, X., K.W. Thoning, and E.J. Dlugokencky: Trends in globally-averaged CH4, N2O, and SF6 determined from NOAA Global Monitoring Laboratory measurements. https://doi.org/10.15138/P8XG-AA10 | File recorded as served. |
| `fixtures/epa_ghgrp/` | U.S. EPA Greenhouse Gas Reporting Program, via Envirofacts. | Responses recorded as returned. |
| `fixtures/natural_earth/` | Made with [Natural Earth](https://www.naturalearthdata.com/). Public domain. | Recorded as served. |
| `app/globe/textures/mars_2k.jpg` | [Solar System Scope](https://www.solarsystemscope.com/textures/), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), based on NASA elevation and imagery data. | None. |

## Fetched at run time (not bundled)

| Source | Licence and attribution |
|---|---|
| ERA5 via ARCO-ERA5 (Google Cloud public dataset) and the Copernicus CDS | CC BY 4.0. Contains modified Copernicus Climate Change Service information. |
| NCEP/NCAR Reanalysis 1 and NCEP-DOE Reanalysis 2 | Data provided by the NOAA PSL, Boulder, Colorado, USA, from their website at https://psl.noaa.gov. |
| OpenMARS MY28-35 standard database v3 | Holmes, J.A., Lewis, S.R., Patel, M.R. et al., The Open University. CC BY 4.0. |
| EDGAR v8.0 | European Commission JRC, CC BY 4.0, (c) European Union 1995-2026. |
| Carbon Mapper | Data by Carbon Mapper®. **Non-commercial, share-alike, revocable licence.** Its responses are never committed; they stay in the local, gitignored cache. |
| Open-Meteo (live API) | CC BY 4.0. **The free API is for non-commercial use only.** |
| Sentinel-5P TROPOMI via Google Earth Engine | Contains modified Copernicus Sentinel data. Earth Engine is free for non-commercial and research use. |
| NASA Blue Marble and topology images (loaded by the globe from the three-globe package on unpkg) | NASA imagery. |

Licence terms were read on 2026-11-25; see `docs/research/2026-11-25-ch4-wind-datasets.md` and
`topoflow/sources/registry.py`. Re-read them before any commercial use.
