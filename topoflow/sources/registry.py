"""Every dataset TopoFlow can read: how to reach it, its licence, and what a person must do when it is closed.

Terms were read on 2026-11-25 (docs/research/2026-11-25-dataset-terms.md). Re-read them before relying on a source
for anything beyond research, and before making the repo public: two sources here are non-commercial only.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..catalog import DataKind, Tier


@dataclass(frozen=True)
class SourceEntry:
    id: str
    name: str
    kind: DataKind
    tier: Tier
    url: str
    license: str
    attribution: str
    terms_url: str
    terms_checked: str  # date, or "unverified: <why>"
    # Environment variable or file that opens a keyed source. Absent means the source is skipped with person_route.
    needs: str = ""
    person_route: str = ""
    # True when the licence bars redistribution: recordings go to the gitignored cache, not fixtures/.
    no_redistribution: bool = False
    note: str = ""


SOURCES: dict[str, SourceEntry] = {
    s.id: s
    for s in [
        SourceEntry(
            "open_meteo_era5", "ERA5 reanalysis via Open-Meteo Historical Weather API", "reanalysis", 0,
            "https://archive-api.open-meteo.com/v1/archive",
            "CC BY 4.0; free API is non-commercial only (600/min, 5,000/h, 10,000/day)",
            "Weather data by Open-Meteo.com (https://open-meteo.com/), ERA5 by ECMWF/Copernicus",
            "https://open-meteo.com/en/terms", "2026-11-25",
            note="Serves 10 m speed and direction, not u and v; TopoFlow converts. ERA5 lags real time by about 5 days.",
        ),
        SourceEntry(
            "open_meteo_cams", "CAMS global methane via Open-Meteo Air Quality API", "model_analysis", 0,
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            "CC BY 4.0; free API is non-commercial only",
            "Air quality data by Open-Meteo.com, CAMS global by ECMWF/Copernicus Atmosphere Monitoring Service",
            "https://open-meteo.com/en/terms", "2026-11-25",
            note="Methane is a mass concentration (ug m-3) at the lowest model level. History starts in early 2025 (tested).",
        ),
        SourceEntry(
            "nasa_power", "NASA POWER daily point data (MERRA-2)", "reanalysis", 0,
            "https://power.larc.nasa.gov/api/temporal/daily/point",
            "No licence named on the pages read",
            "Data obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science Directorate Applied Science Program.",
            "https://power.larc.nasa.gov/docs/referencing/", "unverified: licence not stated; attribution text verified",
            note="A different reanalysis from ERA5, so agreement is evidence and disagreement is a flag. Used for cross-checks only.",
        ),
        SourceEntry(
            "noaa_gml", "NOAA GML global monthly mean CH4", "in_situ", 0,
            "https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.txt",
            "Freely available; cite in publications", "Lan, X., K.W. Thoning, and E.J. Dlugokencky: Trends in globally-averaged CH4, N2O, and SF6 determined from NOAA Global Monitoring Laboratory measurements. https://doi.org/10.15138/P8XG-AA10",
            "https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.txt", "2026-11-25",
            note="Marine boundary layer background. The latest months are preliminary (uncertainty -9.99).",
        ),
        SourceEntry(
            "climate_trace", "Climate TRACE asset emissions (API v6, beta)", "inventory", 0,
            "https://api.climatetrace.org/v6/assets",
            "CC BY 4.0 (some embedded third-party datasets carry their own terms)",
            "Climate TRACE (https://climatetrace.org), changes: filtered to CH4 and the run's region",
            "https://climatetrace.org/terms", "2026-11-25",
            note="Beta API that asks users to keep volume low; TopoFlow caps pages. No area filter, so it filters by country, then by bbox here.",
        ),
        SourceEntry(
            "epa_ghgrp", "EPA GHGRP facilities via Envirofacts", "facility_report", 0,
            "https://data.epa.gov/efservice",
            "US federal data, normally public domain; not stated for this API",
            "U.S. EPA Greenhouse Gas Reporting Program, via Envirofacts",
            "https://www.epa.gov/enviro/envirofacts-data-service-api", "unverified: no dataset licence on the API page",
            note="US only. Emissions are CO2e at the reporting year's GWP; 2023 is the latest year with sector rows (tested).",
        ),
        SourceEntry(
            "carbon_mapper", "Carbon Mapper public CH4 plume catalogue", "plume_detection", 0,
            "https://api.carbonmapper.org/api/v1/catalog/plumes/annotated",
            "Carbon Mapper data licence: revocable, NON-COMMERCIAL, share-alike",
            "Data by Carbon Mapper® (data.carbonmapper.org)",
            "https://carbonmapper.org/terms", "2026-11-25 (effective 2026-01-13)",
            no_redistribution=True,
            note="Recordings stay in the gitignored cache. Each plume carries its own wind estimate, which TopoFlow checks against ERA5.",
        ),
        SourceEntry(
            "edgar", "EDGAR v8.0 CH4 total emissions, 0.1 degree annual grid", "inventory", 2,
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/v80_FT2022_GHG/CH4/TOTALS/emi_nc/",
            "CC BY 4.0, (c) European Union 1995-2026",
            "EDGAR (Emissions Database for Global Atmospheric Research) Community GHG Database v8.0, European Commission JRC; regridded by TopoFlow",
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/v80_FT2022_GHG/CH4/TOTALS/emi_nc/copyright.txt",
            "2026-11-25",
            note="About 15 MB per year, zipped. Downloaded once to data/cache, regridded by summing mass, then the raw file is deleted.",
        ),
        SourceEntry(
            "natural_earth", "Natural Earth 1:110m land polygons", "basemap", 2,
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson",
            "Public domain", "Made with Natural Earth.",
            "https://www.naturalearthdata.com/about/terms-of-use/", "2026-11-25",
        ),
        SourceEntry(
            "cds_era5", "ERA5 single levels via Copernicus Climate Data Store", "reanalysis", 1,
            "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
            "CC BY 4.0", "Contains modified Copernicus Climate Change Service information (ERA5)",
            "https://cds.climate.copernicus.eu/how-to-api", "2026-11-25",
            needs="~/.cdsapirc or CDSAPI_URL + CDSAPI_KEY",
            person_route="Create a free CDS account, copy your token into ~/.cdsapirc (see the how-to-api page), and accept the ERA5 licence once on the dataset page.",
            note="Native u10/v10 at 0.25 degree, no conversion needed. Requests queue; a week of a region takes minutes.",
        ),
        SourceEntry(
            "arco_era5", "ERA5 via ARCO-ERA5 (Google Cloud public dataset)", "reanalysis", 0,
            "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/",
            "CC BY 4.0 (ERA5, Copernicus licence); hosted as a Google Cloud public dataset",
            "Contains modified Copernicus Climate Change Service information (ERA5), via ARCO-ERA5 (Google Research)",
            "https://cds.climate.copernicus.eu/how-to-api",
            "ERA5 licence verified 2026-11-25; the ARCO host's own terms page not read",
            note="One global hour per chunk at 0.25 deg, read anonymously over HTTPS. Used for the 2010-2024 climatology "
                 "because the CDS queue held global monthly requests for over an hour (tested 2026-11-25).",
        ),
        SourceEntry(
            "ncep_r1", "NCEP/NCAR Reanalysis 1, 10 m wind (T62 Gaussian, 6-hourly)", "reanalysis", 0,
            "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/",
            "Freely available from NOAA PSL; acknowledgement requested",
            "NCEP/NCAR Reanalysis data provided by the NOAA PSL, Boulder, Colorado, USA, from their website at https://psl.noaa.gov",
            "https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html", "2026-11-25",
            note="An independent reanalysis (different model, assimilation and grid, about 1.9 deg) for checking which "
                 "stagnation points are robust across products. Yearly files, about 50 MB per variable.",
        ),
        SourceEntry(
            "ncep_r2", "NCEP-DOE Reanalysis 2, 10 m wind (T62 Gaussian, 6-hourly)", "reanalysis", 0,
            "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis2/gaussian_grid/",
            "Freely available from NOAA PSL; acknowledgement requested",
            "NCEP-DOE Reanalysis 2 data provided by the NOAA PSL, Boulder, Colorado, USA, from their website at https://psl.noaa.gov",
            "https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.html", "2026-11-25",
            note="R1 with known errors fixed. A third product for agreement checks.",
        ),
        SourceEntry(
            "gee_tropomi", "Sentinel-5P TROPOMI XCH4 L3 via Google Earth Engine", "satellite", 1,
            "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_CH4",
            "Copernicus Sentinel data terms (open); Earth Engine is free for non-commercial use only",
            "Contains modified Copernicus Sentinel data [year], processed by ESA; accessed via Google Earth Engine",
            "https://developers.google.com/earth-engine/guides/access", "2026-11-25",
            needs="EARTHENGINE_PROJECT and Earth Engine credentials (earthengine authenticate)",
            person_route="Register a Google Cloud project for Earth Engine (noncommercial), run `earthengine authenticate`, and set EARTHENGINE_PROJECT in .env.",
        ),
        SourceEntry(
            "local_era5", "Your own ERA5 u10/v10 NetCDF files", "local_file", 3,
            "file://$TOPOFLOW_U_FILE, $TOPOFLOW_V_FILE", "Whatever licence the files came with (ERA5: CC BY 4.0)",
            "Contains modified Copernicus Climate Change Service information (ERA5)", "", "n/a",
            needs="TOPOFLOW_U_FILE and TOPOFLOW_V_FILE",
            person_route="Download ERA5 10m u/v from the CDS as NetCDF and point TOPOFLOW_U_FILE / TOPOFLOW_V_FILE at them.",
        ),
        SourceEntry(
            "local_ch4", "Your own daily CH4 NetCDF files", "local_file", 3,
            "file://$TOPOFLOW_METHANE_DIR/YYYYMMDD.nc", "Whatever licence the files came with", "", "", "n/a",
            needs="TOPOFLOW_METHANE_DIR (and TOPOFLOW_METHANE_VAR, TOPOFLOW_METHANE_UNIT if the file does not say)",
            person_route="Put one NetCDF per day in a folder, named YYYYMMDD.nc, and set TOPOFLOW_METHANE_DIR.",
            note="The variable's units attribute must name its unit, or TOPOFLOW_METHANE_UNIT must; otherwise the gate refuses it.",
        ),
    ]
}
