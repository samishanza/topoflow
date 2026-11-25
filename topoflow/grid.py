"""The grid and time window every source is asked for, so all fields line up cell for cell."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contract import RunRequest

# west, south, east, north. Regions are picked for what they show, not for coverage.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    # The whole planet at a resolution the free APIs can serve in one sitting.
    "global": (-180.0, -85.0, 175.0, 85.0),
    # Densest oil and gas methane plumes in the US; Carbon Mapper and GHGRP both cover it.
    "permian": (-106.0, 29.0, -100.0, 35.0),
    # Gulf of Mexico coast: land-sea breeze convergence, refineries, wetlands.
    "gulf_coast": (-98.0, 26.0, -88.0, 32.0),
    # Turkmenistan's South Caspian basin: some of the largest observed ultra-emitters.
    "turkmenistan": (52.0, 36.0, 64.0, 42.0),
    # Sudd wetlands, the largest tropical wetland CH4 source in Africa.
    "sudd": (28.0, 5.0, 34.0, 10.0),
}

# Default resolution per region. The global grid is the coarsest the free tier can fill in about ten minutes.
DEFAULT_RESOLUTION = {"global": 5.0, "permian": 0.5, "gulf_coast": 0.5, "turkmenistan": 0.5, "sudd": 0.5}

# CAMS methane on Open-Meteo starts in early 2025, so the default window sits after that.
DEFAULT_START = "2025-07-01"
DEFAULT_END = "2025-07-07"
DEFAULT_HOURS = [0, 6, 12, 18]


def make_request(region: str = "permian", resolution: float | None = None, start: str = DEFAULT_START,
                 end: str = DEFAULT_END, hours: list[int] | None = None,
                 bbox: tuple[float, float, float, float] | None = None) -> RunRequest:
    if bbox is None:
        if region not in REGIONS:
            raise ValueError(f"unknown region {region!r}; known: {', '.join(REGIONS)} (or pass --bbox)")
        bbox = REGIONS[region]
    w, s, e, n = bbox
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError(f"bad bbox {bbox}: expected west<east, south<north, in degrees")
    if pd.Timestamp(end) < pd.Timestamp(start):
        raise ValueError("end is before start")
    res = resolution or DEFAULT_RESOLUTION.get(region, 0.5)
    return RunRequest(region=region, bbox=(w, s, e, n), resolution_deg=res, start=start, end=end,
                      hours_utc=sorted(hours if hours is not None else DEFAULT_HOURS))


def axes(req: RunRequest) -> tuple[np.ndarray, np.ndarray]:
    w, s, e, n = req.bbox
    r = req.resolution_deg
    lats = np.round(np.arange(s, n + r / 2, r), 4)
    lons = np.round(np.arange(w, e + r / 2, r), 4)
    return lats, lons


def times(req: RunRequest) -> np.ndarray:
    days = pd.date_range(req.start, req.end, freq="D")
    stamps = [d + pd.Timedelta(hours=h) for d in days for h in req.hours_utc]
    return np.array(stamps, dtype="datetime64[ns]")


def points(req: RunRequest) -> list[tuple[float, float]]:
    """Every (lat, lon) cell centre, row-major from the south-west corner."""
    lats, lons = axes(req)
    return [(float(la), float(lo)) for la in lats for lo in lons]


def sample_points(req: RunRequest, n: int = 12, seed: int = 7) -> list[tuple[float, float]]:
    """A fixed, spread-out subset of cells for cross-checks. Seeded so a replay asks for the same points."""
    pts = points(req)
    if len(pts) <= n:
        return pts
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(pts), size=n, replace=False))
    return [pts[i] for i in idx]
