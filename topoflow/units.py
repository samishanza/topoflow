"""Unit conversions. Deterministic, tested, and the only place these numbers are written down."""
from __future__ import annotations

import numpy as np

R_GAS = 8.314462618  # J mol-1 K-1
M_CH4 = 16.0425  # g mol-1
EARTH_RADIUS_M = 6_371_008.8
SECONDS_PER_YEAR = 365.25 * 86400
# AR5 100-year GWP used by the EPA GHGRP for reporting years 2014-2023 (40 CFR 98 Table A-1).
# Reporting year 2024 onward uses AR5 too, but check the year's Table A-1 before relying on it.
GWP100_CH4_GHGRP = 25.0

# Accepted spellings of one unit. A proposal's unit must normalise to the catalog unit or it is refused.
_ALIASES = {
    "m/s": "m s-1", "m s**-1": "m s-1", "m s-1": "m s-1", "ms-1": "m s-1",
    "μg/m³": "ug m-3", "µg/m³": "ug m-3", "ug/m3": "ug m-3", "ug m-3": "ug m-3",
    "ppb": "ppb", "nmol/mol": "ppb", "1e-9": "ppb", "mol mol-1": "mol mol-1",
    "pa": "Pa", "hpa": "hPa", "k": "K", "°c": "degC", "degc": "degC",
    "m": "m", "1": "1", "s-1": "s-1", "s**-1": "s-1",
    "kg m-2 s-1": "kg m-2 s-1", "kg m**-2 s**-1": "kg m-2 s-1",
    "t ch4 yr-1": "t CH4 yr-1", "t co2e yr-1": "t CO2e yr-1", "kg h-1": "kg h-1", "kg/hr": "kg h-1",
}

# (from, to) -> factor or function
_LINEAR = {
    ("hPa", "Pa"): 100.0,
    ("mol mol-1", "ppb"): 1e9,
}


def normalise_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    u = unit.strip()
    return _ALIASES.get(u.lower(), _ALIASES.get(u, u))


def convert(values, from_unit: str, to_unit: str):
    """Convert between catalog units. Raises ValueError when there is no exact conversion."""
    f, t = normalise_unit(from_unit), normalise_unit(to_unit)
    if f == t:
        return values
    if (f, t) in _LINEAR:
        return np.asarray(values) * _LINEAR[(f, t)]
    if (f, t) == ("degC", "K"):
        return np.asarray(values) + 273.15
    raise ValueError(f"no exact conversion from {from_unit!r} to {to_unit!r}")


def wind_components(speed, direction_deg):
    """Meteorological convention: direction is where the wind blows FROM, clockwise from north.

    A wind from the north (0 deg) blows toward the south, so v is negative.
    """
    speed = np.asarray(speed, dtype=float)
    rad = np.deg2rad(np.asarray(direction_deg, dtype=float))
    return -speed * np.sin(rad), -speed * np.cos(rad)


def mass_to_mole_fraction_ppb(conc_ug_m3, pressure_pa, temperature_k):
    """CH4 mass concentration to dry-ish mole fraction, using the air density at that cell and time.

    x = (c / M) / (p / (R T)). Water vapour is ignored, which biases x low by up to a few percent in the humid
    tropics. The record carries this as a stated limitation, not a silent one.
    """
    c = np.asarray(conc_ug_m3, dtype=float) * 1e-6  # g m-3
    n_ch4 = c / M_CH4  # mol m-3
    n_air = np.asarray(pressure_pa, dtype=float) / (R_GAS * np.asarray(temperature_k, dtype=float))
    return n_ch4 / n_air * 1e9


def cell_areas_m2(lats_deg: np.ndarray, dlat_deg: float, dlon_deg: float) -> np.ndarray:
    """Area of each latitude band's cell on a sphere, for cells centred on `lats_deg`."""
    lat = np.deg2rad(np.asarray(lats_deg, dtype=float))
    half = np.deg2rad(dlat_deg) / 2
    top = np.clip(lat + half, -np.pi / 2, np.pi / 2)
    bot = np.clip(lat - half, -np.pi / 2, np.pi / 2)
    return EARTH_RADIUS_M**2 * np.deg2rad(dlon_deg) * (np.sin(top) - np.sin(bot))


def area_weighted_mean(field2d: np.ndarray, lats_deg: np.ndarray) -> float:
    w = np.cos(np.deg2rad(lats_deg))[:, None] * np.ones_like(field2d)
    ok = np.isfinite(field2d)
    if not ok.any():
        return float("nan")
    return float((field2d[ok] * w[ok]).sum() / w[ok].sum())
