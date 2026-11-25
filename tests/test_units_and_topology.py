import numpy as np
import pytest

from topoflow.sources.points import parse_gml_monthly
from topoflow.topology import critical_points, div_vort
from topoflow.units import (area_weighted_mean, cell_areas_m2, convert, mass_to_mole_fraction_ppb, normalise_unit,
                            wind_components)


# ------------------------------------------------------------------------------------------- units

def test_wind_from_north_blows_south():
    u, v = wind_components(10.0, 0.0)
    assert u == pytest.approx(0, abs=1e-9) and v == pytest.approx(-10)


def test_wind_from_west_blows_east():
    u, v = wind_components(5.0, 270.0)
    assert u == pytest.approx(5) and v == pytest.approx(0, abs=1e-9)


def test_mass_to_ppb_at_reference_conditions():
    # 1.31 mg/m3 of CH4 at 1013.25 hPa and 15 C is about 1.95 ppm (hand calculation: 1310e-6/16.0425 / 42.29 mol/m3).
    ppb = mass_to_mole_fraction_ppb(1310.0, 101_325.0, 288.15)
    assert ppb == pytest.approx(1931, rel=2e-3)


def test_lower_pressure_means_higher_mole_fraction_for_same_mass():
    assert mass_to_mole_fraction_ppb(1300, 90_000, 288.15) > mass_to_mole_fraction_ppb(1300, 101_325, 288.15)


def test_convert_and_aliases():
    assert convert(np.array([1000.0]), "hPa", "Pa")[0] == 100_000
    assert convert(np.array([0.0]), "degC", "K")[0] == pytest.approx(273.15)
    assert normalise_unit("μg/m³") == "ug m-3"
    assert normalise_unit("m s**-1") == "m s-1"
    with pytest.raises(ValueError):
        convert(np.array([1.0]), "ug m-3", "ppb")  # needs p and T: never a bare unit swap


def test_cell_areas_sum_to_earth_surface():
    lats = np.arange(-89.5, 90, 1.0)
    total = cell_areas_m2(lats, 1.0, 1.0).sum() * 360
    assert total == pytest.approx(4 * np.pi * 6_371_008.8**2, rel=1e-6)


def test_area_weighted_mean_ignores_nan():
    f = np.array([[1.0, np.nan], [3.0, 3.0]])
    assert area_weighted_mean(f, np.array([0.0, 60.0])) == pytest.approx((1 * 1 + 3 * 0.5 * 2) / (1 + 0.5 * 2))


def test_gml_parser_marks_preliminary_rows():
    text = "# header\n  2026       7      2026.542       1929.77         -9.99       1942.60         -9.99\n" \
           "  1983       7      1983.542       1625.96          2.25       1635.08          1.50\n"
    rows = parse_gml_monthly(text)
    assert rows[0] == {"year": 2026, "month": 7, "ppb": 1929.77, "uncertainty": None, "preliminary": True}
    assert rows[1]["uncertainty"] == 2.25 and not rows[1]["preliminary"]


# ------------------------------------------------------------------------------------------- topology

LATS = np.arange(0, 20.01, 1.0)
LONS = np.arange(10, 30.01, 1.0)
LA, LO = np.meshgrid(LATS, LONS, indexing="ij")
X = (LO - 20.7) * np.cos(np.deg2rad(LA))
Y = LA - 10.3

CASES = {
    "repelling_node": (X, Y),
    "attracting_node": (-X, -Y),
    "saddle": (X, -Y),
    "attracting_focus": (-0.2 * X - Y, X - 0.2 * Y),
    "repelling_focus": (0.2 * X - Y, X + 0.2 * Y),
    "center": (-Y, X),
}


@pytest.mark.parametrize("kind", CASES)
def test_classifies_and_locates_each_kind(kind):
    u, v = CASES[kind]
    pts = critical_points(u, v, LATS, LONS)
    assert len(pts) == 1
    p = pts[0]
    assert p["kind"] == kind
    assert p["lat"] == pytest.approx(10.3, abs=0.01)
    assert p["lon"] == pytest.approx(20.7, abs=0.01)
    assert p["index"] == (-1 if kind == "saddle" else 1)


def test_no_zero_means_no_points():
    assert critical_points(np.ones_like(X), np.ones_like(X), LATS, LONS) == []


def test_nan_cells_are_skipped_not_crashed():
    u, v = CASES["saddle"]
    u = u.copy()
    u[10, 10] = np.nan
    assert all(np.isfinite(p["lat"]) for p in critical_points(u, v, LATS, LONS))


def test_global_seam_is_closed():
    lats = np.arange(-10, 10.01, 2.5)
    lons = np.arange(-180, 177.6, 2.5)  # 175.0 + 2.5 wraps to -180
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    # Smooth and periodic: zero at 178.7E (rising, so a source with v = y) and at 1.3W (falling, so a saddle).
    x = np.sin(np.deg2rad(lo - 178.7))
    pts = sorted(critical_points(x, (la - 1.1) * 0.02, lats, lons), key=lambda p: p["lon"])
    assert [p["kind"] for p in pts] == ["saddle", "repelling_node"]
    assert pts[0]["lon"] == pytest.approx(-1.3, abs=0.05)
    assert pts[1]["lon"] == pytest.approx(178.7, abs=0.05)  # this one sits in the cell that crosses the seam


def test_divergence_sign_matches_source_and_sink():
    div_src, _ = div_vort(X, Y, LATS, LONS)
    div_sink, _ = div_vort(-X, -Y, LATS, LONS)
    assert np.nanmean(div_src) > 0 > np.nanmean(div_sink)


def test_vorticity_sign_of_counterclockwise_rotation():
    _, vort = div_vort(-Y, X, LATS, LONS)
    assert np.nanmean(vort) > 0
