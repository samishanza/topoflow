import numpy as np
import pytest

from topoflow import grid
from topoflow.contract import Evidence, Proposal
from topoflow.gatekeeper import gate

REQ = grid.make_request("permian", 1.0, "2025-07-01", "2025-07-01", [0, 12])
LATS, LONS = grid.axes(REQ)
NT = len(grid.times(REQ))
SHAPE = (NT, len(LATS), len(LONS))


def ev(source="open_meteo_era5", origin="real"):
    return Evidence(source_id=source, kind="reanalysis", tier=0, refs=["https://example.test/x"], label="test",
                    retrieved_at="2026-11-25T00:00:00+00:00", data_origin=origin)


def prop(field, value, unit, agent="wind", evidence="default", **kw):
    return Proposal(agent_id=agent, field=field, shape=kw.pop("shape", "grid"), unit=unit,
                    evidence=ev() if evidence == "default" else evidence, confidence=kw.pop("confidence", 0.8),
                    rationale=kw.pop("rationale", "a sufficiently long reason"), value=value, **kw)


def codes(result):
    return {c for r in result.rejected for c in r.codes}


def test_good_wind_is_admitted_and_units_normalised():
    g = gate([prop("wind.u10", np.full(SHAPE, 3.0, "float32"), "m/s")], REQ, {})
    assert len(g.accepted) == 1 and g.accepted[0].unit == "m s-1"


def test_pressure_in_hpa_is_converted_to_pa():
    g = gate([prop("met.sp", np.full(SHAPE, 950.0, "float32"), "hPa")], REQ, {})
    assert g.accepted[0].converted and float(g.accepted[0].value.mean()) == pytest.approx(95_000)


def test_unknown_field():
    assert codes(gate([prop("wind.gust", np.zeros(SHAPE), "m s-1")], REQ, {})) == {"unknown_field"}


def test_agent_outside_its_scope():
    assert "agent_not_permitted" in codes(gate([prop("wind.u10", np.zeros(SHAPE), "m s-1", agent="methane")], REQ, {}))


def test_no_evidence_no_admission():
    assert "missing_evidence" in codes(gate([prop("wind.u10", np.zeros(SHAPE), "m s-1", evidence=None)], REQ, {}))


def test_missing_unit_is_refused():
    assert "unit_error" in codes(gate([prop("wind.u10", np.zeros(SHAPE), "")], REQ, {}))


def test_mass_concentration_cannot_pose_as_ppb():
    g = gate([prop("ch4.surface_ppb", np.full(SHAPE, 1300.0), "ug m-3", agent="methane")], REQ, {})
    assert "unit_error" in codes(g)


def test_wrong_shape():
    assert "shape_mismatch" in codes(gate([prop("wind.u10", np.zeros((2, 2)), "m s-1")], REQ, {}))


def test_impossible_wind_is_out_of_bounds():
    bad = np.full(SHAPE, 3.0, "float32")
    bad[0, :, :] = 300.0
    assert "out_of_bounds" in codes(gate([prop("wind.u10", bad, "m s-1")], REQ, {}))


def test_one_corrupt_cell_in_a_big_field_is_tolerated_but_counted():
    req = grid.make_request("global", 5.0, "2025-07-01", "2025-07-01", [0, 6, 12, 18])
    shape = (4, len(grid.axes(req)[0]), len(grid.axes(req)[1]))
    v = np.full(shape, 3.0, "float32")
    v[0, 0, 0] = 300.0  # 1 cell in ~10,000 is under the 0.1% tolerance
    assert len(gate([prop("wind.u10", v, "m s-1")], req, {}).accepted) == 1


def test_patchy_field_refused():
    v = np.full(SHAPE, 3.0, "float32")
    v[:, : len(LATS) // 2] = np.nan
    assert "too_patchy" in codes(gate([prop("wind.u10", v, "m s-1")], REQ, {}))


def test_regionally_impossible_ch4_refused_on_cross_field_physics():
    background = prop("ch4.global_mean_ppb", {"ppb": 1930.0}, "ppb", agent="methane", shape="scalar",
                      evidence=ev("noaa_gml"))
    doctored = prop("ch4.surface_ppb", np.full(SHAPE, 4800.0, "float32"), "ppb", agent="methane",
                    evidence=ev("local_ch4"))
    # The doctored field comes first; the gate still judges it against the background.
    g = gate([doctored, background], REQ, {})
    assert [a.proposal.field for a in g.accepted] == ["ch4.global_mean_ppb"]
    assert codes(g) == {"cross_field_implausible"}


def test_plausible_ch4_admitted_against_background():
    background = prop("ch4.global_mean_ppb", {"ppb": 1930.0}, "ppb", agent="methane", shape="scalar", evidence=ev("noaa_gml"))
    ok = prop("ch4.surface_ppb", np.full(SHAPE, 2100.0, "float32"), "ppb", agent="methane", evidence=ev("local_ch4"))
    assert len(gate([ok, background], REQ, {}).accepted) == 2


def test_derived_value_needs_admitted_inputs():
    p = prop("flow.divergence", np.zeros(SHAPE, "float32"), "s-1", agent="derive", evidence=None, derived=True,
             derived_from=["wind.u10", "wind.v10"])
    assert "derived_requires_lineage" in codes(gate([p], REQ, {}))
    assert len(gate([p], REQ, {"wind.u10": 1, "wind.v10": 1}).accepted) == 1


def test_points_outside_the_region_refused():
    feats = [{"lon": 10.0, "lat": 50.0, "properties": {"t_ch4_per_year": 5.0}}] * 3
    p = prop("emitters.assets", feats, "t CH4 yr-1", agent="emissions", shape="points", evidence=ev("climate_trace"))
    assert "outside_region" in codes(gate([p], REQ, {}))


def test_confidence_and_rationale_required():
    p = prop("wind.u10", np.zeros(SHAPE, "float32"), "m s-1", confidence=1.5, rationale="ok")
    assert {"bad_confidence", "missing_rationale"} <= codes(gate([p], REQ, {}))
