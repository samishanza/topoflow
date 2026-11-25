"""End-to-end runs with no network: the labelled synthetic fixture, and a replay of the recorded Permian run."""
import json
from pathlib import Path

import pytest

from topoflow import grid
from topoflow.agents.base import AgentContext, Emit
from topoflow.orchestrator import collect
from topoflow.sources import SourceClient
from topoflow.sources.synthetic import FOCUS, SADDLE, SOURCE_NODE, make_tools
from topoflow.store import load_run

REPO = Path(__file__).resolve().parents[1]


def run_synthetic(tmp_path, doctored=False):
    req = grid.make_request("permian", 0.25, "2025-07-01", "2025-07-01", [0, 12])
    run = collect(req, "replay", out_root=tmp_path / "runs", fixtures_dir=tmp_path / "none",
                  cache_dir=tmp_path / "cache", tools=make_tools(doctored=doctored), progress=lambda _: None)
    return load_run(run)


def test_synthetic_run_is_labelled_and_finds_the_planted_points(tmp_path):
    manifest, ds, points = run_synthetic(tmp_path)
    assert "CONTAINS SYNTHETIC DATA" in manifest.labels
    assert manifest.fields["ch4.surface_ppb"].state == "derived"
    cps = [f for f in points["features"] if f["properties"]["field"] == "topology.critical_points"
           and f["properties"]["t_index"] == 0]
    kinds = {f["properties"]["kind"]: f["geometry"]["coordinates"] for f in cps}

    def near(coords, target):
        return abs(coords[1] - target[0]) < 0.3 and abs(coords[0] - target[1]) < 0.3

    assert "attracting_focus" in kinds and near(kinds["attracting_focus"], FOCUS)
    assert any(f["properties"]["kind"] == "saddle" and near(f["geometry"]["coordinates"], SADDLE) for f in cps)
    assert any(f["properties"]["kind"] == "repelling_node" and near(f["geometry"]["coordinates"], SOURCE_NODE) for f in cps)
    # The CH4 bump was planted at the focus, so the focus reads above the regional mean.
    focus = next(f for f in cps if f["properties"]["kind"] == "attracting_focus")
    assert focus["properties"]["ch4_anomaly_ppb"] > 100
    ds.close()


def test_doctored_file_is_refused_flagged_and_changes_nothing(tmp_path):
    clean, ds1, _ = run_synthetic(tmp_path / "a")
    manifest, ds2, _ = run_synthetic(tmp_path / "b", doctored=True)
    refused = [r for a in manifest.agent_runs for r in a.rejected if r.source_id == "local_ch4"]
    assert refused and "cross_field_implausible" in refused[0].codes
    assert any("instruction-like text in local_ch4" in w for w in manifest.watchdog)
    assert "IMPORTANT SYSTEM NOTE" in manifest.untrusted_text["local_ch4"]
    # The decision did not move: same chosen source, same values, cross-checks untouched.
    assert manifest.fields["ch4.surface_ppb"].selected == clean.fields["ch4.surface_ppb"].selected == "derived"
    assert float(ds2["ch4__surface_ppb"].mean()) == pytest.approx(float(ds1["ch4__surface_ppb"].mean()))
    assert manifest.fields["ch4.global_mean_ppb"].state == "observed"
    ds1.close()
    ds2.close()


def test_agent_cannot_call_tools_off_its_list(tmp_path):
    req = grid.make_request("permian", 1.0, "2025-07-01", "2025-07-01", [0])
    from topoflow.http import Fetcher
    sc = SourceClient(Fetcher("replay", tmp_path, cache_dir=tmp_path), req, tools=make_tools())
    e = Emit("methane", AgentContext(request=req, sources=sc, allowed_tools=("open_meteo.cams",)))
    assert e.call("open_meteo.era5") is None
    assert "open_meteo_era5" not in sc.runs  # the source was never touched
    assert any("refused open_meteo.era5" in n for n in e.notes)


@pytest.mark.skipif(not (REPO / "fixtures" / "open_meteo_era5").exists(), reason="no recorded fixtures")
def test_replay_of_recorded_permian_run(tmp_path):
    """Real public data, replayed. Sources whose results live only in the local cache report themselves missing."""
    req = grid.make_request("permian")
    run = collect(req, "replay", out_root=tmp_path / "runs", fixtures_dir=REPO / "fixtures",
                  cache_dir=tmp_path / "empty-cache", progress=lambda _: None)
    m = json.loads((run / "manifest.json").read_text())
    status = {s["source_id"]: s["status"] for s in m["source_runs"]}
    for sid in ("open_meteo_era5", "open_meteo_cams", "noaa_gml", "nasa_power", "natural_earth", "climate_trace", "epa_ghgrp"):
        assert status[sid] == "ok", (sid, status[sid])
    for sid in ("carbon_mapper", "edgar", "cds_era5"):  # licence-restricted or bulk: cache only
        assert status[sid] in ("missing", "barred"), (sid, status[sid])
    assert m["fields"]["topology.critical_points"]["state"] == "derived"
    checks = {c["name"].split(" ")[0]: c["status"] for c in m["cross_checks"]}
    assert checks["ERA5"] == "pass"
    assert "CONTAINS SYNTHETIC DATA" not in m["labels"]
