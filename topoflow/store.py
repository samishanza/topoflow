"""A run on disk: fields.nc (every admitted grid, chosen and alternative), points.geojson, manifest.json.

Nothing is overwritten. A new run is a new folder; data/runs/LATEST names the newest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from . import grid
from .catalog import FIELDS
from .contract import RunManifest, RunRequest
from .gatekeeper import Accepted


def var_name(field: str, source: str | None = None) -> str:
    base = field.replace(".", "__")
    return f"{base}___{source}" if source else base


def write_run(run_dir: Path, req: RunRequest, manifest: RunManifest, selected: dict[str, Any],
              by_field: dict[str, list[Accepted]]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    lats, lons = grid.axes(req)
    times = grid.times(req)
    data_vars = {}
    features = []
    for key, cands in by_field.items():
        spec = FIELDS[key]
        chosen_src = manifest.fields[key].selected
        for a in cands:
            src = a.proposal.evidence.source_id if a.proposal.evidence else "derived"
            is_chosen = src == chosen_src and a.value is selected.get(key)
            attrs = {
                "long_name": spec.label, "units": spec.unit, "source": src,
                "data_origin": a.proposal.evidence.data_origin if a.proposal.evidence else "derived",
                "state": manifest.fields[key].state if is_chosen else "alternative",
                "rationale": a.proposal.rationale[:500],
            }
            if spec.shape == "grid":
                dims = ("lat", "lon") if spec.static else ("time", "lat", "lon")
                data_vars[var_name(key, None if is_chosen else src)] = (dims, np.asarray(a.value, "float32"), attrs)
            elif spec.shape == "points":
                for f in a.value:
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [f["lon"], f["lat"]]},
                        "properties": {"field": key, "source": src, "chosen": is_chosen, **f.get("properties", {})},
                    })
    ds = xr.Dataset(data_vars, coords={"time": times, "lat": lats, "lon": lons},
                    attrs={"title": "TopoFlow run", "run_id": manifest.run_id, "region": req.region,
                           "labels": " | ".join(manifest.labels)})
    enc = {k: {"zlib": True, "complevel": 4} for k in ds.data_vars}
    ds.to_netcdf(run_dir / "fields.nc", encoding=enc)
    (run_dir / "points.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features},
                                                       separators=(",", ":"), default=_json_default))
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    (run_dir.parent / "LATEST").write_text(run_dir.name + "\n")
    return run_dir


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def latest_run(root: str | Path = "data/runs") -> Path | None:
    root = Path(root)
    marker = root / "LATEST"
    if marker.exists():
        p = root / marker.read_text().strip()
        if p.exists():
            return p
    runs = sorted(p for p in root.glob("*") if (p / "manifest.json").exists()) if root.exists() else []
    return runs[-1] if runs else None


def load_run(run_dir: str | Path) -> tuple[RunManifest, xr.Dataset, dict]:
    run_dir = Path(run_dir)
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
    ds = xr.open_dataset(run_dir / "fields.nc")
    points = json.loads((run_dir / "points.geojson").read_text())
    return manifest, ds, points
