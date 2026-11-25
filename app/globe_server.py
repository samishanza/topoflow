"""TopoFlow globe: a 3-D Earth over the processed stagnation climatology.

    python -m app.globe_server            # http://127.0.0.1:8060

Reads only what `topoflow.climatology` wrote (data/era5/months/<res>/...). It never fetches data, so everything on the
globe is gated, traced to its ERA5 month, and labelled with how it was derived.
"""
from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from flask import Flask, jsonify, request, send_from_directory

from topoflow.climatology import LAYERS
from topoflow.sources.registry import SOURCES

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "globe"
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.months = sorted(p.name for p in root.glob("*") if (p / "summary.json").exists())
        if not self.months:
            raise SystemExit(f"No processed months under {root}. Run the ERA5 fetch and `python -m topoflow.climatology process`.")
        self.summaries = {m: json.loads((root / m / "summary.json").read_text()) for m in self.months}
        cl = pd.concat([pd.read_parquet(root / m / "closure.parquet").assign(month=m) for m in self.months],
                       ignore_index=True)
        self.closure = cl.sort_values("time").reset_index(drop=True)
        self.times = pd.DatetimeIndex(self.closure["time"])

    @lru_cache(maxsize=6)
    def points(self, month: str) -> pd.DataFrame:
        return pd.read_parquet(self.root / month / "points.parquet")

    @lru_cache(maxsize=6)
    def display(self, month: str) -> xr.Dataset:
        return xr.open_dataset(self.root / month / "display.nc").load()

    @lru_cache(maxsize=32)
    def occupancy(self, season: str) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        total, steps, lat, lon = None, 0, None, None
        for m in self.months:
            if season != "all" and int(m[5:7]) not in SEASONS[season]:
                continue
            with xr.open_dataset(self.root / m / "occupancy.nc") as ds:
                arr = ds["occupancy"].sel(layer=list(LAYERS)).values
                steps += int(ds.attrs["steps"])
                lat, lon = ds["lat"].values, ds["lon"].values
            total = arr if total is None else total + arr
        return total, steps, lat, lon


def create_app(root: Path) -> Flask:
    holder = {"store": Store(root)}
    app = Flask(__name__, static_folder=None)

    class _Live:
        """Months keep arriving while the climatology fills in; reload when a new one is processed."""

        def __getattr__(self, name):
            s = holder["store"]
            if len(list(root.glob("*/summary.json"))) != len(s.months):
                holder["store"] = s = Store(root)
            return getattr(s, name)

    store = _Live()

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def static_files(name):
        return send_from_directory(STATIC, name)

    @app.get("/api/meta")
    def meta():
        cl = store.closure
        pts = sum(s["points"] for s in store.summaries.values())
        by_kind: dict[str, int] = {}
        for m in store.months:
            for k, v in store.points(m)["kind"].value_counts().items():
                by_kind[k] = by_kind.get(k, 0) + int(v)
        entry = SOURCES[store.summaries[store.months[-1]]["source"]["id"]]
        return jsonify({
            "months": store.months,
            "times": [t.isoformat() + "Z" for t in store.times],
            "resolution": store.summaries[store.months[0]]["source"]["resolution"],
            "points": pts,
            "by_kind": by_kind,
            "closure": {
                "steps": len(cl),
                "mean_winding_sum": round(float(cl.winding_sum.mean()), 4),
                "share_exactly_2": round(float((cl.winding_sum == 2).mean()), 4),
                "identity_holds": bool((cl.winding_sum == 2 + cl.edge_error).all()),
                "mean_points_per_step": round(float(cl.n_points.mean()), 1),
                "mean_edge_flips": round(float(cl.edge_flips.mean()), 2),
            },
            "source": {"name": entry.name, "license": entry.license, "attribution": entry.attribution,
                       "terms": entry.terms_url},
        })

    @app.get("/api/frame")
    def frame():
        i = int(np.clip(int(request.args.get("i", 0)), 0, len(store.times) - 1))
        t = store.times[i]
        row = store.closure.iloc[i]
        month = row["month"]
        ds = store.display(month)
        tdim = "valid_time" if "valid_time" in ds.dims else "time"
        k = int(np.abs(pd.DatetimeIndex(ds[tdim].values) - t).argmin())
        u = ds["u10"].isel({tdim: k}).values
        v = ds["v10"].isel({tdim: k}).values
        blh = ds["blh"].isel({tdim: k}).values
        pts = store.points(month)
        now = pts[pts["time"] == t]
        return jsonify({
            "time": t.isoformat() + "Z", "index": i, "month": month,
            "grid": {"lat": ds["latitude"].values.round(3).tolist(), "lon": ds["longitude"].values.round(3).tolist()},
            "u": np.round(np.nan_to_num(u), 2).ravel().tolist(),
            "v": np.round(np.nan_to_num(v), 2).ravel().tolist(),
            "blh": np.round(np.nan_to_num(blh), 0).ravel().tolist(),
            "points": now[["lat", "lon", "kind", "strength", "divergence", "age_h", "track"]].round(
                {"lat": 3, "lon": 3, "strength": 9, "divergence": 9, "age_h": 1}).to_dict("records"),
            "closure": {k: int(row[k]) for k in ("winding_sum", "edge_error", "edge_flips", "cap_north",
                                                 "cap_south", "n_points")},
        })

    @app.get("/api/persistence")
    def persistence():
        kind = request.args.get("kind", "all")
        season = request.args.get("season", "all")
        occ, steps, lat, lon = store.occupancy(season)
        if occ is None:
            return jsonify({"error": f"no months in season {season}"}), 404
        if kind not in LAYERS:
            return jsonify({"error": f"unknown kind {kind}"}), 400
        # Each step counted once per cell (processing marks a cell when a point is within one cell of it).
        smooth = occ[LAYERS.index(kind)] / max(steps, 1)
        # Hotspots: strongest cells, at least 5 degrees apart so one corridor does not fill the list.
        top, taken = [], []
        for flat in np.argsort(smooth.ravel())[::-1]:
            a, b = np.unravel_index(flat, smooth.shape)
            if any(abs(lat[a] - lat[x]) < 5 and min(abs(lon[b] - lon[y]), 360 - abs(lon[b] - lon[y])) < 5 for x, y in taken):
                continue
            taken.append((a, b))
            top.append((a, b))
            if len(top) == 12:
                break
        return jsonify({
            "kind": kind, "season": season, "steps": steps, "lat": lat.tolist(), "lon": lon.tolist(),
            "max": float(smooth.max()),
            "grid": np.round(smooth, 5).ravel().tolist(),
            "hotspots": [{"lat": float(lat[a]), "lon": float(lon[b]), "freq": round(float(smooth[a, b]), 4)} for a, b in top],
            "note": "Share of 6-hourly steps with a stagnation point of this kind within one grid cell.",
        })

    @app.get("/api/closure")
    def closure():
        cl = store.closure
        return jsonify({"time": [t.isoformat() + "Z" for t in store.times], "winding_sum": cl.winding_sum.tolist(),
                        "edge_error": cl.edge_error.tolist(), "n_points": cl.n_points.tolist()})

    @app.get("/api/cell")
    def cell():
        lat, lon = float(request.args["lat"]), float(request.args["lon"])
        radius_km = float(request.args.get("radius_km", 250))
        rows = []
        for m in store.months:
            p = store.points(m)
            dlat = np.deg2rad(p["lat"] - lat)
            dlon = np.deg2rad(p["lon"] - lon)
            a = np.sin(dlat / 2) ** 2 + np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(p["lat"])) * np.sin(dlon / 2) ** 2
            near = p[2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) <= radius_km]
            steps = store.summaries[m]["steps"]
            fam = near["kind"].map(lambda k: "saddle" if k == "saddle" else ("node" if k.endswith("node") else "focus"))
            rows.append({"month": m, "steps": steps,
                         **{f: round(float(near[fam == f]["time"].nunique()) / steps, 4) for f in ("saddle", "node", "focus")},
                         "attracting": round(float(near[near["kind"].str.startswith("attracting")]["time"].nunique()) / steps, 4),
                         "max_age_h": float(near["age_h"].max()) if len(near) else 0.0})
        df = pd.DataFrame(rows)
        return jsonify({"lat": lat, "lon": lon, "radius_km": radius_km, "monthly": df.to_dict("records"),
                        "mean": {c: round(float(df[c].mean()), 4) for c in ("saddle", "node", "focus", "attracting")},
                        "longest_track_h": float(df["max_age_h"].max()),
                        "note": "Share of 6-hourly steps with a stagnation point of each family within the radius."})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("TOPOFLOW_CLIM_DIR", str(ROOT / "data" / "era5" / "months" / "1deg")))
    ap.add_argument("--port", type=int, default=8060)
    a = ap.parse_args()
    app = create_app(Path(a.data))
    print(f"TopoFlow globe on http://127.0.0.1:{a.port}  (data: {a.data})")
    app.run(port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
