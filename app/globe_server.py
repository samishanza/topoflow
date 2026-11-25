"""TopoFlow globe: Earth and Mars over the processed stagnation climatologies.

    python -m app.globe_server            # http://127.0.0.1:8060

Reads only what `topoflow.climatology` (Earth, ERA5) and `topoflow.mars` (Mars, OpenMARS) wrote. It never fetches,
so everything on the globe passed the gate, is traced to its source file, and says how it was derived. Periods keep
arriving while the climatologies fill in; each planet's store reloads when a new one is processed.
"""
from __future__ import annotations

import argparse
import json
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
EARTH_SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
# Each data product is its own "planet" key; the three Earth reanalyses share Earth's seasons and radius.
PLANETS = {
    "earth": {"root": ROOT / "data" / "era5" / "months" / "1deg", "radius_km": 6371.0, "body": "earth",
              "label": "ERA5 (ECMWF), 1 deg", "seasons": EARTH_SEASONS},
    "earth_ncep_r1": {"root": ROOT / "data" / "ncep_r1" / "months" / "T62", "radius_km": 6371.0, "body": "earth",
                      "label": "NCEP/NCAR R1, T62", "seasons": EARTH_SEASONS},
    "earth_ncep_r2": {"root": ROOT / "data" / "ncep_r2" / "months" / "T62", "radius_km": 6371.0, "body": "earth",
                      "label": "NCEP-DOE R2, T62", "seasons": EARTH_SEASONS},
    "mars": {"root": ROOT / "data" / "mars" / "periods" / "5deg", "radius_km": 3389.5, "body": "mars",
             "label": "OpenMARS, 5 deg",
             # Solar longitude quarters: northern spring, summer, autumn, winter.
             "seasons": {"Ls0-90": (0, 90), "Ls90-180": (90, 180), "Ls180-270": (180, 270), "Ls270-360": (270, 360)}},
}


def _family(k: str) -> str:
    return "saddle" if k == "saddle" else ("node" if k.endswith("node") else "focus")


class Store:
    def __init__(self, planet: str):
        cfg = PLANETS[planet]
        self.planet, self.root = planet, cfg["root"]
        self.periods = sorted(p.name for p in self.root.glob("*") if (p / "summary.json").exists())
        if not self.periods:
            raise LookupError(f"no processed periods for {planet} under {self.root}")
        self.summaries = {m: json.loads((self.root / m / "summary.json").read_text()) for m in self.periods}
        if planet == "mars":  # chronological: Mars Year, then solar longitude (names sort "ls17" before "ls2")
            self.periods.sort(key=lambda m: (self.summaries[m]["mars_year"], self.summaries[m]["ls_range"][0]))
        cl = pd.concat([pd.read_parquet(self.root / m / "closure.parquet").assign(period=m) for m in self.periods],
                       ignore_index=True)
        self.closure = cl.sort_values("time").reset_index(drop=True)
        if "label" not in self.closure:
            self.closure["label"] = pd.DatetimeIndex(self.closure["time"]).strftime("%Y-%m-%d %H:%M UTC")
        self.keys = self.closure["time"].tolist()

    @lru_cache(maxsize=6)
    def points(self, period: str) -> pd.DataFrame:
        return pd.read_parquet(self.root / period / "points.parquet")

    @lru_cache(maxsize=6)
    def display(self, period: str) -> xr.Dataset:
        return xr.open_dataset(self.root / period / "display.nc").load()

    def in_season(self, period: str, season: str) -> bool:
        if season == "all":
            return True
        rng = PLANETS[self.planet]["seasons"][season]
        if PLANETS[self.planet]["body"] == "earth":
            return int(period[5:7]) in rng
        lo, hi = self.summaries[period]["ls_range"]
        mid = (lo + hi) / 2 if hi >= lo else ((lo + hi + 360) / 2) % 360
        return rng[0] <= mid < rng[1]

    @lru_cache(maxsize=32)
    def occupancy(self, season: str):
        total, steps, lat, lon = None, 0, None, None
        for m in self.periods:
            if not self.in_season(m, season):
                continue
            with xr.open_dataset(self.root / m / "occupancy.nc") as ds:
                arr = ds["occupancy"].sel(layer=list(LAYERS)).values
                steps += int(ds.attrs["steps"])
                lat, lon = ds["lat"].values, ds["lon"].values
            total = arr if total is None else total + arr
        return total, steps, lat, lon


_stores: dict[str, Store] = {}
_forecast: dict = {}


def forecast_for(key) -> list | None:
    """TopoNet's 24 h stagnation probability at each icosphere vertex, for held-out test steps only."""
    f = ROOT / "data" / "tdl" / "predictions.npz"
    if not f.exists():
        return None
    mtime = f.stat().st_mtime
    if _forecast.get("mtime") != mtime:
        d = np.load(f)  # never allow_pickle: a crafted .npz could run code
        _forecast.update(mtime=mtime, t=pd.DatetimeIndex(d["t"]), lat=d["lat"], lon=d["lon"], p=d["toponet"])
    try:
        k = _forecast["t"].get_loc(pd.Timestamp(key))
    except (KeyError, TypeError, ValueError):
        return None
    p = _forecast["p"][k].astype(float)
    keep = p >= 0.15
    return np.c_[_forecast["lat"][keep], _forecast["lon"][keep], p[keep]].round(3).tolist()


def store_for(planet: str) -> Store:
    if planet not in PLANETS:
        raise LookupError(f"unknown planet {planet}")
    s = _stores.get(planet)
    n = len(list(PLANETS[planet]["root"].glob("*/summary.json")))
    if s is None or n != len(s.periods):
        s = _stores[planet] = Store(planet)
    return s


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.errorhandler(LookupError)
    def missing(exc):
        return jsonify({"error": str(exc)}), 404

    @app.errorhandler(ValueError)
    def bad_request(exc):
        return jsonify({"error": "bad request parameter"}), 400

    @app.after_request
    def headers(resp):
        # The page loads three.js, Plotly and fonts from fixed CDNs and talks only to this server.
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        return resp

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def static_files(name):
        return send_from_directory(STATIC, name)

    @app.get("/api/planets")
    def planets():
        return jsonify({p: {"periods": len(list(cfg["root"].glob("*/summary.json"))), "seasons": list(cfg["seasons"]),
                            "body": cfg["body"], "label": cfg["label"]} for p, cfg in PLANETS.items()})

    @app.get("/api/meta")
    def meta():
        s = store_for(request.args.get("planet", "earth"))
        cl = s.closure
        by_kind: dict[str, int] = {}
        for m in s.periods:
            for k, v in s.points(m)["kind"].value_counts().items():
                by_kind[k] = by_kind.get(k, 0) + int(v)
        src = s.summaries[s.periods[-1]]["source"]
        entry = SOURCES.get(src["id"])
        starts = cl.groupby("period", sort=False).apply(lambda g: int(g.index.min()), include_groups=False)
        if s.planet == "mars":
            step_h = float(np.median(np.diff(cl["time"].values[:50]))) * 24.6597
        else:
            step_h = float(pd.Series(pd.DatetimeIndex(cl["time"])).diff().dt.total_seconds().median() / 3600)
        return jsonify({
            "planet": s.planet, "body": PLANETS[s.planet]["body"], "radius_km": PLANETS[s.planet]["radius_km"],
            "step_hours": round(step_h, 4), "period_start": {m: int(starts[m]) for m in s.periods},
            "periods": s.periods, "labels": cl["label"].tolist(),
            "resolution": src["resolution"], "points": int(sum(x["points"] for x in s.summaries.values())),
            "by_kind": by_kind, "seasons": list(PLANETS[s.planet]["seasons"]),
            "closure": {
                "steps": len(cl),
                "mean_winding_sum": round(float(cl.winding_sum.mean()), 4),
                "share_exactly_2": round(float((cl.winding_sum == 2).mean()), 4),
                "identity_holds": bool((cl.winding_sum == 2 + cl.edge_error).all()),
                "mean_points_per_step": round(float(cl.n_points.mean()), 1),
                "mean_edge_flips": round(float(cl.edge_flips.mean()), 2),
            },
            "source": {"name": entry.name if entry else src["name"], "license": src.get("license") or entry.license,
                       "attribution": src.get("attribution") or entry.attribution},
        })

    @app.get("/api/frame")
    def frame():
        s = store_for(request.args.get("planet", "earth"))
        i = int(np.clip(int(request.args.get("i", 0)), 0, len(s.keys) - 1))
        row = s.closure.iloc[i]
        key = row["time"]
        ds = s.display(row["period"])
        tdim = "valid_time" if "valid_time" in ds.dims else "time"
        tv = ds[tdim].values
        if np.issubdtype(tv.dtype, np.datetime64):
            k = int(np.abs(pd.DatetimeIndex(tv) - pd.Timestamp(key)).argmin())
        else:
            k = int(np.abs(tv - float(key)).argmin())
        pts = s.points(row["period"])
        now = pts[pts["time"] == key]
        return jsonify({
            "label": row["label"], "index": i, "period": row["period"],
            "grid": {"lat": ds["latitude"].values.round(3).tolist(), "lon": ds["longitude"].values.round(3).tolist()},
            "u": np.round(np.nan_to_num(ds["u10"].isel({tdim: k}).values), 2).ravel().tolist(),
            "v": np.round(np.nan_to_num(ds["v10"].isel({tdim: k}).values), 2).ravel().tolist(),
            "points": now[["lat", "lon", "kind", "strength", "divergence", "age_h", "track"]].round(
                {"lat": 3, "lon": 3, "strength": 9, "divergence": 9, "age_h": 1}).to_dict("records"),
            "closure": {c: int(row[c]) for c in ("winding_sum", "edge_error", "edge_flips", "cap_north",
                                                 "cap_south", "n_points")},
            "forecast": forecast_for(key) if s.planet == "earth" else None,
            "next": min(i + 1, len(s.keys) - 1),
        })

    @app.get("/api/agreement")
    def agreement():
        """Do independent reanalyses agree on where and how often stagnation happens? Same months only."""
        prods = [p for p, c in PLANETS.items() if c["body"] == "earth" and any(c["root"].glob("*/summary.json"))]
        stores = {p: store_for(p) for p in prods}
        common = sorted(set.intersection(*(set(s.periods) for s in stores.values()))) if stores else []
        out = {"products": {}, "common_months": common, "pattern_r": {}}
        if not common:
            return jsonify(out)
        grid_lat, grid_lon = np.arange(-88.75, 90, 2.5), np.arange(-178.75, 180, 2.5)
        fields = {}
        for p, s in stores.items():
            cl = s.closure[s.closure["period"].isin(common)]
            out["products"][p] = {"label": PLANETS[p]["label"], "steps": len(cl),
                                  "share_exactly_2": round(float((cl.winding_sum == 2).mean()), 4),
                                  "mean_points_per_step": round(float(cl.n_points.mean()), 1)}
            total, steps, lat, lon = None, 0, None, None
            for m in common:
                with xr.open_dataset(s.root / m / "occupancy.nc") as ds:
                    a = ds["occupancy"].sel(layer="all").values
                    lat, lon = ds["lat"].values, ds["lon"].values
                    steps += int(ds.attrs["steps"])
                total = a if total is None else total + a
            f = total / steps
            ii = np.abs(lat[:, None] - grid_lat[None, :]).argmin(0)
            jj = np.abs(lon[:, None] - grid_lon[None, :]).argmin(0)
            fields[p] = f[np.ix_(ii, jj)].ravel()
        names = list(fields)
        w = np.repeat(np.cos(np.deg2rad(grid_lat)), len(grid_lon))
        for a_i in range(len(names)):
            for b_i in range(a_i + 1, len(names)):
                x, y = fields[names[a_i]], fields[names[b_i]]
                xm, ym = (x * w).sum() / w.sum(), (y * w).sum() / w.sum()
                r = ((w * (x - xm) * (y - ym)).sum() / np.sqrt((w * (x - xm) ** 2).sum() * (w * (y - ym) ** 2).sum()))
                out["pattern_r"][f"{names[a_i]} vs {names[b_i]}"] = round(float(r), 3)
        out["note"] = ("Area-weighted correlation of where stagnation occurs (share of steps with any stagnation point "
                       "within one cell), on a common 2.5 deg grid, over months every product has.")
        return jsonify(out)

    @app.get("/api/model")
    def model():
        f = ROOT / "data" / "tdl" / "results.json"
        return jsonify(json.loads(f.read_text()) if f.exists() else {"error": "no model trained yet"})

    @app.get("/api/persistence")
    def persistence():
        s = store_for(request.args.get("planet", "earth"))
        kind, season = request.args.get("kind", "all"), request.args.get("season", "all")
        if season != "all" and season not in PLANETS[s.planet]["seasons"]:
            raise ValueError("unknown season")
        occ, steps, lat, lon = s.occupancy(season)
        if occ is None:
            return jsonify({"error": f"no periods in season {season}"}), 404
        if kind not in LAYERS:
            return jsonify({"error": f"unknown kind {kind}"}), 400
        freq = occ[LAYERS.index(kind)] / max(steps, 1)
        top, taken = [], []
        for flat in np.argsort(freq.ravel())[::-1]:
            a, b = np.unravel_index(flat, freq.shape)
            if any(abs(lat[a] - lat[x]) < 5 and min(abs(lon[b] - lon[y]), 360 - abs(lon[b] - lon[y])) < 5
                   for x, y in taken):
                continue
            taken.append((a, b))
            top.append((a, b))
            if len(top) == 12:
                break
        return jsonify({
            "kind": kind, "season": season, "steps": steps, "lat": lat.tolist(), "lon": lon.tolist(),
            "max": float(freq.max()), "grid": np.round(freq, 5).ravel().tolist(),
            "hotspots": [{"lat": float(lat[a]), "lon": float(lon[b]), "freq": round(float(freq[a, b]), 4)} for a, b in top],
            "note": "Share of steps with a stagnation point of this kind within one grid cell.",
        })

    @app.get("/api/closure")
    def closure():
        s = store_for(request.args.get("planet", "earth"))
        cl = s.closure
        return jsonify({"label": cl["label"].tolist(), "winding_sum": cl.winding_sum.tolist(),
                        "edge_error": cl.edge_error.tolist(), "n_points": cl.n_points.tolist()})

    @app.get("/api/cell")
    def cell():
        planet = request.args.get("planet", "earth")
        s = store_for(planet)
        R = PLANETS[planet]["radius_km"]
        lat, lon = float(request.args.get("lat", "nan")), float(request.args.get("lon", "nan"))
        radius_km = float(request.args.get("radius_km", 250 if planet == "earth" else 400))
        if not (-90 <= lat <= 90 and -180 <= lon <= 360 and 0 < radius_km <= 2000):
            raise ValueError("lat, lon or radius out of range")
        rows = []
        for m in s.periods:
            p = s.points(m)
            dlat, dlon = np.deg2rad(p["lat"] - lat), np.deg2rad(p["lon"] - lon)
            a = np.sin(dlat / 2) ** 2 + np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(p["lat"])) * np.sin(dlon / 2) ** 2
            near = p[2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1))) <= radius_km]
            steps = s.summaries[m]["steps"]
            fam = near["kind"].map(_family)
            rows.append({"period": m, "steps": steps,
                         **{f: round(float(near[fam == f]["time"].nunique()) / steps, 4) for f in ("saddle", "node", "focus")},
                         "attracting": round(float(near[near["kind"].str.startswith("attracting")]["time"].nunique()) / steps, 4),
                         "max_age_h": float(near["age_h"].max()) if len(near) else 0.0})
        df = pd.DataFrame(rows)
        return jsonify({"lat": lat, "lon": lon, "radius_km": radius_km, "series": df.to_dict("records"),
                        "mean": {c: round(float(df[c].mean()), 4) for c in ("saddle", "node", "focus", "attracting")},
                        "longest_track_h": float(df["max_age_h"].max()),
                        "note": f"Share of steps with a stagnation point of each family within {radius_km:.0f} km."})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8060)
    a = ap.parse_args()
    print(f"TopoFlow globe on http://127.0.0.1:{a.port}")
    create_app().run(port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
