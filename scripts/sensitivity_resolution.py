"""Resolution dependence of Poincaré-Hopf closure on real ERA5 winds.

For sample times, read ERA5 10 m wind at 0.25 deg from ARCO-ERA5 (two chunks per time, nothing saved) and analyse it
at 0.25, 0.5, 1 and 2 degrees by subsampling. Reports, per resolution: how often the index sum is exactly 2, the mean
absolute departure, the flipped edges that explain it, and the stagnation points found per step.

    python scripts/sensitivity_resolution.py --times 48 --year 2019
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fetch_era5_arco import chunk  # noqa: E402
from topoflow.sphere import analyse  # noqa: E402

RES = (0.25, 0.5, 1.0, 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--times", type=int, default=48)
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--out", default="docs/research/resolution_closure.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.year)
    hours = pd.date_range(f"{a.year}-01-01", f"{a.year}-12-31 18:00", freq="6h")
    sample = sorted(rng.choice(len(hours), a.times, replace=False))
    lat = np.arange(90, -90.0001, -0.25)
    lon = np.arange(0, 360, 0.25)
    rows = []
    for n, k in enumerate(sample):
        t = hours[k]
        u = chunk("10m_u_component_of_wind", t, 1)
        v = chunk("10m_v_component_of_wind", t, 1)
        for res in RES:
            s = int(round(res / 0.25))
            r = analyse(u[::s, ::s], v[::s, ::s], lat[::s], lon[::s])
            rows.append({"time": str(t), "res": res, "sum": r["winding_sum"], "edge_error": r["edge_error"],
                         "flips": r["edge_flips"], "points": r["n_points"], "cap_n": r["cap_north"], "cap_s": r["cap_south"]})
        print(f"{n + 1}/{len(sample)} {t}: " + ", ".join(f"{x['res']}°→{x['sum']}" for x in rows[-len(RES):]), flush=True)
    df = pd.DataFrame(rows)
    summary = df.groupby("res").agg(steps=("sum", "size"), exactly_2=("sum", lambda s: float((s == 2).mean())),
                                    mean_sum=("sum", "mean"), mean_abs_departure=("sum", lambda s: float((s - 2).abs().mean())),
                                    mean_flips=("flips", "mean"), mean_points=("points", "mean"),
                                    identity=("sum", "size")).reset_index()
    summary["identity"] = df.groupby("res").apply(lambda g: bool((g["sum"] == 2 + g["edge_error"]).all())).values
    print(summary.to_string(index=False))
    Path(a.out).write_text(json.dumps({"year": a.year, "times": a.times, "source": "ARCO-ERA5 0.25 deg, subsampled",
                                       "summary": summary.to_dict("records"), "rows": rows}, indent=1, default=float))


if __name__ == "__main__":
    main()
