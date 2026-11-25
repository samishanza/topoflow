"""How much do critical points depend on grid resolution? Measured, not guessed.

For random smooth wind fields over a 10 x 10 degree box, the points found on a 0.05 degree grid are the reference.
Each coarser grid (the resolutions TopoFlow runs at) is scored against it:

  recall     share of reference points with a coarse point of the same family within one coarse cell
  precision  share of coarse points with a reference point of the same family within one coarse cell
  class acc. among matched points, share whose exact kind (attracting/repelling too) agrees

The fields are synthetic and smooth at about 1.5 degrees, roughly like 10 m wind away from fronts and coasts. Real
wind has sharper structure, so these numbers are an upper bound on how well a coarse grid does.

    python scripts/eval_topology.py [--fields 40] [--seed 1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topoflow.topology import critical_points, family  # noqa: E402

BOX = (0.0, 30.0, 10.0, 40.0)  # west, south, east, north
REFERENCE_RES = 0.05
RESOLUTIONS = (0.25, 0.5, 1.0, 2.5)


def random_field(rng: np.random.Generator, n_modes: int = 6, scale_deg: float = 1.5):
    """A smooth (u, v) built from random Fourier modes; returns a function of (lat, lon) arrays."""
    k = rng.normal(0, 1 / scale_deg, size=(n_modes, 2, 2))
    phase = rng.uniform(0, 2 * np.pi, size=(n_modes, 2))
    amp = rng.normal(0, 1, size=(n_modes, 2))
    mean = rng.normal(0, 0.3, size=2)

    def f(la, lo):
        u = np.full_like(la, mean[0])
        v = np.full_like(la, mean[1])
        for m in range(n_modes):
            u += amp[m, 0] * np.sin(k[m, 0, 0] * lo + k[m, 0, 1] * la + phase[m, 0])
            v += amp[m, 1] * np.sin(k[m, 1, 0] * lo + k[m, 1, 1] * la + phase[m, 1])
        return u, v

    return f


def detect(f, res):
    w, s, e, n = BOX
    lats = np.arange(s, n + res / 2, res)
    lons = np.arange(w, e + res / 2, res)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    u, v = f(la, lo)
    return critical_points(u, v, lats, lons)


def match(a, b, tol):
    """Greedy nearest matching within tol degrees and the same family. Returns matched pairs."""
    used, pairs = set(), []
    for p in a:
        best, bd = None, tol
        for j, q in enumerate(b):
            if j in used or family(p["kind"]) != family(q["kind"]):
                continue
            d = np.hypot(p["lat"] - q["lat"], p["lon"] - q["lon"])
            if d <= bd:
                best, bd = j, d
        if best is not None:
            used.add(best)
            pairs.append((p, b[best]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    # Ignore points within one coarse cell of the edge: no grid can see a zero it has no cell around.
    w, s, e, n = BOX
    totals = {r: {"ref": 0, "found": 0, "tp_r": 0, "tp_p": 0, "cls": 0} for r in RESOLUTIONS}
    for _ in range(a.fields):
        f = random_field(rng)
        ref = detect(f, REFERENCE_RES)
        for r in RESOLUTIONS:
            inner = lambda p, m=r: (w + m <= p["lon"] <= e - m) and (s + m <= p["lat"] <= n - m)  # noqa: E731
            R = [p for p in ref if inner(p)]
            C = [p for p in detect(f, r) if inner(p)]
            t = totals[r]
            t["ref"] += len(R)
            t["found"] += len(C)
            pr = match(R, C, tol=r)
            t["tp_r"] += len(pr)
            t["cls"] += sum(p["kind"] == q["kind"] for p, q in pr)
            t["tp_p"] += len(match(C, R, tol=r))
    print(f"{a.fields} random fields, reference grid {REFERENCE_RES} deg, box {BOX}\n")
    print(f"{'grid':>6} {'ref pts':>8} {'found':>7} {'recall':>7} {'precision':>9} {'class acc.':>10}")
    for r, t in totals.items():
        rec = t["tp_r"] / t["ref"] if t["ref"] else float("nan")
        prec = t["tp_p"] / t["found"] if t["found"] else float("nan")
        acc = t["cls"] / t["tp_r"] if t["tp_r"] else float("nan")
        print(f"{r:>5}° {t['ref']:>8} {t['found']:>7} {rec:>7.2f} {prec:>9.2f} {acc:>10.2f}")


if __name__ == "__main__":
    main()
