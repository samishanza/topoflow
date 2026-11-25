"""Stagnation points on the whole sphere, with a Poincaré-Hopf closure test.

Two independent measurements per time step:

1. Cell winding indices (exact integers). For every grid cell, the wind at its four corners is carried into the tangent
   plane at the cell centre and the winding number is counted going counter-clockwise as seen from outside. Each polar
   cap (poleward of the last full latitude ring) gets its winding from the ring, in a frame centred on the pole.
   For a continuous tangent field on a sphere the indices must sum to chi(S^2) = 2 (Hairy Ball / Poincaré-Hopf).
   Corner sampling reads each edge's turn as the shortest rotation, so when the wind turns by more than half a turn
   between neighbouring grid points the count is wrong; at coarse resolution that is what moves the sum off 2.

2. Located and classified points. Inside each cell, sub-grid roots are solved on the two linear triangles and
   classified by the Jacobian (topology.classify). Summing their Poincaré indices (-1 saddle, +1 otherwise) plus the
   caps is a second, independent closure test: it fails when roots are missed, doubled, or degenerate.

Coordinates: lats strictly ascending, poles excluded from the interior; lons ascending and spanning the globe.
"""
from __future__ import annotations

import numpy as np

from .topology import classify  # noqa: F401  (re-exported for callers)

TWO_PI = 2 * np.pi


def _wrap(a):
    return (a + np.pi) % TWO_PI - np.pi


def _frames(lat_deg, lon_deg):
    """3-D east and north unit vectors at (lat, lon). Shapes broadcast."""
    la, lo = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    east = np.stack(np.broadcast_arrays(-np.sin(lo), np.cos(lo), np.zeros_like(lo * la)), -1)
    north = np.stack(np.broadcast_arrays(-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la) + 0 * lo), -1)
    return east, north


def _to3d(u, v, lat_deg, lon_deg):
    e, n = _frames(lat_deg, lon_deg)
    return u[..., None] * e + v[..., None] * n


def prepare(u, v, lats, lons):
    """Sort to ascending lat/lon in [-180, 180), drop exact pole rows, and close the longitude seam."""
    lats = np.asarray(lats, float)
    lons = ((np.asarray(lons, float) + 180) % 360) - 180
    oi, oj = np.argsort(lats), np.argsort(lons)
    lats, lons = lats[oi], lons[oj]
    u = np.asarray(u, float)[..., oi, :][..., :, oj]
    v = np.asarray(v, float)[..., oi, :][..., :, oj]
    keep = np.abs(lats) < 90 - 1e-6
    lats, u, v = lats[keep], u[..., keep, :], v[..., keep, :]
    # Break exact degeneracies (a zero vector on a grid node, exactly opposite neighbours), which packed data such as
    # NCEP's 0.01 m/s storage produces. The perturbation is deterministic and 1e5 times below that precision, so no
    # real value changes; it only decides ties the way "simulation of simplicity" does.
    ii, jj = np.meshgrid(np.arange(u.shape[-2]), np.arange(u.shape[-1]), indexing="ij")
    u = u + 1e-7 * np.sin(1.3 * ii + 2.7 * jj + 0.5)
    v = v + 1e-7 * np.cos(2.1 * ii + 0.9 * jj + 1.1)
    lons_c = np.append(lons, lons[0] + 360)
    u = np.concatenate([u, u[..., :, :1]], axis=-1)
    v = np.concatenate([v, v[..., :, :1]], axis=-1)
    return u, v, lats, lons_c


def _corner_components(u, v, lats, lons_c):
    """Corner vectors of every cell, read in the tangent plane at the cell centre: (4, ny-1, nx, 2), SW SE NE NW."""
    LA, LO = np.meshgrid(lats, lons_c, indexing="ij")
    V = _to3d(u, v, LA, LO)  # (ny, nx+1, 3)
    clat = 0.5 * (lats[:-1] + lats[1:])
    clon = 0.5 * (lons_c[:-1] + lons_c[1:])
    E, N = _frames(clat[:, None], clon[None, :])
    corners = [V[:-1, :-1], V[:-1, 1:], V[1:, 1:], V[1:, :-1]]  # counter-clockwise as seen from outside
    return np.stack([np.stack([(c * E).sum(-1), (c * N).sum(-1)], -1) for c in corners]), clat, clon


def cell_indices(u, v, lats, lons_c):
    """Winding index of each cell (ny-1, nx), and the signed closure error carried by each cell's east and north edges.

    Every interior edge is shared by two cells and read once in each cell's tangent frame. When the wind at the edge's
    two ends points nearly opposite ways, the two readings can turn opposite ways round, and that edge adds +-1 to the
    global sum. `edge_error` records those flips, so sum(edge_error) is exactly the departure from chi = 2 that the
    interior cells cause. It is zero on a fine enough grid.
    """
    C, _, _ = _corner_components(u, v, lats, lons_c)
    ang = np.arctan2(C[..., 1], C[..., 0])
    steps = np.stack([_wrap(ang[(k + 1) % 4] - ang[k]) for k in range(4)])  # 0 S, 1 E, 2 N, 3 W edges
    total = steps.sum(0)
    idx = np.rint(np.nan_to_num(total) / TWO_PI).astype(int)
    err = np.zeros(idx.shape, dtype=int)
    # East edge of (i, j) vs west edge of (i, j+1), read in opposite directions: consistent when they cancel.
    east = steps[1] + np.roll(steps[3], -1, axis=1)
    err += np.rint(np.nan_to_num(east) / TWO_PI).astype(int)
    # North edge of (i, j) vs south edge of (i+1, j).
    north = steps[2][:-1] + steps[0][1:]
    err[:-1] += np.rint(np.nan_to_num(north) / TWO_PI).astype(int)
    # Polar ring edges are read once in the pole's frame (by the cap) and once by the cells along the ring.
    ring = np.zeros((2, idx.shape[1]), dtype=int)
    lo = lons_c[:-1]
    for r_i, (row, cell_step) in enumerate(((-1, steps[2][-1]), (0, steps[0][0]))):
        V = _to3d(u[row, :-1], v[row, :-1], np.full_like(lo, lats[row]), lo)
        a = np.arctan2(V[:, 1], V[:, 0])
        cap = _wrap(np.diff(np.append(a, a[0])))
        ring[r_i] = np.rint(np.nan_to_num(cap + cell_step) / TWO_PI).astype(int)
    err[-1] += ring[0]
    err[0] += ring[1]
    return idx, err


def cap_indices(u, v, lats, lons_c) -> tuple[int, int]:
    """Winding of each polar cap from its boundary ring, counter-clockwise as seen from outside the sphere.

    A winding number does not depend on orientation as long as the loop and the vectors flip together, so both caps
    are counted the same way: the ring traversed with longitude increasing, vectors read in the (x, y) plane.
    (Mirroring only the vectors at the south pole would flip the sign; tests/test_sphere.py pins this.)
    """
    lo = lons_c[:-1]
    out = []
    for row in (-1, 0):
        V = _to3d(u[row, :-1], v[row, :-1], np.full_like(lo, lats[row]), lo)
        a = np.arctan2(V[:, 1], V[:, 0])
        s = _wrap(np.diff(np.append(a, a[0]))).sum()
        out.append(int(np.rint(s / TWO_PI)) if np.isfinite(s) else 0)
    return out[0], out[1]


def stagnation_points(u, v, lats, lons_c, radius_m: float | None = None) -> list[dict]:
    """Sub-grid roots on the two linear triangles of every cell, from the same tangent-plane corner vectors as the
    winding count, so a cell's located points always carry its winding index. Classified by the Jacobian in metres."""
    from .units import EARTH_RADIUS_M

    radius = radius_m or EARTH_RADIUS_M
    C, clat, clon = _corner_components(u, v, lats, lons_c)
    dlat = np.diff(lats)[:, None] * np.ones((1, len(clon)))
    dlon = np.diff(lons_c)[None, :] * np.ones((len(clat), 1))
    ky = radius * np.pi / 180
    kx = ky * np.cos(np.deg2rad(clat))[:, None]
    hx, hy = dlon * kx / 2, dlat * ky / 2
    # Corner positions in metres from the cell centre: SW, SE, NE, NW.
    P = np.stack([np.stack([-hx, -hy], -1), np.stack([hx, -hy], -1), np.stack([hx, hy], -1), np.stack([-hx, hy], -1)])
    out = []
    for a, b, c in ((0, 1, 3), (2, 3, 1)):  # SW-SE-NW and NE-NW-SE
        A0, A1, A2 = C[a], C[b], C[c]
        a11, a12 = A1[..., 0] - A0[..., 0], A2[..., 0] - A0[..., 0]
        a21, a22 = A1[..., 1] - A0[..., 1], A2[..., 1] - A0[..., 1]
        det = a11 * a22 - a12 * a21
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (-A0[..., 0] * a22 + a12 * A0[..., 1]) / det
            r = (-a11 * A0[..., 1] + a21 * A0[..., 0]) / det
            eps = 1e-9
            inside = np.isfinite(s + r) & (np.abs(det) > 1e-12) & (s >= -eps) & (r >= -eps) & (s + r <= 1 + eps)
        for i, j in np.argwhere(inside):
            p0, p1, p2 = P[a][i, j], P[b][i, j], P[c][i, j]
            E = np.column_stack([p1 - p0, p2 - p0])
            Jab = np.array([[a11[i, j], a12[i, j]], [a21[i, j], a22[i, j]]])
            try:
                Jm = Jab @ np.linalg.inv(E)
            except np.linalg.LinAlgError:
                continue
            xy = p0 + s[i, j] * (p1 - p0) + r[i, j] * (p2 - p0)
            lat = clat[i] + xy[1] / ky
            lon = ((clon[j] + xy[0] / kx[i, 0] + 180) % 360) - 180
            out.append({**classify(Jm, float(lat), float(lon)), "cell": (int(i), int(j))})
    return out


def analyse(u, v, lats, lons, radius_m: float | None = None) -> dict:
    """One time step: located points, cell index field, cap indices, and both closure sums."""
    u2, v2, la, lo = prepare(u, v, lats, lons)
    idx, err = cell_indices(u2, v2, la, lo)
    north, south = cap_indices(u2, v2, la, lo)
    pts = stagnation_points(u2, v2, la, lo, radius_m)
    # Points the triangle solver finds in the cells nearest the poles are inside the caps' rings already; the caps
    # are counted from the ring winding, so nothing is double counted here.
    winding_sum = int(idx.sum()) + north + south
    point_sum = int(sum(p["index"] for p in pts)) + north + south
    return {
        "points": pts,
        "cell_index": idx,
        "edge_flips": int(np.abs(err).sum()),
        "edge_error": int(err.sum()),
        "edge_error_field": err,
        "cap_north": north,
        "cap_south": south,
        "winding_sum": winding_sum,
        "point_sum": point_sum,
        "n_points": len(pts),
        "lats": la,
        "lons": lo[:-1],
    }
