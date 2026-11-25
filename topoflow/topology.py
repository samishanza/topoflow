"""Critical points of a 2-D wind field on a lat/lon grid. Deterministic; no model decides anything here.

Method: split every grid cell into two triangles and treat u and v as linear on each (the standard
piecewise-linear interpolant). On a triangle a linear field vanishes at most once, so each zero is isolated and found
exactly by solving a 2x2 system. Its Jacobian, taken in metres so longitude shrinks with latitude, classifies it:

    det J < 0                      saddle
    det J > 0, tr^2 < 4 det        focus  (spiral): attracting if tr < 0, repelling if tr > 0
    det J > 0, tr^2 >= 4 det       node:            attracting (sink) if tr < 0, repelling (source) if tr > 0

Attracting nodes and foci are where surface air converges, so where emitted CH4 can pool under a shallow boundary
layer. That link is a hypothesis this tool helps look at; it is not established by finding the points.
"""
from __future__ import annotations

import numpy as np

from .units import EARTH_RADIUS_M

CLASSES = ("saddle", "attracting_node", "repelling_node", "attracting_focus", "repelling_focus", "center")


def _wrap_global(lons: np.ndarray, *fields: np.ndarray):
    """Close the seam when the grid spans the globe, so a zero between 175E and 180W is not missed."""
    step = float(lons[1] - lons[0]) if len(lons) > 1 else 0
    if len(lons) > 2 and abs((lons[-1] + step) - (lons[0] + 360)) < 1e-6:
        lons2 = np.append(lons, lons[0] + 360)
        return lons2, [np.concatenate([f, f[:, :1]], axis=1) for f in fields]
    return lons, list(fields)


def critical_points(u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray, *,
                    center_tolerance: float = 0.05) -> list[dict]:
    """Zeros of (u, v) on one time slice. u, v are (lat, lon) in m s-1; lats ascending."""
    lons2, (u2, v2) = _wrap_global(np.asarray(lons, float), np.asarray(u, float), np.asarray(v, float))
    lats = np.asarray(lats, float)
    ny, nx = u2.shape
    if ny < 2 or nx < 2:
        return []
    # Corner indices of the two triangles in every cell: (i,j),(i,j+1),(i+1,j) and (i+1,j+1),(i+1,j),(i,j+1).
    I, J = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    I, J = I.ravel(), J.ravel()
    tris = [
        ((I, J), (I, J + 1), (I + 1, J)),
        ((I + 1, J + 1), (I + 1, J), (I, J + 1)),
    ]
    out: list[dict] = []
    seen: set[tuple[float, float]] = set()
    for (a, b, c) in tris:
        U0, U1, U2 = u2[a], u2[b], u2[c]
        V0, V1, V2 = v2[a], v2[b], v2[c]
        ok = np.isfinite(U0 + U1 + U2 + V0 + V1 + V2)
        # Solve [[dU1 dU2],[dV1 dV2]] [s, t] = [-U0, -V0] for the barycentric zero.
        a11, a12, a21, a22 = U1 - U0, U2 - U0, V1 - V0, V2 - V0
        det_ab = a11 * a22 - a12 * a21
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (-U0 * a22 + a12 * V0) / det_ab
            t = (-a11 * V0 + a21 * U0) / det_ab
        eps = 1e-9
        with np.errstate(invalid="ignore"):
            inside = ok & (np.abs(det_ab) > 1e-12) & (s >= -eps) & (t >= -eps) & (s + t <= 1 + eps)
        for k in np.nonzero(inside)[0]:
            ia, ja = a[0][k], a[1][k]
            ib, jb = b[0][k], b[1][k]
            ic, jc = c[0][k], c[1][k]
            lat0, lon0 = lats[ia], lons2[ja]
            lat_p = lat0 + s[k] * (lats[ib] - lat0) + t[k] * (lats[ic] - lat0)
            lon_p = lon0 + s[k] * (lons2[jb] - lon0) + t[k] * (lons2[jc] - lon0)
            lon_p = ((lon_p + 180) % 360) - 180
            key = (round(float(lat_p), 5), round(float(lon_p), 5))
            if key in seen:  # a zero exactly on a shared edge is found by both triangles
                continue
            seen.add(key)
            # Edge vectors in metres at the point's latitude.
            kx = EARTH_RADIUS_M * np.cos(np.deg2rad(lat_p)) * np.pi / 180
            ky = EARTH_RADIUS_M * np.pi / 180
            E = np.array([[(lons2[jb] - lon0) * kx, (lons2[jc] - lon0) * kx],
                          [(lats[ib] - lat0) * ky, (lats[ic] - lat0) * ky]])
            Jab = np.array([[a11[k], a12[k]], [a21[k], a22[k]]])
            try:
                Jm = Jab @ np.linalg.inv(E)
            except np.linalg.LinAlgError:
                continue
            out.append(classify(Jm, float(lat_p), float(lon_p), center_tolerance))
    return out


def classify(Jm: np.ndarray, lat: float, lon: float, center_tolerance: float = 0.05) -> dict:
    det = float(np.linalg.det(Jm))
    tr = float(np.trace(Jm))
    disc = tr * tr - 4 * det
    if det < 0:
        kind = "saddle"
    elif disc < 0:
        if abs(tr) < center_tolerance * 2 * np.sqrt(det):
            kind = "center"
        else:
            kind = "attracting_focus" if tr < 0 else "repelling_focus"
    else:
        kind = "attracting_node" if tr < 0 else "repelling_node"
    return {
        "lat": round(lat, 4), "lon": round(lon, 4), "kind": kind,
        # Poincare index: -1 for a saddle, +1 for everything else.
        "index": -1 if kind == "saddle" else 1,
        "divergence": tr,  # s-1; the trace of J is the local divergence
        "vorticity": float(Jm[1, 0] - Jm[0, 1]),
        "strength": float(np.sqrt(abs(det))),  # s-1; how sharply the flow turns about the point
    }


def family(kind: str) -> str:
    """The three families the original dashboard drew: Node, Saddle, Focus."""
    if kind == "saddle":
        return "Saddle"
    if kind.endswith("node"):
        return "Node"
    return "Focus"


def div_vort(u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal divergence and relative vorticity on the sphere, by centred differences. Arrays are (..., lat, lon)."""
    phi = np.deg2rad(np.asarray(lats, float))
    lam = np.deg2rad(np.asarray(lons, float))
    R = EARTH_RADIUS_M
    cos = np.cos(phi)[:, None]
    tan = np.tan(phi)[:, None]
    du_dlam = np.gradient(u, lam, axis=-1)
    dv_dlam = np.gradient(v, lam, axis=-1)
    du_dphi = np.gradient(u, phi, axis=-2)
    dv_dphi = np.gradient(v, phi, axis=-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        div = du_dlam / (R * cos) + dv_dphi / R - v * tan / R
        vort = dv_dlam / (R * cos) - du_dphi / R + u * tan / R
    polar = np.abs(np.asarray(lats)) > 88
    div[..., polar, :] = np.nan
    vort[..., polar, :] = np.nan
    return div.astype("float32"), vort.astype("float32")
