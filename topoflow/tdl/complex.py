"""The sphere as a simplicial complex: an icosphere with oriented boundary matrices and Hodge Laplacians.

Vertices, edges and triangles of a subdivided icosahedron, projected to the unit sphere.
  B1  (V x E): edge (a -> b), a < b, has -1 at a and +1 at b
  B2  (E x F): triangle (i, j, k) counter-clockwise seen from outside; +1 where an edge runs with the triangle's
               boundary, -1 where against it
  B1 @ B2 == 0 (the boundary of a boundary is empty), and V - E + F == 2 (Euler characteristic of the sphere).

Hodge Laplacians on 1-cochains (edge signals, here: wind flowing along edges):
  L1_down = B1^T B1   (gradient part: how flows meet at vertices, i.e. divergence)
  L1_up   = B2 B2^T   (curl part: how flows circulate around triangles, i.e. vorticity)
On a sphere the first Betti number is 0, so L1 = L1_down + L1_up has no kernel: every edge flow is a gradient plus a
curl, with no harmonic part. That is the same topology that forces stagnation points (Hairy Ball, Poincaré-Hopf).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Complex:
    xyz: np.ndarray        # (V, 3) unit vectors
    lat: np.ndarray        # (V,)
    lon: np.ndarray        # (V,)
    edges: np.ndarray      # (E, 2) with a < b
    faces: np.ndarray      # (F, 3) counter-clockwise from outside
    B1: sp.csr_matrix      # (V, E)
    B2: sp.csr_matrix      # (E, F)

    @property
    def L0(self):
        return (self.B1 @ self.B1.T).tocsr()

    @property
    def L1_down(self):
        return (self.B1.T @ self.B1).tocsr()

    @property
    def L1_up(self):
        return (self.B2 @ self.B2.T).tocsr()


def _icosahedron():
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                  [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
                  [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    return v / np.linalg.norm(v, axis=1, keepdims=True), f


def icosphere(level: int = 5) -> Complex:
    verts, faces = _icosahedron()
    verts = list(verts)
    for _ in range(level):
        cache: dict[tuple[int, int], int] = {}

        def mid(a, b):
            key = (a, b) if a < b else (b, a)
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        new = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = np.array(new)
    xyz = np.array(verts)
    # Make every face counter-clockwise as seen from outside.
    n = np.cross(xyz[faces[:, 1]] - xyz[faces[:, 0]], xyz[faces[:, 2]] - xyz[faces[:, 0]])
    flip = (n * xyz[faces].mean(1)).sum(1) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]

    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges, inverse = np.unique(e, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    V, E, F = len(xyz), len(edges), len(faces)
    B1 = sp.csr_matrix((np.r_[-np.ones(E), np.ones(E)], (np.r_[edges[:, 0], edges[:, 1]], np.r_[np.arange(E), np.arange(E)])),
                       shape=(V, E))
    rows, cols, vals = [], [], []
    for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        a, b = faces[:, i], faces[:, j]
        eid = inverse[k * F:(k + 1) * F]
        rows.append(eid)
        cols.append(np.arange(F))
        vals.append(np.where(a < b, 1.0, -1.0))  # face boundary runs a -> b; the edge is stored low -> high
    B2 = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(E, F))
    lat = np.degrees(np.arcsin(np.clip(xyz[:, 2], -1, 1)))
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    return Complex(xyz, lat, lon, edges, faces, B1, B2)


def edge_frames(cx: Complex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midpoint (lat, lon) of every edge and the unit tangent along it (a -> b), as east/north components."""
    a, b = cx.xyz[cx.edges[:, 0]], cx.xyz[cx.edges[:, 1]]
    m = a + b
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    d = b - a
    d -= (d * m).sum(1, keepdims=True) * m
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    lat = np.arcsin(np.clip(m[:, 2], -1, 1))
    lon = np.arctan2(m[:, 1], m[:, 0])
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], 1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], 1)
    return np.degrees(lat), np.degrees(lon), np.stack([(d * east).sum(1), (d * north).sum(1)], 1)
