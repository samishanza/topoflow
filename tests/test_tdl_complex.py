"""The icosphere is a valid simplicial complex of the sphere, and the wind cochain is built along its edges."""
import numpy as np
import pytest

from topoflow.tdl.complex import edge_frames, icosphere


@pytest.mark.parametrize("level", [0, 2, 3])
def test_euler_characteristic_and_boundary_of_boundary(level):
    cx = icosphere(level)
    V, E, F = len(cx.xyz), len(cx.edges), len(cx.faces)
    assert V - E + F == 2
    assert abs(cx.B1 @ cx.B2).sum() == 0


def test_no_harmonic_edge_flows_on_a_sphere():
    cx = icosphere(2)
    w = np.linalg.eigvalsh((cx.L1_down + cx.L1_up).toarray())
    assert w.min() > 1e-6  # Betti_1(S^2) = 0


def test_one_connected_component():
    cx = icosphere(2)
    assert int((np.abs(np.linalg.eigvalsh(cx.L0.toarray())) < 1e-9).sum()) == 1


def test_edge_tangents_are_unit_and_point_from_a_to_b():
    cx = icosphere(2)
    lat, lon, tan = edge_frames(cx)
    assert np.allclose(np.hypot(tan[:, 0], tan[:, 1]), 1)
    # An eastward wind flows positively along edges whose head lies east of their tail.
    flow = tan[:, 0]
    a, b = cx.lon[cx.edges[:, 0]], cx.lon[cx.edges[:, 1]]
    # Longitude is undefined at a pole vertex, so edges touching one are left out of this check.
    off_pole = (np.abs(cx.lat[cx.edges[:, 0]]) < 89) & (np.abs(cx.lat[cx.edges[:, 1]]) < 89)
    east = ((((b - a + 180) % 360) - 180) > 1) & off_pole
    assert east.sum() > 100 and (flow[east] > 0).all()
