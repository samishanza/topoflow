"""Full-sphere stagnation detection and the Poincaré-Hopf closure test."""
import numpy as np
import pytest

from topoflow.sphere import analyse

LATS = np.arange(-90, 90.1, 2.0)
LONS = np.arange(0, 360, 2.0)
LA, LO = np.meshgrid(LATS, LONS, indexing="ij")
PHI, LAM = np.deg2rad(LA), np.deg2rad(LO)


def tangent_of(F):
    """Project a 3-D field (..., 3) onto the local east and north directions."""
    east = np.stack([-np.sin(LAM), np.cos(LAM), 0 * LAM], -1)
    north = np.stack([-np.sin(PHI) * np.cos(LAM), -np.sin(PHI) * np.sin(LAM), np.cos(PHI)], -1)
    return (F * east).sum(-1), (F * north).sum(-1)


def test_northward_flow_has_a_source_and_a_sink_at_the_poles():
    r = analyse(0 * LA, np.cos(PHI), LATS, LONS)
    assert (r["cap_south"], r["cap_north"]) == (1, 1)  # source and sink both have index +1
    assert r["winding_sum"] == 2 and r["n_points"] == 0


def test_solid_body_rotation_has_two_centres():
    r = analyse(np.cos(PHI), 0 * LA, LATS, LONS)
    assert (r["cap_south"], r["cap_north"], r["winding_sum"]) == (1, 1, 2)


@pytest.mark.parametrize("seed", range(5))
def test_smooth_random_fields_close_at_two(seed):
    rng = np.random.default_rng(seed)
    X = np.stack([np.cos(PHI) * np.cos(LAM), np.cos(PHI) * np.sin(LAM), np.sin(PHI)], -1)
    F = np.zeros_like(X)
    for _ in range(6):
        F += rng.normal(0, 1, 3) * np.sin(X @ rng.normal(0, 2, 3) + rng.uniform(0, 6.3))[..., None]
    u, v = tangent_of(F)
    r = analyse(u, v, LATS, LONS)
    assert r["winding_sum"] == 2
    assert r["point_sum"] == r["winding_sum"]  # located points carry exactly the cells' winding
    assert r["edge_error"] == 0


def test_identity_holds_on_a_field_too_rough_for_the_grid():
    rng = np.random.default_rng(11)
    u, v = rng.normal(size=LA.shape), rng.normal(size=LA.shape)  # white noise: neighbours unrelated
    r = analyse(u, v, LATS, LONS)
    assert r["winding_sum"] == 2 + r["edge_error"]
    assert r["edge_flips"] > 0


def test_longitude_order_and_range_do_not_matter():
    rng = np.random.default_rng(2)
    X = np.stack([np.cos(PHI) * np.cos(LAM), np.cos(PHI) * np.sin(LAM), np.sin(PHI)], -1)
    u, v = tangent_of(np.sin(X @ rng.normal(0, 2, (3, 3))))
    a = analyse(u, v, LATS, LONS)
    shifted = ((LONS + 180) % 360) - 180
    order = np.argsort(shifted)
    b = analyse(u[:, order][::-1], v[:, order][::-1], LATS[::-1], shifted[order])  # -180..180, north first
    assert a["winding_sum"] == b["winding_sum"] and a["n_points"] == b["n_points"]
