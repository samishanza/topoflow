"""The CH4 statistics find a planted effect and do not invent one."""
import numpy as np

from topoflow.ch4stats import band_anomaly, benjamini_hochberg, composite, lag_correlation, persistent_zones

LATS = np.arange(-60, 61, 2.0)
DAYS, NY, NX = 120, len(LATS), 60


def fields(effect_ppb: float, seed: int):
    rng = np.random.default_rng(seed)
    # Smooth, autocorrelated stagnation zones that drift, and CH4 with a latitude gradient, a trend, and noise.
    zones = np.zeros((DAYS, NY, NX), bool)
    c = rng.uniform(0, NX, 8)
    for d in range(DAYS):
        c = (c + rng.normal(0, 0.5, 8)) % NX
        for k, x in enumerate(c):
            zones[d, 10 + 5 * k: 13 + 5 * k, int(x): int(x) + 3] = True
    base = 1900 + 0.5 * LATS[None, :, None] + 0.05 * np.arange(DAYS)[:, None, None]
    ch4 = base + rng.normal(0, 15, (DAYS, NY, NX))
    ch4 = ch4 + effect_ppb * zones
    ch4[rng.random(ch4.shape) < 0.3] = np.nan  # retrieval gaps
    return zones, ch4


def test_band_anomaly_removes_gradient_and_trend():
    _, ch4 = fields(0, 1)
    a = band_anomaly(ch4, LATS)
    assert abs(np.nanmean(a)) < 1
    assert abs(np.corrcoef(np.nanmean(a, axis=(0, 2)), LATS)[0, 1]) < 0.5


def test_planted_enhancement_is_found():
    zones, ch4 = fields(12.0, 2)
    c = composite(band_anomaly(ch4, LATS), zones, LATS, n_boot=300)
    assert 8 < c.difference < 16
    assert c.p_value < 0.01
    assert c.ci95[0] > 0


def test_no_effect_is_not_significant():
    zones, ch4 = fields(0.0, 3)
    c = composite(band_anomaly(ch4, LATS), zones, LATS, n_boot=300)
    assert abs(c.difference) < 3
    assert c.p_value > 0.05


def test_persistent_zones_need_24_hours():
    occ = np.zeros((16, 2, 2), bool)  # 4 days of 6-hourly steps
    occ[2:6, 0, 0] = True  # 24 h run spanning days 0-1
    occ[9:11, 1, 1] = True  # 12 h only
    z = persistent_zones(occ)
    assert z[:, 0, 0].tolist() == [True, True, False, False]
    assert not z[:, 1, 1].any()


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, np.nan])
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True, False, False, False, False, False, False, False]


def test_lag_correlation_recovers_a_two_day_lag():
    rng = np.random.default_rng(5)
    share = np.convolve(rng.normal(size=400), np.ones(5) / 5, mode="same")
    anom = np.roll(share, 2) * 10 + rng.normal(0, 0.5, 400)
    res = {r["lag_days"]: r["r"] for r in lag_correlation(share, anom, n_boot=200)}
    assert max(res, key=res.get) == 2
