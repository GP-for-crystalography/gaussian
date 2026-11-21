"""Testy sekcji 2d (dwumodalność, test mieszaniny, asymetria stron) na danych syntetycznych."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import domain_tests as dt        # noqa: E402


def test_bimodality_detects_two_phases():
    pytest.importorskip("diptest")
    rng = np.random.default_rng(0)
    two = np.r_[rng.normal(0, 0.1, 1500), rng.normal(0.9, 0.15, 500)]
    one = rng.gamma(2.0, 0.2, 2000)                           # jednomodalny, skośny
    b2, b1 = dt.bimodality(two), dt.bimodality(one)
    assert b2["p"] < 0.01 and b1["p"] > 0.05
    assert np.argmin(b2["bic"]) >= 1 and b2["valley_ratio"] < 0.5


def _stack(three_state, seed=0, H=40, W=40, n=120):
    rng = np.random.default_rng(seed)
    i, j = np.mgrid[0:H, 0:W]
    w = gaussian_filter((((i + j) // 6) % 2).astype(float), 1.2)       # ułamek stanu 2
    ph = 2 * np.pi * np.arange(n) / n
    smooth = lambda k: 1 + 0.5 * np.sin(k * ph + rng.uniform(0, 6)) + 0.3 * np.cos((k + 2) * ph)
    Pm, Ps, P3 = smooth(1), smooth(3), smooth(5)
    Y = (1 - w)[..., None] * Pm + w[..., None] * Ps
    if three_state:
        Y = Y + (0.5 * 4 * w * (1 - w))[..., None] * P3                # trzeci stan tylko na krawędziach
    Y = 1000 * (Y + rng.normal(0, 0.02, Y.shape))
    return Y, np.ones((H, W, n), bool), w


def test_mixture_two_states_vs_three_states():
    Y, v, w = _stack(False)
    ok = np.ones(w.shape, bool)
    kw = dict(core=0.95, bg=0.05, mid=(0.2, 0.8))        # rdzeń bez domieszki trzeciego stanu
    r2 = dt.mixture_test(Y, v, w, ok, **kw)
    Y3, v3, w3 = _stack(True)
    r3 = dt.mixture_test(Y3, v3, w3, ok, **kw)
    assert r2["ratio_median"] < 1.5 and r2["w_corr"] > 0.9
    assert r3["ratio_median"] > 3.0
    assert r3["coh_mid"] > 0.3 and r2["coh_mid"] < 0.1
    assert r3["eig_mid"][0] > r3["eig_pure"][0]       # PCA wycentrowane: stała składowa jest w coh


def _stripes(asym, seed=0, H=62, W=62, alpha=49.0, period=9.0):
    rng = np.random.default_rng(seed)
    a = np.deg2rad(alpha)
    ii, jj = np.mgrid[0:H, 0:W]
    u = -np.sin(a) * (jj - (W - 1) / 2) + np.cos(a) * (ii - (H - 1) / 2)
    f = np.zeros((H, W))
    for k in range(-12, 13):
        s = u - 0.3 - k * period
        sig = np.where(s > 0, 1.0, 2.0) if asym else 1.5
        f += np.exp(-0.5 * (s / sig) ** 2)
    return f + rng.normal(0, 0.05, f.shape)


def test_side_asymmetry_detects_asymmetric_and_not_symmetric():
    ok = np.ones((62, 62), bool)
    ra = dt.side_asymmetry(_stripes(True), ok, 49.0)
    rs = dt.side_asymmetry(_stripes(False), ok, 49.0)
    assert ra["n_peaks"] >= 5
    assert ra["A_mean"] < -0.1 and ra["p_t"] < 0.01              # szersza strona u < 0 → P(−s) > P(+s)
    assert abs(rs["A_mean"]) < 0.08 and rs["p_t"] > 0.01
