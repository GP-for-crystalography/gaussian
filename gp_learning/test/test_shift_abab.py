"""Testy sekcji 2e (przesunięcia klatek, znak flank, test ABAB) na danych syntetycznych."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import domain_tests as dt        # noqa: E402

H = W = 62
ALPHA = 49.0
PERIOD = 8.7


def _bumps(u, shift=0.0, sig=1.2):
    s = np.zeros_like(u)
    for k in range(-14, 15):
        s += np.exp(-0.5 * ((u - shift - 0.4 - k * PERIOD) / sig) ** 2)
    return s


def _shifted_stack(seed=0, n=180):
    """Profil piksela = P_matryca(ω) + w(x − δ_k)·ΔP(ω): paski różnią się KSZTAŁTEM profilu
    (różnica samej amplitudy zniknęłaby w z-score)."""
    rng = np.random.default_rng(seed)
    u = dt.u_grid(H, W, ALPHA)
    om = 2.0 * np.arange(n)
    w = np.deg2rad(om)
    Pm = 1 + 0.5 * np.sin(2 * w + 0.5)
    dP = 0.15 * np.cos(3 * w)                                         # zmienia znak, zero-średnia
    delta = 0.6 * np.sin(w) + 0.3                                     # przesunięcie okresowe + stałe
    Wk = np.stack([_bumps(u, delta[k]) for k in range(n)], -1)
    Y = Pm[None, None, :] + Wk * dP[None, None, :] + rng.normal(0, 0.01, (H, W, n))
    return 1000 * Y, np.ones((H, W, n), bool), om, delta, _bumps(u, 0.0)


def test_frame_shifts_recovers_known_shifts_and_correction_removes_them():
    Y, v, om, delta, S = _shifted_stack()
    ok = np.ones((H, W), bool)
    fs = dt.frame_shifts(Y, v, S, ALPHA, ok)
    good = fs["weight"] > np.quantile(fs["weight"], 0.5)
    d_true = delta - np.average(delta, weights=np.where(np.isfinite(fs["weight"]), fs["weight"], 0))
    err = fs["delta_rel"][good] - d_true[good]
    assert np.corrcoef(fs["delta_rel"][good], d_true[good])[0, 1] > 0.95
    assert np.sqrt(np.mean(err ** 2)) < 0.15
    fm = dt.fit_shift_model(om, fs["delta_rel"], fs["weight"], harmonics=2)
    assert abs(fm["amplitude"][0] - 0.6) < 0.15 and fm["r2"] > 0.8   # ~10% obciążenia (drugi rząd)
    Yc, vc = dt.shift_frames(Y, v, fs["delta_rel"], ALPHA)
    fs2 = dt.frame_shifts(Yc, vc, S, ALPHA, ok)
    good2 = fs2["weight"] > np.quantile(fs2["weight"], 0.5)
    assert np.sqrt(np.mean(fs2["delta_rel"][good2] ** 2)) < 0.5 * np.sqrt(np.mean(fs["delta_rel"][good] ** 2))


def _score(sign_pattern, seed=0):
    rng = np.random.default_rng(seed)
    u = dt.u_grid(H, W, ALPHA)
    sc = rng.normal(0, 0.02, (H, W))
    peaks = [0.4 + k * PERIOD for k in range(-4, 5)]
    for k, up in enumerate(peaks):
        s = u - up
        sc += sign_pattern(k) * (s / 1.5) * np.exp(-((s / 2.0) ** 2))
    return sc, np.array(peaks)


def test_flank_signs_constant_vs_alternating():
    sc_c, pk = _score(lambda k: 1.0)
    sc_a, _ = _score(lambda k: (-1.0) ** k)
    _, st_c = dt.flank_signs(sc_c, ALPHA, pk, PERIOD)
    _, st_a = dt.flank_signs(sc_a, ALPHA, pk, PERIOD)
    assert st_c["n_pairs"] >= 5 and st_c["n_same"] == st_c["n_pairs"] and st_c["mean_prod"] > 0.99
    assert st_a["n_same"] == 0 and st_a["mean_prod"] < -0.99


def _plateau_stack(mode, seed=0, n=120, step=0.06):
    rng = np.random.default_rng(seed)
    u = dt.u_grid(H, W, ALPHA)
    ph = 2 * np.pi * np.arange(n) / n
    base = 1 + 0.5 * np.sin(ph)
    e = rng.normal(size=n); e -= e.mean(); e /= np.linalg.norm(e) / np.sqrt(n)
    peaks = np.array([0.4 + k * PERIOD for k in range(-4, 5)])
    which = np.floor((u - peaks[0]) / PERIOD).astype(int)             # numer przedziału
    if mode == "abab":
        amp = step * (which % 2)
    else:
        amp = step * 0.35 * which
    prof = base[None, None, :] + amp[..., None] * e[None, None, :]
    frac = np.clip(_bumps(u), 0, 1)
    Y = 1000 * (prof * (1 - frac[..., None]) + frac[..., None] * (base + 0.3 * np.cos(ph)) +
                rng.normal(0, 0.01, (H, W, n)))
    return Y, np.ones((H, W, n), bool), frac, peaks


def test_abab_regression_detects_parity_not_smooth_trend():
    ok = np.ones((H, W), bool)
    res = {}
    for mode in ("abab", "smooth"):
        Y, v, frac, pk = _plateau_stack(mode)
        pp = dt.plateau_profiles(Y, v, frac, ok, ALPHA, pk, period=PERIOD)
        res[mode] = dt.abab_regression(pp)
    assert res["abab"]["t_gamma"] > 4 and res["abab"]["D1"] > res["abab"]["D2"]
    assert abs(res["smooth"]["t_gamma"]) < 3 and res["smooth"]["beta"] > 0
    assert res["smooth"]["D1"] < res["smooth"]["D2"]
