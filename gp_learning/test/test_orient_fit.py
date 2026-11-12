"""Testy orient_fit na syntetycznym masterze o symetrii 4mm (C4 + lustra, bez inwersji)."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import echord_cone as ec          # noqa: E402
import orient_fit as of           # noqa: E402

THETA, DPSI = 40.0, 0.25


def _master_4mm(d):
    """Pasma Kikuchiego (|n.d| małe) dla normalnych zamkniętych na 4mm + człon polarny."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    normals, w = [], []
    for n, wt in [((1, 0, 0), 1.0), ((1, 1, 0), 0.7), ((1, 0, 1), 0.6), ((1, 1, 1), 0.5),
                  ((2, 0, 1), 0.4), ((0, 0, 1), 0.8)]:
        n = np.array(n, float)
        for k in range(4):                       # C4
            a = k * np.pi / 2
            R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            for m in (np.eye(3), np.diag([-1.0, 1, 1])):   # + lustro
                normals.append(R @ m @ n / np.linalg.norm(n)); w.append(wt)
    f = np.zeros(d.shape[:-1])
    for n, wt in zip(normals, w):
        f += wt * np.exp(-0.5 * ((d @ n) / 0.025) ** 2)
    return f + 0.6 * z + 0.4 * z**3 + 0.3 * z * (x**2 - y**2) ** 2   # polarne: brak inwersji


@pytest.fixture(scope="module")
def mp(tmp_path_factory):
    npx = 250
    t = np.linspace(-1, 1, 2 * npx + 1)
    XY = np.stack(np.meshgrid(t, t, indexing="xy"), axis=-1)       # [row=Y, col=X]
    NH = _master_4mm(ec.lambert_to_sphere(XY, north=True))
    SH = _master_4mm(ec.lambert_to_sphere(XY, north=False))
    p = tmp_path_factory.mktemp("m") / "synthetic_4mm.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("EMData/EBSDmaster")
        g["mLPNH"], g["mLPSH"] = NH, SH
        f["CrystalData/SpaceGroupNumber"] = np.array([99])
    return ec.MasterPattern(str(p))


@pytest.fixture(scope="module")
def model(mp):
    return of.ConeModel(mp, THETA, dpsi=DPSI)


@pytest.fixture(scope="module")
def bank(model):
    return of.build_bank(model, spacing_deg=1.5, verbose=False)


def test_euler_roundtrip():
    g = of.random_orientations(200, seed=3)
    e = of.euler_from_matrix(g)
    assert np.allclose(ec.bunge_to_matrix(e), g, atol=1e-10)
    g0 = ec.bunge_to_matrix([[30.0, 0.0, 20.0]])                       # osobliwość Phi = 0
    assert np.allclose(ec.bunge_to_matrix(of.euler_from_matrix(g0)), g0, atol=1e-10)


def test_misorientation():
    g = of.random_orientations(50, seed=4)
    S = of.c4_ops()
    assert np.allclose(of.misorientation_deg(S[1] @ g, g), 0, atol=1e-6)
    R = ec.bunge_to_matrix([[0.0, 3.0, 0.0]])[0]
    assert np.allclose(of.misorientation_deg(g @ R, g), 3.0, atol=1e-6)


def test_symmetry_of_synthetic_master(mp):
    c = of.mirror_symmetry_check(mp)
    assert c["mirror_x"] > 0.999 and c["mirror_diag"] > 0.999 and c["inversion"] < 0.99


def test_self_check(model):
    r, err = of.self_check(model)
    assert r > 0.99999 and err < 1e-3


def test_shift_correlation_vs_bruteforce(model, bank):
    rng = np.random.default_rng(0)
    y = model.profiles(of.random_orientations(1, 5), 6.0, -1)[0] + rng.normal(0, 0.05, 180)
    mask = rng.random(180) > 0.1
    rows = slice(100, 110)
    r_fft = of.shift_correlation(bank, y, model.omega, 6.0, -1, mask, rows=rows)
    M = bank.M
    pos = of.comb_positions(model.omega, 6.0, -1, DPSI, M)
    for i, Fi in enumerate(bank.F[rows]):
        for j in (0, 1, 7, 777, M - 1):
            r_bf = of.pearson(y, Fi[(pos - j) % M], mask)
            assert abs(r_fft[i, j] - r_bf) < 1e-5


@pytest.mark.parametrize("phi0,sense", [(0.0, +1), (23.0, -1)])
def test_recovery_noise_free(model, bank, phi0, sense):
    g_true = of.random_orientations(8, seed=11)
    Y = model.profiles(g_true, phi0, sense) * 3.0 + 50.0                # gain/offset
    for gt, y in zip(g_true, Y):
        res = of.fit_profile(model, bank, y, phi0, sense, top_k=3)
        assert of.misorientation_deg(res["g"], gt)[0] < 0.05
        assert res["r"] > 0.99999


def test_recovery_noise_and_background(model, bank):
    rng = np.random.default_rng(2)
    g_true = of.random_orientations(6, seed=12)
    Y = model.profiles(g_true, 10.0, +1)
    bg = 0.5 * Y.std() * np.cos(np.deg2rad(model.omega) + 1.0)
    bank2 = of.build_bank(model, spacing_deg=1.5, n_bg_harm=2, verbose=False)
    for gt, y in zip(g_true, Y):
        y = y + bg + rng.normal(0, 0.1 * y.std(), y.size)
        res = of.fit_profile(model, bank2, y, 10.0, +1, top_k=3)
        assert of.misorientation_deg(res["g"], gt)[0] < 0.5


def test_sense_flip_is_exact_ambiguity(model):
    """Profil (g, sense) == profil (sigma_c g sigma_s, -sense, -phi0): sense niewyznaczalne."""
    g = of.random_orientations(10, seed=21)
    a = model.profiles(g, 14.0, +1)
    b = model.profiles(of.sense_flip_equivalent(g), -14.0, -1)
    assert np.all(of.pearson(a, b) > 0.9999)


def test_invert_is_exact_ambiguity(mp):
    """invert=True z g == invert=False z (-sigma_c) g."""
    g = of.random_orientations(10, seed=22)
    m_inv = of.ConeModel(mp, THETA, invert=True)
    m = of.ConeModel(mp, THETA, invert=False)
    sig = np.diag([-1.0, 1.0, 1.0])
    a = m_inv.profiles(g, 5.0, +1)
    b = m.profiles(-sig @ g, 5.0, +1)
    assert np.all(of.pearson(a, b) > 0.9999)
