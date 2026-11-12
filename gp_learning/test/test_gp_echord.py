"""Testy gp_echord na syntetycznym masterze (bez EMsoftu, nic nie uruchamia w tle)."""
import os
import sys
import warnings

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import echord_cone as ec  # noqa: E402
import gp_echord as G  # noqa: E402
from synthetic_master import write_synthetic_master  # noqa: E402


@pytest.fixture(scope="module")
def mp(tmp_path_factory):
    path = tmp_path_factory.mktemp("synth") / "synth_master.h5"
    write_synthetic_master(str(path), npx=150, space_group=99, width=0.006)
    return ec.MasterPattern(str(path))


@pytest.fixture(scope="module")
def patch_model(mp):
    """Mały model na gęstym (~0.5°) planie w łacie (Phi, phi2) — szybki test całości."""
    rng = np.random.default_rng(0)
    ang = np.deg2rad(np.column_stack([rng.uniform(30, 40, 400), rng.uniform(10, 20, 400)]))
    data = G.build_training_data(mp, angles=ang, coneAngle=40, dpsi=0.5, verbose=False)
    gp = G.EchordGP(G.proper_rotations("C4")).fit(data, n_fit=400, maxiter=30, verbose=False)
    return gp, data


# ------------------------------------------------------------------ orientacje
def test_bunge_roundtrip():
    rng = np.random.default_rng(1)
    e = np.column_stack([rng.uniform(0, 2 * np.pi, 500), rng.uniform(0, np.pi, 500),
                         rng.uniform(0, 2 * np.pi, 500)])
    e[:5, 1], e[5:10, 1] = 0.0, np.pi  # przypadki zdegenerowane
    g = ec.bunge_to_matrix(e, degrees=False)
    g2 = ec.bunge_to_matrix(G.matrix_to_bunge(g), degrees=False)
    assert np.abs(g - g2).max() < 1e-12


def test_random_quaternions_unit_and_hemisphere():
    q = G.random_quaternions(1000, 3)
    assert np.allclose(np.linalg.norm(q, axis=1), 1) and np.all(q[:, 0] >= 0)


def test_canonical_angles_dedupe():
    e = np.array([[10, 30, 40], [55, 30, 40], [10, 31, 40]], float)  # 1. i 2. różnią się tylko phi1
    ang, phi1, keep = G.canonical_angles(e, rep="euler")
    assert len(ang) == 2 and list(keep) == [0, 2]
    assert np.allclose(np.rad2deg(phi1), [10, 55, 10])


# ------------------------------------------------------------------- symetria
@pytest.mark.parametrize("name,size", [("C1", 1), ("C2z", 2), ("C3", 3), ("C4", 4), ("C6", 6),
                                       ("D2", 4), ("D2diag", 4), ("D3x", 6), ("D3y", 6),
                                       ("D4", 8), ("D6", 12), ("T", 12), ("O", 24)])
def test_groups(name, size):
    ops = G.proper_rotations(name)
    assert len(ops) == size
    assert np.allclose(np.linalg.det(ops), 1) and np.allclose(ops[0], np.eye(3))
    for A in ops:  # domknięcie
        for B in ops:
            assert any(np.allclose(A @ B, C, atol=1e-9) for C in ops)


def test_space_group_table():
    assert G.proper_group_name(99) == "C4"     # P4mm
    assert G.proper_group_name(38) == "C2z"    # Amm2
    assert G.proper_group_name(221) == "O"     # Pm-3m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert G.proper_group_name(160) == "C3"  # R3m (ustawienie heksagonalne)


def test_symmetry_check(mp):
    assert G.symmetry_check(mp, G.proper_rotations("C4")).min() > 0.999
    assert G.symmetry_check(mp, G.proper_rotations("D4")).min() < 0.99  # złą grupę widać


# ---------------------------------------------------------------------- jądro
@pytest.mark.parametrize("kappa", [0.5, 1.0, 1.3])
def test_kernel_psd_and_invariant(kappa):
    ops = G.proper_rotations("C4")
    k = G.SymMaternKernel(ops, 1.5)
    X = ec.quat_to_matrix(G.random_quaternions(200, 2))
    K = k.gram(X, X, np.deg2rad(5), kappa)
    assert np.abs(K - K.T).max() < 1e-12
    assert np.linalg.eigvalsh(K).min() > -1e-10
    assert np.abs(K - k.gram(ops[1] @ X, X, np.deg2rad(5), kappa)).max() < 1e-12


def test_kernel_distance_geometry():
    """Pochylenie o b -> delta = b; obrót wokół normalnej o a -> delta = kappa * a."""
    k = G.SymMaternKernel([np.eye(3)], 1.5)
    g = ec.quat_to_matrix(G.random_quaternions(1, 4))
    for axis, factor in (([1, 0, 0], 1.0), ([0, 1, 0], 1.0), ([0, 0, 1], 0.7)):
        g2 = g @ G._axis_angle(axis, 0.5)  # obrót w układzie PRÓBKI
        w2f, _, c0, _ = k._weights(0.7)
        delta = np.sqrt(c0 - (g2.reshape(1, 9) * w2f) @ g.reshape(1, 9).T)[0, 0]
        assert np.isclose(np.rad2deg(delta), 0.5 * factor, rtol=1e-4)
    w, _ = k.min_misorientation(g @ G._axis_angle([1, 2, 3], 0.7), g)
    assert np.isclose(w[0], 0.7, rtol=1e-6)


@pytest.mark.parametrize("nu", [1.5, 2.5, np.inf])
def test_nll_gradient(nu):
    rng = np.random.default_rng(0)
    X = G.canonical_matrices(rng.uniform(0, 0.3, (50, 2)))
    Y = rng.normal(size=(50, 5))
    gp = G.EchordGP(G.proper_rotations("C4"), nu=nu)
    th = np.log([np.deg2rad(3), 0.8, 0.8, 0.05])
    _, g = gp.neg_log_marginal_likelihood(th, X, Y)
    h = 1e-6
    num = [(gp.neg_log_marginal_likelihood(th + h * e, X, Y)[0]
            - gp.neg_log_marginal_likelihood(th - h * e, X, Y)[0]) / (2 * h) for e in np.eye(4)]
    assert np.allclose(g, num, atol=1e-6)


# --------------------------------------------------------- wzór (1): przesunięcie
@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("sense", [+1, -1])
def test_shift_exact(mp, invert, sense):
    """phi1 i phi0 na siatce psi -> brak interpolacji -> zgodność do precyzji float32."""
    dpsi = 0.25
    e = np.array([[12.5, 33.0, 71.0], [300.25, 2.0, 5.0], [45.0, 150.0, 200.0]])
    q = ec.bunge_to_quat(e)
    ang, phi1, _ = G.canonical_angles(q, dedupe_tol_deg=None)
    F = G.simulate_canonical(mp, ang, 40, dpsi, invert=invert).astype(float)
    om, direct = mp.cone_profiles(q, 40, 2.0, phi0=17.75, sense=sense, invert=invert)
    shifted = G.profiles_from_canonical(F, dpsi, phi1, om, 17.75, sense)
    assert np.abs(shifted - direct).max() < 1e-5 * np.abs(direct).max()


def test_shift_random(mp):
    r, err = G.check_shift_bookkeeping(mp, 40, 0.1, n=20)
    assert r > 0.999 and err < 0.03


# --------------------------------------------------------------------- GP
def test_gp_beats_nearest_neighbour(mp, patch_model):
    gp, _ = patch_model
    rng = np.random.default_rng(5)
    ang = np.deg2rad(np.column_stack([rng.uniform(31, 39, 60), rng.uniform(11, 19, 60)]))
    res = G.evaluate(gp, mp, angles=ang, verbose=False)
    assert np.median(res["r_gp"]) > 0.9
    assert np.median(res["r_gp"]) > np.median(res["r_nn"])
    assert 0.5 < res["z_sd"] < 2.0


def test_predict_profiles_geometry(mp, patch_model):
    """Predykcja dla pełnej orientacji (phi1, phi0, sense dowolne) = symulacja bezpośrednia."""
    gp, data = patch_model
    Phi, phi2 = np.rad2deg(data.angles[:5]).T
    q = ec.bunge_to_quat(np.column_stack([[17, 250, 3, 90, 181], Phi, phi2]))
    om, P, std = gp.predict_profiles(q, phi0=33.0, sense=-1)
    _, direct = mp.cone_profiles(q, 40, 2.0, phi0=33.0, sense=-1)
    assert G.rowwise_pearson(P, direct).min() > 0.97
    assert P.shape == (5, 180) and std.shape == (5,)


def test_save_load(tmp_path, patch_model):
    gp, data = patch_model
    path = tmp_path / "gp.npz"
    gp.save(path)
    gp2 = G.EchordGP.load(path, verbose=False)
    m1, s1 = gp.predict_canonical(data.X[:10])
    m2, s2 = gp2.predict_canonical(data.X[:10])
    assert np.allclose(m1, m2) and np.allclose(s1, s2)
    assert gp2.meta["coneAngle"] == 40
