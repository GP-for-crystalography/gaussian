"""Testy domain_tests.py na danych syntetycznych (syntetyczny master 4mm z test_orient_fit)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import domain_tests as dt        # noqa: E402
import orient_fit as of          # noqa: E402
import real_data as rd           # noqa: E402
from test_orient_fit import mp   # noqa: E402,F401  (fixture)


def test_stripe_fraction_sign_and_range():
    rng = np.random.default_rng(0)
    f = rng.normal(0, 0.02, 4000)
    f[rng.choice(4000, 600, replace=False)] += 0.5                 # mniejszość: paski
    for sgn in (+1, -1):
        fr = dt.stripe_fraction(sgn * f)
        assert fr.min() >= 0 and fr.max() <= 1
        assert fr[f > 0.4].mean() > 0.6 and fr[f < 0.1].mean() < 0.1


def test_evidence_vs_fraction_recovers_line():
    rng = np.random.default_rng(1)
    n = 3000
    u = rng.uniform(0, 60, n)
    frac = np.clip(0.5 + 0.5 * np.sin(2 * np.pi * u / 10) + rng.normal(0, 0.05, n), 0, 1)
    dL = -2.0 + 3.0 * frac + rng.normal(0, 1.0, n)
    ev = dt.evidence_vs_fraction(dL, frac, dt.block_ids(u, 10.0), np.ones(n, bool), n_boot=200)
    assert abs(ev["b0"] + 2.0) < 0.2 and abs(ev["b1"] - 3.0) < 0.3
    assert ev["se_b1"] > 0 and abs(ev["pure"] - 1.0) < 0.3


def test_group_difference_and_axis_test():
    rng = np.random.default_rng(2)
    P = 2000
    u = rng.uniform(0, 80, P)
    stripe = (np.sin(2 * np.pi * u / 10) > 0.6)
    R = rng.normal(0, 0.05, (P, 3))
    R[stripe] += np.array([0.0, 0.0, 0.10])                        # paski: obrót wokół z
    d, cov, na, nb = dt.group_difference(R, stripe, ~stripe, dt.block_ids(u, 10.0), n_boot=300)
    assert na > 100 and nb > 100 and abs(d[2] - 0.10) < 0.02
    axes = np.eye(3)
    t = dt.axis_hypothesis_test(d, cov, axes, ["x", "y", "z"])
    assert t.p[2] > 0.05 and t.p[0] < 0.01 and t.p[1] < 0.01       # z nieodrzucona, x i y odrzucone
    assert t.attrs["p_zero"] < 0.01


def test_cube_axes_sample_identity():
    G, names = dt.cube_axes_sample(np.eye(3))
    assert np.allclose(G, np.eye(3)) and "x_próbki" in names[0] and "z_próbki" in names[2]


def test_differential_fit_rotation_vs_variant(mp):
    model = of.ConeModel(mp, 40.0)
    V = of.pseudo_variants()
    g = of.random_orientations(1, 11)[0]
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(3)
    f = 0.5
    # (1) paski = matryca obrócona o 0.3 deg wokół z próbki, rozcieńczone f = 0.5
    g_rot = g @ Rotation.from_rotvec(np.deg2rad([0, 0, 0.3])).as_matrix()
    mu_m = model.profiles(g[None])[0]
    mu_r = model.profiles(g_rot[None])[0]
    y_b = mu_m + rng.normal(0, 1e-3, 180)
    y_a = (1 - f) * mu_m + f * mu_r + rng.normal(0, 1e-3, 180)
    out = dt.differential_fit(model, g, y_a, y_b, spec=3, variants=V)
    assert out["expl_rot"] > 0.9
    assert abs(out["delta_deg"][2] - f * 0.3) < 0.05 and abs(out["delta_deg"][0]) < 0.05
    # (2) paski = inny wariant (V2), f = 0.5
    mu_v = model.profiles((V[2] @ g)[None])[0]
    y_a2 = (1 - f) * mu_m + f * mu_v + rng.normal(0, 1e-3, 180)
    out2 = dt.differential_fit(model, g, y_a2, y_b, spec=3, variants=V)
    assert out2["best_variant"] == "V2" and out2["expl_variant"] > 0.9
    assert out2["expl_variant"] > out2["expl_rot"] + 0.2


def test_class_geometry_and_eta():
    L = np.zeros((40, 40), int)
    L[:, ::8] = 1                                                   # cienkie paski
    L[:, 1::8] = 1
    L[20:, :] = np.where(L[20:, :] == 1, 1, 2)                      # duży region klasy 2
    g = dt.class_geometry(L)
    assert g.loc[1, "szerokość_typowa_px"] < g.loc[2, "szerokość_typowa_px"]
    f = (L == 2).astype(float)
    assert dt.eta_squared(L.ravel(), f.ravel(), np.ones(L.size, bool)) > 0.3


def test_polarity_pairs_three_pairs():
    g = of.random_orientations(1, 5)[0]
    pairs = dt.polarity_pairs(of.pseudo_variants(), g)
    assert len(pairs) == 3 and sorted(i for p in pairs for i in p) == list(range(6))


def test_pair_detectability_weights():
    n = 180
    rng = np.random.default_rng(4)
    Zs = rng.normal(size=(2, n)); Zs -= Zs.mean(1, keepdims=True)
    Zs /= np.linalg.norm(Zs, axis=1, keepdims=True)
    bands = [(1, 30), (31, 90)]
    t = dt.pair_detectability(Zs, [(0, 1)], [1.0, 0.0], bands)
    assert np.isclose(t.iloc[0, :2].sum(), 1.0, atol=1e-9)
    assert np.isclose(t.wskaźnik_wykrywalności.iloc[0], t.iloc[0, 0], atol=1e-9)


def test_cross_stripe_profiles_geometry():
    H = W = 60
    n = 40
    rng = np.random.default_rng(6)
    pat_a, pat_b = rng.normal(size=n), rng.normal(size=n)
    alpha = 30.0
    a = np.deg2rad(alpha)
    ii, jj = np.mgrid[0:H, 0:W]
    u = -np.sin(a) * (jj - (W - 1) / 2) + np.cos(a) * (ii - (H - 1) / 2)
    stripe = (np.mod(u, 12.0) < 4.0)
    Y = np.where(stripe[..., None], pat_a, pat_b)[:, :, :] + 5.0 + rng.normal(0, 0.05, (H, W, n))
    valid = np.ones_like(Y, bool)
    cen, prof, msk, cnt = dt.cross_stripe_profiles(Y, valid, alpha, half_length=20, du=1.0, min_pixels=5)
    z = lambda x: (x - x.mean()) / np.linalg.norm(x - x.mean())
    proj = np.array([z(p) @ z(pat_a) for p in prof])
    ind = (np.mod(cen, 12.0) < 4.0).astype(float)
    inside = (np.mod(cen, 12.0) > 0.7) & (np.mod(cen, 12.0) < 3.3) | (np.mod(cen, 12.0) > 4.7) & (np.mod(cen, 12.0) < 11.3)
    assert np.corrcoef(proj[inside], ind[inside])[0, 1] > 0.9
    assert msk.all() and cnt.min() >= 5


def test_fit_series_recovers_rotation_step(mp):
    model = of.ConeModel(mp, 40.0)
    V = of.pseudo_variants()
    g = of.random_orientations(1, 8)[0]
    from scipy.spatial.transform import Rotation
    g2 = g @ Rotation.from_rotvec(np.deg2rad([0, 0, 0.4])).as_matrix()
    profs = model.profiles(np.stack([g, g2]))
    profs = (profs - profs.mean(1, keepdims=True)) / profs.std(1, keepdims=True)
    df = dt.fit_series(model, g, profs, np.ones((2, 180), bool), 3, V, 0)
    assert abs(df.dz[0]) < 0.02 and abs(df.dz[1] - 0.4) < 0.05 and (df.best == 0).all()


def test_twin_hypotheses_identity_and_c_over_a():
    tab = dt.twin_hypotheses(np.eye(3), 49.0)
    row = tab[tab.płaszczyzna == "(1, 0, 1)"].iloc[0]
    assert np.allclose(row.oś_w_próbce, [0, 1, 0]) and row.oś_obrotu == "[010]"
    assert np.isclose(row.psi_obraz_deg, (49.0 - row.ślad_w_próbce_deg) % 180)
    ca = 0.40019998 / 0.39930001
    assert abs(dt.rotation_from_c_over_a(ca) - 0.129) < 0.002
    assert abs(dt.c_over_a_from_rotation(dt.rotation_from_c_over_a(ca)) - ca) < 1e-9
    t2 = dt.twin_hypotheses(np.eye(3), 49.0, d=np.array([0, 0.1, 0]), cov=np.eye(3) * 1e-4)
    assert t2[t2.płaszczyzna == "(1, 0, 1)"].p.iloc[0] > 0.5


def test_cross_stripe_skips_invalid_pixels_and_fit_series_nan(mp):
    H = W = 40
    n = 180
    rng = np.random.default_rng(9)
    Y = rng.normal(size=(H, W, n)) + 5.0
    valid = np.ones_like(Y, bool)
    valid[5:15, 5:15] = False                                       # artefakt: brak ważnych klatek
    cen, prof, msk, cnt = dt.cross_stripe_profiles(Y, valid, 40.0, du=1.0, min_pixels=5)
    assert np.isfinite(prof).all() and (msk.sum(1) > 0).all()
    model = of.ConeModel(mp, 40.0)
    g = of.random_orientations(1, 2)[0]
    short = np.zeros((1, n), bool); short[0, :20] = True            # za mało ważnych klatek
    df = dt.fit_series(model, g, np.zeros((1, n)), short, 3, of.pseudo_variants(), 0)
    assert df.best[0] == -1 and np.isnan(df.dz[0])
