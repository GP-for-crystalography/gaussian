"""Test całego toru prawdziwych danych na sztucznym stosie z paskami domen (syntetyczny master)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration as cal        # noqa: E402
import orient_fit as of          # noqa: E402
import real_data as rd           # noqa: E402
import spatial_gp as sg          # noqa: E402
from test_orient_fit import mp   # noqa: E402,F401  (fixture: syntetyczny master 4mm)

H = W = 16
THETA_TRUE, SENSE = 40.3, -1


@pytest.fixture(scope="module")
def stack(mp, tmp_path_factory):
    import tifffile
    rng = np.random.default_rng(0)
    V = of.pseudo_variants()
    g_ref = of.random_orientations(1, 42)[0]
    i, j = np.mgrid[0:H, 0:W]
    lab = ((i + j) // 4) % 2                                  # paski ukośne
    vidx = np.where(lab == 0, 0, 2)                           # warianty 0 i 2 (90°)
    model = of.ConeModel(mp, THETA_TRUE)
    G = np.einsum("pij,jk->pik", V[vidx.ravel()], g_ref)
    prof = model.profiles(G, 0.0, SENSE)                      # (P, 180)
    prof = (prof - prof.min()) / np.ptp(prof)
    gain = 50000 * (1 + 0.1 * rng.normal(size=(len(prof), 1)))
    Yp = 5000 + gain * prof
    Yp += rng.normal(size=Yp.shape) * Yp.std(1, keepdims=True) / 15
    Yp = np.clip(Yp, 1, 65535)
    Yp[Yp > np.quantile(Yp, 0.995)] = 65535                   # nasycenie
    Yp[0, 10:14] = 0                                          # zera z rejestracji (narożnik)
    art = 5 * W + 5
    Yp[art] = rng.uniform(5000, 60000, Yp.shape[1])           # artefakt
    raw = np.moveaxis(Yp.reshape(H, W, -1), -1, 0).astype(np.uint16)
    path = tmp_path_factory.mktemp("d") / "stack.tif"
    tifffile.imwrite(path, raw)
    return dict(path=str(path), raw=raw, lab=lab, vidx=vidx, g_ref=g_ref, art=art, model=model)


def test_load_and_valid(stack):
    Y, om, raw = rd.load_stack(stack["path"])
    assert Y.shape == (H, W, 180) and np.array_equal(raw, stack["raw"])
    v = rd.valid_mask(raw)
    assert v.shape == Y.shape
    assert not v[0, 0, 10:14].any()
    assert (~v).sum() >= (stack["raw"] == 65535).sum()


def test_quality_and_clean(stack):
    Y, _, raw = rd.load_stack(stack["path"])
    q = rd.quality_maps(Y, rd.valid_mask(raw))
    clean, thr = rd.clean_pixels(q)
    a = stack["art"]
    assert not clean.ravel()[a]
    assert clean.mean() > 0.9
    assert np.nanmedian(q["snr_neigh"]) > 3


def test_model_free_clusters_find_stripes(stack):
    Y, _, raw = rd.load_stack(stack["path"])
    v = rd.valid_mask(raw)
    clean, _ = rd.clean_pixels(rd.quality_maps(Y, v))
    R, _ = rd.residuals_vs_common(Y, v)
    sc, _, _ = rd.residual_pca(R, clean)
    lab = rd.cluster_map(sc, clean, k=2)
    t = stack["lab"][clean]
    l = lab[clean]
    agree = max(np.mean(l == t), np.mean(l == 1 - t))
    assert agree > 0.95


@pytest.mark.parametrize("method", ["hdbscan", "dbscan", "gmm"])
def test_cluster_auto_finds_number_of_clusters(method):
    rng = np.random.default_rng(0)
    h = w = 40
    scores = rng.normal(size=(h, w, 8)) * 0.15
    lab_true = (np.add.outer(np.arange(h), np.arange(w)) // 20) % 3          # 3 pasy (3 klastry)
    centers = np.array([[0, 0, 0, 0], [3, 0, 0, 0], [0, 3, 0, 0]], float)
    scores[..., :4] += centers[lab_true]
    clean = np.ones((h, w), bool)
    clean[5, 5] = False
    lab, info = rd.cluster_auto(scores, clean, method=method, min_cluster_size=30)
    assert info["k"] == 3
    assert lab[5, 5] == -1
    # zgodność z prawdą (z dokładnością do permutacji etykiet)
    m = (lab >= 0) & clean
    conf = np.array([[np.sum((lab == a) & (lab_true == b) & m) for b in range(3)] for a in range(3)])
    assert conf.max(1).sum() / m.sum() > 0.95


def test_theta_calibration_and_map_fit(stack, tmp_path):
    Y, _, raw = rd.load_stack(stack["path"])
    v = rd.valid_mask(raw)
    clean, _ = rd.clean_pixels(rd.quality_maps(Y, v))
    rng = np.random.default_rng(1)
    pix = rng.choice(np.flatnonzero(clean.ravel()), 10, replace=False)
    Yf, vf = Y.reshape(-1, 180), v.reshape(-1, 180)
    G0 = np.repeat(stack["g_ref"][None], len(pix), 0)
    th, scan = cal.calibrate_theta(stack["model"].mp, Yf[pix], G0, 40.0, sense=SENSE,
                                   mask=vf[pix], coarse_half=0.6, verbose=False)
    assert abs(th - THETA_TRUE) < 0.05

    model = of.ConeModel(stack["model"].mp, th)
    sel = rng.choice(np.flatnonzero(clean.ravel()), 30, replace=False)
    ck = tmp_path / "ck.npz"
    fit = rd.fit_map_variants(model, Y, v, sel, stack["g_ref"], sense=SENSE, checkpoint=str(ck),
                              every=10, verbose=False)
    dec = rd.variant_decision(fit)
    assert np.mean(dec["best"][sel] == stack["vidx"].ravel()[sel]) > 0.95
    rot = rd.rotation_field(fit["G"][sel, dec["best"][sel]],
                            of.pseudo_variants()[dec["best"][sel]] @ stack["g_ref"])
    assert np.nanmax(np.linalg.norm(rot, axis=1)) < 0.3
    fit2 = rd.fit_map_variants(model, Y, v, sel, stack["g_ref"], sense=SENSE, checkpoint=str(ck),
                               verbose=False)                     # wznowienie: nic do liczenia
    assert np.allclose(fit2["R"][sel], fit["R"][sel])


def test_spatial_gp_recovers_stripe_direction():
    rng = np.random.default_rng(3)
    X = sg.grid_coords(30, 30)
    beta = np.deg2rad(135.0)                      # normalna pasków -> paski wzdłuż 45°
    f = np.tanh(2 * np.sin(2 * np.pi * (X @ [np.cos(beta), np.sin(beta)]) / 8.0))
    y = f + 0.1 * rng.normal(size=len(f))
    gp = sg.SpatialGP().fit(X, y, n_sub=500, verbose=False)
    a = gp.params["alpha_deg"]
    assert min(abs(a - 45), 180 - abs(a - 45)) < 10
    assert gp.params["l_perp"] < gp.params["l_par"]
    mu, _ = gp.loo()
    assert np.corrcoef(mu, f)[0, 1] > 0.9


def test_wall_traces_identity():
    tr = rd.wall_traces(np.eye(3))
    d = {t[0]: t for t in tr}
    assert np.isclose(d[(1, 0, 1)][1], 90.0) and np.isclose(d[(0, 1, 1)][1] % 180, 0.0)
    assert np.isclose(d[(1, 1, 0)][2], 90.0)


def test_band_projector_matches_fft():
    x = np.random.default_rng(0).normal(size=180)
    P = of.band_projector(180, [(21, 60)])
    X = np.fft.rfft(x); X[:21] = 0; X[61:] = 0
    assert np.allclose(P @ x, np.fft.irfft(X, 180), atol=1e-10)
    assert of.projector_dof(180, [(21, 60)]) == 80
    assert np.allclose(of.make_projector(180, 3), of.bg_projector(180, 3))


def test_shared_latent_gp_recovers_loadings():
    rng = np.random.default_rng(5)
    X = sg.grid_coords(24, 24)
    beta = np.deg2rad(135.0)
    f = np.tanh(2 * np.sin(2 * np.pi * (X @ [np.cos(beta), np.sin(beta)]) / 8.0))
    a = np.array([1.0, 0.03, -0.01, 0.04])                     # PC1, Δx, Δy, Δz
    noise = np.array([0.05, 0.01, 0.01, 0.01])
    Y = f[:, None] * a[None, :] + rng.normal(size=(len(f), 4)) * noise
    gp = sg.SharedLatentGP().fit(X, Y, noise_var=np.broadcast_to(noise ** 2, Y.shape) * 0,
                                 n_sub=300, n_pred=400, verbose=False)
    r = gp.params["ratio_to_first"]
    assert np.allclose(r[1:], a[1:] / a[0], atol=0.008)
    mu, sd = gp.predict_latent(X)
    assert abs(np.corrcoef(mu, f)[0, 1]) > 0.95
