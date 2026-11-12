"""Testy wersji GPyTorch: ma dawać DOKŁADNIE to samo co gp_echord (numpy)."""
import os
import sys
import warnings

import numpy as np
import pytest

gpytorch = pytest.importorskip("gpytorch")
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
warnings.filterwarnings("ignore", category=FutureWarning)

import echord_cone as ec  # noqa: E402
import gp_echord as G  # noqa: E402
import gp_echord_gpytorch as T  # noqa: E402
from synthetic_master import write_synthetic_master  # noqa: E402

OPS = G.proper_rotations("C4")
HYPER = dict(ell=np.deg2rad(1.7), kappa=0.7, s2=0.9, sn2=3e-3)


@pytest.fixture(scope="module")
def mp(tmp_path_factory):
    path = tmp_path_factory.mktemp("synth") / "synth_master.h5"
    write_synthetic_master(str(path), npx=150, space_group=99, width=0.006)
    return ec.MasterPattern(str(path))


@pytest.fixture(scope="module")
def data(mp):
    rng = np.random.default_rng(0)
    ang = np.deg2rad(np.column_stack([rng.uniform(30, 40, 300), rng.uniform(10, 20, 300)]))
    return G.build_training_data(mp, angles=ang, coneAngle=40, dpsi=0.5, verbose=False)


def _torch_gp_with(hyper, X, Yn):
    gp = T.TorchEchordGP(OPS)
    gp._build(T._t(X.reshape(-1, 9)), T._t(Yn).T.contiguous())
    gp.set_hyperparameters(**hyper)
    return gp


def test_kernel_matches_numpy():
    X = ec.quat_to_matrix(G.random_quaternions(150, 1))
    gp = _torch_gp_with(HYPER, X, np.zeros((150, 1)))
    with torch.no_grad():
        Kt = gp.model.covar_module(T._t(X.reshape(-1, 9))).to_dense().numpy()
    Kn = HYPER["s2"] * G.SymMaternKernel(OPS, 1.5).gram(X, X, HYPER["ell"], HYPER["kappa"])
    assert np.abs(Kt - Kn).max() < 1e-12


def test_likelihood_and_gradient_match_numpy(data):
    Yn = (data.Y[:, ::8].astype(float) - data.Y.mean()) / data.Y.std()
    th = np.log([HYPER["ell"], HYPER["kappa"], HYPER["s2"], HYPER["sn2"]])
    f_np, g_np = G.EchordGP(OPS).neg_log_marginal_likelihood(th, data.X, Yn)

    gp = _torch_gp_with(HYPER, data.X, Yn)
    gp.model.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp.model)
    Xt, Yt = gp.model.train_inputs[0], gp.model.train_targets

    def loss_at(t):
        gp.set_hyperparameters(ell=np.exp(t[0]), kappa=np.exp(t[1]), s2=np.exp(t[2]),
                               sn2=np.exp(t[3]))
        with gpytorch.settings.max_cholesky_size(T.MAX_CHOL), torch.no_grad():
            return float(-mll(gp.model(Xt), Yt).mean())

    assert np.isclose(loss_at(th), f_np, rtol=0, atol=1e-10)
    h = 1e-5
    g_t = [(loss_at(th + h * e) - loss_at(th - h * e)) / (2 * h) for e in np.eye(4)]
    assert np.allclose(g_t, g_np, atol=1e-7)


def test_batched_targets_one_factorisation(data):
    """ExactGP z celami (M, N) = suma po kolumnach jednego GP (sprawdza założenie z docstringu)."""
    Yn = (data.Y[:, ::8].astype(float) - data.Y.mean()) / data.Y.std()
    gp = _torch_gp_with(HYPER, data.X, Yn)
    gp.model.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp.model)
    Xt, Yt = gp.model.train_inputs[0], gp.model.train_targets
    with gpytorch.settings.max_cholesky_size(T.MAX_CHOL), torch.no_grad():
        batched = mll(gp.model(Xt), Yt)
        single = torch.stack([mll(gp.model(Xt), Yt[j]) for j in range(3)])
    assert batched.shape == (Yt.shape[0],)
    assert torch.allclose(batched[:3], single, atol=1e-10)


def test_fit_converges_like_numpy(mp):
    """Ta sama optymalizacja ML-II, dwa narzędzia -> te same hiperparametry.

    Plan losowy na SO(3), jak w prawdziwym użyciu. (W gęstej, małej łacie optimum szumu
    leży na dolnej granicy, K jest bardzo źle uwarunkowana i oba optymalizatory stają
    w nieco innych miejscach płaskiego minimum — to nie jest dobry test zgodności.)
    """
    ang, _, _ = G.canonical_angles(G.random_quaternions(500, 3))
    d = G.build_training_data(mp, angles=ang, coneAngle=40, dpsi=0.5, verbose=False)
    gpt = T.TorchEchordGP(OPS).fit(d, n_fit=500, psi_stride=4, n_iter=100, verbose=False)
    gpn = G.EchordGP(OPS).fit(d, n_fit=500, psi_stride=4, verbose=False)
    ht = gpt.hyperparameters()
    assert np.isclose(ht["ell"], gpn.ell, rtol=2e-3)
    assert np.isclose(ht["kappa"], gpn.kappa, rtol=5e-3)
    assert np.isclose(ht["s2"], gpn.signal_var, rtol=5e-3)


def test_predictions_match_numpy(mp, data):
    gpt = _torch_gp_with(HYPER, data.X, np.zeros((len(data), 1)))
    gpt.y_mean, gpt.y_std, gpt.dpsi = float(data.Y.mean()), float(data.Y.std()), data.dpsi
    gpt.meta = dict(data.meta)
    gpt.condition(data, verbose=False)
    gpn = gpt.to_numpy_gp(data)
    Xs = G.canonical_matrices(np.deg2rad([[35.2, 14.1], [31.0, 19.5], [38.7, 11.2]]))
    mt, st = gpt.predict_canonical(Xs)
    mn, sn = gpn.predict_canonical(Xs)
    assert np.allclose(mt, mn, atol=1e-8) and np.allclose(st, sn, atol=1e-8)
    q = ec.bunge_to_quat([[17.0, 35.2, 14.1]])
    _, Pt, _ = gpt.predict_profiles(q, phi0=20.0, sense=-1)
    _, Pn, _ = gpn.predict_profiles(q, phi0=20.0, sense=-1)
    assert Pt.shape == (1, 180) and np.allclose(Pt, Pn, atol=1e-8)


def test_save_load(tmp_path, data):
    gp = T.TorchEchordGP(OPS).fit(data, n_fit=200, psi_stride=8, n_iter=5, verbose=False)
    gp.save(tmp_path / "gp.pt")
    gp2 = T.TorchEchordGP.load(tmp_path / "gp.pt", verbose=False)
    m1, s1 = gp.predict_canonical(data.X[:5])
    m2, s2 = gp2.predict_canonical(data.X[:5])
    assert np.allclose(m1, m2) and np.allclose(s1, s2)
