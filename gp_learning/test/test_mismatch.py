"""Testy mismatch.py i marginesów wariantów (syntetyczny master 4mm z 4 kanałami)."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import echord_cone as ec          # noqa: E402
import mismatch as mm             # noqa: E402
import orient_fit as of           # noqa: E402
from test_orient_fit import _master_4mm   # noqa: E402


@pytest.fixture(scope="module")
def master4(tmp_path_factory):
    """4 'kanały pozycji': części tej samej funkcji 4mm o różnych wagach (suma = 1)."""
    npx = 200
    t = np.linspace(-1, 1, 2 * npx + 1)
    XY = np.stack(np.meshgrid(t, t, indexing="xy"), axis=-1)
    chans = []
    for north in (True, False):
        d = ec.lambert_to_sphere(XY, north=north)
        f = _master_4mm(d)
        z = d[..., 2]
        parts = [0.4 * f, 0.3 * f + 0.05 * z, 0.2 * f - 0.05 * z, 0.1 * f]
        chans.append(np.stack(parts))
    p = tmp_path_factory.mktemp("m") / "syn4ch.h5"
    with h5py.File(p, "w") as fh:
        g = fh.create_group("EMData/ECPmaster")
        g["mLPNH"], g["mLPSH"] = chans
        fh["CrystalData/SpaceGroupNumber"] = np.array([99])
    return str(p), ec.MasterPattern(str(p))


def test_path_model_matches_cone(master4):
    _, mp = master4
    g = of.random_orientations(10, 1)
    a = of.ConeModel(mp, 40.0).profiles(g, 13.0, -1)
    b = mm.PathModel(mp, 40.0).profiles(g, 13.0, -1)
    assert np.allclose(a, b, rtol=1e-10, atol=1e-10)
    ai = of.ConeModel(mp, 40.0, invert=True).profiles(g, 5.0, 1)
    bi = mm.PathModel(mp, 40.0, invert=True).profiles(g, 5.0, 1)
    assert np.allclose(ai, bi)


def test_wobble_zero_is_cone(master4):
    _, mp = master4
    g = of.random_orientations(5, 2)
    a = of.ConeModel(mp, 40.0).profiles(g)
    b = mm.PathModel(mp, mm.wobble(40.0, 0.0)).profiles(g)
    assert np.allclose(a, b)


def test_reweighted_ones_is_identity(master4):
    path, mp = master4
    d = np.random.default_rng(0).normal(size=(500, 3))
    m1 = mm.reweighted_master(mp, path, [1, 1, 1, 1])
    assert np.allclose(m1.lookup(d), mp.lookup(d))
    m2 = mm.reweighted_master(mp, path, [1, 1.5, 1, 1])
    assert not np.allclose(m2.lookup(d), mp.lookup(d))


def test_variant_scores_pick_truth(master4):
    _, mp = master4
    m = of.ConeModel(mp, 40.0)
    for gt in of.random_orientations(4, 3):
        y = m.profiles(gt[None])[0]
        G, R = of.variant_scores(m, y, gt)
        assert np.argmax(R) == 0 and R[0] > 0.99999
        assert of.misorientation_deg(G[0], gt)[0] < 0.01
        dr, dl = of.margin_stats(R, len(y))
        assert dr > 0 and dl > 0
