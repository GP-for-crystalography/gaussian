"""
Wspólne fixture'y dla testów echord_cone.

pytest automatycznie wczytuje ten plik przed testami z tego samego folderu.
Fixture to funkcja, której wynik pytest wstrzykuje do testu po nazwie argumentu;
scope="session" oznacza, że liczy się raz na całe uruchomienie pytest.
"""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import echord_cone as ec

NPX = 300
EKEVS = np.array([10.0, 15.0, 20.0])
WIDTHS = [0.06, 0.05, 0.04]  # każdy kanał energii trochę inny


def make_band_function(seed=0, n_bands=12):
    """Syntetyczne "pasma Kikuchiego": f(d) = sum_k exp(-(d.n_k)^2 / 2w^2).

    Normalne są losowe, więc funkcja NIE ma symetrii kwadratu Lamberta
    (test orientacji tablicy coś rozróżnia), ale f(-d) = f(d).
    """
    rng = np.random.default_rng(seed)
    normals = rng.normal(size=(n_bands, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    def f(d, width):
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        return np.exp(-((d @ normals.T) ** 2) / (2 * width**2)).sum(-1)

    return f


def write_synthetic_master(path, f, npx=NPX, space_group=99, sites=1):
    """Plik .h5 o układzie jak z EMsoft (kształty jak po odczycie przez h5py)."""
    g = np.arange(-npx, npx + 1) / npx
    XX, YY = np.meshgrid(g, g)  # wiersz = Y (indeks j), kolumna = X (indeks i)
    XY = np.stack([XX, YY], -1)
    dN = ec.lambert_to_sphere(XY, True)
    dS = ec.lambert_to_sphere(XY, False)
    dSP = ec.stereo_to_sphere(XY)
    mN = np.stack([f(dN, w) for w in WIDTHS])
    mS = np.stack([f(dS, w) for w in WIDTHS])
    # kilka pozycji atomów: rozbijamy sygnał na równe części, suma = oryginał
    mN = np.repeat(mN[None] / sites, sites, axis=0)  # (sites, nE, n, n)
    mS = np.repeat(mS[None] / sites, sites, axis=0)
    sp = np.stack([np.where(XX**2 + YY**2 <= 1, f(dSP, w), 0) for w in WIDTHS])
    accum_e = np.random.default_rng(1).random((11, 11, EKEVS.size))
    with h5py.File(path, "w") as h:
        grp = h.create_group("EMData/EBSDmaster")
        grp["mLPNH"], grp["mLPSH"] = mN, mS
        grp["EkeVs"], grp["masterSPNH"] = EKEVS, sp
        h["EMData/MCOpenCL/accum_e"] = accum_e
        h["CrystalData/SpaceGroupNumber"] = np.array([space_group])
        h["CrystalData/LatticeParameters"] = np.array([0.3995, 0.3995, 0.4022, 90, 90, 90])
    return accum_e


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """Syntetyczny master pattern + funkcja analityczna, z której powstał."""
    f = make_band_function()
    path = tmp_path_factory.mktemp("master") / "synthetic_master.h5"
    accum_e = write_synthetic_master(path, f)
    return SimpleNamespace(path=str(path), f=f, widths=WIDTHS, ekevs=EKEVS,
                           eweights=accum_e.sum(axis=(0, 1)))


@pytest.fixture(scope="session")
def mp(synthetic):
    """MasterPattern wczytany z pliku syntetycznego (kanał najwyższej energii)."""
    return ec.MasterPattern(synthetic.path, energy="max")


@pytest.fixture
def make_master(tmp_path):
    """Fabryka: zapisuje wariant pliku (inna grupa przestrzenna, kilka pozycji
    atomów, mniejsza siatka) i zwraca ścieżkę. Użycie: make_master(sites=3)."""
    counter = iter(range(1000))

    def _make(npx=60, space_group=99, sites=1):
        path = tmp_path / f"variant_{next(counter)}.h5"
        write_synthetic_master(path, make_band_function(), npx=npx,
                               space_group=space_group, sites=sites)
        return str(path)

    return _make


@pytest.fixture
def rng():
    """Świeży generator liczb losowych o stałym ziarnie dla każdego testu."""
    return np.random.default_rng(12345)
