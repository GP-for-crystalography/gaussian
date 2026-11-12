"""
synthetic_master.py — sztuczny master pattern w formacie EMsoftu (do testów bez EMsoftu).

Intensywność na sferze = tło + suma "pasm Kikuchiego" wzdłuż wielkich kół
prostopadłych do wektorów sieci odwrotnej g = (h, k, l/c) sieci tetragonalnej.
Szerokość pasma ~ |g| (jak kąt Bragga), więc profile po stożku są "szpilkowe".
Amplitudy pasm g i -g są różne (zależą od znaku l), więc M(-d) != M(d):
wzór jest NIECENTROSYMETRYCZNY, ale ma dokładną symetrię C4 wokół osi z
(jak grupa punktowa 4mm, np. P4mm = grupa przestrzenna nr 99).
"""
import itertools

import h5py
import numpy as np

import echord_cone as ec


def synthetic_intensity(d, c_over_a=1.15, hkl_max=3, width=0.0035, chunk=200_000):
    d = np.asarray(d, dtype=float).reshape(-1, 3)
    out = 2.0 + 0.3 * d[:, 2]
    gs, amps, widths = [], [], []
    for h, k, l in itertools.product(range(-hkl_max, hkl_max + 1), repeat=3):
        if h == k == l == 0:
            continue
        g = np.array([h, k, l / c_over_a])
        gn = np.linalg.norm(g)
        gs.append(g / gn)
        amps.append(1.2 / gn**1.5 * (1.0 + 0.35 * np.sign(l)))  # g i -g różne -> brak środka symetrii
        widths.append(width * gn)
    gs, amps, widths = np.array(gs), np.array(amps), np.array(widths)
    for s in range(0, d.shape[0], chunk):
        x = d[s:s + chunk] @ gs.T / widths  # odległość od płaszczyzny w jednostkach szerokości
        band = np.exp(-0.5 * x**2) * (1.0 + 0.5 * np.tanh(x))
        out[s:s + chunk] += band @ amps
    return out


def write_synthetic_master(path, npx=200, space_group=99, **kw):
    """Zapisuje plik .h5 czytelny dla echord_cone.MasterPattern."""
    i = np.arange(-npx, npx + 1) / npx
    XX, YY = np.meshgrid(i, i)  # XX zmienia się wzdłuż kolumn -> img[row=Y, col=X]
    XY = np.stack([XX, YY], axis=-1)
    imgs = []
    for north in (True, False):
        d = ec.lambert_to_sphere(XY, north=north).reshape(-1, 3)
        imgs.append(synthetic_intensity(d, **kw).reshape(XX.shape))
    with h5py.File(path, "w") as f:
        g = f.create_group("EMData/ECPmaster")
        g["mLPNH"] = imgs[0][None].astype(np.float32)
        g["mLPSH"] = imgs[1][None].astype(np.float32)
        g["EkeVs"] = np.array([20.0])
        c = f.create_group("CrystalData")
        c["SpaceGroupNumber"] = np.array([space_group], dtype=np.int32)
        c["LatticeParameters"] = np.array([0.4, 0.4, 0.46, 90.0, 90.0, 90.0])
    return path
