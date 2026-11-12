"""
mismatch.py — generatory danych symulowanych z NIEDOPASOWANIEM modelu (bez nowych masterów).

Dopasowanie zawsze tym samym modelem (ConeModel: theta = 40°, sigma_conv = 0, pełny master),
a dane generowane "prawdą", która się od niego różni:

  * inne theta (stałe)                       -> ConeModel z innym coneAngle
  * rozmycie wiązki / zbieżność              -> ConeModel z sigma_conv_deg > 0
  * bicie stolika: theta(omega) = theta0 + eps*cos(omega + alpha)   -> PathModel
  * nieliniowość detektora: I -> I^gamma     -> nonlinear()
  * inne wagi pozycji atomów w masterze      -> reweighted_master()  (zastępstwo dla błędów
    czynników struktury / Debye'a-Wallera / składu Ta/Nb — przybliżenie, nie fizyka dynamiczna)

Czego NIE da się tak zasymulować (wymaga nowego mastera z EMsoftu): inna energia wiązki,
inne c/a i przesunięcia jonów (polaryzacja), inne DW policzone poprawnie, absorpcja, głębokość.
Gotowy inny master można podać jako "prawdę" bezpośrednio (ConeModel z innym MasterPattern).

Uwaga o rzeczach, które NIE są niedopasowaniem:
  * błąd phi0 to dokładnie przesunięcie phi1 (pochłaniane w każdym pikselu),
  * stałe przechylenie osi obrotu względem normalnej próbki to stały obrót g -> g M
    (wspólny dla całej mapy; przesuwa orientacje względem metody słownikowej, ale nie
    zmienia kształtu profili ani wyboru wariantu).
"""

from __future__ import annotations

import copy

import h5py
import numpy as np


class PathModel:
    """Prawda z dowolną ścieżką: theta może zależeć od omega (bicie stolika).

    theta: liczba [deg] albo funkcja omega[deg] -> theta[deg]. Bez splotu wiązki.
    """

    name = "path"

    def __init__(self, mp, theta, invert=False, omega_step=2.0):
        self.mp, self.invert = mp, invert
        self.theta = theta if callable(theta) else (lambda om, t=float(theta): np.full_like(om, t))
        self.omega = np.arange(int(round(360.0 / omega_step))) * omega_step

    def directions(self, phi0=0.0, sense=+1):
        th = np.deg2rad(self.theta(self.omega))
        b = np.deg2rad(phi0 + sense * self.omega)       # jak echord_cone.cone_directions
        d = np.stack([np.sin(th) * np.cos(b), np.sin(th) * np.sin(b), np.cos(th)], axis=-1)
        return -d if self.invert else d

    def profiles(self, g, phi0=0.0, sense=+1):
        d = self.directions(phi0, sense)
        dc = np.einsum("nij,mj->nmi", np.asarray(g).reshape(-1, 3, 3), d)
        return self.mp.lookup(dc)


def wobble(theta0, eps, alpha=0.0):
    """theta(omega) = theta0 + eps*cos(omega + alpha) [deg]."""
    return lambda om: theta0 + eps * np.cos(np.deg2rad(om + alpha))


def nonlinear(Y, gamma):
    """Nieliniowa odpowiedź detektora: I -> I_max * (I / I_max)^gamma (I_max globalne)."""
    Y = np.clip(np.asarray(Y, dtype=float), 0.0, None)
    m = Y.max()
    return m * (Y / m) ** gamma


def master_channels(mp, path):
    """Surowe kanały mastera (n_ch, ny, nx) dla NH i SH — u Ciebie 4 pozycje atomów."""
    with h5py.File(path, "r") as f:
        NH, SH = f[mp.nh_path][()], f[mp.sh_path][()]
    if NH.ndim == 2:
        return NH[None].astype(float), SH[None].astype(float)
    if mp.ekevs is not None and mp.ekevs.size > 1:
        raise ValueError("master z osią energii — reweighted_master obsługuje tylko kanały pozycji")
    return (NH.reshape(-1, *NH.shape[-2:]).astype(float),
            SH.reshape(-1, *SH.shape[-2:]).astype(float))


def reweighted_master(mp, path, weights):
    """Kopia MasterPattern z sumą kanałów ważoną 'weights' (dla wag = 1 identyczna z mp)."""
    NHc, SHc = master_channels(mp, path)
    w = np.asarray(weights, dtype=float)
    if w.size != NHc.shape[0]:
        raise ValueError(f"master ma {NHc.shape[0]} kanałów, podano {w.size} wag")
    m2 = copy.copy(mp)
    m2._raw = (np.tensordot(w, NHc, axes=1), np.tensordot(w, SHc, axes=1))
    m2._set_orient(*mp._orient)
    return m2


def channel_contributions(mp, path, n=20000, seed=0):
    """Udział i podobieństwo kanałów: średnia, std i korelacja każdego kanału z sumą."""
    NHc, _ = master_channels(mp, path)
    tot = NHc.sum(0)
    rng = np.random.default_rng(seed)
    iy, ix = rng.integers(0, tot.shape[0], n), rng.integers(0, tot.shape[1], n)
    out = []
    for k, c in enumerate(NHc):
        a, b = c[iy, ix], tot[iy, ix]
        out.append(dict(kanal=k, srednia=a.mean(), std=a.std(),
                        udzial_sredniej=a.mean() / b.mean(), r_z_suma=np.corrcoef(a, b)[0, 1]))
    return out
