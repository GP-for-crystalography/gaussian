"""
calibration.py — kalibracja globalnych parametrów geometrii (na razie theta) z danych,
bez etykiet orientacji.

Idea: theta jest wspólne dla całej mapy. Dla każdej wartości theta z siatki dopasowujemy
orientację każdego piksela (6 wariantów, najlepszy wygrywa) i sumujemy log-wiarygodność
najlepszego dopasowania po pikselach:

    S(theta) = sum_p max_v [ -n/2 * log(1 - r_pv(theta)^2) ].

theta_hat = argmax S. Wybór wariantu NIE jest potrzebny do kalibracji (max po wariantach),
więc procedura działa na pomiarze bez etykiet. Startowe orientacje (G0) mogą pochodzić
z pełnego dopasowania przy nominalnym theta albo z metody słownikowej; przy każdym theta
optymalizacja startuje z V G0 dla wszystkich wariantów V (ciepły start).

phi0 nie jest tu potrzebne: przy G0 z pełnego dopasowania błąd phi0 i tak siedzi w phi1.
"""

from __future__ import annotations

import time

import numpy as np

import orient_fit as of


def _pixel_masks(mask, P, n):
    if mask is None:
        return [None] * P, np.full(P, n)
    m = np.asarray(mask, bool)
    if m.ndim == 1:
        return [m] * P, np.full(P, m.sum())
    return [None if mp_.all() else mp_ for mp_ in m], m.sum(1)


def scan_models(models, values, Y, G0, phi0=0.0, sense=+1, variants=None, mask=None,
                n_bg_harm=0, verbose=True, name="param"):
    """Ogólny skan: lista modeli (po jednym na wartość parametru). Y (P, n), G0 (P, 3, 3),
    mask None / (n,) / (P, n) — maski per piksel (np. nasycone klatki).

    Zwraca słownik jak theta_scan, z kluczem "thetas" = values (nazwa historyczna).
    """
    variants = of.pseudo_variants() if variants is None else variants
    Y, G0 = np.asarray(Y, float), np.asarray(G0, float)
    values = np.asarray(values, float)
    P, n = Y.shape
    masks, nk = _pixel_masks(mask, P, n)
    if isinstance(n_bg_harm, (list, tuple)):          # pasmo: n = rząd projektora
        nk = np.array([of.projector_dof(n, n_bg_harm, m) for m in masks])
    T, V = len(values), len(variants)
    R = np.empty((T, P, V))
    G = np.empty((T, P, 3, 3))
    t0 = time.perf_counter()
    for i, m in enumerate(models):
        for p in range(P):
            Gv, R[i, p] = of.variant_scores(m, Y[p], G0[p], variants, phi0, sense, masks[p],
                                            n_bg_harm)
            G[i, p] = Gv[np.argmax(R[i, p])]
        if verbose and (i == 0 or (i + 1) % max(1, T // 5) == 0):
            el = time.perf_counter() - t0
            print(f"    {name} {values[i]:.3f} ({i + 1}/{T}), {el:.0f} s, "
                  f"zostało ~{el / (i + 1) * (T - i - 1):.0f} s", flush=True)
    logL = -0.5 * nk[None, :, None] * np.log(np.clip(1.0 - R ** 2, 1e-300, None))
    return dict(thetas=values, R=R, logL=logL, Lbest=logL.max(2), vbest=logL.argmax(2), G=G)


def theta_scan(mp, Y, G0, thetas, phi0=0.0, sense=+1, invert=False, sigma_conv_deg=0.0,
               variants=None, mask=None, n_bg_harm=0, verbose=True):
    """Skan theta. Y (P, 180), G0 (P, 3, 3); mask None / (n,) / (P, n).

    Zwraca słownik:
      thetas (T,), R (T, P, V) — r każdego wariantu, logL (T, P, V),
      Lbest (T, P) — logL najlepszego wariantu, vbest (T, P) — indeks najlepszego wariantu
      (względem G0: wariant v to V[v] @ G0[p]), G (T, P, 3, 3) — najlepsza orientacja.
    """
    thetas = np.asarray(thetas, float)
    models = [of.ConeModel(mp, th, invert, sigma_conv_deg) for th in thetas]
    return scan_models(models, thetas, Y, G0, phi0, sense, variants, mask, n_bg_harm, verbose,
                       name="theta")


def sigma_scan(mp, Y, G0, theta, sigmas, phi0=0.0, sense=+1, invert=False, variants=None,
               mask=None, n_bg_harm=0, verbose=True):
    """Skan rozmycia wiązki sigma_conv przy ustalonym theta (np. po kalibracji theta)."""
    sigmas = np.asarray(sigmas, float)
    models = [of.ConeModel(mp, theta, invert, s) for s in sigmas]
    return scan_models(models, sigmas, Y, G0, phi0, sense, variants, mask, n_bg_harm, verbose,
                       name="sigma")


def merge_scans(a, b):
    """Łączy dwa skany (np. zgrubny i dokładny) w jeden, posortowany po theta."""
    th = np.concatenate([a["thetas"], b["thetas"]])
    o = np.argsort(th, kind="stable")
    th = th[o]
    keep = np.r_[True, np.diff(th) > 1e-9]          # usuń duplikaty theta
    out = {}
    for k in a:
        v = np.concatenate([a[k], b[k]])[o]
        out[k] = v[keep]
    return out


def peak(x, s, half_window=2):
    """Maksimum funkcji s(x) z dokładnością podsiatkową: parabola na 2*hw+1 punktach wokół
    argmax (działa także dla nierównej siatki). Na brzegu zwraca punkt siatki."""
    x, s = np.asarray(x, float), np.asarray(s, float)
    k = int(np.argmax(s))
    lo, hi = max(0, k - half_window), min(len(x), k + half_window + 1)
    if hi - lo < 3 or k == 0 or k == len(x) - 1:
        return float(x[k])
    a, b, _ = np.polyfit(x[lo:hi] - x[k], s[lo:hi], 2)
    if a >= 0:
        return float(x[k])
    return float(np.clip(x[k] - b / (2 * a), x[lo], x[hi - 1]))


def theta_hat(scan, pixels=None):
    """theta z maksimum S(theta) = sum_p Lbest (opcjonalnie tylko wybrane piksele)."""
    L = scan["Lbest"] if pixels is None else scan["Lbest"][:, pixels]
    return peak(scan["thetas"], L.sum(1))


def calibrate_theta(mp, Y, G0, theta0, coarse_half=1.2, coarse_step=0.1, fine_half=0.15,
                    fine_step=0.025, **kw):
    """Dwustopniowy skan: zgrubny wokół theta0, potem dokładny wokół maksimum zgrubnego."""
    coarse = theta0 + np.arange(-coarse_half, coarse_half + 1e-9, coarse_step)
    s1 = theta_scan(mp, Y, G0, coarse, **kw)
    t1 = theta_hat(s1)
    fine = t1 + np.arange(-fine_half, fine_half + 1e-9, fine_step)
    fine = fine[np.min(np.abs(fine[:, None] - coarse[None, :]), axis=1) > 1e-6]
    s2 = theta_scan(mp, Y, G0, fine, **kw)
    scan = merge_scans(s1, s2)
    return theta_hat(scan), scan
