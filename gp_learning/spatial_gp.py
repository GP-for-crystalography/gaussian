"""
spatial_gp.py — GP na mapie pikseli (2D) do analizy przestrzennej wyników.

Jądro anizotropowe (Matérn 3/2 albo RBF) z obrotem osi:
    d^2 = (u / l_par)^2 + (w / l_perp)^2,  [u, w] = R(-alpha) [x, y],
więc hiperparametry mają bezpośrednią interpretację dla domen paskowych:
    alpha  — kierunek pasków (oś "wzdłuż"),
    l_par  — długość korelacji wzdłuż pasków,
    l_perp — w poprzek (skala szerokości domen).
Szum: znany na piksel (np. z Laplace'a, heteroskedastyczny) + wspólny nugget (uczony).

Dokładny GP (Cholesky) — mapa 62x62 = 3844 punktów mieści się bez problemu.
Hiperparametry: maksymalizacja log-wiarygodności brzegowej na podzbiorze (n_sub punktów),
kilka startów kąta (pasek może mieć dowolny kierunek).

Uwaga: stacjonarny GP rozmywa skoki na ścianach domen. Do etykiet wariantów (dyskretnych)
używamy go jako wygładzenia CIĄGŁEJ miary dowodu (różnica log-wiarygodności dwóch wariantów),
nie jako klasyfikatora — to eksploracja, nie model docelowy.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def grid_coords(H, W):
    """Współrzędne (H*W, 2) w pikselach: kolumna x = j, wiersz y = i (oś y w dół, jak imshow)."""
    i, j = np.mgrid[0:H, 0:W]
    return np.column_stack([j.ravel(), i.ravel()]).astype(float)


class SpatialGP:
    def __init__(self, kernel="matern32"):
        if kernel not in ("matern32", "rbf"):
            raise ValueError("kernel: 'matern32' albo 'rbf'")
        self.kernel = kernel
        self.params = None        # dict: sd_f, l_par, l_perp, alpha_deg, nugget, mean

    # ------------------------------------------------------------- jądro
    def _k(self, A, B, sd_f, l_par, l_perp, alpha):
        c, s = np.cos(alpha), np.sin(alpha)
        Rm = np.array([[c, s], [-s, c]])                  # obrót o -alpha
        a, b = A @ Rm.T, B @ Rm.T
        d2 = ((a[:, None, 0] - b[None, :, 0]) / l_par) ** 2 + \
             ((a[:, None, 1] - b[None, :, 1]) / l_perp) ** 2
        if self.kernel == "rbf":
            return sd_f ** 2 * np.exp(-0.5 * d2)
        r = np.sqrt(3.0 * d2)
        return sd_f ** 2 * (1.0 + r) * np.exp(-r)

    @staticmethod
    def _unpack(theta):
        sd_f, l_par, l_perp, nug = np.exp(theta[[0, 1, 2, 4]])
        return sd_f, l_par, l_perp, theta[3], nug

    def _nll(self, theta, X, y, noise):
        sd_f, l_par, l_perp, alpha, nug = self._unpack(theta)
        K = self._k(X, X, sd_f, l_par, l_perp, alpha)
        K[np.diag_indices_from(K)] += noise + nug + 1e-8 * sd_f ** 2
        try:
            cf = cho_factor(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e25
        a = cho_solve(cf, y, check_finite=False)
        return 0.5 * y @ a + np.sum(np.log(np.diag(cf[0]))) + 0.5 * len(y) * np.log(2 * np.pi)

    # ------------------------------------------------------------- uczenie
    def fit(self, X, y, noise_var=None, n_sub=1200, seed=0, alpha_starts=(0, 45, 90, 135),
            l_init=None, l_bounds=(0.5, 200.0), verbose=True):
        """X (n,2), y (n,), noise_var (n,) — znana wariancja szumu na punkt (albo None)."""
        X, y = np.asarray(X, float), np.asarray(y, float)
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        noise = np.zeros(len(y)) if noise_var is None else np.asarray(noise_var, float)
        ok &= np.isfinite(noise)
        X, y, noise = X[ok], y[ok], noise[ok]
        self.mean_ = float(np.mean(y))
        yc = y - self.mean_
        sd = float(np.std(yc)) or 1.0
        rng = np.random.default_rng(seed)
        sub = rng.choice(len(y), min(n_sub, len(y)), replace=False)
        Xs, ys, ns = X[sub], yc[sub], noise[sub]
        l0 = l_init or max(2.0, 0.1 * np.ptp(X, axis=0).max())
        bnds = [(np.log(sd) - 4, np.log(sd) + 3), (np.log(l_bounds[0]), np.log(l_bounds[1])),
                (np.log(l_bounds[0]), np.log(l_bounds[1])), (None, None),
                (np.log(sd ** 2) - 14, np.log(sd ** 2) + 1)]
        best = None
        for a0 in alpha_starts:
            th0 = np.array([np.log(sd), np.log(l0), np.log(l0 / 2), np.deg2rad(a0),
                            np.log(0.1 * sd ** 2)])
            res = minimize(self._nll, th0, args=(Xs, ys, ns), method="L-BFGS-B", bounds=bnds)
            if verbose:
                p = self._unpack(res.x)
                print(f"  start α={a0:3d}°: nll={res.fun:.2f}, l_par={p[1]:.2f}, "
                      f"l_perp={p[2]:.2f}, α={np.rad2deg(p[3]) % 180:.1f}°")
            if best is None or res.fun < best.fun:
                best = res
        sd_f, l_par, l_perp, alpha, nug = self._unpack(best.x)
        if l_perp > l_par:                     # konwencja: l_par = dłuższa skala
            l_par, l_perp, alpha = l_perp, l_par, alpha + np.pi / 2
        self.params = dict(sd_f=sd_f, l_par=l_par, l_perp=l_perp,
                           alpha_deg=float(np.rad2deg(alpha) % 180), nugget=nug, mean=self.mean_,
                           nll=float(best.fun), n_sub=len(sub))
        # pełne dane do predykcji
        self._X, self._y, self._noise = X, yc, noise
        K = self._k(X, X, sd_f, l_par, l_perp, np.deg2rad(self.params["alpha_deg"]))
        K[np.diag_indices_from(K)] += noise + nug + 1e-8 * sd_f ** 2
        self._cf = cho_factor(K, lower=True, check_finite=False)
        self._alpha_vec = cho_solve(self._cf, yc, check_finite=False)
        return self

    # ------------------------------------------------------------- predykcja
    def predict(self, Xs, return_std=True, chunk=2000):
        """Średnia (i odchylenie) procesu ukrytego f w punktach Xs (bez szumu pomiaru)."""
        p = self.params
        Xs = np.asarray(Xs, float)
        mu, sd = np.empty(len(Xs)), np.empty(len(Xs))
        for s in range(0, len(Xs), chunk):
            Ks = self._k(Xs[s:s + chunk], self._X, p["sd_f"], p["l_par"], p["l_perp"],
                         np.deg2rad(p["alpha_deg"]))
            mu[s:s + chunk] = Ks @ self._alpha_vec + self.mean_
            if return_std:
                v = cho_solve(self._cf, Ks.T, check_finite=False)
                var = p["sd_f"] ** 2 - np.sum(Ks * v.T, axis=1)
                sd[s:s + chunk] = np.sqrt(np.clip(var, 0, None))
        return (mu, sd) if return_std else mu

    def loo(self):
        """Predykcje leave-one-out na danych uczących (dokładne, z jednego Choleskiego):
        mu_i = y_i - [K^-1 y]_i / [K^-1]_ii,  var_i = 1/[K^-1]_ii (z szumem)."""
        Kinv = cho_solve(self._cf, np.eye(len(self._y)), check_finite=False)
        d = np.diag(Kinv)
        mu = self._y - self._alpha_vec / d + self.mean_
        return mu, np.sqrt(1.0 / d)

    def summary(self):
        p = self.params
        return (f"jądro {self.kernel}: sd_f={p['sd_f']:.3g}, l_par={p['l_par']:.2f} px, "
                f"l_perp={p['l_perp']:.2f} px, α={p['alpha_deg']:.1f}°, "
                f"nugget sd={np.sqrt(p['nugget']):.3g}")


class SharedLatentGP:
    """Kilka map (wyjść) napędzanych jednym ukrytym polem f(x) (liniowy model koregionalizacji
    z jednym procesem ukrytym):

        y_j(x) = c_j + a_j * f(x) + eps_j(x),   f ~ GP(0, k_aniso), var f = 1,
        eps_j ~ N(0, noise_j(x) + nugget_j).

    Zastosowanie: PC1 reszt i składowe obrotu sieci zależą od tego samego ułamka pasków
    w pikselu; wspólne f łączy informację z natężeń i z geometrii. Identyfikowalne są
    stosunki a_j / a_0 (np. obrót sieci na jednostkę PC1) i kształt f; skala f (co znaczy
    "czysty pasek") — NIE, trzeba ją zakotwiczyć osobno.

    Wyjścia są standaryzowane wewnętrznie; params["a"] w jednostkach danych.
    """

    def __init__(self, kernel="matern32"):
        self._base = SpatialGP(kernel)
        self.params = None

    def _K(self, X, a, l_par, l_perp, alpha, nv):
        Kf = self._base._k(X, X, 1.0, l_par, l_perp, alpha)
        K = np.kron(np.outer(a, a), Kf)
        K[np.diag_indices_from(K)] += nv.T.ravel() + 1e-8
        return K

    def _unpack(self, th, J):
        l_par, l_perp = np.exp(th[0]), np.exp(th[1])
        alpha = th[2]
        a = th[3:3 + J]
        nug = np.exp(th[3 + J:3 + 2 * J])
        return l_par, l_perp, alpha, a, nug

    def _nll(self, th, X, Z, nv0):
        J = Z.shape[1]
        l_par, l_perp, alpha, a, nug = self._unpack(th, J)
        K = self._K(X, a, l_par, l_perp, alpha, nv0 + nug[None, :])
        try:
            cf = cho_factor(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e25
        y = Z.T.ravel()
        al = cho_solve(cf, y, check_finite=False)
        return 0.5 * y @ al + np.sum(np.log(np.diag(cf[0]))) + 0.5 * len(y) * np.log(2 * np.pi)

    def fit(self, X, Y, noise_var=None, n_sub=500, n_pred=1200, seed=0, alpha0=45.0, l0=(20.0, 3.0),
            names=None, verbose=True):
        """X (n,2), Y (n,J), noise_var (n,J) znana wariancja (NaN/None = 0)."""
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        J = Y.shape[1]
        nv = np.zeros_like(Y) if noise_var is None else np.nan_to_num(np.asarray(noise_var, float))
        ok = np.isfinite(Y).all(1) & np.isfinite(X).all(1)
        X, Y, nv = X[ok], Y[ok], nv[ok]
        self.mu_, self.sd_ = Y.mean(0), Y.std(0)
        Z = (Y - self.mu_) / self.sd_
        nvz = nv / self.sd_ ** 2
        rng = np.random.default_rng(seed)
        sub = rng.choice(len(Z), min(n_sub, len(Z)), replace=False)
        # start: ładunki z korelacji z pierwszym wyjściem
        a0 = np.array([np.corrcoef(Z[:, 0], Z[:, j])[0, 1] for j in range(J)]) * 0.9
        a0[0] = 0.9
        th0 = np.r_[np.log(l0[0]), np.log(l0[1]), np.deg2rad(alpha0), a0, np.log(np.full(J, 0.2))]
        bnds = [(np.log(0.5), np.log(300)), (np.log(0.5), np.log(300)), (None, None)] + \
               [(-5, 5)] * J + [(np.log(1e-4), np.log(5))] * J
        res = minimize(self._nll, th0, args=(X[sub], Z[sub], nvz[sub]), method="L-BFGS-B",
                       bounds=bnds)
        l_par, l_perp, alpha, a, nug = self._unpack(res.x, J)
        if l_perp > l_par:
            l_par, l_perp, alpha = l_perp, l_par, alpha + np.pi / 2
        if a[0] < 0:
            a = -a
        self.params = dict(l_par=l_par, l_perp=l_perp, alpha_deg=float(np.rad2deg(alpha) % 180),
                           a=a * self.sd_, a_std=a, nugget_sd=np.sqrt(nug) * self.sd_,
                           ratio_to_first=(a * self.sd_) / (a[0] * self.sd_[0]),
                           nll=float(res.fun), names=names or [f"y{j}" for j in range(J)])
        # dane do predykcji (podzbiór, żeby macierz J*n x J*n była rozsądna)
        pr = rng.choice(len(Z), min(n_pred, len(Z)), replace=False)
        self._X, self._a = X[pr], a
        K = self._K(X[pr], a, l_par, l_perp, np.deg2rad(self.params["alpha_deg"]),
                    nvz[pr] + nug[None, :])
        self._cf = cho_factor(K, lower=True, check_finite=False)
        self._al = cho_solve(self._cf, Z[pr].T.ravel(), check_finite=False)
        if verbose:
            print(self.summary())
        return self

    def predict_latent(self, Xs, chunk=1500):
        """Średnia i sd ukrytego pola f (standaryzowanego) w punktach Xs."""
        p = self.params
        Xs = np.asarray(Xs, float)
        mu, sd = np.empty(len(Xs)), np.empty(len(Xs))
        for s in range(0, len(Xs), chunk):
            k = self._base._k(Xs[s:s + chunk], self._X, 1.0, p["l_par"], p["l_perp"],
                              np.deg2rad(p["alpha_deg"]))
            Kc = np.hstack([aj * k for aj in self._a])               # cov(f*, y)
            mu[s:s + chunk] = Kc @ self._al
            v = cho_solve(self._cf, Kc.T, check_finite=False)
            sd[s:s + chunk] = np.sqrt(np.clip(1.0 - np.sum(Kc * v.T, axis=1), 0, None))
        return mu, sd

    def predict_outputs(self, Xs):
        """Średnie wyjść (n*, J) w jednostkach danych: c_j + a_j f*."""
        f, _ = self.predict_latent(Xs)
        return self.mu_[None, :] + f[:, None] * self.params["a"][None, :]

    def summary(self):
        p = self.params
        s = (f"wspólne pole: l_par={p['l_par']:.2f} px, l_perp={p['l_perp']:.2f} px, "
             f"α={p['alpha_deg']:.1f}°\n")
        for nm, a, r, ns in zip(p["names"], p["a"], p["ratio_to_first"], p["nugget_sd"]):
            s += f"  {nm:<6} ładunek {a:+.4g}   / pierwsze wyjście {r:+.4g}   nugget sd {ns:.3g}\n"
        return s
