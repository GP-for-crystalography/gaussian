"""
orient_fit.py — regresja: profil eCHORD (180 punktów w omega) -> orientacja (phi1, Phi, phi2).

Etap BEZ pełnej wiarygodności. Koszt = 1 - r (korelacja Pearsona pomiar–model), więc wynik
jest z założenia odporny na nieznany gain/offset piksela. Opcjonalnie n_bg_harm > 0 usuwa
najniższe harmoniki w omega z OBU stron (pomiaru i modelu) — prosty zamiennik modelu tła.

Model do przodu jest wymienny (ten sam interfejs):
  * ConeModel — bezpośrednio echord_cone.MasterPattern.cone_profiles ("wyrocznia"),
  * GPModel   — emulator gp_echord.EchordGP (albo TorchEchordGP).
Dzięki temu można rozdzielić błąd ALGORYTMU (cone -> cone) od błędu EMULATORA (cone -> gp).

Algorytm dla jednego profilu:
  1. Szukanie globalne: bank profili kanonicznych F_i(psi) dla siatki (Phi, phi2) pokrywającej
     sferę normalnej próbki (w układzie kryształu, zredukowaną przez C4). Dla każdego F_i
     korelacja ze WSZYSTKIMI przesunięciami phi1 = j*dpsi naraz — trzy FFT długości M
     (pomiar wstawiony jako "grzebień" w punkty psi_k = phi0 + sense*omega_k).
  2. Kilka najlepszych, różnych (mod C4) kandydatów -> lokalnie least_squares po
     wektorze obrotu v (g = g0 @ R(v)), bez osobliwości Eulera przy Phi = 0.

Konwencje jak w echord_cone: g pasywne próbka -> kryształ, Bunge, symetria kryształu S g.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import fft as sfft
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import echord_cone as ec


# ==========================================================================
# Geometria orientacji
# ==========================================================================
def c4_ops() -> np.ndarray:
    """Obroty C4 wokół osi z kryształu (4,3,3)."""
    out = []
    for k in range(4):
        a = k * np.pi / 2
        c, s = round(np.cos(a)), round(np.sin(a))
        out.append(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float))
    return np.stack(out)


def d4_ops() -> np.ndarray:
    """C4 + osie 2-krotne w płaszczyźnie bazowej (8,3,3) = grupa właściwa Laue 4/mmm.

    Dezorientacja modulo D4 ignoruje POLARNOŚĆ (+c vs -c). Dla fazy 4mm orientacje g
    i C2x g różnią się tylko częścią mastera łamiącą prawo Friedla — porównanie mis_C4
    z mis_D4 pokazuje, czy błędy to głównie pomyłki polarności (domeny 180°).
    """
    c2x = np.diag([1.0, -1.0, -1.0])
    c4 = c4_ops()
    return np.concatenate([c4, c4 @ c2x])


def cubic_ops() -> np.ndarray:
    """24 obroty grupy O (sześcian): macierze permutacji ze znakami o wyznaczniku +1."""
    import itertools
    out = []
    for p in itertools.permutations(range(3)):
        for sg in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            m[range(3), p] = sg
            if np.linalg.det(m) > 0:
                out.append(m)
    return np.stack(out)


def pseudo_variants(ops_hi=None, ops_lo=None) -> np.ndarray:
    """Reprezentanci warstw O / C4 (6 macierzy, pierwsza = I).

    KTN jest pseudoregularny (c/a ~ 1.002): orientacje V g różnią się tylko słabym
    zniekształceniem tetragonalnym i polarnością, więc dają bardzo podobne profile.
    3 wybory osi c (domeny 90°) x 2 zwroty (domeny 180°) = 6 wariantów.
    """
    ops_hi = cubic_ops() if ops_hi is None else ops_hi
    ops_lo = c4_ops() if ops_lo is None else ops_lo
    reps = [np.eye(3)]
    for V in ops_hi:
        covered = any(np.allclose(V, S @ R) for R in reps for S in ops_lo)
        if not covered:
            reps.append(V)
    return np.stack(reps)


def euler_from_matrix(g: np.ndarray) -> np.ndarray:
    """Macierze g (N,3,3) -> kąty Bunge (N,3) [deg], odwrotność ec.bunge_to_matrix.

    Przy Phi ~ 0 (albo 180) phi1 i phi2 są nierozróżnialne — przyjmujemy phi2 = 0
    i całą sumę wkładamy w phi1. Dla profilu to bez znaczenia (liczy się tylko kombinacja).
    """
    g = np.asarray(g, dtype=float).reshape(-1, 3, 3)
    Phi = np.arccos(np.clip(g[:, 2, 2], -1.0, 1.0))
    sPhi = np.sin(Phi)
    phi1 = np.arctan2(g[:, 2, 0], -g[:, 2, 1])
    phi2 = np.arctan2(g[:, 0, 2], g[:, 1, 2])
    deg = sPhi < 1e-8
    if deg.any():
        # Phi = 0: g00 = cos(phi1+phi2), g01 = sin(phi1+phi2)
        # Phi = pi: g00 = cos(phi1-phi2), g01 = sin(phi1-phi2)
        phi1[deg] = np.arctan2(g[deg, 0, 1], g[deg, 0, 0])
        phi2[deg] = 0.0
    return np.rad2deg(np.column_stack([phi1, Phi, phi2])) % [360.0, 360.0, 360.0]


def misorientation_deg(g1, g2, ops=None) -> np.ndarray:
    """Kąt dezorientacji [deg] min po S w ops: kąt obrotu (S g1) g2^T."""
    ops = c4_ops() if ops is None else np.asarray(ops)
    g1 = np.asarray(g1, dtype=float).reshape(-1, 3, 3)
    g2 = np.asarray(g2, dtype=float).reshape(-1, 3, 3)
    # tr(S g1 g2^T) dla każdego S
    d = np.einsum("nij,nkj->nik", g1, g2)            # g1 g2^T
    tr = np.einsum("sij,nji->ns", ops, d)            # tr(S d)
    R = np.einsum("nij,njk->nik", ops[tr.argmax(axis=1)], d)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    # atan2 zamiast arccos: dokładne także dla bardzo małych kątów
    return np.rad2deg(np.arctan2(0.5 * np.linalg.norm(v, axis=1), 0.5 * (tr.max(axis=1) - 1.0)))


def random_orientations(n, seed=0) -> np.ndarray:
    """Jednorodnie losowe g (n,3,3) na SO(3) (rozkład Haara nie zależy od konwencji)."""
    return Rotation.random(n, random_state=seed).as_matrix()


def canonical_candidates(spacing_deg=0.7, n_fold=4):
    """Siatka (Phi, phi2) [deg] ~ równomierna na sferze normalnej próbki (spirala Fibonacciego).

    Kierunek normalnej próbki w układzie kryształu: g z = (s2 sPhi, c2 sPhi, cPhi), więc (Phi, phi2)
    to współrzędne sferyczne. Obrót C4 wokół z kryształu przesuwa phi2 o 90°, więc wystarcza
    phi2 w [0, 360/n_fold). Pełny zakres Phi (0–180°): faza niecentrosymetryczna.
    """
    n_full = int(np.ceil(4 * np.pi / np.deg2rad(spacing_deg) ** 2))
    i = np.arange(n_full) + 0.5
    z = 1.0 - 2.0 * i / n_full
    az = (np.pi * (1.0 + 5 ** 0.5) * i) % (2 * np.pi)
    Phi = np.rad2deg(np.arccos(z))
    phi2 = np.rad2deg(np.arctan2(np.cos(az), np.sin(az))) % 360.0   # x = s2 sPhi, y = c2 sPhi
    keep = phi2 < 360.0 / n_fold
    return np.column_stack([Phi[keep], phi2[keep]])


# ==========================================================================
# Przetwarzanie profili
# ==========================================================================
def highpass(x, n_harm, axis=-1):
    """Usuwa harmoniki 0..n_harm (dokładny rzut dla próbkowania równomiernego pełnego okręgu)."""
    if n_harm is None or n_harm < 0:
        return x
    X = np.fft.rfft(x, axis=axis)
    sl = [slice(None)] * X.ndim
    sl[axis] = slice(0, n_harm + 1)
    X[tuple(sl)] = 0.0
    return np.fft.irfft(X, n=np.shape(x)[axis], axis=axis)


def bg_projector(n_omega, n_harm=0, mask=None):
    """Macierz P (n_keep, n_keep) rzutująca na dopełnienie harmonik 0..n_harm w omega,
    liczona TYLKO na punktach niezamaskowanych (dokładny rzut także przy brakujących klatkach).
    n_harm = 0 -> zwykłe centrowanie. Działa: x_proj = P @ x[keep]."""
    keep = np.ones(n_omega, bool) if mask is None else np.asarray(mask, bool)
    w = 2 * np.pi * np.arange(n_omega)[keep] / n_omega
    cols = [np.ones(keep.sum())]
    for k in range(1, max(0, n_harm) + 1):
        cols += [np.cos(k * w), np.sin(k * w)]
    Q, _ = np.linalg.qr(np.column_stack(cols))
    return np.eye(keep.sum()) - Q @ Q.T


def band_projector(n_omega, bands, mask=None):
    """Macierz P (n_keep, n_keep) rzutująca na podprzestrzeń wybranych harmonik w omega
    (np. bands = [(21, 60)]), liczona na punktach niezamaskowanych. Harmonika 0 (średnia)
    jest pomijana, chyba że pasmo zaczyna się od 0. Korelacja po takim rzucie to korelacja
    "w paśmie" — tylko w skalach, w których model zgadza się z pomiarem."""
    keep = np.ones(n_omega, bool) if mask is None else np.asarray(mask, bool)
    w = 2 * np.pi * np.arange(n_omega)[keep] / n_omega
    cols = []
    for lo, hi in bands:
        for k in range(int(lo), int(hi) + 1):
            if k == 0:
                cols.append(np.ones(keep.sum()))
            else:
                cols += [np.cos(k * w), np.sin(k * w)]
    A = np.column_stack(cols)
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    U = U[:, S > 1e-8 * S.max()]            # np. sin(90 w) = 0 przy pełnym próbkowaniu
    return U @ U.T


def make_projector(n_omega, spec=0, mask=None):
    """spec: int K -> usuń harmoniki 0..K (bg_projector); lista pasm [(lo, hi), ...] ->
    zostaw tylko te harmoniki (band_projector)."""
    if isinstance(spec, (list, tuple)):
        return band_projector(n_omega, spec, mask)
    return bg_projector(n_omega, spec, mask)


def projector_dof(n_omega, spec=0, mask=None):
    """Liczba stopni swobody (rząd) projektora — właściwe n do log-wiarygodności."""
    return int(round(np.trace(make_projector(n_omega, spec, mask))))


def pearson(y, mu, mask=None):
    """Korelacja Pearsona wzdłuż ostatniej osi (mu może być (N,M), y (M,))."""
    if mask is not None:
        y, mu = y[..., mask], mu[..., mask]
    yc = y - y.mean(-1, keepdims=True)
    mc = mu - mu.mean(-1, keepdims=True)
    return (yc * mc).sum(-1) / np.sqrt((yc * yc).sum(-1) * (mc * mc).sum(-1) + 1e-300)


# ==========================================================================
# Modele do przodu (wspólny interfejs)
# ==========================================================================
class ConeModel:
    """Model dokładny: próbkowanie mastera po stożku."""

    name = "cone"

    def __init__(self, mp, coneAngle, invert=False, sigma_conv_deg=0.0, dpsi=0.25,
                 omega_step=2.0):
        self.mp, self.coneAngle, self.invert = mp, coneAngle, invert
        self.sigma_conv_deg, self.dpsi, self.omega_step = sigma_conv_deg, dpsi, omega_step
        self.omega = np.arange(int(round(360.0 / omega_step))) * omega_step

    def profiles(self, g, phi0=0.0, sense=+1):
        return self.mp.cone_profiles(np.asarray(g).reshape(-1, 3, 3), self.coneAngle,
                                     self.omega_step, phi0=phi0, sense=sense,
                                     sigma_conv_deg=self.sigma_conv_deg, rep="matrix",
                                     invert=self.invert)[1]

    def canonical(self, angles_deg, chunk=2048):
        """Profile kanoniczne F(psi), psi = j*dpsi, dla (Phi, phi2) [deg] -> (n, M)."""
        a = np.asarray(angles_deg, dtype=float).reshape(-1, 2)
        e = np.column_stack([np.zeros(len(a)), a[:, 0], a[:, 1]])
        out = []
        for s in range(0, len(e), chunk):
            out.append(self.mp.cone_profiles(e[s:s + chunk], self.coneAngle, self.dpsi,
                                             phi0=0.0, sense=+1,
                                             sigma_conv_deg=self.sigma_conv_deg, rep="euler",
                                             invert=self.invert)[1])
        return np.concatenate(out).astype(np.float32)


class GPModel:
    """Emulator GP (gp_echord.EchordGP albo TorchEchordGP) — średnia jako model do przodu."""

    name = "gp"

    def __init__(self, gp, omega_step=2.0):
        self.gp = gp
        m = gp.meta
        self.coneAngle, self.invert = m["coneAngle"], m["invert"]
        self.sigma_conv_deg = m["sigma_conv_deg"]
        self.dpsi = m.get("dpsi", getattr(gp, "dpsi", None))
        self.omega_step = omega_step
        self.omega = np.arange(int(round(360.0 / omega_step))) * omega_step

    def profiles(self, g, phi0=0.0, sense=+1):
        res = self.gp.predict_profiles(np.asarray(g).reshape(-1, 3, 3), rep="matrix",
                                       omega_deg=self.omega, phi0=phi0, sense=sense,
                                       return_std=False)
        return res[1] if isinstance(res, tuple) else res

    def profiles_std(self, g, phi0=0.0, sense=+1):
        """Pełna predykcja (mu, sd) — do diagnostyki."""
        _, mu, sd = self.gp.predict_profiles(np.asarray(g).reshape(-1, 3, 3), rep="matrix",
                                             omega_deg=self.omega, phi0=phi0, sense=sense,
                                             return_std=True)
        return mu, sd

    @staticmethod
    def canonical_X(angles_deg):
        """(Phi, phi2) [deg] -> wejście GP: R(0, Phi, phi2) spłaszczone do 9 liczb.
        UWAGA: zgodność z G.canonical_matrices sprawdza self_check()."""
        a = np.asarray(angles_deg, dtype=float).reshape(-1, 2)
        e = np.column_stack([np.zeros(len(a)), a])
        return ec.bunge_to_matrix(e).reshape(-1, 9)

    def canonical(self, angles_deg, chunk=512):
        mean = self.gp.predict_canonical(self.canonical_X(angles_deg), return_std=False)
        return np.asarray(mean, dtype=np.float32)


def self_check(model, n=20, phi0=17.0, sense=-1, seed=1):
    """Spójność: profil złożony z banku kanonicznego (grzebień) == model.profiles(g).

    Sprawdza naraz: konwencję wejścia GP, siatkę psi, wzór psi = phi0 + sense*omega - phi1.
    Zwraca (min r, max |różnica| / std). Dla ConeModel ~ (1, 0) z dokładnością
    interpolacji mastera; dla GP < 1e-6, jeśli konwencje się zgadzają.
    """
    g = random_orientations(n, seed)
    e = euler_from_matrix(g)
    # phi1 zaokrąglone do siatki dpsi, żeby test był dokładny (bez interpolacji po psi)
    e[:, 0] = np.round(e[:, 0] / model.dpsi) * model.dpsi
    g = ec.bunge_to_matrix(e)
    phi0 = np.round(phi0 / model.dpsi) * model.dpsi
    F = model.canonical(e[:, 1:])
    M = F.shape[1]
    idx = np.round((phi0 + sense * model.omega[None, :] - e[:, :1]) / model.dpsi).astype(int) % M
    from_bank = np.take_along_axis(F, idx, axis=1)
    direct = model.profiles(g, phi0, sense)
    r = pearson(direct, from_bank)
    err = np.abs(direct - from_bank).max() / direct.std()
    return float(r.min()), float(err)


# ==========================================================================
# Szukanie globalne: bank + korelacja po wszystkich przesunięciach (FFT)
# ==========================================================================
@dataclass
class Bank:
    angles: np.ndarray          # (n, 2) [deg] (Phi, phi2)
    F: np.ndarray               # (n, M) float32 profile kanoniczne (ew. po highpass)
    dpsi: float
    n_bg_harm: int = 0
    _FF: np.ndarray = field(default=None, repr=False)    # rfft(F)
    _FF2: np.ndarray = field(default=None, repr=False)   # rfft(F^2)
    # wariancje F w punktach grzebienia dla każdego przesunięcia; nie zależą od y,
    # tylko od położeń punktów (phi0, sense, maska), więc liczymy je raz na geometrię
    _var_cache: dict = field(default_factory=dict, repr=False)

    @property
    def M(self):
        return self.F.shape[1]

    def prepare(self):
        # Pearson nie zależy od stałej dodanej do F, a odjęcie średniej i podzielenie przez std
        # chroni przed utratą cyfr w s2 - s1^2/n przy obliczeniach w float32
        F = self.F.astype(np.float64)
        F = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-300)
        self._FF = sfft.rfft(F, axis=1, workers=-1).astype(np.complex64)
        self._FF2 = sfft.rfft(F ** 2, axis=1, workers=-1).astype(np.complex64)
        self._var_cache = {}
        return self

    def comb_variance(self, pos, chunk=4096):
        """varF[i, j] = sum_k F_i[n_k-j]^2 - (sum_k F_i[n_k-j])^2 / n  (z pamięci podręcznej)."""
        key = pos.tobytes()
        if key not in self._var_cache:
            if len(self._var_cache) > 4:          # różne maski -> nie trzymaj wszystkich
                self._var_cache.clear()
            M, n = self.M, len(pos)
            c1 = np.zeros(M, np.float32); c1[pos] = 1.0
            C1 = sfft.rfft(c1)
            out = np.empty(self.F.shape, np.float32)
            for s in range(0, len(self.F), chunk):
                s1 = sfft.irfft(C1[None, :] * np.conj(self._FF[s:s + chunk]), n=M, workers=-1)
                s2 = sfft.irfft(C1[None, :] * np.conj(self._FF2[s:s + chunk]), n=M, workers=-1)
                out[s:s + chunk] = np.clip(s2 - s1 * s1 / n, 1e-30, None)
            self._var_cache[key] = out
        return self._var_cache[key]

    def save(self, path):
        np.savez_compressed(path, angles=self.angles, F=self.F, dpsi=self.dpsi,
                            n_bg_harm=self.n_bg_harm)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["angles"], d["F"], float(d["dpsi"]), int(d["n_bg_harm"])).prepare()


def build_bank(model, spacing_deg=0.7, n_fold=4, n_bg_harm=0, verbose=True):
    t0 = time.perf_counter()
    ang = canonical_candidates(spacing_deg, n_fold)
    F = model.canonical(ang)
    if n_bg_harm > 0:
        # w psi: jeden obrót psi = jeden obrót omega, więc harmoniki odpowiadają sobie
        # (dokładnie tylko dla części pasmowo ograniczonej; do startu wystarczy)
        F = highpass(F.astype(np.float64), n_bg_harm).astype(np.float32)
    if verbose:
        print(f"[bank/{model.name}] {len(ang)} kandydatów (Phi, phi2), oczko ~{spacing_deg}°, "
              f"M = {F.shape[1]}, {F.nbytes / 2**20:.0f} MB, {time.perf_counter() - t0:.1f} s")
    return Bank(ang, F, model.dpsi, n_bg_harm).prepare()


def comb_positions(omega, phi0, sense, dpsi, M):
    """Indeksy siatki psi odpowiadające punktom pomiaru dla phi1 = 0."""
    pos = (phi0 + sense * np.asarray(omega)) / dpsi
    ipos = np.round(pos).astype(int)
    if np.abs(pos - ipos).max() > 1e-6:
        # phi0 / krok omega nie leżą na siatce psi: start przybliżony (do dpsi/2),
        # lokalna optymalizacja i tak liczy dokładnie
        pass
    return ipos % M


def shift_correlation(bank: Bank, y, omega, phi0, sense, mask=None, rows=None):
    """Korelacja Pearsona r[i, j] pomiaru y z F_i dla phi1 = j*dpsi, wszystkie j naraz.

    dot(j)  = sum_k y_k F[n_k - j]   -> korelacja wzajemna F z grzebieniem c (c[n_k] = y_k)
    s1(j)   = sum_k F[n_k - j],  s2(j) = sum_k F[n_k - j]^2  -> to samo z grzebieniem jedynek
    """
    M = bank.M
    y = np.asarray(y, dtype=float)
    keep = np.ones(y.size, bool) if mask is None else np.asarray(mask, bool)
    pos = comb_positions(omega, phi0, sense, bank.dpsi, M)[keep]
    yk = y[keep]
    if len(np.unique(pos)) != len(pos):
        raise ValueError("Dwa punkty pomiaru trafiają w ten sam punkt psi — zmniejsz dpsi")
    rows = slice(None) if rows is None else rows
    varF = bank.comb_variance(pos)[rows]                         # liczone raz na geometrię
    c_y = np.zeros(M, np.float32); c_y[pos] = yk - yk.mean()
    Cy = sfft.rfft(c_y)
    dot = sfft.irfft(Cy[None, :] * np.conj(bank._FF[rows]), n=M, workers=-1)   # sum (y-ȳ) F
    vary = np.sum((yk - yk.mean()) ** 2)
    return dot / np.sqrt(varF * vary)


def global_search(bank: Bank, y, omega, phi0, sense, mask=None, top_k=5, min_sep_deg=3.0,
                  ops=None, chunk=4096):
    """Najlepsze top_k startów (g (k,3,3), r (k,)), wzajemnie odległe o >= min_sep_deg (mod ops)."""
    y = np.asarray(y, dtype=float)
    if bank.n_bg_harm > 0:
        if mask is None:
            y = highpass(y, bank.n_bg_harm)
        else:                      # rzut tylko na punktach niezamaskowanych; reszta nieużywana
            keep = np.asarray(mask, bool)
            y = y.copy()
            y[keep] = bg_projector(y.size, bank.n_bg_harm, keep) @ y[keep]
    n = len(bank.angles)
    best_r = np.empty(n)
    best_j = np.empty(n, int)
    for s in range(0, n, chunk):
        r = shift_correlation(bank, y, omega, phi0, sense, mask, rows=slice(s, s + chunk))
        best_j[s:s + chunk] = r.argmax(1)
        best_r[s:s + chunk] = r.max(1)
    order = np.argsort(-best_r)[:max(50, 10 * top_k)]
    e = np.column_stack([best_j[order] * bank.dpsi, bank.angles[order]])
    gs = ec.bunge_to_matrix(e)
    chosen = []
    for i in range(len(order)):
        if all(misorientation_deg(gs[i], gs[c], ops)[0] >= min_sep_deg for c in chosen):
            chosen.append(i)
        if len(chosen) == top_k:
            break
    return gs[chosen], best_r[order][chosen]


# ==========================================================================
# Optymalizacja lokalna
# ==========================================================================
def refine(model, y, g0, phi0, sense, mask=None, n_bg_harm=0, max_nfev=200, diff_step=1e-3):
    """Minimalizuje 1 - r^2 po wektorze obrotu v: g = g0 @ R(v).

    r liczone po rzucie P (make_projector): n_bg_harm = K usuwa średnią i harmoniki 1..K,
    n_bg_harm = [(lo, hi), ...] zostawia tylko te pasma; dokładnie także przy maskach. Gain wyeliminowany analitycznie (rzut zmiennych
    liniowych), więc suma kwadratów reszt = 1 - r^2.

    Jakobian różnicami skończonymi, ale trzy przesunięte profile idą do modelu jednym
    wywołaniem (partia). Zwraca (g, r, info); info["cov_deg2"] — kowariancja v [deg^2]
    z przybliżenia Laplace'a: s^2 (J^T J)^-1, s^2 = SSE / (n - 3 - 1), w układzie próbki
    (v działa po prawej stronie g0, czyli na wektory próbki).
    """
    y = np.asarray(y, dtype=float)
    keep = np.ones(y.size, bool) if mask is None else np.asarray(mask, bool)
    Pm = make_projector(y.size, n_bg_harm, keep)
    yc = Pm @ y[keep]
    ny = np.linalg.norm(yc)
    ncall = [0]

    def model_c(V):
        V = np.atleast_2d(V)
        G = np.einsum("ij,bjk->bik", g0, Rotation.from_rotvec(V).as_matrix())
        mu = model.profiles(G, phi0, sense)
        ncall[0] += 1
        return mu[:, keep] @ Pm.T

    def resid_from(mc):
        a = (mc @ yc) / np.maximum((mc * mc).sum(1), 1e-300)
        return (yc[None, :] - a[:, None] * mc) / ny

    def resid(v):
        return resid_from(model_c(v))[0]

    def jac(v):
        h = diff_step * np.maximum(1.0, np.abs(v))
        V = np.vstack([v, v + np.diag(h)])
        R = resid_from(model_c(V))
        return ((R[1:] - R[0]) / h[:, None]).T

    sol = least_squares(resid, np.zeros(3), jac=jac, method="trf",
                        x_scale=np.full(3, 0.01), max_nfev=max_nfev)
    g = g0 @ Rotation.from_rotvec(sol.x).as_matrix()
    mc = model_c(sol.x)[0]
    r = float(yc @ mc / (ny * np.linalg.norm(mc) + 1e-300))
    J = sol.jac
    dof = max(1, int(round(np.trace(Pm))) - 3 - 1)
    s2 = float(np.sum(sol.fun ** 2)) / dof
    try:
        cov = s2 * np.linalg.inv(J.T @ J) * np.rad2deg(1.0) ** 2
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.nan)
    return g, r, dict(nfev=sol.nfev, ncall=ncall[0], cov_deg2=cov,
                      step_deg=float(np.rad2deg(np.linalg.norm(sol.x))), status=sol.status)


def cost_r(model, y, g, phi0, sense, mask=None, n_bg_harm=0):
    """r (Pearson po highpass) dla danych orientacji — np. w prawdziwej orientacji."""
    mu = model.profiles(np.asarray(g).reshape(-1, 3, 3), phi0, sense)
    y = np.asarray(y, dtype=float)
    keep = np.ones(y.size, bool) if mask is None else np.asarray(mask, bool)
    Pm = make_projector(y.size, n_bg_harm, keep)
    yc, mc = Pm @ y[keep], mu[:, keep] @ Pm.T
    return (mc @ yc) / (np.linalg.norm(mc, axis=1) * np.linalg.norm(yc) + 1e-300)


def variant_scores(model, y, g, variants=None, phi0=0.0, sense=+1, mask=None, n_bg_harm=0,
                   max_nfev=200, return_cov=False):
    """Dopracowuje każdy wariant V g osobno. Zwraca (G (6,3,3), r (6,)) w kolejności variants
    (z return_cov=True także cov (6,3,3) [deg^2]).

    To jest pomiar "jak bardzo dane wolą dany wariant" — do marginesu decyzji o osi c.
    """
    variants = pseudo_variants() if variants is None else variants
    G, R, C = [], [], []
    for V in variants:
        gv, rv, info = refine(model, y, V @ g, phi0, sense, mask, n_bg_harm, max_nfev)
        G.append(gv)
        R.append(rv)
        C.append(info["cov_deg2"])
    if return_cov:
        return np.stack(G), np.asarray(R), np.stack(C)
    return np.stack(G), np.asarray(R)


def margin_stats(r, n):
    """Margines decyzji z wektora r wariantów (bez znajomości prawdy).

    dr    = r_best - r_drugi,
    dlogL = n/2 * log((1 - r_2^2) / (1 - r_1^2)) — różnica log-wiarygodności przy szumie
            gaussowskim i.i.d. o nieznanej wariancji (gain/offset wyeliminowane). Przy
            niedopasowaniu strukturalnym reszty NIE są i.i.d., więc dlogL zawyża pewność.
    """
    o = np.sort(np.asarray(r))[::-1]
    r1, r2 = o[0], o[1]
    return float(r1 - r2), float(0.5 * n * np.log((1 - r2 ** 2) / max(1 - r1 ** 2, 1e-300)))


def fit_profile(model, bank, y, phi0, sense, mask=None, top_k=5, min_sep_deg=3.0,
                n_bg_harm=None, ops=None, max_nfev=200, variants=None, n_refine=None,
                score_variants=False):
    """Pełna procedura dla jednego profilu. Zwraca słownik z najlepszym g i diagnostyką.

    variants: macierze V (np. pseudo_variants()); każdy start g0 jest uzupełniany o V g0,
        więc wszystkie warianty pseudosymetryczne są jawnie sprawdzane.
    n_refine: jeśli startów jest więcej, wszystkie są najpierw OCENIANE jednym wywołaniem
        modelu (r w punkcie startowym), a optymalizowane jest tylko n_refine najlepszych.
        None = optymalizuj wszystkie (najdokładniej, najwolniej).
    score_variants: po znalezieniu najlepszego g dopracuj osobno wszystkie jego warianty
        (variant_g, variant_r) i policz margines (dr, dlogL).
    """
    nbh = bank.n_bg_harm if n_bg_harm is None else n_bg_harm
    seeds, r_seed = global_search(bank, y, model.omega, phi0, sense, mask, top_k,
                                  min_sep_deg, ops)
    if variants is not None:
        seeds = np.einsum("vij,kjl->kvil", variants, seeds).reshape(-1, 3, 3)
        r_seed = np.repeat(r_seed, len(variants))
    n_seeds = len(seeds)
    if n_refine is not None and n_seeds > n_refine:
        r0 = cost_r(model, y, seeds, phi0, sense, mask, nbh)      # jedno wywołanie, partia
        order = np.argsort(-r0)[:n_refine]
        seeds, r_seed = seeds[order], r0[order]
    best, ncall = None, 0
    for k, g0 in enumerate(seeds):
        g, r, info = refine(model, y, g0, phi0, sense, mask, nbh, max_nfev)
        ncall += info["ncall"]
        if best is None or r > best["r"]:
            best = dict(g=g, r=r, seed_rank=k, r_seed=float(r_seed[k]), **info)
    best["r_seed_top"] = float(r_seed[0])
    best["seeds"] = seeds
    best["ncall_total"] = ncall
    best["n_seeds"] = n_seeds
    if score_variants:
        Vs = pseudo_variants() if variants is None else variants
        Gv, Rv = variant_scores(model, y, best["g"], Vs, phi0, sense, mask, nbh, max_nfev)
        k = int(np.argmax(Rv))
        if Rv[k] > best["r"]:                      # wariant po dopracowaniu okazał się lepszy
            best.update(g=Gv[k], r=float(Rv[k]))
        n = int(np.sum(mask)) if mask is not None else len(y)
        best["variant_g"], best["variant_r"] = Gv, Rv
        best["dr"], best["dlogL"] = margin_stats(Rv, n)
    return best


# ==========================================================================
# Diagnostyka symetrii (ważne dla identyfikowalności sense / invert)
# ==========================================================================
def mirror_symmetry_check(mp, n=20000, seed=0):
    """Korelacja M(sigma d) z M(d) dla luster 4mm (x -> -x i przekątna) oraz M(-d) z M(d).

    Jeśli lustra dają ~1, a inwersja nie, to:
      * (g, sense) i (g'', -sense) dają IDENTYCZNE profile, g'' = sigma_c g sigma_s,
      * (g, invert) i ((-sigma_c) g, not invert) też — tzn. sense i invert nie są wyznaczalne
        z samych profili; ustala je konwencja metody słownikowej.
    """
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    base = mp.lookup(d)
    sig_x = np.diag([-1.0, 1.0, 1.0])
    sig_d = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]])
    return {"mirror_x": float(np.corrcoef(base, mp.lookup(d @ sig_x.T))[0, 1]),
            "mirror_diag": float(np.corrcoef(base, mp.lookup(d @ sig_d.T))[0, 1]),
            "inversion": float(np.corrcoef(base, mp.lookup(-d))[0, 1])}


def sense_flip_equivalent(g, sigma_c=None, sigma_s=None):
    """g'' = sigma_c g sigma_s: przy masterze z lustrem sigma_c profil g'' dla sense = -s
    i phi0 -> -phi0 jest identyczny z profilem g dla sense = s (sigma_s: y -> -y w próbce)."""
    sigma_c = np.diag([-1.0, 1.0, 1.0]) if sigma_c is None else sigma_c
    sigma_s = np.diag([1.0, -1.0, 1.0]) if sigma_s is None else sigma_s
    return sigma_c @ np.asarray(g).reshape(-1, 3, 3) @ sigma_s
