"""
gp_echord.py — emulator gaussowski (GP) profili eCHORD, uczony na symulacjach z echord_cone.

Redukcja problemu
-----------------
Dla orientacji g = (phi1, Phi, phi2) (Bunge, pasywnie próbka -> kryształ):

    I(omega) = F(psi; Phi, phi2),      psi = phi0 + sense*omega - phi1            (1)

gdzie F to "profil kanoniczny" = master pattern wzdłuż okręgu

    d_c(psi) = R(0, Phi, phi2) @ (sin th cos psi, sin th sin psi, cos th).

(1) wynika wprost z Rz(phi1) d_s(b) = d_s(b - phi1) (patrz docstring echord_cone;
tam ta sama własność jest zapisana z przeciwnym znakiem argumentu F).
GP uczy się więc tylko odwzorowania (Phi, phi2) -> F(.) na gęstej siatce psi.
phi1, phi0 i sense obsługujemy przy predykcji przesunięciem/odbiciem, bez uczenia.

Model
-----
* Wejście: macierz kanoniczna R(0, Phi, phi2). Odległość między wejściami to
  odległość cięciwowa obrotów
        delta(R1, R2) = ||R1 - R2||_F / sqrt(2) = 2 sin(w/2) ~ w,
  gdzie w = kąt dezorientacji. Na powierzchni kanonicznej metryka dezorientacji
  jest płaska: ds^2 = dPhi^2 + dphi2^2 (zmiana phi2 to obrót wokół osi z kryształu).
* Jądro: Matérn na delta, zsymetryzowane po WŁAŚCIWYCH obrotach grupy punktowej:
        k(R1, R2) = s2 * sum_S m( delta(S R1, R2) / ell ),
  z jednym parametrem anizotropii kappa: obrót wokół normalnej próbki (czyste
  przesunięcie profilu) może dekorelować wolniej/szybciej niż pochylenie.
  Jest dodatnio określone (Matérn w R^9 + suma po grupie), nie ma osobliwości
  przy Phi = 0, a symetria kryształu (M(S d) = M(d)) jest wbudowana, więc wystarczą
  dane uczące z RFZ, a predykcja działa dla dowolnej orientacji.
* Wyjście: M wartości profilu (siatka psi) jako niezależne GP ze WSPÓLNYM jądrem.
  Jedna faktoryzacja Cholesky'ego N x N obsługuje wszystkie M kolumn naraz.
* Uczenie: maksymalizacja log-wiarygodności brzegowej (ML-II) po (ell, kappa, s2, sn2),
  L-BFGS-B z analitycznym gradientem, na losowym podzbiorze n_fit punktów;
  potem warunkowanie (Cholesky) na wszystkich punktach uczących.

Konwencje: kwaterniony skalar pierwszy, P = -1, kąty w API w stopniach,
wewnątrz modułu w radianach (zaznaczone w nazwach / docstringach).
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

import echord_cone as ec

TWO_PI = 2.0 * np.pi


# ==========================================================================
# Orientacje
# ==========================================================================
def matrix_to_bunge(g) -> np.ndarray:
    """Pasywne macierze g (N,3,3) -> kąty Bunge (N,3) w RADIANACH (jak om2eu w EMsoft).

    Odwrotność ec.bunge_to_matrix. Dla Phi = 0 lub pi przyjmuje phi2 = 0.
    """
    g = np.asarray(g, dtype=float).reshape(-1, 3, 3)
    Phi = np.arccos(np.clip(g[:, 2, 2], -1.0, 1.0))
    sin_Phi = np.hypot(g[:, 2, 0], g[:, 2, 1])
    degenerate = sin_Phi < 1e-8
    phi1 = np.where(degenerate, np.arctan2(g[:, 0, 1], g[:, 0, 0]),
                    np.arctan2(g[:, 2, 0], -g[:, 2, 1]))
    phi2 = np.where(degenerate, 0.0, np.arctan2(g[:, 0, 2], g[:, 1, 2]))
    return np.column_stack([np.mod(phi1, TWO_PI), Phi, np.mod(phi2, TWO_PI)])


def canonical_matrices(angles) -> np.ndarray:
    """(N,2) [Phi, phi2] w radianach -> macierze kanoniczne R(0, Phi, phi2) (N,3,3)."""
    a = np.atleast_2d(np.asarray(angles, dtype=float))
    eul = np.column_stack([np.zeros(len(a)), a[:, 0], a[:, 1]])
    return ec.bunge_to_matrix(eul, degrees=False)


def random_quaternions(n, seed=None) -> np.ndarray:
    """n orientacji jednorodnych na SO(3) (miara Haara), kwaterniony skalar pierwszy, q0 >= 0."""
    rng = np.random.default_rng(seed)
    u1, u2, u3 = rng.random((3, n))
    q = np.column_stack([np.sqrt(1 - u1) * np.sin(TWO_PI * u2),
                         np.sqrt(1 - u1) * np.cos(TWO_PI * u2),
                         np.sqrt(u1) * np.sin(TWO_PI * u3),
                         np.sqrt(u1) * np.cos(TWO_PI * u3)])
    q[q[:, 0] < 0] *= -1
    return q


def canonical_angles(orientations, rep="quat", degrees=True, dedupe_tol_deg=1e-4):
    """Orientacje -> (kąty kanoniczne (N',2) [Phi, phi2] w rad, phi1 (N,) w rad, indeksy N').

    Orientacje różniące się tylko phi1 dają ten sam profil kanoniczny, więc
    duplikaty (z tolerancją dedupe_tol_deg) są usuwane; zwracane indeksy
    wskazują zachowane wiersze wejścia.
    """
    eu = matrix_to_bunge(ec.to_matrices(orientations, rep=rep, degrees=degrees))
    ang = eu[:, 1:]
    if dedupe_tol_deg is None:
        return ang, eu[:, 0], np.arange(len(ang))
    key = np.round(ang / np.deg2rad(dedupe_tol_deg)).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    keep = np.sort(first)
    return ang[keep], eu[:, 0], keep


def uniform_canonical_design(n, n_fold=1, Phi_range_deg=(0.0, 180.0), seed=0):
    """Plan quasi-losowy (Sobol) JEDNORODNY w metryce kanonicznej (Phi, phi2).

    Alternatywa dla pliku z EMsampleRFZ: RFZ jest jednorodne na SO(3), więc po
    usunięciu phi1 gęstość punktów w (Phi, phi2) jest ~ sin(Phi) — mało punktów
    przy Phi ~ 0 i ~ 180°. Tu gęstość jest stała. phi2 w [0, 360°/n_fold).
    Zwraca kąty (n,2) w radianach.
    """
    from scipy.stats import qmc

    with warnings.catch_warnings():  # Sobol woli n = 2^k; dla innych n też działa poprawnie
        warnings.simplefilter("ignore")
        u = qmc.Sobol(d=2, scramble=True, seed=seed).random(n)
    lo, hi = np.deg2rad(Phi_range_deg)
    return np.column_stack([lo + (hi - lo) * u[:, 0], TWO_PI / n_fold * u[:, 1]])


# ==========================================================================
# Symetria kryształu (właściwe obroty grupy punktowej)
# ==========================================================================
def _axis_angle(axis, angle_deg) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    t = np.deg2rad(angle_deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


_GENERATORS = {
    "C1": [],
    "C2x": [((1, 0, 0), 180)], "C2y": [((0, 1, 0), 180)], "C2z": [((0, 0, 1), 180)],
    "C3": [((0, 0, 1), 120)], "C4": [((0, 0, 1), 90)], "C6": [((0, 0, 1), 60)],
    "D2": [((0, 0, 1), 180), ((1, 0, 0), 180)],
    "D2diag": [((0, 0, 1), 180), ((1, 1, 0), 180)],    # osie 2 po przekątnych (-4m2)
    "D3x": [((0, 0, 1), 120), ((1, 0, 0), 180)],       # typ 321 (oś 2 || a = x)
    "D3y": [((0, 0, 1), 120), ((0, 1, 0), 180)],       # typ 312 (oś 2 || y)
    "D4": [((0, 0, 1), 90), ((1, 0, 0), 180)],
    "D6": [((0, 0, 1), 60), ((1, 0, 0), 180)],
    "T": [((0, 0, 1), 180), ((1, 1, 1), 120)],
    "O": [((0, 0, 1), 90), ((1, 1, 1), 120)],
}

# (zakres numerów grup przestrzennych, grupa właściwych obrotów); układ osi jak w EMsoft
# (x || a, z || c*; monokliniczne z osią unikalną b; trygonalne w ustawieniu heksagonalnym)
_SG_TABLE = [
    ((1, 2), "C1"), ((3, 15), "C2y"), ((16, 24), "D2"), ((25, 46), "C2z"), ((47, 74), "D2"),
    ((75, 80), "C4"), ((81, 82), "C2z"), ((83, 88), "C4"), ((89, 98), "D4"), ((99, 110), "C4"),
    ((111, 114), "D2"), ((115, 120), "D2diag"), ((121, 122), "D2"), ((123, 142), "D4"),
    ((143, 148), "C3"), ((149, 149), "D3y"), ((150, 150), "D3x"), ((151, 151), "D3y"),
    ((152, 152), "D3x"), ((153, 153), "D3y"), ((154, 155), "D3x"), ((156, 161), "C3"),
    ((162, 163), "D3y"), ((164, 167), "D3x"), ((168, 173), "C6"), ((174, 174), "C3"),
    ((175, 176), "C6"), ((177, 182), "D6"), ((183, 186), "C6"), ((187, 188), "D3y"),
    ((189, 190), "D3x"), ((191, 194), "D6"), ((195, 206), "T"), ((207, 214), "O"),
    ((215, 220), "T"), ((221, 230), "O"),
]
_RHOMBOHEDRAL = {146, 148, 155, 160, 161, 166, 167}


def proper_group_name(space_group) -> str:
    """Nazwa grupy właściwych obrotów dla numeru grupy przestrzennej (None -> 'C1')."""
    if space_group is None:
        warnings.warn("Brak numeru grupy przestrzennej — przyjmuję C1 (bez symetrii).")
        return "C1"
    for (a, b), name in _SG_TABLE:
        if a <= space_group <= b:
            if space_group in _RHOMBOHEDRAL:
                warnings.warn(f"Grupa {space_group} jest romboedryczna — zakładam ustawienie "
                              "heksagonalne (oś 3 || z). Sprawdź symmetry_check().")
            return name
    raise ValueError(f"Nieprawidłowy numer grupy przestrzennej: {space_group}")


def proper_rotations(name) -> np.ndarray:
    """Wszystkie obroty grupy (K,3,3), domknięcie generatorów; pierwszy to identyczność."""
    gens = [_axis_angle(ax, ang) for ax, ang in _GENERATORS[name]]
    ops = [np.eye(3)]
    changed = True
    while changed:
        changed = False
        for A in list(ops):
            for B in gens:
                C = A @ B
                if not any(np.allclose(C, D, atol=1e-9) for D in ops):
                    ops.append(C)
                    changed = True
    return np.array(ops)


def symmetry_check(mp, ops, n=20000, seed=0) -> np.ndarray:
    """Korelacja M(S d) z M(d) dla losowych kierunków, osobno dla każdego obrotu S.

    Wszystkie wartości powinny być ~1. Jeśli któraś jest wyraźnie mniejsza,
    tabela grup nie pasuje do ustawienia osi w pliku — wtedy podaj grupę ręcznie.
    """
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    I0 = mp.lookup(d)
    return np.array([np.corrcoef(I0, mp.lookup(d @ S.T))[0, 1] for S in ops])


# ==========================================================================
# Jądro
# ==========================================================================
class SymMaternKernel:
    """k(R1,R2) = sum_S m(delta(S R1, R2) / ell)  (bez s2).

    delta^2 = 1/2 ||(A - B) W||_F^2, W = diag(kappa, kappa, sqrt(2 - kappa^2)) działa na
    kolumny macierzy g, czyli na osie PRÓBKI (x, y, z = normalna). Dla małych kątów:
      * pochylenie wokół osi w płaszczyźnie próbki o b      -> delta = b,
      * obrót wokół normalnej próbki o a (czyste przesunięcie) -> delta = kappa * a.
    kappa = 1 to zwykła odległość cięciwowa (izotropowo, delta = 2 sin(w/2)).
    Dopuszczalne 0 < kappa < sqrt(2). Jądro pozostaje dodatnio określone
    (Matérn na odległości euklidesowej osadzenia g -> gW w R^9, plus suma po grupie).
    """

    def __init__(self, ops, nu=1.5):
        if nu not in (0.5, 1.5, 2.5, np.inf):
            raise ValueError("nu musi być 0.5, 1.5, 2.5 albo np.inf (RBF)")
        self.ops = np.asarray(ops, dtype=float).reshape(-1, 3, 3)
        self.nu = nu

    def _m(self, r2, ell):
        """m(delta/ell), dm/dlog(ell), dm/d(delta^2) dla delta^2 = r2."""
        r = np.sqrt(r2)
        a = r / ell
        if self.nu == 0.5:
            e = np.exp(-a)
            # dm/d(delta^2) jest osobliwe w delta = 0, ale tam d(delta^2)/dlog(kappa) = 0
            # (granica iloczynu = 0), więc dla delta -> 0 przyjmujemy 0
            tiny = r < 1e-6 * ell
            return e, a * e, np.where(tiny, 0.0, -e / (2 * ell * np.where(tiny, 1.0, r)))
        if self.nu == 1.5:
            u = np.sqrt(3.0) * a
            e = np.exp(-u)
            return (1 + u) * e, u * u * e, -1.5 / ell**2 * e
        if self.nu == 2.5:
            u = np.sqrt(5.0) * a
            e = np.exp(-u)
            return (1 + u + u * u / 3) * e, u * u / 3 * (1 + u) * e, -5.0 / (6 * ell**2) * (1 + u) * e
        e = np.exp(-0.5 * a * a)
        return e, a * a * e, -e / (2 * ell**2)

    @staticmethod
    def _weights(kappa):
        k2 = kappa * kappa
        if not 0.0 < k2 < 2.0:
            raise ValueError("kappa musi być w (0, sqrt(2))")
        w2 = np.array([k2, k2, 2.0 - k2])
        dw2 = np.array([2 * k2, 2 * k2, -2 * k2])  # pochodna po log(kappa)
        # spłaszczenie wierszami: element (i, j) -> indeks 3*i + j, waga kolumny j
        return np.tile(w2, 3), np.tile(dw2, 3), 2.0 + k2, 2 * k2

    def gram(self, X1, X2, ell, kappa=1.0, grad=False, chunk=1024):
        """Macierz jądra (N1,N2); z grad=True także dK/dlog(ell) i dK/dlog(kappa). X: (N,3,3)."""
        X1 = np.asarray(X1).reshape(-1, 3, 3)
        X2f = np.asarray(X2).reshape(-1, 9)
        w2f, dw2f, c0, dc0 = self._weights(kappa)
        K = np.zeros((len(X1), len(X2f)))
        if grad:
            dK_ell, dK_kap = np.zeros_like(K), np.zeros_like(K)
        for s in range(0, len(X1), chunk):
            blk = slice(s, s + chunk)
            for S in self.ops:
                A = (S @ X1[blk]).reshape(-1, 9)
                r2 = np.clip(c0 - (A * w2f) @ X2f.T, 0.0, None)
                m, dm_dlogell, dm_dr2 = self._m(r2, ell)
                K[blk] += m
                if grad:
                    dK_ell[blk] += dm_dlogell
                    dK_kap[blk] += dm_dr2 * (dc0 - (A * dw2f) @ X2f.T)
        return (K, dK_ell, dK_kap) if grad else K

    def diag_value(self, ell, kappa=1.0):
        """k(R,R) — stała (nie zależy od R)."""
        return float(self.gram(np.eye(3)[None], np.eye(3)[None], ell, kappa)[0, 0])

    def min_misorientation(self, X1, X2, chunk=1024):
        """Najmniejszy kąt dezorientacji [deg] (z symetrią) od każdego X1 do zbioru X2 oraz argmin."""
        X1 = np.asarray(X1).reshape(-1, 3, 3)
        X2f = np.asarray(X2).reshape(-1, 9)
        best_tr = np.full(len(X1), -np.inf)
        arg = np.zeros(len(X1), dtype=int)
        for s in range(0, len(X1), chunk):
            blk = slice(s, s + chunk)
            for S in self.ops:
                T = (S @ X1[blk]).reshape(-1, 9) @ X2f.T
                j = np.argmax(T, axis=1)
                t = T[np.arange(len(j)), j]
                better = t > best_tr[blk]
                best_tr[blk] = np.where(better, t, best_tr[blk])
                arg[blk] = np.where(better, j, arg[blk])
        w = np.arccos(np.clip((best_tr - 1.0) / 2.0, -1.0, 1.0))
        return np.rad2deg(w), arg


# ==========================================================================
# Dane uczące
# ==========================================================================
@dataclass
class CanonicalData:
    angles: np.ndarray          # (N,2) [Phi, phi2] w radianach
    Y: np.ndarray               # (N,M) profile kanoniczne, float32
    dpsi: float                 # krok siatki psi [deg]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.X = canonical_matrices(self.angles)

    @property
    def psi(self):
        return np.arange(self.Y.shape[1]) * self.dpsi

    def __len__(self):
        return len(self.angles)


def simulate_canonical(mp, angles, coneAngle=40.0, dpsi=0.25, invert=False,
                       sigma_conv_deg=0.0, chunk=512):
    """Profile kanoniczne F(psi; Phi, phi2) z master patternu. angles: (N,2) w radianach.

    Równoważne cone_profiles dla (0, Phi, phi2) z phi0 = 0, sense = +1, krokiem dpsi.
    """
    nsteps = 360.0 / dpsi
    if abs(nsteps - round(nsteps)) > 1e-9:
        raise ValueError("360/dpsi musi być liczbą całkowitą")
    a = np.atleast_2d(angles)
    eul = np.column_stack([np.zeros(len(a)), a[:, 0], a[:, 1]])
    _, Y = mp.cone_profiles(eul, coneAngle=coneAngle, angleStep=dpsi, phi0=0.0, sense=+1,
                            rep="euler", degrees=False, invert=invert,
                            sigma_conv_deg=sigma_conv_deg, chunk=chunk)
    return Y.astype(np.float32)


def build_training_data(mp, anglefile=None, angles=None, coneAngle=40.0, dpsi=0.25,
                        invert=False, sigma_conv_deg=0.0, verbose=True) -> CanonicalData:
    """Zbiór uczący: orientacje z pliku kątów EMsoftu (qu/eu) albo gotowe kąty kanoniczne.

    Z każdej orientacji bierzemy tylko (Phi, phi2) — phi1 to przesunięcie profilu.
    """
    t0 = time.perf_counter()
    meta = dict(master=str(mp.path), energy=str(mp.energy), space_group=mp.space_group,
                coneAngle=float(coneAngle), dpsi=float(dpsi), invert=bool(invert),
                sigma_conv_deg=float(sigma_conv_deg))
    if anglefile is not None:
        rep, ori = ec.read_anglefile(anglefile)
        angles, _, _ = canonical_angles(ori, rep=rep)
        meta.update(anglefile=str(anglefile), n_in_file=int(len(ori)))
        if verbose:
            print(f"[dane] {anglefile}: {len(ori)} orientacji ({rep}) -> "
                  f"{len(angles)} unikalnych (Phi, phi2)")
    elif angles is None:
        raise ValueError("Podaj anglefile albo angles")
    Y = simulate_canonical(mp, angles, coneAngle, dpsi, invert, sigma_conv_deg)
    if verbose:
        print(f"[dane] profile kanoniczne {Y.shape} (krok psi {dpsi}°), "
              f"{time.perf_counter() - t0:.1f} s, {Y.nbytes / 2**20:.0f} MB")
    return CanonicalData(np.asarray(angles, float), Y, float(dpsi), meta)


# ==========================================================================
# Przesunięcie profilu kanonicznego -> geometria pomiaru
# ==========================================================================
def resample_periodic(F, dpsi, psi_deg):
    """Interpolacja liniowa okresowa. F: (N,M) na siatce k*dpsi, psi_deg: (N,K) -> (N,K)."""
    M = F.shape[1]
    x = np.mod(psi_deg, 360.0) / dpsi
    i0 = np.floor(x).astype(int)
    w = x - i0
    rows = np.arange(F.shape[0])[:, None]
    return (1 - w) * F[rows, i0 % M] + w * F[rows, (i0 + 1) % M]


def profiles_from_canonical(F, dpsi, phi1_rad, omega_deg, phi0=0.0, sense=+1):
    """Profil pomiarowy z kanonicznego: I(omega) = F(phi0 + sense*omega - phi1), wzór (1)."""
    psi = phi0 + sense * np.asarray(omega_deg)[None, :] - np.rad2deg(phi1_rad)[:, None]
    return resample_periodic(F, dpsi, psi)


# ==========================================================================
# Proces gaussowski
# ==========================================================================
def _cholesky(K, jitter0=1e-8, max_tries=6):
    """Cholesky z rosnącym jitterem, gdy macierz jest numerycznie osobliwa."""
    scale = np.mean(np.diag(K))
    jitter = 0.0
    for t in range(max_tries):
        try:
            if jitter:
                K = K.copy()
                K[np.diag_indices_from(K)] += jitter
            return cho_factor(K, lower=True, check_finite=False), jitter
        except LinAlgError:
            jitter = scale * jitter0 * 10**t
    raise LinAlgError("Cholesky nie zbiega się nawet z jitterem")


class EchordGP:
    """Emulator profili kanonicznych.

    Hiperparametry (theta = ich logarytmy):
      ell   — skala korelacji dla pochyleń [rad],
      kappa — czułość na obrót wokół normalnej próbki względem pochyleń (1 = izotropowo),
      s2    — wariancja sygnału, sn2 — nugget/szum (w jednostkach standaryzowanych).
    """

    names = ("ell", "kappa", "s2", "sn2")

    def __init__(self, ops, nu=1.5, ell_deg=1.4, kappa=1.0, signal_var=1.0, noise_var=1e-2,
                 learn_kappa=True):
        self.kernel = SymMaternKernel(ops, nu)
        self.theta = np.log([np.deg2rad(ell_deg), kappa, signal_var, noise_var])
        self.learn_kappa = learn_kappa
        self.history: list[dict] = []
        self.meta: dict = {}
        self._L = None

    # --- hiperparametry w czytelnej postaci
    ell = property(lambda self: float(np.exp(self.theta[0])))
    kappa = property(lambda self: float(np.exp(self.theta[1])))
    signal_var = property(lambda self: float(np.exp(self.theta[2])))
    noise_var = property(lambda self: float(np.exp(self.theta[3])))

    def describe(self, theta=None):
        ell, kap, s2, sn2 = np.exp(self.theta if theta is None else theta)
        return (f"ell = {np.rad2deg(ell):.3f}°, kappa = {kap:.3f}, sd_f = {np.sqrt(s2):.3f}, "
                f"sd_n = {np.sqrt(sn2):.4f}")

    # --- log-wiarygodność brzegowa i gradient
    def neg_log_marginal_likelihood(self, theta, X, Yn):
        """-log p(Y | theta) / (N*M) oraz gradient po theta = log(ell, kappa, s2, sn2).

        M kolumn Y to niezależne GP ze wspólnym jądrem K:
          -log p = 1/2 tr(Y^T K^-1 Y) + M/2 log|K| + NM/2 log(2 pi)
          d(-log p)/d theta_i = -1/2 tr[(A A^T - M K^-1) dK/dtheta_i],  A = K^-1 Y
        """
        ell, kappa, s2, sn2 = np.exp(theta)
        N, M = Yn.shape
        Kb, dK_ell, dK_kap = self.kernel.gram(X, X, ell, kappa, grad=True)
        K = s2 * Kb
        K[np.diag_indices(N)] += sn2
        try:
            c, _ = _cholesky(K)
        except LinAlgError:
            return 1e10, np.zeros(4)
        A = cho_solve(c, Yn, check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(c[0])))
        nll = 0.5 * np.sum(Yn * A) + 0.5 * M * logdet + 0.5 * N * M * np.log(TWO_PI)
        W = A @ A.T - M * cho_solve(c, np.eye(N), check_finite=False)
        grad = -0.5 * np.array([s2 * np.sum(W * dK_ell), s2 * np.sum(W * dK_kap),
                                s2 * np.sum(W * Kb), sn2 * np.trace(W)])
        norm = N * M
        self._last = (np.array(theta, copy=True), nll / norm)
        return nll / norm, grad / norm

    # --- PĘTLA UCZENIA
    def fit(self, data: CanonicalData, n_fit=2000, psi_stride=1, maxiter=100,
            ell_bounds_deg=(0.05, 30.0), seed=0, verbose=True):
        """Uczenie: ML-II na losowym podzbiorze n_fit, potem warunkowanie na całości.

        psi_stride > 1 przerzedza siatkę psi tylko na czas doboru hiperparametrów
        (przyspieszenie); warunkowanie zawsze używa pełnej siatki.
        """
        rng = np.random.default_rng(seed)
        self.y_mean = float(data.Y.mean())
        self.y_std = float(data.Y.std())
        self.dpsi = data.dpsi
        self.meta = dict(data.meta)

        idx = rng.choice(len(data), size=min(n_fit, len(data)), replace=False)
        Xf = data.X[idx]
        Yf = (data.Y[idx][:, ::psi_stride].astype(np.float64) - self.y_mean) / self.y_std
        kap_b = ((np.log(0.2), np.log(np.sqrt(2.0) - 1e-3)) if self.learn_kappa
                 else (self.theta[1], self.theta[1]))
        bounds = [tuple(np.log(np.deg2rad(ell_bounds_deg))), kap_b,
                  (np.log(1e-3), np.log(1e3)), (np.log(1e-8), np.log(10.0))]

        if verbose:
            print(f"[uczenie] ML-II na {len(idx)} z {len(data)} profili, "
                  f"{Yf.shape[1]} punktów psi/profil, nu = {self.kernel.nu}, "
                  f"{len(self.kernel.ops)} obrotów symetrii")
            print(f"[uczenie] start: {self.describe()}")
        t0 = time.perf_counter()
        self.history = []

        def callback(theta):  # wywoływane przez L-BFGS-B po każdej iteracji
            nll = self._last[1] if np.allclose(theta, self._last[0]) else np.nan
            rec = dict(it=len(self.history) + 1, nll=float(nll), t=time.perf_counter() - t0,
                       **{n: float(v) for n, v in zip(self.names, np.exp(theta))})
            self.history.append(rec)
            if verbose:
                print(f"  it {rec['it']:3d}  nll/el = {nll:+.5f}  {self.describe(theta)}"
                      f"  [{rec['t']:.1f} s]")

        res = minimize(self.neg_log_marginal_likelihood, self.theta, args=(Xf, Yf), jac=True,
                       method="L-BFGS-B", bounds=bounds, callback=callback,
                       options=dict(maxiter=maxiter))
        self.theta = res.x
        if verbose:
            print(f"[uczenie] koniec ({res.message}), {time.perf_counter() - t0:.1f} s")
            print(f"[uczenie] {self.describe()}")
            for i, name in enumerate(self.names):
                lo, hi = bounds[i]
                if lo < hi and (np.isclose(res.x[i], lo) or np.isclose(res.x[i], hi)):
                    print(f"  UWAGA: {name} na granicy przedziału — rozważ zmianę granic.")
        self.condition(data, verbose=verbose)
        return self

    # --- warunkowanie na danych (Cholesky na wszystkich punktach)
    def condition(self, data: CanonicalData, verbose=True):
        """Cholesky macierzy K na WSZYSTKICH punktach uczących i A = K^-1 Y.

        Pamięć: ~8*N^2 bajtów (N = 9077 -> ~0.66 GB), faktoryzacja w miejscu.
        """
        t0 = time.perf_counter()
        self.X_train = data.X
        self.angles_train = data.angles
        self.Y_train = data.Y
        self.dpsi = data.dpsi
        N = len(data)
        self._kss = self.signal_var * self.kernel.diag_value(self.ell, self.kappa)
        jitter = 0.0
        for attempt in range(6):
            K = self.signal_var * self.kernel.gram(data.X, data.X, self.ell, self.kappa)
            K[np.diag_indices(N)] += self.noise_var + jitter
            try:
                L, low = cho_factor(K, lower=True, overwrite_a=True, check_finite=False)
                break
            except LinAlgError:
                del K
                jitter = self._kss * 1e-8 * 10**attempt
        else:
            raise LinAlgError("Cholesky nie zbiega się nawet z jitterem")
        Yn = (data.Y.astype(np.float64) - self.y_mean) / self.y_std
        self._L = L  # górny trójkąt to śmieci; solve_triangular(lower=True) czyta tylko dolny
        self._A = cho_solve((L, low), Yn, check_finite=False)
        if verbose:
            extra = f", dodany jitter {jitter:.1e}" if jitter else ""
            print(f"[warunkowanie] Cholesky {N}x{N}{extra}, {time.perf_counter() - t0:.1f} s")
        return self

    # --- predykcja
    def predict_canonical(self, Xs, return_std=True, include_noise=False, chunk=512):
        """Profile kanoniczne dla macierzy kanonicznych Xs (n,3,3).

        Zwraca średnią (n,M) w jednostkach intensywności i odchylenie (n,) — przy
        wspólnym jądrze niepewność jest taka sama dla wszystkich psi danego profilu.
        """
        if self._L is None:
            raise RuntimeError("Najpierw fit() albo condition()")
        Xs = np.asarray(Xs).reshape(-1, 3, 3)
        mean = np.empty((len(Xs), self._A.shape[1]))
        std = np.empty(len(Xs))
        for s in range(0, len(Xs), chunk):
            blk = slice(s, s + chunk)
            Ks = self.signal_var * self.kernel.gram(Xs[blk], self.X_train, self.ell, self.kappa)
            mean[blk] = Ks @ self._A
            if return_std:
                V = solve_triangular(self._L, Ks.T, lower=True, check_finite=False)
                var = np.clip(self._kss - np.sum(V * V, axis=0), 0.0, None)
                std[blk] = np.sqrt(var + (self.noise_var if include_noise else 0.0))
        mean = mean * self.y_std + self.y_mean
        return (mean, std * self.y_std) if return_std else mean

    def predict_profiles(self, orientations, rep="quat", omega_deg=None, phi0=0.0, sense=+1,
                         return_std=True):
        """Profile w geometrii pomiaru dla dowolnych orientacji (domyślnie kwaterniony EMsoftu).

        omega_deg domyślnie 0, 2, ..., 358. coneAngle/invert/zbieżność są te, na których
        uczono (self.meta) — inne wymagają nowego modelu.
        """
        if omega_deg is None:
            omega_deg = np.arange(180) * 2.0
        eu = matrix_to_bunge(ec.to_matrices(orientations, rep=rep))
        out = self.predict_canonical(canonical_matrices(eu[:, 1:]), return_std=return_std)
        F, std = out if return_std else (out, None)
        P = profiles_from_canonical(F, self.dpsi, eu[:, 0], omega_deg, phi0, sense)
        return (np.asarray(omega_deg), P, std) if return_std else (np.asarray(omega_deg), P)

    # --- zapis / odczyt
    def save(self, path):
        """Zapisuje hiperparametry i dane uczące; Cholesky liczony od nowa przy load()."""
        np.savez_compressed(
            path, theta=self.theta, nu=self.kernel.nu, ops=self.kernel.ops,
            y_mean=self.y_mean, y_std=self.y_std, dpsi=self.dpsi,
            angles=self.angles_train, Y=self.Y_train,
            meta=json.dumps(self.meta), history=json.dumps(self.history))

    @classmethod
    def load(cls, path, verbose=True):
        z = np.load(path, allow_pickle=False)
        gp = cls(z["ops"], nu=float(z["nu"]))
        gp.theta = z["theta"]
        gp.y_mean, gp.y_std = float(z["y_mean"]), float(z["y_std"])
        gp.meta = json.loads(str(z["meta"]))
        gp.history = json.loads(str(z["history"]))
        data = CanonicalData(z["angles"], z["Y"], float(z["dpsi"]), gp.meta)
        return gp.condition(data, verbose=verbose)


# ==========================================================================
# Walidacja
# ==========================================================================
def rowwise_pearson(A, B):
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    return np.sum(A * B, axis=1) / np.sqrt(np.sum(A * A, axis=1) * np.sum(B * B, axis=1))


def evaluate(gp: EchordGP, mp, n_val=500, seed=12345, angles=None, verbose=True):
    """Porównanie z prawdziwą symulacją dla losowych (jednorodnych na SO(3)) orientacji.

    Raportuje: korelację Pearsona profilu przewidzianego z prawdziwym (GP i, dla
    odniesienia, najbliższy sąsiad z danych uczących), RMSE, kalibrację niepewności
    oraz zależność jakości od odległości do najbliższego punktu uczącego i od Phi.
    """
    if angles is None:
        angles, _, _ = canonical_angles(random_quaternions(n_val, seed), dedupe_tol_deg=None)
    m = gp.meta
    Ytrue = simulate_canonical(mp, angles, m["coneAngle"], gp.dpsi, m["invert"],
                               m["sigma_conv_deg"]).astype(np.float64)
    Xv = canonical_matrices(angles)
    mean, std = gp.predict_canonical(Xv)
    dist, nn = gp.kernel.min_misorientation(Xv, gp.X_train)

    r_gp = rowwise_pearson(mean, Ytrue)
    r_nn = rowwise_pearson(gp.Y_train[nn].astype(np.float64), Ytrue)
    rmse = np.sqrt(np.mean((mean - Ytrue) ** 2, axis=1))
    z = (Ytrue - mean) / np.maximum(std, 1e-12)[:, None]
    res = dict(r_gp=r_gp, r_nn=r_nn, rmse=rmse, dist_nn_deg=dist, std=std,
               z_sd=float(np.std(z)), cover2=float(np.mean(np.abs(z) < 2)),
               Phi_deg=np.rad2deg(angles[:, 0]))
    if verbose:
        q = lambda x: f"{np.median(x):.3f} [{np.percentile(x, 10):.3f}, {np.percentile(x, 90):.3f}]"
        print(f"[walidacja] {len(angles)} losowych orientacji (mediana [10%, 90%])")
        print(f"  Pearson r  GP: {q(r_gp)}   najbliższy sąsiad: {q(r_nn)}")
        print(f"  RMSE: {q(rmse)}   (sd danych uczących {gp.y_std:.3f})")
        print(f"  odl. do najbliższego punktu uczącego: {q(dist)} °")
        print(f"  kalibracja: sd(z) = {res['z_sd']:.2f} (ideał 1), |z|<2: "
              f"{100 * res['cover2']:.1f}% (ideał ~95%)")
        print("  r_GP wg odległości do najbliższego punktu uczącego:")
        edges = np.percentile(dist, [0, 25, 50, 75, 100])
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (dist >= a) & (dist <= b)
            print(f"    {a:5.2f}–{b:5.2f}°: r_GP = {np.median(r_gp[sel]):.3f}, "
                  f"r_NN = {np.median(r_nn[sel]):.3f}  (n = {sel.sum()})")
        print("  r_GP wg Phi (kąt normalnej próbki od osi z kryształu):")
        for a, b in ((0, 20), (20, 60), (60, 120), (120, 160), (160, 180)):
            sel = (res["Phi_deg"] >= a) & (res["Phi_deg"] < b)
            if sel.any():
                print(f"    Phi {a:3d}–{b:3d}°: r_GP = {np.median(r_gp[sel]):.3f}, "
                      f"odl. NN = {np.median(dist[sel]):.2f}°  (n = {sel.sum()})")
    return res


def check_shift_bookkeeping(mp, coneAngle=40.0, dpsi=0.25, invert=False, sigma_conv_deg=0.0,
                            n=50, seed=7):
    """Sprawdza wzór (1): profil z kanonicznego + przesunięcie vs. bezpośrednie cone_profiles.

    Losowe orientacje, phi0 i oba zwroty sense. Zwraca minimalną korelację i
    maksymalny błąd względny (niezerowy tylko przez interpolację liniową po psi).
    """
    rng = np.random.default_rng(seed)
    q = random_quaternions(n, seed)
    ang, phi1, _ = canonical_angles(q, dedupe_tol_deg=None)
    F = simulate_canonical(mp, ang, coneAngle, dpsi, invert, sigma_conv_deg).astype(np.float64)
    worst_r, worst_err = 1.0, 0.0
    for sense in (+1, -1):
        phi0 = rng.uniform(0, 360)
        omega, direct = mp.cone_profiles(q, coneAngle, 2.0, phi0=phi0, sense=sense,
                                         invert=invert, sigma_conv_deg=sigma_conv_deg)
        shifted = profiles_from_canonical(F, dpsi, phi1, omega, phi0, sense)
        worst_r = min(worst_r, rowwise_pearson(shifted, direct).min())
        worst_err = max(worst_err, np.max(np.abs(shifted - direct)) / np.ptp(direct))
    return float(worst_r), float(worst_err)
