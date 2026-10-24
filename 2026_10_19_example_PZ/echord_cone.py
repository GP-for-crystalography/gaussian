"""
echord_cone.py — wycinanie profili eCHORD (przebiegów po stożku) z master patternu EMsoft.

Geometria
---------
Próbka jest pochylona o kąt theta = coneAngle względem wiązki i obracana wokół
własnej normalnej. W układzie próbki (z = normalna, zwrot do źródła elektronów)
kierunek padania wiązki zakreśla stożek o półkącie theta:

    d_s(omega) = (sin th * cos b,  sin th * sin b,  cos th),   b = phi0 + sense * omega

Kierunek w układzie kryształu:  d_c = g @ d_s,  gdzie g to pasywna macierz Bunge
(próbka -> kryształ), ta sama co eu2om w EMsoft. Intensywność to wartość master
patternu w kierunku d_c: kwadratowa projekcja Rosca-Lambert, półkula N (mLPNH)
dla z_c >= 0, S (mLPSH) w przeciwnym razie, interpolacja dwuliniowa jak w EMsoft.

Nie liczymy więc pełnych ECP dla każdej orientacji (jak w pracy Lafonda),
tylko od razu próbkujemy master pattern wzdłuż okręgu — EMECP i tak robi
dokładnie tę samą interpolację, tylko na całym obrazku.

Uwaga o phi1: g @ d_s = Rz(phi2) Rx(Phi) Rz(phi1 - phi0 - sense*omega) @ e_t,
więc zmiana phi1 to wyłącznie cykliczne przesunięcie profilu w omega.
"""

from __future__ import annotations

import posixpath
import warnings

import h5py
import numpy as np
from scipy.ndimage import map_coordinates, spline_filter

_SQRT_PI = np.sqrt(np.pi)
_L = np.sqrt(np.pi / 2.0)  # półbok kwadratu Lamberta dla sfery jednostkowej


# --------------------------------------------------------------------------
# Inspekcja pliku HDF5
# --------------------------------------------------------------------------
def h5_tree(path: str) -> None:
    """Wypisuje strukturę pliku HDF5 (grupy i datasety z kształtami)."""

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name:65s} {str(obj.shape):22s} {obj.dtype}")
        else:
            print(f"{name}/")

    with h5py.File(path, "r") as f:
        f.visititems(visitor)


def _find_all(f: h5py.File, basename: str) -> list[str]:
    hits = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and posixpath.basename(name) == basename:
            hits.append(name)

    f.visititems(visitor)
    return hits


# --------------------------------------------------------------------------
# Projekcje
# --------------------------------------------------------------------------
def sphere_to_lambert(xyz: np.ndarray) -> np.ndarray:
    """Wektory jednostkowe (...,3) -> współrzędne Lamberta (...,2) w [-1, 1].

    Konwencja EMsoft (LambertSphereToSquare podzielone przez sqrt(pi/2)).
    Półkulę wybiera się osobno po znaku z; obie półkule mają to samo (X, Y).
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    q = np.sqrt(np.clip(2.0 * (1.0 - np.abs(z)), 0.0, None))
    region1 = np.abs(y) <= np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        X1 = np.sign(x) * q * (_SQRT_PI / 2.0)
        Y1 = np.sign(x) * q * (2.0 / _SQRT_PI) * np.arctan(y / x)
        X2 = np.sign(y) * q * (2.0 / _SQRT_PI) * np.arctan(x / y)
        Y2 = np.sign(y) * q * (_SQRT_PI / 2.0)
    X = np.nan_to_num(np.where(region1, X1, X2))  # biegun: x = y = 0 -> (0, 0)
    Y = np.nan_to_num(np.where(region1, Y1, Y2))
    return np.stack([X, Y], axis=-1) / _L


def lambert_to_sphere(XY: np.ndarray, north: bool = True) -> np.ndarray:
    """Odwrotność sphere_to_lambert: (...,2) w [-1, 1] -> wektory jednostkowe (...,3)."""
    a = XY[..., 0] * _L
    b = XY[..., 1] * _L
    region1 = np.abs(b) <= np.abs(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = 2.0 * a / np.pi * np.sqrt(np.clip(np.pi - a * a, 0.0, None))
        x1 = r1 * np.cos(b * np.pi / (4.0 * a))
        y1 = r1 * np.sin(b * np.pi / (4.0 * a))
        z1 = 1.0 - 2.0 * a * a / np.pi
        r2 = 2.0 * b / np.pi * np.sqrt(np.clip(np.pi - b * b, 0.0, None))
        x2 = r2 * np.sin(a * np.pi / (4.0 * b))
        y2 = r2 * np.cos(a * np.pi / (4.0 * b))
        z2 = 1.0 - 2.0 * b * b / np.pi
    x = np.nan_to_num(np.where(region1, x1, x2))
    y = np.nan_to_num(np.where(region1, y1, y2))
    z = np.where(region1, z1, z2)
    if not north:
        z = -z
    return np.stack([x, y, z], axis=-1)


def stereo_to_sphere(XY: np.ndarray) -> np.ndarray:
    """Rzut stereograficzny (dysk jednostkowy) -> półkula północna."""
    X, Y = XY[..., 0], XY[..., 1]
    r2 = X * X + Y * Y
    return np.stack([2 * X, 2 * Y, 1 - r2], axis=-1) / (1 + r2)[..., None]


# --------------------------------------------------------------------------
# Orientacje i geometria stożka
# --------------------------------------------------------------------------
def bunge_to_matrix(eulers, degrees: bool = True) -> np.ndarray:
    """Kąty Bunge (N,3) -> pasywne macierze obrotu g (N,3,3), próbka -> kryształ."""
    e = np.atleast_2d(np.asarray(eulers, dtype=float))
    if degrees:
        e = np.deg2rad(e)
    c1, s1 = np.cos(e[:, 0]), np.sin(e[:, 0])
    c, s = np.cos(e[:, 1]), np.sin(e[:, 1])
    c2, s2 = np.cos(e[:, 2]), np.sin(e[:, 2])
    g = np.empty((e.shape[0], 3, 3))
    g[:, 0, 0] = c1 * c2 - s1 * s2 * c
    g[:, 0, 1] = s1 * c2 + c1 * s2 * c
    g[:, 0, 2] = s2 * s
    g[:, 1, 0] = -c1 * s2 - s1 * c2 * c
    g[:, 1, 1] = -s1 * s2 + c1 * c2 * c
    g[:, 1, 2] = c2 * s
    g[:, 2, 0] = s1 * s
    g[:, 2, 1] = -c1 * s
    g[:, 2, 2] = c
    return g


def cone_directions(coneAngle, angleStep, phi0=0.0, sense=+1, nSteps=None):
    """Kierunki padania wiązki w układzie próbki dla kolejnych kroków obrotu.

    Zwraca (omega [deg] kształtu (M,), d_s kształtu (M,3)).
    phi0  — azymut pochylenia w układzie próbki dla omega = 0 [deg]
    sense — +1 / -1, w którą stronę azymut wiązki rośnie z obrotem stolika
    """
    if nSteps is None:
        nSteps = int(round(360.0 / angleStep))
    omega = np.arange(nSteps) * angleStep
    th = np.deg2rad(coneAngle)
    b = np.deg2rad(phi0 + sense * omega)
    d = np.stack(
        [np.sin(th) * np.cos(b), np.sin(th) * np.sin(b), np.full_like(b, np.cos(th))],
        axis=-1,
    )
    return omega, d


def _tangent_basis(d: np.ndarray):
    """Dwa wektory jednostkowe prostopadłe do d (i do siebie)."""
    helper = np.where(np.abs(d[..., 2:3]) < 0.9, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    t1 = np.cross(helper, d)
    t1 /= np.linalg.norm(t1, axis=-1, keepdims=True)
    t2 = np.cross(d, t1)
    return t1, t2


def convergence_directions(d: np.ndarray, sigma_deg: float, n_quad: int = 3):
    """Rozmycie kierunków gaussowsko (zbieżność wiązki), kwadratura Gaussa-Hermite'a 2D.

    d: (M,3) -> (M,Q,3) kierunki oraz wagi (Q,), Q = n_quad**2, suma wag = 1.
    """
    x, w = np.polynomial.hermite.hermgauss(n_quad)
    u, v = np.meshgrid(x, x, indexing="ij")
    W = (np.outer(w, w) / np.pi).ravel()
    s = np.sqrt(2.0) * np.deg2rad(sigma_deg)
    t1, t2 = _tangent_basis(d)
    dd = (
        d[:, None, :]
        + s * u.ravel()[None, :, None] * t1[:, None, :]
        + s * v.ravel()[None, :, None] * t2[:, None, :]
    )
    dd /= np.linalg.norm(dd, axis=-1, keepdims=True)
    return dd, W


# --------------------------------------------------------------------------
# Master pattern
# --------------------------------------------------------------------------
def _reduce_master(arr, ekevs, energy, eweights):
    """Sprowadza mLPNH/mLPSH/masterSPNH do 2D: sumuje po pozycjach atomów,
    wybiera/uśrednia energię. Kształt z h5py: (..., ny, nx)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    lead = arr.shape[:-2]
    e_axis = None
    if ekevs is not None and ekevs.size > 1:
        for ax in reversed(range(len(lead))):
            if lead[ax] == ekevs.size:
                e_axis = ax
                break
    if e_axis is None:  # brak osi energii (np. master ECP) -> suma po reszcie
        return arr.reshape(-1, *arr.shape[-2:]).sum(axis=0)
    other = tuple(a for a in range(len(lead)) if a != e_axis)
    if other:
        arr = arr.sum(axis=other)  # teraz (nE, ny, nx)
    if isinstance(energy, (int, np.integer)):
        return arr[energy]
    if energy == "max":
        return arr[int(np.argmax(ekevs))]
    if energy == "weighted":
        w = eweights if eweights is not None else np.ones(arr.shape[0])
        w = np.asarray(w, dtype=float) / np.sum(w)
        return np.tensordot(w, arr, axes=1)
    raise ValueError("energy musi być 'max', 'weighted' albo indeksem kanału energii")


class MasterPattern:
    """Master pattern EMsoft (EBSD lub ECP) w kwadratowej projekcji Lamberta.

    Parametry
    ---------
    path       : plik .h5 z wynikiem EMEBSDmaster(OpenCL) / EMECPmaster
    energy     : 'max' (kanał najwyższej energii = energia wiązki, domyślnie),
                 'weighted' (średnia ważona histogramem energii z Monte Carlo)
                 albo indeks kanału energii
    transpose, flip_x, flip_y : korekta orientacji tablicy, gdyby
                 validate_with_stereo() pokazało, że jest potrzebna
    order      : 1 = dwuliniowa (jak EMsoft), 3 = splajn kubiczny
    """

    def __init__(self, path, energy="max", transpose=False, flip_x=False,
                 flip_y=False, order=1):
        self.path = path
        self.energy = energy
        self.order = order

        with h5py.File(path, "r") as f:
            nh_hits, sh_hits = _find_all(f, "mLPNH"), _find_all(f, "mLPSH")
            if not nh_hits or not sh_hits:
                raise KeyError(f"Brak mLPNH/mLPSH w {path}; sprawdź h5_tree(path).")
            if len(nh_hits) > 1:
                warnings.warn(f"Kilka mLPNH w pliku: {nh_hits}; biorę pierwszy.")
            self.nh_path, self.sh_path = nh_hits[0], sh_hits[0]
            grp = posixpath.dirname(self.nh_path)
            self.group = grp

            ek_path = grp + "/EkeVs"
            self.ekevs = np.atleast_1d(f[ek_path][()]).astype(float) if ek_path in f else None

            self.eweights = None
            acc = _find_all(f, "accum_e")
            if acc and self.ekevs is not None:
                a = np.asarray(f[acc[0]][()], dtype=float)
                axes = [i for i, n in enumerate(a.shape) if n == self.ekevs.size]
                if axes:
                    ax = axes[-1] if a.shape[-1] == self.ekevs.size else axes[0]
                    self.eweights = a.sum(axis=tuple(i for i in range(a.ndim) if i != ax))
            if energy == "weighted" and self.eweights is None:
                warnings.warn("Nie znalazłem accum_e pasującego do EkeVs — wagi równe.")

            self.raw_shape = f[self.nh_path].shape
            NH = _reduce_master(f[self.nh_path][()], self.ekevs, energy, self.eweights)
            SH = _reduce_master(f[self.sh_path][()], self.ekevs, energy, self.eweights)

            sp = _find_all(f, "masterSPNH")
            self._sp_raw = f[sp[0]][()] if sp else None

        if NH.shape[0] != NH.shape[1] or NH.shape[0] % 2 == 0:
            raise ValueError(f"Nieoczekiwany kształt obrazu Lamberta: {NH.shape}")
        self.npx = (NH.shape[0] - 1) // 2
        self._raw = (NH, SH)
        self._set_orient(transpose, flip_x, flip_y)

    # ---------------------------------------------------------------- utils
    def _set_orient(self, transpose=False, flip_x=False, flip_y=False):
        """Ustawia orientację tablic względem surowego odczytu z pliku."""
        self._orient = (transpose, flip_x, flip_y)
        imgs = []
        for img in self._raw:
            if transpose:
                img = img.T
            if flip_x:
                img = img[:, ::-1]
            if flip_y:
                img = img[::-1, :]
            imgs.append(np.ascontiguousarray(img))
        self.NH, self.SH = imgs
        if self.order > 1:  # współczynniki splajnu liczone raz, nie przy każdym wywołaniu
            self._NHc = spline_filter(self.NH, order=self.order)
            self._SHc = spline_filter(self.SH, order=self.order)
        else:
            self._NHc, self._SHc = self.NH, self.SH

    def __repr__(self):
        ek = "brak" if self.ekevs is None else (
            f"{self.ekevs.min():.1f}-{self.ekevs.max():.1f} keV ({self.ekevs.size} kanałów)")
        return (f"MasterPattern('{self.path}')\n"
                f"  dataset : {self.nh_path}, surowy kształt {self.raw_shape}\n"
                f"  npx     : {self.npx}  (obraz {2 * self.npx + 1}^2)\n"
                f"  energie : {ek}, wybór: {self.energy}\n"
                f"  accum_e : {'tak' if self.eweights is not None else 'nie'}, "
                f"masterSPNH: {'tak' if self._sp_raw is not None else 'nie'}")

    # --------------------------------------------------------------- lookup
    def lookup(self, dc: np.ndarray) -> np.ndarray:
        """Intensywność master patternu dla kierunków w układzie kryształu (...,3)."""
        dc = np.asarray(dc, dtype=float)
        shape = dc.shape[:-1]
        dc = dc.reshape(-1, 3)
        dc = dc / np.linalg.norm(dc, axis=1, keepdims=True)
        xy = sphere_to_lambert(dc)
        col = (xy[:, 0] + 1.0) * self.npx  # X Lamberta -> kolumna (indeks Fortranu i)
        row = (xy[:, 1] + 1.0) * self.npx  # Y Lamberta -> wiersz  (indeks Fortranu j)
        north = dc[:, 2] >= 0.0
        out = np.empty(dc.shape[0])
        kw = dict(order=self.order, mode="nearest", prefilter=False)
        for mask, img in ((north, self._NHc), (~north, self._SHc)):
            if mask.any():
                out[mask] = map_coordinates(img, [row[mask], col[mask]], **kw)
        return out.reshape(shape)

    # ------------------------------------------------------------- profiles
    def cone_profiles(self, eulers, coneAngle, angleStep, phi0=0.0, sense=+1,
                      sigma_conv_deg=0.0, n_quad=3, degrees=True, chunk=1024):
        """Profile eCHORD dla N orientacji.

        eulers : (N,3) kąty Bunge (phi1, Phi, phi2)
        Zwraca (omega (M,), profile (N, M)).
        """
        omega, ds = cone_directions(coneAngle, angleStep, phi0, sense)
        if sigma_conv_deg > 0:
            ds_all, w = convergence_directions(ds, sigma_conv_deg, n_quad)
        else:
            ds_all, w = ds[:, None, :], np.ones(1)
        G = bunge_to_matrix(eulers, degrees=degrees)
        out = np.empty((G.shape[0], omega.size))
        for s in range(0, G.shape[0], chunk):
            dc = np.einsum("nij,mqj->nmqi", G[s:s + chunk], ds_all)
            out[s:s + chunk] = self.lookup(dc) @ w
        return omega, out

    def cone_path_crystal(self, euler, coneAngle, angleStep, phi0=0.0, sense=+1,
                          degrees=True):
        """Kierunki ścieżki w układzie kryształu (M,3) — do wykresów/diagnostyki."""
        _, ds = cone_directions(coneAngle, angleStep, phi0, sense)
        g = bunge_to_matrix(euler, degrees=degrees)[0]
        return ds @ g.T

    # ----------------------------------------------------------- validation
    def validate_with_stereo(self):
        """Porównuje odtworzony przez nas rzut stereograficzny z masterSPNH,
        który policzył EMsoft. Zwraca korelację dla 8 wariantów orientacji
        tablicy (transpose, flip_x, flip_y). Najlepszy powinien być (F, F, F).
        Dla kryształów m-3m wszystkie warianty dadzą ~1 (to symetrie) i wtedy
        wybór jest obojętny."""
        if self._sp_raw is None:
            print("Brak masterSPNH w pliku — pomijam test.")
            return None
        sp = _reduce_master(self._sp_raw, self.ekevs, self.energy, self.eweights)
        n = sp.shape[0]
        nsp = (n - 1) // 2
        i = np.arange(-nsp, nsp + 1) / nsp
        XX, YY = np.meshg540brid(i, i)  # XX zmienia się wzdłuż kolumn (indeks i Fortranu)
        inside = XX ** 2 + YY ** 2 < 0.98
        d = stereo_to_sphere(np.stack([XX, YY], -1))[inside]

        saved = self._orient
        results = {}
        for t in (False, True):
            for fx in (False, True):
                for fy in (False, True):
                    self._set_orient(t, fx, fy)
                    ours = self.lookup(d)
                    results[(t, fx, fy)] = float(np.corrcoef(ours, sp[inside])[0, 1])
        self._set_orient(*saved)
        for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
            print(f"  transpose={k[0]!s:5s} flip_x={k[1]!s:5s} flip_y={k[2]!s:5s}  r = {v:.5f}")
        return results

    # ----------------------------------------------------------------- plot
    def plot_cone(self, euler, coneAngle, angleStep, phi0=0.0, sense=+1, ax=None):
        """Master pattern (N i S w Lambercie) z naniesioną ścieżką stożka."""
        import matplotlib.pyplot as plt

        dc = self.cone_path_crystal(euler, coneAngle, angleStep, phi0, sense)
        dc /= np.linalg.norm(dc, axis=1, keepdims=True)
        xy = sphere_to_lambert(dc)
        north = dc[:, 2] >= 0
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(10, 5))
        ext = (-1, 1, -1, 1)
        for a, img, mask, title in ((ax[0], self.NH, north, "półkula N"),
                                    (ax[1], self.SH, ~north, "półkula S")):
            a.imshow(img, origin="lower", extent=ext, cmap="gray")
            a.plot(xy[mask, 0], xy[mask, 1], ".", ms=3, color="tab:red")
            if mask.any():
                k0 = np.flatnonzero(mask)[0]
                a.plot(*xy[k0], "o", color="tab:orange", ms=6)
            a.set_title(title)
            a.set_xlabel("X (Lambert)")
            a.set_ylabel("Y (Lambert)")
        return ax
