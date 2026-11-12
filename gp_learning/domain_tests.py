"""
domain_tests.py — testy hipotez o domenach (paski vs matryca) na wynikach dopasowania mapy.

Wszystko tu jest odporne na to, że niedopasowanie modelu do danych jest duże (r ~ 0.6–0.8):
zamiast ufać pojedynczym pikselom, porównujemy GRUPY pikseli (paski vs matryca), regresujemy
dowód na ułamek pasków w pikselu i liczymy niepewności blokowym bootstrapem (bloki w poprzek
pasków, żeby uwzględnić korelację przestrzenną).

Zawartość:
  * ułamek pasków bez modelu (z gładkiego pola PC1): stripe_fraction, stripe_masks,
  * dowód o wariancie vs ułamek pasków: evidence_vs_fraction (ekstrapolacja do czystego paska),
  * różnica obrotu sieci paski − matryca z kowariancją i test osi: group_difference,
    axis_hypothesis_test, cube_axes_sample,
  * różnicowe dopasowanie PROFILI grup (obrót vs inny wariant): group_profile, differential_fit,
  * geometria klas (tło czy paski): class_geometry, eta_squared,
  * polaryzacja: polarity_pairs, pair_detectability,
  * dane bez binningu: cross_stripe_profiles, fit_series,
  * geometria i tetragonalność: twin_hypotheses, c_over_a_from_rotation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation
from scipy.stats import chi2, skew

import orient_fit as of
import real_data as rd

CRYSTAL_AXES = ["[100]", "[010]", "[001]"]


# ==========================================================================
# Ułamek pasków (bez modelu) i maski
# ==========================================================================
def stripe_fraction(field, valid=None, q_hi=0.99):
    """Model-free "ułamek pasków" z gładkiego pola (np. średnia GP z PC1).

    Znak jest ustalany tak, żeby paski (mniejszość, długi ogon) miały wartości dodatnie;
    mediana (matryca) = 0, kwantyl q_hi = 1, obcięte do [0, 1]. To skala orientacyjna:
    nie wiemy, jak wygląda czysty pasek (rozcieńczenie przez binning).
    """
    f = np.asarray(field, float).copy()
    v = np.isfinite(f) if valid is None else (np.asarray(valid, bool) & np.isfinite(f))
    if skew(f[v]) < 0:
        f = -f
    f0, f1 = np.median(f[v]), np.quantile(f[v], q_hi)
    return np.clip((f - f0) / (f1 - f0), 0.0, 1.0)


def stripe_masks(frac, valid, core=0.6, bg=0.08):
    """(rdzeń pasków, czysta matryca) jako maski bool."""
    valid = np.asarray(valid, bool)
    return valid & (frac >= core), valid & (frac <= bg)


def block_ids(u, width):
    """Numery bloków wzdłuż współrzędnej u (w poprzek pasków); szerokość ~ okres pasków."""
    u = np.asarray(u, float)
    return np.floor((u - np.nanmin(u)) / width).astype(int)


def _blocks(block_id, idx):
    return {b: idx[block_id[idx] == b] for b in np.unique(block_id[idx])}


# ==========================================================================
# Dowód o wariancie vs ułamek pasków
# ==========================================================================
def evidence_vs_fraction(dL, frac, block_id, ok, n_boot=400, seed=0):
    """Regresja dL = b0 + b1 * frac z blokowym bootstrapem.

    dL — różnica log-wiarygodności wariantu j względem wariantu matrycy (na piksel).
    b0 — przewidywanie dla matrycy (frac = 0), b0 + b1 — dla czystego paska (frac = 1).
    Jeśli paski są INNYM wariantem, b1 > 0 (dla j = ich wariantu) i b0 + b1 > 0 przy b0 < 0.
    """
    ok = np.asarray(ok, bool) & np.isfinite(dL) & np.isfinite(frac)
    idx = np.flatnonzero(ok)
    blocks = _blocks(block_id, idx)
    keys = np.array(list(blocks))

    def fit(ii):
        A = np.column_stack([np.ones(len(ii)), frac[ii]])
        return np.linalg.lstsq(A, dL[ii], rcond=None)[0]

    c0 = fit(idx)
    rng = np.random.default_rng(seed)
    B = np.empty((n_boot, 2))
    for i in range(n_boot):
        pick = rng.choice(keys, len(keys), replace=True)
        B[i] = fit(np.concatenate([blocks[k] for k in pick]))
    se = B.std(0, ddof=1)
    return dict(b0=float(c0[0]), b1=float(c0[1]), pure=float(c0.sum()),
                se_b0=float(se[0]), se_b1=float(se[1]), se_pure=float(B.sum(1).std(ddof=1)),
                n=int(len(idx)), n_blocks=int(len(keys)))


# ==========================================================================
# Różnica obrotu sieci między grupami + test osi
# ==========================================================================
def group_difference(R, mask_a, mask_b, block_id, n_boot=500, seed=0):
    """Mediana R w grupie a minus w grupie b (R (P,3) [deg]) i kowariancja z blokowego
    bootstrapu (3,3) [deg^2]. Zwraca (d, cov, n_a, n_b)."""
    R = np.asarray(R, float)
    fin = np.isfinite(R).all(1)
    ma, mb = np.asarray(mask_a, bool) & fin, np.asarray(mask_b, bool) & fin
    d = np.median(R[ma], 0) - np.median(R[mb], 0)
    idx = np.flatnonzero(ma | mb)
    blocks = _blocks(block_id, idx)
    keys = np.array(list(blocks))
    rng = np.random.default_rng(seed)
    D = []
    for _ in range(n_boot):
        sel = np.concatenate([blocks[k] for k in rng.choice(keys, len(keys), replace=True)])
        a, b = sel[ma[sel]], sel[mb[sel]]
        if len(a) >= 3 and len(b) >= 3:
            D.append(np.median(R[a], 0) - np.median(R[b], 0))
    return d, np.cov(np.array(D).T), int(ma.sum()), int(mb.sum())


def cube_axes_sample(G):
    """Osie sześcianu [100], [010], [001] w układzie próbki (wiersze G) i ich opisy."""
    G = np.asarray(G, float).reshape(3, 3)
    names = []
    for k in range(3):
        j = int(np.argmax(np.abs(G[k])))
        ang = np.degrees(np.arccos(np.clip(abs(G[k, j]), 0, 1)))
        names.append(f"{CRYSTAL_AXES[k]} ≈ {'+' if G[k, j] > 0 else '−'}{'xyz'[j]}_próbki ({ang:.1f}°)")
    return G.copy(), names


def axis_hypothesis_test(d, cov, axes, names=None):
    """Czy wektor d (3,) [deg] jest równoległy do jednej z osi `axes` (K,3)?

    Dla każdej osi a: najlepsza amplituda A (znak dowolny) minimalizująca
    chi2 = (d - A a)^T C^-1 (d - A a), 2 stopnie swobody (kierunek jest zadany, amplituda
    swobodna). attrs: chi2_zero / p_zero — test d = 0 (3 stopnie swobody).
    """
    d = np.asarray(d, float)
    C = np.asarray(cov, float) + 1e-18 * np.eye(3)
    Ci = np.linalg.inv(C)
    rows = []
    for k, a in enumerate(np.asarray(axes, float)):
        a = a / np.linalg.norm(a)
        A = (a @ Ci @ d) / (a @ Ci @ a)
        r = d - A * a
        x2 = float(r @ Ci @ r)
        rows.append(dict(oś=names[k] if names is not None else str(k), amplituda_deg=float(A),
                         kąt_do_d=float(np.degrees(np.arccos(np.clip(abs(d @ a) / np.linalg.norm(d), 0, 1)))),
                         chi2=x2, p=float(chi2.sf(x2, 2))))
    df = pd.DataFrame(rows)
    x0 = float(d @ Ci @ d)
    df.attrs["chi2_zero"], df.attrs["p_zero"] = x0, float(chi2.sf(x0, 3))
    return df


# ==========================================================================
# Różnicowe dopasowanie profili grup: obrót sieci czy inny wariant?
# ==========================================================================
def group_profile(Y, valid, mask, min_frac=0.5):
    """Średni z-score profil grupy pikseli (ważony ważnymi klatkami).

    Y, valid: (H, W, n); mask: (H, W) albo (H*W,). Zwraca (profil (n,), maska klatek (n,),
    liczba pikseli). Klatki, w których ważnych jest < min_frac pikseli, są odrzucane.
    """
    H, W, n = Y.shape
    m = np.asarray(mask, bool).reshape(H, W)
    Z = rd._zscore_masked(Y, valid)
    cnt = valid[m].sum(0)
    prof = Z[m].sum(0) / np.maximum(cnt, 1)
    return prof, cnt >= min_frac * m.sum(), int(m.sum())


def differential_fit(model, g_m, y_a, y_b, mask=None, spec=0, variants=None, phi0=0.0, sense=+1,
                     h_deg=0.1):
    """Czy różnica profili dwóch grup to MAŁY OBRÓT sieci, czy INNY WARIANT?

    D = ŷ_a − ŷ_b (profile po rzucie spec i normalizacji do jednostkowej normy; wspólne
    niedopasowanie modelu w dużej części się znosi). Hipotezy (grupa a względem grupy b o
    orientacji g_m):
      R:  D ≈ J δ              — obrót δ (3 składowe, układ próbki), J = pochodna profilu modelu,
      V_j: D ≈ s (μ̂(V_j g_m) − μ̂(g_m)), s ∈ R — inny wariant; s ~ ułamek domeny w grupie,
      RV_j: oba naraz.
    expl = 1 − |reszta|² / |D|² (ułamek wyjaśnionej różnicy). Obrót w pierwszym rzędzie.
    Zwraca słownik: D_norm, delta_deg (3), expl_rot, tabela wariantów (DataFrame),
    best_variant, expl_variant, expl_joint.
    """
    variants = of.pseudo_variants() if variants is None else variants
    ya0, yb0 = np.asarray(y_a, float), np.asarray(y_b, float)
    n = len(ya0)
    keep = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    P = of.make_projector(n, spec, keep)
    ya, yb = P @ ya0[keep], P @ yb0[keep]
    ya, yb = ya / np.linalg.norm(ya), yb / np.linalg.norm(yb)
    D = ya - yb
    DD = float(D @ D)

    def mu_hat(G):
        m = model.profiles(np.asarray(G).reshape(-1, 3, 3), phi0, sense)[:, keep] @ P.T
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    h = np.deg2rad(h_deg)
    I3 = np.eye(3)
    Gp = np.einsum("ij,bjk->bik", g_m, Rotation.from_rotvec(h * I3).as_matrix())
    Gm = np.einsum("ij,bjk->bik", g_m, Rotation.from_rotvec(-h * I3).as_matrix())
    J = ((mu_hat(Gp) - mu_hat(Gm)) / (2 * h)).T                      # (n_keep, 3) na radian
    delta = np.linalg.lstsq(J, D, rcond=None)[0]
    res = D - J @ delta
    expl_rot = 1.0 - float(res @ res) / DD

    base = mu_hat(g_m[None])[0]
    Vg = np.einsum("vij,jk->vik", variants, g_m)
    M = mu_hat(Vg) - base
    rows = []
    for j in range(1, len(variants)):
        m = M[j]
        s = float(D @ m / (m @ m))
        r1 = D - s * m
        A = np.column_stack([m, J])
        c = np.linalg.lstsq(A, D, rcond=None)[0]
        r2 = D - A @ c
        rows.append(dict(wariant=f"V{j}", s=s, expl=1 - float(r1 @ r1) / DD,
                         expl_z_obrotem=1 - float(r2 @ r2) / DD,
                         delta_deg_z_wariantem=np.rad2deg(c[1:])))
    tab = pd.DataFrame(rows)
    k = int(tab.expl.values.argmax())
    return dict(D_norm=float(np.sqrt(DD)), delta_deg=np.rad2deg(delta), expl_rot=expl_rot,
                table=tab, best_variant=tab.wariant[k], expl_variant=float(tab.expl[k]),
                s_best=float(tab.s[k]), expl_joint=float(tab.expl_z_obrotem[k]), n_keep=int(keep.sum()))


# ==========================================================================
# Geometria klas: tło czy paski
# ==========================================================================
def class_geometry(labels, ignore=-1):
    """Dla każdej klasy: liczba pikseli, udział sąsiadów (4) tej samej klasy, typowa szerokość
    (2 x średnia odległość od brzegu) i największy wpisany promień. Paski: wąskie (szerokość ~ 2–4 px),
    tło / regiony: szerokie."""
    L = np.asarray(labels)
    cross = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    rows = []
    for c in np.unique(L[L != ignore]):
        m = L == c
        dist = ndi.distance_transform_edt(m)
        nb_same = ndi.convolve(m.astype(float), cross, mode="constant")[m]
        rows.append(dict(klasa=int(c), n=int(m.sum()), sąsiedzi_tej_samej_klasy=float(nb_same.mean() / 4),
                         szerokość_typowa_px=float(2 * dist[m].mean()), max_promień_px=float(dist.max())))
    return pd.DataFrame(rows).set_index("klasa")


def eta_squared(labels, field, ok):
    """Udział wariancji pola wyjaśniony samymi etykietami klas (η²)."""
    ok = np.asarray(ok, bool) & np.isfinite(field)
    l, f = np.asarray(labels)[ok], np.asarray(field, float)[ok]
    tot = float(np.sum((f - f.mean()) ** 2))
    bet = sum(float((l == c).sum() * (f[l == c].mean() - f.mean()) ** 2) for c in np.unique(l))
    return bet / tot if tot > 0 else np.nan


# ==========================================================================
# Polaryzacja: gdzie żyje różnica i czy model ją odtwarza
# ==========================================================================
def polarity_pairs(variants, g_ref, tol=0.5):
    """Pary wariantów (i, j) o tej samej osi c i przeciwnej polaryzacji: dezorientacja
    ≈ 0 modulo D4 (obroty właściwe grupy 4/mmm), ale ≠ 0 modulo C4."""
    G = np.einsum("vij,jk->vik", variants, np.asarray(g_ref).reshape(3, 3))
    D4 = of.d4_ops()
    out = []
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            m4 = of.misorientation_deg(G[i], G[j], D4)[0]
            mc = of.misorientation_deg(G[i], G[j])[0]
            if m4 < tol and mc > 10:
                out.append((i, j))
    return out


def pair_detectability(Zs, pairs, coherence, bands):
    """Widmo różnicy profili par wariantów po pasmach harmonik.

    Zs (V,n) jednostkowe profile symulowanych wariantów (po tym samym rzucie co dane),
    coherence — zgodność model–pomiar w pasmach (z tabeli sekcji 11, ta sama kolejność co
    bands). udział_w_pasmach: ułamek energii różnicy w paśmie; wskaźnik = Σ udział × zgodność²
    (ile różnicy leży tam, gdzie model zgadza się z pomiarem)."""
    w = np.clip(np.asarray(coherence, float), 0, None) ** 2
    rows = []
    for i, j in pairs:
        F = np.abs(np.fft.rfft(Zs[i] - Zs[j])) ** 2
        e = np.array([F[lo:hi + 1].sum() for lo, hi in bands]) / F.sum()
        rows.append(dict(para=f"V{i}–V{j}", **{f"{lo}–{hi}": ek for (lo, hi), ek in zip(bands, e)},
                         wskaźnik_wykrywalności=float(e @ w)))
    return pd.DataFrame(rows).set_index("para")


# ==========================================================================
# Dane bez binningu: profil w poprzek pasków
# ==========================================================================
def cross_stripe_profiles(Y, valid, alpha_deg, center=None, half_length=None, du=1.0,
                          min_pixels=15):
    """Uśrednia z-score'owane profile pikseli w przedziałach współrzędnej u w poprzek pasków.

    alpha_deg — kierunek PASKÓW (jak w GP: od osi kolumn ku osi wierszy, wiersze w dół);
    t = wzdłuż paska, u = w poprzek: u = −sin(a)(x−x0) + cos(a)(y−y0), x = kolumna, y = wiersz.
    half_length — połowa długości odcinka wzdłuż pasków (None = cała mapa).
    Zwraca (u_centers (K,), profiles (K,n), masks (K,n), counts (K,)).
    """
    H, W, n = Y.shape
    a = np.deg2rad(alpha_deg)
    x0, y0 = ((W - 1) / 2, (H - 1) / 2) if center is None else center
    ii, jj = np.mgrid[0:H, 0:W]
    x, y = jj - x0, ii - y0
    t = np.cos(a) * x + np.sin(a) * y
    u = -np.sin(a) * x + np.cos(a) * y
    inside = np.ones((H, W), bool) if half_length is None else (np.abs(t) <= half_length)
    inside &= valid.any(-1)                       # piksele bez żadnej ważnej klatki (artefakty) nie liczą się
    Z = rd._zscore_masked(Y, valid)
    edges = np.arange(np.floor(u[inside].min()), np.ceil(u[inside].max()) + du, du)
    idx = np.digitize(u, edges) - 1
    cen, prof, msk, cnt = [], [], [], []
    for b in range(len(edges) - 1):
        sel = inside & (idx == b)
        ns = int(sel.sum())
        if ns < min_pixels:
            continue
        c = valid[sel].sum(0)
        prof.append(Z[sel].sum(0) / np.maximum(c, 1))
        msk.append(c >= 0.5 * ns)
        cnt.append(ns)
        cen.append(0.5 * (edges[b] + edges[b + 1]))
    return np.array(cen), np.array(prof), np.array(msk), np.array(cnt)


def fit_series(model, g_ref, profiles, masks, spec, variants, vb, phi0=0.0, sense=+1, min_valid=60):
    """Dla każdego profilu serii: 6 wariantów dopracowanych z V g_ref; obrót sieci względem
    wariantu vb (dx, dy, dz [deg]), r_best, najlepszy wariant, ΔlogL(V_j − V_vb) dla wszystkich j.
    Profile z mniej niż min_valid ważnymi klatkami dostają wiersz NaN (best = −1)."""
    rows = []
    G_vb = variants[vb] @ g_ref
    for y, keep in zip(profiles, masks):
        if keep.sum() < min_valid:
            rows.append(dict(dx=np.nan, dy=np.nan, dz=np.nan, r_best=np.nan, best=-1,
                             **{f"dL{j}": np.nan for j in range(len(variants))}))
            continue
        mk = None if keep.all() else keep
        G, R = of.variant_scores(model, y, g_ref, variants, phi0, sense, mk, spec)
        nk = of.projector_dof(len(keep), spec, keep)
        logL = -0.5 * nk * np.log(np.clip(1.0 - R ** 2, 1e-300, None))
        rot = rd.rotation_field(G[vb], G_vb)
        rows.append(dict(dx=rot[0], dy=rot[1], dz=rot[2], r_best=R.max(), best=int(R.argmax()),
                         **{f"dL{j}": logL[j] - logL[vb] for j in range(len(variants))}))
    return pd.DataFrame(rows)


# ==========================================================================
# Geometria ścian i tetragonalność
# ==========================================================================
# płaszczyzna ściany bliźniaczej, osie c obu domen (indeksy osi sześcianu), oś obrotu sieci
_TWIN = [((1, 0, 1), (0, 2), 1), ((1, 0, -1), (0, 2), 1),
         ((0, 1, 1), (1, 2), 0), ((0, 1, -1), (1, 2), 0),
         ((1, 1, 0), (0, 1), 2), ((1, -1, 0), (0, 1), 2)]


def twin_hypotheses(G_m, alpha_img_deg, d=None, cov=None):
    """Tabela hipotez: ściana bliźniacza (domeny 90° w fazie tetragonalnej).

    Dla każdej płaszczyzny: osie c obu domen, przewidywana OŚ obrotu sieci (trzecia oś sześcianu,
    leży w płaszczyźnie ściany) w układzie próbki, przewidywany kąt śladu ściany na powierzchni
    (układ próbki EMsoftu) i niejednoznaczny obrót obraz ↔ próbka psi, jaki z niego wynika
    (psi = alfa_obrazu − ślad; druga kolumna — przy odbiciu obrazu). Jeśli podano d i cov
    (obrót paski − matryca), dodaje chi2 / p dla osi tej hipotezy (2 stopnie swobody).
    """
    G_m = np.asarray(G_m, float).reshape(3, 3)
    traces = rd.wall_traces(G_m, families=[t[0] for t in _TWIN])
    rows = []
    for (hkl, cp, ax), (_, ang, incl) in zip(_TWIN, traces):
        axis = G_m[ax]
        row = dict(płaszczyzna=str(hkl), osie_c_domen=" ↔ ".join(CRYSTAL_AXES[i] for i in cp),
                   oś_obrotu=CRYSTAL_AXES[ax], oś_w_próbce=np.round(axis, 3),
                   ślad_w_próbce_deg=ang, nachylenie_ściany_deg=incl,
                   psi_obraz_deg=(alpha_img_deg - ang) % 180, psi_obraz_odbity_deg=(-alpha_img_deg - ang) % 180)
        if d is not None:
            t = axis_hypothesis_test(d, cov, axis[None])
            row.update(chi2=t.chi2[0], p=t.p[0])
        rows.append(row)
    return pd.DataFrame(rows)


def c_over_a_from_rotation(omega_deg):
    """Stosunek c/a z kąta obrotu sieci między domenami 90°: omega = 2 atan(c/a) − 90°
    (przybliżenie: c/a − 1 ≈ omega w radianach)."""
    return np.tan(np.deg2rad(45.0 + np.asarray(omega_deg, float) / 2))


def rotation_from_c_over_a(ca):
    """Kąt obrotu sieci [deg] między domenami 90° dla danego c/a."""
    return 2 * np.degrees(np.arctan(np.asarray(ca, float))) - 90.0
