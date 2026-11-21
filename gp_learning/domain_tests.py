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


# ==========================================================================
# Struktura pasków bez modelu (sekcja 2d): dwumodalność, test mieszaniny, asymetria stron
# ==========================================================================
def fraction_unclipped(field, valid=None, q_hi=0.99):
    """Jak stripe_fraction, ale BEZ obcięcia do [0, 1] (obcięcie tworzy sztuczne piki w 0 i 1,
    więc do testu dwumodalności się nie nadaje). Mediana = 0, kwantyl q_hi = 1."""
    f = np.asarray(field, float).copy()
    v = np.isfinite(f) if valid is None else (np.asarray(valid, bool) & np.isfinite(f))
    if skew(f[v]) < 0:
        f = -f
    f0, f1 = np.median(f[v]), np.quantile(f[v], q_hi)
    return (f - f0) / (f1 - f0)


def lattice_mask(H, W, step=3):
    """Co `step`-ty piksel w obu kierunkach (zmniejsza korelację przestrzenną próbek w testach
    zakładających niezależność, np. dip test)."""
    m = np.zeros((H, W), bool)
    m[::step, ::step] = True
    return m


def bimodality(x, bins=40, k_max=3, seed=0):
    """Czy rozkład 1-D jest wielomodalny? Zwraca dip (statystyka Hartigana) i p (pakiet `diptest`,
    NaN jeśli go brak), BIC dla mieszanin 1..k_max gaussów 1-D oraz głębokość doliny histogramu
    (min między dwoma największymi pikami / niższy z pików; 1 = brak doliny).
    Uwaga: p zakłada niezależne próbki; piksele sąsiednie są skorelowane (użyj lattice_mask)."""
    from scipy.signal import find_peaks
    from sklearn.mixture import GaussianMixture
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    out = dict(n=len(x), dip=np.nan, p=np.nan)
    try:
        from diptest import diptest
        out["dip"], out["p"] = (float(v) for v in diptest(x))
    except Exception:
        pass
    X = x.reshape(-1, 1)
    out["bic"] = np.array([GaussianMixture(k, n_init=2, random_state=seed).fit(X).bic(X)
                           for k in range(1, k_max + 1)])
    h, e = np.histogram(x, bins=bins)
    hs = np.convolve(h, [0.25, 0.5, 0.25], mode="same")
    pk, _ = find_peaks(np.r_[0, hs, 0])
    pk = pk - 1
    out["valley_ratio"] = np.nan
    if len(pk) >= 2:
        top = np.sort(pk[np.argsort(-hs[pk])[:2]])
        out["valley_ratio"] = float(hs[top[0]:top[1] + 1].min() / hs[top].min())
    out["hist"] = (h, e)
    return out


def _neigh_corr(S):
    """Korelacja wartości z sąsiadem (4-sąsiedztwo) po parach, gdzie obie są skończone."""
    a = np.r_[S[:, :-1][np.isfinite(S[:, :-1]) & np.isfinite(S[:, 1:])],
              S[:-1][np.isfinite(S[:-1]) & np.isfinite(S[1:])]]
    b = np.r_[S[:, 1:][np.isfinite(S[:, :-1]) & np.isfinite(S[:, 1:])],
              S[1:][np.isfinite(S[:-1]) & np.isfinite(S[1:])]]
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else np.nan


def _coherence(R):
    """|średnia reszta|² / średnia |reszta|² po pikselach (szum: ≈ 1/liczba pikseli)."""
    m = R.mean(0)
    return float(m @ m / max(np.mean((R * R).sum(1)), 1e-300))


def mixture_test(Y, valid, frac, ok, core=0.6, bg=0.08, mid=(0.15, 0.85), n_pc=3, seed=0):
    """Czy profile pikseli pośrednich to kombinacja dwóch stanów końcowych?

    Profile końcowe: średnie z-score'owane profile czystej matrycy (frac <= bg) i rdzenia pasków
    (frac >= core). Piksel o pośrednim ułamku jest dopasowywany jako a·P_matryca + b·P_pasek + c.
    (Z-score jest liniowy w intensościach z dokładnością do skalara, więc mieszanina intensywności
    jest kombinacją z-score'ów — nie trzeba dopasowywać surowych.)
    Zwraca energię reszt (po pikselach) dla grup, PCA reszt pikseli pośrednich i czystych,
    mapę wyniku PC1 reszt oraz udział systematyczny reszt coh = |średnia reszta|² / średnia |reszta|²
    (≈ 1/n dla szumu; duży = wszystkie piksele pośrednie mają tę samą brakującą składową = trzeci stan).
    """
    H, W, n = Y.shape
    ok = np.asarray(ok, bool) & valid.any(-1)
    lab = np.full((H, W), -1)
    lab[ok & (frac <= bg)] = 0
    lab[ok & (frac >= core)] = 1
    prof, cnt = rd.class_profiles(Y, valid, lab)
    Z = rd._zscore_masked(Y, valid).reshape(-1, n)
    V = valid.reshape(-1, n)

    def fit(sel):
        idx = np.flatnonzero(sel.ravel())
        coef = np.full((len(idx), 3), np.nan)
        R = np.zeros((len(idx), n))
        e = np.full(len(idx), np.nan)
        for q, p in enumerate(idx):
            k = V[p]
            if k.sum() < 20:
                continue
            A = np.column_stack([prof[0][k], prof[1][k], np.ones(k.sum())])
            c, *_ = np.linalg.lstsq(A, Z[p, k], rcond=None)
            r = Z[p, k] - A @ c
            R[q, k], coef[q], e[q] = r, c, r @ r
        return idx, coef, R, e

    m_mid = ok & (frac > mid[0]) & (frac < mid[1])
    i_mid, c_mid, R_mid, e_mid = fit(m_mid)
    i_m, _, R_m, e_m = fit(lab == 0)
    i_s, _, R_s, e_s = fit(lab == 1)
    good = np.isfinite(e_mid)
    i_mid, c_mid, R_mid, e_mid = i_mid[good], c_mid[good], R_mid[good], e_mid[good]

    def pca(R, nmax):
        Rc = R - R.mean(0)
        U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
        return (S[:n_pc] ** 2) / len(R), U[:, :n_pc] * S[:n_pc], Vt[:n_pc]

    rng = np.random.default_rng(seed)
    pure_R = np.vstack([R_m, R_s])
    sub = rng.choice(len(pure_R), size=min(len(R_mid), len(pure_R)), replace=False)
    eig_mid, sc_mid, comp_mid = pca(R_mid, n_pc)
    eig_pure, sc_pure, _ = pca(pure_R[sub], n_pc)
    smap = np.full(H * W, np.nan); smap[i_mid] = sc_mid[:, 0]; smap = smap.reshape(H, W)
    # tych samych pikseli pure -> mapa wyniku dla grupy czystej
    pmap = np.full(H * W, np.nan)
    pidx = np.r_[i_m, i_s][sub]
    pmap[pidx] = sc_pure[:, 0]
    pmap = pmap.reshape(H, W)
    w_hat = c_mid[:, 1] / np.where(np.abs(c_mid[:, 0] + c_mid[:, 1]) > 1e-9, c_mid[:, 0] + c_mid[:, 1], np.nan)
    fr_mid = frac.reshape(-1)[i_mid]
    ok_w = np.isfinite(w_hat)
    return dict(
        n_mid=int(len(i_mid)), n_matrix=int(len(i_m)), n_core=int(len(i_s)),
        e_mid=e_mid, e_matrix=e_m[np.isfinite(e_m)], e_core=e_s[np.isfinite(e_s)],
        ratio_median=float(np.median(e_mid) / np.median(np.r_[e_m[np.isfinite(e_m)], e_s[np.isfinite(e_s)]])),
        eig_mid=eig_mid, eig_pure=eig_pure, comp_mid=comp_mid,
        score_mid=smap, score_pure=pmap,
        coh_mid=_coherence(R_mid), coh_pure=_coherence(pure_R[sub]),
        w_hat=w_hat, frac_mid=fr_mid,
        w_corr=float(np.corrcoef(w_hat[ok_w], fr_mid[ok_w])[0, 1]) if ok_w.sum() > 10 else np.nan,
        endmember_dist=float(1 - np.corrcoef(prof[0], prof[1])[0, 1]),
    )


def side_asymmetry(field, ok, alpha_deg, du=0.5, width=None, min_prominence=0.25, min_sep=3.0,
                   s_min=1.0):
    """Asymetria profilu pola w poprzek pasków względem środka każdego pasa.

    u = −sin(a)(x−x0) + cos(a)(y−y0), x = kolumna, y = wiersz (jak w cross_stripe_profiles).
    Pole uśredniane po całej długości pasków w przedziałach u (du), środki pasów = maksima
    (dokładność ułamka przedziału, parabola), potem każdy pas ma profil P_p(s) w s = u − u_p
    ∈ [−width, width]. Asymetria pasa: A_p = średnia po |s| ≥ s_min z (P(+s) − P(−s)) / wysokość.
    Zwraca średnią A, jej błąd (po pasach), test t i test znaku, uśredniony profil i D(s) = P(+s) − P(−s).
    Dodatnia asymetria: pole większe po stronie rosnącego u, czyli w kierunku (−sin a, cos a) w
    (kolumna, wiersz). Nie mierzy samego nachylenia ściany — odróżnia tylko symetrię od asymetrii.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    from scipy.stats import binomtest, t as tdist
    H, W = field.shape
    a = np.deg2rad(alpha_deg)
    ii, jj = np.mgrid[0:H, 0:W]
    x, y = jj - (W - 1) / 2, ii - (H - 1) / 2
    u = -np.sin(a) * x + np.cos(a) * y
    ok = np.asarray(ok, bool) & np.isfinite(field)
    uu, ff = u[ok], field[ok]
    edges = np.arange(np.floor(uu.min()), np.ceil(uu.max()) + 2 * du, du)
    idx = np.digitize(uu, edges) - 1
    nb = len(edges) - 1
    cnt = np.bincount(idx, minlength=nb)[:nb]
    sm = np.bincount(idx, ff, minlength=nb)[:nb]
    F = np.where(cnt >= 3, sm / np.maximum(cnt, 1), np.nan)
    cen = edges[:-1] + du / 2
    good = np.isfinite(F)
    Fi = np.interp(cen, cen[good], F[good])
    Fs = gaussian_filter1d(Fi, 0.7 / du)
    rngF = np.nanpercentile(Fs, 99) - np.nanpercentile(Fs, 1)
    pk, _ = find_peaks(Fs, prominence=min_prominence * rngF, distance=max(int(min_sep / du), 1))
    pk = pk[(pk > 0) & (pk < nb - 1)]
    up = []
    for i in pk:                                              # parabola przez 3 punkty
        y0, y1, y2 = Fs[i - 1], Fs[i], Fs[i + 1]
        den = y0 - 2 * y1 + y2
        up.append(cen[i] + (0.5 * (y0 - y2) / den * du if abs(den) > 1e-12 else 0.0))
    up = np.array(up)
    base = float(np.nanmedian(F))
    out = dict(u_centers=cen, F=F, Fs=Fs, u_peaks=up, n_peaks=0)
    if len(up) < 3:
        return out
    period = float(np.median(np.diff(up)))
    w = 0.45 * period if width is None else width
    n = int(np.floor(w / du))
    sedges = np.arange(-n, n + 1) * du
    scen = 0.5 * (sedges[:-1] + sedges[1:])                   # 2n przedziałów, symetrycznie
    P, H_, keep = [], [], []
    for k, upk in enumerate(up):
        if upk - w < uu.min() or upk + w > uu.max():
            continue
        s = u - upk
        sel = ok & (np.abs(s) < n * du - 1e-9)
        j = np.digitize(s[sel], sedges) - 1
        c = np.bincount(j, minlength=2 * n)[:2 * n]
        m = np.bincount(j, field[sel], minlength=2 * n)[:2 * n]
        P.append(np.where(c >= 2, m / np.maximum(c, 1), np.nan))
        H_.append(Fs[pk[k]] - base)
        keep.append(k)
    if len(P) < 3:
        return out
    P, H_ = np.array(P), np.array(H_)
    right = P[:, n:]                                          # s > 0
    left = P[:, :n][:, ::-1]                                  # s < 0 odbite
    D = right - left
    sc_ = scen[n:]
    use = sc_ >= s_min
    A = np.nanmean(D[:, use], axis=1) / H_
    A = A[np.isfinite(A)]
    npk = len(A)
    Am, Ase = float(A.mean()), float(A.std(ddof=1) / np.sqrt(npk))
    tval = Am / Ase if Ase > 0 else np.nan
    out.update(n_peaks=npk, period=period, width=w, A=A, A_mean=Am, A_se=Ase, t=float(tval),
               p_t=float(2 * tdist.sf(abs(tval), npk - 1)) if np.isfinite(tval) else np.nan,
               n_pos=int((A > 0).sum()),
               p_sign=float(binomtest(int((A > 0).sum()), npk, 0.5).pvalue),
               s_all=scen, mean_profile=np.nanmean(P / H_[:, None], axis=0),
               se_profile=np.nanstd(P / H_[:, None], axis=0, ddof=1) / np.sqrt(npk),
               s_pos=sc_, D_mean=np.nanmean(D / H_[:, None], axis=0),
               D_se=np.nanstd(D / H_[:, None], axis=0, ddof=1) / np.sqrt(npk))
    return out


# ==========================================================================
# Sekcja 2e: przesunięcia klatek między ω i test naprzemienności domen (ABAB)
# ==========================================================================
def u_grid(H, W, alpha_deg):
    """Współrzędna w poprzek pasków (jak w side_asymmetry): u = −sin(a)(x−x0) + cos(a)(y−y0)."""
    a = np.deg2rad(alpha_deg)
    ii, jj = np.mgrid[0:H, 0:W]
    return -np.sin(a) * (jj - (W - 1) / 2) + np.cos(a) * (ii - (H - 1) / 2)


def frame_shifts(Y, valid, S, alpha_deg, ok, poly=2, min_pix=200, norm="mean"):
    """Przesunięcie wzoru pasków w poprzek pasków (wzdłuż u) w każdej klatce ω.

    Dla klatki k (z-score po ω, więc zero-średnie): Z_k ≈ wielomian(x, y) + c_k·S + d_k·∂_u S,
    S = gładka mapa pasków (znormalizowana). Z_k(x) = c·S(x − δ) ≈ c·S − c·δ·∂_u S → δ_k = −d_k/c_k [px].
    Wielomian (stopień `poly`) pochłania wolne tło jasności. Mierzalna jest tylko składowa prostopadła
    do pasków (wzdłuż pasków nie ma gradientu). Dodatnie δ: wzór przesunięty ku rosnącemu u.
    Dokładność amplitudy ≈ ±15% (przy δ ~ szerokość paska/2 amplituda jest lekko zawyżona, drugi rząd);
    zbyt mocno wygładzone S obciąża amplitudę, kształt zależności od ω pozostaje.
    Zwraca dict: c, d, se_c, se_d, delta, se_delta, weight (=1/se²), delta_rel (minus średnia ważona), n_pix.
    """
    H, W, n = Y.shape
    a = np.deg2rad(alpha_deg)
    okm = np.asarray(ok, bool) & np.isfinite(S)
    from scipy.ndimage import map_coordinates
    S0 = (S - np.nanmean(S[okm])) / np.nanstd(S[okm])
    S0 = np.where(np.isfinite(S0), S0, 0.0)
    ii, jj = np.mgrid[0:H, 0:W]
    hh = 0.5                                                  # pochodna po u: różnica centralna na splajnie
    gp = map_coordinates(S0, [ii + hh * np.cos(a), jj - hh * np.sin(a)], order=3, mode="nearest")
    gm = map_coordinates(S0, [ii - hh * np.cos(a), jj + hh * np.sin(a)], order=3, mode="nearest")
    G = (gp - gm) / (2 * hh)
    x, y = (jj - (W - 1) / 2) / (W / 2), (ii - (H - 1) / 2) / (H / 2)
    cols = [np.ones_like(x)]
    if poly >= 1:
        cols += [x, y]
    if poly >= 2:
        cols += [x * x, x * y, y * y]
    B = np.stack(cols, -1)
    nb = B.shape[-1]
    if norm == "zscore":
        Z = rd._zscore_masked(Y, valid)
    else:
        w_ = valid.astype(float)
        mu_ = (Y * w_).sum(-1, keepdims=True) / np.maximum(w_.sum(-1, keepdims=True), 1)
        Z = np.where(valid, (Y - mu_) / np.maximum(np.abs(mu_), 1e-12), 0.0)
    out = {k: np.full(n, np.nan) for k in ("c", "d", "se_c", "se_d", "delta", "se_delta", "n_pix")}
    for k in range(n):
        sel = okm & valid[..., k]
        npx = int(sel.sum())
        out["n_pix"][k] = npx
        if npx < min_pix:
            continue
        A = np.column_stack([B[sel], S0[sel], G[sel]])
        z = Z[..., k][sel]
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        r = z - A @ coef
        cov = (r @ r / max(npx - A.shape[1], 1)) * np.linalg.inv(A.T @ A)
        c, d = coef[nb], coef[nb + 1]
        vcc, vdd, vcd = cov[nb, nb], cov[nb + 1, nb + 1], cov[nb, nb + 1]
        out["c"][k], out["d"][k] = c, d
        out["se_c"][k], out["se_d"][k] = np.sqrt(vcc), np.sqrt(vdd)
        if abs(c) > 1e-12:
            out["delta"][k] = -d / c
            v = vdd / c ** 2 + d ** 2 * vcc / c ** 4 - 2 * d * vcd / c ** 3
            out["se_delta"][k] = np.sqrt(max(v, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / out["se_delta"] ** 2
    w = np.where(np.isfinite(w), w, 0.0)
    out["weight"] = w
    good = np.isfinite(out["delta"]) & (w > 0)
    out["delta_rel"] = out["delta"] - (np.average(out["delta"][good], weights=w[good]) if good.any() else 0.0)
    return out


def fit_shift_model(omega_deg, delta, weight=None, harmonics=2, trend=True):
    """Dopasowanie δ(ω) ≈ d0 + Σ_h (A_h cos hω + B_h sin hω) + m·(ω − ω̄)/360° (ważone).
    Okresowość 360° (h = 1) wskazuje mimośrodowy obrót próbki; trend — dryf. Zwraca współczynniki,
    amplitudy i fazy harmonicznych, trend [px/360°], R² ważone, rms reszt, krzywą fit."""
    om = np.asarray(omega_deg, float)
    w = np.deg2rad(om)
    wt = np.ones_like(om) if weight is None else np.asarray(weight, float)
    good = np.isfinite(delta) & np.isfinite(wt) & (wt > 0)
    cols = [np.ones_like(w)]
    for h in range(1, harmonics + 1):
        cols += [np.cos(h * w), np.sin(h * w)]
    if trend:
        cols.append((om - om[good].mean()) / 360.0)
    A = np.column_stack(cols)
    sw = np.sqrt(wt[good])
    coef, *_ = np.linalg.lstsq(A[good] * sw[:, None], np.asarray(delta)[good] * sw, rcond=None)
    fit = A @ coef
    res = np.asarray(delta)[good] - fit[good]
    dbar = np.average(np.asarray(delta)[good], weights=wt[good])
    ss_tot = np.sum(wt[good] * (np.asarray(delta)[good] - dbar) ** 2)
    ss_res = np.sum(wt[good] * res ** 2)
    amp = [float(np.hypot(coef[1 + 2 * h], coef[2 + 2 * h])) for h in range(harmonics)]
    ph = [float(np.degrees(np.arctan2(coef[2 + 2 * h], coef[1 + 2 * h]))) for h in range(harmonics)]
    return dict(coef=coef, fit=fit, amplitude=amp, phase_deg=ph, trend=float(coef[-1]) if trend else 0.0,
                r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                rms=float(np.sqrt(np.average(res ** 2, weights=wt[good]))), n_good=int(good.sum()))


def shift_frames(Y, valid, delta_u, alpha_deg):
    """Koryguje przesunięcie: Y'_k(x) = Y_k(x + δ_k·û), û = (−sin a, cos a) w (kolumna, wiersz).
    Interpolacja liniowa z uwzględnieniem pikseli nieważnych (waga = maska). Zwraca (Y', valid')."""
    from scipy.ndimage import map_coordinates
    H, W, n = Y.shape
    a = np.deg2rad(alpha_deg)
    ii, jj = np.mgrid[0:H, 0:W].astype(float)
    Yc = np.zeros_like(Y, dtype=float)
    vc = np.zeros(Y.shape, bool)
    for k in range(n):
        dk = delta_u[k]
        if not np.isfinite(dk):
            dk = 0.0
        rows, cols = ii + dk * np.cos(a), jj - dk * np.sin(a)
        v = valid[..., k].astype(float)
        num = map_coordinates(np.where(valid[..., k], Y[..., k], 0.0), [rows, cols], order=1, mode="nearest")
        den = map_coordinates(v, [rows, cols], order=1, mode="nearest")
        vc[..., k] = den > 0.99
        Yc[..., k] = np.where(vc[..., k], num / np.maximum(den, 1e-9), 0.0)
    return Yc, vc


def flank_signs(score, alpha_deg, u_peaks, period, wfrac=0.45, s_min=0.3):
    """Znak różnicy wyniku (np. PC1 reszt) między prawą a lewą flanką każdego pasa.
    Stały znak dla kolejnych pasów → przesunięcie / asymetryczna ściana; naprzemienny → domeny ABAB.
    Zwraca (DataFrame po pasach, statystyki: n_pairs, n_same, p_same, mean_prod)."""
    import pandas as pd
    from scipy.stats import binomtest
    u = u_grid(*score.shape, alpha_deg)
    w = wfrac * period
    rows = []
    for k, up in enumerate(u_peaks):
        s = u - up
        L = score[(s < -s_min) & (s > -w)]
        R = score[(s > s_min) & (s < w)]
        L, R = L[np.isfinite(L)], R[np.isfinite(R)]
        if len(L) < 5 or len(R) < 5:
            continue
        rows.append((k, float(up), float(L.mean()), float(R.mean()), len(L), len(R)))
    df = pd.DataFrame(rows, columns=["pas", "u", "L", "R", "nL", "nR"])
    df["roznica"] = df["R"] - df["L"]
    df["znak"] = np.sign(df["roznica"]).astype(int)
    pairs = [(df.znak.iloc[i], df.znak.iloc[i + 1]) for i in range(len(df) - 1)
             if df.pas.iloc[i + 1] - df.pas.iloc[i] == 1]
    n_same = int(sum(a == b for a, b in pairs))
    st = dict(n_pairs=len(pairs), n_same=n_same,
              p_same=float(binomtest(n_same, len(pairs), 0.5).pvalue) if pairs else np.nan,
              mean_prod=float(np.mean([a * b for a, b in pairs])) if pairs else np.nan)
    return df, st


def plateau_profiles(Y, valid, frac, ok, alpha_deg, u_peaks, margin=None, period=None, frac_max=0.15,
                     min_pix=30, seed=0, region=None):
    """Średnie z-score'owane profile 'plateau' (matryca, frac ≤ frac_max) między kolejnymi pasami oraz
    macierz D_ij = 1 − r (profile scentrowane). `region` (maska bool) ogranicza piksele (np. tylko jedna klasa). noise_ij = (h_i + h_j)/4, h = 1 − r między losowymi
    połówkami pikseli przedziału: oszacowanie wkładu szumu do D."""
    H, W, n = Y.shape
    u = u_grid(H, W, alpha_deg)
    m = margin if margin is not None else max(1.0, 0.2 * (period if period else 8.0))
    Z = rd._zscore_masked(Y, valid)
    rng = np.random.default_rng(seed)

    def cr(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-300))

    idx, prof, cnt, h = [], [], [], []
    for i in range(len(u_peaks) - 1):
        sel = np.asarray(ok, bool) & (frac <= frac_max) & (u > u_peaks[i] + m) & (u < u_peaks[i + 1] - m)
        if region is not None:                                   # np. tylko strefa jednej klasy / okno wzdłuż pasków
            sel &= np.asarray(region, bool)
        npx = int(sel.sum())
        if npx < min_pix:
            continue
        Zs, Vs = Z[sel], valid[sel]
        prof.append(Zs.sum(0) / np.maximum(Vs.sum(0), 1))
        perm = rng.permutation(npx)
        a_, b_ = perm[: npx // 2], perm[npx // 2:]
        pa = Zs[a_].sum(0) / np.maximum(Vs[a_].sum(0), 1)
        pb = Zs[b_].sum(0) / np.maximum(Vs[b_].sum(0), 1)
        h.append(1 - cr(pa, pb))
        idx.append(i)
        cnt.append(npx)
    prof = np.array(prof)
    mm = len(prof)
    D = np.array([[1 - cr(prof[a], prof[b]) for b in range(mm)] for a in range(mm)]) if mm else np.zeros((0, 0))
    h = np.array(h)
    noise = (h[:, None] + h[None, :]) / 4 if mm else np.zeros((0, 0))
    return dict(idx=np.array(idx), prof=prof, cnt=np.array(cnt), D=D, noise=noise, h=h)


def abab_regression(pp, power=2):
    """D_ij − noise_ij = a + β·dist^power + γ·[dist nieparzysty], dist = różnica numerów pasów.
    Gładkie wygięcie → β > 0, γ ≈ 0 (D rośnie z odległością). Naprzemienne domeny ABAB → γ > 0:
    pary o nieparzystej odległości różnią się bardziej, niż daje trend. Błędy: jackknife po przedziałach.
    D1, D2 = średnie D (minus szum) dla odległości 1 i 2: ABAB → D1 > D2, trend gładki → D1 < D2."""
    idx, D, noise = pp["idx"], pp["D"], pp["noise"]
    m = len(idx)
    nan = dict(a=np.nan, beta=np.nan, gamma=np.nan, se_beta=np.nan, se_gamma=np.nan, t_gamma=np.nan,
               D1=np.nan, D2=np.nan, n_pairs=0, n_int=m)
    if m < 4:
        return nan
    P = [(a, b, int(idx[b] - idx[a]), D[a, b] - noise[a, b]) for a in range(m) for b in range(a + 1, m)]
    arr = np.array(P, float)

    def fit(sel):
        X = np.column_stack([np.ones(sel.sum()), arr[sel, 2] ** power, (arr[sel, 2] % 2)])
        return np.linalg.lstsq(X, arr[sel, 3], rcond=None)[0]

    c0 = fit(np.ones(len(arr), bool))
    jk = np.array([fit((arr[:, 0] != j) & (arr[:, 1] != j)) for j in range(m)])
    se = np.sqrt((m - 1) / m * ((jk - jk.mean(0)) ** 2).sum(0))
    d1 = arr[arr[:, 2] == 1, 3]
    d2 = arr[arr[:, 2] == 2, 3]
    return dict(a=float(c0[0]), beta=float(c0[1]), gamma=float(c0[2]), se_beta=float(se[1]),
                se_gamma=float(se[2]), t_gamma=float(c0[2] / se[2]) if se[2] > 0 else np.nan,
                D1=float(d1.mean()) if len(d1) else np.nan, D2=float(d2.mean()) if len(d2) else np.nan,
                n_pairs=len(arr), n_int=m)
