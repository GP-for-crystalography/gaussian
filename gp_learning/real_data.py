"""
real_data.py — prawdziwe dane eCHORD: wczytanie stosu, kontrola jakości, analiza bez modelu,
dopasowanie orientacji na całej mapie.

Konwencje:
  * stos z pliku: (n_omega, H, W); tu trzymany jako Y (H, W, n_omega), klatka k <-> omega = k*step,
  * valid (H, W, n_omega): True = wartość użyteczna (nie nasycona, nie zero z rejestracji),
  * piksele adresowane (i, j) = (wiersz, kolumna), płasko: p = i*W + j.

Obserwacja z pierwszego pomiaru (KTN_014): wszystkie piksele mają niemal ten sam profil
(r z profilem medianowym ~0.995), a domeny widać dopiero w resztach — to jedna orientacja
sieci pseudoregularnej z domenami wariantów. Stąd tryb "referencyjny": jedna orientacja
odniesienia g_ref z profilu uśrednionego, potem dla każdego piksela 6 wariantów V g_ref
dopracowanych lokalnie (szybko, etykiety wariantów spójne na całej mapie).
"""

from __future__ import annotations

import os
import time

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.transform import Rotation

import orient_fit as of

SATURATION = 65535


# ==========================================================================
# Wczytanie i maski
# ==========================================================================
def load_stack(path, omega_step=2.0):
    """Stos .tif (n_omega, H, W) -> Y (H, W, n_omega) float64, omega [deg], surowe uint."""
    import tifffile
    raw = tifffile.imread(path)
    if raw.ndim != 3:
        raise ValueError(f"oczekiwano stosu 3D, jest {raw.shape}")
    Y = np.moveaxis(raw.astype(np.float64), 0, -1)
    omega = np.arange(Y.shape[-1]) * omega_step
    return Y, omega, raw


def valid_mask(raw_or_Y, saturation=SATURATION, zero_invalid=True):
    """True tam, gdzie wartość jest użyteczna. Przyjmuje Y (H, W, n) albo surowy (n, H, W)."""
    A = np.asarray(raw_or_Y)
    if A.shape[0] != A.shape[1] and A.shape[1] == A.shape[2]:      # (n, H, W) -> (H, W, n)
        A = np.moveaxis(A, 0, -1)
    v = A < saturation
    if zero_invalid:
        v &= A > 0
    return v


def _zscore_masked(Y, valid):
    """z-score każdego profilu tylko po ważnych klatkach; nieważne -> 0."""
    w = valid.astype(float)
    n = np.maximum(w.sum(-1, keepdims=True), 1)
    m = (Y * w).sum(-1, keepdims=True) / n
    d = (Y - m) * w
    s = np.sqrt((d * d).sum(-1, keepdims=True))
    return d / np.maximum(s, 1e-300)


def _masked_corr(A, B, valid):
    """Korelacja Pearsona A z B (te same kształty ..., n) po klatkach ważnych w valid."""
    w = valid.astype(float)
    n = np.maximum(w.sum(-1), 1)
    ma = (A * w).sum(-1) / n
    mb = (B * w).sum(-1) / n
    da, db = (A - ma[..., None]) * w, (B - mb[..., None]) * w
    return (da * db).sum(-1) / np.sqrt((da * da).sum(-1) * (db * db).sum(-1) + 1e-300)


def quality_maps(Y, valid):
    """Mapy kontroli jakości (H, W):
      mean, n_sat_or_zero (liczba nieważnych klatek), r_median (korelacja z profilem
      medianowym po ważnych klatkach), r_neigh (mediana korelacji z 4 sąsiadami),
      snr_neigh = sqrt(r/(1-r)) — dolne oszacowanie SNR (zakłada, że sąsiedzi mają ten sam
      profil; prawdziwe różnice między sąsiadami obniżają tę liczbę).
    """
    H, W, n = Y.shape
    med = np.nanmedian(np.where(valid, Y, np.nan).reshape(-1, n), axis=0)
    r_med = _masked_corr(Y, np.broadcast_to(med, Y.shape), valid)
    rn = np.full((H, W, 4), np.nan)
    for k, (di, dj) in enumerate([(0, 1), (0, -1), (1, 0), (-1, 0)]):
        Ys = np.roll(Y, (di, dj), axis=(0, 1))
        vs = np.roll(valid, (di, dj), axis=(0, 1)) & valid
        r = _masked_corr(Y, Ys, vs)
        if di == 1: r[0, :] = np.nan
        if di == -1: r[-1, :] = np.nan
        if dj == 1: r[:, 0] = np.nan
        if dj == -1: r[:, -1] = np.nan
        rn[..., k] = r
    r_neigh = np.nanmedian(rn, axis=-1)
    snr = np.sqrt(np.clip(r_neigh, 0, None) / np.clip(1 - r_neigh, 1e-9, None))
    return dict(mean=(Y * valid).sum(-1) / np.maximum(valid.sum(-1), 1),
                n_invalid=(~valid).sum(-1), r_median=r_med, r_neigh=r_neigh, snr_neigh=snr,
                median_profile=med)


def clean_pixels(qmaps, max_invalid=30, r_median_k=6.0, r_median_min=None, border=0):
    """Maska pikseli nadających się do analizy (H, W).

    Odrzuca: > max_invalid nieważnych klatek; r_median poniżej mediany o r_median_k odchyleń
    MAD (artefakty — jasny obiekt, ciemne plamki); opcjonalnie twardy próg i ramkę brzegu.
    """
    r = qmaps["r_median"]
    med = np.nanmedian(r)
    mad = 1.4826 * np.nanmedian(np.abs(r - med))
    thr = med - r_median_k * mad if r_median_min is None else r_median_min
    ok = (qmaps["n_invalid"] <= max_invalid) & (r >= thr) & np.isfinite(r)
    if border > 0:
        ok[:border] = ok[-border:] = False
        ok[:, :border] = ok[:, -border:] = False
    return ok, thr


# ==========================================================================
# Analiza bez modelu: reszty względem profilu wspólnego, PCA, klastry
# ==========================================================================
def residuals_vs_common(Y, valid, common=None, n_bg_harm=0):
    """Reszty profili po dopasowaniu a*common + b (+ harmoniki 1..n_bg_harm) na ważnych klatkach.

    Zwraca R (H, W, n) w jednostkach z-score profilu (nieważne klatki = 0) i common.
    Domeny o nieco innych profilach widać w R mimo r ~ 0.995 z profilem wspólnym.
    """
    H, W, n = Y.shape
    Z = _zscore_masked(Y, valid)
    if common is None:
        common = np.median(Z.reshape(-1, n)[valid.reshape(-1, n).all(1)], axis=0)
    w = 2 * np.pi * np.arange(n) / n
    cols = [common, np.ones(n)]
    for k in range(1, n_bg_harm + 1):
        cols += [np.cos(k * w), np.sin(k * w)]
    A = np.column_stack(cols)
    Zf, vf = Z.reshape(-1, n), valid.reshape(-1, n)
    R = np.zeros_like(Zf)
    for p in range(len(Zf)):
        k = vf[p]
        if k.sum() < A.shape[1] + 5:
            continue
        c, *_ = np.linalg.lstsq(A[k], Zf[p, k], rcond=None)
        R[p, k] = Zf[p, k] - A[k] @ c
    return R.reshape(H, W, n), common


def residual_pca(R, clean, n_comp=8):
    """PCA reszt na pikselach czystych. Zwraca scores (H, W, n_comp) (NaN poza czystymi),
    components (n_comp, n), explained (n_comp,)."""
    H, W, n = R.shape
    X = R.reshape(-1, n)[clean.ravel()]
    X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    sc = np.full((H * W, n_comp), np.nan)
    sc[clean.ravel()] = U[:, :n_comp] * S[:n_comp]
    return sc.reshape(H, W, n_comp), Vt[:n_comp], (S[:n_comp] ** 2) / np.sum(S ** 2)


def cluster_map(scores, clean, k=2, n_use=4, seed=0):
    """k-średnich na pierwszych n_use składowych -> etykiety (H, W) (-1 poza czystymi),
    posortowane malejąco według liczności (0 = największy klaster)."""
    H, W, _ = scores.shape
    X = scores.reshape(-1, scores.shape[-1])[clean.ravel(), :n_use]
    Xs = X / X.std()               # wspólna skala: składowe ważone swoją wariancją
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(10):
        cen, lab = kmeans2(Xs, k, minit="++", seed=rng)
        inertia = np.sum((Xs - cen[lab]) ** 2)
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    lab = best[1]
    order = np.argsort(-np.bincount(lab, minlength=k))
    remap = np.empty(k, int); remap[order] = np.arange(k)
    out = np.full(H * W, -1)
    out[clean.ravel()] = remap[lab]
    return out.reshape(H, W)


def cluster_auto(scores, clean, method="hdbscan", n_use=4, min_cluster_size=40, min_samples=None,
                 eps=None, selection="eom", k_max=8, xy_weight=0.0, seed=0):
    """Klasteryzacja ze sprawdzaną liczbą klastrów (zamiast k-średnich z zadanym k).

    method:
      "hdbscan" — gęstościowa, bez eps; liczba klastrów wynika z min_cluster_size
                  (selection="eom" łączy, "leaf" daje drobniejsze klastry),
      "dbscan"  — eps z kolana krzywej k-odległości (jeśli eps=None), min_samples domyślnie 2*n_use,
      "gmm"     — mieszanina gaussowska; k = argmin BIC dla k = 1..k_max (klastry eliptyczne).
    xy_weight > 0 dodaje współrzędne piksela (std = 1, razy xy_weight) do cech, więc klastry
    muszą być też spójne przestrzennie.
    Zwraca (labels (H, W), info). Etykiety 0.. posortowane malejąco wg liczności; -1 = poza maską
    `clean` albo szum (info["n_noise"]).
    """
    from sklearn.cluster import DBSCAN, HDBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import NearestNeighbors
    H, W, _ = scores.shape
    idx = np.flatnonzero(clean.ravel())
    X = scores.reshape(-1, scores.shape[-1])[idx, :n_use]
    Xs = X / X.std()
    if xy_weight > 0:
        ii, jj = np.divmod(idx, W)
        xy = np.column_stack([ii, jj]).astype(float)
        xy = (xy - xy.mean(0)) / xy.std()
        Xs = np.column_stack([Xs, xy_weight * xy])
    info = dict(method=method)
    if min_samples is None:
        min_samples = max(2 * Xs.shape[1], 5)
    if method == "hdbscan":
        lab = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                      cluster_selection_method=selection, copy=True).fit_predict(Xs)
    elif method == "dbscan":
        if eps is None:
            d = np.sort(NearestNeighbors(n_neighbors=min_samples).fit(Xs).kneighbors(Xs)[0][:, -1])
            t = np.linspace(0, 1, len(d))                     # kolano = największa odległość od cięciwy
            dn = (d - d[0]) / max(d[-1] - d[0], 1e-12)
            eps = d[np.argmax(t - dn)]
        info["eps"] = float(eps)
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)
    elif method == "gmm":
        bic, models = [], []
        for k in range(1, k_max + 1):
            g = GaussianMixture(k, covariance_type="full", n_init=3, random_state=seed).fit(Xs)
            bic.append(g.bic(Xs)); models.append(g)
        kb = int(np.argmin(bic))
        info["bic"] = np.array(bic)
        lab = models[kb].predict(Xs)
    else:
        raise ValueError(method)
    info["n_noise"] = int((lab < 0).sum())
    ks = np.unique(lab[lab >= 0])
    cnt = np.array([(lab == c).sum() for c in ks])
    remap = {int(c): r for r, c in enumerate(ks[np.argsort(-cnt)])}
    lab2 = np.array([remap.get(int(c), -1) for c in lab])
    info["k"] = len(ks)
    info["sizes"] = np.sort(cnt)[::-1]
    out = np.full(H * W, -1)
    out[idx] = lab2
    return out.reshape(H, W), info


def detrend_scores(scores, clean, sigma=10.0):
    """Odejmuje gładkie tło przestrzenne od składowych (normalizowany splot gaussowski po pikselach
    czystych; sigma w pikselach). Usuwa wolnozmienny gradient (np. wygięcie), zostawia strukturę
    drobniejszą niż ~sigma (paski). sigma=None → bez zmian. Poza `clean` zwraca NaN."""
    from scipy.ndimage import gaussian_filter
    if sigma is None:
        return scores.copy()
    out = np.full(scores.shape, np.nan)
    c = clean.astype(float)
    den = np.maximum(gaussian_filter(c, sigma), 1e-3)
    for k in range(scores.shape[-1]):
        s = np.where(clean, scores[..., k], 0.0)
        out[..., k] = np.where(clean, scores[..., k] - gaussian_filter(s, sigma) / den, np.nan)
    return out


def gmm_scan(scores, clean, n_use=4, k_list=range(1, 11), covariance_type="full", xy_weight=0.0,
             n_init=3, seed=0):
    """GMM dla każdego k w k_list na pierwszych n_use składowych (wspólna skala, opcjonalnie xy).
    Zwraca dict: k_list, bic, aic, labels {k: (H, W)} (klasy malejąco wg liczności, -1 poza maską),
    sizes {k: liczności}."""
    from sklearn.mixture import GaussianMixture
    H, W, _ = scores.shape
    idx = np.flatnonzero(clean.ravel())
    X = scores.reshape(-1, scores.shape[-1])[idx, :n_use]
    Xs = X / X.std()
    if xy_weight > 0:
        ii, jj = np.divmod(idx, W)
        xy = np.column_stack([ii, jj]).astype(float)
        Xs = np.column_stack([Xs, xy_weight * (xy - xy.mean(0)) / xy.std()])
    res = dict(k_list=list(k_list), bic=[], aic=[], labels={}, sizes={})
    for k in res["k_list"]:
        g = GaussianMixture(k, covariance_type=covariance_type, n_init=n_init, random_state=seed).fit(Xs)
        res["bic"].append(g.bic(Xs)); res["aic"].append(g.aic(Xs))
        lab = g.predict(Xs)
        cnt = np.bincount(lab, minlength=k)
        order = np.argsort(-cnt)
        remap = np.empty(k, int); remap[order] = np.arange(k)
        out = np.full(H * W, -1); out[idx] = remap[lab]
        res["labels"][k] = out.reshape(H, W)
        res["sizes"][k] = np.sort(cnt)[::-1]
    res["bic"], res["aic"] = np.array(res["bic"]), np.array(res["aic"])
    return res


def class_profiles(Y, valid, labels):
    """Średni z-score profil każdej klasy (k, n) (po ważnych klatkach) i liczności."""
    Z = _zscore_masked(Y, valid)
    k = labels.max() + 1
    prof, cnt = [], []
    for c in range(k):
        sel = labels == c
        num = Z[sel].sum(0)
        den = valid[sel].sum(0)
        prof.append(num / np.maximum(den, 1))
        cnt.append(sel.sum())
    return np.array(prof), np.array(cnt)


# ==========================================================================
# Dopasowanie mapy w trybie referencyjnym
# ==========================================================================
def fit_map_variants(model, Y, valid, pixels, g_ref, sense=+1, n_bg_harm=0, variants=None,
                     checkpoint=None, every=100, min_valid=60, verbose=True):
    """Dla każdego piksela z listy (indeksy płaskie) dopracowuje 6 wariantów V g_ref.

    Y (H, W, n), valid (H, W, n), g_ref (3,3) wspólne albo (H*W, 3, 3) na piksel.
    Wynik (słownik, tablice po WSZYSTKICH pikselach mapy, NaN gdzie nie liczono):
      G (P, V, 3, 3), R (P, V), C (P, V, 3, 3) — kowariancja [deg^2], n_keep (P,), done (P,).
    checkpoint: ścieżka .npz — zapis co `every` pikseli i wznowienie po przerwaniu.
    """
    H, W, n = Y.shape
    P = H * W
    variants = of.pseudo_variants() if variants is None else variants
    V = len(variants)
    Yf, vf = Y.reshape(P, n), valid.reshape(P, n)
    out = dict(G=np.full((P, V, 3, 3), np.nan), R=np.full((P, V), np.nan),
               C=np.full((P, V, 3, 3), np.nan), n_keep=np.zeros(P, int), done=np.zeros(P, bool))
    if checkpoint and os.path.exists(checkpoint):
        d = np.load(checkpoint)
        for k in out:
            out[k] = d[k]
        if verbose:
            print(f"[wznowienie] {out['done'].sum()} pikseli już policzonych z {checkpoint}")
    g_ref = np.asarray(g_ref)
    todo = [p for p in np.asarray(pixels).ravel() if not out["done"][p]]
    t0 = time.perf_counter()
    for i, p in enumerate(todo):
        keep = vf[p]
        if keep.sum() >= min_valid:
            g0 = g_ref if g_ref.ndim == 2 else g_ref[p]
            G, R, C = of.variant_scores(model, Yf[p], g0, variants, 0.0, sense,
                                        None if keep.all() else keep, n_bg_harm,
                                        return_cov=True)
            out["G"][p], out["R"][p], out["C"][p] = G, R, C
            out["n_keep"][p] = keep.sum()
        out["done"][p] = True
        if verbose and (i + 1) % every == 0:
            el = time.perf_counter() - t0
            print(f"  {i + 1}/{len(todo)}  {el:.0f} s, zostało ~{el / (i + 1) * (len(todo) - i - 1):.0f} s",
                  flush=True)
        if checkpoint and (i + 1) % every == 0:
            np.savez_compressed(checkpoint, **out)
    if checkpoint:
        np.savez_compressed(checkpoint, **out)
    return out


def variant_decision(fit, n_eff=None):
    """Najlepszy wariant, marginesy i log-wiarygodności z wyniku fit_map_variants (P,).
    n_eff: liczba stopni swobody do logL (dla dopasowania w paśmie: projector_dof);
    domyślnie liczba ważnych klatek."""
    R = fit["R"]
    nk = np.maximum(fit["n_keep"], 1) if n_eff is None else np.broadcast_to(n_eff, R.shape[:1])
    ok = np.isfinite(R).all(1)
    best = np.full(len(R), -1)
    best[ok] = np.nanargmax(R[ok], axis=1)
    logL = -0.5 * nk[:, None] * np.log(np.clip(1 - R ** 2, 1e-300, None))
    dlogL, dr, r_best = (np.full(len(R), np.nan) for _ in range(3))
    srt = -np.sort(-logL[ok], axis=1)
    rs = -np.sort(-R[ok], axis=1)
    dlogL[ok], dr[ok], r_best[ok] = srt[:, 0] - srt[:, 1], rs[:, 0] - rs[:, 1], rs[:, 0]
    return dict(best=best, dlogL=dlogL, dr=dr, r_best=r_best, logL=logL, ok=ok)


# ==========================================================================
# Reprezentacje orientacji do analizy przestrzennej
# ==========================================================================
def c_axis_sample(G):
    """Kierunek osi c kryształu w układzie próbki: g^T [0,0,1] (g pasywne próbka -> kryształ)."""
    G = np.asarray(G)
    return G[..., 2, :]          # trzeci wiersz g = g^T e_z


def rotation_field(G, G_ref):
    """Wektor obrotu [deg] Δ taki, że G = G_ref @ R(Δ) (Δ w układzie próbki).
    Dla pikseli dopasowanych od V g_ref to mała "rotacja sieci" względem odniesienia."""
    G, G_ref = np.asarray(G), np.asarray(G_ref)
    D = np.einsum("...ji,...jk->...ik", np.broadcast_to(G_ref, G.shape), G)   # G_ref^T G
    flat = D.reshape(-1, 3, 3)
    ok = np.isfinite(flat).all((1, 2))
    out = np.full((len(flat), 3), np.nan)
    if ok.any():
        out[ok] = np.rad2deg(Rotation.from_matrix(flat[ok]).as_rotvec())
    return out.reshape(G.shape[:-2] + (3,))


def wall_traces(g, families=None):
    """Kierunki śladów płaszczyzn na powierzchni próbki dla orientacji g (eksploracja).

    Dla domen 90° w fazie tetragonalnej (perowskit) ściany leżą na {101} (domeny c/a)
    i {110} (domeny a1/a2). Ślad płaszczyzny o normalnej n_s (układ próbki) na powierzchni
    z = 0 ma kierunek n_s x z. Zwraca listę (płaszczyzna, kąt śladu [deg, 0-180, od osi x
    próbki], nachylenie płaszczyzny do powierzchni [deg]).

    UWAGA: kąt jest w układzie próbki EMsoftu; związek z osiami obrazu (wiersz/kolumna)
    zależy od phi0, sense i ewentualnego odbicia obrazu — porównuj RÓŻNICE kątów rodzin
    albo kalibruj obrót obraz <-> próbka osobno.
    """
    if families is None:
        families = [(1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1), (1, 1, 0), (1, -1, 0)]
    g = np.asarray(g, float).reshape(3, 3)
    out = []
    for hkl in families:
        n_c = np.asarray(hkl, float) / np.linalg.norm(hkl)
        n_s = g.T @ n_c
        t = np.cross(n_s, [0.0, 0.0, 1.0])
        if np.linalg.norm(t) < 1e-9:
            out.append((hkl, np.nan, 0.0))
            continue
        ang = np.rad2deg(np.arctan2(t[1], t[0])) % 180
        incl = np.rad2deg(np.arccos(abs(n_s[2])))
        out.append((hkl, float(ang), float(incl)))
    return out
