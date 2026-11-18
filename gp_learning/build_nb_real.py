import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Prawdziwe dane eCHORD (KTN_014): jakość → domeny bez modelu → kalibracja → mapa orientacji → GP

Kolejność jest celowa: najpierw wszystko, co da się powiedzieć **bez mastera** (jakość danych,
struktura domen z samych profili), potem model: orientacja odniesienia, kalibracja θ (i σ_conv),
dopasowanie całej mapy w trybie referencyjnym, na końcu analiza przestrzenna GP.

Wstępna inspekcja tego pliku pokazała:
* wszystkie piksele mają niemal ten sam profil (r ≈ 0.995 z profilem medianowym) → jedna
  orientacja sieci pseudoregularnej; domeny widać dopiero w resztach (paski ukośne),
* SNR pojedynczego piksela ≈ 12 (z korelacji sąsiadów) — w zakresie testowanym w symulacjach,
* nasycenie (65 535) w jasnych klatkach i przy obiekcie ~(31,31), zera w narożnikach (rejestracja).

**Uwaga o konwencjach:** sense i invert są niewyznaczalne z profili (lustra 4mm) — tu ustalone
umownie; orientacje bezwzględne trzeba będzie przeliczyć do konwencji metody słownikowej.
φ0 = 0: błąd φ0 siedzi w φ1 i nie wpływa na nic poza porównaniem z metodą słownikową.
""")

md("## 0. Parametry")
code(r"""
%load_ext autoreload
%autoreload 2
%matplotlib inline

import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import echord_cone as ec
import orient_fit as of
import calibration as cal
import real_data as rd
import spatial_gp as sg

TIF    = "../2026_10_19_example_PZ/KTN_014_s2_fin_crop_372x372_bin_62x62.tif"
MASTER = "../2026_10_19_example_PZ/masterKTN_014_01.h5"
OMEGA_STEP = 2.0
THETA0, SENSE, INVERT = 40.0, +1, False   # sense/invert: umowne (niewyznaczalne z profili)
N_BG_HARM = 3                     # usuwane harmoniki tła w ω (symulacje: bez utraty informacji)
SPACING, TOP_K, N_REFINE = 0.7, 5, 6
BANK_CACHE = "bank"
N_CAL_PER_QUAD = 10               # pikseli kalibracyjnych na ćwiartkę mapy
K_CLUSTERS = 4
TIF_FULL = None                   # stos 372x372 bez binningu (sekcja 19) albo None
BIN = 6                           # współczynnik binningu: 372 / 62
SEED = 0

plt.rcParams.update({"figure.dpi": 110, "axes.grid": False})
VARIANTS = of.pseudo_variants()
D4, O24 = of.d4_ops(), of.cubic_ops()
""")

md("## 1. Wczytanie i kontrola jakości")
code(r"""
Y, omega, raw = rd.load_stack(TIF, OMEGA_STEP)
H, W, NW = Y.shape
valid = rd.valid_mask(raw)
q = rd.quality_maps(Y, valid)
clean, thr = rd.clean_pixels(q)
print(f"stos {raw.shape} {raw.dtype}, mapa {H}x{W}, {NW} klatek co {OMEGA_STEP}°")
print(f"nieważne wartości: {(~valid).sum()} (nasycone {(raw == rd.SATURATION).sum()}, zera {(raw == 0).sum()})")
print(f"czyste piksele: {clean.sum()} / {H * W}  (próg r_median = {thr:.4f})")
print(f"SNR z sąsiadów (mediana, czyste): {np.nanmedian(q['snr_neigh'][clean]):.1f}")

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
fmean = (Y * valid).reshape(-1, NW).sum(0) / valid.reshape(-1, NW).sum(0)
ax[0, 0].plot(omega, fmean, "k", lw=0.8); ax[0, 0].set_title("średnia klatki vs ω"); ax[0, 0].set_xlabel("ω [°]")
for a_, m_, t_, kw in [(ax[0, 1], q["mean"], "średnia po ω", {}),
                       (ax[0, 2], q["n_invalid"], "liczba nieważnych klatek", {}),
                       (ax[1, 0], q["r_median"], "r z profilem medianowym", dict(vmin=0.98, vmax=1)),
                       (ax[1, 1], np.where(clean, q["snr_neigh"], np.nan), "SNR (z sąsiadów)", {}),
                       (ax[1, 2], clean, "czyste piksele", dict(cmap="gray"))]:
    im = a_.imshow(m_, **kw); a_.set_title(t_); plt.colorbar(im, ax=a_)
plt.tight_layout(); plt.show()
""")

md(r"""
## 2. Domeny bez modelu: reszty względem profilu wspólnego

Każdy profil (z-score) dopasowany jako a·wspólny + b; reszty niosą różnice między domenami.
PCA reszt i k-średnich na kilku pierwszych składowych → mapa klas, której **nie wyznacza
żaden model** — punkt odniesienia dla mapy wariantów z dopasowania.
""")
code(r"""
Rres, common = rd.residuals_vs_common(Y, valid)
sc, comp, ex = rd.residual_pca(Rres, clean, n_comp=8)
labs = {k: rd.cluster_map(sc, clean, k=k, n_use=4, seed=SEED) for k in (2, 3, 4)}
print("udział wariancji reszt PC1..8:", np.round(ex, 3))

fig, ax = plt.subplots(2, 4, figsize=(19, 9))
for k in range(4):
    im = ax[0, k].imshow(np.where(clean, sc[..., k], np.nan), cmap="RdBu_r")
    ax[0, k].set_title(f"PC{k + 1} reszt ({ex[k]:.0%})"); plt.colorbar(im, ax=ax[0, k])
for a_, k in zip(ax[1, :3], (2, 3, 4)):
    a_.imshow(np.where(labs[k] >= 0, labs[k], np.nan), cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    a_.set_title(f"klasy k={k}: " + ", ".join(str(c) for c in np.bincount(labs[k][labs[k] >= 0])))
ax[1, 3].plot(omega, comp[0], lw=0.8, label="PC1"); ax[1, 3].plot(omega, comp[1], lw=0.8, label="PC2")
ax[1, 3].set_title("kształt PC w ω"); ax[1, 3].legend(); ax[1, 3].set_xlabel("ω [°]")
plt.tight_layout(); plt.show()
""")
code(r"""
labK = labs[K_CLUSTERS]
cprof, ccnt = rd.class_profiles(Y, valid, labK)
z = lambda x: (x - x.mean()) / np.linalg.norm(x - x.mean())
M = np.array([[1 - z(a) @ z(b) for b in cprof] for a in cprof])
print("1 − r między średnimi profilami klas (porównaj: warianty w symulacji ~0.03–0.04):")
display(pd.DataFrame(M, index=[f"klasa {c} ({n})" for c, n in enumerate(ccnt)],
                     columns=[f"klasa {c}" for c in range(len(ccnt))]).style.format("{:.4f}"))
fig, ax = plt.subplots(figsize=(12, 3.5))
for c in range(len(ccnt)):
    ax.plot(omega, cprof[c] - cprof.mean(0), lw=0.8, label=f"klasa {c}")
ax.set_xlabel("ω [°]"); ax.set_ylabel("z-score − średnia klas"); ax.legend(); ax.grid(alpha=0.3)
plt.show()
""")

md(r"""
### 2b. Liczba klas wyznaczona automatycznie (HDBSCAN, DBSCAN, GMM + BIC)

Zamiast zadawać k: HDBSCAN (liczba klastrów z gęstości; `min_cluster_size` ≈ najmniejsza
sensowna domena w pikselach), DBSCAN (eps z kolana krzywej k-odległości) i GMM z wyborem k
przez BIC. Wersje `xy_weight > 0` wymagają też spójności przestrzennej.
**Uwaga:** dla pikseli rozcieńczonych (ułamek pasków zmienia się ciągle) klastrów nie ma —
metody gęstościowe wtedy albo łączą wszystko w jeden, albo tną gradient dowolnie. Liczba klas
jest więc wiarygodna tylko tam, gdzie metody się zgadzają i różnią się profilami klas.
""")
code(r"""
runs = {
    "hdbscan (eom)":        dict(method="hdbscan", selection="eom"),
    "hdbscan (leaf)":       dict(method="hdbscan", selection="leaf"),
    "dbscan":               dict(method="dbscan"),
    "gmm (BIC)":            dict(method="gmm"),
    "hdbscan + xy 0.5":     dict(method="hdbscan", xy_weight=0.5),
}
auto = {}
for name, kw in runs.items():
    auto[name] = rd.cluster_auto(sc, clean, n_use=4, min_cluster_size=30, seed=SEED, **kw)
    lab_, inf_ = auto[name]
    print(f"{name:20s} k = {inf_['k']}, szum = {inf_['n_noise']}, liczności = {inf_['sizes'].tolist()}")
fig, ax = plt.subplots(1, len(runs), figsize=(4 * len(runs), 4.2))
for a_, (name, (lab_, inf_)) in zip(ax, auto.items()):
    a_.imshow(np.where(lab_ >= 0, lab_, np.nan), cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    a_.set_title(f"{name}: k={inf_['k']}")
plt.tight_layout(); plt.show()
if "bic" in auto["gmm (BIC)"][1]:
    plt.figure(figsize=(5, 3)); plt.plot(range(1, 9), auto["gmm (BIC)"][1]["bic"], "o-")
    plt.xlabel("k"); plt.ylabel("BIC"); plt.grid(alpha=0.3); plt.show()
""")

md(r"""
## 3. GP przestrzenny na PC1 (bez modelu)

Jądro anizotropowe: α = kierunek pasków (od osi kolumn w stronę wierszy, oś wierszy w dół
jak na obrazie), l_par / l_perp = długości korelacji wzdłuż / w poprzek pasków.
To pierwsza ilościowa charakterystyka struktury domen — bez żadnych założeń o krysztale.
""")
code(r"""
XY = sg.grid_coords(H, W)
m = clean.ravel()
gp_pc1 = sg.SpatialGP("matern32").fit(XY[m], sc[..., 0].ravel()[m], n_sub=1200, seed=SEED)
print(gp_pc1.summary())
mu, sd = gp_pc1.predict(XY)
loo_mu, loo_sd = gp_pc1.loo()
zl = (sc[..., 0].ravel()[m] - loo_mu) / loo_sd
print(f"LOO: korelacja {np.corrcoef(loo_mu, sc[..., 0].ravel()[m])[0, 1]:.3f}, sd(z) = {zl.std():.2f}")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for a_, im_, t_ in [(ax[0], np.where(clean, sc[..., 0], np.nan), "PC1"),
                    (ax[1], mu.reshape(H, W), "GP: średnia"), (ax[2], sd.reshape(H, W), "GP: sd")]:
    im = a_.imshow(im_, cmap="RdBu_r" if a_ is not ax[2] else "viridis"); a_.set_title(t_)
    plt.colorbar(im, ax=a_)
a = np.deg2rad(gp_pc1.params["alpha_deg"])
ax[1].arrow(W / 2, H / 2, 12 * np.cos(a), 12 * np.sin(a), color="k", width=0.4)
plt.tight_layout(); plt.show()
""")

md(r"""
## 4. Model: orientacja odniesienia przy nominalnym θ

Dopasowujemy (pełne szukanie z bankiem + 6 wariantów) profil wspólny i średnie profile klas.
Średnie klas uśredniają setki pikseli → szum znika, zostaje tylko niedopasowanie modelu.
""")
code(r"""
mp = ec.MasterPattern(MASTER)
print(mp)
print("symetria luster:", {k: round(v, 4) for k, v in of.mirror_symmetry_check(mp).items()})
fit0 = of.ConeModel(mp, THETA0, INVERT)
path = f"{BANK_CACHE}_cone_th{THETA0}_sp{SPACING}_bg{N_BG_HARM}.npz" if BANK_CACHE else None
if path and os.path.exists(path):
    bank = of.Bank.load(path)
else:
    bank = of.build_bank(fit0, SPACING, n_bg_harm=N_BG_HARM)
    if path:
        bank.save(path)


def fit_refs(model, profiles, names, g_start=None):
    rows, gs = [], []
    for k, (y, nm) in enumerate(zip(profiles, names)):
        if g_start is None:
            res = of.fit_profile(model, bank, y, 0.0, SENSE, top_k=TOP_K, variants=VARIANTS,
                                 n_refine=N_REFINE, n_bg_harm=N_BG_HARM, score_variants=True)
            g, Rv = res["g"], res["variant_r"]
        else:                                    # tylko dopracowanie (np. po kalibracji)
            Gv, Rv = of.variant_scores(model, y, g_start, VARIANTS, 0.0, SENSE, None, N_BG_HARM)
            g = Gv[np.argmax(Rv)]
        dr, dl = of.margin_stats(Rv, len(y))
        e = of.euler_from_matrix(g)[0]
        rows.append(dict(profil=nm, r=Rv.max(), dr=dr, dlogL=dl, phi1=e[0], Phi=e[1], phi2=e[2]))
        gs.append(g)
    return pd.DataFrame(rows).set_index("profil"), np.stack(gs)


ref_profiles = np.vstack([z(common)[None], cprof])
ref_names = ["wspólny"] + [f"klasa {c}" for c in range(len(cprof))]
t0 = time.perf_counter()
tab0, G_refs0 = fit_refs(fit0, ref_profiles, ref_names)
print(f"{time.perf_counter() - t0:.0f} s")
display(tab0.style.format({"r": "{:.4f}", "dr": "{:.4f}", "dlogL": "{:.1f}",
                           "phi1": "{:.2f}", "Phi": "{:.2f}", "phi2": "{:.2f}"}))
g_ref0 = G_refs0[0]
""")

md(r"""
## 5. Kalibracja θ (bez etykiet)

Piksele kalibracyjne: po `N_CAL_PER_QUAD` losowych czystych pikseli z każdej ćwiartki
(żeby od razu sprawdzić, czy θ jest stałe na mapie). Starty: orientacja odniesienia
(wszystkie piksele to ta sama sieć; max po wariantach obsługuje domeny). Dodatkowo
skan na średnich profilach klas (bez szumu).
""")
code(r"""
rng = np.random.default_rng(SEED)
ii, jj = np.mgrid[0:H, 0:W]
quad = (ii >= H // 2) * 2 + (jj >= W // 2)
cal_pix, cal_quad = [], []
for qd in range(4):
    cand = np.flatnonzero((clean & (quad == qd)).ravel())
    pick = rng.choice(cand, min(N_CAL_PER_QUAD, len(cand)), replace=False)
    cal_pix += list(pick); cal_quad += [qd] * len(pick)
cal_pix, cal_quad = np.array(cal_pix), np.array(cal_quad)
Yf, vf = Y.reshape(-1, NW), valid.reshape(-1, NW)

t0 = time.perf_counter()
th_hat, scan_th = cal.calibrate_theta(mp, Yf[cal_pix], np.repeat(g_ref0[None], len(cal_pix), 0),
                                      THETA0, sense=SENSE, invert=INVERT, mask=vf[cal_pix],
                                      n_bg_harm=N_BG_HARM, variants=VARIANTS)
th_quad = {qd: cal.theta_hat(scan_th, np.flatnonzero(cal_quad == qd)) for qd in range(4)}
th_cls, scan_cls = cal.calibrate_theta(mp, ref_profiles, np.repeat(g_ref0[None], len(ref_profiles), 0),
                                       THETA0, sense=SENSE, invert=INVERT, n_bg_harm=N_BG_HARM,
                                       variants=VARIANTS, verbose=False)
print(f"θ̂ (piksele) = {th_hat:.3f}°,  θ̂ (średnie klas) = {th_cls:.3f}°   [{time.perf_counter() - t0:.0f} s]")
if min(abs(th_hat - scan_th["thetas"].min()), abs(th_hat - scan_th["thetas"].max())) < 1e-6:
    print("UWAGA: θ̂ na brzegu siatki skanu — poszerz zakres (coarse_half w calibrate_theta)")
print("θ̂ w ćwiartkach:", {f"Q{k}": round(v, 3) for k, v in th_quad.items()},
      f" rozrzut {np.ptp(list(th_quad.values())):.3f}°  (tolerancja z symulacji ~±0.07°)")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(scan_th["thetas"], scan_th["R"].max(2).mean(1), ".-", label="piksele")
ax[0].plot(scan_cls["thetas"], scan_cls["R"].max(2).mean(1), ".-", label="średnie klas")
ax[0].axvline(th_hat, color="tab:red", ls="--"); ax[0].axvline(THETA0, color="0.6", ls="-.")
ax[0].set_xlabel("θ modelu [°]"); ax[0].set_ylabel("średnie r_best"); ax[0].legend(); ax[0].grid(alpha=0.3)
for qd in range(4):
    sel = cal_quad == qd
    ax[1].plot(scan_th["thetas"], scan_th["R"].max(2)[:, sel].mean(1), ".-", ms=3, label=f"Q{qd}")
ax[1].set_xlim(th_hat - 0.4, th_hat + 0.4); ax[1].set_xlabel("θ [°]"); ax[1].legend(); ax[1].grid(alpha=0.3)
ax[1].set_title("ćwiartki")
plt.tight_layout(); plt.show()
""")
md(r"""
Jak czytać: ostry pik i r_best w maksimum blisko wartości z symulacji (≈0.99 przy SNR ~12)
→ model dobrze opisuje dane. Płaski pik / niskie r → duże niedopasowanie (master, tło, dryf).
Różnice θ̂ między ćwiartkami > ~0.07° → θ zmienia się na próbce (krzywizna, pochylenie) —
wtedy trzeba kalibrować lokalnie.
""")

md("## 6. Rozmycie wiązki σ_conv przy θ̂")
code(r"""
SIGMAS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
scan_sg = cal.sigma_scan(mp, Yf[cal_pix], scan_th["G"][np.argmin(abs(scan_th["thetas"] - th_hat))],
                         th_hat, SIGMAS, sense=SENSE, invert=INVERT, mask=vf[cal_pix],
                         n_bg_harm=N_BG_HARM, variants=VARIANTS)
sg_hat = max(0.0, cal.peak(scan_sg["thetas"], scan_sg["Lbest"].sum(1)))
print(f"σ̂_conv = {sg_hat:.2f}°")
if sg_hat >= max(SIGMAS) - 1e-9:
    print("UWAGA: σ̂ na górnym brzegu siatki — poszerz SIGMAS; duże σ̂ może też oznaczać, że "
          "rozmycie 'udaje' inne niedopasowanie (master, tło, dryf)")
plt.figure(figsize=(6, 3.5))
plt.plot(SIGMAS, scan_sg["R"].max(2).mean(1), "o-"); plt.axvline(sg_hat, color="tab:red", ls="--")
plt.xlabel("σ_conv [°]"); plt.ylabel("średnie r_best"); plt.grid(alpha=0.3); plt.show()
""")

md(r"""
## 7. Orientacja odniesienia po kalibracji i hipoteza „czym są paski?”

Porównujemy orientacje dopasowane do średnich profili klas:
* **ten sam wariant, różnica orientacji ~0.1°** → paski to domeny 90° widziane przez lekki
  obrót sieci (dla c/a = 1.0023 sąsiednie domeny 90° są obrócone o ~0.13°) albo inne małe
  odkształcenia — wtedy wybór wariantu z pojedynczego profilu jest niemożliwy i liczy się
  pole obrotu sieci,
* **różne warianty (90° mod D4 / 180° polarność)** → paski to domeny wariantów, zgodnie
  z modelem; jeśli przy tym 1 − r między klasami (sekcja 2) jest dużo mniejsze niż ~0.03,
  to piksele są mieszaninami albo model przecenia różnice wariantów.
""")
code(r"""
model = of.ConeModel(mp, th_hat, INVERT, sigma_conv_deg=sg_hat)
tab1, G_refs = fit_refs(model, ref_profiles, ref_names, g_start=g_ref0)
g_ref = G_refs[0]
mis = lambda A, B, ops=None: of.misorientation_deg(A, B, ops)[0]
tab1["mis C4 vs wspólny [°]"] = [mis(g, g_ref) for g in G_refs]
tab1["mis D4 [°]"] = [mis(g, g_ref, D4) for g in G_refs]
tab1["mis O [°]"] = [mis(g, g_ref, O24) for g in G_refs]
display(tab1.style.format("{:.4f}", subset=["r", "dr"]).format("{:.2f}", subset=[
    "dlogL", "phi1", "Phi", "phi2", "mis C4 vs wspólny [°]", "mis D4 [°]", "mis O [°]"]))
print("oś c w układzie próbki:", np.round(rd.c_axis_sample(g_ref), 4))
""")

md(r"""
## 8. Dopasowanie całej mapy (tryb referencyjny)

Dla każdego czystego piksela: 6 wariantów V·g_ref dopracowanych przy θ̂, σ̂; nasycone klatki
maskowane (rzut tła liczony dokładnie na pozostałych). Punkt kontrolny pozwala przerwać
i wznowić. Etykiety wariantów są spójne na całej mapie (wszystkie względem tego samego g_ref).
""")
code(r"""
CHECKPOINT = f"mapfit_th{th_hat:.3f}_sg{sg_hat:.2f}_bg{N_BG_HARM}.npz"
pix = np.flatnonzero(clean.ravel())
fitmap = rd.fit_map_variants(model, Y, valid, pix, g_ref, sense=SENSE, n_bg_harm=N_BG_HARM,
                             variants=VARIANTS, checkpoint=CHECKPOINT, every=200)
dec = rd.variant_decision(fitmap)
best = dec["best"]
okp = dec["ok"]
print("udział wariantów (V0 = odniesienie):",
      {f"V{v}": f"{np.mean(best[okp] == v):.1%}" for v in range(len(VARIANTS))})
""")
code(r"""
to_map = lambda v: np.where(okp, v, np.nan).reshape(H, W)
G_best = fitmap["G"][np.arange(H * W), np.clip(best, 0, None)]
rot = rd.rotation_field(G_best, VARIANTS[np.clip(best, 0, None)] @ g_ref)      # (P,3) [deg]
rot[~okp] = np.nan
C_best = fitmap["C"][np.arange(H * W), np.clip(best, 0, None)]
rot_sd = np.sqrt(np.clip(np.diagonal(C_best, axis1=1, axis2=2), 0, None))
rot_sd[~okp] = np.nan

fig, ax = plt.subplots(2, 4, figsize=(19, 9))
ax[0, 0].imshow(to_map(best), cmap="tab10", vmin=0, vmax=9, interpolation="nearest"); ax[0, 0].set_title("najlepszy wariant")
for a_, im_, t_, kw in [(ax[0, 1], to_map(dec["r_best"]), "r_best", {}),
                        (ax[0, 2], to_map(np.log10(np.clip(dec["dlogL"], 1e-2, None))), "log10 ΔlogL", {}),
                        (ax[0, 3], to_map(dec["dr"]), "Δr", {})]:
    im = a_.imshow(im_, **kw); a_.set_title(t_); plt.colorbar(im, ax=a_)
for k in range(3):
    lim = np.nanpercentile(np.abs(rot[:, k] - np.nanmedian(rot[:, k])), 98)
    im = ax[1, k].imshow(to_map(rot[:, k] - np.nanmedian(rot[:, k])), cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[1, k].set_title(f"obrót sieci Δ{'xyz'[k]} − mediana [°]"); plt.colorbar(im, ax=ax[1, k])
im = ax[1, 3].imshow(to_map(np.linalg.norm(rot_sd, axis=1))); ax[1, 3].set_title("σ obrotu (Laplace) [°]")
plt.colorbar(im, ax=ax[1, 3])
plt.tight_layout(); plt.show()

print("tabela: klasy bez modelu (wiersze) vs najlepszy wariant (kolumny)")
display(pd.crosstab(labK.ravel()[okp], best[okp], rownames=["klasa"], colnames=["wariant"]))
""")
md(r"""
Porównanie z sekcją 2: jeśli mapa wariantów odtwarza paski z klas bez modelu — model widzi
te same domeny i przypisuje im warianty. Jeśli warianty są jednolite, a paski widać
w **obrocie sieci** (dolny wiersz) — domeny różnią się orientacją o ułamki stopnia,
a nie wariantem (zgodnie z sekcją 7). Ostrożnie z ΔlogL: przy niedopasowaniu strukturalnym
zawyża pewność (symulacje w `mismatch_tests.ipynb`).
""")

md(r"""
## 9. GP przestrzenny na wynikach modelu

**(a) Pole obrotu sieci** — trzy składowe Δ z szumem heteroskedastycznym z Laplace'a.
Długości korelacji i kierunek mówią o skali i geometrii odkształceń sieci; średnia GP to
wygładzone pole z niepewnością (np. do gradientów obrotu ~ gęstości dyslokacji/odkształceń).
""")
code(r"""
gp_rot = {}
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
for k in range(3):
    yk = rot[:, k] - np.nanmedian(rot[:, k])
    m_ = okp & np.isfinite(yk)
    gp = sg.SpatialGP("matern32").fit(XY[m_], yk[m_], noise_var=rot_sd[m_, k] ** 2,
                                      n_sub=1200, seed=SEED, verbose=False)
    gp_rot[k] = gp
    print(f"Δ{'xyz'[k]}: {gp.summary()}")
    mu_k, sd_k = gp.predict(XY)
    lim = np.nanpercentile(np.abs(mu_k), 98)
    im = ax[0, k].imshow(mu_k.reshape(H, W), cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[0, k].set_title(f"GP: Δ{'xyz'[k]} [°]"); plt.colorbar(im, ax=ax[0, k])
    im = ax[1, k].imshow(sd_k.reshape(H, W)); ax[1, k].set_title(f"GP sd Δ{'xyz'[k]} [°]")
    plt.colorbar(im, ax=ax[1, k])
plt.tight_layout(); plt.show()
""")

md(r"""
**(b) Dowód o wariancie** — dla dwóch najczęstszych wariantów a, b: s = (logL_a − logL_b) / n.
GP wygładza tę ciągłą miarę: znak średniej = wariant, |średnia| / sd = pewność. Uwzględnia
sąsiedztwo (domeny są ciągłe), czego decyzja piksel po pikselu nie robi. To eksploracja —
docelowo klasyfikacja GP / pole losowe na 6 etykietach.
""")
code(r"""
cnt = np.bincount(best[okp], minlength=len(VARIANTS))
va, vb = np.argsort(-cnt)[:2]
print(f"dwa najczęstsze warianty: V{va} ({cnt[va]}), V{vb} ({cnt[vb]})")
if cnt[vb] >= 20:
    s = (dec["logL"][:, va] - dec["logL"][:, vb]) / np.maximum(fitmap["n_keep"], 1)
    m_ = okp & np.isfinite(s)
    gp_ev = sg.SpatialGP("matern32").fit(XY[m_], s[m_], n_sub=1200, seed=SEED, verbose=False)
    print(gp_ev.summary())
    mu_s, sd_s = gp_ev.predict(XY)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    lim = np.nanpercentile(np.abs(s[m_]), 98)
    im = ax[0].imshow(to_map(s), cmap="RdBu_r", vmin=-lim, vmax=lim); ax[0].set_title(f"s piksel (V{va} > 0 > V{vb})")
    plt.colorbar(im, ax=ax[0])
    im = ax[1].imshow(mu_s.reshape(H, W), cmap="RdBu_r", vmin=-lim, vmax=lim); ax[1].set_title("GP: średnia s")
    plt.colorbar(im, ax=ax[1])
    im = ax[2].imshow((np.abs(mu_s) / sd_s).reshape(H, W), vmin=0, vmax=5); ax[2].set_title("|średnia| / sd")
    plt.colorbar(im, ax=ax[2])
    plt.tight_layout(); plt.show()
else:
    print("drugi wariant prawie nie występuje — dowód o wariancie jednolity, pomijam")
""")

md(r"""
**(c) Geometria pasków vs krystalografia (eksploracja).** Ślady płaszczyzn {101} i {110}
dla g_ref na powierzchni próbki. Kąty są w układzie próbki EMsoftu, a α z GP w układzie
obrazu — ich związek (obrót, ewentualne odbicie) nie jest jeszcze skalibrowany, więc
porównuj przede wszystkim **różnice** kątów między rodzinami pasków, jeśli jest ich kilka.
""")
code(r"""
tr = pd.DataFrame(rd.wall_traces(g_ref), columns=["płaszczyzna", "kąt śladu [°]", "nachylenie [°]"])
display(tr.style.format({"kąt śladu [°]": "{:.1f}", "nachylenie [°]": "{:.1f}"}))
print(f"kierunek pasków z GP (PC1, układ obrazu): α = {gp_pc1.params['alpha_deg']:.1f}°")
""")

md("## 10. Zapis wyników")
code(r"""
OUT = "real_results.npz"
np.savez_compressed(OUT, theta_hat=th_hat, sigma_hat=sg_hat, g_ref=g_ref, clean=clean,
                    best=best.reshape(H, W), r_best=dec["r_best"].reshape(H, W),
                    dlogL=dec["dlogL"].reshape(H, W), dr=dec["dr"].reshape(H, W),
                    rot=rot.reshape(H, W, 3), rot_sd=rot_sd.reshape(H, W, 3),
                    labels_model_free=labK, pca_scores=sc, sense=SENSE, invert=INVERT,
                    n_bg_harm=N_BG_HARM)
print("zapisano", OUT)
""")


md(r"""
# Część 2: diagnostyka modelu, dopasowanie w paśmie, wspólne pole GP

Sekcje 11–13 zbierają testy z rozmowy (A–D) w powtarzalną formę i idą dalej:
* **11** — podobieństwo wariantów vs różnice klas, zgodność model–pomiar w pasmach harmonik,
  obrót sieci w poprzek pasków,
* **12** — dopasowanie tylko w paśmie harmonik, w którym model zgadza się z pomiarem
  (domyślnie 21–60), i powtórzenie testów: klasy, θ dla klas, mapa wariantów,
* **13** — GP z jednym wspólnym polem ukrytym dla PC1 i składowych obrotu sieci
  (ułamek pasków w pikselu), z ładunkami = obrót sieci na jednostkę PC1.
""")

md("## 11. Diagnostyka modelu")
code(r"""
BANDS_DIAG = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 60), (61, 90)]
unit = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
lblV = [f"V{v}" for v in range(len(VARIANTS))]


def variant_vs_class(spec):
    # 1 - r między symulowanymi wariantami g_ref i między zmierzonymi klasami, po rzucie spec
    Pp = of.make_projector(NW, spec)
    Zs_ = unit(model.profiles(VARIANTS @ g_ref, 0.0, SENSE) @ Pp.T)
    Zc_ = unit(cprof @ Pp.T)
    return Zs_, Zc_, 1 - Zs_ @ Zs_.T, 1 - Zc_ @ Zc_.T


def coherence_table(y, mu, Zs_, bands=BANDS_DIAG):
    FY, FM = np.fft.rfft(y), np.fft.rfft(mu)
    rows = []
    for lo, hi in bands:
        s = slice(lo, hi + 1)
        coh = np.real(np.sum(FY[s] * np.conj(FM[s]))) / np.sqrt(np.sum(abs(FY[s])**2) * np.sum(abs(FM[s])**2))
        dv = np.mean([np.sum(abs(np.fft.rfft(Zs_[i] - Zs_[j])[s])**2) / np.sum(abs(np.fft.rfft(Zs_[i] - Zs_[j]))**2)
                      for i in range(len(Zs_)) for j in range(len(Zs_)) if i != j])
        rows.append(dict(harmoniki=f"{lo}–{hi}", zgodność=coh,
                         udział_pomiaru=np.sum(abs(FY[s])**2) / np.sum(abs(FY)**2), udział_różnic_wariantów=dv))
    return pd.DataFrame(rows)


def class_pairs(Zs_, Zc_):
    rows = []
    for i in range(len(Zc_)):
        for j in range(i + 1, len(Zc_)):
            dm = Zc_[i] - Zc_[j]
            cand = sorted([((dm @ (Zs_[a] - Zs_[b])) / np.linalg.norm(dm) / np.linalg.norm(Zs_[a] - Zs_[b]), a, b)
                           for a in range(len(Zs_)) for b in range(len(Zs_)) if a != b], reverse=True)
            rows.append(dict(klasy=f"{i}−{j}", r_1=cand[0][0], pary_1=f"V{cand[0][1]}−V{cand[0][2]}",
                             r_2=cand[1][0], pary_2=f"V{cand[1][1]}−V{cand[1][2]}"))
    return pd.DataFrame(rows)


Zs, Zc, simV, measC = variant_vs_class(N_BG_HARM)
print("symulacja: 1 − r między wariantami g_ref (pełne widmo bez tła)")
display(pd.DataFrame(simV, lblV, lblV).style.format("{:.4f}"))
print("pomiar: 1 − r między klasami")
display(pd.DataFrame(measC).style.format("{:.4f}"))

Pm = of.bg_projector(NW, N_BG_HARM)
y0, gb0 = ref_profiles[0], G_refs[0]
a_, b_ = Pm @ y0, Pm @ model.profiles(gb0[None], 0.0, SENSE)[0]
b_ *= (a_ @ b_) / (b_ @ b_)
print("zgodność model–pomiar (profil wspólny) w pasmach harmonik:")
display(coherence_table(a_, b_, Zs).style.format(precision=3))
""")
code(r"""
# obrót sieci względem WSPÓLNEGO odniesienia (wariant 0) i profil w poprzek pasków
rot0 = rd.rotation_field(fitmap["G"][:, 0], g_ref)
var0 = np.diagonal(fitmap["C"][:, 0], axis1=1, axis2=2)                 # [deg^2]
a_str = np.deg2rad(gp_pc1.params["alpha_deg"])
u_perp = XY @ np.array([-np.sin(a_str), np.cos(a_str)])                 # w poprzek pasków [px]


def cross_profile(V, m, step=0.5, min_n=6):
    V = np.asarray(V).reshape(len(u_perp), -1)
    edges = np.arange(np.floor(u_perp[m].min()), np.ceil(u_perp[m].max()) + step, step)
    idx = np.digitize(u_perp, edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    out = np.array([np.nanmedian(V[m & (idx == b)], axis=0) if (m & (idx == b)).sum() >= min_n
                    else np.full(V.shape[1], np.nan) for b in range(1, len(edges))])
    return mid, out


m_ok = okp & np.isfinite(rot0).all(1)
mid, prof_rot = cross_profile(rot0, m_ok)
_, prof_pc1 = cross_profile(sc[..., 0].ravel(), m_ok)
fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
for k in range(3):
    ax[0].plot(mid, prof_rot[:, k] - np.nanmedian(prof_rot[:, k]), lw=1, label=f"Δ{'xyz'[k]}")
ax[0].set_ylabel("obrót sieci [°]"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(mid, prof_pc1[:, 0], "k", lw=1); ax[1].set_ylabel("PC1"); ax[1].set_xlabel("w poprzek pasków [px]")
ax[1].grid(alpha=0.3); plt.show()
""")

md(r"""
## 12. Dopasowanie w paśmie harmonik

Korelacja liczona tylko w paśmie `BAND` (rzut na wybrane harmoniki, dokładny także przy
zamaskowanych klatkach). Log-wiarygodność liczona z n = liczba stopni swobody pasma
(`N_EFF`), nie liczba klatek. Wybór pasma: tam, gdzie zgodność model–pomiar z sekcji 11
jest wysoka **i** jest duży udział różnic wariantów.
""")
code(r"""
BAND = [(21, 60)]
N_EFF = of.projector_dof(NW, BAND)
print(f"pasmo {BAND}: {N_EFF} stopni swobody")

rows, G_band = [], []
for k, (y, nm) in enumerate(zip(ref_profiles, ref_names)):
    Gv, Rv = of.variant_scores(model, y, G_refs[k], VARIANTS, 0.0, SENSE, None, BAND)
    Gv0, Rv0 = of.variant_scores(model, y, G_refs[k], VARIANTS, 0.0, SENSE, None, N_BG_HARM)
    dr, dl = of.margin_stats(Rv, N_EFF)
    vb = int(np.argmax(Rv))
    G_band.append(Gv[vb])
    rows.append(dict(profil=nm, r_pełne=Rv0.max(), r_pasmo=Rv.max(), wariant_pasmo=f"V{vb}",
                     wariant_pełne=f"V{int(np.argmax(Rv0))}", dr_pasmo=dr, dlogL_pasmo=dl,
                     mis_O_do_wspólnego=of.misorientation_deg(Gv[vb], G_band[0] if k else Gv[vb], O24)[0]))
G_band = np.stack(G_band)
display(pd.DataFrame(rows).set_index("profil").style.format(precision=4))
""")
code(r"""
Zs_b, Zc_b, simV_b, measC_b = variant_vs_class(BAND)
print("W PAŚMIE — symulacja: 1 − r między wariantami")
display(pd.DataFrame(simV_b, lblV, lblV).style.format("{:.4f}"))
print("W PAŚMIE — pomiar: 1 − r między klasami")
display(pd.DataFrame(measC_b).style.format("{:.4f}"))
print("W PAŚMIE — dopasowanie różnicowe klas do par wariantów")
display(class_pairs(Zs_b, Zc_b).style.format({"r_1": "{:.3f}", "r_2": "{:.3f}"}))
""")
md(r"""
### 12b. Czy „efekt θ” klasy 2 jest realny?

Skan θ dla profili referencyjnych (wspólny + klasy) w pełnym widmie i w paśmie. Dla każdego
profilu: r przy wspólnym θ̂ i przy własnym θ̂ oraz zysk log-wiarygodności. Duży zysk tylko
dla klasy 2 → jej profil różni się w sposób, którego obrót sieci nie oddaje.
""")
code(r"""
TH_GRID = th_hat + np.arange(-0.4, 0.4001, 0.05)
res_th = {}
for lab, spec, n_eff in [("pełne", N_BG_HARM, of.projector_dof(NW, N_BG_HARM)), ("pasmo", BAND, N_EFF)]:
    scn = cal.theta_scan(mp, ref_profiles, G_refs, TH_GRID, sense=SENSE, invert=INVERT,
                         sigma_conv_deg=sg_hat, n_bg_harm=spec, variants=VARIANTS, verbose=False)
    i0 = int(np.argmin(abs(TH_GRID - th_hat)))
    Rb = scn["R"].max(2)
    tab = pd.DataFrame({
        "θ̂ własne": [cal.peak(TH_GRID, scn["Lbest"][:, k]) for k in range(len(ref_names))],
        "r przy θ̂ wspólnym": Rb[i0], "r przy θ̂ własnym": Rb.max(0),
        "zysk ΔlogL": scn["Lbest"].max(0) - scn["Lbest"][i0]}, index=ref_names)
    res_th[lab] = (scn, tab)
    print(f"--- {lab} (n_eff = {n_eff})")
    display(tab.style.format(precision=4))
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
for a_x, lab in zip(ax, res_th):
    scn = res_th[lab][0]
    for k, nm in enumerate(ref_names):
        a_x.plot(TH_GRID, scn["R"].max(2)[:, k], ".-", label=nm)
    a_x.axvline(th_hat, color="0.5", ls=":"); a_x.set_title(lab); a_x.set_xlabel("θ [°]")
    a_x.set_ylabel("r_best"); a_x.legend(fontsize=7); a_x.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")
md("### 12c. Mapa wariantów w paśmie")
code(r"""
CHECKPOINT_B = f"mapfit_band{BAND[0][0]}-{BAND[-1][1]}_th{th_hat:.3f}_sg{sg_hat:.2f}.npz"
fitmap_b = rd.fit_map_variants(model, Y, valid, pix, g_ref, sense=SENSE, n_bg_harm=BAND,
                               variants=VARIANTS, checkpoint=CHECKPOINT_B, every=200)
dec_b = rd.variant_decision(fitmap_b, n_eff=N_EFF)
best_b, okb = dec_b["best"], dec_b["ok"]
print("udział wariantów (pasmo):", {f"V{v}": f"{np.mean(best_b[okb] == v):.1%}" for v in range(len(VARIANTS))})
print(f"mediana r_best: pełne {np.nanmedian(dec['r_best']):.3f}, pasmo {np.nanmedian(dec_b['r_best']):.3f}")
to_map_b = lambda v: np.where(okb, v, np.nan).reshape(H, W)
fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
ax[0].imshow(to_map_b(best_b), cmap="tab10", vmin=0, vmax=9, interpolation="nearest"); ax[0].set_title("wariant (pasmo)")
for a_x, im_, t_ in [(ax[1], to_map_b(dec_b["r_best"]), "r_best (pasmo)"),
                     (ax[2], to_map_b(np.log10(np.clip(dec_b["dlogL"], 1e-2, None))), "log10 ΔlogL (pasmo)"),
                     (ax[3], to_map_b(dec_b["dr"]), "Δr (pasmo)")]:
    im = a_x.imshow(im_); a_x.set_title(t_); plt.colorbar(im, ax=a_x)
plt.tight_layout(); plt.show()
print("klasy bez modelu (wiersze) vs wariant w paśmie (kolumny)")
display(pd.crosstab(labK.ravel()[okb], best_b[okb], rownames=["klasa"], colnames=["wariant"]))

rot0_b = rd.rotation_field(fitmap_b["G"][:, 0], g_ref)
m_b = okb & np.isfinite(rot0_b).all(1)
mid_b, prof_rot_b = cross_profile(rot0_b, m_b)
fig, ax = plt.subplots(figsize=(13, 3.5))
for k in range(3):
    ax.plot(mid, prof_rot[:, k] - np.nanmedian(prof_rot[:, k]), lw=0.8, alpha=0.5, label=f"Δ{'xyz'[k]} pełne")
    ax.plot(mid_b, prof_rot_b[:, k] - np.nanmedian(prof_rot_b[:, k]), lw=1.3, label=f"Δ{'xyz'[k]} pasmo")
ax.set_xlabel("w poprzek pasków [px]"); ax.set_ylabel("obrót sieci [°]"); ax.legend(ncol=2, fontsize=8)
ax.grid(alpha=0.3); plt.show()
""")


md(r"""
## 12d. Obrót sieci w paśmie względem wariantu GLOBALNEGO, bez gładkiego wygięcia

* wariant odniesienia VB = dominujący wariant w paśmie (obrót liczony względem V_VB·g_ref, a nie
  względem V0, który dla matrycy jest wariantem złym),
* gładkie wygięcie sieci (wielomian 2. stopnia) dopasowane tylko na czystej matrycy
  (ułamek pasków ≤ 0.08, z gładkiego pola PC1 — bez modelu) i odjęte od wszystkich pikseli.
""")
code(r"""
import domain_tests as dt

pc1_gp_mu, _ = gp_pc1.predict(XY)                        # gładkie pole PC1 (sekcja 3, bez modelu)
frac_pc1 = dt.stripe_fraction(pc1_gp_mu, okb)            # ułamek pasków bez modelu, 0..1
VB = int(np.bincount(best_b[okb]).argmax())              # dominujący wariant w paśmie
G_m = VARIANTS[VB] @ g_ref                               # orientacja matrycy (w paśmie)
rot_b = rd.rotation_field(fitmap_b["G"][:, VB], G_m)     # (P,3) [deg]
cov_b = fitmap_b["C"][:, VB]                             # (P,3,3) [deg^2], Laplace
ok_rb = okb & np.isfinite(rot_b).all(1)
is_matrix = ok_rb & (frac_pc1 <= 0.08)

A2 = np.column_stack([np.ones(H * W), XY, XY ** 2, XY[:, 0] * XY[:, 1]])
bend = np.zeros_like(rot_b)
for k in range(3):
    c_, *_ = np.linalg.lstsq(A2[is_matrix], rot_b[is_matrix, k], rcond=None)
    bend[:, k] = A2 @ c_
rot_bd = rot_b - bend
u_blk = dt.block_ids(u_perp, width=10.0)                 # bloki w poprzek pasków (~1 okres)
print(f"VB = V{VB}; czysta matryca: {is_matrix.sum()} pikseli, rdzeń pasków (frac ≥ 0.6): "
      f"{(ok_rb & (frac_pc1 >= 0.6)).sum()}, bloków: {len(np.unique(u_blk))}")

mid3, prof3 = cross_profile(rot_bd, ok_rb)
fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
for k in range(3):
    ax[0].plot(mid3, prof3[:, k] - np.nanmedian(prof3[:, k]), lw=1.2, label=f"Δ{'xyz'[k]}")
ax[0].set_ylabel("obrót bez wygięcia [°]"); ax[0].legend(); ax[0].grid(alpha=0.3)
_, prof_fr = cross_profile(frac_pc1, ok_rb); ax[1].plot(mid3, prof_fr[:, 0], "k"); ax[1].set_ylabel("ułamek pasków")
_, prof_bd = cross_profile(bend, ok_rb)
for k in range(3):
    ax[2].plot(mid3, prof_bd[:, k], lw=1, label=f"wygięcie Δ{'xyz'[k]}")
ax[2].set_ylabel("gładkie wygięcie [°]"); ax[2].set_xlabel("w poprzek pasków [px]"); ax[2].legend(); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

md(r"""
## 13. Wspólne pole ukryte: PC1 + obrót sieci (GP)

Model: y_j(x) = c_j + a_j · f(x) + szum, jedno pole f ~ GP (jądro anizotropowe), dla
y = (PC1, Δx, Δy, Δz). Szum obrotu: wariancja z Laplace'a na piksel (+ nugget).
Identyfikowalne: kierunek i szerokości korelacji pola, **stosunki ładunków** (obrót sieci
na jednostkę PC1 → oś obrotu), mapa f z niepewnością. Skala f (co znaczy "czysty pasek")
nie jest identyfikowalna — kotwiczymy ją najjaśniejszymi paskami, co daje **dolne
oszacowanie** obrotu dla czystej domeny.
""")
code(r"""
ROT_SOURCE = "band_ref"      # "full" (sekcja 8), "band" (12c, wariant 0) albo "band_ref" (12d)
if ROT_SOURCE == "full":
    R_use, V_use, ok_use = rot0, var0, okp
elif ROT_SOURCE == "band":
    R_use, V_use, ok_use = rot0_b, np.diagonal(fitmap_b["C"][:, 0], axis1=1, axis2=2), okb
else:
    R_use, V_use, ok_use = rot_bd, np.diagonal(cov_b, axis1=1, axis2=2), ok_rb
Ymat = np.column_stack([sc[..., 0].ravel(), R_use - np.nanmedian(R_use[ok_use], axis=0)])
NV = np.column_stack([np.zeros(H * W), V_use])
m13 = ok_use & np.isfinite(Ymat).all(1) & np.isfinite(NV).all(1)

# szybka kontrola: regresja liniowa składowych obrotu na PC1
for k in range(3):
    sl, ic = np.polyfit(Ymat[m13, 0], Ymat[m13, 1 + k], 1)
    print(f"Δ{'xyz'[k]} ≈ {sl:+.4f} · PC1   (r = {np.corrcoef(Ymat[m13, 0], Ymat[m13, 1 + k])[0, 1]:+.3f})")

t0 = time.perf_counter()
lgp = sg.SharedLatentGP("matern32").fit(
    XY[m13], Ymat[m13], noise_var=NV[m13], n_sub=500, n_pred=1200, seed=SEED,
    alpha0=gp_pc1.params["alpha_deg"], l0=(gp_pc1.params["l_par"], gp_pc1.params["l_perp"]),
    names=["PC1", "Δx", "Δy", "Δz"])
print(f"[{time.perf_counter() - t0:.0f} s]")
""")
code(r"""
f_mu, f_sd = lgp.predict_latent(XY)
a_rot = lgp.params["a"][1:]                                  # [deg na jednostkę f]
axis = a_rot / np.linalg.norm(a_rot)
f_mat = np.median(f_mu[m13])                                 # matryca (większość pikseli)
f_hi = np.quantile(f_mu[m13], 0.99)                          # najjaśniejsze paski
tilt_lb = np.linalg.norm(a_rot) * (f_hi - f_mat)
th_tw = 2 * np.degrees(np.arctan(0.40019998 / 0.39930001)) - 90
print(f"oś obrotu związana z paskami (układ próbki): {np.round(axis, 3)}")
print(f"obrót na jednostkę f: {np.linalg.norm(a_rot):.4f}°; między matrycą a najjaśniejszymi paskami: "
      f"{tilt_lb:.3f}°  (dolne oszacowanie; teoria domen 90°: {th_tw:.3f}°)")
print("kąty osi z osiami próbki x, y, z [°]:", np.round(np.degrees(np.arccos(np.abs(axis))), 1))

frac = np.clip((f_mu - f_mat) / (f_hi - f_mat), 0, 1)
pred = lgp.predict_outputs(XY)
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
mk = lambda v: np.where(m13, v, np.nan).reshape(H, W)
for a_x, im_, t_, kw in [(ax[0, 0], f_mu.reshape(H, W), "pole f (średnia)", dict(cmap="RdBu_r")),
                         (ax[0, 1], f_sd.reshape(H, W), "pole f (sd)", {}),
                         (ax[0, 2], frac.reshape(H, W), "ułamek pasków (zakotwiczony, orientacyjnie)", dict(vmin=0, vmax=1)),
                         (ax[1, 0], mk(Ymat[:, 3]), "Δz obserwowane [°]", dict(cmap="RdBu_r", vmin=-0.08, vmax=0.08)),
                         (ax[1, 1], pred[:, 3].reshape(H, W), "Δz z pola wspólnego [°]", dict(cmap="RdBu_r", vmin=-0.08, vmax=0.08)),
                         (ax[1, 2], mk(Ymat[:, 3] - pred[:, 3]), "reszta Δz [°]", dict(cmap="RdBu_r", vmin=-0.08, vmax=0.08))]:
    im = a_x.imshow(im_, **kw); a_x.set_title(t_); plt.colorbar(im, ax=a_x)
plt.tight_layout(); plt.show()
""")
md(r"""
**Jak czytać:**
* **oś obrotu** — teoria dla Twojej orientacji (osie sześcianu ≈ osie próbki): domeny a1/a2 →
  oś ≈ z próbki, domeny c/a → oś ≈ x próbki. Oś pośrednia może znaczyć mieszankę typów
  domen albo artefakt dopasowania (porównaj ROT_SOURCE = "full" i "band"),
* **dolne oszacowanie obrotu** — jeśli zbliża się do 0.129°, dane zgadzają się z domenami 90°
  przy c/a z mastera; jeśli zostaje wyraźnie poniżej mimo braku rozcieńczenia (dane bez
  binningu!), kryształ jest mniej tetragonalny,
* **reszta Δz** bez struktury pasków → jedno pole f wystarcza (PC1 i obrót opisują to samo zjawisko).
""")
code(r"""
OUT2 = "real_results_part2.npz"
np.savez_compressed(OUT2, band=np.array(BAND), n_eff=N_EFF, best_band=best_b.reshape(H, W),
                    r_best_band=dec_b["r_best"].reshape(H, W), dlogL_band=dec_b["dlogL"].reshape(H, W),
                    rot0=rot0.reshape(H, W, 3), rot0_band=rot0_b.reshape(H, W, 3),
                    latent_f=f_mu.reshape(H, W), latent_sd=f_sd.reshape(H, W),
                    loadings=lgp.params["a"], latent_axis=axis, tilt_lower_bound=tilt_lb)
print("zapisano", OUT2)
""")


md(r"""
# Część 3: czym są paski? Testy hipotez na grupach pikseli (sekcje 14–20)

Pojedynczy piksel nie rozstrzyga wariantu (marginesy ≈ szum + niedopasowanie), więc tu
porównujemy **grupy**: rdzeń pasków vs czysta matryca (z gładkiego pola PC1, bez modelu),
z niepewnościami z blokowego bootstrapu (bloki ~ jeden okres pasków w poprzek).

* **14** — dowód o wariancie jako funkcja ułamka pasków (ekstrapolacja do czystego paska),
* **15** — czy klasy k-średnich to paski, czy ciągłe tło (geometria klas, wygięcie sieci),
* **16** — różnicowe dopasowanie PROFILI grup: obrót sieci czy inny wariant?,
* **17** — oś obrotu paski − matryca z kowariancją: test hipotez o osi,
* **18** — polaryzacja: gdzie żyje różnica i czy model ją odtwarza,
* **19** — dane bez binningu (opcjonalnie): ostry profil w poprzek ścian,
* **20** — geometria ścian, obrót obraz ↔ próbka, c/a z amplitudy obrotu (punkt E).
""")

md(r"""
## 14. Czy paski są innym wariantem? Dowód vs ułamek pasków

Dla każdego wariantu j ≠ VB: dL_j = logL_j − logL_VB na piksel, regresja na ułamek pasków
(frac): dL_j = b0 + b1·frac. **b0** = preferencja matrycy, **b0 + b1** = ekstrapolacja do
czystego paska. Jeśli paski są wariantem j, to b1 > 0 istotnie i b0 + b1 > 0 > b0. Poprzednia
wersja tego testu (suma po klasie 2) mieszała czyste i rozcieńczone piksele oraz zależała
od tego, że klasa 2 faktycznie jest paskami.
""")
code(r"""
logL_b = dec_b["logL"]
rows14, dL14 = [], {}
for j in range(len(VARIANTS)):
    if j == VB:
        continue
    dLj = logL_b[:, j] - logL_b[:, VB]
    dL14[j] = dLj
    ev = dt.evidence_vs_fraction(dLj, frac_pc1, u_blk, okb, n_boot=400, seed=SEED)
    rows14.append(dict(wariant=f"V{j}", matryca_b0=ev["b0"], nachylenie_b1=ev["b1"], se_b1=ev["se_b1"],
                       z_b1=ev["b1"] / ev["se_b1"], czysty_pasek=ev["pure"], se_pasek=ev["se_pure"]))
tab14 = pd.DataFrame(rows14).set_index("wariant")
print(f"referencja: V{VB} (matryca); bloków bootstrapu: {ev['n_blocks']}, pikseli: {ev['n']}")
display(tab14.style.format(precision=3))

edges = np.linspace(0, 1, 7)
fig, ax = plt.subplots(figsize=(8, 4))
for j, dLj in dL14.items():
    mids, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = okb & np.isfinite(dLj) & (frac_pc1 >= lo) & (frac_pc1 < hi + (hi == 1))
        if s.sum() >= 20:
            mids.append(0.5 * (lo + hi)); means.append(dLj[s].mean())
    ax.plot(mids, means, "o-", label=f"V{j} − V{VB}")
ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("ułamek pasków (model-free)"); ax.set_ylabel("średnie ΔlogL na piksel")
ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.show()
""")
md(r"""
**Jak czytać:** wariant j jest kandydatem na wariant pasków tylko wtedy, gdy `z_b1` ≳ 3 **i**
`czysty_pasek` > 0 (czyli paski wolą j bardziej niż VB). Brak takiego wariantu przy wyraźnym
`z_b1` dla jakiegokolwiek j oznacza: dane (z tym modelem) nie odróżniają pasków od matrycy
wariantem — paski to wtedy obrót sieci / inna cecha, nie inny wariant. Uwaga: niedopasowanie
jest skorelowane między pikselami, błąd z bootstrapu blokowego jest więc uczciwszy niż
błąd z pojedynczych pikseli, ale i tak zakłada, że bloki są niezależne.
""")

md(r"""
## 15. Czy klasy k-średnich to paski, czy ciągłe tło?

* geometria klas: szerokość typowa (paski ≈ 2–4 px, regiony szeroko) i spójność przestrzenna,
* η²: jaka część wariancji gładkich pól (położenie, wygięcie sieci, ułamek pasków) jest
  wyjaśniona samą etykietą klasy. Klasa-tło wyjaśnia dużo wariancji wygięcia/położenia,
  klasa-paski — ułamka pasków.
""")
code(r"""
Lmap = np.where(okb, labK.ravel(), -1).reshape(H, W)
geo = dt.class_geometry(Lmap)
geo["średni ułamek pasków"] = [frac_pc1[(labK.ravel() == c) & okb].mean() for c in geo.index]
display(geo.style.format(precision=3))
fields = {"x": XY[:, 0], "y": XY[:, 1], "wygięcie Δx": bend[:, 0], "wygięcie Δy": bend[:, 1],
          "wygięcie Δz": bend[:, 2], "ułamek pasków": frac_pc1}
eta = pd.Series({nm: dt.eta_squared(labK.ravel(), f, okb) for nm, f in fields.items()}, name="η² (klasy)")
display(eta.to_frame().style.format("{:.3f}"))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
ax[0].imshow(np.where(Lmap >= 0, Lmap, np.nan), cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
ax[0].set_title("klasy (k-średnich)")
ax[1].imshow(np.where(Lmap >= 0, Lmap, np.nan), cmap="tab10", vmin=0, vmax=9, interpolation="nearest", alpha=0.6)
cs = ax[1].contour(bend[:, 2].reshape(H, W), levels=8, colors="k", linewidths=0.8)
ax[1].set_title("klasy + poziomice wygięcia Δz")
im = ax[2].imshow(frac_pc1.reshape(H, W), vmin=0, vmax=1); ax[2].set_title("ułamek pasków (bez modelu)")
plt.colorbar(im, ax=ax[2]); plt.tight_layout(); plt.show()
""")
md(r"""
**Jak czytać:** klasa o dużej szerokości, niskim `średnim ułamku pasków` i dużym η² względem
wygięcia/położenia to tło (k-średnich pocięło ciągły gradient), nie domena. Klasa wąska
z wysokim ułamkiem pasków to paski. Jeśli klasa-paski jest jedna, a pozostałe są tłem — liczba
„4 klas” z k-średnich nie odpowiada 4 domenom krystalograficznym (np. polaryzacja może być
w danych niewidoczna, patrz sekcja 18).
""")

md(r"""
## 16. Różnicowe dopasowanie profili grup: obrót sieci czy inny wariant?

Dla pary grup (a, b): D = ŷ_a − ŷ_b (średnie profile po rzucie na pasmo, jednostkowa norma).
Wspólne niedopasowanie modelu w dużej części się znosi. Hipotezy względem orientacji matrycy G_m:
**R** — D ≈ J·δ (mały obrót, 3 składowe), **V_j** — D ≈ s·(μ̂(V_j G_m) − μ̂(G_m)) (inny wariant;
s ~ ułamek domeny), **RV** — oba. `expl` = ułamek wyjaśnionej różnicy.
Wiersze kontrolne: matryca lewa vs prawa (spodziewany tylko gładki gradient) i niezależne
połowy mapy (powtarzalność).
""")
code(r"""
stripe_core, matrix_pure = dt.stripe_masks(frac_pc1, ok_rb, core=0.6, bg=0.08)
left = XY[:, 0] < W / 2
GR = {"paski (rdzeń)": stripe_core, "matryca": matrix_pure,
      "paski lewa": stripe_core & left, "matryca lewa": matrix_pure & left,
      "paski prawa": stripe_core & ~left, "matryca prawa": matrix_pure & ~left}
for c in range(labK.max() + 1):
    GR[f"klasa {c}"] = ok_rb & (labK.ravel() == c)
gprof = {nm: dt.group_profile(Y, valid, m) for nm, m in GR.items()}
pairs16 = [("paski (rdzeń)", "matryca"), ("paski lewa", "matryca lewa"), ("paski prawa", "matryca prawa"),
           ("matryca lewa", "matryca prawa")] + \
          [(f"klasa {i}", f"klasa {j}") for i in range(labK.max() + 1) for j in range(i + 1, labK.max() + 1)]
rows16, diff_res = [], {}
for na_, nb_ in pairs16:
    pa, ka, cnta = gprof[na_]; pb, kb, cntb = gprof[nb_]
    out = dt.differential_fit(model, G_m, pa, pb, mask=ka & kb, spec=BAND, variants=VARIANTS, phi0=0.0, sense=SENSE)
    diff_res[(na_, nb_)] = out
    dpix = np.nanmedian(rot_bd[GR[na_] & ok_rb], 0) - np.nanmedian(rot_bd[GR[nb_] & ok_rb], 0)
    rows16.append({"a": na_, "b": nb_, "n_a": cnta, "n_b": cntb, "|D|": out["D_norm"],
                   "expl R": out["expl_rot"], "δx": out["delta_deg"][0], "δy": out["delta_deg"][1],
                   "δz": out["delta_deg"][2], "|δ|": np.linalg.norm(out["delta_deg"]),
                   "δ pikselowe |.|": np.linalg.norm(dpix), "najlepszy V": out["best_variant"],
                   "s": out["s_best"], "expl V": out["expl_variant"], "expl RV": out["expl_joint"]})
tab16 = pd.DataFrame(rows16)
sim_pairs = np.sqrt(2 * simV_b[np.triu_indices(len(VARIANTS), 1)])
print(f"dla porównania: |D| symulowanych wariantów w paśmie: {sim_pairs.min():.2f}–{sim_pairs.max():.2f}; "
      f"obrót 0.129° (wzór): zob. sekcja 20")
display(tab16.style.format(precision=3))
""")
md(r"""
**Jak czytać:**
* `expl R` ≫ `expl V` → różnica profili to obrót sieci (δ = obrót, rozcieńczony ułamkiem domeny),
* `expl V` ≫ `expl R` i `s` ~ 0.3–1 → paski to inny wariant (`najlepszy V`, `s` ~ ułamek),
* oba niskie → różnica nie jest ani jednym, ani drugim (np. polaryzacja poza pasmem modelu,
  efekty niekrystalograficzne),
* wiersz `matryca lewa vs prawa` i `paski lewa vs prawa` pokazują, ile `expl` daje samo
  wygięcie / szum; wiersze „lewa”/„prawa” dla pasków vs matrycy sprawdzają powtarzalność
  (δ i `najlepszy V` powinny się zgadzać),
* `|δ|` z profili vs `δ pikselowe`: dwa niezależne oszacowania tej samej wielkości.
""")

md(r"""
## 17. Oś obrotu paski − matryca: test hipotez o osi

Różnica obrotu sieci (mediana pasków − mediana matrycy, po odjęciu wygięcia, względem V_VB) z
kowariancją z blokowego bootstrapu. Test: czy wektor jest równoległy do jednej z osi sześcianu
(w układzie próbki)? Hipotezy dla bliźniaków 90° na {110} → oś [001]-kryształu; na {101} → oś
trzecia (prostopadła do obu osi c). χ² z 2 stopniami swobody (amplituda swobodna).
""")
code(r"""
d_rot, cov_rot, n_s, n_m = dt.group_difference(rot_bd, stripe_core, matrix_pure, u_blk, n_boot=500, seed=SEED)
axes_c, axnames = dt.cube_axes_sample(G_m)
t17 = dt.axis_hypothesis_test(d_rot, cov_rot, axes_c, axnames)
sd_rot = np.sqrt(np.diag(cov_rot))
print(f"paski ({n_s}) − matryca ({n_m}): d = {np.round(d_rot, 4)} °,  sd (bootstrap) = {np.round(sd_rot, 4)} °,  "
      f"|d| = {np.linalg.norm(d_rot):.4f}°")
print(f"test d = 0: chi2 = {t17.attrs['chi2_zero']:.1f}, p = {t17.attrs['p_zero']:.2g}  (3 stopnie swobody)")
dil = frac_pc1[stripe_core].mean() - frac_pc1[matrix_pure].mean()
print(f"średni ułamek pasków: rdzeń {frac_pc1[stripe_core].mean():.2f}, matryca {frac_pc1[matrix_pure].mean():.2f}; "
      f"|d| / różnica ułamków = {np.linalg.norm(d_rot) / dil:.4f}° (orientacyjna korekta na rozcieńczenie)")
display(t17.style.format({"amplituda_deg": "{:+.4f}", "kąt_do_d": "{:.1f}", "chi2": "{:.2f}", "p": "{:.3f}"}))
d_prof = diff_res[("paski (rdzeń)", "matryca")]["delta_deg"]
print("niezależnie, z różnicy profili (sekcja 16): δ =", np.round(d_prof, 4), "°; kąty do osi sześcianu [°]:",
      np.round(np.degrees(np.arccos(np.clip(np.abs(axes_c @ d_prof) / np.linalg.norm(d_prof), 0, 1))), 1))
""")
md(r"""
**Jak czytać:** oś o największym `p` (a przy tym `p` ≳ 0.05) jest zgodna z danymi; osie z
`p` ≪ 0.05 są odrzucone. Jeśli żadna nie jest zgodna — oś jest mieszanką (np. dwa typy
ścian) albo pomiar jest zanieczyszczony. Zgodność oszacowań z pikseli i z profili wzmacnia wynik.
`kąt_do_d` pokazuje, jak daleko jest d od danej osi. Oś **[001]-kryształu** → domeny
c w płaszczyźnie (a1/a2); osie [100]/[010] → domeny c/a (c wzdłuż normalnej lub w płaszczyźnie).
""")

md(r"""
## 18. Polaryzacja: gdzie żyje różnica i czy model ją odtwarza

Pary wariantów o tej samej osi c i przeciwnej polaryzacji (dezorientacja ≈ 0 mod D4, ≠ 0 mod C4).
Dla nich i dla par 90°: ułamek energii różnicy profili w pasmach harmonik oraz wskaźnik
wykrywalności = Σ(udział × zgodność²), czyli ile różnicy leży tam, gdzie model zgadza się
z pomiarem (zgodność model–pomiar z profilem wspólnym).
""")
code(r"""
Zs18, _, _, _ = variant_vs_class(N_BG_HARM)
Pm18 = of.bg_projector(NW, N_BG_HARM)
a18 = Pm18 @ ref_profiles[0]
b18 = Pm18 @ model.profiles(G_refs[0][None], 0.0, SENSE)[0]
b18 = b18 * (a18 @ b18) / (b18 @ b18)
tabc = coherence_table(a18, b18, Zs18)
print("zgodność model–pomiar w pasmach (profil wspólny):")
display(tabc.set_index("harmoniki")[["zgodność"]].T.style.format("{:.2f}"))

pol = dt.polarity_pairs(VARIANTS, g_ref)
oth = [(i, j) for i in range(len(VARIANTS)) for j in range(i + 1, len(VARIANTS)) if (i, j) not in pol]
det_pol = dt.pair_detectability(Zs18, pol, tabc["zgodność"].values, BANDS_DIAG)
det_oth = dt.pair_detectability(Zs18, oth, tabc["zgodność"].values, BANDS_DIAG)
print(f"polaryzacja ({len(pol)} pary), średni wskaźnik wykrywalności: {det_pol.wskaźnik_wykrywalności.mean():.3f}")
display(det_pol.style.format(precision=3))
print(f"pary 90° ({len(oth)}), średni wskaźnik wykrywalności: {det_oth.wskaźnik_wykrywalności.mean():.3f}")
display(det_oth.style.format(precision=3))
""")
md(r"""
Jeśli wskaźnik dla polaryzacji jest wyraźnie niższy niż dla par 90° — różnica polaryzacji leży
w pasmach, w których model nie zgadza się z pomiarem; przy tym modelu polaryzacja jest
niewyznaczalna i można to uczciwie stwierdzić, a „4. klasa” może być domeną 180°,
niewidoczną w danych / modelu. Podobne wskaźniki → polaryzacja jest w zasadzie wykrywalna
i brak jej w wynikach wymaga innego wyjaśnienia.
""")

md(r"""
## 19. Dane bez binningu: ostry profil w poprzek ścian (opcjonalne)

Ustaw `TIF_FULL` na stos 372 × 372 (sekcja 0). Profile pikseli uśredniane są wzdłuż
pasków (odcinek ± `HALF_LEN` px, kierunek α z GP sekcji 3) w przedziałach współrzędnej u
w poprzek pasków (1 px oryginalny = 1/6 piksela zbinowanego). Dla każdego u: 6 wariantów
w paśmie, obrót względem V_VB, ΔlogL. Płaskie plateau obrotu w paskach = prawdziwy obrót
bez rozcieńczenia binningiem; szerokość schodka = szerokość ściany.
""")
code(r"""
HALF_LEN, DU, U_STRIDE = 90, 1.0, 1
if TIF_FULL is None:
    print("TIF_FULL = None — pomijam sekcję 19")
else:
    Yfull, _, raw_full = rd.load_stack(TIF_FULL, OMEGA_STEP)
    Hf, Wf, _ = Yfull.shape
    assert (Hf, Wf) == (H * BIN, W * BIN), f"oczekiwano {(H * BIN, W * BIN)}, jest {(Hf, Wf)}"
    valid_full = rd.valid_mask(raw_full) & np.kron(clean, np.ones((BIN, BIN), bool))[..., None]
    alpha_s = gp_pc1.params["alpha_deg"]
    cenU, profU, maskU, cntU = dt.cross_stripe_profiles(Yfull, valid_full, alpha_s, half_length=HALF_LEN, du=DU)
    sel_u = np.arange(0, len(cenU), U_STRIDE)
    print(f"{len(cenU)} przedziałów u, ~{int(np.median(cntU))} pikseli w przedziale; dopasowuję {len(sel_u)} ...")
    t0 = time.perf_counter()
    ser = dt.fit_series(model, g_ref, profU[sel_u], maskU[sel_u], BAND, VARIANTS, VB, sense=SENSE)
    ser.insert(0, "u_px", cenU[sel_u])
    print(f"[{time.perf_counter() - t0:.0f} s]")
    a_s = np.deg2rad(alpha_s)
    u_c = -np.sin(a_s) * (W - 1) / 2 + np.cos(a_s) * (H - 1) / 2           # środek mapy zbinowanej
    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for k, nm in enumerate(["dx", "dy", "dz"]):
        ax[0].plot(ser.u_px, ser[nm] - ser[nm].median(), lw=1, label=f"Δ{nm[1]}")
    ax[0].set_ylabel("obrót względem matrycy [°]"); ax[0].legend()
    for j in range(len(VARIANTS)):
        if j != VB:
            ax[1].plot(ser.u_px, ser[f"dL{j}"], lw=0.8, label=f"V{j} − V{VB}")
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_ylabel("ΔlogL"); ax[1].legend(ncol=5, fontsize=7)
    ax[2].plot(ser.u_px, ser.r_best, "k", lw=1); ax[2].set_ylabel("r_best (pasmo)")
    ax[3].plot(BIN * (mid - u_c), prof_pc1[:, 0], "k", lw=1); ax[3].set_ylabel("PC1 (zbinowane, do porównania)")
    ax[3].set_xlabel("u w poprzek pasków [px oryginalne]")
    for a_x in ax: a_x.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
""")
md(r"""
**Jak czytać:** wysokość plateau Δz/Δx/Δy w paskach vs ~0.1° z danych zbinowanych mówi, ile
amplitudy zjadał binning; jeśli plateau sięga ~0.13° — kryształ zgodny z masterem (c/a ≈ 1.0023);
jeśli wyraźnie niżej mimo ostrych ścian — mniejsza tetragonalność. Skoki `ΔlogL` dla jakiegoś
V_j w paskach, a nie w matrycy → ten wariant jest kandydatem na wariant pasków (uśrednianie
wzdłuż pasków daje SNR kilkadziesiąt razy lepszy niż pojedynczy piksel).
""")

md(r"""
## 20. Geometria ścian, obrót obraz ↔ próbka i c/a (punkt E)

**(a) Hipotezy o ścianach.** Dla orientacji matrycy: sześć płaszczyzn {110} i {101}; dla każdej
osie c obu domen, przewidywana oś obrotu sieci (trzecia oś sześcianu, leży w płaszczyźnie
ściany), ślad ściany na powierzchni (układ próbki EMsoftu) i obrót obraz ↔ próbka psi, jaki z
niego wynika przy zmierzonym kierunku pasków α (obraz; druga kolumna: przy odbitym obrazie).
`chi2`/`p`: zgodność przewidywanej osi z d z sekcji 17. **To zawęża kandydatów, nie rozstrzyga**:
psi musisz porównać z geometrią skanu (patrz lista poniżej).

**(b) c/a z amplitudy obrotu.** ω = 2·atan(c/a) − 90° (c/a − 1 ≈ ω w radianach). Zmierzona amplituda to
dolne oszacowanie (rozcieńczenie binningiem i mieszaniem), stąd kolumna „czystość”.
""")
code(r"""
alpha_img = gp_pc1.params["alpha_deg"]
tw = dt.twin_hypotheses(G_m, alpha_img, d=d_rot, cov=cov_rot)
display(tw.style.format({"ślad_w_próbce_deg": "{:.1f}", "nachylenie_ściany_deg": "{:.1f}", "psi_obraz_deg": "{:.1f}",
                         "psi_obraz_odbity_deg": "{:.1f}", "chi2": "{:.2f}", "p": "{:.3f}"}))

ca_master = 0.40019998 / 0.39930001
print(f"master: c/a = {ca_master:.5f} → ω = {dt.rotation_from_c_over_a(ca_master):.3f}°")
rows20 = []
amps = {"paski − matryca (piksele)": np.linalg.norm(d_rot), "profile grup (sekcja 16)": np.linalg.norm(d_prof)}
if TIF_FULL is not None:
    amps["plateau z danych bez binningu (sekcja 19)"] = float(np.linalg.norm(
        [np.percentile(ser[c] - ser[c].median(), 95) for c in ("dx", "dy", "dz")]))
for nm, amp in amps.items():
    for pur in (1.0, 0.8, 0.6, 0.4):
        om = amp / pur
        rows20.append({"źródło amplitudy": nm, "amplituda [°]": amp, "czystość": pur,
                       "ω [°]": om, "c/a": float(dt.c_over_a_from_rotation(om))})
display(pd.DataFrame(rows20).style.format({"amplituda [°]": "{:.4f}", "czystość": "{:.1f}", "ω [°]": "{:.4f}", "c/a": "{:.5f}"}))
""")
md(r"""
**(c) Czego nie da się wyliczyć z danych — do zebrania (wpływa na interpretację):**
1. **Skład x** w KTa₁₋ₓNbₓO₃ i **temperatura pomiaru** względem temperatury Curie. Blisko T_c c/a i
   polaryzacja są mniejsze niż w strukturze mastera (c/a = 1.0023), więc mniejsze będą zarówno
   obrót sieci, jak i różnice profili wariantów.
2. **Skąd struktura w masterze:** parametry sieci i przesunięcia jonów (polaryzacja), temperatura
   struktury. Jeśli master jest dla innej temperatury/składu — nowe mastery (c/a, przesunięcia)
   i emulator GP po tych parametrach to naturalny następny krok.
3. **Geometria skanu:** czy obrazy były rejestrowane / de-rotowane do układu próbki, który z kierunków
   obrazu odpowiada osi pochylenia, czy obraz jest odbity. Z tego wynika psi (kolumna w tabeli (a))
   i wybór między hipotezami ścian.
4. **Dryf próbki między klatkami** (korekcja obrazu) — psuje wąskie struktury w profilach i
   jest zwykle ukryty w niedopasowaniu modelu.
""")
code(r"""
OUT3 = "real_results_part3.npz"
np.savez_compressed(OUT3, frac_pc1=frac_pc1, VB=VB, G_m=G_m, rot_bd=rot_bd, bend=bend,
                    d_rot=d_rot, cov_rot=cov_rot, d_prof=d_prof,
                    evidence_table=tab14.reset_index().to_numpy(dtype=object),
                    diff_table=tab16.to_numpy(dtype=object))
tab14.to_csv("part3_evidence.csv"); tab16.to_csv("part3_differential.csv"); t17.to_csv("part3_axis_test.csv")
print("zapisano", OUT3, "oraz part3_*.csv")
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "real_data.ipynb")
print(len(C), "komórek")
