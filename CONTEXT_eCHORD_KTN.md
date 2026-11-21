# Kontekst: orientacje i domeny KTN z profili eCHORD (stan na 2026-11-18)

Dokument do przekazania następnemu czatowi. Zawiera: cel, konwencje, opis plików, wyniki
symulacji i prawdziwych danych, otwarte pytania, potencjał procesów gaussowskich i kolejne kroki.
Wszystko, co oznaczone **[hipoteza]**, nie jest rozstrzygnięte.

## 1. Cel i podejście

Wyznaczanie orientacji (domen) KTN (KTa₁₋ₓNbₓO₃, pseudoregularny, c/a ≈ 1.0023) z profili eCHORD
dla całej mapy pikseli, a docelowo analiza przestrzenna wyników procesami gaussowskimi.
Ścieżka: model do przodu (próbkowanie mastera po stożku, `cone_profiles`) → regresja
profil → orientacja (korelacja, 6 wariantów pseudosymetrii) → kalibracja θ → mapa → testy
hipotez o domenach → GP przestrzenny. Emulator GP profili jest gotowy, ale do problemu odwrotnego
okazał się niepotrzebny (sekcja 4).

## 2. Pomiar, master, konwencje

- Próbka pochylona o θ ≈ 40° (po kalibracji θ̂ ≈ 40.5°), obrót co Δω = 2°, 180 klatek.
  Stos `KTN_014_s2_fin_crop_372x372_bin_62x62.tif`: (180, 62, 62), uint16 (binning 6×6
  ze stosu 372×372; oryginał bez binningu nie był jeszcze używany).
- Dane: 554 wartości nasycone (65 535) w 345 pikselach (jasne klatki ω ≈ 138–232° i obiekt
  przy (31,31)), 67 zer w 19 pikselach (narożniki), artefakty (jasny obiekt, ciemne plamki).
  Czystych pikseli: 3675 / 3844. SNR pojedynczego piksela (z korelacji sąsiadów) ≈ 12.
- Master: `masterKTN_014_01.h5`, grupa 99 (P4mm), niecentrosymetryczna, a = 0.3993, c = 0.4002 nm
  (c/a = 1.00225 → obrót sieci między domenami 90° ω = 2·atan(c/a) − 90° = 0.129°),
  mLPNH (4, 1025, 1025): 4 kanały to pozycje atomów, sumowane (kanał 1 ≈ 55% natężenia).
  Brak osi energii, więc „energie: brak, wybór: max” jest niegroźne.
- Konwencje: kwaterniony EMsoftu (skalar pierwszy, P = −1, obrót pasywny próbka → kryształ),
  Bunge (φ1, Φ, φ2); osie kryształu w układzie próbki = wiersze macierzy g.
- **sense i invert nie są wyznaczalne z profili** (lustra 4mm: `mirror_x = mirror_diag = 1.0`):
  ustala je konwencja metody słownikowej. φ0 = 0 (błąd φ0 wchłania się w φ1). Łamanie prawa
  Friedla jest słabe: korelacja M(d) z M(−d) = 0.953.
- Pseudosymetria: każda orientacja ma **6 wariantów** (3 wybory osi c × 2 polaryzacje),
  profile różnią się o 1 − r ≈ 0.02–0.04. Warianty V_j to reprezentanci warstw O/C4
  (`of.pseudo_variants()`), V0 = tożsamość, V1 = ta sama oś c z odwróconą polaryzacją.

## 3. Pliki

Model i regresja:
- `echord_cone.py` — model do przodu (MasterPattern, `cone_profiles`, konwersje orientacji).
- `gp_echord.py`, `gp_echord_gpytorch.py`, `train_gp.py`, `gp_ktn_th40.npz` — emulator GP profili
  (redukcja do profilu kanonicznego F(ψ; Φ, φ2), jądro Matérn 3/2 na macierzach, suma po C4).
- `orient_fit.py` — rdzeń: `ConeModel`/`GPModel`; bank profili kanonicznych i szukanie globalne
  przez FFT (korelacja po wszystkich φ1 naraz); `refine` (least_squares po wektorze obrotu,
  gain eliminowany analitycznie, niepewność Laplace'a); warianty (`pseudo_variants`,
  `variant_scores`, `margin_stats`); rzuty tła (`bg_projector` = usuń harmoniki 0..K,
  `band_projector` = zostaw pasma, dokładne przy maskach); `fit_profile`.
- `calibration.py` — kalibracja θ (i σ_conv) z danych bez etykiet: S(θ) = Σ max_wariant logL.
- `mismatch.py` — generatory niedopasowania do symulacji (θ, bicie stolika, γ, wagi kanałów).
- `run_inversion_test.py` — CLI do testów na symulacjach.

Prawdziwe dane i analiza przestrzenna:
- `real_data.py` — wczytanie stosu, maski nasycenia, mapy jakości, analiza bez modelu (reszty
  względem profilu wspólnego, PCA, klastry), `fit_map_variants` (mapa w trybie referencyjnym
  z punktem kontrolnym), `variant_decision`, `rotation_field`, `wall_traces`.
- `spatial_gp.py` — `SpatialGP` (GP 2D, anizotropowy Matérn/RBF z kątem i dwiema długościami
  korelacji, szum na piksel, LOO) i `SharedLatentGP` (kilka map napędzanych jednym ukrytym polem).
- `domain_tests.py` — testy hipotez o paskach: ułamek pasków bez modelu, regresja dowodu na ułamek
  pasków, bootstrap blokowy, test osi obrotu, różnicowe dopasowanie profili grup (obrót vs wariant),
  geometria klas, wykrywalność polaryzacji, profile w poprzek pasków (dane bez binningu),
  hipotezy o ścianach, c/a z amplitudy obrotu.

Notatniki: `inversion_tests.ipynb` (symulacje: algorytm, szum, tło, emulator),
`mismatch_tests.ipynb` (niedopasowanie modelu, margines wariantu, kalibracja θ),
`real_data.ipynb` (sekcje 0–20: jakość → domeny bez modelu → GP na PC1 → orientacja odniesienia →
kalibracja θ/σ → mapa → GP na obrocie sieci → diagnostyka modelu → dopasowanie w paśmie →
wspólne pole ukryte → testy hipotez o paskach → dane bez binningu → geometria i c/a).
Testy: `tests/` (34 testy na syntetycznym masterze 4mm; przechodzą). Notatniki są generowane
skryptami `build_nb*.py`.

Domyślne ustawienia `real_data.ipynb`: `BAND = [(21, 60)]`, `N_BG_HARM = 3`, `K_CLUSTERS = 4`,
`ROT_SOURCE = "band_ref"`, `TIF_FULL = None`, `BIN = 6`. Mapa w trybie referencyjnym (6 wariantów
V·g_ref na piksel) trwa ok. 9 min i zapisuje postęp.

## 4. Wyniki symulacji (model dopasowywany tym samym modelem, który generuje dane)

- Algorytm (bank + FFT + `refine` + 6 wariantów): bez szumu 0 porażek, precyzja ~10⁻¹⁰°.
  Z szumem mediana błędu 0.005 / 0.009 / 0.018 / 0.029° dla SNR 20 / 10 / 5 / 3 (rośnie jak 1/SNR);
  pierwsze pomyłki wariantu dopiero przy SNR 3 (3%). Czas ≈ 0.35–0.65 s na profil (cone).
- Usuwanie harmonik tła 0..3 nie traci informacji o orientacji ani wariancie.
- Emulator GP: mediana błędu 1 − r ≈ 0.03 (torch; numpy ≈ 0.065), czyli tyle, co różnice między
  wariantami. Wybiera zły wariant 90° w 28% (torch) / 58% (numpy) przypadków. Próg: poniżej
  1 − r_true ≈ 0.02 emulator trafia. **Wniosek: do problemu odwrotnego przy ustalonym masterze
  używać cone** (szybszy ~10× i dokładniejszy).
- Niedopasowanie modelu: **θ jest krytyczny** (Δθ = 0.2° → 70–82% poprawnych wariantów,
  0.5° → 42–50%, 1° → 27–33%; bias orientacji ≈ 0.5·Δθ). Bez wpływu na wybór wariantu: rozmycie
  wiązki do 1°, nieliniowość I^γ, wagi kanałów ±20% (słaby test). Bicie stolika zamienia się w bias
  orientacji ≈ 0.65·ε.
- Margines Δr nie jest wiarygodnym wskaźnikiem pewności; ΔlogL jest lepszy (przy błędzie θ
  błędne decyzje mają ΔlogL < 10).
- Kalibracja θ z danych bez etykiet: błąd θ̂ ≤ 0.012°, wystarczą ~3 piksele (rozrzut 0.004°),
  tolerancja θ dla poprawnego wariantu ±0.07–0.1°.

## 5. Co wiemy o prawdziwych danych (KTN_014)

Struktura i jakość:
- Cała mapa ma **jedną orientację sieci**: każdy piksel koreluje z profilem medianowym na
  r ≈ 0.995. Domeny widać tylko w resztach: **ukośne paski**, kierunek α ≈ 48.6–49.3° (układ
  obrazu, wiersze w dół), korelacja w poprzek ≈ 2 px, wzdłuż ≈ 30–40 px (GP Matérn 3/2 na PC1,
  LOO r ≈ 0.98); ok. 7 pasków, okres 8–13 px, szerokość połówkowa 2–3 px.
- Reszty profilu względem modelu są **addytywne, nie proporcjonalne do jasności** (nie jest to
  zmienny prąd wiązki); z tła liczy się tylko 1. harmonika (r: 0.48 → 0.63).

Orientacja i model:
- Orientacja odniesienia: φ1 ≈ 359°, Φ ≈ 93°, φ2 ≈ 1.6° (powierzchnia ≈ {100}); oś c wariantu V0
  ≈ −y próbki. Osie sześcianu ≈ osie próbki (z dokładnością kilku stopni).
- θ̂ ≈ 40.5° (40.40–40.58° zależnie od profilu i pasma). Różnice θ̂ między klasami ≤ 0.1°
  z zyskiem ΔlogL < 1: **brak dowodu na lokalną zmianę θ**.
- Zgodność model–pomiar: **r = 0.63 w pełnym widmie, 0.79 w paśmie harmonik 21–60**.
  Zgodność w pasmach: 1–5: 0.96, 6–10: 0.60, 11–20: 0.44, **21–40: 0.81**, 41–60: 0.68, 61–90: 0.12.
  W paśmie 21–40 leży największa część różnic między wariantami (36%).
  Granicą jest niedopasowanie modelu, nie szum (średnie profile klas są bez szumu).

Domeny:
- Symulowane różnice wariantów (1 − r, pasmo 21–60): 0.016–0.039. Zmierzone różnice klas:
  0–1: 0.002, 0–3: 0.009, 1–3: 0.009, klasa 2 vs reszta: 0.015–0.016.
- Dopasowanie różnicowe klas do par wariantów nie daje spójnego przypisania (r 0.1–0.44, drugi
  wynik prawie równie dobry, etykiety niespójne).
- Pojedynczy piksel nie rozstrzyga wariantu: w paśmie 99.3% pikseli wybiera V3 (c ∥ normalna),
  z marginesami Δr ≈ 0.02, ΔlogL ≈ 1–3. Paski widać w mapach ΔlogL i Δr, ale nie w etykietach.
- **Paski wyglądają na lekki obrót sieci (ewentualnie z nachyleniem powierzchni), a nie na wykryty inny wariant — ale patrz 5a, bo część argumentów okazała się słabsza.** Różnicowe dopasowanie profili (pasmo 21–60),
  paski (rdzeń) vs matryca: |D| = 0.20 (porównywalne z różnicą wariantów 0.18–0.28);
  obrót wyjaśnia 29% różnicy (δ = (0.045, −0.051, 0.100)°, |δ| = 0.12°), inny wariant 9% (V3, s = 0.28),
  oba razem 34%. Powtarzalne na niezależnych połowach mapy: |δ| = 0.110 (lewa) i 0.127 (prawa),
  δz = 0.081 i 0.109. Kontrola (matryca lewa vs prawa): |D| = 0.05, |δ| = 0.025° (wygięcie sieci
  ~5× mniejsze niż efekt pasków). Oszacowanie z pikseli (mediana pasków − matryca): |d| = 0.087°.
- Profil obrotu w poprzek pasków pokrywa się z PC1: wąskie piki Δz do ≈ 0.05–0.10° (po odjęciu
  wygięcia), Δy w tych samych miejscach ujemne. Gładkie wygięcie sieci jest małe (ok. 0.02° wzdłuż
  mapy; reszta „wygięcia” to stały offset względem g_ref).
- Klasy k-średnich (k=4): klasa 2 zachowuje się jak paski (różnice od klas 0/1/3: |D| ≈ 0.18, |δ| ≈ 0.10,
  jak paski vs matryca); klasy 0 i 1 są prawie identyczne (|D| = 0.07, |δ| = 0.027); klasa 3 jest
  pośrednia (|D| ≈ 0.13 do 0 i 1) **[hipoteza: krawędzie pasków / piksele mieszane]**. Czyli
  „4 klasy” z k-średnich niekoniecznie odpowiadają 4 typom domen. Potwierdzenie: sekcja 15 notatnika
  (geometria klas, η²) — wyniki jeszcze nie przeanalizowane.
- Oś obrotu paski − matryca leży najbliżej **osi [001]-kryształu ≈ normalna próbki** (χ² = 229, oś [100]:
  560, oś [010]: 1019). Formalnie wszystkie osie są odrzucone, bo bootstrap uwzględnia tylko szum
  statystyczny, nie błąd systematyczny (niedopasowanie modelu); składowe x i y (~0.05°) są
  istotne i nie wiadomo, czy to zanieczyszczenie dopasowania, czy mieszanie typów ścian.
- Tabela hipotez o ścianach: dla płaszczyzn {110} ściany są prawie pionowe (nachylenie 87–89°)
  i dają obrót obraz ↔ próbka ψ ≈ 4.7° (albo 94.8° / 87.4° / 177.5° przy odbiciu), a więc niewielki
  i naturalny; płaszczyzny {101} mają nachylenie 42–48° i ψ = 48°–143°.
  **[hipoteza]** Obraz spójny: domeny a1/a2 (c w płaszczyźnie), ściany {110}, oś obrotu ≈ normalna,
  pionowe proste paski. **Napięcie:** takie domeny wymagają c w płaszczyźnie w matrycy, a pasmowe
  dopasowanie wybiera globalnie V3 (c ∥ normalna) na słabym dowodzie (prawdopodobnie artefakt
  niedopasowania modelu). Wybór globalnego wariantu pozostaje nierozstrzygnięty.
- Amplituda obrotu → c/a: 0.087° (piksele) i 0.121° (profile) przy czystości pasków 1.0 dają c/a
  1.0015 i 1.0021; przy 0.8: 1.0019 i 1.0026. Master: 1.00225 (0.129°). Brak oznak wyraźnie mniejszej
  tetragonalności niż w masterze, ale niepewność systematyczna jest duża.
- Polaryzacja (domeny 180°): wskaźnik wykrywalności w pasmach jest taki sam dla par polaryzacji
  (0.42) jak dla par 90° (0.41). **Brak czwartej klasy nie wynika więc z tego, że różnica polaryzacji
  leży poza pasmami zgodności**; może jej nie być w polu widzenia albo model/master słabo ją oddaje.

## 5a. Korekty po ocenie sekcji 16–20 (czytać razem z sekcją 5)

1. **Ułamek wyjaśniony nie odróżnia pasków od kontroli.** Kontrola (matryca lewa vs prawa, |D| = 0.048)
   daje expl R = 0.26 i expl V = 0.05; paski vs matryca: 0.29 i 0.09. To ogólny poziom, na jakim
   3-parametrowy obrót i 5 wariantów opisują dowolną małą różnicę profili. Argumentem za obrotem nie jest
   więc expl, tylko: niezależne oszacowanie z pikseli (0.087°), przestrzenna zgodność z paskami (profil
   w poprzek pasków, sekcja 12d) i powtarzalność lewa/prawa (0.110° / 0.127°).
2. **δ z profili ma podłogę ≈ 0.025° i jest zawyżone.** Kontrola: profile 0.025° vs piksele 0.004°;
   klasy 0–1: 0.027 vs 0.008; paski: 0.121 vs 0.087 (×1.4). Do c/a używać oszacowania z pikseli.
   Po poprawce na rozcieńczenie (kontrast ułamka pasków rdzeń − matryca ≈ 0.78 − 0.01; skala ułamka jest
   tylko orientacyjna, kotwiczona w 99. percentylu): ≈ 0.11° (piksele) … 0.16° (profile) vs 0.129° (master).
3. **2/3 różnicy profili pasków i matrycy jest niewyjaśnione** (obrót + najlepszy wariant: 0.34).
   |D| = 0.20, po podzieleniu przez kontrast ≈ 0.26 dla czystego paska, jest tego samego rzędu co
   pełna różnica wariantów (0.18–0.28). **Test do zrobienia (tylko prawdziwy master, bez nowych
   symulacji):** policzyć |D| przewidywane dla czystego obrotu 0.129° i dla każdego wariantu i porównać
   z 0.26. Jeśli obrót daje wielokrotnie mniej, zmiana profili w paskach nie jest tylko obrotem.
4. **Dekompozycja obrotu niezależna od φ0:** składowa wzdłuż normalnej δz ≈ 0.100° (rozcieńczone) i w
   płaszczyźnie √(δx² + δy²) ≈ 0.068°, czyli oś ≈ 34° od normalnej. Czysty bliźniak a1/a2 (c w płaszczyźnie,
   ściany {110}) dałby oś wzdłuż normalnej; czysty bliźniak c/a (ściany {101}) oś w płaszczyźnie. Część
   „w płaszczyźnie" może pochodzić z nachylenia powierzchni (relief ścian), które jest równoważne
   obrotowi próbki; jedyną czysto sieciową składową jest δz.
5. **Globalny wariant matrycy jest niestabilny:** pełne widmo → V0 (87%), pasmo 21–60 → V3 (99%),
   ΔlogL ≈ 1–3 na piksel. Tabela ścian z sekcji 20 jest liczona dla V_VB·g_ref (V3, c ∥ normalna).
   Dla matrycy z c ∥ normalna możliwe są tylko ściany {101} z osią obrotu w płaszczyźnie, a to **nie pasuje**
   do dominującego δz. Spójniejszy obraz: matryca z c w płaszczyźnie, domeny a1/a2, ściany {110} niemal
   pionowe (87–89°), ślad 43.9° vs paski 48.6°, ψ ≈ 5°. **[hipoteza]** Wymaga φ0/sense/invert z metody
   słownikowej i/lub danych bez binningu.
6. **ψ (obrót obraz ↔ próbka) nie jest wyznaczalny z samych danych:** φ0 = 0 jest umowne, zmierzone φ1
   zawiera φ0. Kalibracja φ0 na pikselach z orientacjami słownikowymi daje φ0, sense, invert i ψ naraz.
7. **Polaryzacja nie jest wykluczona:** klasy 0 vs 1 (|D| = 0.07) różnią się ≈ 4× mniej niż pełna różnica
   polaryzacji (≈ 0.28), więc nie są czystymi domenami 180°, ale różnica udziału polaryzacji ≈ 25% jest
   możliwa. Wskaźnik wykrywalności (0.42 vs 0.41) też jej nie wyklucza.

## 5b. Wyniki sekcji 14, 15, 17, 20 (ocenione)

- **Sekcja 14 (dowód wariantu vs ułamek pasków, odniesienie V3, 9 bloków, 3675 pikseli):** V3 wygrywa
  w całym zakresie ułamka, także dla czystego paska (ΔlogL względem V3: V0 −1.6, V1 −3.1, V2 −5.3,
  V4 −4.4, V5 −5.1 na piksel). Dla V0, V1, V2, V5 przewaga V3 maleje z ułamkiem o ok. 1 logL (nachylenie
  0.84–1.21), dla V4 rośnie (−0.24). Żaden wariant nie przekracza zera, więc paski nie są wykrytym
  innym wariantem. Maleje tylko luka (rozcieńczenie / mniejszy kontrast), a nie widać zmiany wariantu.
  Matryca: V0 jest tylko 2.5 gorszy od V3, co zgadza się z niestabilnością V0/V3 (5a.5).
- **Sekcja 15 (geometria klas):** klasy 0, 1 = matryca (szerokie, ułamek pasków < 0.06); 2, 3 = paski
  (wąskie, 0.63 i 0.27). Podział 0 vs 1 idzie wzdłuż poziomic wygięcia Δz (η² = 0.31), czyli to gładki
  gradient, a nie osobne domeny. η² dla ułamka pasków 0.77 jest częściowo kołowe. `max_promień_px` = 2.000 dla
  klas 2, 3 to ucięcie algorytmu, więc szerokość pasków (2.2 px) jest dolnym oszacowaniem.
  Realne są dwie populacje (matryca, paski); cztery klasy k-średnich to 2 + gradient + rozcieńczenie.
  Do sprawdzenia: po której stronie zielonego paska leży czerwony (nachylona ściana vs dwa typy).
- **Sekcja 17 (test osi):** d (piksele) = (0.023, −0.027, 0.080)°, |d| = 0.087°, sd bootstrapu 0.002–0.004°.
  Kąt do osi: [001] 22.8° (χ² 229), [100] 72.8° (560), [010] 75.5° (1019). Wszystkie odrzucone, więc jest błąd
  systematyczny, ale oś [001] ≈ normalna próbki jest zdecydowanie najbliższa. Profile dają ten sam
  kierunek w płaszczyźnie (0.045, −0.051), ale ok. 2× większy, a w z 0.100 (kąty 66°, 68°, 33°).
  Kierunek składowej w płaszczyźnie (≈ −49°) zgadza się co do wartości bezwzględnej z kątem pasków (48.6°):
  do sprawdzenia, czy leży wzdłuż śladu ściany (oś obrotu w płaszczyźnie ściany jak w granicy nachylenia).
- **Sekcja 20 (ściany i c/a):** hipotezy {110} z osią [001] mają najmniejsze χ² (229) i prawie pionową ścianę
  (87–89°); ślad 43.9° vs zmierzone 48.6° → ψ ≈ 4.7° (odbite 87.4°). Pozostałe ściany wymagają dużego ψ
  (≈ 40–50° lub 130–143°). Ale ψ nie jest znane (5a.6), więc ślad nie rozstrzyga bez kalibracji geometrii.
  c/a z c = 1/czystość: piksele 1.0015 (czystość 1.0), 1.0019 (0.8), 1.0025 (0.6); profile 1.0021, 1.0026, 1.0035.
  Realna czystość ≈ 0.8 (rdzeń 0.78 − matryca 0.01), więc c/a ≈ 1.0019–1.0026 vs master 1.00225: zgodne,
  ale zakłada, że cały obrót to bliźniak c/a (nie nachylenie powierzchni ani gradient).
- **Sekcja 19** (bez binningu) nie była uruchomiona: brak stosu 372×372.

## 5c. Liczba stanów: klasteryzacja (2b, 2c) i sekcja 2d

- **Klasteryzacja automatyczna nie ma naturalnego k.** DBSCAN: jeden spójny klaster (3522 px), HDBSCAN: k = 6–8
  i 44–63% szumu, GMM: BIC maleje monotonicznie do k = 9–10 (ΔBIC setki na składnik) bez kolana, także po
  usunięciu gładkiego tła (σ = 6, 10, 15). Wniosek: w przestrzeni PC1–4 jest kontinuum (gradient wygięcia +
  rozcieńczenie pasków), a nie skupiska; BIC zawyża k dla rozkładów niegaussowskich. Po detrendzie klasy
  różnią się profilami o 1 − r = 0.003–0.005 (warianty: 0.03–0.04), a podział k = 2 (≈ 50/50) jest częściowo
  artefaktem filtra górnoprzepustowego.
- **2d, test 1 (dwumodalność ułamka pasków, n = 423 po próbkowaniu co 3 px):** bez `diptest` (p = NaN);
  ΔBIC(1→2) = −72 (surowe) / −86 (GP), dolina 0.67 / 0.59. Histogram: dwa garby w „matrycy” (≈ −0.3 i ≈ 0 —
  prawdopodobnie gradient wygięcia) i ciągły ogon do 1 bez piku. Test nie rozstrzyga, bo przy binningu 6×6 nie
  ma pikseli czysto paskowych (paski ~2 px), więc piku przy ≈ 1 nie można oczekiwać.
- **2d, test 2 (mieszanina):** matryca 2191, rdzeń 473, pośrednie 1167 px; 1 − r(profil matrycy, rdzenia) = 0.008;
  energia reszt: pośrednie 0.0109 vs czyste 0.0083–0.0093 (×1.29); coh = 0.0224 vs 0.0004 (szum ≈ 0.0009);
  wartości własne PC1/PC2 reszt: ×1.7/×1.8 względem czystych; w = b/(a+b) vs ułamek: r = 0.955, ale przesunięte
  o ≈ +0.2 (nieskalibrowana skala ułamka lub norma profili). Dwa stany + mieszanie wyjaśniają większość, ale
  zostaje mały, istotny i spójny przestrzennie składnik (≈ 1/6 energii różnicy stanów), o znaku zmieniającym się
  po obu stronach pasów, z kształtem w ω podobnym do pochodnej profilu (piki przy ω ≈ 0–30°, 100–160°, 290–330°)
  **[hipoteza: małe przesunięcie pasm = składowa obrotu o zmiennym znaku w poprzek pasa; do sprawdzenia przez
  rzut na jakobian modelu]**.
- **2d, test 3 (asymetria stron, α = 48.6°, 8 pasów):** A = +0.227 ± 0.054 (t = 4.2, p = 0.004, 8/8 dodatnie,
  p znaku = 0.008); α + 3°: A = +0.227 (p = 0.002); α − 3°: A = −0.123 (p = 0.27) — **znak zależy od α**.
  Profil: peak przy s ≈ −0.25 px, ramię po stronie rosnącego u (w obrazie ku lewemu dołowi) przy s = 1.5–3 px.
  Dwa możliwe artefakty (jeszcze niewyeliminowane): (1) błąd α przy niesymetrycznym zakresie t pasów,
  (2) gładki trend tła (wygięcie): różnica poziomu po obu stronach (−0.32 vs −0.18) ≈ D(3.25) = 0.14.
  Do zrobienia: α z maksymalizacji ostrości pików, okno wzdłuż t symetryczne, odjęcie trendu tła.

## 5d. Informacja od użytkownika z AFM (dane AFM jeszcze nieprzygotowane)

- Z AFM powinny wychodzić **4 domeny**. W k-średnich k = 4 (niebieska = klasa 0, pomarańczowa = 1, zielona = 2,
  czerwona = 3) użytkownik uważa, że **czerwona jest artefaktem** (piksele przesuwały się między zdjęciami
  przy kolejnych ω), a niebieska, zielona i pomarańczowa są prawdziwe. Czwarta, prawdziwa klasa powinna dzielić
  pomarańczową na dwie, zmieniające się po przecięciu zielonego paska (układ typu ABAB), podobnie jak wzór
  naprzemiennych pasów w PC1 reszt (2d, test 2).
- **Hipoteza pasująca do danych (do weryfikacji):** przesunięcie pola widzenia δ(ω) między klatkami + gradient
  ułamka na krawędzi pasa daje resztę ∝ (P_pasek − P_matryca)·δ(ω)·∇w, o znaku przeciwnym po obu stronach pasa
  i jednostronną asymetrię profilu. Tłumaczyłoby to czerwoną klasę (jednostronna, pośrednia), składnik reszt
  z 2d i część asymetrii A.
- **Rozróżnienie od prawdziwych domen ABAB:** przy przesunięciu kolejność znaku flanek jest taka sama dla każdego
  pasa; przy ABAB zmienia się co pas. Drugi test: profile plateau między pasami (macierz 1 − r) — parzystość
  vs gładki trend wygięcia.
- **Sekcja 2e jest napisana** (`domain_tests.py`: u_grid, frame_shifts, fit_shift_model, shift_frames, flank_signs,
  plateau_profiles, abab_regression; testy w `tests/test_shift_abab.py`; komórki w notebooku po 2d), przetestowana
  tylko na danych syntetycznych, jeszcze nie uruchomiona na prawdziwych. Uwaga: `frame_shifts` domyślnie normuje
  względnym kontrastem (Y − ⟨Y⟩_ω)/⟨Y⟩_ω, a nie z-score — z-score dawał ~30% obciążenia δ w dół; amplituda δ ≈ ±15%.
  Opis planu: (a) δ_u(ω) z dopasowania klatek Z_k ≈ b_k + c_k S + d_k ∂_u S (δ_k = −d_k/c_k),
  korekta resamplingiem i powtórzenie k-średnich oraz 2d; (b) test ABAB (znak flanek po pasach + macierz
  profili plateau). AFM (po rejestracji) da niezależną maskę pasków, nachylenie powierzchni (0.1° ≈ 1.7 nm na
  1 µm) do porównania ze składową obrotu w płaszczyźnie oraz kalibrację ψ i śladu ścian.

## 5e. Wyniki sekcji 2e (przesunięcia klatek, ABAB) i struktura mastera

- **δ(ω):** istnieje, gładkie, ~±0.5 px (binning 6×6): ≈ −0.2 px dla ω ≈ 50–150°, +0.5…+1 px dla ω ≈ 250–330°;
  h = 1: 0.285 px, trend +0.26 px/360°, R² = 0.44, rms reszt 0.26 px ≫ se 0.046 px (model nie opisuje wszystkiego).
  Kontrast pasków c_k jest istotny tylko w oknach ω ≈ 100–200° i 280–330°, tylko tam δ jest mierzalne.
- **Korekta modelowa (rms δ 0.364 → 0.206 px)** poprawia część wskaźników: λ1 reszt pośrednich −36%, asymetria A
  0.227 → 0.107 (t 4.2 → 4.3), udział PC1 0.19 → 0.25; NIE usuwa: coh 0.022 → 0.028, klasa czerwona (563 → 537,
  ułamek 0.27 → 0.22), wzór par czerwono-niebieskich w PC1 reszt, stały znak flanek. Wniosek: przesunięcie
  jest realne i tłumaczy ok. 1/3–1/2 składnika reszt i ok. połowę asymetrii, ale nie jest główną przyczyną
  klasy czerwonej. Do zrobienia: CORR = "raw" na dobrych klatkach i druga iteracja.
- **ABAB niepotwierdzone:** znak (prawa − lewa flanka) dodatni dla wszystkich 9 pasów (8/8 par, p = 0.008), przed i
  po korekcie; plateau: D1 = 1.3e-4, D2 = 3.1e-4, D rośnie do 1.6e-3 przy odległości 7, β(dist²) = 2.9e-5 ± 2.7e-6,
  γ(parzystość) = −4.9e-5 ± 2.8e-5 (t = −1.7), D1 < D2, szum ~1e-5. Profil matrycy zmienia się gładko jak dist²
  (liniowy gradient obrotu = wygięcie); granica parzystości γ ≲ 1e-5 (2σ), > 1000× mniej niż różnice wariantów
  w symulacji (0.02–0.04). Test obejmuje wszystkie przedziały naraz; ABAB tylko w części pomarańczowej miałby małą moc.
- **Kandydat na stały znak flanek:** gradient wygięcia matrycy (sąsiednie plateau różnią się, a test mieszaniny
  używa jednego globalnego profilu matrycy). Do sprawdzenia: mieszanina z lokalnymi profilami lewego/prawego plateau.
- **Struktura mastera (z repo, `KTN_lattice_modified_KNO.xtal`):** P4mm (99), a = 0.3993, c = 0.4002 nm; atomy
  (x, y, z; DW [nm²]): K (½, ½, ½; 0.1024), Nb (0, 0, 0.0149; 0.0043), O (0, ½, 0.974; 0.005), O (0, 0, 0.4716;
  0.005), obsadzenia 1. To współrzędne tetragonalnego KNbO₃ z siatką KTN: **brak Ta** (tylko Nb, Z = 41),
  skład x nieznany; przesunięcia polarne Nb–O ≈ 0.17 Å (typowe dla KNbO₃, prawdopodobnie za duże dla KTN w temp.
  pokojowej przy c/a = 1.002) **[hipoteza: wyjaśniałoby, czemu różnice między klasami w danych (0.003–0.005,
  plateau < 1e-5) są ≫ mniejsze niż między wariantami w masterze (0.02–0.04)]**. **DW dla K = 0.1024 nm² (~10 Å²)
  jest ~20× większe niż dla Nb i O (typowo ~0.005–0.01 nm²) — do sprawdzenia, możliwa literówka.** Źródło wartości
  liczbowych: literatura (użytkownik nie zna dokładnego źródła). MC (`openCL_014_01.nml`): tryb bse1 (ECP),
  10 keV, 2·10⁹ elektronów, głębokość 100 nm; master: npx = 512, dmin = 0.05 nm; ECP: tilt 40°, WD 10.6 mm,
  detektor pierścieniowy Rin 4.3, Rout 10 mm. Pomiary w temperaturze pokojowej. Wpływ c/a, skali przesunięć
  jonów, Ta i DW — do zbadania w osobnym czacie (tu emulator GP po parametrach ma sens).
- Notatka z repo (`info_dominik.md`): w danych słownikowych Dominika kierunek (oś) działa dobrze, problemem jest
  zwrot (polaryzacja); drugi słownik (~8 h) rozróżnia orientacje pseudosymetryczne.

## 5f. ABAB w strefie pomarańczowej (uruchomione na prawdziwym stosie z repo, α = 48.6°, 9 pasów)

Użytkownik: AFM wskazuje, że ABAB występuje tylko w strefie pomarańczowej. Poprzedni test brał całe przedziały
(każdy jest po części pomarańczowy, po części niebieski; pomarańczowa to elipsa wokół środka, ułamek klasy 1
w przedziałach 0.3–0.6). Powtórzenie z plateau tylko z pikseli klasy 1 (min 20 px; przedziały 0–7, liczności
70, 51, 71, 129, 63, 253, 149, 22): D1 = 1.26e-4, D2 = 2.29e-4, β = 3.1e-5 ± 3.4e-6, γ = −7e-6 ± 4.8e-5 (t = −0.1).
Przedziały 4–7 (u ≈ 0–32, rdzeń strefy) tworzą blok niemal identycznych profili (D − szum 0.5–2.8e-4, D1 ≈ 0.9e-4
< D2 ≈ 1.9e-4), więc **bez naprzemienności na poziomie ≳ 1e-4 także w samej strefie pomarańczowej**; nie jest to
jednak mocny dowód (4 przedziały, część z mało pikseli). Okno wzdłuż pasków |t| ≤ 12 daje to samo (γ t = −0.6).
Skok profilu jest przy przedziale 3 → 4 (granica niebieski/pomarańczowy), nie parzysty.

## 5g. Dane bez binningu i obserwacja wizualna użytkownika

- **Użytkownik dysponuje teraz stosem bez binningu (372×372)**; w tym czacie nie był jeszcze analizowany
  (nazwa pliku i lokalizacja do uzupełnienia przez użytkownika; w repo jest tylko wersja 62×62). Sekcja 19
  notatnika (`TIF_FULL`) jest na to przygotowana: czyste piksele paskowe (paski ~12 px), dip test ułamka
  pasków, c/a bez korekty na rozcieńczenie, profil w poprzek pasków i test ABAB w natywnej rozdzielczości.
- **Obserwacja użytkownika (pojedyncza klatka, widok bezpośredni):** pasy ukośne i strefa pomarańczowa są dobrze
  widoczne gołym okiem; **naprzemienności ABAB użytkownik w obrazach w ogóle nie widzi**, co zgadza się z
  wynikiem negatywnym testów 2e/5f. W centrum klatki jest jasna plamka (to artefakt maskowany przez
  `clean_pixels` na mapach przy ok. (30, 30)). ABAB z AFM może więc być niewidoczny w kontraście
  intensywności eCHORD (natura A/B pozostaje otwarta: AFM sam może pokazywać relief/inny sygnał niż eCHORD).

## 6. Pułapki i niepewne punkty

- Etykiety wariantów są względne: V_j jest liczone względem orientacji startowej danego profilu.
  Porównywać można tylko wyniki względem wspólnego g_ref lub przez dezorientację orientacji
  końcowych modulo C4/D4. Dezorientacja modulo O **nie** odróżnia wariantów z definicji.
- Wniosek z jednego z wcześniejszych czatów, że „klasa 2 wybiera inny wariant”, był nieuzasadniony
  (porównanie etykiet z różnych startów); nie używać.
- Bootstrap blokowy ma ok. 9 bloków (okres pasków) → niepewności zgrubne i optymistyczne.
- Różnicowe dopasowanie traktuje obrót liniowo, a ułamek domeny (rozcieńczenie binningiem) jest
  nieznany (skala pola pasków nie jest identyfikowalna; `frac` jest zakotwiczony orientacyjnie
  w 99. percentylu).
- Obrót sieci liczony względem V0 w paśmie jest niestabilny (skoki do −0.2°); używać odniesienia VB
  (sekcja 12d).
- Wszystkie testy kodu były wykonane na syntetycznym masterze 4mm; sekcje 14–20 notatnika
  nie były jeszcze uruchamiane na prawdziwym masterze.

## 7. Potencjał procesów gaussowskich

Już zrobione i działa:
1. GP 2D na PC1 (anizotropowy Matérn): kierunek, szerokości, struktura domen bez żadnego modelu krystalicznego.
2. `SharedLatentGP`: PC1 i składowe obrotu sieci jako liniowe funkcje jednego ukrytego pola f(x);
   daje ładunki (obrót na jednostkę f → oś), mapę f z niepewnością. Skala f jest nieidentyfikowalna.
3. GP z szumem heteroskedastycznym z Laplace'a (niepewność orientacji na piksel).

Najbardziej obiecujące, w kolejności:
1. **GP na niedopasowaniu modelu (dyskrepancja Kennedy'ego–O'Hagana).** Niedopasowanie (r = 0.63–0.79)
   jest wspólne dla wszystkich pikseli, bo orientacja jest wspólna. Można je modelować jako gładką
   funkcję b(ω) (jądro po ω), uczoną z setek niemal bezszumowych pikseli matrycy, i odjąć. To
   uogólnia różnicowe dopasowanie z sekcji 16 i jest najlepszą szansą na rozróżnienie wariantów.
2. **Emulator GP po parametrach fizycznych** (c/a, przesunięcia jonów = polaryzacja, energia, DW):
   tu emulator ma sens, bo każdy nowy master z EMsoftu kosztuje minuty–godziny. Pozwala wyznaczyć
   tetragonalność z danych (amplituda obrotu, różnice wariantów) z propagacją niepewności emulacji.
   Przy ustalonym masterze cone jest lepszy (sekcja 4).
3. **Model przestrzenny świadomy ścian:** stacjonarny GP rozmywa skoki na ścianach. Opcje:
   GP po współrzędnej w poprzek pasków (1D, jądro schodkowe do szerokości ściany), niestacjonarne
   jądro, GP-klasyfikacja albo pole losowe na 6 wariantach z ΔlogL jako członem danych.
4. **Gładkie pole obrotu z niepewnością** do gradientów (miary odkształceń); autodiff w wersji GPyTorch.
5. Wspólne pole ukryte dla większej liczby wyjść (np. dodać różnicę r i ΔlogL, ułamek klas).

Ograniczenia: dokładny GP dla 3844 punktów jest tani; na danych bez binningu (138 tys. pikseli) trzeba
przybliżeń (podzbiór, indukujące punkty) albo pracy na profilach uśrednionych wzdłuż pasków.

## 8. Proponowane następne kroki

0. **Najważniejsze:** skalibrować φ0, sense, invert (i ψ) na pikselach z orientacjami z metody słownikowej
   (użytkownik je ma dla całej mapy). Bez tego globalny wariant matrycy i typ ścian pozostają
   nierozstrzygnięte (5a.5–5a.6). Potem wykonać test z 5a.3 (|D| dla czystego obrotu vs zmierzone).
1. Przeanalizować wyniki sekcji 14, 15 i 17 (opisane w 5b; zostaje sekcja 19): czy któryś wariant ma istotne
   `z_b1`; czy klasa 2 jest wąska, a klasy 0, 1, 3 to tło/krawędzie.
2. **Dane bez binningu** (372×372): ostry profil w poprzek ścian, plateau obrotu bez rozcieńczenia
   (sekcja 19, parametr `TIF_FULL`).
3. Rozstrzygnąć wariant globalny (V3 vs warianty z c w płaszczyźnie) i oś obrotu: dyskrepancja GP
   z pkt 7.1 oraz dopasowanie z kilkoma ścianami naraz.
4. Zebrać metadane: skład x, temperatura względem T_c, źródło struktury w masterze, geometria skanu
   (rejestracja obrazu, odbicie, związek osi obrazu z osią pochylenia → ψ).
5. Jeśli metadane lub model wskażą inną strukturę: nowe mastery (c/a, przesunięcia jonów) i emulator GP po parametrach.
6. Przeliczenie wyników do konwencji metody słownikowej (sense, invert) i porównanie z jej wynikiem
   dla całej mapy; raportować dezorientację modulo C4, D4 i O.


## 9. Stan na koniec czatu i priorytety (czytać najpierw)

**Co jest solidne:** (1) jedna orientacja sieci na mapie, mierzona z dokładnością ~0.1°; (2) dwie populacje:
matryca i paski (obrót paski − matryca ≈ 0.09–0.12°, zgodny z c/a mastera 1.00225, z dużą niepewnością
systematyczną), rozmyte rozmiarem piksela; (3) matryca zmienia się gładko wzdłuż u jak dist² (wygięcie);
(4) przesunięcia klatek δ(ω) ~ ±0.5 px istnieją, ale nie tłumaczą klasy czerwonej; (5) w danych nie ma
rozróżnienia wariantów pseudosymetrii (różnice klas 0.003–0.005, parzystość ≲ 1e-4 vs 0.02–0.04 w masterze);
(6) klasteryzacja nie ma naturalnego k (kontinuum).
**Co jest otwarte:** globalny wariant (V3, c ∥ normalna vs c w płaszczyźnie), typ ścian ({110} a1/a2 vs {101}),
φ0/sense/invert/ψ, natura klasy czerwonej i stałego znaku flanek (wygięcie vs reszta przesunięcia vs asymetria
prawdziwa), natura domen A/B z AFM (niewidoczne w eCHORD), polaryzacja, wpływ struktury mastera (brak Ta,
przesunięcia KNbO₃, DW K, c/a, skład x) na r = 0.63.

**Priorytety (kolejność):**
1. *Z obecnymi danymi:* (a) test mieszaniny z lokalnymi profilami plateau (lewy/prawy sąsiad pasa) zamiast globalnej
   matrycy + `CORR = "raw"` i druga iteracja δ(ω); (b) rzut pierwszej składowej reszt na jakobian modelu
   (oś obrotu, porównanie z d = (0.023, −0.027, 0.080)°); (c) przeliczenie D ∝ dist² na gradient obrotu
   [°/px] (jakobian), porównanie z polem obrotu z SpatialGP; (d) GP na rozbieżność modelu b(ω).
2. *Wymaga nowych danych:* (a) orientacje ze słownika dla całej mapy → kalibracja φ0, sense, invert, ψ, wariant
   globalny; (b) stos 372×372 bez binningu — **już otrzymany przez użytkownika, nieanalizowany** (`TIF_FULL`, sekcja 19;
   pierwsza rzecz do zrobienia w nowym czacie): paski ~12 px, czyste piksele, dip test, c/a bez korekty na
   rozcieńczenie, ABAB w natywnej rozdzielczości; (c) AFM (po rejestracji, potrzebne: pole widzenia, rozmiar piksela,
   topografia czy sygnał piezo): maska pasków jako etykiety, nachylenie powierzchni (0.1° ≈ 1.7 nm/µm) vs
   składowa obrotu w płaszczyźnie, natura A/B; (d) rozmiar piksela eCHORD (µm) i metadane (skład x).
3. *Master / fizyka (osobny czat):* siatka masterów po (c/a, skala przesunięć jonów λ, udział Ta x, DW K)
   i emulator GP po tych parametrach; sprawdzić, czy λ ≪ 1 godzi różnice wariantów z danymi; sprawdzić DW K.

**Pliki do przekazania** (wersje końcowe w /mnt/user-data/outputs): `real_data.py` (cluster_auto, detrend_scores,
gmm_scan), `domain_tests.py` (sekcje 2d i 2e + `region` w plateau_profiles), `build_nb_real.py` + `real_data.ipynb`
(sekcje 2b–2e), `tests/test_structure_2d.py`, `tests/test_shift_abab.py`, `tests/test_real_data.py`, ten dokument;
pozostałe moduły bez zmian: `orient_fit.py`, `calibration.py`, `mismatch.py`, `spatial_gp.py`,
`run_inversion_test.py`, notatniki `inversion_tests.ipynb`, `mismatch_tests.ipynb`; `echord_cone.py` i emulator
GP są w repo użytkownika. Zależności: `scikit-learn` (HDBSCAN/DBSCAN/GMM), `diptest` (opcjonalny), `h5py`, `tifffile`.
Dane z repo (`2026_10_19_example_PZ/`): stos 62×62, `masterKTN_014_01.h5`, `KTN_lattice_modified_KNO.xtal`, nml.
Odtworzenie wyniku na prawdziwych danych bez notatnika: skrypt w sekcji 5f używa tylko `real_data` i `domain_tests`.
