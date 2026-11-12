"""
train_gp.py — uczenie emulatora GP profili eCHORD na orientacjach z pliku kątów EMsoftu.

Przykład:
    python train_gp.py --master ../2026_10_19_example_PZ/masterKTN_014_01.h5 --angles ../2026_10_19_example_PZ/test_ang_quaterion.csv --theta 40 --dpsi 0.25 --out gp_ktn_th40.npz

    python train_gp.py --master ../2026_10_19_example_PZ/masterKTN_014_01.h5 --angles design_C4_sobol16k.txt --theta 40 --dpsi 0.25 --out gp_ktn_th40.npz

Kolejne kroki:
  1. wczytanie master patternu i ustalenie grupy właściwych obrotów (z nr grupy przestrzennej),
     test symetrii na samym masterze,
  2. profile kanoniczne F(psi; Phi, phi2) dla orientacji z pliku (phi1 odrzucone),
  3. test wzoru I(omega) = F(phi0 + sense*omega - phi1) na losowych orientacjach,
  4. pętla uczenia: ML-II (L-BFGS-B) na podzbiorze --n-fit, warunkowanie na całości,
  5. walidacja na losowych orientacjach spoza zbioru uczącego, zapis modelu.

Uwaga: plik celowo NIE nazywa się test_*.py (pytest / VS Code by go importowały).
"""




import argparse
import json
import time

import numpy as np

import echord_cone as ec
import gp_echord as G


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", required=True, help="master pattern .h5 z EMsoftu")
    p.add_argument("--angles", required=True,
                   help="plik kątów EMsoftu ('qu' albo 'eu'), np. wynik EMsampleRFZ")
    p.add_argument("--energy", default="max", help="'max', 'weighted' albo indeks kanału")
    p.add_argument("--theta", type=float, default=40.0, help="kąt pochylenia próbki [deg]")
    p.add_argument("--dpsi", type=float, default=0.25, help="krok siatki psi [deg]")
    p.add_argument("--invert", action="store_true", help="kierunek -d zamiast d")
    p.add_argument("--sigma-conv", type=float, default=0.0, help="zbieżność wiązki [deg]")
    p.add_argument("--sym", default="auto",
                   help="grupa właściwych obrotów: auto (z pliku .h5) albo np. C4, C2z, C3, D4, O")
    p.add_argument("--nu", type=float, default=1.5, help="Matérn: 0.5, 1.5, 2.5 lub inf")
    p.add_argument("--ell0", type=float, default=1.4, help="startowa skala korelacji [deg]")
    p.add_argument("--n-fit", type=int, default=2000, help="liczba profili do doboru hiperparam.")
    p.add_argument("--psi-stride", type=int, default=1,
                   help="przerzedzenie psi przy doborze hiperparametrów (przyspieszenie)")
    p.add_argument("--maxiter", type=int, default=100)
    p.add_argument("--n-val", type=int, default=500, help="liczba orientacji walidacyjnych")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="gp_echord.npz")
    return p.parse_args()


def main():
    a = parse_args()
    t_start = time.perf_counter()
    energy = int(a.energy) if a.energy.lstrip("-").isdigit() else a.energy
    nu = np.inf if a.nu == np.inf else float(a.nu)

    # 1. master + symetria
    mp = ec.MasterPattern(a.master, energy=energy)
    print(mp)
    sym = G.proper_group_name(mp.space_group) if a.sym == "auto" else a.sym
    ops = G.proper_rotations(sym)
    corr = G.symmetry_check(mp, ops)
    print(f"[symetria] grupa właściwych obrotów: {sym} ({len(ops)} op.), "
          f"korelacja M(Sd) z M(d): {np.round(corr, 4)}")
    if corr.min() < 0.99:
        print("[symetria] UWAGA: master nie ma tej symetrii — podaj --sym ręcznie "
              "(np. C1, jeśli nie wiesz).")

    # 2. dane uczące z pliku EMsoftu
    data = G.build_training_data(mp, anglefile=a.angles, coneAngle=a.theta, dpsi=a.dpsi,
                                 invert=a.invert, sigma_conv_deg=a.sigma_conv)
    Phi = np.rad2deg(data.angles[:, 0])
    print(f"[dane] Phi: min {Phi.min():.1f}°, max {Phi.max():.1f}°; "
          f"punktów z Phi < 15°: {np.sum(Phi < 15)}, z Phi > 165°: {np.sum(Phi > 165)}")

    # 3. test przesunięcia (niezależny od GP)
    r, err = G.check_shift_bookkeeping(mp, a.theta, a.dpsi, a.invert, a.sigma_conv)
    print(f"[test wzoru (1)] min r = {r:.5f}, max błąd wzgl. = {err:.4f} "
          "(różnica tylko przez interpolację po psi)")

    # 4. pętla uczenia
    gp = G.EchordGP(ops, nu=nu, ell_deg=a.ell0)
    gp.fit(data, n_fit=a.n_fit, psi_stride=a.psi_stride, maxiter=a.maxiter, seed=a.seed)
    gp.meta["sym"] = sym

    # 5. walidacja i zapis
    res = G.evaluate(gp, mp, n_val=a.n_val, seed=a.seed + 12345)
    gp.meta["validation"] = dict(
        r_gp_median=float(np.median(res["r_gp"])), r_nn_median=float(np.median(res["r_nn"])),
        dist_nn_median_deg=float(np.median(res["dist_nn_deg"])), z_sd=res["z_sd"],
        cover2=res["cover2"])
    gp.save(a.out)
    print(f"[zapis] {a.out}; całość {time.perf_counter() - t_start:.1f} s")
    print(json.dumps(gp.meta["validation"], indent=2))


if __name__ == "__main__":
    main()
