"""
run_inversion_test.py — test regresji profil -> orientacja na danych SYMULOWANYCH.

Prawda: losowe orientacje g, profile z ConeModel (dokładne cone_profiles), opcjonalnie
zaburzone (gain/offset, tło niskich harmonik, szum). Dopasowanie: ConeModel ("wyrocznia")
i/lub GPModel (emulator). Porównanie rozdziela trzy źródła błędu:

  cone -> cone, bez szumu   : błąd samego algorytmu (powinien być ~0)
  cone -> gp,   bez szumu   : błąd emulatora
  ... z szumem / tłem       : odporność / identyfikowalność

Dla każdego profilu zapisujemy też r_true = korelację modelu w PRAWDZIWEJ orientacji.
Jeśli dezorientacja jest duża, to:
  r_est <  r_true  -> porażka SZUKANIA (istnieje lepsze minimum, nie znaleźliśmy go),
  r_est >= r_true  -> problem MODELU / IDENTYFIKOWALNOŚCI (model woli złą orientację).

Przykłady:
  python run_inversion_test.py --master ../2026_10_19_example_PZ/masterKTN_014_01.h5 --n-test 200
  python run_inversion_test.py --master masterKTN_014_01.h5 --gp gp_ktn_th40.npz \
      --snr inf 20 5 --phi0 13 --sense -1 --out inv_th40.npz
  python run_inversion_test.py --master ... --gp ... --near-angles DI_wyniki.txt --near-spread 2

Plik celowo nie nazywa się test_*.py.
# 1. algorytm (powinno wyjść ~0°)
python run_inversion_test.py --master ../2026_10_19_example_PZ/masterKTN_014_01.h5 --n-test 200 --bank-cache bank
python run_inversion_test.py --master ../2026_10_19_example_PZ/masterKTN_014_01.h5 --gp gp_ktn_th40.npz --snr inf 20 5 --phi0 13 --sense -1 --gain-sd 0.3 --bank-cache bank


"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

import echord_cone as ec
import orient_fit as of


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", required=True)
    p.add_argument("--energy", default="max")
    p.add_argument("--gp", default=None, help="plik .npz emulatora (gp_echord.EchordGP)")
    p.add_argument("--no-cone", action="store_true", help="pomiń dopasowanie ConeModel")
    p.add_argument("--theta", type=float, default=40.0, help="ignorowane, gdy podano --gp")
    p.add_argument("--invert", action="store_true", help="ignorowane, gdy podano --gp")
    p.add_argument("--sigma-conv", type=float, default=0.0, help="ignorowane, gdy podano --gp")
    p.add_argument("--dpsi", type=float, default=0.25, help="ignorowane, gdy podano --gp")
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--near-angles", default=None,
                   help="plik kątów EMsoftu: orientacje testowe losowane WOKÓŁ nich")
    p.add_argument("--near-spread", type=float, default=2.0, help="[deg] rozrzut wokół nich")
    p.add_argument("--phi0", type=float, default=0.0)
    p.add_argument("--sense", type=int, default=+1, choices=[-1, 1])
    p.add_argument("--snr", type=float, nargs="+", default=[np.inf],
                   help="std(profilu)/std(szumu); 'inf' = bez szumu; można kilka")
    p.add_argument("--gain-sd", type=float, default=0.0, help="względny rozrzut gainu")
    p.add_argument("--bg-amp", type=float, default=0.0,
                   help="amplituda tła (harmoniki 1..3) względem std profilu")
    p.add_argument("--n-bg-harm", type=int, default=0,
                   help="usuń harmoniki 0..K w omega z pomiaru i modelu (0 = tylko średnia)")
    p.add_argument("--spacing", type=float, default=0.7, help="[deg] oczko banku (Phi, phi2)")
    p.add_argument("--bank-from", choices=["cone", "model"], default="cone",
                   help="bank startów z cone_profiles (szybko, dokładnie) albo z modelu")
    p.add_argument("--bank-cache", default=None, help="prefiks plików .npz z bankiem")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="inversion_test.npz")
    return p.parse_args()


def make_truth(a, n):
    if a.near_angles is None:
        return of.random_orientations(n, a.seed)
    rep, data = ec.read_anglefile(a.near_angles)
    G0 = ec.to_matrices(data, rep=rep)
    rng = np.random.default_rng(a.seed)
    base = G0[rng.integers(len(G0), size=n)]
    v = rng.normal(size=(n, 3))
    v *= np.deg2rad(a.near_spread) * rng.random((n, 1)) / np.linalg.norm(v, axis=1, keepdims=True)
    return base @ Rotation.from_rotvec(v).as_matrix()


def perturb(a, Y, omega, snr, rng):
    s = Y.std(axis=1, keepdims=True)
    gain = 1.0 + a.gain_sd * rng.normal(size=(len(Y), 1))
    off = s * rng.normal(size=(len(Y), 1))
    bg = np.zeros_like(Y)
    if a.bg_amp > 0:
        w = np.deg2rad(omega)[None, :]
        for k in (1, 2, 3):
            ph = rng.uniform(0, 2 * np.pi, size=(len(Y), 1))
            bg += a.bg_amp * s / k * np.cos(k * w + ph)
    noise = 0.0 if np.isinf(snr) else rng.normal(size=Y.shape) * s / snr
    return gain * Y + off + bg + noise


def get_bank(name, model, a):
    path = f"{a.bank_cache}_{name}_sp{a.spacing}_bg{a.n_bg_harm}.npz" if a.bank_cache else None
    if path:
        try:
            b = of.Bank.load(path)
            print(f"[bank] wczytany {path}")
            return b
        except FileNotFoundError:
            pass
    b = of.build_bank(model, a.spacing, n_bg_harm=a.n_bg_harm)
    if path:
        b.save(path)
    return b


def summarize(tag, mis, mis_d4, r_est, r_true):
    big = mis > 2.0
    search_fail = big & (r_est < r_true - 1e-6)
    model_fail = big & ~search_fail
    print(f"  {tag:<22} mediana {np.median(mis):7.3f}°  p90 {np.percentile(mis, 90):7.3f}°  "
          f">1°: {np.mean(mis > 1):5.1%}  >2°: {np.mean(big):5.1%} "
          f"(szukanie {search_fail.sum()}, model {model_fail.sum()})  "
          f"med r_est {np.median(r_est):.4f}  med r_true {np.median(r_true):.4f}")
    pol = big & (mis_d4 < 2.0)
    print(f"  {'':<22} w tym pomyłki polarności (+c/-c, poprawne mod D4): {pol.sum()}")


def main():
    a = parse_args()
    energy = int(a.energy) if a.energy.lstrip("-").isdigit() else a.energy
    mp = ec.MasterPattern(a.master, energy=energy)
    print(mp)
    print("[symetria luster]", {k: round(v, 4) for k, v in of.mirror_symmetry_check(mp).items()})

    fit_models = {}
    if a.gp:
        import gp_echord as G
        gp = G.EchordGP.load(a.gp)
        gpm = of.GPModel(gp)
        a.theta, a.invert = gpm.coneAngle, gpm.invert
        a.sigma_conv, a.dpsi = gpm.sigma_conv_deg, gpm.dpsi
        r, err = of.self_check(gpm)
        print(f"[self_check GP] min r = {r:.8f}, max błąd/std = {err:.2e}")
        if r < 0.9999:
            raise SystemExit("Bank kanoniczny GP nie zgadza się z predict_profiles — sprawdź "
                             "GPModel.canonical_X (konwencja wejścia) zanim pójdziesz dalej.")
    truth = of.ConeModel(mp, a.theta, a.invert, a.sigma_conv, a.dpsi)
    print(f"[geometria] theta = {a.theta}°, invert = {a.invert}, sigma_conv = {a.sigma_conv}°, "
          f"dpsi = {a.dpsi}°, phi0 = {a.phi0}°, sense = {a.sense:+d}")
    if not a.no_cone:
        fit_models["cone"] = truth
    if a.gp:
        fit_models["gp"] = gpm

    banks = {}
    for name, m in fit_models.items():
        src = truth if a.bank_from == "cone" else m
        key = src.name
        if key not in banks:
            banks[key] = get_bank(key, src, a)
    bank_of = {name: banks[truth.name if a.bank_from == "cone" else m.name]
               for name, m in fit_models.items()}

    D4 = of.d4_ops()
    g_true = make_truth(a, a.n_test)
    Y0 = truth.profiles(g_true, a.phi0, a.sense)
    rng = np.random.default_rng(a.seed + 1)
    results = {}
    for snr in a.snr:
        Y = perturb(a, Y0, truth.omega, snr, rng)
        for name, m in fit_models.items():
            t0 = time.perf_counter()
            rec = dict(mis=[], mis_d4=[], r_est=[], r_true=[], seed_rank=[], r_seed=[],
                       euler=[])
            for i, y in enumerate(Y):
                res = of.fit_profile(m, bank_of[name], y, a.phi0, a.sense, top_k=a.top_k,
                                     n_bg_harm=a.n_bg_harm)
                rec["mis"].append(of.misorientation_deg(res["g"], g_true[i])[0])
                rec["mis_d4"].append(of.misorientation_deg(res["g"], g_true[i], D4)[0])
                rec["r_est"].append(res["r"])
                rec["r_true"].append(of.cost_r(m, y, g_true[i], a.phi0, a.sense,
                                               n_bg_harm=a.n_bg_harm)[0])
                rec["seed_rank"].append(res["seed_rank"])
                rec["r_seed"].append(res["r_seed"])
                rec["euler"].append(of.euler_from_matrix(res["g"])[0])
            rec = {k: np.asarray(v) for k, v in rec.items()}
            if name == "gp":
                _, rec["sd_true"] = gpm.profiles_std(g_true, a.phi0, a.sense)
            dt = time.perf_counter() - t0
            tag = f"{name}, snr={snr:g}"
            summarize(tag, rec["mis"], rec["mis_d4"], rec["r_est"], rec["r_true"])
            print(f"  {'':<22} {dt / len(Y):.2f} s/profil, start z rangą > 0: "
                  f"{np.mean(rec['seed_rank'] > 0):.1%}")
            results[tag] = rec

    flat = {f"{t}/{k}": v for t, rec in results.items() for k, v in rec.items()}
    np.savez_compressed(a.out, g_true=g_true, euler_true=of.euler_from_matrix(g_true),
                        Y0=Y0, args=str(vars(a)), **flat)
    print(f"[zapis] {a.out}")


if __name__ == "__main__":
    main()
