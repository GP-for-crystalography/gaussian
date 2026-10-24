"""
Test instalacji EMsoft: symulacja wzorca kanałowania elektronów (ECP)
dla pliku KNO_tetragonal.xtal.

Uruchom z terminala:   python test_emsoft_ecp.py
(wymaga: pip install numpy h5py matplotlib)

Co robi skrypt:
  1. EMMCOpenCL   - Monte Carlo (tryb bse1, wymagany dla ECP)
  2. EMECPmaster  - master pattern ECP
  3. EMECP        - wzorzec dla jednej orientacji i zadanego kąta thetac
  4. rysuje obraz z pliku wynikowego HDF5

Szablony namelist są generowane przez sam EMsoft (opcja -t), a skrypt tylko
podmienia w nich wartości. Dzięki temu nie zależymy od tego, czy pamiętamy
dokładny układ pliku .nml.
"""
import re
import shutil
import subprocess
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- ustawienia
DATA = Path.home() / "EMsoft_data"             # = EMdatapathname z konfiguracji
XTAL = "KNO_tetragonal.xtal"                   # plik w XtalFolder
XTAL_PATH = DATA / "XtalFolder" / XTAL
MC_H5 = "KNO-ECPmaster-15kV.h5"                # plik z Monte Carlo (do niego
                                               # EMECPmaster dopisuje master pattern)
EKEV = 15.0                                    # napięcie przyspieszające [keV]
THETAC = 13.0                                  # połowa kąta rozwarcia stożka [stopnie]
SIGEND = 15.0                                  # największy kąt w Monte Carlo [stopnie],# trzymaj go >= THETAC
TOTNUM_EL = 1_000_000_000                      # mało elektronów = szybki test (domyślnie 2e9)
NPX = 500                                      # rozmiar master patternu = (2*NPX+1)^2 px
NTHREADS = 8
EULER_LIST = [
    (0.0, 0.0, 0.0),      # punkt odniesienia
    (45.0, 0.0, 0.0),     # obrót tylko wokół osi c o 45°
    (30.0, 20.0, 10.0),   # orientacja ogólna
]                         # orientacja (Bunge, w stopniach)



def main():
    # ------------------------------------------------------------------ narzędzia
    def run(cmd):
        """Uruchamia program EMsoft w folderze DATA; wyjście widać na bieżąco."""
        print(f"\n$ {' '.join(cmd)}   (w {DATA})", flush=True)
        subprocess.run(cmd, cwd=DATA, check=True)


    def patch(text, changes, program):
        """W tekście namelisty podmienia wartości pól: nazwa = wartość,"""
        for key, value in changes.items():
            # wartość kończy się na przecinku, wykrzykniku (komentarz) lub końcu linii
            pattern = re.compile(rf"^(\s*{key}\s*=\s*)[^,!\n]*,?", re.MULTILINE | re.IGNORECASE)
            if not pattern.search(text):
                print(f"UWAGA: w szablonie {program} nie znalazłem pola '{key}'"
                    " - sprawdź plik .nml ręcznie.")
                continue
            text = pattern.sub(lambda m: m.group(1) + value + ", ", text, count=1)
        return text


    def make_nml(program, nml_name, changes):
        """program -t  ->  szablon  ->  podmiana wartości  ->  plik .nml."""
        run([program, "-t"])
        template = DATA / f"{program}.template"
        text = patch(template.read_text(), changes, program)
        (DATA / nml_name).write_text(text)
        print(f"Zapisano {DATA / nml_name}")


    def q(s):
        """Wartość tekstowa w apostrofach, jak w namelistach Fortrana."""
        return f"'{s}'"


    # ----------------------------------------------------------------------- start
    if not XTAL_PATH.exists():
        raise SystemExit(f"Brak pliku struktury: {XTAL_PATH}\n"
                        f"Skopiuj {XTAL} do {XTAL_PATH.parent}")

    start = time.time()

    # 1. Monte Carlo (tryb bse1 jest wymagany dla ECP)
    make_nml("EMMCOpenCL", "mc.nml", {
        "xtalname": q(XTAL),
        "mode": q("bse1"),
        "sigstart": "0.0",
        "sigend": f"{SIGEND}",
        "sigstep": "1.0",
        "EkeV": f"{EKEV}D0",
        "Ehistmin": f"{EKEV - 1}D0",
        "Ebinsize": "1.0D0",
        "totnum_el": f"{TOTNUM_EL}",
        "dataname": q(MC_H5),
        "platid": "1",     # numer platformy Rusticl, licząc od 1
        "devid": "1",
    })
    run(["EMMCOpenCL", "mc.nml"])

    if not (DATA / MC_H5).exists():
        raise SystemExit("Monte Carlo nie utworzył pliku wynikowego.")

    # 2. Master pattern ECP (BetheParameters.nml musi leżeć obok)
    make_nml("EMECPmaster", "ecpmaster.nml", {
        "npx": f"{NPX}",
        "energyfile": q(MC_H5),
        "nthreads": f"{NTHREADS}",
    })
    shutil.copy(DATA / "BetheParameters.template", DATA / "BetheParameters.nml")
    run(["EMECPmaster", "ecpmaster.nml"])

    # 3. Wzorzec ECP dla jednej orientacji
    lines = ["eu", str(len(EULER_LIST))] + [", ".join(str(a) for a in e) for e in EULER_LIST]
    (DATA / "angles.txt").write_text("\n".join(lines) + "\n")

    before = {p: p.stat().st_mtime for p in DATA.glob("*.h5")}
    make_nml("EMECP", "ecp.nml", {
        "xtalname": q(XTAL),
        "npix": "261",
        "thetac": f"{THETAC}",
        "maskpattern": q("n"),
        "energyfile": q(MC_H5),
        "masterfile": q(MC_H5),
        "anglefile": q("angles.txt"),
        "eulerconvention": q("hkl"),
        "datafile": q("KNO_ecp.h5"),
    })
    print("\n--- pola z nazwami plików w ecp.nml (sprawdź nazwę pliku wyjściowego) ---")
    for line in (DATA / "ecp.nml").read_text().splitlines():
        if "file" in line.lower() or "name" in line.lower():
            print(line)
    run(["EMECP", "ecp.nml"])
    if not (DATA / "KNO_ecp.h5").exists():
        raise SystemExit("EMECP nie utworzył pliku KNO_ecp.h5.")

    print(f"\nGotowe w {time.time() - start:.0f} s.")

    # 4. Szukamy nowego pliku HDF5 i rysujemy pierwszy sensowny obraz 2D
    new_files = [p for p in DATA.glob("*.h5")
                if p not in before or p.stat().st_mtime > before[p]]
    new_files = [p for p in new_files if p.name != MC_H5]
    if not new_files:
        raise SystemExit("Nie znalazłem nowego pliku .h5 z wynikiem EMECP.")
    out = max(new_files, key=lambda p: p.stat().st_mtime)
    print(f"Plik wynikowy: {out}")

    candidates = []
    with h5py.File(out, "r") as f:
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fiu":
                shape = tuple(s for s in obj.shape if s > 1)
                if len(shape) >= 2 and min(shape[-2:]) >= 64:
                    candidates.append((name, obj.shape))
        f.visititems(collect)
        print("Kandydaci na obraz:", candidates)
        if candidates:
            name = candidates[0][0]
            img = np.squeeze(f[name][()])
            while img.ndim > 2:       # jeśli wzorców jest więcej, bierzemy pierwszy
                img = img[0]
            plt.imshow(img, cmap="gray")
            plt.title(f"{out.name}:{name}")
            plt.colorbar()
            plt.show()

if __name__ == "__main__":
    main()