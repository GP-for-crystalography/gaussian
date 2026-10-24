"""
Test na syntetycznym master patternie zapisanym w układzie jak z EMsoft
(EMData/EBSDmaster/mLPNH (nsites, nE, n, n), EkeVs, masterSPNH, MCOpenCL/accum_e).
Sprawdza mechanikę kodu; zgodność z prawdziwym EMsoftem sprawdza
MasterPattern.validate_with_stereo() na Twoim pliku.
"""
import time

import h5py
import numpy as np

import echord_cone as ec

rng = np.random.default_rng(0)

# --- syntetyczna "funkcja na sferze": pasma Kikuchiego wokół losowych płaszczyzn
normals = rng.normal(size=(12, 3))
normals /= np.linalg.norm(normals, axis=1, keepdims=True)


def f_true(d, width):
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return np.exp(-((d @ normals.T) ** 2) / (2 * width**2)).sum(-1)


npx, ekevs = 300, np.array([10.0, 15.0, 20.0])
widths = [0.06, 0.05, 0.04]  # każdy kanał energii trochę inny
g = np.arange(-npx, npx + 1) / npx
XX, YY = np.meshgrid(g, g)  # wiersz = Y (indeks j), kolumna = X (indeks i)
XY = np.stack([XX, YY], -1)
dN, dS = ec.lambert_to_sphere(XY, True), ec.lambert_to_sphere(XY, False)
dSP = ec.stereo_to_sphere(XY)
mN = np.stack([f_true(dN, w) for w in widths])[None]  # (1, nE, n, n)
mS = np.stack([f_true(dS, w) for w in widths])[None]
sp = np.stack([np.where(XX**2 + YY**2 <= 1, f_true(dSP, w), 0) for w in widths])

path = "/tmp/synthetic_master.h5"
with h5py.File(path, "w") as h:
    grp = h.create_group("EMData/EBSDmaster")
    grp["mLPNH"], grp["mLPSH"], grp["EkeVs"], grp["masterSPNH"] = mN, mS, ekevs, sp
    h["EMData/MCOpenCL/accum_e"] = rng.random((11, 11, ekevs.size))
    h["CrystalData/SpaceGroupNumber"] = np.array([99])  # P4mm, bez środka symetrii
    h["CrystalData/LatticeParameters"] = np.array([0.3995, 0.3995, 0.4022, 90, 90, 90])

ec.h5_tree(path)
mp = ec.MasterPattern(path, energy="max")
print(mp)

# 1) Lambert: tam i z powrotem
d = rng.normal(size=(100000, 3))
d /= np.linalg.norm(d, axis=1, keepdims=True)
back = ec.lambert_to_sphere(ec.sphere_to_lambert(d), True)
back[:, 2] *= np.sign(d[:, 2])
print("1) Lambert round-trip, max błąd:", np.abs(back - d).max())

# 2) interpolacja vs funkcja analityczna (kanał max energii)
err = mp.lookup(d) - f_true(d, widths[-1])
print("2) lookup vs analityczna: RMS =", np.sqrt(np.mean(err**2)),
      " (skala sygnału ~", f_true(d, widths[-1]).std(), ")")

# 3) macierz Bunge = Rz(phi2) Rx(Phi) Rz(phi1) (obroty pasywne)
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
e = np.deg2rad([37.0, 52.0, 119.0])
print("3) Bunge vs złożenie obrotów:",
      np.abs(ec.bunge_to_matrix(np.rad2deg(e))[0] - Rz(e[2]) @ Rx(e[1]) @ Rz(e[0])).max())

# 3b) kwaterniony EMsoft -> ta sama macierz co z kątów Eulera
E = np.column_stack([rng.uniform(0, 360, 1000), rng.uniform(0, 180, 1000), rng.uniform(0, 360, 1000)])
print("3b) quat_to_matrix(bunge_to_quat(e)) vs bunge_to_matrix(e):",
      np.abs(ec.quat_to_matrix(ec.bunge_to_quat(E)) - ec.bunge_to_matrix(E)).max())
_, pe = mp.cone_profiles(E[:20], 40, 2)
_, pq = mp.cone_profiles(ec.bunge_to_quat(E[:20]), 40, 2, rep="quat")
print("    profile z Eulera vs z kwaternionów:", np.abs(pe - pq).max())

# 4) phi1 = cykliczne przesunięcie profilu (sense=+1: phi1 + k*step -> roll o +k)
coneAngle, angleStep = 40, 2
_, p = mp.cone_profiles([[10.0, 52.0, 119.0], [10.0 + 3 * angleStep, 52.0, 119.0]],
                        coneAngle=coneAngle, angleStep=angleStep)
print("4) phi1 -> roll, max różnica:", np.abs(np.roll(p[0], 3) - p[1]).max())

# 5) test orientacji tablicy przez masterSPNH
print("5) validate_with_stereo:")
res = mp.validate_with_stereo()
assert max(res, key=res.get) == (False, False, False)

# 6) szybkość: 10^4 orientacji x 180 kroków
eul = np.column_stack([rng.uniform(0, 360, 10000), np.rad2deg(np.arccos(rng.uniform(-1, 1, 10000))),
                       rng.uniform(0, 360, 10000)])
t0 = time.perf_counter()
omega, P = mp.cone_profiles(eul, coneAngle=coneAngle, angleStep=angleStep)
print(f"6) {P.shape[0]} profili x {P.shape[1]} punktów: {time.perf_counter() - t0:.2f} s")

# 7) zbieżność wiązki: małe sigma ~ brak rozmycia
_, p0 = mp.cone_profiles(eul[:50], coneAngle, angleStep)
_, p1 = mp.cone_profiles(eul[:50], coneAngle, angleStep, sigma_conv_deg=0.08)
print("7) rozmycie 0.08 deg, względna zmiana:", np.abs(p1 - p0).max() / np.abs(p0).max())

# 8) wykres
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 4))
ax = [fig.add_subplot(1, 3, i) for i in (1, 2, 3)]
mp.plot_cone([300.7, 40.6, 6.7], coneAngle, angleStep, ax=ax[:2])
_, pr = mp.cone_profiles([[300.7, 40.6, 6.7]], coneAngle, angleStep)
ax[2].plot(omega, pr[0])
ax[2].set_xlabel("omega [deg]")
ax[2].set_title("profil")
fig.tight_layout()
fig.savefig("/tmp/synthetic_cone.png", dpi=110)
print("wykres: /tmp/synthetic_cone.png")