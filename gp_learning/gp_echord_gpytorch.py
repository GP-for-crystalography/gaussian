"""
gp_echord_gpytorch.py — ten sam emulator co gp_echord.py, ale na GPyTorch.

Model jest IDENTYCZNY (te same wzory, te same 4 hiperparametry), zmienia się narzędzie:
  * jądro      : własna klasa gpytorch.kernels.Kernel; w środku zwykły MaternKernel
                 z GPyTorch na osadzeniu obrotów w R^9 + suma po obrotach symetrii,
  * model      : gpytorch.models.ExactGP + GaussianLikelihood,
  * uczenie    : ExactMarginalLogLikelihood i pętla z optymalizatorem torcha (L-BFGS albo
                 Adam), gradienty z autodiff — żadnych ręcznych pochodnych,
  * dane, symetria, przesunięcie profili, walidacja: z gp_echord (nic nie jest powielane).

Wiele wyjść (1440 punktów psi)
------------------------------
Sprawdzone na GPyTorch 1.15 / linear_operator 0.6:
  * UCZENIE: ExactGP z celami w kształcie (M, N) i jądrem BEZ batch_shape liczy jedną
    faktoryzację — czas i pamięć nie rosną z M, a log-wiarygodność zgadza się z ręczną.
    Z tego korzystamy.
  * PREDYKCJA wbudowana w ExactGP przy takich celach nie działa: błąd kształtu przy
    składaniu wyniku, a przy większym M cache średniej woła torch.cholesky_solve
    z rozgłaszaniem, które materializuje M kopii czynnika Cholesky'ego (dla N=1000,
    M=720 próbowało zaalokować 5,8 GB). Dlatego warunkowanie i predykcję robimy sami:
    jeden Cholesky N x N i Y jako macierz M prawych stron (bez wymiarów batch).
  * MultitaskKernel (ICM) z B = I w tej konfiguracji składał gęstą macierz (NM) x (NM)
    (dla N=1000, M=64: 33 GB) — nie używamy.
GPyTorch domyślnie przy N > 800 przechodzi na metody iteracyjne (CG/Lanczos); tu
wymuszamy dokładnego Cholesky'ego (settings.max_cholesky_size).

Plik celowo nie nazywa się test_*.py.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings

import numpy as np
import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Interval

import echord_cone as ec
import gp_echord as G

DTYPE = torch.float64
MAX_CHOL = 1_000_000  # "zawsze Cholesky"


def _t(x):
    return torch.as_tensor(np.asarray(x), dtype=DTYPE)


# ==========================================================================
# Jądro
# ==========================================================================
class SymRotationKernel(gpytorch.kernels.Kernel):
    """k(R1,R2) = sum_S m(delta(S R1, R2) / ell),  delta^2 = 1/2 ||(R1 - R2) W||_F^2.

    Wejście: macierze kanoniczne spłaszczone do (n, 9). W = diag(kappa, kappa,
    sqrt(2 - kappa^2)) waży kolumny g, czyli osie próbki (jak w gp_echord).
    Osadzenie z = vec(g W)/sqrt(2) daje ||z1 - z2|| = delta, więc wystarczy zwykły
    MaternKernel z GPyTorch (z jego lengthscale = ell w radianach).
    """

    is_stationary = False  # suma po symetriach — GPyTorch nie powinien zakładać stacjonarności

    def __init__(self, ops, nu=1.5, kappa=1.0, learn_kappa=True, **kwargs):
        super().__init__(**kwargs)
        self.base = (gpytorch.kernels.RBFKernel() if nu == np.inf
                     else gpytorch.kernels.MaternKernel(nu=nu))
        self.nu = nu
        self.register_buffer("ops", _t(np.asarray(ops).reshape(-1, 3, 3)))
        self.register_parameter("raw_kappa", torch.nn.Parameter(torch.zeros(1, dtype=DTYPE)))
        self.register_constraint("raw_kappa", Interval(0.05, math.sqrt(2.0) - 1e-3))
        self.kappa = kappa
        self.raw_kappa.requires_grad_(learn_kappa)

    @property
    def kappa(self):
        return self.raw_kappa_constraint.transform(self.raw_kappa)

    @kappa.setter
    def kappa(self, value):
        value = torch.as_tensor(value, dtype=self.raw_kappa.dtype).reshape(1)
        self.initialize(raw_kappa=self.raw_kappa_constraint.inverse_transform(value))

    def _embed(self, x, S=None):
        g = x.reshape(*x.shape[:-1], 3, 3)
        if S is not None:
            g = torch.matmul(S, g)
        k2 = self.kappa.reshape(()) ** 2
        w = torch.sqrt(torch.stack([k2, k2, 2.0 - k2]))  # waga kolumny j = oś próbki j
        return (g * w).reshape(*x.shape[:-1], 9) * math.sqrt(0.5)

    def forward(self, x1, x2, diag=False, **params):
        z2 = self._embed(x2)
        out = None
        for S in self.ops:
            k = self.base.forward(self._embed(x1, S), z2, diag=diag, **params)
            out = k if out is None else out + k
        return out


class CanonicalProfileGP(gpytorch.models.ExactGP):
    """ExactGP: wejście (n, 9), cele (M, n) — M niezależnych wyjść ze wspólnym jądrem."""

    def __init__(self, train_x, train_y, likelihood, ops, nu=1.5, learn_kappa=True):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            SymRotationKernel(ops, nu=nu, learn_kappa=learn_kappa))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


# ==========================================================================
# Emulator (ten sam interfejs co gp_echord.EchordGP)
# ==========================================================================
class TorchEchordGP:
    """Emulator profili kanonicznych na GPyTorch.

    Interfejs jak w gp_echord.EchordGP: fit / condition / predict_canonical /
    predict_profiles / save / load, więc działają też G.evaluate i reszta narzędzi.
    """

    def __init__(self, ops, nu=1.5, ell_deg=1.4, kappa=1.0, signal_var=1.0, noise_var=1e-2,
                 learn_kappa=True):
        self.ops = np.asarray(ops, dtype=float).reshape(-1, 3, 3)
        self.nu = nu
        self.learn_kappa = learn_kappa
        self.init = dict(ell=np.deg2rad(ell_deg), kappa=kappa, s2=signal_var, sn2=noise_var)
        # jądro numpy tylko do kątów dezorientacji (baseline "najbliższy sąsiad" w G.evaluate)
        self.kernel = G.SymMaternKernel(self.ops, nu)
        self.model = None
        self.likelihood = None
        self.history: list[dict] = []
        self.meta: dict = {}
        self._L = None

    # --- budowa modelu GPyTorch
    def _build(self, train_x, train_y):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=GreaterThan(1e-8)).to(DTYPE)
        self.model = CanonicalProfileGP(train_x, train_y, self.likelihood, self.ops, self.nu,
                                        self.learn_kappa).to(DTYPE)
        self.set_hyperparameters(**self.init)

    def set_hyperparameters(self, ell=None, kappa=None, s2=None, sn2=None):
        sk = self.model.covar_module
        f64 = lambda v: torch.as_tensor(v, dtype=DTYPE)  # bez tego GPyTorch zaokrągla do float32
        if ell is not None:
            sk.base_kernel.base.lengthscale = f64(ell)
        if kappa is not None:
            sk.base_kernel.kappa = f64(kappa)
        if s2 is not None:
            sk.outputscale = f64(s2)
        if sn2 is not None:
            self.likelihood.noise = f64(sn2)

    def hyperparameters(self):
        sk = self.model.covar_module
        with torch.no_grad():
            return dict(ell=sk.base_kernel.base.lengthscale.item(),
                        kappa=sk.base_kernel.kappa.item(),
                        s2=sk.outputscale.item(), sn2=self.likelihood.noise.item())

    ell = property(lambda self: self.hyperparameters()["ell"])
    kappa = property(lambda self: self.hyperparameters()["kappa"])
    signal_var = property(lambda self: self.hyperparameters()["s2"])
    noise_var = property(lambda self: self.hyperparameters()["sn2"])

    def describe(self):
        h = self.hyperparameters()
        return (f"ell = {np.rad2deg(h['ell']):.3f}°, kappa = {h['kappa']:.3f}, "
                f"sd_f = {np.sqrt(h['s2']):.3f}, sd_n = {np.sqrt(h['sn2']):.4f}")

    # --- PĘTLA UCZENIA
    def fit(self, data: G.CanonicalData, n_fit=2000, psi_stride=1, optimizer="lbfgs",
            n_iter=100, lr=0.05, tol=1e-10, gtol=1e-6, patience=3, seed=0, verbose=True,
            log_every=1):
        """ML-II na losowym podzbiorze n_fit (ten sam co w gp_echord przy tym samym seed),
        potem warunkowanie na całości.

        optimizer = 'lbfgs' (domyślnie; dla 4 parametrów zbiega w kilkudziesięciu krokach)
                    albo 'adam' (klasyczna pętla z tutoriali GPyTorch; potrzebuje ~200+ kroków).
        Strata = -log p(Y)/(N*M), ta sama skala co 'nll/el' w gp_echord.
        Stop: max |gradient| po parametrach < gtol albo względna zmiana straty < tol
        przez `patience` kolejnych kroków.
        """
        rng = np.random.default_rng(seed)
        self.y_mean = float(data.Y.mean())
        self.y_std = float(data.Y.std())
        self.dpsi = data.dpsi
        self.meta = dict(data.meta)

        idx = rng.choice(len(data), size=min(n_fit, len(data)), replace=False)
        Xf = _t(data.X[idx].reshape(-1, 9))
        Yf = _t((data.Y[idx][:, ::psi_stride].astype(np.float64) - self.y_mean) / self.y_std)
        Yf = Yf.T.contiguous()  # (M, n): każde wyjście to jeden "batch" celów
        self._build(Xf, Yf)
        self.model.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        params = [p for p in self.model.parameters() if p.requires_grad]

        if verbose:
            print(f"[uczenie/GPyTorch] ML-II na {len(idx)} z {len(data)} profili, "
                  f"{Yf.shape[0]} punktów psi/profil, nu = {self.nu}, "
                  f"{len(self.ops)} obrotów symetrii, optymalizator {optimizer}")
            print(f"[uczenie/GPyTorch] start: {self.describe()}")

        def loss_fn():
            return -mll(self.model(Xf), Yf).mean()

        t0 = time.perf_counter()
        self.history = []
        prev, n_small = np.inf, 0
        with gpytorch.settings.max_cholesky_size(MAX_CHOL):
            if optimizer == "lbfgs":
                # max_iter=1: jeden krok L-BFGS (z przeszukiwaniem liniowym) na obrót pętli,
                # historia krzywizny zostaje w optymalizatorze między krokami
                # max_eval trzeba podać: domyślnie max_iter*5//4 = 1, a wtedy przeszukiwanie
                # liniowe ma tylko jedną ewaluację i optymalizator potrafi stanąć w miejscu
                opt = torch.optim.LBFGS(params, lr=1.0, max_iter=1, max_eval=25, history_size=20,
                                        line_search_fn="strong_wolfe")

                def closure():
                    opt.zero_grad()
                    loss = loss_fn()
                    loss.backward()
                    return loss
            elif optimizer == "adam":
                opt = torch.optim.Adam(params, lr=lr)
            else:
                raise ValueError("optimizer musi być 'lbfgs' albo 'adam'")

            for it in range(1, n_iter + 1):
                if optimizer == "lbfgs":
                    loss = opt.step(closure).item()
                else:
                    opt.zero_grad()
                    l_ = loss_fn()
                    l_.backward()
                    opt.step()
                    loss = l_.item()
                gmax = max(p.grad.abs().max().item() for p in params if p.grad is not None)
                rec = dict(it=it, nll=loss, grad_max=gmax, t=time.perf_counter() - t0,
                           **self.hyperparameters())
                self.history.append(rec)
                if verbose and (it % log_every == 0 or it == 1):
                    print(f"  it {it:3d}  nll/el = {loss:+.6f}  {self.describe()}  "
                          f"|grad| = {gmax:.1e}  [{rec['t']:.1f} s]")
                n_small = n_small + 1 if abs(prev - loss) < tol * max(1.0, abs(loss)) else 0
                if gmax < gtol or n_small >= patience:
                    break
                prev = loss
        if verbose:
            print(f"[uczenie/GPyTorch] koniec po {len(self.history)} krokach, "
                  f"{time.perf_counter() - t0:.1f} s")
            print(f"[uczenie/GPyTorch] {self.describe()}")
        self.condition(data, verbose=verbose)
        return self

    # --- warunkowanie: JEDEN Cholesky na wszystkich punktach
    @torch.no_grad()
    def condition(self, data: G.CanonicalData, chunk=1024, verbose=True):
        """K (N x N) blokami, Cholesky, A = K^-1 Y z Y jako macierzą (N, M) prawych stron."""
        t0 = time.perf_counter()
        self.model.eval()
        self.X_train = data.X
        self.angles_train = data.angles
        self.Y_train = data.Y
        self.dpsi = data.dpsi
        X = _t(data.X.reshape(-1, 9))
        N = len(X)
        kern = self.model.covar_module
        self._X = X
        self._kss = float(kern(X[:1], X[:1], diag=True).squeeze())
        K = torch.empty(N, N, dtype=DTYPE)
        for s in range(0, N, chunk):
            K[s:s + chunk] = kern(X[s:s + chunk], X).to_dense()
        K.diagonal().add_(self.noise_var)
        jitter = 0.0
        for attempt in range(6):
            L, info = torch.linalg.cholesky_ex(K)
            if int(info) == 0:
                break
            jitter = self._kss * 1e-8 * 10**attempt
            K.diagonal().add_(jitter)
        else:
            raise RuntimeError("Cholesky nie zbiega się nawet z jitterem")
        del K
        Yn = _t((data.Y.astype(np.float64) - self.y_mean) / self.y_std)
        self._L = L
        self._A = torch.cholesky_solve(Yn, L)
        if verbose:
            extra = f", dodany jitter {jitter:.1e}" if jitter else ""
            print(f"[warunkowanie/GPyTorch] Cholesky {N}x{N}{extra}, "
                  f"{time.perf_counter() - t0:.1f} s")
        return self

    # --- predykcja
    @torch.no_grad()
    def predict_canonical(self, Xs, return_std=True, include_noise=False, chunk=512):
        """Jak gp_echord.EchordGP.predict_canonical (zwraca tablice numpy)."""
        if self._L is None:
            raise RuntimeError("Najpierw fit() albo condition()")
        Xs = _t(np.asarray(Xs).reshape(-1, 9))
        kern = self.model.covar_module
        means, stds = [], []
        for s in range(0, len(Xs), chunk):
            Ks = kern(Xs[s:s + chunk], self._X).to_dense()
            means.append(Ks @ self._A)
            if return_std:
                V = torch.linalg.solve_triangular(self._L, Ks.T, upper=False)
                var = (self._kss - (V * V).sum(0)).clamp_min(0.0)
                stds.append(torch.sqrt(var + (self.noise_var if include_noise else 0.0)))
        mean = torch.cat(means).numpy() * self.y_std + self.y_mean
        if not return_std:
            return mean
        return mean, torch.cat(stds).numpy() * self.y_std

    def predict_profiles(self, orientations, rep="quat", omega_deg=None, phi0=0.0, sense=+1,
                         return_std=True):
        """Profile w geometrii pomiaru (domyślnie omega = 0, 2, ..., 358°), jak w gp_echord."""
        return G.EchordGP.predict_profiles(self, orientations, rep, omega_deg, phi0, sense,
                                           return_std)

    # --- most do wersji numpy (porównania / testy)
    def to_numpy_gp(self, data: G.CanonicalData = None, verbose=False) -> G.EchordGP:
        """EchordGP z gp_echord z tymi samymi hiperparametrami (opcjonalnie od razu uwarunkowany)."""
        h = self.hyperparameters()
        gp = G.EchordGP(self.ops, nu=self.nu)
        gp.theta = np.log([h["ell"], h["kappa"], h["s2"], h["sn2"]])
        gp.y_mean, gp.y_std, gp.dpsi = self.y_mean, self.y_std, self.dpsi
        gp.meta = dict(self.meta)
        return gp.condition(data, verbose=verbose) if data is not None else gp

    # --- zapis / odczyt
    def save(self, path):
        torch.save(dict(hyper=self.hyperparameters(), ops=self.ops, nu=self.nu,
                        learn_kappa=self.learn_kappa, y_mean=self.y_mean, y_std=self.y_std,
                        dpsi=self.dpsi, angles=self.angles_train, Y=self.Y_train,
                        meta=json.dumps(self.meta), history=json.dumps(self.history)), path)

    @classmethod
    def load(cls, path, verbose=True):
        d = torch.load(path, weights_only=False)
        gp = cls(d["ops"], nu=d["nu"], learn_kappa=d["learn_kappa"])
        gp.y_mean, gp.y_std = d["y_mean"], d["y_std"]
        gp.meta, gp.history = json.loads(d["meta"]), json.loads(d["history"])
        data = G.CanonicalData(d["angles"], d["Y"], d["dpsi"], gp.meta)
        gp._build(_t(data.X[:2].reshape(-1, 9)), _t(np.zeros((1, 2))))
        gp.set_hyperparameters(**d["hyper"])
        return gp.condition(data, verbose=verbose)


# ==========================================================================
# CLI
# ==========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Uczenie emulatora GP (GPyTorch) profili eCHORD.")
    p.add_argument("--master", required=True)
    p.add_argument("--angles", required=True, help="plik kątów EMsoftu ('qu' albo 'eu')")
    p.add_argument("--energy", default="max")
    p.add_argument("--theta", type=float, default=40.0)
    p.add_argument("--dpsi", type=float, default=0.25)
    p.add_argument("--invert", action="store_true")
    p.add_argument("--sigma-conv", type=float, default=0.0)
    p.add_argument("--sym", default="auto")
    p.add_argument("--nu", type=float, default=1.5)
    p.add_argument("--ell0", type=float, default=1.4)
    p.add_argument("--n-fit", type=int, default=2000)
    p.add_argument("--psi-stride", type=int, default=1)
    p.add_argument("--optimizer", choices=["lbfgs", "adam"], default="lbfgs")
    p.add_argument("--n-iter", type=int, default=100, help="kroki optymalizatora (Adam: ~300)")
    p.add_argument("--lr", type=float, default=0.05, help="tylko Adam")
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0, help="wątki torcha (0 = domyślnie)")
    p.add_argument("--compare-numpy", action="store_true",
                   help="dla porównania wytrenuj też gp_echord.EchordGP na tym samym podzbiorze")
    p.add_argument("--out", default="gp_echord_torch.pt")
    return p.parse_args()


def main():
    a = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    if a.threads:
        torch.set_num_threads(a.threads)
    t_start = time.perf_counter()
    energy = int(a.energy) if a.energy.lstrip("-").isdigit() else a.energy

    mp = ec.MasterPattern(a.master, energy=energy)
    print(mp)
    sym = G.proper_group_name(mp.space_group) if a.sym == "auto" else a.sym
    ops = G.proper_rotations(sym)
    corr = G.symmetry_check(mp, ops)
    print(f"[symetria] {sym} ({len(ops)} op.), korelacja M(Sd) z M(d): {np.round(corr, 4)}")
    if corr.min() < 0.99:
        print("[symetria] UWAGA: master nie ma tej symetrii — podaj --sym ręcznie.")

    data = G.build_training_data(mp, anglefile=a.angles, coneAngle=a.theta, dpsi=a.dpsi,
                                 invert=a.invert, sigma_conv_deg=a.sigma_conv)

    gp = TorchEchordGP(ops, nu=a.nu, ell_deg=a.ell0)
    gp.fit(data, n_fit=a.n_fit, psi_stride=a.psi_stride, optimizer=a.optimizer,
           n_iter=a.n_iter, lr=a.lr, seed=a.seed)
    gp.meta["sym"] = sym
    res = G.evaluate(gp, mp, n_val=a.n_val, seed=a.seed + 12345)
    gp.meta["validation"] = dict(r_gp_median=float(np.median(res["r_gp"])),
                                 r_nn_median=float(np.median(res["r_nn"])),
                                 z_sd=res["z_sd"], cover2=res["cover2"])

    if a.compare_numpy:
        print("\n[porównanie] gp_echord.EchordGP na tym samym podzbiorze:")
        gpn = G.EchordGP(ops, nu=a.nu, ell_deg=a.ell0)
        gpn.fit(data, n_fit=a.n_fit, psi_stride=a.psi_stride, seed=a.seed, verbose=False)
        resn = G.evaluate(gpn, mp, n_val=a.n_val, seed=a.seed + 12345, verbose=False)
        print(f"  numpy  : {gpn.describe()}, mediana r = {np.median(resn['r_gp']):.4f}")
        print(f"  GPyTorch: {gp.describe()}, mediana r = {np.median(res['r_gp']):.4f}")

    gp.save(a.out)
    print(f"[zapis] {a.out}; całość {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
