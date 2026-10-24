"""
Testy echord_cone (pytest).

Uruchamianie z folderu, w którym leży echord_cone.py:
    pytest -v                 # wszystkie testy, po jednej linii na test
    pytest -v -k quat         # tylko testy z "quat" w nazwie
    pytest -x                 # stop na pierwszym błędzie

Testy sprawdzają mechanikę kodu na syntetycznym master patternie (conftest.py).
Test na prawdziwym pliku EMsoftu (klasa TestRealMaster) jest pomijany, dopóki
nie podasz ścieżki w zmiennej środowiskowej:
    ECHORD_MASTER=/ścieżka/do/masterKTN_014_01.h5 pytest -v -s -k Real
"""
import os

import h5py
import numpy as np
import pytest

import echord_cone as ec


# ---------------------------------------------------------------- pomocnicze
def random_unit_vectors(rng, n):
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def random_eulers(rng, n):
    """Orientacje jednorodne na SO(3) (cos Phi jednorodny)."""
    return np.column_stack([rng.uniform(0, 360, n),
                            np.rad2deg(np.arccos(rng.uniform(-1, 1, n))),
                            rng.uniform(0, 360, n)])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


# ------------------------------------------------------------ projekcja Lamberta
class TestLambert:
    def test_roundtrip(self, rng):
        d = random_unit_vectors(rng, 100_000)
        back = ec.lambert_to_sphere(ec.sphere_to_lambert(d), north=True)
        back[:, 2] *= np.sign(d[:, 2])  # półkulę wybieramy po znaku z
        np.testing.assert_allclose(back, d, atol=1e-12)

    def test_range_and_equator_on_boundary(self, rng):
        d = random_unit_vectors(rng, 10_000)
        xy = ec.sphere_to_lambert(d)
        assert np.all(np.abs(xy) <= 1 + 1e-12)
        phi = rng.uniform(0, 2 * np.pi, 1000)
        eq = np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
        np.testing.assert_allclose(np.abs(ec.sphere_to_lambert(eq)).max(axis=1), 1.0)

    @pytest.mark.parametrize("z", [1.0, -1.0])
    def test_poles_map_to_centre(self, z):
        xy = ec.sphere_to_lambert(np.array([[0.0, 0.0, z]]))
        assert np.all(np.isfinite(xy))
        np.testing.assert_allclose(xy, 0.0)

    def test_equal_area(self, rng):
        """Punkty jednorodne na półkuli -> jednorodne na kwadracie (to definicja
        projekcji równopolowej)."""
        d = random_unit_vectors(rng, 400_000)
        d[:, 2] = np.abs(d[:, 2])
        xy = ec.sphere_to_lambert(d)
        counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=10, range=[[-1, 1], [-1, 1]])
        expected = d.shape[0] / 100
        assert np.all(np.abs(counts - expected) < 5 * np.sqrt(expected))

    def test_stereo_centre_and_rim(self):
        d = ec.stereo_to_sphere(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, -1.0]]))
        np.testing.assert_allclose(d, [[0, 0, 1], [1, 0, 0], [0, -1, 0]], atol=1e-15)


# ----------------------------------------------------------------- orientacje
class TestOrientations:
    @pytest.mark.parametrize("euler", [(0, 0, 0), (37, 52, 119), (300.7, 40.6, 6.7),
                                       (90, 90, 90), (359, 179, 1)])
    def test_bunge_is_rz_rx_rz(self, euler):
        e = np.deg2rad(euler)
        expected = Rz(e[2]) @ Rx(e[1]) @ Rz(e[0])
        np.testing.assert_allclose(ec.bunge_to_matrix(euler)[0], expected, atol=1e-14)

    def test_bunge_is_proper_rotation(self, rng):
        g = ec.bunge_to_matrix(random_eulers(rng, 1000))
        np.testing.assert_allclose(g @ g.transpose(0, 2, 1), np.broadcast_to(np.eye(3), g.shape),
                                   atol=1e-13)
        np.testing.assert_allclose(np.linalg.det(g), 1.0, atol=1e-13)

    def test_radians_flag(self):
        e = np.array([[37.0, 52.0, 119.0]])
        np.testing.assert_allclose(ec.bunge_to_matrix(np.deg2rad(e), degrees=False),
                                   ec.bunge_to_matrix(e))

    def test_quat_matches_euler(self, rng):
        e = random_eulers(rng, 1000)
        np.testing.assert_allclose(ec.quat_to_matrix(ec.bunge_to_quat(e)),
                                   ec.bunge_to_matrix(e), atol=1e-13)

    def test_quat_scalar_part_nonnegative(self, rng):
        assert np.all(ec.bunge_to_quat(random_eulers(rng, 1000))[:, 0] >= 0)

    def test_quat_sign_and_norm_invariance(self, rng):
        q = ec.bunge_to_quat(random_eulers(rng, 100))
        g = ec.quat_to_matrix(q)
        np.testing.assert_allclose(ec.quat_to_matrix(-q), g, atol=1e-14)
        np.testing.assert_allclose(ec.quat_to_matrix(3.7 * q), g, atol=1e-14)

    def test_to_matrices_all_reps_agree(self, rng):
        e = random_eulers(rng, 50)
        g = ec.to_matrices(e, rep="euler")
        np.testing.assert_allclose(ec.to_matrices(ec.bunge_to_quat(e), rep="quat"), g, atol=1e-13)
        np.testing.assert_allclose(ec.to_matrices(g, rep="matrix"), g)

    def test_to_matrices_bad_rep(self):
        with pytest.raises(ValueError):
            ec.to_matrices([[0, 0, 0]], rep="rodrigues")


# ------------------------------------------------------------ pliki kątów
class TestAnglefile:
    def _write(self, path, kind, data, n=None, sep="  "):
        n = len(data) if n is None else n
        lines = [kind, str(n)] + [sep.join(f"{v:.9f}" for v in row) for row in data]
        path.write_text("\n".join(lines) + "\n")
        return str(path)

    def test_quaternions(self, tmp_path, rng):
        q = ec.bunge_to_quat(random_eulers(rng, 25))
        rep, data = ec.read_anglefile(self._write(tmp_path / "a.txt", "qu", q))
        assert rep == "quat"
        np.testing.assert_allclose(data, q, atol=1e-9)

    def test_euler_degrees(self, tmp_path, rng):
        e = random_eulers(rng, 25)
        rep, data = ec.read_anglefile(self._write(tmp_path / "a.txt", "eu", e))
        assert rep == "euler"
        np.testing.assert_allclose(ec.to_matrices(data, rep=rep), ec.bunge_to_matrix(e), atol=1e-8)

    def test_emsoft_layout_from_real_file(self, tmp_path):
        """Dokładnie taki początek, jak w test_ang_quaterion.csv."""
        p = tmp_path / "a.csv"
        p.write_text("qu\n    2\n      0.856243800     -0.347426546     -0.347426546     -0.159487129\n"
                     "      0.856243800     -0.352796707     -0.352796707     -0.133848874\n")
        rep, data = ec.read_anglefile(str(p))
        assert rep == "quat" and data.shape == (2, 4)
        np.testing.assert_allclose(np.linalg.norm(data, axis=1), 1.0, atol=1e-6)

    def test_comma_separated(self, tmp_path, rng):
        q = ec.bunge_to_quat(random_eulers(rng, 5))
        _, data = ec.read_anglefile(self._write(tmp_path / "a.csv", "qu", q, sep=", "))
        np.testing.assert_allclose(data, q, atol=1e-9)

    def test_count_mismatch_warns(self, tmp_path, rng):
        q = ec.bunge_to_quat(random_eulers(rng, 5))
        with pytest.warns(UserWarning):
            ec.read_anglefile(self._write(tmp_path / "a.txt", "qu", q, n=7))

    def test_unsupported_type(self, tmp_path):
        with pytest.raises(ValueError):
            ec.read_anglefile(self._write(tmp_path / "a.txt", "ro", np.ones((2, 4))))


# ------------------------------------------------------------ geometria stożka
class TestConeGeometry:
    @pytest.mark.parametrize("coneAngle", [10, 40])
    @pytest.mark.parametrize("angleStep", [1, 2])
    def test_cone_directions(self, coneAngle, angleStep):
        omega, d = ec.cone_directions(coneAngle, angleStep)
        assert omega.size == 360 // angleStep
        assert omega[0] == 0 and omega[-1] == 360 - angleStep
        np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0)
        np.testing.assert_allclose(np.rad2deg(np.arccos(d[:, 2])), coneAngle)

    def test_sense_reverses_azimuth(self):
        _, dp = ec.cone_directions(40, 2, sense=+1)
        _, dm = ec.cone_directions(40, 2, sense=-1)
        np.testing.assert_allclose(dm[:, 1], -dp[:, 1], atol=1e-15)  # odbicie y -> -y
        np.testing.assert_allclose(dm[:, [0, 2]], dp[:, [0, 2]], atol=1e-15)

    def test_convergence_weights_and_spread(self, rng):
        _, d = ec.cone_directions(40, 2)
        sigma = 0.5
        dd, w = ec.convergence_directions(d, sigma_deg=sigma, n_quad=3)
        assert dd.shape == (d.shape[0], 9, 3)
        np.testing.assert_allclose(w.sum(), 1.0)
        np.testing.assert_allclose(np.linalg.norm(dd, axis=-1), 1.0)
        # dla 2D gaussa o odchyleniu sigma na oś: <theta^2> = 2 sigma^2
        theta2 = np.arccos(np.clip(np.einsum("mqi,mi->mq", dd, d), -1, 1)) ** 2
        np.testing.assert_allclose(theta2 @ w, 2 * np.deg2rad(sigma) ** 2, rtol=1e-3)


# --------------------------------------------------------- odczyt master patternu
class TestReduceMaster:
    def test_2d_passthrough(self):
        a = np.arange(9.0).reshape(3, 3)
        np.testing.assert_array_equal(ec._reduce_master(a, None, "max", None), a)

    def test_no_energy_axis_sums_leading(self):
        a = np.random.default_rng(0).random((4, 5, 5))
        np.testing.assert_allclose(ec._reduce_master(a, None, "max", None), a.sum(0))

    def test_sites_summed_energy_selected(self):
        a = np.random.default_rng(0).random((2, 3, 5, 5))  # (sites, nE, ny, nx)
        ek = np.array([10.0, 20.0, 15.0])                  # max nie musi być ostatni
        np.testing.assert_allclose(ec._reduce_master(a, ek, "max", None), a.sum(0)[1])
        np.testing.assert_allclose(ec._reduce_master(a, ek, 2, None), a.sum(0)[2])

    def test_weighted(self):
        a = np.random.default_rng(0).random((1, 3, 5, 5))
        w = np.array([1.0, 2.0, 7.0])
        expected = np.tensordot(w / w.sum(), a[0], axes=1)
        np.testing.assert_allclose(ec._reduce_master(a, np.array([1., 2., 3.]), "weighted", w),
                                   expected)

    def test_bad_energy(self):
        a = np.zeros((1, 3, 5, 5))
        with pytest.raises(ValueError):
            ec._reduce_master(a, np.array([1., 2., 3.]), "average", None)


class TestMasterPatternLoading:
    def test_metadata(self, mp, synthetic):
        assert mp.npx == 300
        assert mp.NH.shape == mp.SH.shape == (601, 601)
        assert mp.group == "EMData/EBSDmaster"
        np.testing.assert_array_equal(mp.ekevs, synthetic.ekevs)
        assert mp.eweights is not None
        assert "kryształ" in repr(mp)

    def test_energy_choices(self, synthetic):
        with h5py.File(synthetic.path, "r") as f:
            raw = f["EMData/EBSDmaster/mLPNH"][()]
        top = ec.MasterPattern(synthetic.path, energy="max")
        first = ec.MasterPattern(synthetic.path, energy=0)
        wtd = ec.MasterPattern(synthetic.path, energy="weighted")
        np.testing.assert_allclose(top.NH, raw[0, 2])
        np.testing.assert_allclose(first.NH, raw[0, 0])
        w = synthetic.eweights / synthetic.eweights.sum()
        np.testing.assert_allclose(wtd.NH, np.tensordot(w, raw[0], axes=1))

    def test_multiple_sites_are_summed(self, make_master):
        one = ec.MasterPattern(make_master(sites=1))
        three = ec.MasterPattern(make_master(sites=3))
        np.testing.assert_allclose(three.NH, one.NH, rtol=1e-12)

    def test_missing_dataset_raises(self, tmp_path):
        path = tmp_path / "empty.h5"
        with h5py.File(path, "w") as h:
            h["EMData/something"] = np.zeros(3)
        with pytest.raises(KeyError):
            ec.MasterPattern(str(path))

    @pytest.mark.parametrize("sg, centro", [
        (1, False), (2, True), (3, False), (10, True), (15, True), (16, False),
        (46, False), (47, True), (74, True), (75, False), (83, True), (88, True),
        (89, False), (99, False), (122, False), (123, True), (142, True), (143, False),
        (147, True), (148, True), (149, False), (161, False), (162, True), (167, True),
        (168, False), (175, True), (176, True), (177, False), (186, False), (191, True),
        (194, True), (195, False), (200, True), (206, True), (207, False), (220, False),
        (221, True), (225, True), (230, True)])  # granice każdego przedziału
    def test_centrosymmetric(self, make_master, sg, centro):
        assert ec.MasterPattern(make_master(space_group=sg)).centrosymmetric is centro


# ------------------------------------------------------------------- lookup
class TestLookup:
    def test_matches_analytic(self, mp, synthetic, rng):
        d = random_unit_vectors(rng, 50_000)
        truth = synthetic.f(d, synthetic.widths[-1])
        rms = np.sqrt(np.mean((mp.lookup(d) - truth) ** 2))
        assert rms < 0.01 * truth.std()

    def test_cubic_more_accurate_than_bilinear(self, synthetic, rng):
        d = random_unit_vectors(rng, 50_000)
        truth = synthetic.f(d, synthetic.widths[-1])
        rms = {o: np.sqrt(np.mean((ec.MasterPattern(synthetic.path, order=o).lookup(d)
                                   - truth) ** 2)) for o in (1, 3)}
        assert rms[3] < rms[1]

    def test_continuous_across_equator(self, mp, rng):
        phi = rng.uniform(0, 2 * np.pi, 2000)
        eps = 1e-6
        up = np.column_stack([np.cos(phi), np.sin(phi), np.full_like(phi, eps)])
        down = up * [1, 1, -1]
        np.testing.assert_allclose(mp.lookup(up), mp.lookup(down), atol=1e-3)

    def test_unnormalized_input(self, mp, rng):
        d = random_unit_vectors(rng, 100)
        np.testing.assert_allclose(mp.lookup(5.0 * d), mp.lookup(d))

    def test_keeps_input_shape(self, mp, rng):
        d = random_unit_vectors(rng, 24).reshape(2, 3, 4, 3)
        assert mp.lookup(d).shape == (2, 3, 4)


# ------------------------------------------------------------------- profile
class TestConeProfiles:
    def test_shapes(self, mp, rng):
        omega, p = mp.cone_profiles(random_eulers(rng, 7), coneAngle=40, angleStep=2)
        assert omega.shape == (180,)
        assert p.shape == (7, 180)
        assert np.all(np.isfinite(p))

    @pytest.mark.parametrize("sense", [+1, -1])
    @pytest.mark.parametrize("m", [1, 3, 17])
    def test_phi1_is_cyclic_shift(self, mp, sense, m):
        """I(omega) = F(phi1 - phi0 - sense*omega): phi1 + m*step -> roll o sense*m."""
        step = 2
        e = np.array([[10.0, 52.0, 119.0], [10.0 + m * step, 52.0, 119.0]])
        _, p = mp.cone_profiles(e, coneAngle=40, angleStep=step, sense=sense)
        np.testing.assert_allclose(p[1], np.roll(p[0], sense * m), atol=1e-12)

    def test_phi0_equivalent_to_phi1(self, mp, rng):
        e = random_eulers(rng, 10)
        e_shift = e.copy()
        e_shift[:, 0] -= 25.0
        _, a = mp.cone_profiles(e, 40, 2, phi0=25.0)
        _, b = mp.cone_profiles(e_shift, 40, 2, phi0=0.0)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_invert_is_noop_for_symmetric_function(self, mp, rng):
        """Syntetyczne f spełnia f(-d) = f(d), więc invert nie może nic zmienić."""
        e = random_eulers(rng, 10)
        _, a = mp.cone_profiles(e, 40, 2)
        _, b = mp.cone_profiles(e, 40, 2, invert=True)
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_quat_input_matches_euler(self, mp, rng):
        e = random_eulers(rng, 20)
        _, a = mp.cone_profiles(e, 40, 2)
        _, b = mp.cone_profiles(ec.bunge_to_quat(e), 40, 2, rep="quat")
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_chunking_does_not_change_result(self, mp, rng):
        e = random_eulers(rng, 50)
        _, a = mp.cone_profiles(e, 40, 2, chunk=7)
        _, b = mp.cone_profiles(e, 40, 2, chunk=1024)
        np.testing.assert_array_equal(a, b)

    def test_small_convergence_barely_changes(self, mp, rng):
        e = random_eulers(rng, 30)
        _, a = mp.cone_profiles(e, 40, 2)
        _, b = mp.cone_profiles(e, 40, 2, sigma_conv_deg=0.08)
        assert np.abs(b - a).max() < 0.02 * np.abs(a).max()

    def test_profile_matches_analytic(self, mp, synthetic, rng):
        """Profil z interpolacji vs funkcja analityczna w tych samych kierunkach."""
        e = random_eulers(rng, 5)
        _, p = mp.cone_profiles(e, 40, 2)
        _, ds = ec.cone_directions(40, 2)
        dc = np.einsum("nij,mj->nmi", ec.bunge_to_matrix(e), ds)
        truth = synthetic.f(dc, synthetic.widths[-1])
        assert np.sqrt(np.mean((p - truth) ** 2)) < 0.01 * truth.std()


# --------------------------------------------------------------- diagnostyka
class TestDiagnostics:
    def test_validate_with_stereo_picks_identity(self, mp):
        res = mp.validate_with_stereo()
        best = max(res, key=res.get)
        assert best == (False, False, False)
        assert res[best] > 0.999
        assert mp._orient == (False, False, False)  # test przywraca stan

    def test_validate_detects_wrong_orientation(self, synthetic):
        wrong = ec.MasterPattern(synthetic.path, transpose=True)
        res = wrong.validate_with_stereo()
        assert max(res, key=res.get) == (False, False, False)
        assert res[(True, False, False)] < 0.9  # bieżące ustawienie wypada źle

    def test_plot_cone_smoke(self, mp):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = mp.plot_cone([300.7, 40.6, 6.7], coneAngle=40, angleStep=2)
        assert len(ax) == 2
        plt.close("all")


# ------------------------------------------------- prawdziwy plik EMsoftu
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.abspath(
    os.path.join(ROOT_DIR, "../2026_10_19_example_PZ")
)
REAL_MASTER = os.environ.get("ECHORD_MASTER", os.path.join(DATA_DIR, "masterKTN_014_01.h5"))


@pytest.fixture(scope="module")
def real():
    return ec.MasterPattern(REAL_MASTER)


@pytest.mark.skipif(not REAL_MASTER, reason="ustaw ECHORD_MASTER=ścieżka/do/master.h5")
class TestRealMaster:

    def test_loads(self, real):
        print(real)
        assert np.all(np.isfinite(real.NH)) and np.all(np.isfinite(real.SH))
        assert real.NH.std() > 0

    def test_array_orientation_matches_emsoft_stereo(self, real):
        res = real.validate_with_stereo()
        if res is None:
            pytest.skip("w pliku nie ma masterSPNH")
        identity = res[(False, False, False)]
        assert identity >= max(res.values()) - 1e-3  # przy symetrii m-3m remis jest OK
        assert identity > 0.9

    def test_profiles_finite(self, real):
        _, p = real.cone_profiles([[300.7, 40.6, 6.7], [10, 20, 30]], 40, 2)
        assert np.all(np.isfinite(p)) and p.std() > 0
