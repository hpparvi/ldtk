import numpy.testing as npt
import pytest

from numpy import array, cov, linspace, log, zeros
from numpy.random import default_rng
from scipy.stats import multivariate_normal

from ldtk.ldmodel import Power2Model
from ldtk.ldtk import LDPSet, ldm_with_edge
from ldtk.loglikelihood import ReducedRankLL

COEFFS = array([0.55, 0.45])


def create_synthetic_ldpset(nmu=80, nsamples=200, nfilters=2, seed=0, **kwargs):
    """Create an LDPSet from synthetic power-2 profiles with a smootherstep edge.

    A tiny noise floor stands in for the numerical noise of the real
    (teff, logg, z) interpolation: without it the sample std is exactly zero
    at mu=1 where all normalized profiles equal one.
    """
    rng = default_rng(seed)
    mu = linspace(0.02, 1.0, nmu)
    ldp = zeros((nfilters, nsamples, nmu))
    for iflt in range(nfilters):
        c = rng.normal(0.55 + 0.1 * iflt, 0.02, nsamples)
        a = rng.normal(0.45, 0.02, nsamples)
        for i in range(nsamples):
            ldp[iflt, i] = ldm_with_edge(mu, 0.05, 0.15, array([c[i], a[i]]))
    ldp += rng.normal(0.0, 1e-8, ldp.shape)
    return LDPSet([f'f{i}' for i in range(nfilters)], mu, ldp, **kwargs)


@pytest.fixture(scope='module')
def ps_rr():
    return create_synthetic_ldpset()


@pytest.fixture(scope='module')
def ps_dg():
    return create_synthetic_ldpset(likelihood='diagonal')


def test_matches_analytic_gaussian():
    """With all eigenmodes kept, the lnlike must equal the analytic MVN log-density."""
    rng = default_rng(1)
    ndim, nsamples = 6, 50_000
    a = rng.normal(size=(ndim, ndim))
    samples = rng.multivariate_normal(rng.normal(size=ndim), a @ a.T + ndim * 0.1, size=nsamples)

    ll = ReducedRankLL(linspace(0, 1, ndim), samples, cev=1.0)
    assert ll.nk == ndim
    mvn = multivariate_normal(samples.mean(0), cov(samples, rowvar=False))
    for model in (zeros(ndim), samples[0], samples.mean(0)):
        assert ll(model) == pytest.approx(mvn.logpdf(model), abs=1e-6)


def test_methods_agree():
    rng = default_rng(2)
    samples = rng.multivariate_normal(zeros(4), array([[2., 1., 0., 0.],
                                                       [1., 2., 0., 0.],
                                                       [0., 0., 1., 0.],
                                                       [0., 0., 0., 1.]]), size=10_000)
    mu = linspace(0, 1, 4)
    ll_svd = ReducedRankLL(mu, samples, method='svd')
    ll_eig = ReducedRankLL(mu, samples, method='eigh')
    model = array([0.5, -0.5, 0.1, 0.0])
    assert ll_svd(model) == pytest.approx(ll_eig(model), abs=1e-8)


def test_em_scaling():
    """Scaling the covariance by em**2 must divide chisq by em**2 and shift the normalization."""
    rng = default_rng(3)
    samples = rng.normal(size=(1000, 5))
    ll = ReducedRankLL(linspace(0, 1, 5), samples)
    model = rng.normal(size=5)
    em = 2.0
    chisq = -2.0 * ll(model) - ll.log_det - ll.log_twopi
    expected = -0.5 * (chisq / em ** 2 + ll.log_det + 2 * ll.nk * log(em) + ll.log_twopi)
    assert ll(model, em) == pytest.approx(expected, abs=1e-8)


def test_batched_evaluation():
    rng = default_rng(4)
    samples = rng.normal(size=(1000, 5))
    ll = ReducedRankLL(linspace(0, 1, 5), samples)
    models = rng.normal(size=(7, 5))
    batched = ll(models)
    assert batched.shape == (7,)
    npt.assert_allclose(batched, [ll(m) for m in models], atol=1e-10)


def test_diagonal_mode_unchanged(ps_dg):
    """The diagonal mode must reproduce the legacy (LDTk <= 1.8) per-filter likelihood."""
    m = Power2Model.evaluate(ps_dg._mu, COEFFS)[0, 0]
    expected = ps_dg._lnc1 + ps_dg._lnc2[0] - 0.5 * ((ps_dg._mean[0] - m) ** 2 / ps_dg._err2[0]).sum()
    assert ps_dg.lnlike_p2(COEFFS, flt=0) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize('mode', ['reduced-rank', 'diagonal'])
def test_coefficient_array_shapes(mode, ps_rr, ps_dg):
    """1D+flt, 2D [ipb, icf], and 3D [ipv, ipb, icf] inputs must be consistent."""
    ps = ps_rr if mode == 'reduced-rank' else ps_dg
    per_filter = [ps.lnlike_p2(COEFFS, flt=i) for i in range(2)]
    ldcs2d = array([COEFFS, COEFFS])
    assert ps.lnlike_p2(ldcs2d) == pytest.approx(sum(per_filter), abs=1e-6)
    lnl3d = ps.lnlike_p2(array([ldcs2d, ldcs2d]))
    assert lnl3d.shape == (2,)
    assert lnl3d[0] == pytest.approx(sum(per_filter), abs=1e-6)
    assert lnl3d[0] == pytest.approx(lnl3d[1], abs=1e-10)


def test_resampling_invariance():
    """The reduced-rank likelihood shape must be ~invariant to the mu resolution,
    while the diagonal likelihood sharpens with the number of mu points."""
    c_off = array([0.65, 0.45])

    def dlnl(ps):
        return ps.lnlike_p2(COEFFS, flt=0) - ps.lnlike_p2(c_off, flt=0)

    ratios = {}
    for name, ps in (('rr', create_synthetic_ldpset()),
                     ('dg', create_synthetic_ldpset(likelihood='diagonal'))):
        ps.resample_linear_mu(100)
        d100 = dlnl(ps)
        ps.resample_linear_mu(300)
        ratios[name] = dlnl(ps) / d100

    assert ratios['dg'] > 2.0
    assert abs(ratios['rr'] - 1.0) < 0.5


def test_uncertainty_multiplier():
    ps = create_synthetic_ldpset()
    c_off = array([0.65, 0.45])
    d1 = ps.lnlike_p2(COEFFS, flt=0) - ps.lnlike_p2(c_off, flt=0)
    ps.set_uncertainty_multiplier(2.0)
    d2 = ps.lnlike_p2(COEFFS, flt=0) - ps.lnlike_p2(c_off, flt=0)
    assert d2 == pytest.approx(d1 / 4.0, abs=1e-6)


def test_coeffs_wider_uncertainties(ps_rr, ps_dg):
    """The reduced-rank likelihood must give (much) wider coefficient
    uncertainties than the overconfident diagonal likelihood.

    The synthetic profiles are driven by two latent parameters, and the
    default cumulative-explained-variance truncation should identify the
    two significant eigenmodes on its own.
    """
    assert ps_rr._rrll[0].nk == 2
    _, err_rr = ps_rr.coeffs_qd()
    _, err_dg = ps_dg.coeffs_qd()
    assert (err_rr > err_dg).all()


def test_diagnostics(ps_rr, capsys):
    ps_rr.diagnostics()
    out = capsys.readouterr().out
    for iflt in range(2):
        nk = ps_rr._rrll[iflt].nk
        nall = ps_rr._rrll[iflt].all_eigenvalues.size
        assert f'f{iflt}: {nk} of {nall} eigenmodes kept' in out


def test_likelihood_mode_validation(ps_dg):
    with pytest.raises(ValueError):
        create_synthetic_ldpset(likelihood='full')
    ps = create_synthetic_ldpset()
    ps.set_likelihood_mode('diagonal')
    assert ps.lnlike_p2(COEFFS, flt=0) == pytest.approx(ps_dg.lnlike_p2(COEFFS, flt=0), abs=1e-8)
