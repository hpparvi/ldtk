"""
Limb darkening toolkit
Copyright (C) 2015-2026  Hannu Parviainen <hpparvi@gmail.com>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

from functools import partial
from pathlib import Path
from pickle import load, dump
from typing import Optional, Union, List

import astropy.io.fits as pf
from numba import njit
from numpy import argmin, zeros, sqrt, array, diff, log, linspace, ones, diag, exp, cov, asarray, percentile, arange, \
    clip, full_like, inf, ndarray, full, isfinite
from numpy.random import normal, multivariate_normal, uniform
from scipy.interpolate import interp1d, LinearNDInterpolator as NDI
from scipy.optimize import fmin, minimize
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type

from .client import Client
from .core import TWO_PI, dx2, a_lims_hilo, a_lims, TEFF_POINTS, LOGG_POINTS, Z_POINTS, is_root, with_mpi, comm, message
from .loglikelihood import ReducedRankLL
from .rbf import RBFProfileInterpolator
from .ldmodel import (LinearModel, QuadraticModel, TriangularQuadraticModel, SquareRootModel, NonlinearModel,
                      GeneralModel, Power2Model, Power2MPModel, models, ld_power_2)


def load_ldpset(filename):
    with open(filename, 'rb') as fin:
        return LDPSet(load(fin), load(fin), load(fin))


@njit
def lnlike1d(model, fid, _lnc1, _lnc2, _mean, _err2):
    return _lnc1 + _lnc2[fid] - 0.5 * ((_mean[fid] - model) ** 2 / _err2[fid]).sum()


@njit
def lnlike2d(model, _lnc1, _lnc2, _mean, _err2):
    nfilters = model.shape[0]
    lnl = 0.0
    for fid in range(nfilters):
        lnl += _lnc1 + _lnc2[fid] - 0.5 * ((_mean[fid] - model[fid]) ** 2 / _err2[fid]).sum()
    return lnl


@njit
def lnlike3d(model, _lnc1, _lnc2, _mean, _err2):
    npv = model.shape[0]
    nfilters = model.shape[1]

    lnl = zeros(npv)
    for ipv in range(npv):
        for fid in range(nfilters):
            lnl[ipv] += _lnc1 + _lnc2[fid] - 0.5 * ((_mean[fid] - model[ipv, fid]) ** 2 / _err2[fid]).sum()
    return lnl


def smootherstep(x, e0: float, e1: float):
    x = clip((x-e0) / (e1-e0), 0.0, 1.0)
    return x * x * x * (x * (6*x - 15.) + 10.)


def ldm_with_edge(mu, e0, e1, ldc, ldm=ld_power_2):
    if e0 > e1:
        return full_like(mu, inf)
    nmu = clip((mu-e1)/(1-e1), 0.0, 1.0)
    return smootherstep(mu, e0, e1) * ldm(nmu, ldc)


# Main classes
# ============
class LDPSet(object):
    """Limb darkening profile set

    Parameters
    ----------
    filters : list
        List of Filter-instances defining the passbands
    mu : array_like
        Array of mu values
    ldp_samples : list
        A list containing arrays of limb darkening profile samples for each filter
    likelihood : str, optional
        Log-likelihood mode, either 'reduced-rank' (default) or 'diagonal'.
        The reduced-rank mode evaluates a Normal log-likelihood in the
        principal subspace of the profile sample covariance, which accounts
        for the strong correlations between the mu points and makes the
        likelihood insensitive to the mu-grid resolution set by `resample`.
        The 'diagonal' mode restores the legacy (LDTk <= 1.8) behavior that
        assumes the mu points are independent: it overestimates the constraining power
        of the profiles by a large factor and its sharpness scales with the
        number of mu points, so it is kept only for comparison purposes.
    cev : float, optional
        Cumulative explained variance threshold for the reduced-rank
        likelihood: the smallest number of leading eigenmodes explaining at
        least this fraction of the total profile sample variance is kept.
    nk : int, optional
        Optional hard limit on the number of eigenmodes kept by the
        reduced-rank likelihood.
    """

    def __init__(self, filters, mu, ldp_samples, likelihood: str = 'reduced-rank',
                 cev: float = 0.999, nk: Optional[int] = None):
        self._filters = filters
        self._nfilters = len(filters)
        self._mu = mu
        self._z = sqrt(1 - mu ** 2)
        self._ldps = ldp_samples
        self._mean = array([ldp_samples[i, :, :].mean(0) for i in range(self._nfilters)])
        self._std = array([ldp_samples[i, :, :].std(0) for i in range(self._nfilters)])
        self._samples = {m.abbr: [] for m in models.values()}

        self._ldps_orig = self._ldps.copy()
        self._mu_orig = self._mu.copy()
        self._z_orig = self._z.copy()
        self._mean_orig = self._mean.copy()
        self._std_orig = self._std.copy()
        self._em = 1.0
        self._rr_cev = cev
        self._rr_nk = nk
        self._rrll: List[ReducedRankLL] = []
        self.set_likelihood_mode(likelihood)

        self.fit_limb()

        self._lnl = zeros(self._nfilters)
        self.set_uncertainty_multiplier(1.)
        self._update()

        self.lnlike_ln = partial(self._lnlike, ldmodel=LinearModel)
        self.lnlike_qd = partial(self._lnlike, ldmodel=QuadraticModel)
        self.lnlike_tq = partial(self._lnlike, ldmodel=TriangularQuadraticModel)
        self.lnlike_sq = partial(self._lnlike, ldmodel=SquareRootModel)
        self.lnlike_nl = partial(self._lnlike, ldmodel=NonlinearModel)
        self.lnlike_ge = partial(self._lnlike, ldmodel=GeneralModel)
        self.lnlike_p2 = partial(self._lnlike, ldmodel=Power2Model)
        self.lnlike_p2mp = partial(self._lnlike, ldmodel=Power2MPModel)

        self.coeffs_ln = partial(self._coeffs, ldmodel=LinearModel)
        self.coeffs_qd = partial(self._coeffs, ldmodel=QuadraticModel)
        self.coeffs_tq = partial(self._coeffs, ldmodel=TriangularQuadraticModel)
        self.coeffs_sq = partial(self._coeffs, ldmodel=SquareRootModel)
        self.coeffs_nl = partial(self._coeffs, ldmodel=NonlinearModel)
        self.coeffs_ge = partial(self._coeffs, ldmodel=GeneralModel)
        self.coeffs_p2 = partial(self._coeffs, ldmodel=Power2Model)
        self.coeffs_p2mp = partial(self._coeffs, ldmodel=Power2MPModel)

        self.lnlike_ln.__doc__ = "Linear limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_qd.__doc__ = "Quadratic limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_tq.__doc__ = "Triangular quadratic limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_sq.__doc__ = "Square root limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_nl.__doc__ = "Nonlinear limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_ge.__doc__ = "General limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_p2.__doc__ = "Power-2 limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_p2mp.__doc__ = "Power-2 limb darkening model with an alternative parametrisation\n(coeffs, join=True, flt=None)"

        self.coeffs_ln.__doc__ = "Estimate the linear limb darkening model coefficients, see LPDSet._coeffs for details."
        self.coeffs_qd.__doc__ = "Estimate the quadratic limb darkening model coefficients, see LPDSet._coeffs for details."
        self.coeffs_tq.__doc__ = "Estimate the triangular quadratic limb darkening model coefficients, see LPDSet._coeffs for details."
        self.coeffs_sq.__doc__ = "Estimate the square root limb darkening model coefficients, see LPDSet._coeffs for details."
        self.coeffs_nl.__doc__ = "Estimate the nonlinear limb darkening model coefficients, see LPDSet._coeffs for details."
        self.coeffs_ge.__doc__ = "Estimate the general limb darkening model coefficients, see LPDSet._coeffs for details."
        self.coeffs_p2.__doc__ = "Estimate the power-2 limb darkening model coefficients, see LPDSet._coeffs for details."
        self.coeffs_p2mp.__doc__ = "Estimate the power-2 MP limb darkening model coefficients, see LPDSet._coeffs for details."

    def save(self, filename):
        """Saves the LDPSet as a pickle

        Parameters
        ----------
        filename : string
            Filename
        """
        with open(filename, 'wb') as f:
            dump(self._filters, f)
            dump(self._mu_orig, f)
            dump(self._ldps_orig, f)

    def _update(self):
        self._nmu = self._mu.size
        self._lnc1 = -0.5 * self._nmu * log(TWO_PI)  ## 1st ln likelihood term
        self._lnc2 = array([-log(self._em * e).sum() for e in self._std])  ## 2nd ln likelihood term
        self._err2 = array([(self._em * e) ** 2 for e in self._std])  ## variances
        self._rrll = [ReducedRankLL(self._mu, self._ldps[i], cev=self._rr_cev, nk=self._rr_nk)
                      for i in range(self._nfilters)]

    def set_likelihood_mode(self, mode: str):
        """Set the log-likelihood mode.

        Parameters
        ----------
        mode : str
            Either 'reduced-rank' (default, accounts for the correlations
            between the mu points) or 'diagonal' (the legacy LDTk <= 1.8
            behavior that assumes independent mu points).
        """
        if mode not in ('reduced-rank', 'diagonal'):
            raise ValueError(f"Unknown likelihood mode '{mode}', should be either 'reduced-rank' or 'diagonal'.")
        self._lh_mode = mode

    def fit_limb(self):
        def minfun(x, mu, flux):
            return ((flux - ldm_with_edge(mu, x[0], x[1], x[2:])) ** 2).sum()
        mu_new = linspace(self._mu_orig[0], 1, 1500)
        flux_new = interp1d(self._mu_orig, self._mean_orig.mean(0), 'quadratic')(mu_new)
        res = minimize(minfun, array([0.05, 0.15, 0.5, 1.5]), (mu_new, flux_new), method='Nelder-Mead')
        self._limb_minimization = res
        self.set_limb_mu(res.x[1])

    def set_limb_z(self, z):
        """Set the z value that defines the edge of the stellar disk

        Parameters
        ----------
        z : float
            The z that defines the edge of the stellar disk
        """
        self._limb_z = z
        self._limb_i = argmin(abs(self._z_orig - z))
        self._limb_mu = sqrt(1. - z ** 2)
        self.reset_sampling()

    def set_limb_mu(self, mu):
        self._limb_i = argmin(abs(self._mu_orig - mu))
        self._limb_mu = self._mu_orig[self._limb_i]
        self._limb_z = sqrt(1. - self._limb_mu**2)
        self.reset_sampling()

    def redefine_limb(self):
        self._z = self._z_orig[self._limb_i:] / self._limb_z
        self._mu = sqrt(1. - self._z ** 2)
        self._ldps = self._ldps_orig[:, :, self._limb_i:].copy()
        self._mean = self._mean_orig[:, self._limb_i:].copy()
        self._std = self._std_orig[:, self._limb_i:].copy()

    def set_uncertainty_multiplier(self, em):
        """Set a multiplier that scales the profile uncertainties.

        Scales the profile sample covariance by em**2 to account for the
        imperfections of the stellar atmosphere models: the simulated
        profiles can contain unknown biases and trends that do not agree
        with observations. With the reduced-rank likelihood it is no longer
        needed to correct for the overconfidence of the diagonal likelihood.
        """
        self._em = em
        self._update()

    def reset_sampling(self):
        self.redefine_limb()
        self._update()

    def resample_linear_z(self, nz=100):
        self.resample(z=linspace(0, 1, nz))

    def resample_linear_mu(self, nmu=100):
        self.resample(mu=linspace(0, 1, nmu))

    def resample(self, mu=None, z=None):
        muc = self._mu.copy()
        if z is not None:
            self._z = z
            self._mu = sqrt(1 - self._z ** 2)
        elif mu is not None:
            self._mu = mu
            self._z = sqrt(1 - self._mu ** 2)

        self._ldps = array([interp1d(muc, f, kind='cubic')(self._mu) for f in self._ldps])
        self._mean = array([self._ldps[i, :, :].mean(0) for i in range(self._nfilters)])
        self._std = array([self._ldps[i, :, :].std(0) for i in range(self._nfilters)])
        self._update()

    def _coeffs(self, return_cm=False, do_mc=False, n_mc_samples=20000, mc_thin=25, mc_burn=25,
                ldmodel=QuadraticModel, ngc=4):
        """
        Estimate the limb darkening coefficients and their uncertainties for a given limb darkening  model.

        Parameters

          return_cm    bool     returns the full covariance matrix if set true, otherwise returns
                                the std of the likelihood distribution for each parameter.

          do_mc        bool     estimates the coefficient uncertainties using MCMC sampling

          n_mc_samples int      number of MCMC iterations to run if MCMC is used

          mc_thin      int      MCMC chain thinning factor

          mc_burn      int      MCMC chain burn in

          ldmodel      LDModel  limb darkening model to fit
        """
        npar = ldmodel.npar or ngc

        qcs = []
        x0 = full(npar, 0.1)
        for iflt in range(self._nfilters):
            qcs.append(fmin(lambda pv: -self._lnlike(pv, flt=iflt, ldmodel=ldmodel), x0, disp=0))
            x0 = qcs[-1]

        covs = []
        for iflt, qc in enumerate(qcs):
            s = zeros(npar)
            for ic in range(npar):
                s[ic] = (1. / sqrt(-dx2(lambda x: self._lnlike(x, flt=iflt, ldmodel=ldmodel), qc, 1e-5, dim=ic)))

            ## Simple MCMC uncertainty estimation
            ## ----------------------------------
            if do_mc:
                logl = zeros(n_mc_samples)
                chain = zeros([n_mc_samples, npar])

                chain[0, :] = qc
                logl[0] = self._lnlike(chain[0], flt=iflt, ldmodel=ldmodel)

                for i in range(1, n_mc_samples):
                    pos_t = multivariate_normal(chain[i - 1], diag(s ** 2))
                    logl_t = self._lnlike(pos_t, flt=iflt, ldmodel=ldmodel)
                    if uniform() < exp(logl_t - logl[i - 1]):
                        chain[i, :] = pos_t
                        logl[i] = logl_t
                    else:
                        chain[i, :] = chain[i - 1, :]
                        logl[i] = logl[i - 1]
                self._samples[ldmodel.abbr].append(chain)
                ch = chain[mc_burn::mc_thin, :]

                if return_cm:
                    covs.append(cov(ch, rowvar=0))
                else:
                    covs.append(sqrt(cov(ch, rowvar=0)) if npar == 1 else sqrt(cov(ch, rowvar=0).diagonal()))

            else:
                if return_cm:
                    covs.append(s ** 2 if npar == 1 else diag(s ** 2))
                else:
                    covs.append(s)

        return array(qcs), array(covs)

    def _lnlike(self, ldcs, joint=None, flt=None, ldmodel=QuadraticModel):
        ldcs = asarray(ldcs)

        if joint is not None:
            raise DeprecationWarning(
                "The argument 'joint' has been deprecated in LDTk 1.1 and will be removed in the future.")
        if (ldcs.ndim == 1) and (flt is None) and (self._nfilters > 1):
            raise ValueError(
                'Need to give the filter id `flt` if evaluating a single set of coefficients with multiple filters defined.')

        m = ldmodel.evaluate(self._mu, ldcs)

        if self._lh_mode == 'reduced-rank':
            if flt is not None:
                return self._rrll[flt](m[0, 0], self._em)
            elif ldcs.ndim == 2:
                return sum(self._rrll[i](m[0, i], self._em) for i in range(self._nfilters))
            elif ldcs.ndim == 3:
                lnl = zeros(m.shape[0])
                for i in range(self._nfilters):
                    lnl += self._rrll[i](m[:, i, :], self._em)
                return lnl
        else:
            if flt is not None:
                return lnlike1d(m, flt, self._lnc1, self._lnc2, self._mean, self._err2)
            elif ldcs.ndim == 2:
                return lnlike2d(m[0], self._lnc1, self._lnc2, self._mean, self._err2)
            elif ldcs.ndim == 3:
                return lnlike3d(m, self._lnc1, self._lnc2, self._mean, self._err2)

    def diagnostics(self, nmodes: int = 6):
        """Print a per-filter summary of the reduced-rank likelihood decomposition.

        Shows the number of eigenmodes kept by the cumulative-explained-variance
        truncation and the leading relative eigenvalue spectrum. A kept mode
        count that grows beyond ~3-4 indicates that the stellar parameter
        posterior explores a strongly nonlinear region of the spectrum grid
        (or that the truncation threshold needs attention).

        Parameters
        ----------
        nmodes : int, optional
            Number of leading eigenmodes to list per filter.
        """
        for name, ll in zip(self._filters, self._rrll):
            rel = ll.all_eigenvalues / ll.all_eigenvalues.sum()
            cum = rel.cumsum()
            n = min(nmodes, rel.size)
            print(f"{name}: {ll.nk} of {rel.size} eigenmodes kept (cev = {self._rr_cev})")
            print("  mode  rel. eigenvalue  cumulative")
            for i in range(n):
                kept = '*' if i < ll.nk else ' '
                print(f"  {i + 1:3d}{kept}  {rel[i]:15.3e}  {cum[i]:10.6f}")

    @property
    def profile_averages(self):
        """The average limb darkening profiles for each passband
        """
        return self._mean

    @property
    def profile_uncertainties(self):
        """The limb darkening profile uncertainties for each passband
        """
        return self._std


class LDPSetCreator(object):
    """Creates a limb darkening profile set.

    Parameters
    ----------
    teff : tuple or 1D ndarray
        Effective stellar temperature either as a (value, uncertainty) tuple
        or a 1D ndarray of posterior samples.

    logg : tuple or 1D ndarray
        Log g either as a (value, uncertainty) tuple or a 1D ndarray of posterior
        samples.

    metal : tuple or 1D ndarray
        Stellar metallicity (z) either as a  (value, uncertainty) tuple or a
        1D ndarray of posterior samples.

    filters : list of Filter instances
        List of filters defining the passbands for which to calculate the
        stellar intensity profiles.

    offline_mode : bool, optional
        If True, skips any attempts to connect to the FTP server, and uses only cached
        files.

    force_download: bool, optional
        Download all the files from the FTP server, including the ones already in cache.

    verbose : bool

    cache : str, optional
        Path to the cache directory.

    photon_counting: bool, optional
        If true, calculate photon-weighted averages (e.g., for a CCD), otherwise calculate energy-weighted averages.

    dataset: str, optional
        Set of stellar spectrum models to use. Options are "vis", "vis-lowres", "visir", "visir-lowres"

    interpolation: str, optional
        Method used to evaluate the limb darkening profiles inside the model
        grid, either 'linear' (default) or 'rbf'. The default interpolates
        the grid profiles piecewise-linearly. The 'rbf' option uses a
        smooth radial basis function interpolant
        (`ldtk.rbf.RBFProfileInterpolator`) that captures the nonlinearity of
        the simulated profiles between the grid nodes and extrapolates
        smoothly instead of returning NaN outside the convex hull of the
        available nodes.
    """

    def __init__(self, teff, logg, z, filters: Optional[List] = None,
                 qe=None, limits=None, offline_mode: bool = False,
                 force_download: bool = False, verbose: bool = False, cache: Optional[Union[str, Path]] = None,
                 photon_counting: bool = True, lowres: bool = False, dataset: str = 'vis-lowres',
                 save_memory: bool = True, interpolation: str = 'linear'):

        self.teff = teff
        self.logg = logg
        self.metal = z
        self.photon_counting = photon_counting
        self.save_memory = save_memory

        if interpolation not in ('linear', 'rbf'):
            raise ValueError(f"Unknown interpolation method '{interpolation}', should be either 'linear' or 'rbf'.")
        self.interpolation = interpolation

        if lowres:
            raise DeprecationWarning('lowres option is deprecated in LDTk 1.5, please use dataset="vis-lowres" instead.')

        def set_lims(ms_or_samples, pts, plims=(0.135, 100 - 0.135)):
            if len(ms_or_samples) > 2:
                return a_lims_hilo(pts, *percentile(ms_or_samples, plims))
            else:
                return a_lims(pts, *ms_or_samples)

        if not limits:
            teff_lims = set_lims(teff, TEFF_POINTS)
            logg_lims = set_lims(logg, LOGG_POINTS)
            metal_lims = set_lims(z, Z_POINTS)
        else:
            teff_lims, logg_lims, metal_lims = limits

        if verbose:
            print("Teff limits: " + str(teff_lims))
            print("logg limits: " + str(logg_lims))
            print("Fe/H limits: " + str(metal_lims))

        self.client = Client(limits=[teff_lims, logg_lims, metal_lims], cache=cache, lowres=lowres, dataset=dataset)
        self.files = self.client.local_filenames
        self.nfiles = len(self.files)
        self.qe = qe or (lambda wl: 1.)

        self.filters: Optional[list] = None
        self.nfilters: Optional[int] = None
        self.fluxes: Optional[ndarray] = None
        self.raw_spectra: Optional[list] = None

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(15), retry=retry_if_exception_type(Exception))
        def download_files():
            if self.client.download_uncached_files(force=force_download):
                self.client.__init__(limits=[teff_lims, logg_lims, metal_lims], cache=cache, lowres=lowres)
                raise Exception

        if is_root and not offline_mode:
            download_files()
        if with_mpi:
            comm.Barrier()

        # Initialize the basic arrays
        # ---------------------------
        with pf.open(self.files[0]) as hdul:
            wl0 = hdul[0].header['crval1'] * 1e-1  # Wavelength at d[:,0] [nm]
            dwl = hdul[0].header['cdelt1'] * 1e-1  # Delta wavelength     [nm]
            nwl = hdul[0].header['naxis1']  # Number of wl samples
            self.wl = wl0 + arange(nwl) * dwl
            self.mu = hdul[1].data
            self.z = sqrt(1 - self.mu ** 2)
            self.nmu = self.mu.size

        if not self.save_memory:
            self.raw_spectra = []
            for did, df in enumerate(self.files):
                self.raw_spectra.append(pf.getdata(df))

        if filters is not None:
            self.init_filters(filters)

    def init_filters(self, filters: List):
        self.filters = filters
        self.nfilters = len(filters)

        # Read in the fluxes
        # ------------------
        self.fluxes = zeros([self.nfilters, self.nfiles, self.nmu])
        for did, df in enumerate(self.files):
            if self.save_memory:
                d =  pf.getdata(df)
            else:
                d = self.raw_spectra[did]
            for fid, f in enumerate(self.filters):
                if self.photon_counting:
                    w = self.wl * self.qe(self.wl)
                else:
                    w = self.qe(self.wl)

                self.fluxes[fid, did, :] = f.integrate(self.wl, d*w)
                self.fluxes[fid, did, :] /= self.fluxes[fid, did, -1]

        # Create n_filter interpolators
        # -----------------------------
        points = array([[f.teff, f.logg, f.z] for f in self.client.files])
        if self.interpolation == 'rbf':
            self.itps = [RBFProfileInterpolator(points, self.fluxes[i, :, :]) for i in range(self.nfilters)]
        else:
            self.itps = [NDI(points, self.fluxes[i, :, :]) for i in range(self.nfilters)]

    def create_profiles(self, nsamples=100, teff=None, logg=None, metal=None):
        """Creates a set of limb darkening profiles

           Parameters
           ----------
           nsamples : int number of limb darkening profiles
           teff  : array_like [optional]
           logg  : array_like [optional]
           metal : array_like [optional]

           Notes
           -----
           Teff, logg, and z are by default read in from the previously-created
           object. However, alternative posterior distributions can be passed in via
           (teff_in, logg_in, metal_in).
        """

        def sample(a, b):
            return a if a is not None else (b if len(b) != 2 else normal(*b, size=nsamples))

        if self.filters is None:
            raise ValueError("Can't create LD profiles without filters")

        teff = sample(teff, self.teff)
        logg = sample(logg, self.logg)
        metal = sample(metal, self.metal)

        minsize = min(nsamples, min(map(len, [teff, logg, metal])))
        samples = ones([minsize, 3])
        samples[:, 0] = clip(teff, *self.client.teffl)[:minsize]
        samples[:, 1] = clip(logg, *self.client.loggl)[:minsize]
        samples[:, 2] = clip(metal, *self.client.zl)[:minsize]

        self.ldp_samples = zeros([self.nfilters, minsize, self.nmu])
        for iflt in range(self.nfilters):
            self.ldp_samples[iflt, :, :] = self.itps[iflt](samples)

        # Drop samples that fall outside the available PHOENIX model grid
        # ---------------------------------------------------------------
        # The (teff, logg, z) samples are clipped to the rectangular grid limits, but
        # the available models can be missing some grid nodes (e.g. hot, low-gravity
        # stars are not in the library). The LinearNDInterpolator returns NaN for any
        # sample falling outside the convex hull of the available models. A single such
        # sample would poison the mean profile and the limb fit, turning the whole
        # LDPSet into NaNs, so we drop the offending samples here.
        finite = isfinite(self.ldp_samples).all((0, 2))
        n_bad = int((~finite).sum())
        if finite.sum() == 0:
            raise ValueError(
                "All parameter samples fall outside the available PHOENIX model grid. The requested "
                "(teff, logg, z) lies in a region where models are missing (e.g. hot, low-gravity "
                "stars). Adjust the parameters or their uncertainties.")
        if n_bad > 0:
            message(f"Dropped {n_bad} of {minsize} samples that fell outside the available model grid.")
            self.ldp_samples = self.ldp_samples[:, finite, :]

        return LDPSet(self.filter_names, self.mu, self.ldp_samples)

    @property
    def filter_names(self):
        return [f.name for f in self.filters]
