"""
Limb darkening toolkit
Copyright (C) 2015  Hannu Parviainen <hpparvi@gmail.com>

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
from numpy import argmin, zeros, sqrt, array, diff, log, linspace, ones, diag, exp, cov, asarray, percentile, arange, clip
from numpy.random import normal, multivariate_normal, uniform
from scipy.interpolate import interp1d, LinearNDInterpolator as NDI
from scipy.optimize import fmin
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type

from .client import Client
from .core import TWO_PI, dx2, a_lims_hilo, a_lims, TEFF_POINTS, LOGG_POINTS, Z_POINTS, is_root, with_mpi, comm
from .ldmodel import (LinearModel, QuadraticModel, TriangularQuadraticModel, SquareRootModel, NonlinearModel,
                      GeneralModel, Power2Model, Power2MPModel, models)


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

    """

    def __init__(self, filters, mu, ldp_samples):
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

        self._limb_i = abs(diff(self._mean_orig.mean(0))).argmax()
        self._limb_z = self._z_orig[self._limb_i]
        self._limb_mu = sqrt(1. - self._z_orig[self._limb_i] ** 2)
        self.redefine_limb()

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
        self._limb_mu = mu
        self._limb_i = argmin(abs(self._mu_orig - mu))
        self._limb_z = sqrt(1. - mu ** 2)
        self.reset_sampling()

    def redefine_limb(self):
        self._z = self._z_orig[self._limb_i:] / self._limb_z
        self._mu = sqrt(1. - self._z ** 2)
        self._ldps = self._ldps_orig[:, :, self._limb_i:].copy()
        self._mean = self._mean_orig[:, self._limb_i:].copy()
        self._std = self._std_orig[:, self._limb_i:].copy()

    def set_uncertainty_multiplier(self, em):
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

        self._mean = array([interp1d(muc, f, kind='cubic')(self._mu) for f in self._mean])
        self._std = array([interp1d(muc, f, kind='cubic')(self._mu) for f in self._std])
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
        qcs = [fmin(lambda pv: -self._lnlike(pv, flt=iflt, ldmodel=ldmodel), 0.1 * ones(npar), disp=0) for iflt in range(self._nfilters)]
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

        if flt is not None:
            return lnlike1d(m, flt, self._lnc1, self._lnc2, self._mean, self._err2)
        elif ldcs.ndim == 2:
            return lnlike2d(m, self._lnc1, self._lnc2, self._mean, self._err2)
        elif ldcs.ndim == 3:
            return lnlike3d(m, self._lnc1, self._lnc2, self._mean, self._err2)

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

    lowres: bool, optional
        If true, use model spectra binned to 5 nm, otherwise use the original files.

    """

    def __init__(self, teff, logg, z, filters: List,
                 qe=None, limits=None, offline_mode: bool = False,
                 force_download: bool = False, verbose: bool = False, cache: Optional[Union[str, Path]] = None,
                 photon_counting: bool = True, lowres: bool = True):

        self.teff = teff
        self.logg = logg
        self.metal = z

        self.use_lowres = lowres

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

        self.client = Client(limits=[teff_lims, logg_lims, metal_lims], cache=cache, lowres=lowres)
        self.files = self.client.local_filenames
        self.filters = filters
        self.nfiles = len(self.files)
        self.nfilters = len(filters)
        self.qe = qe or (lambda wl: 1.)

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
            wl = wl0 + arange(nwl) * dwl
            self.mu = hdul[1].data
            self.z = sqrt(1 - self.mu ** 2)
            self.nmu = self.mu.size

        # Read in the fluxes
        # ------------------
        self.fluxes = zeros([self.nfilters, self.nfiles, self.nmu])
        for fid, f in enumerate(self.filters):
            if photon_counting:
                w = f(wl) * wl * self.qe(wl)
            else:
                w = f(wl) * self.qe(wl)

            for did, df in enumerate(self.files):
                self.fluxes[fid, did, :] = (pf.getdata(df) * w).mean(1)
                self.fluxes[fid, did, :] /= self.fluxes[fid, did, -1]

        # Create n_filter interpolators
        # -----------------------------
        points = array([[f.teff, f.logg, f.z] for f in self.client.files])
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

        return LDPSet(self.filter_names, self.mu, self.ldp_samples)

    @property
    def filter_names(self):
        return [f.name for f in self.filters]
