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

import pyfits as pf
from functools import partial
from scipy.interpolate import LinearNDInterpolator as NDI
from scipy.interpolate import interp1d
from scipy.optimize import fmin

from ld_models import LinearModel, QuadraticModel, NonlinearModel, GeneralModel, models
from client import Client
from core import *


def load_ldpset(filename):
    with open(filename,'r') as fin:
        return LDPSet(load(fin), load(fin), load(fin))


## Main classes
## ============        
class LDPSet(object):
    def __init__(self, filters, mu, ldp_samples):
        self._filters  = filters 
        self._nfilters = len(filters)
        self._mu       = mu
        self._z        = sqrt(1-mu**2)
        self._ldps     = ldp_samples
        self._mean     = array([ldp_samples[i,:,:].mean(0) for i in range(self._nfilters)])
        self._std      = array([ldp_samples[i,:,:].std(0)  for i in range(self._nfilters)])
        self._samples  = {m.abbr:[] for m in models.values()}

        self._ldps_orig = self._ldps.copy()
        self._mu_orig   = self._mu.copy()
        self._z_orig    = self._z.copy()
        self._mean_orig = self._mean.copy()
        self._std_orig  = self._std.copy()

        self._limb_i   = abs(diff(self._mean.mean(0))).argmax()
        self._limb_z   = self._z[i]
        self._limb_mu  = sqrt(1.-self._z[i]**2)
        self.redefine_limb()

        self._lnl     = zeros(self._nfilters)
        self.set_uncertainty_multiplier(1.)
        self._update()

        self.lnlike_ln = partial(self._lnlike, ldmodel=LinearModel)
        self.lnlike_qd = partial(self._lnlike, ldmodel=QuadraticModel)
        self.lnlike_nl = partial(self._lnlike, ldmodel=NonlinearModel)
        self.lnlike_ge = partial(self._lnlike, ldmodel=GeneralModel)

        self.coeffs_ln = partial(self._coeffs, ldmodel=LinearModel)
        self.coeffs_qd = partial(self._coeffs, ldmodel=QuadraticModel)
        self.coeffs_nl = partial(self._coeffs, ldmodel=NonlinearModel)
        self.coeffs_ge = partial(self._coeffs, ldmodel=GeneralModel)

        self.lnlike_ln.__doc__ = "Linear limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_qd.__doc__ = "Quadratic limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_nl.__doc__ = "Nonlinear limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_ge.__doc__ = "General limb darkening model\n(coeffs, join=True, flt=None)"

        self.coeffs_ln.__doc__ = "Estimate the linear limb darkening model coefficients, see LPDSet._coeffs for details." 
        self.coeffs_qd.__doc__ = "Estimate the quadratic limb darkening model coefficients, see LPDSet._coeffs for details."
        self.coeffs_nl.__doc__ = "Estimate the nonlinear limb darkening model coefficients, see LPDSet._coeffs for details."
        self.coeffs_ge.__doc__ = "Estimate the general limb darkening model coefficients, see LPDSet._coeffs for details."

        
    def save(self, filename):
        with open(filename,'w') as f:
            dump(self._filters, f)
            dump(self._mu_orig, f)
            dump(self._ldps_orig, f)


    def _update(self):
        self._nmu      = self._mu.size
        self._lnc1    = -0.5*self._nmu*log(TWO_PI)                    ## 1st ln likelihood term
        self._lnc2    = [-log(self._em*e).sum() for e in self._std]   ## 2nd ln likelihood term
        self._err2    = [(self._em*e)**2 for e in self._std]          ## variances


    def set_limb_z(self, z):
        self._limb_z = z
        self._limb_i = argmin(abs(self._z_orig-z))
        self._limb_mu = sqrt(1.-z**2)
        self.reset_sampling()


    def set_limb_mu(self, mu):
        self._limb_mu = mu
        self._limb_i  = argmin(abs(self._mu_orig-mu))
        self._limb_z = sqrt(1.-mu**2) 
        self.reset_sampling()


    def redefine_limb(self):
        self._z  = self._z_orig[self._limb_i:] / self._limb_z
        self._mu = sqrt(1.-self._z**2) 
        self._ldps = self._ldps_orig[:,:,self._limb_i:].copy()
        self._mean = self._mean_orig[:,self._limb_i:].copy()
        self._std  = self._std_orig[:,self._limb_i:].copy()


    def set_uncertainty_multiplier(self, em):
        self._em      = em
        self._update()
        

    def reset_sampling(self):
        self.redefine_limb()
        self._update()


    def resample_linear_z(self, nz=100):
        self.resample(z=linspace(0,1,nz))
 

    def resample_linear_mu(self, nmu=100):
        self.resample(mu=linspace(0,1,nmu))
     

    def resample(self, mu=None, z=None):
        muc = self._mu.copy()
        if z is not None:
            self._z  = z
            self._mu = sqrt(1-self._z**2) 
        elif mu is not None:
            self._mu = mu
            self._z  = sqrt(1-self._mu**2) 

        self._mean = array([interp1d(muc, f, kind='cubic')(self._mu) for f in self._mean])
        self._std  = array([interp1d(muc, f, kind='cubic')(self._mu) for f in self._std])
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
        qcs  = [fmin(lambda pv:-self._lnlike(pv, flt=iflt, ldmodel=ldmodel), 0.1*ones(npar), disp=0) for iflt in range(self._nfilters)]
        covs = []
        for iflt, qc in enumerate(qcs):
            s = zeros(npar)
            for ic in range(npar):
                s[ic] = (1./sqrt(-dx2(lambda x:self._lnlike(x, flt=iflt, ldmodel=ldmodel), qc, 1e-5, dim=ic)))

            ## Simple MCMC uncertainty estimation
            ## ----------------------------------
            if do_mc:
                logl  = zeros(n_mc_samples)
                chain = zeros([n_mc_samples,npar])
                
                chain[0,:] = qc
                logl[0]    = self._lnlike(chain[0], flt=iflt, ldmodel=ldmodel)

                for i in xrange(1,n_mc_samples):
                    pos_t  = multivariate_normal(chain[i-1], diag(s**2))
                    logl_t = self._lnlike(pos_t, flt=iflt, ldmodel=ldmodel)
                    if uniform() < exp(logl_t-logl[i-1]):
                        chain[i,:] = pos_t
                        logl[i]    = logl_t
                    else:
                        chain[i,:] = chain[i-1,:]
                        logl[i]    = logl[i-1]
                self._samples[ldmodel.abbr].append(chain)
                ch = chain[mc_burn::mc_thin,:]

                if return_cm:
                    covs.append(cov(ch, rowvar=0))
                else:
                    covs.append(sqrt(cov(ch, rowvar=0)) if npar == 1 else sqrt(cov(ch, rowvar=0).diagonal()))

            else:
                if return_cm:
                    covs.append(s**2 if npar == 1 else diag(s**2))
                else:
                    covs.append(s)

        return array(qcs), array(covs)

            
    def _lnlike(self, ldcs, joint=True, flt=None, ldmodel=QuadraticModel):
        if flt is None:
            for fid, ldc in enumerate(asarray(ldcs).reshape([self._nfilters,-1])):
                model = ldmodel.evaluate(self._mu, ldc)
                self._lnl[fid] = self._lnc1 + self._lnc2[fid] -0.5*((self._mean[fid]-model)**2/self._err2[fid]).sum()
            return self._lnl.sum() if joint else self._lnl
        else:
            model = ldmodel.evaluate(self._mu, asarray(ldcs))
            self._lnl[flt] = self._lnc1 + self._lnc2[flt] -0.5*((self._mean[flt]-model)**2/self._err2[flt]).sum()
            return self._lnl[flt]


    @property
    def profile_averages(self):
        return self._mean


    @property
    def profile_uncertainties(self):
        return self._std


class LDPSetCreator(object):
    def __init__(self, teff, logg, z, filters, qe=None, limits=None, force_download=False):
        self.teff  = teff
        self.logg  = logg
        self.metal = z

        if not limits:
            teff_lims  = a_lims(TEFF_POINTS, *teff)
            logg_lims  = a_lims(LOGG_POINTS, *logg)
            metal_lims = a_lims(Z_POINTS, *z)
        else:
            teff_lims, logg_lims, metal_lims = lims

        self.client   = c = Client(limits=[teff_lims, logg_lims, metal_lims])
        self.files    = self.client.local_filenames
        self.filters  = filters
        self.nfiles   = len(self.files)
        self.nfilters = len(filters)
        self.qe       = qe or (lambda wl: 1.)

        self.client.download_uncached_files(force=force_download)

        ## Initialize the basic arrays
        ## ---------------------------
        with pf.open(self.files[0]) as hdul:
            wl0  = hdul[0].header['crval1'] * 1e-1 # Wavelength at d[:,0] [nm]
            dwl  = hdul[0].header['cdelt1'] * 1e-1 # Delta wavelength     [nm]
            nwl  = hdul[0].header['naxis1']        # Number of wl samples
            wl   = wl0 + arange(nwl)*dwl
            self.mu   = hdul[1].data
            self.z    = sqrt(1-self.mu**2)
            self.nmu  = self.mu.size
        
        ## Read in the fluxes
        ## ------------------
        self.fluxes   = zeros([self.nfilters, self.nfiles, self.nmu])
        for fid,f in enumerate(self.filters):
            w = f(wl) * self.qe(wl)
            for did,df in enumerate(self.files):
                self.fluxes[fid,did,:]  = (pf.getdata(df)*w).mean(1)
                self.fluxes[fid,did,:] /= self.fluxes[fid,did,-1]

        ## Create n_filter interpolators
        ## -----------------------------
        points = array([[f.teff,f.logg,f.z] for f in self.client.files])
        self.itps = [NDI(points, self.fluxes[i,:,:]) for i in range(self.nfilters)]
         
        
    def create_profiles(self, nsamples=20, mode=0, nmu=100):
        self.ldp_samples = zeros([self.nfilters, nsamples, self.nmu])
        samples = ones([nsamples,3])
        samples[:,0] = clip(normal(*self.teff,  size=nsamples), *self.client.teffl)
        samples[:,1] = clip(normal(*self.logg,  size=nsamples), *self.client.loggl)
        samples[:,2] = clip(normal(*self.metal, size=nsamples), *self.client.zl)
                
        for iflt in range(self.nfilters):
            self.ldp_samples[iflt,:,:] = self.itps[iflt](samples)

        return LDPSet(self.filter_names, self.mu, self.ldp_samples)

    @property
    def filter_names(self):
        return [f.name for f in self.filters]
