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
from scipy.interpolate import LinearNDInterpolator as NDI
from scipy.interpolate import interp1d
from scipy.optimize import fmin

from .ldmodel import LinearModel, QuadraticModel, SquareRootModel, NonlinearModel, GeneralModel, models
from .client import Client
from .core import *

def load_ldpset(filename):
    with open(filename,'r') as fin:
        return LDPSet(load(fin), load(fin), load(fin))

##TODO: Give an option to use Kernel density estimation if given a set of teff,logg,z samples.

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

        self._limb_i   = abs(diff(self._mean_orig.mean(0))).argmax()
        self._limb_z   = self._z_orig[self._limb_i]
        self._limb_mu  = sqrt(1.-self._z_orig[self._limb_i]**2)
        self.redefine_limb()

        self._lnl     = zeros(self._nfilters)
        self.set_uncertainty_multiplier(1.)
        self._update()

        self.lnlike_ln = partial(self._lnlike, ldmodel=LinearModel)
        self.lnlike_qd = partial(self._lnlike, ldmodel=QuadraticModel)
        self.lnlike_sq = partial(self._lnlike, ldmodel=SquareRootModel)
        self.lnlike_nl = partial(self._lnlike, ldmodel=NonlinearModel)
        self.lnlike_ge = partial(self._lnlike, ldmodel=GeneralModel)

        self.coeffs_ln = partial(self._coeffs, ldmodel=LinearModel)
        self.coeffs_qd = partial(self._coeffs, ldmodel=QuadraticModel)
        self.coeffs_sq = partial(self._coeffs, ldmodel=SquareRootModel)
        self.coeffs_nl = partial(self._coeffs, ldmodel=NonlinearModel)
        self.coeffs_ge = partial(self._coeffs, ldmodel=GeneralModel)

        self.lnlike_ln.__doc__ = "Linear limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_qd.__doc__ = "Quadratic limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_sq.__doc__ = "Square root limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_nl.__doc__ = "Nonlinear limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_ge.__doc__ = "General limb darkening model\n(coeffs, join=True, flt=None)"

        self.coeffs_ln.__doc__ = "Estimate the linear limb darkening model coefficients, see LPDSet._coeffs for details." 
        self.coeffs_qd.__doc__ = "Estimate the quadratic limb darkening model coefficients, see LPDSet._coeffs for details."
        self.coeffs_sq.__doc__ = "Estimate the square root limb darkening model coefficients, see LPDSet._coeffs for details."
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
    def __init__(self, teff, logg, z, filters,
                 qe=None, limits=None, offline_mode=False,
                 force_download=False, verbose=False, cache=None): 
        """Creates a limb darkening profile set (LDPSet).

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
        """
        self.teff  = teff
        self.logg  = logg
        self.metal = z

        def set_lims(ms_or_samples, pts, plims=[0.135,100-0.135] ):
            if len(ms_or_samples) > 2:
                return a_lims_hilo(pts, *percentile(ms_or_samples, plims))
            else:
                return a_lims(pts, *ms_or_samples)
        
        if not limits:
            teff_lims  = set_lims(teff, TEFF_POINTS)
            logg_lims  = set_lims(logg, LOGG_POINTS)
            metal_lims = set_lims(z,    Z_POINTS)
        else:
            teff_lims, logg_lims, metal_lims = limits

        if verbose:
            print("Teff limits: " + str(teff_lims))
            print("logg limits: " + str(logg_lims))
            print("Fe/H limits: " + str(metal_lims))

        self.client   = c = Client(limits=[teff_lims, logg_lims, metal_lims], cache=cache)
        self.files    = self.client.local_filenames
        self.filters  = filters
        self.nfiles   = len(self.files)
        self.nfilters = len(filters)
        self.qe       = qe or (lambda wl: 1.)

        if is_root and not offline_mode:
            self.client.download_uncached_files(force=force_download)
        if with_mpi:
            comm.Barrier()
                
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

        def sample(a,b):
            return a if a is not None else (b if len(b)!=2 else normal(*b, size=nsamples))

        teff  = sample(teff,  self.teff)
        logg  = sample(logg,  self.logg)
        metal = sample(metal, self.metal)
    
        minsize = min(nsamples, min(map(len, [teff, logg, metal])))
        samples = ones([minsize,3])
        samples[:,0] = clip(teff,  *self.client.teffl)[:minsize]
        samples[:,1] = clip(logg,  *self.client.loggl)[:minsize]
        samples[:,2] = clip(metal, *self.client.zl)[:minsize]
        
        self.ldp_samples = zeros([self.nfilters, minsize, self.nmu])
        for iflt in range(self.nfilters):
            self.ldp_samples[iflt,:,:] = self.itps[iflt](samples)

        return LDPSet(self.filter_names, self.mu, self.ldp_samples)

    @property
    def filter_names(self):
        return [f.name for f in self.filters]

