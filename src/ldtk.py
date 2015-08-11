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

import os
from os.path import join, exists
import warnings
import pyfits as pf

from glob import glob
from ftplib import FTP
from itertools import product
from functools import partial
from os.path import exists, join, basename
from cPickle import dump, load
from numpy import (array, asarray, arange, linspace, zeros, zeros_like, ones, ones_like, delete,
                   diag, poly1d, polyfit, vstack, diff, cov, exp, log, sqrt, clip, pi)
from numpy.random import normal, uniform, multivariate_normal
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import LinearNDInterpolator as NDI
from scipy.interpolate import interp1d
from scipy.optimize import fmin
from itertools import product

from ld_models import LinearModel, QuadraticModel, NonlinearModel, GeneralModel, models

## Test if we're running inside IPython
## ------------------------------------
try:
    __IPYTHON__
    from IPython.display import display, HTML
    with_ipython = True
except NameError:
    with_ipython = False

warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    try:
        from IPython.display import display, clear_output
        from IPython.html.widgets import IntProgress
        w = IntProgress()
        with_notebook = True
    except AttributeError:
        with_notebook = False

ldtk_root  = os.getenv('LDTK_ROOT') or join(os.getenv('HOME'),'.ldtk')
ldtk_cache = join(ldtk_root,'cache')
ldtk_server_file_list = join(ldtk_root, 'server_file_list.pkl')

if not exists(ldtk_root):
    os.mkdir(ldtk_root)
if not exists(ldtk_cache):
    os.mkdir(ldtk_cache)
    
## Constants
## =========

TWO_PI      = 2*pi
TEFF_POINTS = delete(arange(2300,12001,100), [27])
LOGG_POINTS = arange(0,6.1,0.5)
Z_POINTS    = array([-4.0, -3.0, -2.0, -1.5, -1.0, -0.0, 0.5, 1.0])

## Utility functions
## =================

def message(text):
    if with_ipython:
        display(HTML(text))
    else:
        print text

class ProgressBar(object):
    def __init__(self, max_v):
        if with_notebook:
            self.pb = IntProgress(value=0, max=max_v)
            display(self.pb)

    def increase(self, v=1):
        if with_notebook:
            self.pb.value += v


def dxdx(f, x, h):
    return (f(x+h) - 2*f(x) + f(x-h)) / h**2

def dx2(f, x0, h, dim):
    xp,xm = array(x0), array(x0)
    xp[dim] += h
    xm[dim] -= h
    return (f(xp) - 2*f(x0) + f(xm)) / h**2

def dxdy(f,x,y,h=1e-5):
    return ((f([x+h,y+h])-f([x+h,y-h]))-(f([x-h,y+h])-f([x-h,y-h])))/(4*h**2)

def v_from_poly(lf, x0, dx=1e-3, nx=5):
    """Estimates the variance of a log distribution approximating it as a normal distribution."""
    xs  = linspace(x0-dx, x0+dx, nx)
    fs  = array([lf(x) for x in xs])
    p   = poly1d(polyfit(xs, fs, 2))
    x0  = p.deriv(1).r
    var = -1./p.deriv(2)(x0)
    return var

def is_inside(a,lims):
    """Is a value inside the limits"""
    return a[(a>=lims[0])&(a<=lims[1])]


def a_lims(a,v,e,s=3):
    return a[[max(0,a.searchsorted(v-s*e)-1),min(a.size-1, a.searchsorted(v+s*e))]]


## Main classes
## ============

class SpecIntFile(object):
    def __init__(self, teff, logg, z):
        tmpl = 'lte{teff:05d}-{logg:4.2f}{z:+3.1f}.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'
        self.teff = int(teff)
        self.logg = logg
        self.z    = z
        self.name  = tmpl.format(teff=self.teff, logg=self.logg, z=self.z)
        self._zstr = 'Z'+self.name[13:17]
        
    @property
    def local_path(self):
        return join(ldtk_cache,self._zstr,self.name)

    @property
    def local_exists(self):
        return exists(self.local_path)
    
    
class Client(object):
    def __init__(self, limits=None, verbosity=1, update_server_file_list=False):
        self.fnt = 'lte{teff:05d}-{logg:4.2f}{z:+3.1f}.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'
        self.eftp = 'phoenix.astro.physik.uni-goettingen.de'
        self.edir = 'SpecIntFITS/PHOENIX-ACES-AGSS-COND-SPECINT-2011'
        self.files = None
        self.verbosity = verbosity

        if exists(ldtk_server_file_list) and not update_server_file_list:
            with open(ldtk_server_file_list, 'r') as fin:
                self.files_in_server = load(fin)
        else:
            self.files_in_server = self.get_server_file_list()
            with open(ldtk_server_file_list, 'w') as fout:
                dump(self.files_in_server, fout)

        if limits:
            self.set_limits(*limits)


    def _local_path(self, teff_or_fn, logg=None, z=None):
        """Creates the path to the local version of the file."""
        fn = teff_or_fn if isinstance(teff_or_fn, str) else self.create_name(teff_or_fn,logg,z)
        return join(ldtk_cache,'Z'+fn[13:17],fn)
        

    def _local_exists(self, teff_or_fn, logg=None, z=None):
        """Tests if a file exists in the local cache. """
        return exists(self._local_path(teff_or_fn, logg, z))
        

    def create_name(self, teff, logg, z):
        """Creates a SPECINT filename given teff, logg, and z."""
        return self.fnt.format(teff=int(teff), logg=logg, z=z)


    def set_limits(self, teff_lims, logg_lims, z_lims):
        self.teffl = teff_lims
        self.teffs = is_inside(TEFF_POINTS, teff_lims)
        self.nteff = len(self.teffs)
        self.loggl = logg_lims
        self.loggs = is_inside(LOGG_POINTS, logg_lims)
        self.nlogg = len(self.loggs)
        self.zl    = z_lims
        self.zs    = is_inside(Z_POINTS, z_lims)
        self.nz    = len(self.zs)
        self.pars  = [p for p in product(self.teffs,self.loggs,self.zs)]
        self.files = [SpecIntFile(*p) for p in product(self.teffs,self.loggs,self.zs)]
        self.clean_file_list()

        self.not_cached =  len(self.files) - sum([f.local_exists for f in self.files])
        if self.not_cached > 0:
            message("Need to download {:d} files, approximately {} MB".format(self.not_cached, 16*self.not_cached))
    

    def get_server_file_list(self):
        ftp = FTP(self.eftp)
        ftp.login()
        ftp.cwd(self.edir)
        files_in_server = {}
        zdirs = sorted(ftp.nlst())
        for zdir in zdirs:
            ftp.cwd(zdir)
            files_in_server[zdir] = sorted(ftp.nlst())
            ftp.cwd('..')
        ftp.close()
        return files_in_server


    def files_exist(self, files=None):
        """Tests if a file exists in the FTP server."""
        return [f.name in self.files_in_server[f._zstr] for f in self.files]
            
    
    def clean_file_list(self):
        """Removes files not in the FTP server."""
        self.files = [f for f,e in zip(self.files,self.files_exist()) if e]


    def download_uncached_files(self, force=False):
        """Downloads the uncached files to a local cache."""
        ftp = FTP(self.eftp)
        ftp.login()
        ftp.cwd(self.edir)

        if self.not_cached > 0 or force:
            pbar = ProgressBar(self.not_cached if not force else len(self.files))
        
        for fid,f in enumerate(self.files):
            if not exists(join(ldtk_cache,f._zstr)):
                os.mkdir(join(ldtk_cache,f._zstr))
            if not f.local_exists or force:
                ftp.cwd(f._zstr)
                localfile = open(f.local_path, 'wb')
                ftp.retrbinary('RETR '+f.name, localfile.write)
                localfile.close()
                ftp.cwd('..')
                self.not_cached -= 1
                pbar.increase()
            else:
                if self.verbosity > 1:
                    print 'Skipping an existing file: ', f.name
        ftp.close()
 

    @property
    def local_filenames(self):
        return [f.local_path for f in self.files]
        
        
class LDPSet(object):
    def __init__(self, filters, mu, ldp, ldp_s):
        self._filters  = filters 
        self._nfilters = len(filters)
        self._mu       = mu
        self._mu_orig  = mu.copy()
        self._z        = sqrt(1-mu**2)
        self._z_orig   = self._z.copy()
        self._mean     = ldp
        self._std      = ldp_s
        self._mean_orig= ldp.copy()
        self._std_orig = ldp_s.copy()
        self._samples  = {m.abbr:[] for m in models.values()}

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


    def _update(self):
        self._nmu      = self._mu.size
        self._lnc1    = -0.5*self._nmu*log(TWO_PI)                   ## 1st ln likelihood term
        self._lnc2    = [-log(self._em*e).sum() for e in self._std]   ## 2nd ln likelihood term
        self._err2    = [(self._em*e)**2 for e in self._std]          ## variances


    def set_uncertainty_multiplier(self, em):
        self._em      = em                                      ## uncertainty multiplier
        self._update()
        

    def reset_sampling(self):
        self._mu   = self._mu_orig.copy()
        self._z    = self._z_orig.copy()
        self._mean = self._mean_orig.copy()
        self._std  = self._std_orig.copy()
        self._update()


    def resample_linear_z(self, nz=100):
        self.resample(z=linspace(0,1,nz))
 

    def resample_linear_mu(self, nmu=100):
        self.resample(mu=linspace(0,1,nmu))
     

    def resample(self, mu=None, z=None):
        if z is not None:
            self._z  = z
            self._mu = sqrt(1-self._z**2) 
        elif mu is not None:
            self._mu = mu
            self._z  = sqrt(1-self._mu**2) 

        self._mean = array([interp1d(self._mu_orig, f, kind='cubic')(self._mu) for f in self._mean_orig])
        self._std  = array([interp1d(self._mu_orig, f, kind='cubic')(self._mu) for f in self._std_orig])
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
        self.vals = zeros([self.nfilters, nsamples, self.nmu])
        samples = ones([nsamples,3])
        samples[:,0] = clip(normal(*self.teff,  size=nsamples), *self.client.teffl)
        samples[:,1] = clip(normal(*self.logg,  size=nsamples), *self.client.loggl)
        samples[:,2] = clip(normal(*self.metal, size=nsamples), *self.client.zl)
                
        for iflt in range(self.nfilters):
            self.vals[iflt,:,:] = self.itps[iflt](samples)

        ldp_m = array([self.vals[i,:,:].mean(0) for i in range(self.nfilters)])
        ldp_s = array([self.vals[i,:,:].std(0)  for i in range(self.nfilters)])

        ## Clip the arrays and renormalise the z and mu ranges
        ##
        i = diff(ldp_m.mean(0)).argmax()
        z  = self.z[i:] / self.z[i]
        mu = sqrt(1-z**2) 

        ldp_m = ldp_m[:,i:].copy()
        ldp_s = ldp_s[:,i:].copy()

        return LDPSet(self.filter_names, mu, ldp_m, ldp_s)

    @property
    def filter_names(self):
        return [f.name for f in self.filters]



## Utility classes
## ===============

class SIS(object):
    """Simple wrapper for a specific intensity spectrum file."""
    def __init__(self, fname):
        self.filename = fname
        with pf.open(fname) as hdul:
            self.wl0  = hdul[0].header['crval1'] * 1e-1 # Wavelength at d[:,0] [nm]
            self.dwl  = hdul[0].header['cdelt1'] * 1e-1 # Delta wavelength     [nm]
            self.nwl  = hdul[0].header['naxis1']        # Number of samples
            self.data = hdul[0].data
            self.mu   = hdul[1].data
            self.z    = sqrt(1-self.mu**2)
            self.wl   = self.wl0 + arange(self.nwl)*self.dwl
                
    def intensity_profile(self, l0=0, l1=1e5):
        ip = self.data[:,(self.wl>l0)&(self.wl<l1)].mean(1)
        return ip/ip[-1]
    
    
class IntegratedIP(object):
    def __init__(self, dfile, l0, l1):
        with pf.open(dfile) as hdul:
            wl0  = hdul[0].header['crval1'] * 1e-1 # Wavelength at d[:,0] [nm]
            dwl  = hdul[0].header['cdelt1'] * 1e-1 # Delta wavelength     [nm]
            nwl  = hdul[0].header['naxis1']        # Number of wl samples
            wl   = wl0 + arange(nwl)*dwl
            msk  = (wl > l0) & (wl < l1)
            
            self.flux  = hdul[0].data[:,msk].mean(1)
            self.flux /= self.flux[-1]
            self.mu   = hdul[1].data
            self.z    = sqrt(1-self.mu**2)
