import os
import pyfits as pf

from glob import glob
from ftplib import FTP
from itertools import product
from functools import partial
from os.path import exists, join, basename
from numpy import (array, asarray, arange, linspace, zeros, zeros_like, ones, ones_like, delete,
                   poly1d, polyfit, vstack, cov, exp, log, sqrt, clip, pi)
from numpy.random import normal, uniform, multivariate_normal
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import fmin

from ldtool import ldtk_cache, with_notebook

from ld_models import LinearModel, QuadraticModel, NonlinearModel, GeneralModel 

## Set up some constants
## ---------------------
TWO_PI      = 2*pi
TEFF_POINTS = delete(arange(2300,12001,100), [27])
LOGG_POINTS = arange(0,6.1,0.5)
Z_POINTS    = array([-4.0, -3.0, -2.0, -1.5, -1.0, 0, 0.5, 1.0])

if with_notebook:
    from IPython.display import display, clear_output
    from IPython.html.widgets import IntProgressWidget


def dxdx(f, x, h):
    return (f(x+h) - 2*f(x) + f(x-h)) / h**2


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


def inside(a,lims):
    return a[(a>=lims[0])&(a<=lims[1])]


def a_lims(a,v,e,s=3):
    return a[[max(0,a.searchsorted(v-s*e)-1),min(a.size-1, a.searchsorted(v+s*e))]]


class SpecIntFile(object):
    def __init__(self, teff, logg, z):
        tmpl = 'lte{teff:05d}-{logg:4.2f}-{z:3.1f}.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'
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
    def __init__(self, limits=None, verbosity=1):
        self.fnt = 'lte{teff:05d}-{logg:4.2f}-{z:3.1f}.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'
        self.eftp = 'phoenix.astro.physik.uni-goettingen.de'
        self.edir = 'SpecIntFITS/PHOENIX-ACES-AGSS-COND-SPECINT-2011'
        self.files = None
        self.verbosity = verbosity
        
        if limits:
            self.set_limits(*limits)

    def _local_path(self, teff_or_fn, logg=None, z=None):
        fn = teff_or_fn if isinstance(teff_or_fn, str) else self.create_name(teff_or_fn,logg,z)
        return join(ldtk_cache,'Z'+fn[13:17],fn)
        
    def _local_exists(self, teff_or_fn, logg=None, z=None):
        print self._local_path(teff_or_fn, logg, z)
        return exists(self._local_path(teff_or_fn, logg, z))
        
    def create_name(self, teff, logg, z):
        return self.fnt.format(teff=int(teff), logg=logg, z=z)
    
    def set_limits(self, teff_lims, logg_lims, z_lims):
        self.teffl = teff_lims
        self.teffs = inside(TEFF_POINTS, teff_lims)
        self.nteff = len(self.teffs)
        self.loggl = logg_lims
        self.loggs = inside(LOGG_POINTS, logg_lims)
        self.nlogg = len(self.loggs)
        self.zl    = z_lims
        self.zs    = inside(Z_POINTS, z_lims)
        self.nz    = len(self.zs)
        self.pars  = [p for p in product(self.teffs,self.loggs,self.zs)]
        self.files = [SpecIntFile(*p) for p in product(self.teffs,self.loggs,self.zs)]

        self.not_cached =  len(self.files) - sum([exists(f) for f in self.local_filenames])
        if self.not_cached > 0:
            print "Need to download {:d} files, approximately {} MB".format(self.not_cached, 16*self.not_cached)
    
    def files_exist(self, files=None):
        ftp = FTP(self.eftp)
        ftp.login()
        ftp.cwd(self.edir)
        ftp.cwd('Z-0.0')
        efiles = []
        ftp.retrlines('list',lambda s: efiles.append(s.split()[-1].replace('+','-')))
        ftp.close()
        return [f in efiles for f in (files or self.files)]
    
    def download_uncached_files(self, force=False):
        if with_notebook:
            pbar = IntProgressWidget()
            if (self.verbosity > 0) & (self.not_cached > 0):
                display(pbar)
            pbar.max = max(1, self.not_cached if not force else len(self.files))

        ftp = FTP(self.eftp)
        ftp.login()
        ftp.cwd(self.edir)
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
                if with_notebook:
                    pbar.value += 1
            else:
                if self.verbosity > 1:
                    print 'Skipping an existing file: ', f.name
        ftp.close()
        if with_notebook:
            pbar.value = pbar.max

    @property
    def local_filenames(self):
        return [f.local_path for f in self.files]
        
        
class LDPSet(object):
    def __init__(self, filters, mu, ldp, ldp_s):
        self._filters  = filters 
        self._nfilters = len(filters)
        self._mu       = mu
        self._z        = sqrt(1-mu**2)
        self._mean     = ldp
        self._std      = ldp_s
        self._nmu      = mu.size
        self._samples  = {}

        self._lnl     = zeros(self._nfilters)
        self._lnc1    = -0.5*self._nmu*log(TWO_PI)              ## 1st ln likelihood term
        self.set_uncertainty_multiplier(1.)

        self.lnlike_ln = partial(self._lnlike, ldmodel=LinearModel)
        self.lnlike_qd = partial(self._lnlike, ldmodel=QuadraticModel)
        self.lnlike_nl = partial(self._lnlike, ldmodel=NonlinearModel)
        self.lnlike_ge = partial(self._lnlike, ldmodel=GeneralModel)

        self.lnlike_ln.__doc__ = "Linear limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_qd.__doc__ = "Quadratic limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_nl.__doc__ = "Nonlinear limb darkening model\n(coeffs, join=True, flt=None)"
        self.lnlike_ge.__doc__ = "General limb darkening model\n(coeffs, join=True, flt=None)"


    def set_uncertainty_multiplier(self, em):
        self._em      = em                                      ## uncertainty multiplier
        self._lnc2    = [-log(em*e).sum() for e in self._std]   ## 2nd ln likelihood term
        self._err2    = [(em*e)**2 for e in self._std]          ## variances


    def coeffs_qd(self, return_cm=False, do_mc=False, n_mc_samples=20000, mc_thin=25, mc_burn=25, return_chain=False):
        qcs  = [fmin(lambda pv:-self.lnlike_qd(pv, flt=iflt), [0.2,0.1], disp=0) for iflt in range(self._nfilters)]
        covs = []
        for iflt, qc in enumerate(qcs):
            s1 = 1/sqrt(-dxdx(lambda x:self.lnlike_qd([x,qc[1]], flt=iflt), qc[0], 1e-5))
            s2 = 1/sqrt(-dxdx(lambda x:self.lnlike_qd([qc[0],x], flt=iflt), qc[1], 1e-5))

            ## Simple MCMC uncertainty estimation
            ## ----------------------------------
            if do_mc:
                self._samples['quadratic'] = []
                logl  = zeros(n_mc_samples)
                chain = zeros([n_mc_samples,2])
                
                chain[0,:] = qc
                logl[0]    = self.lnlike_qd(chain[0], flt=iflt)

                for i in xrange(1,n_mc_samples):
                    pos_t  = multivariate_normal(chain[i-1], [[s1**2,0.0],[0.0,s2**2]])
                    logl_t = self.lnlike_qd(pos_t, flt=iflt)
                    if uniform() < exp(logl_t-logl[i-1]):
                        chain[i,:] = pos_t
                        logl[i]    = logl_t
                    else:
                        chain[i,:] = chain[i-1,:]
                        logl[i]    = logl[i-1]
                self._samples['quadratic'].append(chain)
                ch = chain[mc_burn::mc_thin,:]

                if return_cm:
                    covs.append(cov(ch[:,0], ch[:,1]))
                else:
                    covs.append(sqrt(cov(ch[:,0], ch[:,1]).diagonal()))

            else:
                if return_cm:
                    covs.append([[s1**2,0],[0,s2**2]])
                else:
                    covs.append([s1,s2])

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

        with pf.open(self.files[0]) as hdul:
            wl0  = hdul[0].header['crval1'] * 1e-1 # Wavelength at d[:,0] [nm]
            dwl  = hdul[0].header['cdelt1'] * 1e-1 # Delta wavelength     [nm]
            nwl  = hdul[0].header['naxis1']        # Number of wl samples
            wl   = wl0 + arange(nwl)*dwl
            self.mu_orig   = hdul[1].data
            self.z_orig    = sqrt(1-self.mu_orig**2)
            self.nmu_orig  = self.mu_orig.size
        
        self.fluxes   = zeros([self.nfilters, self.nfiles, self.nmu_orig])
        for fid,f in enumerate(self.filters):
            w = f(wl) * self.qe(wl)
            for did,df in enumerate(self.files):
                self.fluxes[fid,did,:]  = (pf.getdata(df)*w).mean(1)
                self.fluxes[fid,did,:] /= self.fluxes[fid,did,-1]

        ## Create n_filter interpolator objects
        ##
        self.itps = [RGI((c.teffs, c.loggs, c.zs, self.mu_orig), 
                         self.fluxes[i,:,:].reshape([c.nteff, c.nlogg, c.nz, self.nmu_orig])) for i in range(self.nfilters)]
        
        self.z  = linspace(0,0.995,self.nmu_orig)
        self.mu = sqrt(1-self.z**2)
        self.nmu = self.mu.size
       

    def create_profiles(self, nsamples=20):
        self.vals = zeros([self.nfilters, nsamples, self.nmu])
        for iflt in range(self.nfilters):
            for ismp in range(nsamples):
                a = ones([self.nmu,4])
                a[:,0] = clip(normal(*self.teff),  *self.client.teffl)
                a[:,1] = clip(normal(*self.logg),  *self.client.loggl)
                a[:,2] = clip(normal(*self.metal), *self.client.zl)
                a[:,3] = self.mu
                self.vals[iflt,ismp,:] = self.itps[iflt](a)

        ldp_m = array([self.vals[i,:,:].mean(0) for i in range(self.nfilters)])
        ldp_s = array([self.vals[i,:,:].std(0)  for i in range(self.nfilters)])
        return LDPSet(self.filter_names, self.mu, ldp_m, ldp_s)

    @property
    def filter_names(self):
        return [f.name for f in self.filters]



## UTILITY CLASSES
## ---------------

class SpectralIP(object):
    def __init__(self, dfile):
        with pf.open(dfile) as hdul:
            self.wl0  = hdul[0].header['crval1'] * 1e-1 # Wavelength at d[:,0] [nm]
            self.dwl  = hdul[0].header['cdelt1'] * 1e-1 # Delta wavelength     [nm]
            self.nwl  = hdul[0].header['naxis1']        # Number of samples
            self.data = hdul[0].data
            self.mu   = hdul[1].data
            self.z    = sqrt(1-self.mu**2)
            self.wl   = self.wl0 + arange(self.nwl)*self.dwl
                
    def intensity_profile(self, l0, l1):
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
