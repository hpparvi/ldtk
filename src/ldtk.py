import os
import pyfits as pf

from glob import glob
from ftplib import FTP
from itertools import product
from os.path import exists, join, basename
from numpy import (array, asarray, arange, linspace, zeros, zeros_like, ones, ones_like,
                   log, sqrt, clip, pi)
from numpy.random import normal
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import fmin

try:
    from IPython.display import display, clear_output
    from IPython.html.widgets import IntProgressWidget
    w = IntProgressWidget()
    with_notebook = True
except AttributeError:
    with_notebook = False

home = os.environ['HOME']
ddir = join(home,'work/Projects/RoPACS/data/phoenix_specint')

TWO_PI = 2*pi

def quadratic_law(mu,ld):
    """Quadratic limb-darkening law as  described in (Mandel & Agol, 2001).
    """
    return 1. - ld[0]*(1.-mu) - ld[1]*(1.-mu)**2


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
        return join(ddir,self._zstr,self.name)

    @property
    def local_exists(self):
        return exists(self.local_path)
    
    
class Runner(object):
    def __init__(self, limits=None, verbosity=1):
        self.teff_points = arange(2300,12001,100)
        self.logg_points = arange(0,6.1,0.5)
        self.z_points    = array([-4.0, -3.0, -2.0, -1.5, -1.0, 0, 0.5, 1.0])
        self.fnt = 'lte{teff:05d}-{logg:4.2f}-{z:3.1f}.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'
        self.eftp = 'phoenix.astro.physik.uni-goettingen.de'
        self.edir = 'SpecIntFITS/PHOENIX-ACES-AGSS-COND-SPECINT-2011'
        self.files = None
        self.verbosity = verbosity
        
        if limits:
            self.set_limits(*limits)

    def _local_path(self, teff_or_fn, logg=None, z=None):
        fn = teff_or_fn if isinstance(teff_or_fn, str) else self.create_name(teff_or_fn,logg,z)
        return join(ddir,'Z'+fn[13:17],fn)
        
    def _local_exists(self, teff_or_fn, logg=None, z=None):
        print self._local_path(teff_or_fn, logg, z)
        return exists(self._local_path(teff_or_fn, logg, z))
        
    def create_name(self, teff, logg, z):
        return self.fnt.format(teff=int(teff), logg=logg, z=z)
    
    def set_limits(self, teff_lims, logg_lims, z_lims):
        self.teffl = teff_lims
        self.teffs = inside(self.teff_points, teff_lims)
        self.nteff = len(self.teffs)
        self.loggl = logg_lims
        self.loggs = inside(self.logg_points, logg_lims)
        self.nlogg = len(self.loggs)
        self.zl    = z_lims
        self.zs    = inside(self.z_points, z_lims)
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
    
    def download_files(self, force=False):
        if with_notebook:
            pbar = IntProgressWidget()
            pbar.max = self.not_cached if not force else len(self.files)
            if self.verbosity > 0:
                display(pbar)
            
        ftp = FTP(self.eftp)
        ftp.login()
        ftp.cwd(self.edir)
        for fid,f in enumerate(self.files):
            if not exists(join(ddir,f._zstr)):
                os.mkdir(join(ddir,f._zstr))
            if not f.local_exists or force:
                localfile = open(f.local_path, 'wb')
                ftp.cwd(f._zstr)
                ftp.retrbinary('RETR '+f.name, localfile.write)
                ftp.cwd('..')
                localfile.close()
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
    def __init__(self, filters, mu, ldp, ldp_e):
        self.filters  = filters 
        self.nfilters = len(filters)
        self.mu       = mu
        self.z        = sqrt(1-mu**2)
        self.mean     = ldp
        self.error    = ldp_e
        self.nmu      = mu.size

        self._lnl     = zeros(self.nfilters)
        self._lnc1    = -0.5*self.nmu*log(TWO_PI)              ## 1st ln likelihood term
        self.set_uncertainty_multiplier(1.)

    def set_uncertainty_multiplier(self, em):
        self._em      = em                                     ## uncertainty multiplier
        self._lnc2    = [-log(em*e).sum() for e in self.error] ## 2nd ln likelihood term
        self._err2    = [(em*e)**2 for e in self.error]        ## variances

    @property
    def quadratic_coeffs(self):
        return [fmin(lambda pv:((self.mean[iflt]-quadratic_law(self.mu, pv))**2).sum(), [0.2,0.1], disp=0) for iflt in range(self.nfilters)]

    def lnlike_quadratic(self, ldcs, joint=False):
        for fid, ldc in enumerate(asarray(ldcs).reshape([-1,2])):
            model = quadratic_law(self.mu, ldc)
            self._lnl[fid] = self._lnc1 + self._lnc2[fid] -0.5*((self.mean[fid]-model)**2/self._err2[fid]).sum()
        return self._lnl.sum() if joint else self._lnl


class LDPSetCreator(object):
    def __init__(self, teff, logg, metal, runner, filters, limits=None):
        self.teff  = teff
        self.logg  = logg
        self.metal = metal

        self.runner   = r = runner
        self.files    = runner.local_filenames
        self.filters  = filters
        self.nfiles   = len(self.files)
        self.nfilters = len(filters)

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
            w = f(wl)
            for did,df in enumerate(self.files):
                self.fluxes[fid,did,:]  = (pf.getdata(df)*w).mean(1)
                self.fluxes[fid,did,:] /= self.fluxes[fid,did,-1]

        ## Create n_filter interpolator objects
        ##
        self.itps = [RGI((r.teffs,r.loggs, r.zs, self.mu_orig), 
                         self.fluxes[i,:,:].reshape([r.nteff, r.nlogg, r.nz, self.nmu_orig])) for i in range(self.nfilters)]
        
        self.z  = linspace(0,0.995,self.nmu_orig)
        self.mu = sqrt(1-self.z**2)
        self.nmu = self.mu.size
       

    def create_profiles(self, nsamples=20):
        self.vals = zeros([self.nfilters, nsamples, self.nmu])
        for iflt in range(self.nfilters):
            for ismp in range(nsamples):
                a = ones([self.nmu,4])
                a[:,0] = clip(normal(*self.teff),  *self.runner.teffl)
                a[:,1] = clip(normal(*self.logg),  *self.runner.loggl)
                a[:,2] = clip(normal(*self.metal), *self.runner.zl)
                a[:,3] = self.mu
                self.vals[iflt,ismp,:] = self.itps[iflt](a)

        ldp   = array([self.vals[i,:,:].mean(0) for i in range(self.nfilters)])
        ldp_e = array([self.vals[i,:,:].std(0)/sqrt(nsamples) for i in range(self.nfilters)])
        return LDPSet(self.filter_names, self.mu, ldp, ldp_e)

    @property
    def filter_names(self):
        return [f.name for f in self.filters]


class StepFilter(object):
    def __init__(self, name, wl_min, wl_max):
        self.name = name
        self.wl_min = wl_min
        self.wl_max = wl_max
        
    def __call__(self, wl):
        w = zeros_like(wl)
        w[(wl>self.wl_min) & (wl<self.wl_max)] = 1.
        return w


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
