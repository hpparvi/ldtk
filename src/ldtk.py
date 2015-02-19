import os
import pyfits as pf

from glob import glob
from ftplib import FTP
from itertools import product
from os.path import exists, join, basename
from numpy import (array, arange, linspace, zeros, zeros_like, ones, ones_like,
                   sqrt, clip)
from numpy.random import normal
from scipy.interpolate import RegularGridInterpolator as RGI

try:
    from IPython.display import display, clear_output
    from IPython.html.widgets import IntProgressWidget
    w = IntProgressWidget()
    with_notebook = True
except AttributeError:
    with_notebook = False

home = os.environ['HOME']
ddir = join(home,'work/Projects/RoPACS/data/phoenix_specint')

def inside(a,lims):
    return a[(a>=lims[0])&(a<=lims[1])]

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
    def __init__(self, verbosity=1):
        self.teff_points = arange(2300,12001,100)
        self.logg_points = arange(0,6.1,0.5)
        self.z_points    = array([-4.0, -3.0, -2.0, -1.5, -1.0, 0, 0.5, 1.0])
        self.fnt = 'lte{teff:05d}-{logg:4.2f}-{z:3.1f}.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'
        self.eftp = 'phoenix.astro.physik.uni-goettingen.de'
        self.edir = 'SpecIntFITS/PHOENIX-ACES-AGSS-COND-SPECINT-2011'
        self.files = None
        self.verbosity = verbosity
        
    def _local_path(self, teff_or_fn, logg=None, z=None):
        fn = teff_or_fn if isinstance(teff_or_fn, str) else self.create_name(teff_or_fn,logg,z)
        return join(ddir,'Z'+fn[13:17],fn)
        
    def _local_exists(self, teff_or_fn, logg=None, z=None):
        print self._local_path(teff_or_fn, logg, z)
        return exists(self._local_path(teff_or_fn, logg, z))
        
    def create_name(self, teff, logg, z):
        return self.fnt.format(teff=int(teff), logg=logg, z=z)
    
    def set_limits(self, teff_lims, logg_lims, z_lims):
        self.teffs = inside(self.teff_points, teff_lims)
        self.nteff = len(self.teffs)
        self.loggs = inside(self.logg_points, logg_lims)
        self.nlogg = len(self.loggs)
        self.zs    = inside(self.z_points, z_lims)
        self.nz    = len(self.zs)
        self.pars  = [p for p in product(self.teffs,self.loggs,self.zs)]
        self.files = [SpecIntFile(*p) for p in product(self.teffs,self.loggs,self.zs)]
    
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
            pbar.max = len(self.files)
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
            else:
                if self.verbosity > 1:
                    print 'Skipping an existing file: ', f.name
            if with_notebook:
                pbar.value += 1
        ftp.close()
        
    @property
    def local_filenames(self):
        return [f.local_path for f in self.files]
        
        
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

            
class IIPSet(object):
    """Set of intensity profiles integrated over given spectral passbands
    
    """
    def __init__(self, dfiles, filters):
        with pf.open(dfiles[0]) as hdul:
            wl0  = hdul[0].header['crval1'] * 1e-1 # Wavelength at d[:,0] [nm]
            dwl  = hdul[0].header['cdelt1'] * 1e-1 # Delta wavelength     [nm]
            nwl  = hdul[0].header['naxis1']        # Number of wl samples
            wl   = wl0 + arange(nwl)*dwl
            self.mu   = hdul[1].data
            self.z    = sqrt(1-self.mu**2)
            self.nmu  = self.mu.size
        
        self.files    = dfiles
        self.filters  = filters
        self.nfiles   = len(dfiles)
        self.nfilters = len(filters)
        self.fluxes   = zeros([self.nfilters, self.nfiles, self.mu.size])
        
        for fid,f in enumerate(self.filters):
            w = f(wl)
            for did,df in enumerate(dfiles):
                self.fluxes[fid,did,:]  = (pf.getdata(df)*w).mean(1)
                self.fluxes[fid,did,:] /= self.fluxes[fid,did,-1]
        
    @property
    def filter_names(self):
        return [f.name for f in self.filters]
                
        
class LDProfile(object):
    def __init__(self, runner, iipset):
        self.runner  = r = runner
        self.iipset  = i = iipset
        self.interp  = RGI((r.teffs,r.loggs, r.zs, i.mu), 
                              i.fluxes[0,:,:].reshape([r.nteff, r.nlogg, r.nz, i.nmu]))
        
        self.z  = linspace(0,0.995,i.mu.size)
        self.mu = sqrt(1-self.z**2)
       
    def create_profile(self, nsamples=20):
        self.vals = zeros([nsamples, self.iipset.mu.size])
        for j in range(self.vals.shape[0]):
            a = ones([self.iipset.mu.size,4])
            a[:,-1] = self.iipset.mu
            a[:,-1] = self.mu
            a[:,0] = clip(normal(5450,100),5300,5600)
            a[:,1] = normal(4.5,0.05)
            a[:,2] = clip(normal(0.25,0.05), 0, 0.5)
            self.vals[j,:] = self.interp(a)
        return self.vals.mean(0), self.vals.std(0)/sqrt(nsamples)

class StepFilter(object):
    def __init__(self, name, wl_min, wl_max):
        self.name = name
        self.wl_min = wl_min
        self.wl_max = wl_max
        
    def __call__(self, wl):
        w = zeros_like(wl)
        w[(wl>self.wl_min) & (wl<self.wl_max)] = 1.
        return w
