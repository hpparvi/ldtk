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
import astropy.io.fits as pf
from glob import glob
from os.path import exists, join, basename
from pickle import dump, load
from numpy import (array, asarray, arange, linspace, zeros, zeros_like, ones, ones_like, delete, append,
                   diag, poly1d, polyfit, vstack, diff, cov, exp, log, sqrt, clip, pi, percentile)
from numpy.random import normal, uniform, multivariate_normal
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
    
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    with_mpi = True
except ImportError:
    comm = None
    mpi_rank = 0
    mpi_size = 1
    with_mpi = False

is_root = mpi_rank == 0

ldtk_root  = os.getenv('LDTK_ROOT') or join(os.getenv('HOME'),'.ldtk')
if not exists(ldtk_root):
    os.mkdir(ldtk_root)

## Constants
## =========

TWO_PI      = 2*pi

TEFF_POINTS = append(arange(2300,7001,100), arange(7200,12001,200)) # The database uses two TEFF step sizes
TEFF_POINTS = delete(TEFF_POINTS, TEFF_POINTS.searchsorted(5000))   # The 5000 K data are missing from the database
LOGG_POINTS = arange(0,6.1,0.5)
Z_POINTS    = array([-4.0, -3.0, -2.0, -1.5, -1.0, -0.0, 0.5, 1.0])
FN_TEMPLATE = 'lte{teff:05d}-{logg:4.2f}{z:+3.1f}.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'

## Utility functions
## =================

def message(text):
    tqdm.write(text)

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

def a_lims_hilo(a,lo,hi):
    return a[[max(0,a.searchsorted(lo)-1),min(a.size-1, a.searchsorted(hi))]]

## Utility classes
## ===============
class SpecIntFile(object):
    def __init__(self, teff, logg, z, cache):
        self.teff = int(teff)
        self.logg = logg
        self.z    = z
        self.name  = FN_TEMPLATE.format(teff=self.teff, logg=self.logg, z=self.z)
        self._zstr = 'Z'+self.name[13:17]
        self.cache = cache
        
    @property
    def local_path(self):
        return join(self.cache, self._zstr, self.name)

    @property
    def local_exists(self):
        return exists(self.local_path)


class SIS(object):
    """Simple wrapper for a specific intensity spectrum file."""
    def __init__(self, fname):
        self.filename = fname
        with pf.open(fname) as hdul:
            self.wl0  = hdul[0].header['crval1'] * 1e-1 # Wavelength at d[:,0] [nm]
            self.dwl  = hdul[0].header['cdelt1'] * 1e-1 # Delta wavelength     [nm]
            self.nwl  = hdul[0].header['naxis1']        # Number of samples
            self.data = hdul[0].data.copy()
            self.mu   = hdul[1].data.copy()
            self.z    = sqrt(1-self.mu**2)
            self.wl   = self.wl0 + arange(self.nwl)*self.dwl
        del hdul[0].data
        del hdul[1].data

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

        del hdul[0].data
        del hdul[1].data
