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
from typing import Optional

import pandas as pd

from pathlib import Path

from matplotlib.pyplot import subplots, setp
from numpy import array, argsort, zeros_like, arange, loadtxt, linspace, floor
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar


class Filter:
    def __init__(self, name):
        self.name = name

    def __call__(self, wl):
        raise NotImplementedError

    def integrate(self, wavelengths, values):
        raise NotImplementedError

    def plot(self, ax=None, wl_min=300, wl_max=1000, wl_res=500):
        if ax is None:
            fig, ax = subplots()
        wl = linspace(wl_min, wl_max, wl_res)
        ax.plot(wl, self(wl))
        setp(ax, xlabel='Wavelength [nm]', ylabel='Transmission')
        return ax


class SVOFilter(Filter):
    shortcuts = dict(kepler='Kepler/Kepler.k',
                     tess='TESS/TESS.Red',
                     sdss_g='SLOAN/SDSS.g',
                     sdss_r='SLOAN/SDSS.r',
                     sdss_i='SLOAN/SDSS.i',
                     sdss_z='SLOAN/SDSS.z')

    def __init__(self, name: str):
        """Creates a filter using the Spanish Virtual Observatory (SVO) Filter Profile Service.

        Creates a filter using the Spanish Virtual Observatory (SVO) Filter Profile
        Service. The filter name can be either an SVO filter name such as "SLOAN/SDSS.z"
        or "Kepler/Kepler.k" or a name shortcut. You can get a dictionary of available
        shortcuts from `SVOFilter.shortcuts`.

        Notes
        -----
        - Requires an internet connection.
        - The SVO FPS is hosted at http://svo2.cab.inta-csic.es/theory/fps/

        Parameters
        ----------
        name : str
            SVO filter name such as "SLOAN/SDSS.z" or a name shortcut such as "TESS".
        """
        from astroquery.svo_fps import SvoFps as svo
        if name.lower() in self.shortcuts.keys():
            name = self.shortcuts[name.lower()]
        super().__init__(name)
        self._svo_data = svo.get_transmission_data(name)
        self.wavelength = wl = self._svo_data['Wavelength'].compressed().astype('d') / 10
        self.transmission = tr = self._svo_data['Transmission'].compressed().astype('d')
        self.transmission /= self.transmission.max()
        self.bbox = wl[tr > 1e-2][[0, -1]]
        self._ip = interp1d(self.wavelength, self.transmission, kind='cubic', bounds_error=False, fill_value=0.0)

    def __call__(self, wl):
        return self._ip(wl)

    def sample(self, n: Optional[int] = 100):
        return self.wavelength, self.transmission

    def plot(self, bbox: bool = False, ax=None):
        if ax is None:
            fig, ax = subplots()
        else:
            fig, ax = None, ax

        ax.plot(self.wavelength, self.transmission)
        if bbox:
            ax.axvspan(*self.bbox, fill=False, ls='--', ec='k')
        setp(ax, xlabel='Wavelength [nm]', ylabel='Transmission')
        return ax


class TabulatedFilter(Filter):
    """Tabulated filter where the transmission is
       listed as a function of wavelength.
    """
    def __init__(self, name, wl_or_fname, tm=None, tmf=1.):
        """
        Parameters

        name        : str           filter name
        wl_or_fname : array or str  a list of wavelength or filename
        tm          : array         a list of transmission values
        tmf         : float         should be set to 1e-2 if the transmission is given in percents
        """
        super(TabulatedFilter,self).__init__(name)
        if isinstance(wl_or_fname,str):
            self.wl, self.tm = loadtxt(wl_or_fname).T
        else:
            self.wl = array(wl_or_fname)
            self.tm = array(tm)
        self.tm *= tmf

        sid = argsort(self.wl)
        self.wl = self.wl[sid]
        self.tm = self.tm[sid]
        self._ip = interp1d(self.wl, self.tm, kind='cubic', bounds_error=False, fill_value=0.)

    def __call__(self, wl):
        return self._ip(wl)

    def integrate(self, wavelengths, values):
        w = self(wavelengths)
        if values.ndim == 2 and values.shape[1] == wavelengths.size:
            return (values * w).sum(1)
        elif values.ndim == 1 and values.size == wavelengths.size:
            return (values * w).sum()
        else:
            raise ValueError


class BoxcarFilter(Filter):
    """Boxcar filter."""
    def __init__(self, name, wl_min, wl_max):
        """
        Parameters

        name        : str           filter name
        wl_min      : float         minimum wavelength [nm]
        wl_max      : float         maximum wavelength [nm]
        """
        super(BoxcarFilter,self).__init__(name)
        self.wl_min = wl_min
        self.wl_max = wl_max
        
    def __call__(self, wl):
        w = zeros_like(wl)
        w[(wl>self.wl_min) & (wl<self.wl_max)] = 1.
        return w

    def integrate(self, wavelengths, values):
        w = self(wavelengths)
        if values.ndim == 2 and values.shape[1] == wavelengths.size:
            return (values * w).sum(1)
        elif values.ndim == 1 and values.size == wavelengths.size:
            return (values * w).sum()
        else:
            raise ValueError


class DeltaFilter(Filter):
    def __init__(self, name, wl):
        super().__init__(name)
        self.wl = wl

    def integrate(self, wavelengths, values):
        i = int(root_scalar(lambda i: wavelengths[int(floor(i))]-self.wl, bracket=[0, wavelengths.size-1]).root)
        if wavelengths[i] > self.wl:
            i -= 1
        x = (self.wl - wavelengths[i])/(wavelengths[i+1]-wavelengths[i])

        if values.ndim == 2 and values.shape[1] == wavelengths.size:
            return (1-x)*values[:,i] + x*values[:,i+1]
        elif values.ndim == 1 and values.size == wavelengths.size:
            return (1-x)*values[i] + x*values[i+1]
        else:
            raise ValueError


def create_tess():
    df = pd.read_csv(Path(__file__).parent.joinpath('filter_files','tess.csv'), comment='#', header=None, names=['wavelength','response'])
    return TabulatedFilter('TESS', df.wavelength.values, df.response.values)


sdss_g = BoxcarFilter("g'", 400, 550)
sdss_r = BoxcarFilter("r'", 570, 690)
sdss_i = BoxcarFilter("i'", 710, 790)
sdss_z = BoxcarFilter("z'", 810, 900)

kepler = TabulatedFilter('kepler',
                         arange(350, 960, 25),
                         array([0.000, 0.001, 0.000, 0.056, 0.465, 0.536, 0.624, 0.663,
                                0.681, 0.715, 0.713, 0.696, 0.670, 0.649, 0.616, 0.574,
                                0.541, 0.490, 0.468, 0.400, 0.332, 0.279, 0.020, 0.000,
                                0.000]))

tess = create_tess()

__all__ = 'Filter TabulatedFilter BoxcarFilter DeltaFilter sdss_g sdss_r sdss_i sdss_z kepler tess'.split()
