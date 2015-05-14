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

from numpy import array, argsort, zeros_like, interp, loadtxt

class Filter(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, wl):
        raise NotImplementedError


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


    def __call__(self, wl):
        return interp(wl, self.wl, self.tm, 0., 0.)


class BoxcarFilter(Filter):
    """A simple boxcar filter."""
    def __init__(self, name, wl_min, wl_max):
        super(BoxcarFilter,self).__init__(name)
        self.wl_min = wl_min
        self.wl_max = wl_max
        
    def __call__(self, wl):
        w = zeros_like(wl)
        w[(wl>self.wl_min) & (wl<self.wl_max)] = 1.
        return w
