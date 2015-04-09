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

from numpy import asarray, zeros_like, interp, loadtxt

class QE(object):
    def __init__(self, instrument):
        self.instrument = instrument

    def __call__(self, wl):
        raise NotImplementedError


class TabulatedQE(QE):
    """Quantum efficiency tabulated as a function of wavelength.
    """
    def __init__(self, instrument, wl_or_fname, tm=None):
        """
        Parameters

        name        : str           filter name
        wl_or_fname : array or str  a list of wavelength or filename
        tm          : array         a list of transmission values
        """
        super(TabulatedQE,self).__init__(instrument)
        if isinstance(wl_or_fname,str):
            self.wl, self.tm = loadtxt(wl_or_fname).T
        else:
            self.wl = asarray(wl)
            self.tm = asarray(tm)
        self.tm *= tmf

    def __call__(self, wl):
        return interp(wl, self.wl, self.tm, 0., 0.)
