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

import numpy as np

from numba import njit
from numpy import ndarray, asarray, sqrt, log2, atleast_2d, zeros

from .models import m_quadratic, m_triangular

@njit
def _get_dims(mu, pv):
    if pv.ndim == 1:
        npv, npb = 1, 1
    elif pv.ndim == 2:
        npv, npb = 1, pv.shape[0]
    else:
        npv, npb = pv.shape[0], pv.shape[1]
    return npv, npb, npv*npb, mu.size


class LDModel(object):
    npar = None
    name = None
    abbr = None

    def __init__(self):
        raise NotImplementedError

    def __call__(self, mu, pv):
        raise NotImplementedError

    @classmethod
    def evaluate(cls, mu: ndarray, pv: ndarray) -> ndarray:
        raise NotImplementedError


class LinearModel(LDModel):
    """Linear limb darkening model (Mandel & Agol, 2001)."""
    npar = 1
    name = 'linear'
    abbr = 'ln'

    @classmethod
    def evaluate(cls, mu: ndarray, pv: ndarray) -> ndarray:
        assert len(pv) == cls.npar
        mu = asarray(mu)
        return 1. - pv[0]*(1.-mu)


class QuadraticModel(LDModel):
    """Quadratic limb darkening model (Kopal, 1950)."""
    npar = 2
    name = 'quadratic'
    abbr = 'qd'

    @classmethod
    def evaluate(cls, mu: ndarray, pv: ndarray) -> ndarray:
        return m_quadratic(mu, pv)


class TriangularQuadraticModel(LDModel):
    """Quadratic limb darkening model with the parametrisation described by Kipping (MNRAS 435, 2013)."""
    npar = 2
    name = 'triangular_quadratic'
    abbr = 'tq'

    @classmethod
    def evaluate(cls, mu: ndarray, pv: ndarray) -> ndarray:
        return m_triangular(mu, pv)


class SquareRootModel(LDModel):
    """Square root limb darkening model (van Hamme, 1993)."""
    npar = 2
    name = 'sqrt'
    abbr = 'sq'

    @classmethod
    def evaluate(cls, mu: ndarray, pv: ndarray) -> ndarray:
        assert len(pv) == cls.npar
        mu = asarray(mu)
        return 1. - pv[0]*(1.-mu) - pv[1]*(1.-mu**0.5)


# Nonlinear model
# ---------------
@njit(fastmath=True)
def _evaluate_nl(mu, coeffs):
    npv, npb, ncf, npt = _get_dims(mu, coeffs)
    cf = coeffs.reshape((ncf, 4))
    fl = zeros((ncf, npt))
    for i in range(ncf):
        fl[i, :] = (1. - (cf[i, 0] * (1. - mu**0.5) + cf[i, 1] * (1. - mu) +
                          cf[i, 2] * (1. - mu**1.5) + cf[i, 3] * (1. - mu**2)))
    return fl.reshape((npv, npb, npt))

class NonlinearModel(LDModel):
    """Nonlinear limb darkening model (Claret, 2000)."""
    npar = 4
    name = 'nonlinear'
    abbr = 'nl'

    @classmethod
    def evaluate(cls, mu: ndarray, pv: ndarray) -> ndarray:
        assert len(pv) == cls.npar
        return _evaluate_nl(mu, pv).squeeze()


class GeneralModel(LDModel):
    """General limb darkening model (Gimenez, 2006)"""
    npar = None
    name = 'general'
    abbr = 'ge'

    @classmethod
    def evaluate(cls, mu: ndarray, pv: ndarray) -> ndarray:
        mu = asarray(mu)
        return 1. - np.sum([c*(1.-mu**(i+1)) for i,c in enumerate(pv)], 0)


# Power2 model
# ------------
@njit(fastmath=True)
def _evaluate_p2(mu, coeffs):
    npv, npb, ncf, npt = _get_dims(mu, coeffs)
    cf = coeffs.reshape((ncf, 2))
    fl = zeros((ncf, npt))
    for i in range(ncf):
        fl[i, :] = 1. - cf[i,0]*(1.-(mu**cf[i,1]))
    return fl.reshape((npv, npb, npt))


class Power2Model(LDModel):
    """Power-2 limb darkening model (Morello et al, 2017)."""
    npar = 2
    name = 'power2'
    abbr = 'p2'

    @classmethod
    def evaluate(cls, mu: ndarray, pv: ndarray) -> ndarray:
        return _evaluate_p2(mu, pv).squeeze()


# Power2 model, alternative parametrisation
# -----------------------------------------
@njit(fastmath=True)
def _evaluate_p2mp(mu, coeffs):
    npv, npb, ncf, npt = _get_dims(mu, coeffs)
    cf = coeffs.reshape((ncf, 2))
    fl = zeros((ncf, npt))
    for i in range(ncf):
        c = 1. - cf[i, 0] + cf[i, 1]
        a = log2(c / cf[i, 1])
        fl[i, :] = 1. - c * (1. - mu**a)
    return fl.reshape((npv, npb, npt))


class Power2MPModel(LDModel):
    """Power-2 limb darkening model with an alternative parametrisation (Maxted, P.F.L., 2018)."""
    npar = 2
    name = 'power2mp'
    abbr = 'p2mp'

    @classmethod
    def evaluate(cls, mu: ndarray, pv: ndarray) -> ndarray:
        return _evaluate_p2mp(mu, pv).squeeze()


models = {'linear':    LinearModel,
          'quadratic': QuadraticModel,
          'triquadratic': TriangularQuadraticModel,
          'sqrt':      SquareRootModel,
          'nonlinear': NonlinearModel,
          'general':   GeneralModel,
          'power2':    Power2Model,
          'power2mp': Power2MPModel}

