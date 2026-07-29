"""
Limb darkening toolkit
Copyright (C) 2015-2026  Hannu Parviainen <hpparvi@gmail.com>

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

from numpy import asarray, atleast_2d, ndarray
from scipy.interpolate import RBFInterpolator


class RBFProfileInterpolator:
    """Smooth radial basis function interpolator for limb darkening profiles.

    A drop-in replacement for the piecewise-linear
    `scipy.interpolate.LinearNDInterpolator` used by `LDPSetCreator`: it
    mirrors the call signature (parameters in, profiles out), but

    - is smooth, so the nonlinearity of the simulated profiles between the
      grid nodes is captured instead of being linearized away inside each
      grid simplex,
    - works on scattered nodes, so missing grid points and irregular node
      spacings need no special handling, and
    - extrapolates smoothly instead of returning NaN outside the convex hull
      of the available nodes.

    The stellar parameter space is normalized to the unit cube before
    interpolation so that a single isotropic kernel can act on teff (~1e3 K),
    logg, and z alike.

    Parameters
    ----------
    points
        Grid node stellar parameters with shape (n_nodes, 3): teff, logg, z.
    profiles
        Simulated profiles at the grid nodes with shape (n_nodes, n_mu).
    kernel
        Radial basis function, passed to `scipy.interpolate.RBFInterpolator`.
        The default 'thin_plate_spline' has no shape parameter to tune and
        gives an exact, smooth interpolant.
    smoothing
        Smoothing parameter: 0 (default) interpolates the nodes exactly,
        larger values allow the interpolant to deviate from the node values.
    neighbors
        If given, the interpolant at each point is computed using only this
        many nearest nodes, which reduces the cost for very large grids.
    """

    def __init__(self, points: ndarray, profiles: ndarray, kernel: str = 'thin_plate_spline',
                 smoothing: float = 0.0, neighbors: Optional[int] = None):
        points = asarray(points)
        self._pmin = points.min(0)
        self._pscale = points.max(0) - self._pmin
        self._pscale[self._pscale == 0.0] = 1.0
        self._rbf = RBFInterpolator((points - self._pmin) / self._pscale, asarray(profiles),
                                    kernel=kernel, smoothing=smoothing, neighbors=neighbors)

    def __call__(self, theta: ndarray) -> ndarray:
        """Interpolate the profiles for a set of stellar parameters.

        Parameters
        ----------
        theta
            Stellar parameters with shape (3,) or (n, 3): teff, logg, z.

        Returns
        -------
        ndarray
            Interpolated profiles with shape (n, n_mu).
        """
        x = (atleast_2d(asarray(theta)) - self._pmin) / self._pscale
        return self._rbf(x)
