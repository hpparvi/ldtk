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

from typing import Literal, Optional

from numpy import asarray, cov, log, ndarray, pi, searchsorted
from numpy.linalg import eigh, svd


class ReducedRankLL:
    """Reduced-rank Normal log-likelihood for limb darkening profile samples.

    The limb darkening profile samples are smooth functions of a small number
    of latent stellar parameters (teff, logg, z), so their empirical covariance
    over the mu grid is strongly rank-deficient: the profiles carry only a
    handful of effective degrees of freedom no matter how densely they are
    sampled in mu. This class projects the model residuals into the principal
    subspace of the sample covariance (Karhunen-Loeve compression) and
    evaluates the Normal log-likelihood there, which makes the likelihood
    insensitive to the mu-grid resolution.

    The log-likelihood is evaluated as

    .. math:: \\ln \\mathcal{L} = -\\frac{1}{2} \\left[ \\sum_{i=1}^{K} \\frac{p_i^2}{\\lambda_i} + \\sum_{i=1}^{K} \\ln \\lambda_i + K \\ln 2\\pi \\right]

    where :math:`\\lambda_i` are the significant eigenvalues of the sample
    covariance and :math:`p_i` are the projections of the residuals onto the
    corresponding eigenvectors.

    Parameters
    ----------
    mu
        The mu grid with shape (M,) on which the profile samples and models
        are evaluated.
    samples
        The profile samples with shape (N_samples, M).
    cev
        Cumulative explained variance threshold: the smallest number of
        leading eigenmodes whose eigenvalues sum to at least ``cev`` times
        the total variance is kept. The threshold adapts to the shape of the
        eigenvalue spectrum and discards the near-noise modes whose tiny
        eigenvalues would otherwise dominate the chi-square through the
        1/lambda weights when fitting structurally mismatched LD models.
    nk
        Optional hard upper limit on the number of eigenmodes to keep.
    method
        Decomposition method, either ``'svd'`` (default, decomposes the
        centered sample matrix directly) or ``'eigh'`` (decomposes the
        empirical covariance matrix).

    References
    ----------
    Tegmark, M., Taylor, A. N., & Heavens, A. F. (1997). Karhunen-Loeve
    eigenvalue problems in cosmology: how should we tackle large data sets?
    *The Astrophysical Journal*, 480(1), 22.
    """

    def __init__(self, mu: ndarray, samples: ndarray, cev: float = 0.999, nk: Optional[int] = None,
                 method: Literal['svd', 'eigh'] = 'svd'):
        self.mu = mu
        samples = asarray(samples)
        self.mean = samples.mean(0)

        if method == 'svd':
            _, sigma, evecs = svd(samples - self.mean, full_matrices=False)
            evals = sigma ** 2 / (samples.shape[0] - 1)
            evecs = evecs.T
        elif method == 'eigh':
            evals, evecs = eigh(cov(samples, rowvar=False))
            evals, evecs = evals[::-1], evecs[:, ::-1]
        else:
            raise ValueError(f"Unknown decomposition method '{method}', should be either 'svd' or 'eigh'.")

        nkeep = int(searchsorted(evals.cumsum() / evals.sum(), cev)) + 1
        if nk is not None:
            nkeep = min(nkeep, nk)
        self.eigenvalues = evals[:nkeep]
        self.eigenvectors = evecs[:, :nkeep]
        self.nk = self.eigenvalues.size
        self.log_det = log(self.eigenvalues).sum()
        self.log_twopi = self.nk * log(2 * pi)

    def __call__(self, model: ndarray, em: float = 1.0):
        """Evaluate the log-likelihood of one or more model profiles.

        Parameters
        ----------
        model
            Model profile(s) with shape (M,) or (npv, M) evaluated on the
            mu grid given in the initialization.
        em
            Uncertainty multiplier: scales the sample covariance by em**2.

        Returns
        -------
        float or ndarray
            The log-likelihood, a float for a (M,) model and an (npv,)
            array for an (npv, M) model array.
        """
        p = (self.mean - asarray(model)) @ self.eigenvectors
        chisq = (p ** 2 / (em ** 2 * self.eigenvalues)).sum(-1)
        return -0.5 * (chisq + self.log_det + 2 * self.nk * log(em) + self.log_twopi)
