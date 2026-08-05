.. _likelihoods:

Coefficients and log-likelihoods
================================

Limb darkening models
---------------------

LDTk supports eight limb darkening laws. Each has a ``coeffs_*`` method for
coefficient estimation and an ``lnlike_*`` method for log-likelihood
evaluation, where ``*`` is the model abbreviation:

========== =============================== ==== =========================
Abbr.      Model                           npar Reference
========== =============================== ==== =========================
``ln``     Linear                          1    Schwarzschild (1906)
``qd``     Quadratic                       2    Kopal (1950)
``tq``     Triangular quadratic            2    Kipping (2013)
``sq``     Square root                     2    van Hamme (1993)
``nl``     Nonlinear (four-parameter)      4    Claret (2000)
``ge``     General                         any  Giménez (2006)
``p2``     Power-2                         2    Morello et al. (2017)
``p2mp``   Power-2, alt. parametrisation   2    Maxted (2018)
========== =============================== ==== =========================

Estimating coefficients
-----------------------

The ``coeffs_*`` methods maximize the profile log-likelihood per filter and
return arrays of coefficients and uncertainties:

.. code-block:: python

    qc, qe = ps.coeffs_qd()            # uncertainties from the curvature of
                                       # the log-likelihood
    qc, qe = ps.coeffs_qd(do_mc=True)  # uncertainties from a built-in
                                       # Metropolis MCMC
    qc, qc_cov = ps.coeffs_qd(return_cm=True)  # full covariance matrices

Evaluating log-likelihoods
--------------------------

The ``lnlike_*`` methods accept coefficient arrays in three shapes:

.. code-block:: python

    ps.lnlike_qd([0.25, 0.05], flt=0)  # 1D: one coefficient set for the
                                       #     filter given by flt
    ps.lnlike_qd(ldcs_2d)              # 2D [ifilter, icoeff]: one set per
                                       #     filter, returns the joint lnL
    ps.lnlike_qd(ldcs_3d)              # 3D [iset, ifilter, icoeff]: many
                                       #     proposal sets at once, returns
                                       #     an array of joint lnL values

The 3D form is convenient inside samplers that evaluate a population of
parameter vectors per step.

How the likelihood is calculated
--------------------------------

Since version 1.9, LDTk calculates the likelihood using reduced-rank Normal
log-likelihood method (:class:`~ldtk.loglikelihood.ReducedRankLL`) following
the Karhunen–Loève eigenmode formalism of `Tegmark et al. (1997)
<https://ui.adsabs.harvard.edu/abs/1997ApJ...480...22T/>`_.

The empirical covariance of the profile samples is eigendecomposed, and the
likelihood of a model profile is evaluated in the subspace spanned by the
significant eigenmodes:

.. math::

    \ln \mathcal{L} = -\frac{1}{2} \left[ \sum_{i=1}^{K}
    \frac{p_i^2}{\lambda_i} + \sum_{i=1}^{K} \ln \lambda_i
    + K \ln 2\pi \right],

where :math:`\lambda_i` are the leading eigenvalues of the sample covariance
and :math:`p_i` are the projections of the residuals (mean profile minus
model) onto the corresponding eigenvectors. The number of kept modes
:math:`K` is chosen as the smallest number of leading eigenmodes that
together explain a fraction ``cev`` (default 0.999) of the total sample
variance; an optional hard cap ``nk`` can also be given. Both are set in the
:class:`~ldtk.ldtk.LDPSet` constructor.

Compared to the earlier likelihood that treated every :math:`\mu` point as an
independent measurement, this

* yields more realistic limb darkening coefficient uncertainties, and
* makes the likelihood insensitive to the :math:`\mu` resampling resolution:
  the information content is set by the eigenvalue spectrum, not by the
  number of tabulated points.

Typically :math:`K` is 2–4. If the profiles were exactly linear in the three
stellar parameters, :math:`K` could not exceed three; values above that
appear when the parameter posterior spans several grid cells in a region
where the simulated profiles change nonlinearly from node to node (cool
stars and molecular-band-dominated passbands, for example).
:meth:`~ldtk.ldtk.LDPSet.diagnostics` prints the kept-mode count and the
leading relative eigenvalue spectrum per filter:

.. code-block:: text

    >>> ps.diagnostics()
    b: 2 of 100 eigenmodes kept (cev = 0.999)
      mode  rel. eigenvalue  cumulative
        1*        9.935e-01    0.993534
        2*        5.676e-03    0.999210
        3         6.821e-04    0.999893
        ...

The legacy likelihood
---------------------

The pre-1.9 likelihood, which assumes independent :math:`\mu` points with a
diagonal covariance, is available for comparison by passing
``likelihood='diagonal'`` to :class:`~ldtk.ldtk.LDPSet` or by calling
:meth:`~ldtk.ldtk.LDPSet.set_likelihood_mode`. Note that it can be overconfident
and that its sharpness grows with the number of :math:`\mu` points set by the resampling.

The uncertainty multiplier
--------------------------

:meth:`~ldtk.ldtk.LDPSet.set_uncertainty_multiplier` scales the profile
sample covariance by ``em**2``. Its purpose is to account for the fact that
the stellar atmosphere models themselves are imperfect: the simulated
profiles can contain unknown biases and trends that do not agree with
observations, and this model uncertainty is not captured by propagating the
stellar parameter uncertainties alone. In earlier LDTk versions the
multiplier also had to compensate for the overconfidence of the
independent-points likelihood; with the reduced-rank likelihood that role is
gone, and the multiplier serves only its original purpose.
