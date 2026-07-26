.. _profiles:

Limb darkening profiles
=======================

Creating a profile set
----------------------

:class:`~ldtk.ldtk.LDPSetCreator` turns stellar parameters and a set of
filters into limb darkening profile samples:

.. code-block:: python

    sc = LDPSetCreator(teff=(6400, 50), logg=(4.5, 0.2), z=(0.25, 0.05),
                       filters=filters)
    ps = sc.create_profiles(nsamples=500)

Each of ``teff``, ``logg``, and ``z`` can be given either as a
``(value, uncertainty)`` tuple — in which case the samples are drawn from a
normal distribution — or as a 1D array of posterior samples from, e.g., a
previous spectroscopic analysis. Alternative sample arrays can also be passed
directly to :meth:`~ldtk.ldtk.LDPSetCreator.create_profiles`.

For every parameter sample, the specific intensity profile is interpolated
from the PHOENIX grid for each filter, giving a set of profile samples whose
scatter propagates the stellar parameter uncertainties. Samples that fall
outside the coverage of the model library (some corners of the grid, e.g. hot
low-gravity stars, have no models) are dropped with a message.

By default, LDTk calculates photon-weighted averages appropriate for
photon-counting detectors (CCDs); set ``photon_counting=False`` for
energy-weighted averages, and use the ``qe`` argument to supply a detector
quantum efficiency curve.

.. _datasets:

Spectrum datasets
-----------------

The ``dataset`` argument of :class:`~ldtk.ldtk.LDPSetCreator` selects the
spectrum model library:

================== ================== ============================================
Dataset            Wavelength range   Notes
================== ================== ============================================
``vis``            50–2600 nm         The original Husser et al. (2013) library.
``vis-lowres``     50–2600 nm         Binned to 5 nm resolution. **Default.**
``visir``          50–5500 nm         Extended version of the original library.
``visir-lowres``   50–5500 nm         Extended library binned to 5 nm resolution.
================== ================== ============================================

The low-resolution sets are sufficient for most broadband and transmission
spectroscopy work and are much faster to download; use the full-resolution
sets when a spectral resolution better than 5 nm is required, and the
``visir`` variants when working in the infrared.

The stellar limb and resampling
-------------------------------

The PHOENIX specific intensity profiles extend beyond the photospheric edge of
the star, so the geometric :math:`\mu = \sqrt{1 - z^2}` does not correspond
directly to the :math:`\mu` used in transit models. On construction,
:class:`~ldtk.ldtk.LDPSet` locates the true stellar limb automatically by
fitting a limb darkening model together with a smootherstep edge function
(``fit_limb``), redefines :math:`\mu` to place zero at the fitted limb, and
resamples the profiles to a linear sampling in :math:`\mu`.

The sampling can be changed afterwards with
:meth:`~ldtk.ldtk.LDPSet.resample_linear_mu`,
:meth:`~ldtk.ldtk.LDPSet.resample_linear_z`, or
:meth:`~ldtk.ldtk.LDPSet.resample`, and the limb definition with
:meth:`~ldtk.ldtk.LDPSet.set_limb_mu` or
:meth:`~ldtk.ldtk.LDPSet.set_limb_z`. Thanks to the reduced-rank likelihood
(see :ref:`likelihoods`), the resampling resolution does not affect the
inferred limb darkening coefficients or their uncertainties.

The mean profiles and their uncertainties are available per filter through the
:attr:`~ldtk.ldtk.LDPSet.profile_averages` and
:attr:`~ldtk.ldtk.LDPSet.profile_uncertainties` properties.

Persistence
-----------

:meth:`~ldtk.ldtk.LDPSet.save` pickles the filters, the original :math:`\mu`
grid, and the original profile samples; :func:`~ldtk.ldtk.load_ldpset`
restores the set and rebuilds the derived state (limb fit, resampling, and the
likelihood decompositions) from scratch, so a saved set can be reused without
network access.
