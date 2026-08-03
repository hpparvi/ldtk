Quickstart
==========

A typical LDTk session has three steps: define the passbands, create the limb
darkening profiles for your star, and either estimate limb darkening
coefficients (to use as priors) or evaluate the profile log-likelihood directly
inside your transit model's log posterior.

1. Define the passbands
-----------------------

Passbands are represented by :ref:`filter objects <filters>`. Boxcar filters
are the usual choice for transmission spectroscopy, and named instruments can
be fetched from the SVO Filter Profile Service:

.. code-block:: python

    from ldtk import BoxcarFilter, SVOFilter, tess

    filters = [BoxcarFilter('a', 450, 550),   # wavelengths in nm
               BoxcarFilter('b', 650, 750),
               BoxcarFilter('c', 850, 950)]

    cheops = SVOFilter('CHEOPS/CHEOPS.band')  # any SVO FPS filter name

2. Create the limb darkening profiles
-------------------------------------

:class:`~ldtk.ldtk.LDPSetCreator` takes the stellar parameters (each either as
a ``(value, uncertainty)`` tuple or as an array of posterior samples) and the
filter list. It downloads (and caches) the PHOENIX spectra bracketing the
parameter ranges, integrates them over each passband, and builds an
interpolator over (T\ :sub:`eff`, log g, z):

.. code-block:: python

    from ldtk import LDPSetCreator

    sc = LDPSetCreator(teff=(6400, 50), logg=(4.5, 0.2), z=(0.25, 0.05),
                       filters=filters)
    ps = sc.create_profiles(nsamples=500)

The returned :class:`~ldtk.ldtk.LDPSet` holds ``nsamples`` limb darkening
profile samples per filter, propagating the stellar parameter uncertainties.

3a. Estimate limb darkening coefficients
----------------------------------------

The ``coeffs_*`` methods fit a limb darkening model to the profiles and return
the coefficients and their uncertainties per filter, ready to be used as
priors:

.. code-block:: python

    cq, eq = ps.coeffs_qd(do_mc=True)   # quadratic law, MCMC uncertainties
    cp, ep = ps.coeffs_p2()             # power-2 law, curvature uncertainties

3b. ...or use the log-likelihood directly
-----------------------------------------

The ``lnlike_*`` methods evaluate the log-likelihood of a set of proposed
coefficients against the profiles, which can be added straight into a transit
model's log posterior:

.. code-block:: python

    lnl = ps.lnlike_qd([0.25, 0.05], flt=0)   # one filter, one coefficient set

    lnl = ps.lnlike_qd([[0.45, 0.15],         # one coefficient set per filter,
                        [0.35, 0.10],         # returns the joint likelihood
                        [0.25, 0.05]])

Both families exist for all eight supported limb darkening laws — see
:ref:`likelihoods` for the method list and for how the likelihood is
calculated.

Saving and loading
------------------

A profile set can be pickled and reloaded without re-downloading or
re-integrating anything:

.. code-block:: python

    from ldtk import load_ldpset

    ps.save('star.ldps')
    ps = load_ldpset('star.ldps')
