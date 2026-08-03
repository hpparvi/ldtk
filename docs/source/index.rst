.. module:: ldtk

Limb Darkening Toolkit
======================

LDTk automates the calculation of custom stellar limb darkening (LD) profiles
and model-specific limb darkening coefficients (LDC) using the library of
PHOENIX-generated specific intensity spectra by `Husser et al. (2013)
<https://ui.adsabs.harvard.edu/abs/2013A%26A...553A...6H/>`_.

The aim of the package is to facilitate exoplanet transit light curve modeling,
especially transmission spectroscopy where the modeling is carried out for
custom narrow passbands. The package can be

1. used to construct model-specific priors on the limb darkening coefficients
   prior to the transit light curve modeling, or
2. directly integrated into the log posterior computation of any pre-existing
   transit modeling code with minimal modifications.

The second approach constrains the LD model parameter space directly by the LD
profile, allowing for marginalization over the whole parameter space that can
explain the profile without the need to approximate this constraint by a prior
distribution. This is useful when using a high-order limb darkening model where
the coefficients are often correlated, and priors estimated from tabulated
values usually fail to include these correlations.

Example
-------

.. code-block:: python

    from ldtk import LDPSetCreator, BoxcarFilter

    filters = [BoxcarFilter('a', 450, 550),  # Define your passbands
               BoxcarFilter('b', 650, 750),  # - Boxcar filters useful in
               BoxcarFilter('c', 850, 950)]  #   transmission spectroscopy

    sc = LDPSetCreator(teff=(6400,   50),    # Define your star, and the code
                       logg=(4.50, 0.20),    # downloads the uncached stellar
                          z=(0.25, 0.05),    # spectra from the Husser et al.
                         filters=filters)    # server automatically.

    ps = sc.create_profiles()                # Create the limb darkening profiles
    cq, eq = ps.coeffs_qd(do_mc=True)        # Estimate quadratic law coefficients

    lnl = ps.lnlike_qd([[0.45, 0.15],        # Calculate the quadratic law log
                        [0.35, 0.10],        # likelihood for a set of coefficients,
                        [0.25, 0.05]])       # one per filter (joint likelihood)

    lnl = ps.lnlike_qd([0.25, 0.05], flt=0)  # Quadratic law log L for one filter

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   filters
   profiles
   likelihoods
   api

Citing LDTk
-----------

If you use LDTk in your research, please cite

    Parviainen, H. & Aigrain, S. (2015). *ldtk: Limb Darkening Toolkit.*
    MNRAS, 453(4), 3821–3826. `doi:10.1093/mnras/stv1857
    <https://doi.org/10.1093/mnras/stv1857>`_

and the paper describing the PHOENIX spectrum library the profiles are based on

    Husser, T.-O., et al. (2013). *A new extensive library of PHOENIX stellar
    atmospheres and synthetic spectra.* A&A, 553, A6.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
