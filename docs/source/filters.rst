.. _filters:

Filters
=======

Filters define the passbands the limb darkening profiles are integrated over.
All wavelengths in LDTk are given in nanometers. Every filter implements
``integrate(wavelengths, values)``, so custom filters can be added by
subclassing :class:`~ldtk.filters.Filter`.

Boxcar filters
--------------

:class:`~ldtk.filters.BoxcarFilter` has a uniform response between two
wavelengths and zero outside. This is the standard choice for transmission
spectroscopy, where the light curves are modeled for a set of narrow
wavelength bins:

.. code-block:: python

    from ldtk import BoxcarFilter

    bins = [BoxcarFilter(f'{wlc}', wlc - 10, wlc + 10)
            for wlc in range(550, 751, 20)]

Tabulated filters
-----------------

:class:`~ldtk.filters.TabulatedFilter` takes wavelength and transmission
arrays (or the name of a file to read them from):

.. code-block:: python

    from ldtk import TabulatedFilter

    flt = TabulatedFilter('example',
                          wl_or_fname=[400, 500, 600, 700],
                          tm=[0.2, 0.7, 0.9, 0.1])

SVO Filter Profile Service filters
----------------------------------

:class:`~ldtk.filters.SVOFilter` fetches a named instrument response from the
`Spanish Virtual Observatory Filter Profile Service
<http://svo2.cab.inta-csic.es/theory/fps/>`_ (over 10000 filters) using
`astroquery`:

.. code-block:: python

    from ldtk import SVOFilter

    flt = SVOFilter('CHEOPS/CHEOPS.band')

Delta filters
-------------

:class:`~ldtk.filters.DeltaFilter` selects a single wavelength, which can be
useful for testing and for approximating very narrow bins.

Predefined filters
------------------

LDTk ships ready-made instances for common passbands: ``sdss_g``, ``sdss_r``,
``sdss_i``, ``sdss_z`` (boxcar approximations), and tabulated ``kepler`` and
``tess`` response functions:

.. code-block:: python

    from ldtk import sdss_g, sdss_r, sdss_i, sdss_z, kepler, tess
