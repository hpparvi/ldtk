Installation
============

LDTk can be installed from PyPI with

.. code-block:: bash

    pip install ldtk

or, for development, cloned from GitHub and installed in editable mode

.. code-block:: bash

    git clone https://github.com/hpparvi/ldtk.git
    cd ldtk
    pip install -e ".[test]"     # test extra installs pytest
    pip install -e ".[docs]"     # docs extra installs Sphinx and the theme

The core dependencies (numpy, scipy, numba, astropy, and others) are installed
automatically. Using :class:`~ldtk.filters.SVOFilter` additionally requires
`astroquery <https://astroquery.readthedocs.io/>`_.

Spectrum cache
--------------

LDTk downloads the PHOENIX specific intensity spectra it needs from the
Göttingen spectrum server on first use and caches them locally. The cache
location defaults to ``~/.ldtk`` and can be changed either by setting the
``LDTK_ROOT`` environment variable or by giving the ``cache`` argument to
:class:`~ldtk.ldtk.LDPSetCreator`. Each spectrum :ref:`dataset <datasets>` has
its own cache subdirectory.

Downloads are retried on failure, and the FITS files are verified with
checksums: corrupted files are deleted and re-fetched automatically. Once the
needed spectra are cached, LDTk can be used without a network connection by
passing ``offline_mode=True`` to :class:`~ldtk.ldtk.LDPSetCreator`.

Running the tests
-----------------

.. code-block:: bash

    pytest tests

The test suite uses synthetic limb darkening profiles and does not download
any spectra.
