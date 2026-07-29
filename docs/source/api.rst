.. _api:

API
===

Profile creation
----------------

.. autoclass:: ldtk.ldtk.LDPSetCreator
   :members:

Limb darkening profile set
--------------------------

.. autoclass:: ldtk.ldtk.LDPSet
   :members:

.. autofunction:: ldtk.ldtk.load_ldpset

Log-likelihood
--------------

.. autoclass:: ldtk.loglikelihood.ReducedRankLL
   :members:
   :special-members: __call__

Grid interpolation
------------------

.. autoclass:: ldtk.rbf.RBFProfileInterpolator
   :members:
   :special-members: __call__

Filters
-------

.. autoclass:: ldtk.filters.Filter
   :members:

.. autoclass:: ldtk.filters.BoxcarFilter
   :members:

.. autoclass:: ldtk.filters.TabulatedFilter
   :members:

.. autoclass:: ldtk.filters.SVOFilter
   :members:

.. autoclass:: ldtk.filters.DeltaFilter
   :members:

Pre-defined filters
^^^^^^^^^^^^^^^^^^^

.. autodata:: ldtk.filters.sdss_g
.. autodata:: ldtk.filters.sdss_r
.. autodata:: ldtk.filters.sdss_i
.. autodata:: ldtk.filters.sdss_z
.. autodata:: ldtk.filters.kepler
.. autodata:: ldtk.filters.tess

Limb darkening models
---------------------

.. automodule:: ldtk.ldmodel
   :members:
   :exclude-members: evaluate

Client
------

.. autoclass:: ldtk.client.Client
   :members:
