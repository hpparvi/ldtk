.. _api:
..module:: ldtk

API
===

Client
------

.. module:: ldtk.client
.. autoclass:: ldtk.client.Client
   :members:

Filters
-------

.. module:: ldtk.filters
.. autoclass:: ldtk.filters.Filter
   :members:

.. autoclass:: ldtk.filters.BoxcarFilter
   :members:

.. autoclass:: ldtk.filters.TabulatedFilter
   :members:

Pre-defined filters
^^^^^^^^^^^^^^^^^^^

.. autodata:: ldtk.filters.sdss_g
.. autodata:: ldtk.filters.sdss_r
.. autodata:: ldtk.filters.sdss_i
.. autodata:: ldtk.filters.sdss_z
.. autodata:: ldtk.filters.kepler
   :annotation: Blah

Limb darkening profile set
--------------------------

.. module:: ldtk.ldtk
.. autoclass:: ldtk.ldtk.LDPSet
   :members:

.. autoclass:: ldtk.ldtk.LDPSetCreator
   :members:


Limb darkening models
---------------------

.. module:: ldtk.ldmodel
.. automodule:: ldtk.ldmodel
   :members:
