import os
from os.path import join, exists

ldtk_root  = os.getenv('LDTK_ROOT') or join(os.getenv('HOME'),'.ldtk')
ldtk_cache = join(ldtk_root,'cache')

if not exists(ldtk_root):
    os.mkdir(ldtk_root)
if not exists(ldtk_cache):
    os.mkdir(ldtk_cache)

from ldtk import LDPSetCreator, LDPSet
from ldtool.filters import BoxcarFilter, TabulatedFilter

__all__ = ['LDPSetCreator','LDPSet','BoxcarFilter','TabulatedFilter']
