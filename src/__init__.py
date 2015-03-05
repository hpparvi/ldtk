import os
from os.path import join

home_dir  = os.environ['HOME']
cache_dir = join(home_dir,'work/Projects/RoPACS/data/phoenix_specint')

try:
    from IPython.display import display, clear_output
    from IPython.html.widgets import IntProgressWidget
    w = IntProgressWidget()
    with_notebook = True
except AttributeError:
    with_notebook = False

from ldtk import LDPSetCreator, LDPSet, quadratic_law
from ldtool.filters import BoxcarFilter, TabulatedFilter

__all__ = ['LDPSetCreator','LDPSet','quadratic_law','BoxcarFilter','TabulatedFilter']
