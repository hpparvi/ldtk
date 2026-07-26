# LDTk documentation build configuration.
#
# Full list of options: https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import version as get_version

# -- Project information ---------------------------------------------------

project = 'LDTk'
author = 'Hannu Parviainen'
copyright = '2017-2026, Hannu Parviainen'

release = get_version('LDTk')
version = '.'.join(release.split('.')[:2])

# -- General configuration -------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'myst_parser',
    'numpydoc',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

root_doc = 'index'
language = 'en'
templates_path = ['_templates']
exclude_patterns = []

numpydoc_show_class_members = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

# -- Options for HTML output -----------------------------------------------

html_theme = 'sphinx_book_theme'
html_title = f'LDTk {release}'
html_static_path = ['_static']

html_theme_options = {
    'repository_url': 'https://github.com/hpparvi/ldtk',
    'use_repository_button': True,
    'use_issues_button': True,
    'path_to_docs': 'docs/source',
}
