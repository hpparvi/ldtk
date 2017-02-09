from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration
import distutils.sysconfig as ds

long_description = ''

setup(name='LDTk',
      version='1.0-beta.4',
      description='Toolkit to calculate stellar limb darkening profiles for arbitrary filters.',
      long_description=long_description,
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='https://github.com/hpparvi/ldtk',
      package_dir={'ldtk':'src'},
      packages=['ldtk'],
      install_requires=["numpy","scipy>=0.16","astropy","tqdm","traitlets"],
      license='GPLv2',
      classifiers=[
          "Topic :: Scientific/Engineering",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3"
      ]
     )
