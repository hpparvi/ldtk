from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration
import distutils.sysconfig as ds

#with open('README.rst') as file:
#    long_description = file.read()

long_description = ''

setup(name='LDTool',
      version='0.5',
      description='Tool to calculate stellar limb darkening profiles for arbitrary filters.',
      long_description=long_description,
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='https://github.com/hpparvi/LDTool',
      package_dir={'ldtool':'src'},
      packages=['ldtool'],
      install_requires=["numpy"],
      license='GPLv2',
      classifiers=[
          "Topic :: Scientific/Engineering",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
          "Operating System :: OS Independent",
          "Programming Language :: Python"
      ]
     )
