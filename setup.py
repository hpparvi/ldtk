from setuptools import setup, find_packages

setup(name='LDTk',
      version='1.0',
      description='Toolkit to calculate stellar limb darkening profiles for arbitrary filters.',
      long_description='Toolkit to calculate stellar limb darkening profiles for arbitrary filters.',
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='https://github.com/hpparvi/ldtk',
      packages=find_packages(),
      install_requires=["numpy", "scipy>=0.16", "astropy", "tqdm", "traitlets"],
      include_package_data=True,
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
      ],
      keywords= 'astronomy astrophysics exoplanets'
     )
