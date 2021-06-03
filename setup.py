from setuptools import setup, find_packages

version = {}
with open("ldtk/version.py") as fp:
    exec(fp.read(), version)

setup(name='LDTk',
      version=str(version['__version__']),
      description='Toolkit to calculate stellar limb darkening profiles for arbitrary filters.',
      long_description='Toolkit to calculate stellar limb darkening profiles for arbitrary filters.',
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='https://github.com/hpparvi/ldtk',
      packages=find_packages(),
      install_requires=["numpy", "numba", "scipy>=0.16", "astropy", "tqdm", "traitlets", "tenacity", 
                        "semantic_version", "pathlib~=1.0 ; python_version<='3.3'"],
      include_package_data=True,
      package_data={'ldtk': ['filter_files/*','filter_files/gtc/osiris/*']},
      license='GPLv2',
      classifiers=[
          "Topic :: Scientific/Engineering",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 3"
      ],
      keywords= 'astronomy astrophysics exoplanets'
     )
