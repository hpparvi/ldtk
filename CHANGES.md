# Changelog

All notable changes to LDTk (the Limb Darkening Toolkit) will be documented in
this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LDPSetCreator.create_profiles` accepts the `likelihood`, `cev`, and `nk`
  arguments and passes them on to the `LDPSet`.
- A changelog page in the documentation.

### Changed

- The `coeffs_*` and `lnlike_*` methods are now regular `LDPSet` methods with
  full docstrings, so they appear in the API documentation and in `help()`.
- `astroquery` is now a required dependency, so `SVOFilter` works without a
  separate install.
- LDTk now declares Python 3.8 as its minimum version.
- Updated the README to reflect the current state of LDTk.

### Removed

- The unused `traitlets` dependency.

## [1.9.2] - 2026-10-15

### Added

- This changelog.

### Fixed

- Importing LDTk no longer silences all warnings in the process, only the
  Astropy warnings raised when reading the PHOENIX files.

## [1.9.1] - 2026-08-05

No code changes from 1.9.0; merges the `master` branch history into the
release.

## [1.9.0] - 2026-08-05

### Added

- Reduced-rank log-likelihood that accounts for the correlations between mu
  points (Tegmark et al. 1997). Configurable with the `LDPSet` arguments
  `likelihood`, `cev`, and `nk`, or with `set_likelihood_mode()`.
- `LDPSet.diagnostics()` for inspecting the reduced-rank decomposition.
- Smooth radial basis function (RBF) interpolation over the model grid,
  selected with the `LDPSetCreator` argument `interpolation`.
- Log-likelihood tests and a `test` install extra.

### Changed

- **The default likelihood is now `'reduced-rank'`**, giving wider and more
  realistic coefficient uncertainties. Use `likelihood='diagonal'` for the
  old behavior.
- **The default grid interpolation is now `'rbf'`.** Use
  `interpolation='linear'` for the old piecewise-linear interpolation.
- `set_uncertainty_multiplier` is now optional rather than a required
  correction.
- Rewrote the documentation.
- Migrated the tests to pytest.
- The package version is now derived from git tags with `setuptools-scm`.

### Removed

- The `semantic_version` dependency.

### Fixed

- `GeneralModel` was missing its leading `1 −` term, giving wrong
  `coeffs_ge`/`lnlike_ge` results.
- Multi-filter log-likelihoods with 2D coefficient arrays used only the first
  filter's profile.

## [1.8.6] - 2026-06-04

### Changed

- The server file list is stored in the custom cache directory when one is
  given, for systems where `$LDTK_ROOT` is not writable (#33).

### Fixed

- `LDPSet` creation no longer fails when the profile interpolator returns
  NaNs: parameter samples falling outside the convex hull of the available
  PHOENIX models (e.g. hot, low-gravity stars missing from the library) are
  dropped with a message, and a clear error is raised if no samples remain.

## [1.8.5] - 2024-11-07

### Changed

- Switched to a modern `pyproject.toml`-based build system (setuptools
  backend).
- Improved boxcar filter integration.

## [1.8.4] - 2023-12-26

### Added

- Citation metadata (`CITATION.cff`) with a preferred journal citation.

### Fixed

- `SVOFilter` raised `NotImplementedError` since 1.8.1 because it lacked its
  own `integrate` method.

## [1.8.3] - 2022-11-25

### Changed

- Improved limb darkening coefficient fitting.

## [1.8.2] - 2022-11-18

### Changed

- Performance improvements.

## [1.8.1] - 2022-10-13

### Added

- `DeltaFilter` for single-wavelength passbands.

### Changed

- Filters now integrate the spectra themselves through `Filter.integrate`.

## [1.7.1] - 2022-06-15

### Fixed

- Resampling an `LDPSet` now also resamples the individual profile samples.

## [1.7.0] - 2021-07-22

### Added

- `LDPSet` now locates the stellar limb during initialization with the new
  `fit_limb` method (an LD model fitted together with a smootherstep edge
  function), replacing the manual limb definition.

### Changed

- Model spectra are resampled to linear sampling in mu by default.

## [1.6.4] - 2021-06-21

### Fixed

- `SVOFilter` returns zero transmission outside its wavelength range instead
  of raising an error.

## [1.6.3] - 2021-06-03

### Fixed

- `Power2MPModel` used `log` instead of `log2` (#23).
- More robust `LDTK_ROOT` handling on Windows and when the environment
  variable is not set.

## [1.6.2] - 2021-06-03

### Fixed

- LDTk now works on Windows (#22, #24).

## [1.6.1] - 2021-05-21

### Fixed

- `LDPSetCreator` failed with its default `dataset` value.

## [1.6.0] - 2021-05-04

### Added

- `SVOFilter`, which creates a filter from its name in the
  [SVO Filter Profile Service](http://svo2.cab.inta-csic.es/theory/fps/).

## [1.5.0] - 2021-03-03

### Added

- Four spectrum libraries selectable with the `LDPSetCreator` argument
  `dataset`: `vis`, `vis-lowres` (default), `visir`, and `visir-lowres`. The
  `visir` sets extend the wavelength range from 2.6 µm to 5.5 µm.

### Deprecated

- The `lowres` argument; it now raises an error pointing to `dataset`.

## [1.4.1] - 2020-08-20

### Fixed

- FITS files are now closed properly after reading.
- `mpi4py` bugfix.

## [1.4.0] - 2020-08-14

### Added

- Corrupted FITS files are detected and re-downloaded automatically, and
  downloads are retried on failure (contributed by T. Fatahi).

## [1.3.1] - 2020-08-14

### Changed

- Low-resolution spectra are again the default.

## [1.3.0] - 2020-07-31

### Changed

- All limb darkening models are computed with numba, which is considerably
  faster.
- Original-resolution spectra are the default again.

## [1.2.0] - 2020-07-21

### Added

- Support for low-resolution spectra binned to 5 nm, toggled with the new
  `LDPSetCreator` argument `lowres`. **Low-resolution spectra are the
  default**, which greatly reduces download times and storage space.

## [1.1.1] - 2020-06-03

### Fixed

- Packaging: the version was passed to `setup` as a `Version` object instead
  of a string.

## [1.1.0] - 2020-06-02

### Added

- `TriangularQuadraticModel`, the quadratic model with the triangular
  parametrization of Kipping (2013).
- `Power2Model`, the power-2 model of Morello et al. (2017) (contributed by
  K. Hay), and `Power2MPModel`, its alternative parametrization by Maxted
  (2018).
- Predefined TESS filter (`tess`).

### Changed

- The quadratic, triangular quadratic, nonlinear, and power-2 models and the
  log-likelihoods are computed with numba.
- Passband averages are photon-weighted by default.

## [1.0] - 2017-09-18

First stable release.

### Added

- Predefined filters `sdss_g`, `sdss_r`, `sdss_i`, `sdss_z`, and `kepler`.
- `load_ldpset` is exported from the package, and the package has a version
  number.

### Changed

- Improved `TabulatedFilter`.

### Fixed

- Python 3 compatibility.

## 1.0 beta releases - 2016-01-14 to 2017-05-19

The first public releases, from 1.0-beta to 1.0-beta.5 (1.0-beta.5 was never
uploaded to PyPI), accompanying the LDTk paper (Parviainen & Aigrain 2015).

- `LDPSetCreator` downloads and caches the PHOENIX spectra, and interpolates
  the limb darkening profiles over Teff, log g, and metallicity. Stellar
  parameters can be given as a value and uncertainty or as posterior samples
  (contributed by I. Crossfield).
- `LDPSet` estimates linear, quadratic, nonlinear, and general model
  coefficients, with uncertainties from numerical derivatives or MCMC, and
  evaluates their log-likelihoods. Profile sets can be resampled, their limb
  redefined, and saved and loaded.
- `BoxcarFilter` and `TabulatedFilter` passbands.
- Square-root model and a nonlinear model fix (contributed by R. Wichmann).
- Offline mode, MPI-aware downloads, and initial Python 3 support.

[Unreleased]: https://github.com/hpparvi/ldtk/compare/v1.9.2...HEAD
[1.9.2]: https://github.com/hpparvi/ldtk/compare/v1.9.1...v1.9.2
[1.9.1]: https://github.com/hpparvi/ldtk/compare/v1.9.0...v1.9.1
[1.9.0]: https://github.com/hpparvi/ldtk/compare/v1.8.4...v1.9.0
[1.8.4]: https://github.com/hpparvi/ldtk/compare/v1.7.0...v1.8.4
[1.7.0]: https://github.com/hpparvi/ldtk/compare/v1.6.3...v1.7.0
[1.6.3]: https://github.com/hpparvi/ldtk/compare/v1.6...v1.6.3
[1.6.0]: https://github.com/hpparvi/ldtk/compare/v1.4.0...v1.6
[1.4.0]: https://github.com/hpparvi/ldtk/compare/v1.3.1...v1.4.0
[1.3.1]: https://github.com/hpparvi/ldtk/compare/v1.0...v1.3.1
[1.0]: https://github.com/hpparvi/ldtk/compare/v1.0-beta.5...v1.0
