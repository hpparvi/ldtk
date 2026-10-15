# Changelog

All notable changes to LDTk (the Limb Darkening Toolkit) will be documented in
this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Smooth RBF grid interpolation with `LDPSetCreator(..., interpolation='rbf')`.
- Log-likelihood tests and a `test` install extra.

### Changed

- **The default likelihood is now `'reduced-rank'`**, giving wider and more
  realistic coefficient uncertainties. Use `likelihood='diagonal'` for the
  old behavior.
- `set_uncertainty_multiplier` is now optional rather than a required
  correction.
- The server file list is stored in the custom cache directory when one is
  given, for systems where `$LDTK_ROOT` is not writable (#33).
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

- `SVOFilter` bugfix.

## [1.7.0] - 2021-07-22

### Added

- `LDPSet` now locates the stellar limb during initialization with the new
  `fit_limb` method (an LD model fitted together with a smootherstep edge
  function), replacing the manual limb definition.

### Changed

- Model spectra are resampled to linear sampling in mu by default.

## [1.6.3] - 2021-06-03

### Fixed

- `Power2MPModel` used `log` instead of `log2` (#23).
- LDTk now works on Windows: `LDTK_ROOT` handling supports Windows paths and
  missing environment variables (#22, #24).

Releases before 1.6.3 predate this changelog and are not documented here.

[Unreleased]: https://github.com/hpparvi/ldtk/compare/v1.8.4...HEAD
[1.8.4]: https://github.com/hpparvi/ldtk/compare/v1.7.0...v1.8.4
[1.7.0]: https://github.com/hpparvi/ldtk/compare/v1.6.3...v1.7.0
[1.6.3]: https://github.com/hpparvi/ldtk/compare/v1.6...v1.6.3
