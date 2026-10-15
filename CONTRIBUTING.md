# Contributing to LDTk

Thanks for your interest in LDTk! Bug reports, questions, documentation fixes
and code are all welcome. This guide covers how to report problems, set up a
development checkout, and get a pull request merged.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs and asking questions

Open an issue on [GitHub](https://github.com/hpparvi/ldtk/issues). A good bug
report includes:

- the LDTk version (`python -c "import ldtk; print(ldtk.__version__)"`), plus
  your Python, NumPy, SciPy, Numba and Astropy versions;
- a minimal, self-contained example that reproduces the problem;
- what you expected and what you got instead.

For problems with the limb darkening profiles or coefficients, include the
stellar parameters (`teff`, `logg`, `z` and their uncertainties), the filters
(their type and wavelength range, or the SVO filter ID), the `dataset`, and
any non-default `interpolation`, `likelihood`, `cev` or `nk` settings. 

For problems with downloading spectra, include the full traceback and say
whether you set `LDTK_ROOT` or used `offline_mode=True`. The spectra come from
the Göttingen PHOENIX server, so a failed download is sometimes caused by the
server and not by LDTk.

## Development setup

```bash
git clone https://github.com/hpparvi/ldtk.git
cd ldtk
pip install -e ".[test]"
```

Add the `docs` extra to build the documentation:
`pip install -e ".[test,docs]"`.

The package version comes from the git tags through `setuptools-scm`, so make
a full clone that includes the tags (not `--depth 1`). The generated
`ldtk/_version.py` is ignored by git. Never edit or commit it, and never
hardcode a version anywhere.

## Running the tests

```bash
pytest tests                                  # everything
pytest tests/test_ld_models.py::test_quadratic  # a single test
```

The test suite does not download any spectra. The likelihood tests build an
`LDPSet` from synthetic profiles with `create_synthetic_ldpset()` in
`tests/test_loglikelihood.py`. New tests should do the same, since CI has no
spectrum cache.

To measure coverage (with `pytest-cov` installed), disable the JIT, because
the tracer cannot see inside compiled Numba kernels:

```bash
NUMBA_DISABLE_JIT=1 pytest tests --cov=ldtk
```

## Making changes

### API stability

The stable public API is:

- the names exported from `ldtk/__init__.py`: `LDPSetCreator`, `LDPSet`,
  `load_ldpset`, `ReducedRankLL`, the filter classes, and the predefined
  filters;
- the `coeffs_*` and `lnlike_*` methods of `LDPSet`, and its sampling and
  configuration methods;
- the limb darkening model classes in `ldtk.ldmodel`.

Breaking changes to these are possible when they clearly improve clarity or
usability, but they must be recorded under a **Breaking** note in
[`CHANGES.md`](CHANGES.md). Names with a leading underscore are private.

`LDPSet.save()` writes a pickle that users keep between sessions. A change to
what it stores must keep `load_ldpset()` able to read files saved by earlier
versions.

### Numba kernels

The hot numerical paths (the `ld_*` model kernels, `evaluate_ld`, and the
diagonal likelihood kernels `lnlike1d/2d/3d`) are `@njit` module-level
functions, kept separate from the object-oriented layer. Keep them
Numba-compatible: no Python objects, only typed arrays. Note that
`evaluate_ld` always returns a 3D array `[ipv, ipb, imu]`, also for 1D and 2D
coefficient input, so index it before passing slices on.

### Adding a limb darkening model

1. Add an `LDModel` subclass and its Numba kernel to `ldtk/ldmodel.py`, and
   register it in the `models` dict.
2. Add a description of the model to `_MODEL_DESCRIPTIONS` in `ldtk/ldtk.py`.
3. Add one `coeffs_xx = _coeffs_method(...)` and one
   `lnlike_xx = _lnlike_method(...)` line to the `LDPSet` class body.
4. Add the model to the documentation and to the tests in
   `tests/test_ld_models.py`.

### Testing numerical code

- Compare new code against an independent reference: closed-form profiles for
  the limb darkening models, and `scipy.stats.multivariate_normal` for the
  likelihoods.
- Add a tiny noise floor to synthetic profiles. They are normalised to unity
  at the disk centre, so the sample variance there is otherwise exactly zero
  and the diagonal likelihood becomes infinite.
- Test multi-filter cases as well as single-filter ones. At least one past
  bug only showed up with more than one filter.
- Mutation-test new code: flip a sign or perturb a coefficient, and check that
  a test fails.

### Code and documentation style

- Docstrings follow the NumPy style (numpydoc).
- Wavelengths are in nanometres throughout. The PHOENIX FITS headers store
  them in Ångström, and they are converted when the files are read.
- The code must run on Python 3.8, the oldest version tested in CI.

Build the documentation with `make -C docs html`; the output goes to
`docs/_build/html/`.

## Pull requests

- Keep each pull request focused on one change, and describe what it does and
  why.
- Add or update tests for any behaviour you change, and make sure
  `pytest tests` passes. CI runs the tests on Python 3.8 to 3.13.
- Add an entry under `[Unreleased]` in [`CHANGES.md`](CHANGES.md) for
  user-visible changes. The changelog follows
  [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
- Update the documentation when you change public behaviour. The narrative
  docs contain examples that are not checked by the build.
- Do not bump the version. Releases are made by the maintainer by tagging
  `vX.Y.Z`.

## Generative AI policy

Contributions developed with LLMs are welcome, but each one needs a human
author who oversees the work and takes responsibility for it. Purely agentic
contributions, made without human oversight, are not accepted.

## License

LDTk is released under the GNU General Public License v2.0 or later. By
contributing, you agree that your contributions are licensed under the same
terms.
