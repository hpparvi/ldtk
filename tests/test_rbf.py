import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator

from ldtk.rbf import RBFProfileInterpolator

MU = np.linspace(0.02, 1.0, 40)


def profile(teff, logg, z):
    """A smooth, nonlinear toy model of a limb darkening profile grid."""
    c = 0.9 - 5e-5 * (teff - 3000.0) + 0.02 * (logg - 4.5) + 0.01 * z
    a = 0.4 + 1e-8 * (teff - 3000.0) ** 2 + 0.01 * z
    return 1.0 - c * (1.0 - MU ** a)


def make_grid(missing=()):
    """An irregular grid with optionally removed nodes."""
    teffs = [2900.0, 3000.0, 3100.0, 3300.0, 3600.0]   # non-constant spacing
    loggs = [4.0, 4.5, 5.0]
    zs = [-0.5, 0.0, 0.5]
    points, profiles = [], []
    for t in teffs:
        for g in loggs:
            for z in zs:
                if (t, g, z) in missing:
                    continue
                points.append((t, g, z))
                profiles.append(profile(t, g, z))
    return np.array(points), np.array(profiles)


@pytest.fixture(scope='module')
def interpolator():
    points, profiles = make_grid()
    return RBFProfileInterpolator(points, profiles)


def test_exact_at_nodes(interpolator):
    points, profiles = make_grid()
    np.testing.assert_allclose(interpolator(points), profiles, atol=1e-10)


def test_interpolation_accuracy(interpolator):
    """Off-node predictions must track the true nonlinear profiles."""
    for theta in ((3050.0, 4.7, 0.2), (3200.0, 4.2, -0.3), (3450.0, 4.9, 0.4)):
        pred = interpolator(np.array(theta))[0]
        np.testing.assert_allclose(pred, profile(*theta), atol=5e-3)


def test_more_accurate_than_linear_at_heldout_plane():
    """The RBF interpolant must beat piecewise-linear interpolation when the
    grid has a hole along the nonlinear (teff) direction.

    The whole teff = 3100 plane is removed so that both interpolants must
    bridge the 3000-3300 gap along teff, where the toy profiles are
    nonlinear. (Removing a single node is not enough: the profiles are
    exactly linear in logg, so the piecewise-linear interpolant can recover
    a lone missing node exactly from its logg neighbors.)
    """
    heldout = (3100.0, 4.5, 0.0)
    missing = tuple((3100.0, g, z) for g in (4.0, 4.5, 5.0) for z in (-0.5, 0.0, 0.5))
    points, profiles = make_grid(missing=missing)
    rbf_err = np.abs(RBFProfileInterpolator(points, profiles)(np.array(heldout))[0]
                     - profile(*heldout)).max()
    lin_err = np.abs(LinearNDInterpolator(points, profiles)(np.atleast_2d(heldout))[0]
                     - profile(*heldout)).max()
    assert rbf_err < lin_err


def test_no_nans_outside_hull(interpolator):
    """Unlike the piecewise-linear interpolator, the RBF interpolant must
    return finite values outside the convex hull of the grid nodes."""
    pred = interpolator(np.array([2850.0, 5.2, 0.6]))
    assert np.isfinite(pred).all()


def test_call_shapes(interpolator):
    assert interpolator(np.array([3000.0, 4.5, 0.0])).shape == (1, MU.size)
    theta = np.tile([3000.0, 4.5, 0.0], (7, 1))
    assert interpolator(theta).shape == (7, MU.size)
