import numpy as np
import numpy.testing as npt
import pytest

from ldtk.ldmodel import (LinearModel, QuadraticModel, TriangularQuadraticModel, SquareRootModel,
                          NonlinearModel, GeneralModel, Power2Model, Power2MPModel, models)

MU = np.array([0.05, 0.25, 0.5, 0.75, 1.0])

GENERIC_COEFFS = {LinearModel: [0.6],
                  QuadraticModel: [0.4, 0.2],
                  TriangularQuadraticModel: [0.36, 0.4],
                  SquareRootModel: [0.3, 0.2],
                  NonlinearModel: [0.2, 0.3, 0.1, 0.05],
                  GeneralModel: [0.4, 0.2, 0.1],
                  Power2Model: [0.6, 0.45],
                  Power2MPModel: [0.35, 0.25]}


@pytest.mark.parametrize('model', list(models.values()), ids=lambda m: m.name)
def test_unit_intensity_at_disk_center(model):
    ldp = model.evaluate(MU, np.array(GENERIC_COEFFS[model]))
    assert ldp[0, 0, -1] == pytest.approx(1.0)


def test_linear():
    npt.assert_allclose(LinearModel.evaluate(MU, np.array([0.0]))[0, 0], np.ones_like(MU))
    npt.assert_allclose(LinearModel.evaluate(MU, np.array([1.0]))[0, 0], MU)
    npt.assert_allclose(LinearModel.evaluate(MU, np.array([0.6]))[0, 0], 1.0 - 0.6 * (1.0 - MU))


def test_quadratic():
    u, v = 0.4, 0.2
    npt.assert_allclose(QuadraticModel.evaluate(MU, np.array([u, v]))[0, 0],
                        1.0 - u * (1.0 - MU) - v * (1.0 - MU) ** 2)


def test_triangular_quadratic_matches_quadratic():
    q1, q2 = 0.36, 0.4
    a, b = np.sqrt(q1), 2.0 * q2
    u, v = a * b, a * (1.0 - b)
    npt.assert_allclose(TriangularQuadraticModel.evaluate(MU, np.array([q1, q2]))[0, 0],
                        QuadraticModel.evaluate(MU, np.array([u, v]))[0, 0])


def test_square_root():
    u, v = 0.3, 0.2
    npt.assert_allclose(SquareRootModel.evaluate(MU, np.array([u, v]))[0, 0],
                        1.0 - u * (1.0 - MU) - v * (1.0 - np.sqrt(MU)))


def test_nonlinear():
    npt.assert_allclose(NonlinearModel.evaluate(MU, np.zeros(4))[0, 0], np.ones_like(MU))
    npt.assert_allclose(NonlinearModel.evaluate(MU, np.array([1.0, 0, 0, 0]))[0, 0], np.sqrt(MU))


def test_general_matches_gimenez_definition():
    pv = np.array([0.4, 0.2, 0.1])
    expected = 1.0 - sum(c * (1.0 - MU ** (i + 1)) for i, c in enumerate(pv))
    npt.assert_allclose(GeneralModel.evaluate(MU, pv)[0, 0], expected)


def test_general_single_term_matches_linear():
    npt.assert_allclose(GeneralModel.evaluate(MU, np.array([0.6]))[0, 0],
                        LinearModel.evaluate(MU, np.array([0.6]))[0, 0])


def test_power2():
    c, alpha = 0.6, 0.45
    npt.assert_allclose(Power2Model.evaluate(MU, np.array([c, alpha]))[0, 0],
                        1.0 - c * (1.0 - MU ** alpha))


def test_power2mp_matches_power2():
    h1, h2 = 0.35, 0.25
    c = 1.0 - h1 + h2
    alpha = np.log2(c / h2)
    npt.assert_allclose(Power2MPModel.evaluate(MU, np.array([h1, h2]))[0, 0],
                        Power2Model.evaluate(MU, np.array([c, alpha]))[0, 0])


def test_evaluate_shape_convention():
    pv1 = np.array([0.4, 0.2])
    pv2 = np.tile(pv1, (3, 1))
    pv3 = np.tile(pv1, (2, 3, 1))
    assert QuadraticModel.evaluate(MU, pv1).shape == (1, 1, MU.size)
    assert QuadraticModel.evaluate(MU, pv2).shape == (1, 3, MU.size)
    assert QuadraticModel.evaluate(MU, pv3).shape == (2, 3, MU.size)
    npt.assert_allclose(QuadraticModel.evaluate(MU, pv3)[1, 2],
                        QuadraticModel.evaluate(MU, pv1)[0, 0])
