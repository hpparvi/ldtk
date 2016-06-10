import unittest
import numpy as np
import numpy.testing as npt

from ldtk.ld_models import *

class TestModels(unittest.TestCase):
    """Test the limb darkening models.
    """
    def setUp(self):
        self.mu = np.array([0, 0.1, 0.5, 1.])

        
    def test_linear(self):
        npt.assert_array_almost_equal(LinearModel.evaluate([0.0, 0.5, 1.0], [0]), [1.0, 1.0, 1.0])
        npt.assert_array_almost_equal(LinearModel.evaluate([0.0, 0.5, 1.0], [1]), [0.0, 0.5, 1.0])

        
    def test_quadratic(self):
        npt.assert_array_almost_equal(QuadraticModel.evaluate(0., [0,0]), 1.)
        npt.assert_array_almost_equal(QuadraticModel.evaluate(0., [1,0]), 0.)
        npt.assert_array_almost_equal(QuadraticModel.evaluate(1., [0,0]), 1.)
        npt.assert_array_almost_equal(QuadraticModel.evaluate(1., [1,0]), 1.)

        
    def test_nonlinear(self):
        npt.assert_array_almost_equal(NonlinearModel.evaluate(0., [0,0,0,0]), 1.)
        npt.assert_array_almost_equal(NonlinearModel.evaluate(0., [1,0,0,0]), 0.)
        npt.assert_array_almost_equal(NonlinearModel.evaluate(1., [0,0,0,0]), 1.)
        npt.assert_array_almost_equal(NonlinearModel.evaluate(1., [1,0,0,0]), 1.)

        
    def test_general(self):
        npt.assert_array_almost_equal(GeneralModel.evaluate(0., [0]), 1.)
        npt.assert_array_almost_equal(GeneralModel.evaluate(0., [1]), 0.)
        npt.assert_array_almost_equal(GeneralModel.evaluate(1., [0]), 1.)
        npt.assert_array_almost_equal(GeneralModel.evaluate(1., [1]), 1.)
