from qddynamics.inference.model import UniformPrior, JefferysPrior

import numpy as np

import unittest
from unittest import TestCase
import nose 


class TestPriors_value(TestCase):
    """
    Class for testing functionality of the prior function works, giving the right values
    """
    def test_uniform(self):
        assert np.allclose(np.exp(UniformPrior(3, 5).logp(4)), .5)

    def test_jefferys(self):
        assert np.allclose(np.exp(JefferysPrior(10, 1000).logp(100)),
                           0.0021714724095162588)

class TestPriors_limit(Testcase):
    """
    Class for testing the limit of the prior classes.
    See if inserting the value that is off from limits still makes sense.
    """
    def test_uniform_limit(self):
        assert np.allclose(np.exp(UniformPrior(3,5).logp(8)), 0)
    def test_jefferys_limit(self):
        assert np.allclose(np.exp(JeffreysPrior(10,1000).logp(1)), 0)
        



if __name__ == '__main__':
#    unittest.main()
    nose.main()