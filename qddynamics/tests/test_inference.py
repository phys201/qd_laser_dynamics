from example.inference.model import UniformPrior, JefferysPrior

import numpy as np

import unittest
from unittest import TestCase
import nose 


class TestPriors(TestCase):
    """
    Class for testing functionality of the prior function works
    """
    def test_uniform(self):
        assert np.allclose(np.exp(UniformPrior(3, 5).logp(4)), .5)

    def test_jefferys(self):
        assert np.allclose(np.exp(JefferysPrior(10, 1000).logp(100)),
                           0.0021714724095162588)

if __name__ == '__main__':
#    unittest.main()
    nose.main()