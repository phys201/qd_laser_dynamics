import unittest
from example.inference.model import LogLikelihood
from data_files.io import get_example_data_file_path, load_data
import pandas as  pd
import numpy as np

class TestLikelihood(unittest.TestCase):
    '''
    Class for testing functionality of the likelihood function
    Loads the real data of x, y, sigma_y, and make sure that for a reasonable theta, likelihood does not diverge
    '''
    def test_likelihood(self):
        data = load_data(get_example_data_file_path('simulated_data_csv.csv'))
        x = data.x.values
        y = data.y.values
        sigma_y = data.sigma_y.values
        theta = 10**(-20) #C value in O'Brian et al. 2004
        assert  0<= np.exp(LogLikelihood(theta, x, y, sigma_y).logllh()) < 1


if __name__ == '__main__':
    unittest.main()
