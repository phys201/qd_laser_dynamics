import unittest
from qddynamics.inference.model import LogLikelihood
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
        initial_guess = np.array([2E21,0.5,1E15])
        theta = (10**(-20), 2e15) #C, Nd value in O'Brian et al. 2004
        print(LogLikelihood(theta, x, y, sigma_y, initial_guess).logllh())
        assert  0<= np.exp(LogLikelihood(theta, x, y, sigma_y, initial_guess).logllh()) < 1


if __name__ == '__main__':
    unittest.main()