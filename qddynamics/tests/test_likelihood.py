import unittest
from qddynamics.inference.model import LogLikelihood
from qddynamics.io import get_example_data_file_path, load_data
import pandas as  pd
import numpy as np

class TestLikelihood(unittest.TestCase):
    '''
    Class for testing functionality of the likelihood function
    Loads the real data of x, y, sigma_y, and make sure that for a reasonable theta, likelihood does not diverge
    '''    
    def test_likelihood(self):
        fpath = '..\data_files'
        data = load_data(get_example_data_file_path(fpath+'\simulated_data_csv.csv'))
        x = data.x.values
        y = data.y.values
        sigma_y = data.sigma_y.values
        initial_guess = np.array([2E21,0.5,1E15])
        theta = (10**(-20), 2e15) #C, Nd value in O'Brian et al. 2004
        assert  0<= np.exp(LogLikelihood(theta, x, y, sigma_y, initial_guess).logllh()) < 1
        
    def test_rate_equation_isnan(self):
        '''
        test for the rate equation in the model.py
        give the initial guess and known parameter i (previously solved) and check if it returns meaningful data
        '''
        fpath = '..\data_files'
        data = load_data(get_example_data_file_path(fpath+'\simulated_data_csv.csv'))
        x = data.x.values
        y = data.y.values
        sigma_y = data.sigma_y.values
        initial_guess = np.array([2E21,0.5,1E15])
        theta = (10**(-20), 2e15) #C, Nd value in O'Brian et al. 2004
        i = np.array([1.75368533e-06,1.65518014e-06,1.52930613e-06,1.46383433e-06, 1.40455534e-06,1.33762973e-06,1.28532535e-06,1.22174511e-06,1.14973982e-06,2.95348733e06,-1.25451824e-06,6.21484259e-06,1.11144801e-06,8.73169618e-07,2.90547955e-06 ]) #A test value that we know will return a meaningful value

        assert ~np.isnan(LogLikelihood(theta,x,y,sigma_y,intial_guess).rateEquations(initial_guess, i)).all()


if __name__ == '__main__':
    unittest.main()
