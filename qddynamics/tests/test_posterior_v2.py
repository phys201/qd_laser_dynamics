import unittest
from qddynamics.inference.model import Posterior
from qddynamics.io import get_example_data_file_path, load_data
import pandas as  pd
import numpy as np


class TestPosterior_v2(unittest.TestCase):
    '''
    Class for testing functionality of the likelihood function
    Loads the real data of x, y, sigma_y, and make sure that for a reasonable theta, likelihood does not diverge
    '''

    def test_log_posterior(self):
        '''
        Tests functionality of our log_posterior function
        '''
        fpath = '..\data_files'
        data = load_data(get_example_data_file_path(fpath+'\simulated_data_csv.csv'))
        x = data.x.values
        #Posterior.y = data.y.values
        y = data.y.values
        sigma_y = data.sigma_y.values
        initial_guess = np.array([2E21,0.5,1E15])
        g0_bounds = (0.15E-13, 0.15E-7)
        CNg0_exp = (10e-20, 1e14, 0.15E-10)
        C_bounds = (10e-21, 10e-19)
        Nd_bounds = (1E13, 1E15)
        
        ls_result = [0.8e-20, 1.5E14, 0.15E-10]
        theta = (10**(-20), 2e15, 0.15E-10) #C, Nd value in O'Brian et al. 2004
        logpost = Posterior(initial_guess, C_bounds, Nd_bounds, g0_bounds, CNg0_exp)
        post = logpost.log_posterior(theta, x, y, sigma_y)
        assert ~np.isnan(post)
        assert -3000 < post < -2000

    def test_parameters_C(self):
        '''
        Tests to see if the parameter inference on "C" returns a reasonable
        value.
        '''
        fpath = '..\data_files'
        data = load_data(get_example_data_file_path(fpath+'\simulated_data_csv.csv'))
        x = data.x.values
        #Posterior.y = data.y.values
        y = data.y.values
        sigma_y = data.sigma_y.values
        initial_guess = np.array([2E21,0.5,1E15])
        g0_bounds = (0.15E-13, 0.15E-7)
        CNg0_exp = (10e-20, 1e14, 0.15E-10)
        C_bounds = (10e-21, 10e-19)
        Nd_bounds = (1E13, 1E15)
        
        ls_result = [0.8e-20, 1.5E14, 0.15E-10]
        theta = (10**(-20), 2e15, 0.15E-10) #C, Nd value in O'Brian et al. 2004
        logpost = Posterior(initial_guess, C_bounds, Nd_bounds, g0_bounds, CNg0_exp)
        param_chain = logpost.mc(x,y, sigma_y, ls_result, nwalkers = 7, nsteps = 300, plot_chains=False)
        params =  logpost.extract_parameters(param_chain[0])
        assert 5e-21 <= params['C'][0.5] <= 10e-21
        assert 7e13 <= params['Nd'][0.5] <= 15e13
        assert 0.15E-13 <= params['g0'][0.5]<=0.15E-7
        assert logpost.plot_parameters(param_chain[0]) == 'C'
        

if __name__ == '__main__':
    unittest.main()
