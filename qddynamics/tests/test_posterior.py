import unittest
from qddynamics.inference.model import Posterior
from data_files.io import get_example_data_file_path, load_data
import pandas as  pd
import numpy as np


class TestPosterior(unittest.TestCase):
    '''
    Class for testing functionality of the likelihood function
    Loads the real data of x, y, sigma_y, and make sure that for a reasonable theta, likelihood does not diverge
    '''

    def test_log_posterior(self):
        data = load_data(get_example_data_file_path('simulated_data_csv.csv'))
        x = data.x.values
        Posterior.y = data.y.values
        sigma_y = data.sigma_y.values
        initial_guess = np.array([2E21,0.5,1E15])
        CN_exp = (10e-20, 1e14)
        C_bounds = (10e-21, 10e-19)
        Nd_bounds = (1E13, 1E15)
        initial_guess = np.array([2E21,0.5,1E15])
        ls_result = [0.8e-20, 1.5E14]

        theta = (10**(-20), 2e15) #C, Nd value in O'Brian et al. 2004
        logpost = Posterior(initial_guess, C_bounds, Nd_bounds, CN_exp)
        post = logpost.log_posterior(theta, x, y, sigma_y)
        assert post == -2127.8337330441004


    def test_mc(self):
        data = load_data(get_example_data_file_path('simulated_data_csv.csv'))
        x = data.x.values
        Posterior.y = data.y.values
        sigma_y = data.sigma_y.values
        initial_guess = np.array([2E21,0.5,1E15])
        CN_exp = (10e-20, 1e14)
        C_bounds = (10e-21, 10e-19)
        Nd_bounds = (1E13, 1E15)
        initial_guess = np.array([2E21,0.5,1E15])
        ls_result = [0.8e-20, 1.5E14]
        theta = (10**(-20), 2e15) #C, Nd value in O'Brian et al. 2004
        logpost = Posterior(initial_guess, C_bounds, Nd_bounds, CN_exp)
        param_chain = logpost.mc(x,y, sigma_y, ls_result, plot_chains=False)
        assert 8.7e-21 <= param_chain <= 8.1e-21

    def test_parameters_C(self):
        data = load_data(get_example_data_file_path('simulated_data_csv.csv'))
        x = data.x.values
        Posterior.y = data.y.values
        sigma_y = data.sigma_y.values
        initial_guess = np.array([2E21,0.5,1E15])
        CN_exp = (10e-20, 1e14)
        C_bounds = (10e-21, 10e-19)
        Nd_bounds = (1E13, 1E15)
        initial_guess = np.array([2E21,0.5,1E15])
        ls_result = [0.8e-20, 1.5E14]
        theta = (10**(-20), 2e15) #C, Nd value in O'Brian et al. 2004
        logpost = Posterior(initial_guess, C_bounds, Nd_bounds, CN_exp)
        param_chain = logpost.mc(x,y, sigma_y, ls_result, plot_chains=False)
                params =  logpost.extract_parameters(param_chain)
        assert 8e-21 <= params['C'][1] <= 11e-21


    def test_parameters_Nd(self):
        data = load_data(get_example_data_file_path('simulated_data_csv.csv'))
        x = data.x.values
        Posterior.y = data.y.values
        sigma_y = data.sigma_y.values
        initial_guess = np.array([2E21,0.5,1E15])
        CN_exp = (10e-20, 1e14)
        C_bounds = (10e-21, 10e-19)
        Nd_bounds = (1E13, 1E15)
        initial_guess = np.array([2E21,0.5,1E15])
        ls_result = [0.8e-20, 1.5E14]
        theta = (10**(-20), 2e15) #C, Nd value in O'Brian et al. 2004
        logpost = Posterior(initial_guess, C_bounds, Nd_bounds, CN_exp)
        param_chain = logpost.mc(x,y, sigma_y, ls_result, plot_chains=False)
        params =  logpost.extract_parameters(param_chain)
        assert 7e13 <= params['Nd'][1] <= 12e13

    def test_plot_parameters(self):
        data = load_data(get_example_data_file_path('simulated_data_csv.csv'))
        x = data.x.values
        Posterior.y = data.y.values
        sigma_y = data.sigma_y.values
        initial_guess = np.array([2E21,0.5,1E15])
        CN_exp = (10e-20, 1e14)
        C_bounds = (10e-21, 10e-19)
        Nd_bounds = (1E13, 1E15)
        initial_guess = np.array([2E21,0.5,1E15])
        ls_result = [0.8e-20, 1.5E14]
        theta = (10**(-20), 2e15) #C, Nd value in O'Brian et al. 2004
        logpost = Posterior(initial_guess, C_bounds, Nd_bounds, CN_exp)
        param_chain = logpost.mc(x,y, sigma_y, ls_result, plot_chains=False)
        assert logpost.plot_parameters(param_chain == None)

if __name__ == '__main__':
    unittest.main()
