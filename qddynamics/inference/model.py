import numpy as np
from numpy import log
from scipy.optimize import fsolve
import emcee
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class Prior:
    """
    Prior Base class. This can store the shape parameters in the object instance
    then be used as a function
    """
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax


class UniformPrior(Prior):
    """
    Returns the value of the uniform prior at position x for range xmin to xmax
    """
    def logp(self, x):
        '''
        Parameters:
        --------
        x: ndarray
        returns: uniform prior probability
        '''
        return np.where(
            np.logical_and(x <= self.xmax, x >= self.xmin),
            -log(self.xmax - self.xmin), -np.inf)


class JefferysPrior(Prior):
    """
    Returns the value of the Jefferys prior at position x for range xmin to xmax
    """

    def logp(self, x):
        '''
        Parameters:
        --------
        x: ndarray
        returns: jefferys prior probability
        '''
        return np.where(
            np.logical_and(x <= self.xmax, x >= self.xmin),
            -log(x) - log(log(self.xmax / self.xmin)), -np.inf)

class LogLikelihood:
    """
    Returns the log of the likelihood of the model.
    """
    def __init__(self, theta, x, y, sigma_y, initial_guess):
        self.theta = theta
        self.x = x
        self.y = y
        self.sigma_y = sigma_y
        self.initial_guess = initial_guess

    def rateEquations(self, z, i):
        '''
        Differential equations that describe the lasing behaviour of quantum dots
        for a given input current

        Returns solutions to differential equations for lasing rates in steady state
        -------------
        Parameters:
        z := ndarray
            initial guess for solutions
        i := ndarray
            input currents
        '''
        C, Nd = self.theta

        ## Constants--currently same constansts as used
        ## in O'Brien.
        Resc = 0
        B = 0
        ## Nd = 1E14 # m^-2
        v = 2.4E22 #2*Nd*confinement_factor/d
        tn = 1E-9 # seconds
        td = tn
        ts = 3E-12 # seconds
        g0 = 0.15E-10 # m^3/seconds
        q = 1.60217662E-19 # Charge of electron (coulombs)

        S = z[0]
        p = z[1]
        N = z[2]*i/self.x[0]

        F = np.empty((3))
        F[0] = -(S/ts) + g0*v*(2*p - 1)*S
        F[1] = -(p/td) - g0*(2*p - 1)*S + ((C)*(N**2) + (B*N))*(1-p) - Resc*p
        F[2] = (i/q) - (N/tn) - 2*Nd*(((C)*(N**2) + (B*N))*(1-p) - Resc*p)
        return F

    def logllh(self):
        """
        returns log of likelihood

        Parameters:
            theta: model parameters (specified as a tuple)
            x: independent data (array of length N)
            y: measurements (array of length N)
            sigma_y: uncertainties on y (array of length N)
        """


        S_out = []
        p_out = []
        N_out = []

        o = 0

        zGuess = self.initial_guess

        fout = lambda ip: fsolve(self.rateEquations, zGuess, args = ip)
        z = np.array(list(map(fout, self.x)))

        S_out = z[:,0]
        p_out = z[:,1]
        N_out = z[:,2]

        residual = (self.y - S_out)**2
        chi_square = np.sum(residual/(self.sigma_y**2))
        constant = np.sum(log(1/np.sqrt(2.0*np.pi*self.sigma_y**2)))
        return constant - 0.5*chi_square

class Posterior:
    '''
    class for computing posterior given lasing rate model
    returns posterior probability
    Parameters
    -----------
    theta: lasing rate starting point
    initial_guess: tuple
        initial guess to solutions to lasing rate equations
    C_bounds: tuple
        lower and upper bounds on C parameter
    Nd_bounds: tuple
        lower and upper bounds on Nd Parameter
    CN_exp: tuple
        expected C and Nd
    '''
    def __init__(self, initial_guess, C_bounds, Nd_bounds, CN_exp):
        self.initial_guess = initial_guess
        self.C_bounds = C_bounds
        self.Nd_bounds = Nd_bounds
        self.CN_exp = CN_exp

    def log_posterior(self, theta, x, y, sigma_y):
        '''
        Computes the log of posterior probability given bounds on variables
        paramters:
        x: ndarray
            Control variables
        y: ndarray
            Photoluminescence of quantum dots
        sigma_y: ndarray
            uncertainty on measurement
        returns: ndarray
            log of posterior probability
        '''
        C, Nd = self.CN_exp
        C_lower, C_upper = self.C_bounds
        Nd_lower, Nd_upper = self.Nd_bounds
        Nd_prior = UniformPrior(Nd_lower, Nd_upper).logp(Nd)
        C_prior =  UniformPrior(C_lower, C_upper).logp(C)
        likelihood = LogLikelihood(theta, x, y, sigma_y, self.initial_guess).logllh()
        return Nd_prior + C_prior + likelihood

    def mc(self, x, y, sigma_y, ls_result, nwalkers=50, nsteps=500, plot_chains=True):
        '''
        Uses emcee to do MCMC sampling to find estimate C and Nd parameters given model
        Paramters
        ---------
        x: ndarray
            Control variables
        y: ndarray
            Photoluminescence of quantum dots
        sigma_y: ndarray
            uncertainty on measurement
        ls_result: ndarray
            starting point for walkers
        nwalkers: int
            number of walkers
        nsteps: int
            number of steps
        returns: DataFrame
            data frame of parameter sampler chains
        '''
        ndim = 2
        gaussian_ball = 1e-4 * np.random.randn(nwalkers, ndim)
        starting_positions = (1 + gaussian_ball) * ls_result
        sampler = emcee.EnsembleSampler(nwalkers, ndim,self.log_posterior,
                                        args=(x, y, sigma_y))
        sampler.run_mcmc(starting_positions, nsteps)
        print('Done')
        if plot_chains:
            fig, (ax_C, ax_Nd) = plt.subplots(2)
            ax_C.set(ylabel='C')
            ax_Nd.set(ylabel='Nd')
            for i in range(50):
                df = pd.DataFrame({'C': sampler.chain[i,:,0], 'Nd':sampler.chain[i,:,1]})
                sns.lineplot(data=df, x=df.index, y='C', ax=ax_C)
                sns.lineplot(data=df, x=df.index, y='Nd', ax=ax_Nd)

        samples = sampler.chain[:,100:,:]
        traces = samples.reshape(-1, ndim).T
        parameter_samples = pd.DataFrame({'C': traces[0], 'Nd': traces[1]})
        return parameter_samples

    def extract_parameters(self, parameter_samples):
        '''
        extract MAP from parameter chains
        parameters:
        -----------
        parameter_samples: DataFrame
            sampler chains from MCMC
        returns: ndarray
            MAP Values for parameter estimation
        '''
        q = parameter_samples.quantile([0.16,0.50,0.84], axis=0)
        print("C = {:.2e} + {:.2e} - {:.2e}".format(q['C'][0.50],
                                                    q['C'][0.84]-q['C'][0.50],
                                                    q['C'][0.50]-q['C'][0.16]))
        print("Nd = {:.2e} + {:.2e} - {:.2e}".format(q['Nd'][0.50],
                                                    q['Nd'][0.84]-q['Nd'][0.50],
                                                    q['Nd'][0.50]-q['Nd'][0.16]))
        return q

    def plot_parameters(self, parameter_samples):
        '''
        plot KDE of parameters when given chains
        parameters
        ---------
        paramter_samples: DataFrame
            data frame of sampler chains from MCMC Sampler
        returns: str
            variable plotted on x axis
        '''
        joint_kde = sns.jointplot(x='C', y='Nd', data=parameter_samples, kind='scatter', s=0.2)
        return joint_kde.x.name
