import numpy as np
from numpy import log
from scipy.optimize import fsolve
import emcee
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


# def linear_fit(x, y, sigma_y, data, g0 = 0.15E-10):
#     inverse_variance = 1./(sigma_y ** 2)
#     linear_model = smf.wls(formula='y~x', data = data, weights=inverse_variance)
#     linear_result = linear_model.fit(cov_type='fixed_scale')
#     print(linear_result.params)
#     parameter_ftn = linear_result.params.array
#     #Solving the equation, the gradient is tau_s/theta the intercept -tau_s/theta*(N/tau_n+2*Nd*rho/tau_d)
#     ## Nd = 1E14 # m^-2
#     v = 2.4E22
#     tn = 1E-9 # seconds
#     td = tn
#     ts = 3E-12 # seconds
#    # g0 = 0.15E-10 # m^3/seconds
#     q = 1.60217662E-19 # Charge of electron (coulombs)
#     p = 0.9
#     theta = 1/g0/(2*p-1)/tau_s
#     N = g0*y[0]/x[0]
#     theta = tau_s/parameter_ftn[1]
#     p = (1/g0/theta/tau_s+1)/2
#     parameter_ftn[0]+x_data*parameter_ftn[1]

#     N_d = (parameter_ftn[0]*(0-theta/tau_s)-N/tau_n)/2/p*tau_d
#     C = (p/tau_d+np.average(x)/tau_s/v)/(1-p)/N**2

#     Isresult = np.array([Nd,C])
#     return Isresult

class Prior:
    """
    Prior Base class. This can store the shape parameters in the object instance
    then be used as a function
    """
    def __init__(self, xmin, xmax):
        '''
        Initialize Prior Class
        Parameters:
        -----------
        xmin, xmax: float
            minimum and maximum possible values of parameters
        '''
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
        '''
        Initialize LogLikelihood Class
        Parameters:
        -----------
        theta: tuple
            model parameters
        x: ndarray
            input currents
        y: ndarray
            photoluminescence data
        sigma_y: ndarray
            uncertainty on photoluminescence data
        initial_guess: ndarray
            initial guess to lasing rate equation solution
        '''
        self.theta = theta
        self.x = x
        self.y = y
        self.sigma_y = sigma_y
        self.initial_guess = initial_guess

    def rateEquations(self,z,i,
                      Resc = 0, B = 0, confinement_factor = 0.06, d = 10E-9,
                      tn = 1E-9, ts = 3E-12):
        '''
        Differential equations that describe the lasing behaviour of quantum dots
        for a given input current

        Constants default to those reported by O'Brien et. al. (2004)
        -------------
        Parameters:
        z: ndarray
            initial guess for solutions
        i: ndarray
            input currents
        Resc: float
            Quantum dot carrier escape rate
        B: float
            carrier-phonon coefficient
        confinement_factor: float
            field confinment in the quantum dots
        d: float
            thickness of quantum well/quantum dot layer
        tn: float
            carrier lifetime in quantum wells and dots
        ts: float
            photon lifetime

        Returns: ndarray
            solutions to differential equations for lasing rates in steady state
            S: Photon density
            P: Dot population
            N: Photon Number

        '''
        C, Nd, g0 = self.theta
        v = 2*Nd*confinement_factor/d
        td = tn
        ts = 3E-12 # seconds
        q = 1.60217662E-19 # Charge of electron (coulombs)

        S = z[0]
        p = z[1]
        N = z[2]

        F = np.empty((3))
        F[0] = -(S/ts) + g0*v*(2*p - 1)*S
        F[1] = -(p/td) - g0*(2*p - 1)*S + ((C)*(N**2) + (B*N))*(1-p) - Resc*p
        F[2] = ((i)/q) - (N/tn) - 2*Nd*(((C)*(N**2) + (B*N))*(1-p) - Resc*p)
        return F

    def logllh(self):
        """
        Computes log likelihood probability
        Parameters:
        -----------
            theta: model parameters (specified as a tuple)
            x: independent data (array of length N)
            y: measurements (array of length N)
            sigma_y: uncertainties on y (array of length N)

        Returns: float
            log likelihood of probability
        """
        scaling_factor = 1E2

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

    '''
    def __init__(self, initial_guess, C_bounds, Nd_bounds, g0_bounds, CNg0_exp):
        '''
        Initialize Posterior class
        Parameters
        -----------
        initial_guess: tuple
            initial guess to solutions to lasing rate equations
        C_bounds: tuple
            lower and upper bounds on C parameter
        Nd_bounds: tuple
            lower and upper bounds on Nd Parameter
        CN_exp: tuple
            expected C and Nd
        '''
        self.initial_guess = initial_guess
        self.C_bounds = C_bounds
        self.Nd_bounds = Nd_bounds
        self.g0_bounds = g0_bounds
        self.CNg0_exp = CNg0_exp

    def log_posterior(self, theta, x, y, sigma_y):
        '''
        Computes the log of posterior probability given bounds on variables
        parameters:
        ----------
        x: ndarray
            Control variables
        y: ndarray
            Photoluminescence of quantum dots
        sigma_y: ndarray
            uncertainty on measurement
        returns: ndarray
            log of posterior probability
        '''
        C, Nd, g0 = self.CNg0_exp
        C_lower, C_upper = self.C_bounds
        Nd_lower, Nd_upper = self.Nd_bounds
        g0_lower, g0_upper = self.g0_bounds
        Nd_prior = UniformPrior(Nd_lower, Nd_upper).logp(Nd)
        C_prior =  UniformPrior(C_lower, C_upper).logp(C)
#         C_prior =  JefferysPrior(C_lower, C_upper).logp(C)
#         g0_prior = UniformPrior(g0_lower, g0_upper).logp(g0)
        g0_prior = JefferysPrior(g0_lower, g0_upper).logp(g0)
        likelihood = LogLikelihood(theta, x, y, sigma_y, self.initial_guess).logllh()
        return Nd_prior + C_prior + g0_prior + likelihood

    def mc(self, x, y, sigma_y, ls_result, nwalkers=100, nsteps=500, plot_chains=True):
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
        ndim = 3
        gaussian_ball = 1e-4 * np.random.randn(nwalkers, ndim)
        starting_positions = (1 + gaussian_ball) * ls_result
        sampler = emcee.EnsembleSampler(nwalkers, ndim,self.log_posterior,
                                        args=(x, y, sigma_y))
        sampler.run_mcmc(starting_positions, nsteps)
        print('Done')
        if plot_chains:
            fig, (ax_C, ax_Nd, ax_g0) = plt.subplots(3)
            ax_C.set(ylabel='C')
            ax_Nd.set(ylabel='Nd')
            ax_g0.set(ylabel='g0')
            for i in range(50):
                df = pd.DataFrame({'C': sampler.chain[i,:,0], 'Nd':sampler.chain[i,:,1], 'g0':sampler.chain[i,:,2]})
                sns.lineplot(data=df, x=df.index, y='C', ax=ax_C)
                sns.lineplot(data=df, x=df.index, y='Nd', ax=ax_Nd)
                sns.lineplot(data=df, x=df.index, y='g0', ax=ax_g0)

        samples = sampler.chain[:,100:,:]
        traces = samples.reshape(-1, ndim).T
        parameter_samples = pd.DataFrame({'C': traces[0], 'Nd': traces[1], 'g0': traces[2]})
        return parameter_samples, sampler, starting_positions

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
        print("g0 = {:.2e} + {:.2e} - {:.2e}".format(q['g0'][0.50],
                                                    q['g0'][0.84]-q['g0'][0.50],
                                                    q['g0'][0.50]-q['g0'][0.16]))
        return q

    def plot_parameters(self, parameter_samples, xname, yname):
        '''
        plot KDE of parameters when given chains
        parameters
        ---------
        paramter_samples: DataFrame
            data frame of sampler chains from MCMC Sampler
        returns: str
            variable plotted on x axis
        '''
        joint_kde = sns.jointplot(x=xname, y=yname, data=parameter_samples, kind='scatter', s=0.2)
        return joint_kde.x.name
