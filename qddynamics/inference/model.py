import numpy as np
from numpy import log
from scipy.optimize import fsolve
import emcee

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
        return np.where(
            np.logical_and(x <= self.xmax, x >= self.xmin),
            -log(self.xmax - self.xmin), -np.inf)


class JefferysPrior(Prior):
    """
    Returns the value of the Jefferys prior at position x for range xmin to xmax
    """

    def logp(self, x):
        return np.where(
            np.logical_and(x <= self.xmax, x >= self.xmin),
            -log(x) - log(log(self.xmax / self.xmin)), -np.inf)

class Likelihood:
    """
    Returns the likelihood of the model.
    """
    def __init__(self, theta, x, y, sigma_y, initial_guess):
        self.theta = theta
        self.x = x
        self.y = y
        self.sigma_y = sigma_y
         ## Initial guesses for solving rate equations below
        self.initial_guess = initial_guess


class LogLikelihood(Likelihood):
    """
    Returns the log of the likelihood given the likelihood defined above.
    """
    def logllh(self):
        """
        returns log of likelihood

        Parameters:
            theta: model parameters (specified as a tuple)
            x: independent data (array of length N)
            y: measurements (array of length N)
            sigma_y: uncertainties on y (array of length N)
        """
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

        S_out = []
        p_out = []
        N_out = []

        o = 0


        zGuess = self.initial_guess

        ## solve the rate equations in steady state
        ## for each value of input current

        def rateEquations(z, i):
            '''
            Differential equations that describe the lasing behaviour of quantum dots
            for a given input current
            -------------
            Parameters:
            z := ndarray
                initial guess for solutions
            i := ndarray
                input currents
            '''

            S = z[0]
            p = z[1]
            N = z[2]*i/self.x[0]

            F = np.empty((3))
            F[0] = -(S/ts) + g0*v*(2*p - 1)*S
            F[1] = -(p/td) - g0*(2*p - 1)*S + ((C)*(N**2) + (B*N))*(1-p) - Resc*p
            F[2] = (i/q) - (N/tn) - 2*Nd*(((C)*(N**2) + (B*N))*(1-p) - Resc*p)


            return F

        fout = lambda ip: fsolve(rateEquations, zGuess, args = ip)
        z = np.array(list(map(fout, self.x)))

        S_out = z[:,0]
        p_out = z[:,1]
        N_out = z[:,2]

        residual = (self.y - S_out)**2
        chi_square = np.sum(residual/(self.sigma_y**2))
        constant = np.sum(log(1/np.sqrt(2.0*np.pi*self.sigma_y**2)))
        return constant - 0.5*chi_square

class Posterior:
     def __init__(self, theta, x, y, sigma_y,initial_guess, prior_upbnd_C, prior_lowerbnd_C, prior_upbnd_Nd, prior_lowerbnd_Nd):
        self.theta = theta
        self.x = x
        self.y = y
        self.sigma_y = sigma_y
        self.initial_guess = initial_guess
        self.prior_up_C = prior_upbnd_C
        self.prior_low_C = prior_lowerbnd_C
        self.prior_up_Nd = prior_upbnd_Nd
        self.prior_low_Nd = prior_lowerbnd_Nd
    def logp(self):
        theta = self.theta
        C, Nd = self.theta
        up_bnd_C, low_bnd_C = self.prior_up_C, self.prior_low_C
        up_bnd_Nd, low_bnd_Nd = self.prior_up_Nd, self.prior_low_Nd    
        log_prior = UniformPrior(low_bnd_C, up_bnd_C).logp(C) + UniformPrior(low_bnd_Nd, up_bnd_Nd).logp(Nd)
        log_likelihood = LogLikelihood(theta, self.x, self.y, self.sigma_y, self_initial_guess).logllh()
        return log_prior+log_likelihood

     def run_mcmc(self,ndim, nwalkers, nsteps, ls_result,gaussian_ball_size):
        gaussian_ball = gaussian_ball_size * np.random.randn(nwalkers, ndim)
        starting_positions = (1 + gaussian_ball) * ls_result
        sampler = emcee.EnsembleSampler(nwalkers, ndim, Posterior.logp(), 
                                args=(posterior))
        # run the sampler. We use iPython's %time directive to tell us 
        # how long it took (in a script, you would leave out "%time")
        %time sampler.run_mcmc(starting_positions, nsteps)
        print('Done')