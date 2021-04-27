import numpy as np
from numpy import log
from scipy.optimize import fsolve

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