import numpy as np
from numpy import log

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