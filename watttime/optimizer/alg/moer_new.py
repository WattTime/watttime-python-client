# moer.py

import numpy as np 
class Moer: 
    """
    Represents Marginal Operating Emissions Rate (MOER) for electricity grid emissions modeling.

    This class handles calculations related to emissions and utilities based on
    MOER data, supporting both diagonal and non-diagonal penalty matrices.

    Attributes:
    -----------
    __mu : numpy.ndarray
        Mean emissions rate for each time step.
    __T : int
        Total number of time steps.

    Methods:
    --------
    __len__()
        Returns the number of time steps.
    get_emission_at(i, xi=1)
        Calculates emission at a specific time step.
    get_emission_interval(start, end, xi=1)
        Calculates emissions for a time interval.
    get_emissions(x)
        Calculates emissions per interval for a given schedule.
    get_total_emission(x)
        Calculates total emission for a given schedule.
    """
    def __init__(self, mu): 
        self.__mu = np.array(mu).flatten()
        self.__T = self.__mu.shape[0]

    def __len__(self):
        return self.__T
    
    def is_diagonal(self):
        return True
    
    def get_emission_at(self, i, usage): 
        return self.__mu[i] * usage
    
    def get_emission_interval(self, start, end, usage): 
        return np.dot(self.__mu[start:end], usage)
        
    def get_emissions(self, usage):
        usage = np.array(usage).flatten()
        return self.__mu[:usage.shape[0]] * usage

    def get_total_emission(self, usage): 
        return self.get_emissions(usage).sum()
