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
    get_emission_at(i, usage)
        Calculates emission at a specific time step.
    get_emission_interval(start, end, usage)
        Calculates sum of emissions for a time interval.
    get_emissions(x)
        Calculates emissions per interval for a given schedule.
    get_total_emission(x)
        Calculates total emission for a given schedule.
    
    """

    def __init__(self, mu):
        """
        Initializes the Moer object.

        Parameters:
        -----------
        mu : array-like
            Emissions rate for each time step.
        """
        self.__mu = np.array(mu).flatten()
        self.__T = self.__mu.shape[0]

    def __len__(self):
        """
        Returns the length of the time series.

        Returns:
        --------
        int
            The number of time steps in the series.
        """
        return self.__T

    def get_emission_at(self, i, usage):
        """
        Calculates the emission at a specific time step.

        Parameters:
        -----------
        i : int
            The time step index.
        usage : float, optional
            The power usage.

        Returns:
        --------
        float
            The calculated emission value.
        """
        return self.__mu[i] * usage

    def get_emission_interval(self, start, end, usage):
        """
        Calculates emissions for a given time interval.

        Parameters:
        -----------
        start : int
            The start index of the interval.
        end : int
            The end index of the interval.
        usage : float, optional
            The emission multiplier. Default is 1.

        Returns:
        --------
        numpy.ndarray
            An array of emission values for the specified interval.
        """
        return np.dot(self.__mu[start:end], usage)

    def get_emissions(self, usage):
        """
        Calculates emissions for a given set of emission multipliers.

        Parameters:
        -----------
        usage : array-like
            The emission multipliers.

        Returns:
        --------
        numpy.ndarray
            An array of calculated emission values.
        """
        usage = np.array(usage).flatten()
        return self.__mu[:usage.shape[0]] * usage

    def get_total_emission(self, usage):
        """
        Calculates the total emission for a given set of emission multipliers.

        Parameters:
        -----------
        usage : array-like
            The emission multipliers.

        Returns:
        --------
        float
            The total calculated emission.
        """
        usage = np.array(usage).flatten()
        return np.dot(self.__mu[: usage.shape[0]], usage)