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
    __diagonal : bool
        Whether the penalty matrix is diagonal.
    __Sigma : numpy.ndarray
        Penalty matrix for emissions rates.

    Methods:
    --------
    __len__()
        Returns the number of time steps.
    is_diagonal()
        Returns whether the penalty matrix is diagonal.
    get_emission_at(i, xi=1)
        Calculates emission at a specific time step.
    get_emission_interval(start, end, xi=1)
        Calculates emissions for a time interval.
    get_diagonal_util(xi, i, ra=0.)
        Calculates utility for diagonal penalty case.
    get_marginal_util(xi, i, x_old=None, ra=0.)
        Calculates marginal utility.
    get_emissions(x)
        Calculates total emissions for a given schedule.
    get_total_emission(x)
        Calculates total emission for a given schedule.
    get_total_util(x, ra=0.)
        Calculates total utility including risk aversion factor.

    """

    def __init__(self, mu, isDiagonal=True, Sigma=None, sig2=0.0, ac1=0.0):
        """
        Initializes the Moer object.

        Parameters:
        -----------
        mu : array-like
            Emissions rate for each time step.
        isDiagonal : bool, optional
            Whether the penalty matrix is diagonal. Default is True.
        Sigma : numpy.ndarray, optional
            Penalty matrix if not diagonal. Default is None.
        sig2 : float or array-like, optional
            Variance(s) for diagonal case. Default is 0.
        ac1 : float, optional
            First-order autocorrelation coefficient for non-diagonal case. Default is 0.
        """
        self.__mu = np.array(mu).flatten()
        self.__T = self.__mu.shape[0]
        self.__diagonal = isDiagonal
        if isinstance(sig2, float):
            sig2 = np.array([sig2] * self.__T)
        else:
            sig2 = np.array(sig2).flatten()
        if not self.__diagonal:
            if Sigma is not None:
                assert Sigma.shape == (self.__T, self.__T)
                self.__Sigma = Sigma
            else:
                from scipy.linalg import toeplitz

                x = ac1 ** np.arange(0, self.__T)
                self.__Sigma = toeplitz(x, x) * np.diag(sig2)
        else:
            self.__Sigma = sig2

    def __len__(self):
        """
        Returns the length of the time series.

        Returns:
        --------
        int
            The number of time steps in the series.
        """
        return self.__T

    def is_diagonal(self):
        """
        Whether the penalty matrix is diagonal.

        Returns:
        --------
        bool
            True if the penalty matrix is diagonal, False otherwise.
        """
        return self.__diagonal

    def get_emission_at(self, i, xi=1):
        """
        Calculates the emission at a specific time step.

        Parameters:
        -----------
        i : int
            The time step index.
        xi : float, optional
            The emission multiplier. Default is 1.

        Returns:
        --------
        float
            The calculated emission value.
        """
        return self.__mu[i] * xi

    def get_emission_interval(self, start, end, xi=1):
        """
        Calculates emissions for a given time interval.

        Parameters:
        -----------
        start : int
            The start index of the interval.
        end : int
            The end index of the interval.
        xi : float, optional
            The emission multiplier. Default is 1.

        Returns:
        --------
        numpy.ndarray
            An array of emission values for the specified interval.
        """
        return self.__mu[start:end] * xi

    def get_diagonal_util(self, xi, i, ra=0.0):
        """
        Calculates the diagonal utility for a given time step.

        Parameters:
        -----------
        xi : float
            The emission multiplier.
        i : int
            The time step index.
        ra : float, optional
            The risk aversion factor. Default is 0.0.

        Returns:
        --------
        float
            The calculated diagonal utility.
        """
        return self.get_emission_at(i, xi) + ra * self.__Sigma[i] * xi**2

    def get_marginal_util(self, xi, i, x_old=None, ra=0.0):
        """
        Calculates the marginal utility for a given time step.

        Parameters:
        -----------
        xi : float
            The emission multiplier.
        i : int
            The time step index.
        x_old : numpy.ndarray, optional
            Previous emission values. Default is None.
        ra : float, optional
            The risk aversion factor. Default is 0.0.

        Returns:
        --------
        float or tuple
            The calculated marginal utility, and updated x_old if not diagonal.
        """
        if self.__diagonal:
            return self.get_diagonal_util(xi, i, ra)
        else:
            return self.get_emission_at(xi, i) + ra * (
                2 * xi * self.__Sigma[i, :i] @ x_old[:i] + self.__Sigma[i, i] * xi**2
            ), np.hstack((x_old, xi))

    def get_emissions(self, x):
        """
        Calculates emissions for a given set of emission multipliers.

        Parameters:
        -----------
        x : array-like
            The emission multipliers.

        Returns:
        --------
        numpy.ndarray
            An array of calculated emission values.
        """
        x = np.array(x).flatten()
        return self.__mu[: x.shape[0]] * x

    def get_total_emission(self, x):
        """
        Calculates the total emission for a given set of emission multipliers.

        Parameters:
        -----------
        x : array-like
            The emission multipliers.

        Returns:
        --------
        float
            The total calculated emission.
        """
        x = np.array(x).flatten()
        return np.dot(self.__mu[: x.shape[0]], x)

    def get_total_util(self, x, ra=0.0):
        """
        Calculates the total utility for a given set of emission multipliers.

        Parameters:
        -----------
        x : array-like
            The emission multipliers.
        ra : float, optional
            The risk aversion factor. Default is 0.0.

        Returns:
        --------
        float
            The total calculated utility.
        """
        x = np.array(x).flatten()
        if self.__diagonal:
            return (
                self.get_total_emission(x)
                + ra * np.multiply(self.__Sigma[: x.shape[0]], x**2).sum()
            )
        else:
            return (
                self.get_total_emission(x)
                + ra * x.reshape((1, -1)) @ self.__Sigma[: x.shape[0], : x.shape[0]] @ x
            )
