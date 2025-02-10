from typing import List
import numpy as np
from numpy.typing import ArrayLike, NDArray


class Moer:
    """Handles Marginal Operating Emissions Rate (MOER) calculations for electricity grid emissions.

    A class for processing and analyzing emissions data based on MOER measurements,
    supporting various calculations including time-specific emissions and interval summations.

    Parameters
    ----------
    mu : ArrayLike
        Emissions rate data for each time step.

    Attributes
    ----------
    __mu : NDArray[np.float64]
        Mean emissions rate for each time step.
    __T : int
        Total number of time steps.

    Examples
    --------
    >>> moer = Moer([0.5, 0.6, 0.7])
    >>> moer.get_emission_at(0, 2.0)
    1.0
    >>> moer.get_total_emission([1.0, 2.0, 1.5])
    2.75
    """

    def __init__(self, mu: ArrayLike) -> None:
        self.__mu: NDArray[np.float64] = np.array(mu).flatten()
        self.__T: int = self.__mu.shape[0]

    def __len__(self) -> int:
        """Return the number of time steps in the series.

        Returns
        -------
        int
            Length of the time series.
        """
        return self.__T

    def get_emission_at(self, i: int, usage: float) -> float:
        """Calculate emission at a specific time step.

        Parameters
        ----------
        i : int
            Time step index.
        usage : float
            Power usage multiplier.

        Returns
        -------
        float
            Calculated emission value.
        """
        return self.__mu[i] * usage

    def get_emission_interval(self, start: int, end: int, usage: float) -> float:
        """Calculate total emissions for a time interval.

        Parameters
        ----------
        start : int
            Start index of interval.
        end : int
            End index of interval.
        usage : float
            Emission multiplier.

        Returns
        -------
        float
            Sum of emissions for the interval.
        """
        return np.dot(self.__mu[start:end], usage)

    def get_emissions(self, usage: ArrayLike) -> NDArray[np.float64]:
        """Calculate emissions for each time step given usage values.

        Parameters
        ----------
        usage : ArrayLike
            Array of emission multipliers.

        Returns
        -------
        NDArray[np.float64]
            Array of calculated emissions.
        """
        usage_array: NDArray[np.float64] = np.array(usage).flatten()
        return self.__mu[: usage_array.shape[0]] * usage_array

    def get_total_emission(self, usage: ArrayLike) -> float:
        """Calculate total emissions across all time steps.

        Parameters
        ----------
        usage : ArrayLike
            Array of emission multipliers.

        Returns
        -------
        float
            Total emission value.
        """
        usage_array: NDArray[np.float64] = np.array(usage).flatten()
        return np.dot(self.__mu[: usage_array.shape[0]], usage_array)