# optCharger.py

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from .moer import Moer

TOL: float = 1e-4  # tolerance
EMISSION_FN_TOL: float = 1e-9  # emissions functions tolerance in kw


class OptCharger:
    """
    Represents an Optimal Charger for managing charging schedules.

    This class handles the optimization of charging schedules based on various parameters
    such as charge rates, emission overheads, and other constraints.

    Methods:
    --------
    __init__()
        Initializes the OptCharger object with the given parameters.
    """

    def __init__(self, verbose: bool) -> None:
        """
        Initializes the OptCharger object.
        """
        self.__optimal_charging_emission: Optional[float] = None
        self.__optimal_charging_schedule: Optional[List[int]] = None
        self.__verbose: bool = verbose

    def __collect_results(self, moer: Moer) -> None:
        """
        Translates the optimal charging schedule into a series of emission multiplier values and calculates various emission-related metrics.

        This function processes the optimal charging schedule to generate emission multipliers,
        calculates energy and emissions over time, and computes the total emissions including
        overhead from starting, stopping, and maintaining the charging process.

        Parameters:
        -----------
        moer : Moer
            An object representing Marginal Operating Emissions Rate, used for emissions calculations.

        Returns:
        --------
        None
            The function updates several instance variables with the calculated results.

        Side Effects:
        -------------
        Updates the following instance variables:
        - __optimal_charging_energy_over_time
        - __optimal_charging_emissions_over_time
        - __optimal_charging_emission

        The function also populates the emission_multipliers list, which is used in the calculations.
        """
        emission_multipliers: List[float] = []
        current_charge_time_units: int = 0
        for i in range(len(self.__optimal_charging_schedule)):
            if self.__optimal_charging_schedule[i] == 0:
                emission_multipliers.append(0.0)
            else:
                old_charge_time_units = current_charge_time_units
                current_charge_time_units += self.__optimal_charging_schedule[i]
                power_rate = self.emission_multiplier_fn(
                    old_charge_time_units, current_charge_time_units
                )
                emission_multipliers.append(power_rate)

        self.__optimal_charging_energy_over_time = np.array(
            self.__optimal_charging_schedule
        ) * np.array(emission_multipliers)
        self.__optimal_charging_emissions_over_time = moer.get_emissions(
            self.__optimal_charging_energy_over_time
        )
        self.__optimal_charging_emission = (
            self.__optimal_charging_emissions_over_time.sum()
        )

    def verbose_on(self, statement: str) -> None:
        if self.__verbose:
            print(statement)

    @staticmethod
    def __sanitize_emission_multiplier(emission_multiplier_fn: Callable[[float, float], float], total_charge: float) -> Callable[[float, float], float]:
        """
        Sanitizes the emission multiplier function to handle edge cases and ensure valid outputs.

        This function wraps the original emission_multiplier_fn to handle cases where the
        end charge (ec) exceeds the total charge or when the start charge (sc) is beyond
        the total charge limit.

        Parameters:
        -----------
        emission_multiplier_fn : callable
            The original emission multiplier function to be sanitized.
        total_charge : int or float
            The maximum total charge value.

        Returns:
        --------
        callable
            A new lambda function that sanitizes the inputs before calling the original
            emission_multiplier_fn.

        Behavior:
        ---------
        - If sc < total_charge:
            - Calls the original function with ec capped at total_charge.
        - If sc >= total_charge:
            - Returns 1.0, assuming no additional emissions beyond total charge.

        Note:
        -----
        This function is useful for preventing out-of-bounds errors and ensuring
        consistent behavior when dealing with charge values near or beyond the total
        charge limit.
        """
        return lambda sc, ec: (
            emission_multiplier_fn(sc, min(ec, total_charge))
            if (sc < total_charge)
            else 0.0
        )

    @staticmethod
    def __check_constraint(t_start: int, c_start: int, dc: int, constraints: Dict[int, Tuple[int, int]]) -> bool:
        # assuming constraints[t] is the bound on total charge after t intervals
        for t in range(t_start + 1, t_start + dc):
            if (t in constraints) and (
                (c_start + t - t_start < constraints[t][0])
                or (c_start + t - t_start > constraints[t][1])
            ):
                return False
        return True

    def __greedy_fit(self, total_charge: int, total_time: int, moer: Moer) -> None:
        """
        Performs a "greedy" fit for charging schedule optimization.

        It charges at the maximum possible rate until the total charge is reached or
        the time limit is hit.

        Parameters:
        -----------
        total_charge : int
            The total amount of charge needed.
        total_time : int
            The total time available for charging.
        moer : Moer
            An object representing Marginal Operating Emissions Rate.

        Calls __collect_results to process the results.
        """
        self.verbose_on("== Baseline fit! ==")
        schedule = [1] * min(total_charge, total_time) + [0] * max(
            0, total_time - total_charge
        )
        self.__optimal_charging_schedule = schedule
        self.__collect_results(moer)

    def __simple_fit(self, total_charge: int, total_time: int, moer: Moer) -> None:
        """
        Performs a "simple" fit for charging schedule optimization.

        This method implements a straightforward optimization strategy. It sorts
        time intervals by MOER (Marginal Operating Emissions Rate) and charges
        during the cleanest intervals until the total charge is reached.

        Parameters:
        -----------
        total_charge : int
            The total amount of charge needed.
        total_time : int
            The total time available for charging.
        moer : Moer
            An object representing Marginal Operating Emissions Rate.

        Calls __collect_results to process the results.
        """
        self.verbose_on("== Simple fit! ==")
        sorted_times = np.argsort(moer.get_emission_interval(0, total_time, 1))

        charge_to_do = total_charge
        schedule, t = [0] * total_time, 0
        while (charge_to_do > 0) and (t < total_time):
            charge_to_do -= 1
            schedule[sorted_times[t]] = 1
            t += 1
        self.__optimal_charging_schedule = schedule
        self.__collect_results(moer)

    def __diagonal_fit(
        self,
        total_charge: int,
        total_time: int,
        moer: Moer,
        emission_multiplier_fn: Callable[[float, float], float],
        constraints: Dict[int, Tuple[Optional[int], Optional[int]]] = {},
    ) -> None:
        """
        Performs a sophisticated diagonal fit for charging schedule optimization using dynamic programming.

        This method implements a more complex optimization strategy using dynamic programming.
        It considers various factors such as emission rates, charging constraints, and overhead costs
        to find an optimal charging schedule.

        Parameters:
        -----------
        total_charge : int
            The total amount of charge needed.
        total_time : int
            The total time available for charging.
        moer : Moer
            An object representing Marginal Operating Emissions Rate.
        emission_multiplier_fn : callable
            A function that calculates emission multipliers.
        constraints : dict, optional
            A dictionary of charging constraints for specific time steps.

        Calls __collect_results to process the results.

        Raises:
        -------
        Exception
            If no valid solution is found.
        """
        self.verbose_on("== Sophisticated fit! ==")
        # This is a matrix with size = number of charge states x number of actions {not charging = 0, charging = 1}
        max_util = np.full((total_charge + 1), np.nan)
        max_util[0] = 0.0
        path_history = np.full((total_time, total_charge + 1), -1, dtype=int)
        for t in range(1, total_time + 1):
            if t in constraints:
                min_charge, max_charge = constraints[t]
                min_charge = 0 if min_charge is None else max(0, min_charge)
                max_charge = (
                    total_charge
                    if max_charge is None
                    else min(max_charge, total_charge)
                )
            else:
                min_charge, max_charge = 0, total_charge
            new_max_util = np.full(max_util.shape, np.nan)
            for c in range(min_charge, max_charge + 1):
                ## not charging
                init_val = True
                if not np.isnan(max_util[c]):
                    new_max_util[c] = max_util[c]
                    path_history[t - 1, c] = c
                    init_val = False
                ## charging
                if (c > 0) and not np.isnan(max_util[c - 1]):
                    # moer.get_emission_at gives lbs/MWh. emission function needs to be how many MWh the interval consumes
                    # which would be power_in_kW * 0.001 * 5/60
                    new_util = max_util[c - 1] - moer.get_emission_at(
                        t - 1, emission_multiplier_fn(c - 1, c)
                    )
                    if init_val or (new_util > new_max_util[c]):
                        new_max_util[c] = new_util
                        path_history[t - 1, c] = c - 1
                    init_val = False
            max_util = new_max_util

        if np.isnan(max_util[total_charge]):
            raise Exception(
                "Solution not found! Please check that constraints are satisfiable."
            )
        curr_state, t_curr = total_charge, total_time

        schedule_reversed = []
        schedule_reversed.append(curr_state)
        while t_curr > 0:
            curr_state = path_history[t_curr - 1, curr_state]
            schedule_reversed.append(curr_state)
            t_curr -= 1
        optimal_path = np.array(schedule_reversed)[::-1]
        self.__optimal_charging_schedule = list(np.diff(optimal_path))
        self.__collect_results(moer)

    def __contiguous_fit(
        self,
        total_charge: int,
        total_time: int,
        moer: Moer,
        emission_multiplier_fn: Callable[[float, float], float],
        charge_per_interval: List[int] = [],
        constraints: Dict[int, Tuple[Optional[int], Optional[int]]] = {},
    ) -> None:
        """
        Performs a contiguous fit for charging schedule optimization using dynamic programming.

        This method implements a sophisticated optimization strategy that considers contiguous
        charging intervals. It uses dynamic programming to find an optimal charging schedule
        while respecting the specified length of each charging interval.

        Parameters:
        -----------
        total_charge : int
            The total amount of charge needed.
        total_time : int
            The total time available for charging.
        moer : Moer
            An object representing Marginal Operating Emissions Rate.
        emission_multiplier_fn : callable
            A function that calculates emission multipliers.
        charge_per_interval : list of int
            The exact charging amount per interval.
        constraints : dict, optional
            A dictionary of charging constraints for specific time steps. Constraints are one-indexed: t:(a,b) means that after t minutes, we have to have charged for between a and b minutes inclusive, so that 1<=t<=total_time

        Calls __collect_results to process the results.

        Raises:
        -------
        Exception
            If no valid solution is found.

        Note:
        -----
        This is the __diagonal_fit() algorithm with further constraint on contiguous charging intervals and their respective length
        """
        self.verbose_on("== Fixed contiguous fit! ==")
        total_interval = len(charge_per_interval)
        # This is a matrix with size = number of time states x number of intervals charged so far
        max_util = np.full((total_time + 1, total_interval + 1), np.nan)
        max_util[0, 0] = 0.0
        path_history = np.full((total_time, total_interval + 1), False, dtype=bool)
        cum_charge = [0]
        for c in charge_per_interval:
            cum_charge.append(cum_charge[-1] + c)

        charge_array_cache = [
            emission_multiplier_fn(x, x + 1) for x in range(0, total_charge + 1)
        ]
        # ("Cumulative charge", cum_charge)
        for t in range(1, total_time + 1):
            if t in constraints:
                min_charge, max_charge = constraints[t]
                min_charge = 0 if min_charge is None else max(0, min_charge)
                max_charge = (
                    total_charge
                    if max_charge is None
                    else min(max_charge, total_charge)
                )
                constraints[t] = (min_charge, max_charge)
            else:
                min_charge, max_charge = 0, total_charge
            for k in range(0, total_interval + 1):
                # print(t,k)
                ## not charging
                init_val = True
                if not np.isnan(max_util[t - 1, k]):
                    max_util[t, k] = max_util[t - 1, k]
                    init_val = False
                ## charging
                if (k > 0) and (charge_per_interval[k - 1] <= t):
                    dc = charge_per_interval[k - 1]
                    if not np.isnan(
                        max_util[t - dc, k - 1]
                    ) and OptCharger.__check_constraint(
                        t - dc, cum_charge[k - 1], dc, constraints
                    ):
                        marginal_cost = moer.get_emission_interval(
                            t - dc,
                            t,
                            charge_array_cache[cum_charge[k - 1] : cum_charge[k]],
                        )
                        new_util = max_util[t - dc, k - 1] - marginal_cost
                        if init_val or (new_util > max_util[t, k]):
                            max_util[t, k] = new_util
                            path_history[t - 1, k] = True
                        init_val = False

        if np.isnan(max_util[total_time, total_interval]):
            raise Exception(
                "Solution not found! Please check that constraints are satisfiable."
            )
        curr_state, t_curr = total_interval, total_time

        schedule_reversed = []
        interval_ids_reversed = []
        while t_curr > 0:
            delta_interval = path_history[t_curr - 1, curr_state]
            if not delta_interval:
                ## did not charge
                schedule_reversed.append(0)
                interval_ids_reversed.append(-1)
                t_curr -= 1
            else:
                ## charge
                dc = charge_per_interval[curr_state - 1]
                t_curr -= dc
                curr_state -= 1
                if dc > 0:
                    schedule_reversed.extend([1] * dc)
                    interval_ids_reversed.extend([curr_state] * dc)
        optimal_path = np.array(schedule_reversed)[::-1]
        self.__optimal_charging_schedule = list(optimal_path)
        self.__interval_ids = list(interval_ids_reversed[::-1])
        self.__collect_results(moer)

    def __variable_contiguous_fit(
        self,
        total_charge: int,
        total_time: int,
        moer: Moer,
        emission_multiplier_fn: Callable[[float, float], float],
        charge_per_interval: List[Tuple[int, int]] = [],
        use_all_intervals: bool = True,
        constraints: Dict[int, Tuple[Optional[int], Optional[int]]] = {},
    ) -> None:
        """
        Performs a contiguous fit for charging schedule optimization using dynamic programming.

        This method implements a sophisticated optimization strategy that considers contiguous
        charging intervals. It uses dynamic programming to find an optimal charging schedule
        while respecting constraints on the length of each charging interval.

        Parameters:
        -----------
        total_charge : int
            The total amount of charge needed.
        total_time : int
            The total time available for charging.
        moer : Moer
            An object representing Marginal Operating Emissions Rate.
        emission_multiplier_fn : callable
            A function that calculates emission multipliers.
        charge_per_interval : list of (int, int)
            The minimium and maximum (inclusive) charging amount per interval.
        use_all_intervals : bool
            If true, use all intervals provided by charge_per_interval; if false, can use the first few intervals and skip the rest.
        constraints : dict, optional
            A dictionary of charging constraints for specific time steps. Constraints are one-indexed: t:(a,b) means that after t minutes, we have to have charged for between a and b minutes inclusive, so that 1<=t<=total_time

        Calls __collect_results to process the results.

        Raises:
        -------
        Exception
            If no valid solution is found.

        Note:
        -----
        This is the __diagonal_fit() algorithm with further constraint on contiguous charging intervals and their respective length
        """
        self.verbose_on("== Variable contiguous fit! ==")
        total_interval = len(charge_per_interval)
        # This is a matrix with size = number of time states x number of charge states x number of intervals charged so far
        max_util = np.full(
            (total_time + 1, total_charge + 1, total_interval + 1), np.nan
        )
        max_util[0, 0, 0] = 0.0
        path_history = np.full(
            (total_time, total_charge + 1, total_interval + 1, 2), 0, dtype=int
        )

        charge_array_cache = [
            emission_multiplier_fn(x, x + 1) for x in range(0, total_charge + 1)
        ]

        for t in range(1, total_time + 1):
            if t in constraints:
                min_charge, max_charge = constraints[t]
                min_charge = 0 if min_charge is None else max(0, min_charge)
                max_charge = (
                    total_charge
                    if max_charge is None
                    else min(max_charge, total_charge)
                )
                constraints[t] = (min_charge, max_charge)
            else:
                min_charge, max_charge = 0, total_charge
            for k in range(0, total_interval + 1):
                for c in range(min_charge, max_charge + 1):
                    ## not charging
                    init_val = True
                    if not np.isnan(max_util[t - 1, c, k]):
                        max_util[t, c, k] = max_util[t - 1, c, k]
                        path_history[t - 1, c, k, :] = [0, 0]
                        init_val = False
                    ## charging
                    if k > 0:
                        for dc in range(
                            charge_per_interval[k - 1][0],
                            min(charge_per_interval[k - 1][1], t, c) + 1,
                        ):
                            if not np.isnan(
                                max_util[t - dc, c - dc, k - 1]
                            ) and OptCharger.__check_constraint(
                                t - dc, c - dc, dc, constraints
                            ):
                                marginal_cost = moer.get_emission_interval