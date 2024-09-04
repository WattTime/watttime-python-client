# optCharger.py
import numpy as np
from .moer import Moer

TOL = 1e-4  # tolerance
EMISSION_FN_TOL = 1e-9  # emissions functions tolerance in kw


class OptCharger:
    """
    Represents an Optimal Charger for managing electric vehicle charging schedules.

    This class handles the optimization of charging schedules based on various parameters
    such as charge rates, emission overheads, and other constraints.

    Attributes:
    -----------
    minChargeRate : int
        The minimum charging rate allowed.
    maxChargeRate : int
        The maximum charging rate allowed.
    emissionOverhead : bool
        Indicates whether emission overhead is considered in calculations.
    startEmissionOverhead : float
        The emission overhead for starting a charging session.
    keepEmissionOverhead : float
        The emission overhead for maintaining a charging session.
    stopEmissionOverhead : float
        The emission overhead for stopping a charging session.
    __optimalChargingEmission : float
        The calculated optimal charging emission (private attribute).
    __optimalTotalEmission : float
        The calculated total optimal emission including overheads (private attribute).
    __optimalChargingSchedule : list
        The optimal charging schedule determined by the optimization process (private attribute).

    Methods:
    --------
    __init__(fixedChargeRate=None, minChargeRate=None, maxChargeRate=None,
             emissionOverhead=False, startEmissionOverhead=0.0,
             keepEmissionOverhead=0.0, stopEmissionOverhead=0.0)
        Initializes the OptCharger object with the given parameters.
    """

    def __init__(
        self,
        fixedChargeRate: int = None,
        minChargeRate: int = None,
        maxChargeRate: int = None,
        emissionOverhead: bool = False,
        startEmissionOverhead: float = 0.0,
        keepEmissionOverhead: float = 0.0,
        stopEmissionOverhead: float = 0.0,
    ):
        """
        Initializes the OptCharger object.

        Parameters:
        -----------
        fixedChargeRate : int, optional
            If provided, sets both minChargeRate and maxChargeRate to this value.
        minChargeRate : int, optional
            The minimum charging rate. Used if fixedChargeRate is not provided.
        maxChargeRate : int, optional
            The maximum charging rate. Used if fixedChargeRate is not provided.
        emissionOverhead : bool, optional
            Whether to consider emission overheads in calculations. Default is False.
        startEmissionOverhead : float, optional
            The emission overhead for starting a charging session. Default is 0.0.
        keepEmissionOverhead : float, optional
            The emission overhead for maintaining a charging session. Default is 0.0.
        stopEmissionOverhead : float, optional
            The emission overhead for stopping a charging session. Default is 0.0.

        Note:
        -----
        The method uses a tolerance value (TOL) to determine if emission overheads
        are significant enough to be considered in calculations.
        """
        if fixedChargeRate is not None:
            self.minChargeRate = fixedChargeRate
            self.maxChargeRate = fixedChargeRate
        else:
            self.minChargeRate = minChargeRate
            self.maxChargeRate = maxChargeRate
        if emissionOverhead:
            self.startEmissionOverhead = startEmissionOverhead
            self.keepEmissionOverhead = keepEmissionOverhead
            self.stopEmissionOverhead = stopEmissionOverhead
            self.emissionOverhead = (
                (startEmissionOverhead > TOL)
                or (keepEmissionOverhead > TOL)
                or (stopEmissionOverhead > TOL)
            )
        else:
            self.emissionOverhead = False
            self.startEmissionOverhead = 0.0
            self.keepEmissionOverhead = 0.0
            self.stopEmissionOverhead = 0.0
        self.__optimalChargingEmission = None
        self.__optimalTotalEmission = None
        self.__optimalChargingSchedule = None

    def __collect_results(self, moer: Moer):
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
        - __optimalChargingEnergyOverTime
        - __optimalChargingEmissionsOverTime
        - __optimalChargingEmission
        - __optimalTotalEmission

        The function also populates the emission_multipliers list, which is used in the calculations.
        """
        emission_multipliers = []
        current_charge_time_units = 0
        for i in range(len(self.__optimalChargingSchedule)):
            if self.__optimalChargingSchedule[i] == 0:
                emission_multipliers.append(0.0)
            else:
                old_charge_time_units = current_charge_time_units
                current_charge_time_units += self.__optimalChargingSchedule[i]
                power_rate = self.emission_multiplier_fn(
                    old_charge_time_units, current_charge_time_units
                )
                emission_multipliers.append(power_rate)

        self.__optimalChargingEnergyOverTime = np.array(
            self.__optimalChargingSchedule
        ) * np.array(emission_multipliers)
        self.__optimalChargingEmissionsOverTime = moer.get_emissions(
            np.array(self.__optimalChargingSchedule) * np.array(emission_multipliers)
        )
        self.__optimalChargingEmission = moer.get_total_emission(
            np.array(self.__optimalChargingSchedule) * np.array(emission_multipliers)
        )
        y = np.hstack((0, self.__optimalOnOffSchedule, 0))
        yDiff = y[1:] - y[:-1]
        self.__optimalTotalEmission = (
            self.__optimalChargingEmission
            + np.sum(y) * self.keepEmissionOverhead
            + np.sum(yDiff == 1) * self.startEmissionOverhead
            + np.sum(yDiff == -1) * self.stopEmissionOverhead
        )

    @staticmethod
    def __sanitize_emission_multiplier(emission_multiplier_fn, totalCharge):
        """
        Sanitizes the emission multiplier function to handle edge cases and ensure valid outputs.

        This function wraps the original emission_multiplier_fn to handle cases where the
        end charge (ec) exceeds the total charge or when the start charge (sc) is beyond
        the total charge limit.

        Parameters:
        -----------
        emission_multiplier_fn : callable
            The original emission multiplier function to be sanitized.
        totalCharge : int or float
            The maximum total charge value.

        Returns:
        --------
        callable
            A new lambda function that sanitizes the inputs before calling the original
            emission_multiplier_fn.

        Behavior:
        ---------
        - If sc < totalCharge:
            - Calls the original function with ec capped at totalCharge.
        - If sc >= totalCharge:
            - Returns 1.0, assuming no additional emissions beyond total charge.

        Note:
        -----
        This function is useful for preventing out-of-bounds errors and ensuring
        consistent behavior when dealing with charge values near or beyond the total
        charge limit.
        """
        return lambda sc, ec: (
            emission_multiplier_fn(sc, min(ec, totalCharge))
            if (sc < totalCharge)
            else 1.0
        )

    def __greedy_fit(self, totalCharge: int, totalTime: int, moer: Moer):
        """
        Performs a "greedy" fit for charging schedule optimization.

        It charges at the maximum possible rate until the total charge is reached or
        the time limit is hit.

        Parameters:
        -----------
        totalCharge : int
            The total amount of charge needed.
        totalTime : int
            The total time available for charging.
        moer : Moer
            An object representing Marginal Operating Emissions Rate.

        Side Effects:
        -------------
        Updates the following instance variables:
        - __optimalChargingSchedule
        - __optimalOnOffSchedule

        Calls __collect_results to process the results.
        """
        print("== Baseline fit! ==")
        chargeToDo = totalCharge
        cs, t = [], 0
        while (chargeToDo > 0) and (t < totalTime):
            c = max(min(self.maxChargeRate, chargeToDo), self.minChargeRate)
            chargeToDo -= c
            cs.append(c)
            t += 1
        self.__optimalChargingSchedule = cs + [0] * (totalTime - t)
        self.__optimalOnOffSchedule = [1] * t + [0] * (totalTime - t)
        self.__collect_results(moer)

    def __simple_fit(self, totalCharge: int, totalTime: int, moer: Moer):
        """
        Performs a "simple" fit for charging schedule optimization.

        This method implements a straightforward optimization strategy. It sorts
        time intervals by MOER (Marginal Operating Emissions Rate) and charges
        during the cleanest intervals until the total charge is reached.

        Parameters:
        -----------
        totalCharge : int
            The total amount of charge needed.
        totalTime : int
            The total time available for charging.
        moer : Moer
            An object representing Marginal Operating Emissions Rate.

        Side Effects:
        -------------
        Updates the following instance variables:
        - __optimalChargingSchedule
        - __optimalOnOffSchedule

        Calls __collect_results to process the results.
        """
        print("== Simple fit! ==")
        sorted_times = [
            x
            for _, x in sorted(
                zip(moer.get_emission_interval(0, totalTime), range(totalTime))
            )
        ]
        chargeToDo = totalCharge
        cs, schedule, t = [0] * totalTime, [0] * totalTime, 0
        while (chargeToDo > 0) and (t < totalTime):
            c = max(min(self.maxChargeRate, chargeToDo), self.minChargeRate)
            chargeToDo -= c
            cs[sorted_times[t]] = c
            schedule[sorted_times[t]] = 1
            t += 1
        self.__optimalChargingSchedule = cs
        self.__optimalOnOffSchedule = schedule
        self.__collect_results(moer)

    def __diagonal_fit(
        self,
        totalCharge: int,
        totalTime: int,
        moer: Moer,
        emission_multiplier_fn,
        ra: float = 0.0,
        constraints: dict = {},
    ):
        """
        Performs a sophisticated diagonal fit for charging schedule optimization using dynamic programming.

        This method implements a more complex optimization strategy using dynamic programming.
        It considers various factors such as emission rates, charging constraints, and overhead costs
        to find an optimal charging schedule.

        Parameters:
        -----------
        totalCharge : int
            The total amount of charge needed.
        totalTime : int
            The total time available for charging.
        moer : Moer
            An object representing Marginal Operating Emissions Rate.
        emission_multiplier_fn : callable
            A function that calculates emission multipliers.
        ra : float, optional
            Risk aversion factor. Default is 0.
        constraints : dict, optional
            A dictionary of charging constraints for specific time steps.

        Side Effects:
        -------------
        Updates the following instance variables:
        - __optimalChargingSchedule
        - __optimalOnOffSchedule

        Calls __collect_results to process the results.

        Raises:
        -------
        Exception
            If no valid solution is found.

        Note:
        -----
        This method uses a complex dynamic programming approach to optimize the
        charging schedule considering various factors and constraints.
        """
        # maxUtil[{0..totalCharge},{0,1}] = emission (with risk penalty)
        # path[t,{0..totalCharge},{0,1},:] = [charge,{0,1]}]
        """
        This is the DP algorithm 
        """
        print("== Sophisticated fit! ==")
        # This is a matrix with size = number of charge states x number of actions {not charging = 0, charging = 1}
        maxUtil = np.full((totalCharge + 1, 2), np.nan)
        maxUtil[0, 0] = 0.0
        pathHistory = np.full((totalTime, totalCharge + 1, 2, 2), 0, dtype=int)
        for t in range(totalTime):
            if t in constraints:
                minCharge, maxCharge = constraints[t]
                minCharge = 0 if minCharge is None else max(0, minCharge)
                maxCharge = (
                    totalCharge if maxCharge is None else min(maxCharge, totalCharge)
                )
            else:
                minCharge, maxCharge = 0, totalCharge
            # print("=== Time step", t, "===")
            newMaxUtil = np.full(maxUtil.shape, np.nan)
            for c in range(minCharge, maxCharge + 1):
                ## update (c,0)
                # print("-- charge", c, "| charging off --")
                initVal = True
                if not np.isnan(maxUtil[c, 0]):
                    newMaxUtil[c, 0] = maxUtil[c, 0]
                    pathHistory[t, c, 0, :] = [c, 0]
                    initVal = False
                if not np.isnan(maxUtil[c, 1]):
                    newUtil = maxUtil[c, 1] - self.stopEmissionOverhead
                    if initVal or (newUtil > newMaxUtil[c, 0]):
                        newMaxUtil[c, 0] = newUtil
                        pathHistory[t, c, 0, :] = [c, 1]

                ## update (c,1)
                # print("-- charge", c, "| charging on --")
                initVal = True
                for ct in range(self.minChargeRate, min(c, self.maxChargeRate) + 1):
                    if not np.isnan(maxUtil[c - ct, 0]):
                        # moer.get_marginal_util gives lbs/MWh. emission function needs to be how many MWh the interval consumes
                        # which would be power_in_kW * 0.001 * 5/60
                        newUtil = (
                            maxUtil[c - ct, 0]
                            - moer.get_marginal_util(ct, t, ra=ra)
                            * emission_multiplier_fn(c - ct, c)
                            - self.startEmissionOverhead
                            - self.keepEmissionOverhead
                        )
                        if initVal or (newUtil > newMaxUtil[c, 1]):
                            newMaxUtil[c, 1] = newUtil
                            pathHistory[t, c, 1, :] = [c - ct, 0]
                        initVal = False
                    if not np.isnan(maxUtil[c - ct, 1]):
                        newUtil = (
                            maxUtil[c - ct, 1]
                            - moer.get_marginal_util(ct, t, ra=ra)
                            * emission_multiplier_fn(c - ct, c)
                            - self.keepEmissionOverhead
                        )
                        if initVal or (newUtil > newMaxUtil[c, 1]):
                            newMaxUtil[c, 1] = newUtil
                            pathHistory[t, c, 1, :] = [c - ct, 1]
                        initVal = False
            maxUtil = newMaxUtil
            # print(maxUtil)

        solution_found = False
        if not np.isnan(maxUtil[totalCharge, 0]):
            max_util = maxUtil[totalCharge, 0]
            m_final = 0
            solution_found = True
        if not np.isnan(maxUtil[totalCharge, 1]):
            newUtil = maxUtil[totalCharge, 1] - self.stopEmissionOverhead
            if not solution_found or (newUtil > max_util):
                max_util = newUtil
                m_final = 1
            solution_found = True
        if not solution_found:
            ## TODO: In this case we should still return the best possible plan
            ## which would probably to just charge for the entire window
            raise Exception("Solution not found!")
        curr_state, t_curr = [totalCharge, m_final], totalTime - 1
        # This gives the schedule in reverse
        schedule = []
        schedule.append(curr_state)
        while t_curr >= 0:
            curr_state = pathHistory[t_curr, curr_state[0], curr_state[1], :]
            schedule.append(curr_state)
            t_curr -= 1
        optimalPath = np.array(schedule)[::-1, :]
        self.__optimalChargingSchedule = list(np.diff(optimalPath[:, 0]))
        self.__optimalOnOffSchedule = optimalPath[:, 1]
        self.__collect_results(moer)

    def __contiguous_fit(
        self,
        totalCharge: int,
        totalTime: int,
        moer: Moer,
        emission_multiplier_fn,
        totalIntervals: int = 1,
        ra: float = 0.0,
        constraints: dict = {},
    ):
        """
        Performs a contiguous fit for charging schedule optimization using dynamic programming.

        This method implements a sophisticated optimization strategy that considers contiguous
        charging intervals. It uses dynamic programming to find an optimal charging schedule
        while respecting constraints on the number of charging intervals.

        Parameters:
        -----------
        totalCharge : int
            The total amount of charge needed.
        totalTime : int
            The total time available for charging.
        moer : Moer
            An object representing Marginal Operating Emissions Rate.
        emission_multiplier_fn : callable
            A function that calculates emission multipliers.
        totalIntervals : int, optional
            The maximum number of contiguous charging intervals allowed. Default is 1.
        ra : float, optional
            Risk aversion factor. Default is 0.
        constraints : dict, optional
            A dictionary of charging constraints for specific time steps.

        Side Effects:
        -------------
        Updates the following instance variables:
        - __optimalChargingSchedule
        - __optimalOnOffSchedule

        Calls __collect_results to process the results.

        Raises:
        -------
        Exception
            If no valid solution is found.

        Note:
        -----
        This method uses a complex dynamic programming approach to optimize the
        charging schedule considering contiguous intervals and various constraints.
        """
        print("== Sophisticated contiguous fit! ==")
        # This is a matrix with size = number of charge states x number of actions {not charging = 0, charging = 1}
        maxUtil = np.full((totalCharge + 1, 2, totalIntervals + 1), np.nan)
        maxUtil[0, 0, 0] = 0.0
        pathHistory = np.full(
            (totalTime, totalCharge + 1, 2, totalIntervals + 1, 3), 0, dtype=int
        )
        for t in range(totalTime):
            if t in constraints:
                minCharge, maxCharge = constraints[t]
                minCharge = 0 if minCharge is None else max(0, minCharge)
                maxCharge = (
                    totalCharge if maxCharge is None else min(maxCharge, totalCharge)
                )
            else:
                minCharge, maxCharge = 0, totalCharge
            newMaxUtil = np.full(maxUtil.shape, np.nan)
            for k in range(0, totalIntervals + 1):
                for c in range(minCharge, maxCharge + 1):
                    ## update (c,0,k)
                    initVal = True
                    if not np.isnan(maxUtil[c, 0, k]):
                        newMaxUtil[c, 0, k] = maxUtil[c, 0, k]
                        pathHistory[t, c, 0, k, :] = [c, 0, k]
                        initVal = False
                    if not np.isnan(maxUtil[c, 1, k]):
                        newUtil = maxUtil[c, 1, k] - self.stopEmissionOverhead
                        if initVal or (newUtil > newMaxUtil[c, 0, k]):
                            newMaxUtil[c, 0, k] = newUtil
                            pathHistory[t, c, 0, k, :] = [c, 1, k]
                    ## update (c,1,k)
                    if k == 0:
                        # If charging is on, we must have k > 0
                        continue
                    initVal = True
                    for ct in range(self.minChargeRate, min(c, self.maxChargeRate) + 1):
                        if not np.isnan(maxUtil[c - ct, 0, k - 1]):
                            newUtil = (
                                maxUtil[c - ct, 0, k - 1]
                                - moer.get_marginal_util(ct, t, ra=ra)
                                * emission_multiplier_fn(c - ct, c)
                                - self.startEmissionOverhead
                                - self.keepEmissionOverhead
                            )
                            if initVal or (newUtil > newMaxUtil[c, 1, k]):
                                newMaxUtil[c, 1, k] = newUtil
                                pathHistory[t, c, 1, k, :] = [c - ct, 0, k - 1]
                            initVal = False
                        if not np.isnan(maxUtil[c - ct, 1, k]):
                            newUtil = (
                                maxUtil[c - ct, 1, k]
                                - moer.get_marginal_util(ct, t, ra=ra)
                                * emission_multiplier_fn(c - ct, c)
                                - self.keepEmissionOverhead
                            )
                            if initVal or (newUtil > newMaxUtil[c, 1, k]):
                                newMaxUtil[c, 1, k] = newUtil
                                pathHistory[t, c, 1, k, :] = [c - ct, 1, k]
                            initVal = False
            maxUtil = newMaxUtil

        solution_found = False
        for k in range(0, totalIntervals + 1):
            if not np.isnan(maxUtil[totalCharge, 0, k]):
                newUtil = maxUtil[totalCharge, 0, k]
                if not solution_found or (newUtil > max_util):
                    max_util = newUtil
                    m_final = (0, k)
                solution_found = True
            if not np.isnan(maxUtil[totalCharge, 1, k]):
                newUtil = maxUtil[totalCharge, 1, k] - self.stopEmissionOverhead
                if not solution_found or (newUtil > max_util):
                    max_util = newUtil
                    m_final = (1, k)
                solution_found = True
        if not solution_found:
            ## TODO: In this case we should still return the best possible plan
            ## which would probably to just charge for the entire window
            raise Exception("Solution not found!")
        curr_state, t_curr = [totalCharge, *m_final], totalTime - 1
        # This gives the schedule in reverse
        schedule = []
        schedule.append(curr_state)
        while t_curr >= 0:
            curr_state = pathHistory[
                t_curr, curr_state[0], curr_state[1], curr_state[2], :
            ]
            schedule.append(curr_state)
            t_curr -= 1
        optimalPath = np.array(schedule)[::-1, :]
        self.__optimalChargingSchedule = list(np.diff(optimalPath[:, 0]))
        self.__optimalOnOffSchedule = optimalPath[:, 1]
        self.__collect_results(moer)

    def fit(
        self,
        totalCharge: int,
        totalTime: int,
        moer: Moer,
        totalIntervals: int = 0,
        constraints: dict = {},
        ra: float = 0.0,
        emission_multiplier_fn=None,
        optimization_method: str = "auto",
    ):
        """
        Fits an optimal charging schedule based on the given parameters and constraints.

        This method serves as the main entry point for the charging optimization process.
        It selects the appropriate optimization method based on the input parameters and
        constraints.

        Parameters:
        -----------
        totalCharge : int
            The total amount of charge needed.
        totalTime : int
            The total time available for charging.
        moer : Moer
            An object representing Marginal Operating Emissions Rate.
        totalIntervals : int, optional
            The maximum number of contiguous charging intervals allowed. Default is 0 (no limit).
        constraints : dict, optional
            A dictionary of charging constraints for specific time steps.
        ra : float, optional
            Risk aversion factor. Default is 0.
        emission_multiplier_fn : callable, optional
            A function that calculates emission multipliers. If None, assumes constant 1kW power usage.
        optimization_method : str, optional
            The optimization method to use. Can be 'auto', 'baseline', 'simple', or 'sophisticated'.
            Default is 'auto'.

        Raises:
        -------
        Exception
            If the charging task is impossible given the constraints, or if an unsupported
            optimization method is specified.

        Note:
        -----
        This method chooses between different optimization strategies based on the input
        parameters and the characteristics of the problem.
        """
        assert len(moer) >= totalTime
        if emission_multiplier_fn is None:
            print(
                "Warning: OptCharger did not get an emission_multiplier_fn. Assuming that device uses constant 1kW of power"
            )
            emission_multiplier_fn = lambda sc, ec: 1.0
        # Store emission_multiplier_fn for evaluation
        self.emission_multiplier_fn = emission_multiplier_fn
        constant_emission_multiplier = (
            np.std(
                [emission_multiplier_fn(sc, sc + 1) for sc in list(range(totalCharge))]
            )
            < EMISSION_FN_TOL
        )
        if totalCharge > totalTime * self.maxChargeRate:
            raise Exception(
                f"Impossible to charge {totalCharge} within {totalTime} intervals."
            )
        if optimization_method == "baseline":
            self.__greedy_fit(totalCharge, totalTime, moer)
        elif (
            (optimization_method == "auto")
            and (
                not self.emissionOverhead
                and ra < TOL
                and not constraints
                and totalIntervals <= 0
                and constant_emission_multiplier
            )
            or (optimization_method == "simple")
        ):
            if not constant_emission_multiplier:
                print(
                    "Warning: Emissions function is non-constant. Using the simple algorithm is suboptimal."
                )
            self.__simple_fit(totalCharge, totalTime, moer)
        elif (
            (optimization_method == "auto")
            and moer.is_diagonal()
            or (optimization_method == "sophisticated")
        ):
            if totalIntervals <= 0:
                self.__diagonal_fit(
                    totalCharge,
                    totalTime,
                    moer,
                    OptCharger.__sanitize_emission_multiplier(
                        emission_multiplier_fn, totalCharge
                    ),
                    ra,
                    constraints,
                )
            else:
                self.__contiguous_fit(
                    totalCharge,
                    totalTime,
                    moer,
                    OptCharger.__sanitize_emission_multiplier(
                        emission_multiplier_fn, totalCharge
                    ),
                    totalIntervals,
                    ra,
                    constraints,
                )
        else:
            raise Exception("Non diagonal risk not implemented!")

    def get_energy_usage_over_time(self) -> list:
        """
        Returns:
        --------
        list
            The energy due to charging at each interval in MWh.
        """
        return self.__optimalChargingEnergyOverTime

    def get_charging_emissions_over_time(self) -> list:
        """
        Returns:
        --------
        list
            The emissions due to charging at each interval in lbs.
        """
        return self.__optimalChargingEmissionsOverTime

    def get_charging_emission(self) -> float:
        """
        Returns:
        --------
        float
            The summed emissions due to charging in lbs.
            This excludes penalty terms due to risk aversion.
        """
        return self.__optimalChargingEmission

    def get_total_emission(self) -> float:
        """
        Returns:
        --------
        float
            The summed emissions due to charging and penalty terms in lbs.
        """
        return self.__optimalTotalEmission

    def get_schedule(self) -> list:
        """
        Returns the optimal charging schedule.

        Returns:
        --------
        list
            The charging schedule as a list, in minutes to charge for each interval.
        """
        return self.__optimalChargingSchedule

    def summary(self):
        print("-- Model Summary --")
        print("Expected charging emissions: %.2f lbs" % self.__optimalChargingEmission)
        print("Expected total emissions: %.2f lbs" % self.__optimalTotalEmission)
        print("Optimal charging schedule:", self.__optimalChargingSchedule)
        print("=" * 15)
