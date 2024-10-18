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

    Methods:
    --------
    __init__()
        Initializes the OptCharger object with the given parameters.
    """

    def __init__(self):
        """
        Initializes the OptCharger object.

        Parameters:
        -----------
        fixedChargeRate : int, optional
            If provided, sets both minChargeRate and maxChargeRate to this value.

        Note:
        -----
        The method uses a tolerance value (TOL) to determine if emission overheads
        are significant enough to be considered in calculations.
        """
        self.__optimalChargingEmission = None
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

        Calls __collect_results to process the results.
        """
        print("== Baseline fit! ==")
        chargeToDo = totalCharge
        cs, t = [], 0
        while (chargeToDo > 0) and (t < totalTime):
            chargeToDo -= 1
            cs.append(1)
            t += 1
        self.__optimalChargingSchedule = cs + [0] * (totalTime - t)
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

        Calls __collect_results to process the results.
        """
        print("== Simple fit! ==")
        sorted_times = [
            x
            for _, x in sorted(
                zip(moer.get_emission_interval(0, totalTime,1), range(totalTime))
            )
        ]
        chargeToDo = totalCharge
        cs, schedule, t = [0] * totalTime, [0] * totalTime, 0
        while (chargeToDo > 0) and (t < totalTime):
            chargeToDo -= 1
            cs[sorted_times[t]] = 1
            schedule[sorted_times[t]] = 1
            t += 1
        self.__optimalChargingSchedule = cs
        self.__collect_results(moer)

    def __diagonal_fit(
        self,
        totalCharge: int,
        totalTime: int,
        moer: Moer,
        emission_multiplier_fn,
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
        constraints : dict, optional
            A dictionary of charging constraints for specific time steps.

        Calls __collect_results to process the results.

        Raises:
        -------
        Exception
            If no valid solution is found.
        """
        print("== Sophisticated fit! ==")
        # This is a matrix with size = number of charge states x number of actions {not charging = 0, charging = 1}
        maxUtil = np.full((totalCharge + 1), np.nan)
        maxUtil[0] = 0.0
        pathHistory = np.full((totalTime, totalCharge + 1), -1, dtype=int)
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
                ## Do not charge
                initVal = True
                if not np.isnan(maxUtil[c]):
                    newMaxUtil[c] = maxUtil[c]
                    pathHistory[t, c] = [c]
                    initVal = False
                if not np.isnan(maxUtil[c-1]):
                    # moer.get_emission_at gives lbs/MWh. emission function needs to be how many MWh the interval consumes
                    # which would be power_in_kW * 0.001 * 5/60
                    newUtil = maxUtil[c-1]-moer.get_emission_at(t, emission_multiplier_fn(c-1,c))
                    if initVal or (newUtil > newMaxUtil[c]):
                        newMaxUtil[c] = newUtil
                        pathHistory[t, c] = c-1
                    initVal = False
            maxUtil = newMaxUtil
        
        if not np.isnan(maxUtil[totalCharge]):
            maxUtil = maxUtil[totalCharge]
        else:
            ## TODO: In this case we should still return the best possible plan
            ## which would probably to just charge for the entire window
            raise Exception("Solution not found!")
        curr_state, t_curr = totalCharge, totalTime - 1
        # This gives the schedule in reverse
        schedule = []
        schedule.append(curr_state)
        while t_curr >= 0:
            curr_state = pathHistory[t_curr, curr_state]
            schedule.append(curr_state)
            t_curr -= 1
        optimalPath = np.array(schedule)[::-1]
        self.__optimalChargingSchedule = list(np.diff(optimalPath))
        self.__collect_results(moer)

    def __contiguous_fit(
        self,
        totalCharge: int,
        totalTime: int,
        moer: Moer,
        emission_multiplier_fn,
        charge_per_interval: list = [],
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
        charge_per_interval : int, optional
            The maximum number of contiguous charging intervals allowed and their lengths.
        constraints : dict, optional
            A dictionary of charging constraints for specific time steps.

        Calls __collect_results to process the results.

        Raises:
        -------
        Exception
            If no valid solution is found.

        Note:
        -----
        This is the __diagonal_fit() algorithm with further constraint on # intervals  
        """
        print("== Variable contiguous fit! ==")
        # This is a matrix with size = number of charge states x number of actions {not charging = 0, charging = 1}
        totalInterval = len(charge_per_interval)
        maxUtil = np.full((totalTime+1,totalCharge+1,totalInterval+1), np.nan)
        maxUtil[0, 0, 0] = 0.0
        pathHistory = np.full(
            (totalTime, totalCharge + 1, totalInterval + 1, 2), 0, dtype=int
        )
        for t in range(1, totalTime+1):
            if t-1 in constraints:
                minCharge, maxCharge = constraints[t-1]
                minCharge = 0 if minCharge is None else max(0, minCharge)
                maxCharge = totalCharge if maxCharge is None else min(maxCharge, totalCharge)
                constraints[t-1] = (minCharge, maxCharge)
            else:
                minCharge, maxCharge = 0, totalCharge
            for k in range(0, totalInterval + 1):
                for c in range(minCharge, maxCharge + 1):
                    # print(t,c,k)
                    ## not charging
                    initVal = True
                    if not np.isnan(maxUtil[t-1, c, k]):
                        maxUtil[t, c, k] = maxUtil[t-1, c, k]
                        pathHistory[t-1, c, k, :] = [0,0]
                        initVal = False
                    ## charging
                    if k > 0: 
                        for dc in range(charge_per_interval[k-1][0],min(charge_per_interval[k-1][1],t,c)+1):
                            if not np.isnan(maxUtil[t-dc,c-dc,k-1]) and OptCharger.__check_constraint(t-dc,c-dc,dc,constraints): 
                                marginalcost = moer.get_emission_interval(t-dc,t,OptCharger.__avg_to_interval(emission_multiplier_fn,c-dc,c))
                                newUtil = maxUtil[t-dc,c-dc,k-1] - marginalcost
                                if initVal or (newUtil > maxUtil[t,c,k]): 
                                    maxUtil[t,c,k] = newUtil
                                    pathHistory[t-1,c,k,:] = [dc,1]
                                initVal = False
                            
        if np.isnan(maxUtil[totalTime,totalCharge,totalInterval]): 
            ## TODO: In this case we should still return the best possible plan
            ## which would probably to just charge for the entire window
            raise Exception("Solution not found!")
        curr_state, t_curr = [totalCharge,totalInterval], totalTime
        # This gives the schedule in reverse
        schedule = []
        while t_curr > 0: 
            dc,di = pathHistory[t_curr-1, curr_state[0], curr_state[1], :]
            if di==0: 
                ## did not charge 
                schedule.append(0)
                t_curr -= 1  
            else: 
                ## charge
                t_curr -= dc
                curr_state = [curr_state[0]-dc, curr_state[1]-di]
                if dc>0: 
                    schedule.extend([1]*dc)         
        optimalPath = np.array(schedule)[::-1]
        self.__optimalChargingSchedule = optimalPath
        self.__collect_results(moer)

    def fit(
        self,
        totalCharge: int,
        totalTime: int,
        moer: Moer,
        charge_per_interval=None,
        constraints: dict = {},
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
        charge_per_interval : list, optional
            The maximum number of contiguous charging intervals allowed. Default is 0 (no limit).
        constraints : dict, optional
            A dictionary of charging constraints for specific time steps.
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
        assert optimization_method in ['baseline','simple','sophisticated','auto']
        if emission_multiplier_fn is None:
            print(
                "Warning: OptCharger did not get an emission_multiplier_fn. Assuming that device uses constant 1kW of power"
            )
            emission_multiplier_fn = lambda sc, ec: 1.0
            constant_emission_multiplier = True
        else:
            constant_emission_multiplier = np.std([emission_multiplier_fn(sc,sc+1) for sc in list(range(totalCharge))]) < EMISSION_FN_TOL
        # Store emission_multiplier_fn for evaluation
        self.emission_multiplier_fn = emission_multiplier_fn
        if totalCharge > totalTime * self.maxChargeRate:
            raise Exception(
                f"Impossible to charge {totalCharge} within {totalTime} intervals."
            )
        if optimization_method == "baseline":
            self.__greedy_fit(totalCharge, totalTime, moer)
        elif (
            not constraints
            and not charge_per_interval 
            and constant_emission_multiplier
            and optimization_method=='auto'
         ) or (optimization_method=='simple'):
            if not constant_emission_multiplier:
                print("Warning: Emissions function is non-constant. Using the simple algorithm is suboptimal.")
            self.__simple_fit(totalCharge, totalTime, moer)
        elif not charge_per_interval:
            self.__diagonal_fit(
                totalCharge,
                totalTime,
                moer,
                OptCharger.__sanitize_emission_multiplier(
                    emission_multiplier_fn, totalCharge
                    ), constraints
                )
        else:
            self.__contiguous_fit(
                totalCharge,
                totalTime,
                moer,
                OptCharger.__sanitize_emission_multiplier(
                    emission_multiplier_fn, totalCharge
                    ), constraints
                )

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
        print("Optimal charging schedule:", self.__optimalChargingSchedule)
        print("=" * 15)
