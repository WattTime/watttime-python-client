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
        """
        self.__optimal_charging_emission = None
        self.__optimal_charging_schedule = None

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
        - __optimal_charging_energy_over_time
        - __optimal_charging_emissions_over_time
        - __optimal_charging_emission

        The function also populates the emission_multipliers list, which is used in the calculations.
        """
        emission_multipliers = []
        current_charge_time_units = 0
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

        self.__optimal_charging_energy_over_time = np.array(self.__optimal_charging_schedule) * np.array(emission_multipliers)
        self.__optimal_charging_emissions_over_time = moer.get_emissions(
            self.__optimal_charging_energy_over_time
        )
        self.__optimal_charging_emission = self.__optimal_charging_emissions_over_time.sum()

    @staticmethod
    def __sanitize_emission_multiplier(emission_multiplier_fn, total_charge):
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
    def __check_constraint(t_start, c_start, dc, constraints): 
        # assuming constraints[t] is the bound on total charge after t intervals
        for t in range(t_start+1, t_start+dc): 
            if (t in constraints) and ((c_start+t-t_start < constraints[t][0]) or (c_start+t-t_start > constraints[t][1])): 
                return False
        return True

    def __greedy_fit(self, total_charge: int, total_time: int, moer: Moer):
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
        print("== Baseline fit! ==")
        schedule = [1] * min(total_charge, total_time) + [0] * max(0, total_time - total_charge)
        self.__optimal_charging_schedule = schedule
        self.__collect_results(moer)

    def __simple_fit(self, total_charge: int, total_time: int, moer: Moer):
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
        print("== Simple fit! ==")
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
        print("== Sophisticated fit! ==")
        # This is a matrix with size = number of charge states x number of actions {not charging = 0, charging = 1}
        max_util = np.full((total_charge + 1), np.nan)
        max_util[0] = 0.0
        path_history = np.full((total_time, total_charge + 1), -1, dtype=int)
        for t in range(1,total_time+1):
            if t in constraints:
                min_charge, max_charge = constraints[t]
                min_charge = 0 if min_charge is None else max(0, min_charge)
                max_charge = (
                    total_charge if max_charge is None else min(max_charge, total_charge)
                )
            else:
                min_charge, max_charge = 0, total_charge
            # print("=== Time step", t, "===")
            new_max_util = np.full(max_util.shape, np.nan)
            # print("min_charge, max_charge =",min_charge,max_charge)
            for c in range(min_charge, max_charge + 1):
                ## not charging
                init_val = True
                if not np.isnan(max_util[c]):
                    new_max_util[c] = max_util[c]
                    path_history[t-1, c] = c
                    init_val = False
                ## charging
                if (c>0) and not np.isnan(max_util[c-1]):
                    # moer.get_emission_at gives lbs/MWh. emission function needs to be how many MWh the interval consumes
                    # which would be power_in_kW * 0.001 * 5/60
                    new_util = max_util[c-1]-moer.get_emission_at(t-1, emission_multiplier_fn(c-1,c))
                    if init_val or (new_util > new_max_util[c]):
                        new_max_util[c] = new_util
                        path_history[t-1, c] = c-1
                    init_val = False
            max_util = new_max_util
        
        if np.isnan(max_util[total_charge]):
            ## TODO: In this case we should still return the best possible plan
            ## which would probably to just charge for the entire window
            raise Exception("Solution not found!")
        curr_state, t_curr = total_charge, total_time
        # This gives the schedule in reverse
        schedule = []
        schedule.append(curr_state)
        while t_curr > 0:
            curr_state = path_history[t_curr-1, curr_state]
            schedule.append(curr_state)
            t_curr -= 1
        optimal_path = np.array(schedule)[::-1]
        self.__optimal_charging_schedule = list(np.diff(optimal_path))
        self.__collect_results(moer)

    def __contiguous_fit(
        self,
        total_charge: int,
        total_time: int,
        moer: Moer,
        emission_multiplier_fn,
        charge_per_interval: list = [],
        constraints: dict = {},
    ):
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
        print("== Fixed contiguous fit! ==")
        # print("Charge per interval constraints:", charge_per_interval)
        total_interval = len(charge_per_interval)
        # This is a matrix with size = number of time states x number of intervals charged so far
        max_util = np.full((total_time+1,total_interval+1), np.nan)
        max_util[0,0] = 0.0
        path_history = np.full(
            (total_time, total_interval + 1), False, dtype=bool
        )
        cum_charge = [0]
        for c in charge_per_interval: 
            cum_charge.append(cum_charge[-1]+c)

        charge_array_cache = [emission_multiplier_fn(x,x+1) for x in range(0,total_charge+1)]
        print("Cumulative charge", cum_charge)
        for t in range(1, total_time+1):
            if t in constraints:
                min_charge, max_charge = constraints[t]
                min_charge = 0 if min_charge is None else max(0, min_charge)
                max_charge = total_charge if max_charge is None else min(max_charge, total_charge)
                constraints[t] = (min_charge, max_charge)
            else:
                min_charge, max_charge = 0, total_charge
            for k in range(0, total_interval + 1):
                # print(t,k)
                ## not charging
                init_val = True
                if not np.isnan(max_util[t-1, k]):
                    max_util[t, k] = max_util[t-1, k]
                    init_val = False
                ## charging
                if (k>0) and (charge_per_interval[k-1]<=t): 
                    dc = charge_per_interval[k-1]
                    if not np.isnan(max_util[t-dc,k-1]) and OptCharger.__check_constraint(t-dc,cum_charge[k-1],dc,constraints): 
                        marginal_cost = moer.get_emission_interval(t-dc,t,charge_array_cache[cum_charge[k-1]:cum_charge[k]])
                        new_util = max_util[t-dc,k-1] - marginal_cost
                        if init_val or (new_util > max_util[t,k]): 
                            max_util[t,k] = new_util
                            path_history[t-1,k] = True
                        init_val = False
                            
        if np.isnan(max_util[total_time,total_interval]): 
            ## TODO: In this case we should still return the best possible plan
            ## which would probably to just charge for the entire window
            raise Exception("Solution not found!")
        curr_state, t_curr = total_interval, total_time
        # This gives the schedule in reverse
        schedule = []
        interval_ids_reversed = []
        while t_curr > 0: 
            delta_interval = path_history[t_curr-1, curr_state]
            if not delta_interval: 
                ## did not charge 
                schedule.append(0)
                interval_ids_reversed.append(-1)
                t_curr -= 1  
            else: 
                ## charge
                dc = charge_per_interval[curr_state-1]
                t_curr -= dc
                curr_state -= 1
                if dc>0: 
                    schedule.extend([1]*dc)    
                    interval_ids_reversed.extend([curr_state]*dc)     
        optimal_path = np.array(schedule)[::-1]
        self.__optimal_charging_schedule = list(optimal_path)
        self.__interval_ids = list(interval_ids_reversed[::-1])
        self.__collect_results(moer)

    def __variable_contiguous_fit(
        self,
        total_charge: int,
        total_time: int,
        moer: Moer,
        emission_multiplier_fn,
        charge_per_interval: list = [],
        use_all_intervals: bool = True,
        constraints: dict = {},
    ):
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
        print("== Variable contiguous fit! ==")
        # print(" erval constraints:", charge_per_interval)
        total_interval = len(charge_per_interval) 
        # This is a matrix with size = number of time states x number of charge states x number of intervals charged so far
        max_util = np.full((total_time+1,total_charge+1,total_interval+1), np.nan)
        max_util[0,0,0] = 0.0
        path_history = np.full(
            (total_time, total_charge + 1, total_interval + 1, 2), 0, dtype=int
        )

        charge_array_cache = [emission_multiplier_fn(x,x+1) for x in range(0,total_charge+1)]

        for t in range(1, total_time+1):
            if t in constraints:
                min_charge, max_charge = constraints[t]
                min_charge = 0 if min_charge is None else max(0, min_charge)
                max_charge = total_charge if max_charge is None else min(max_charge, total_charge)
                constraints[t] = (min_charge, max_charge)
            else:
                min_charge, max_charge = 0, total_charge
            for k in range(0, total_interval + 1):
                for c in range(min_charge, max_charge + 1):
                    ## not charging
                    init_val = True
                    if not np.isnan(max_util[t-1, c, k]):
                        max_util[t, c, k] = max_util[t-1, c, k]
                        path_history[t-1, c, k, :] = [0,0]
                        init_val = False
                    ## charging
                    if k > 0: 
                        for dc in range(charge_per_interval[k-1][0],min(charge_per_interval[k-1][1],t,c)+1):
                            if not np.isnan(max_util[t-dc,c-dc,k-1]) and OptCharger.__check_constraint(t-dc,c-dc,dc,constraints):
                                marginal_cost = moer.get_emission_interval(t-dc,t,charge_array_cache[c-dc:c])
                                new_util = max_util[t-dc,c-dc,k-1] - marginal_cost
                                if init_val or (new_util > max_util[t,c,k]): 
                                    max_util[t,c,k] = new_util
                                    path_history[t-1,c,k,:] = [dc,1]
                                init_val = False
        optimal_interval, optimal_util = total_interval, max_util[total_time,total_charge,total_interval]
        if not use_all_intervals: 
            for k in range(0,total_interval):
                if np.isnan(max_util[total_time,total_charge,optimal_interval]) or (not np.isnan(max_util[total_time,total_charge,k]) and max_util[total_time,total_charge,k] > max_util[total_time,total_charge,optimal_interval]): 
                    optimal_interval = k
        if np.isnan(max_util[total_time,total_charge,optimal_interval]): 
            ## TODO: In this case we should still return the best possible plan
            ## which would probably to just charge for the entire window
            raise Exception("Solution not found!")
        curr_state, t_curr = [total_charge,optimal_interval], total_time
        # This gives the schedule in reverse
        schedule = []
        interval_ids_reversed = []
        while t_curr > 0: 
            dc,delta_interval = path_history[t_curr-1, curr_state[0], curr_state[1], :]
            if delta_interval==0: 
                ## did not charge 
                schedule.append(0)
                interval_ids_reversed.append(-1)
                t_curr -= 1  
            else: 
                ## charge
                t_curr -= dc
                curr_state = [curr_state[0]-dc, curr_state[1]-delta_interval]
                if dc>0: 
                    schedule.extend([1]*dc)
                    interval_ids_reversed.extend([curr_state[1]]*dc)
        optimal_path = np.array(schedule)[::-1]
        self.__optimal_charging_schedule = list(optimal_path)
        self.__interval_ids = list(interval_ids_reversed[::-1])
        self.__collect_results(moer)

    def fit(
        self,
        total_charge: int,
        total_time: int,
        moer: Moer,
        charge_per_interval=None,
        use_all_intervals: bool = True,
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
        total_charge : int
            The total amount of charge needed.
        total_time : int
            The total time available for charging.
        moer : Moer
            An object representing Marginal Operating Emissions Rate.
        charge_per_interval : list of int or (int,int), optional
            The minimium and maximum (inclusive) charging amount per interval. If int instead of tuple, interpret as both min and max. 
        use_all_intervals : bool
            If true, use all intervals provided by charge_per_interval; if false, can use the first few intervals and skip the rest. This can only be false if charge_per_interval is provided as a range. 
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
        assert len(moer) >= total_time
        assert optimization_method in ['baseline','simple','sophisticated','auto']
        if emission_multiplier_fn is None:
            print(
                "Warning: OptCharger did not get an emission_multiplier_fn. Assuming that device uses constant 1kW of power"
            )
            emission_multiplier_fn = lambda sc, ec: 1.0
            constant_emission_multiplier = True
        else:
            constant_emission_multiplier = np.std([emission_multiplier_fn(sc,sc+1) for sc in list(range(total_charge))]) < EMISSION_FN_TOL
        # Store emission_multiplier_fn for evaluation
        self.emission_multiplier_fn = emission_multiplier_fn
        if total_charge > total_time:
            # TODO: might want to just print out charging all the time instead of failing
            raise Exception(
                f"Impossible to charge {total_charge} within {total_time} intervals."
            )
        if optimization_method == "baseline":
            self.__greedy_fit(total_charge, total_time, moer)
        elif (
            not constraints
            and not charge_per_interval 
            and constant_emission_multiplier
            and optimization_method=='auto'
         ) or (optimization_method=='simple'):
            if not constant_emission_multiplier:
                print("Warning: Emissions function is non-constant. Using the simple algorithm is suboptimal.")
            self.__simple_fit(total_charge, total_time, moer)
        elif not charge_per_interval:
            self.__diagonal_fit(
                total_charge,
                total_time,
                moer,
                OptCharger.__sanitize_emission_multiplier(
                    emission_multiplier_fn, total_charge
                ), 
                constraints
            )
        else:
            single_cpi, tuple_cpi, use_fixed_alg = [], [], True
            def convert_input(c):
                ## Converts the interval format
                if isinstance(c,int): 
                    return c,(c,c),True
                if c[0]==c[1]: 
                    return c[0],c,True
                return None,c,False
            for c in charge_per_interval: 
                if use_fixed_alg:
                    sc,tc,use_fixed_alg = convert_input(c)
                    single_cpi.append(sc)
                    tuple_cpi.append(tc)
                else: 
                    tuple_cpi.append(convert_input(c)[1])
            if use_fixed_alg: 
                assert use_all_intervals, "Must use all intervals when interval lengths are fixed!"
                self.__contiguous_fit(
                    total_charge,
                    total_time,
                    moer,
                    OptCharger.__sanitize_emission_multiplier(
                        emission_multiplier_fn, total_charge
                    ), 
                    single_cpi, 
                    constraints
                )
            else: 
                self.__variable_contiguous_fit(
                    total_charge,
                    total_time,
                    moer,
                    OptCharger.__sanitize_emission_multiplier(
                        emission_multiplier_fn, total_charge
                    ), 
                    tuple_cpi, 
                    use_all_intervals,
                    constraints
                )

    def get_energy_usage_over_time(self) -> list:
        """
        Returns list of the energy due to charging at each interval in MWh.
        """
        return self.__optimal_charging_energy_over_time

    def get_charging_emissions_over_time(self) -> list:
        """
        Returns list of the emissions due to charging at each interval in lbs.
        """
        return self.__optimal_charging_emissions_over_time

    def get_total_emission(self) -> float:
        """
        Returns the summed emissions due to charging in lbs.
        """
        return self.__optimal_charging_emission

    def get_schedule(self) -> list:
        """
        Returns list of the optimal charging schedule of units to charge for each interval.
        """
        return self.__optimal_charging_schedule

    def get_interval_ids(self) -> list:
        """
        Returns list of the interval ids for each interval. Has a value of -1 for non-charging intervals.
        Intervals are labeled starting from 0 to n-1 when there are n intervals

        Only defined when charge_per_interval variable is given to some fit function
        """
        return self.__interval_ids

    def summary(self):
        print("-- Model Summary --")
        print("Expected charging emissions: %.2f lbs" % self.__optimal_charging_emission)
        print("Optimal charging schedule:", self.__optimal_charging_schedule)
        print("=" * 15)
