# optCharger.py
import numpy as np
from .moer_new import Moer

TOL = 1e-4
class OptCharger: 
    def __init__(self):
        self.__optimalChargingEmission = None
        self.__optimalChargingSchedule = None
    
    def __collect_results(self, moer:Moer):
        emission_multipliers = []
        current_charge_time_units = 0
        for i in range(len(self.__optimalChargingSchedule)):
            if self.__optimalChargingSchedule[i] == 0:
                emission_multipliers.append(0.0)
            else:
                old_charge_time_units = current_charge_time_units
                current_charge_time_units += self.__optimalChargingSchedule[i]
                power_rate = self.emission_multiplier_fn(old_charge_time_units, current_charge_time_units)
                emission_multipliers.append(power_rate)

        self.__optimalChargingEnergyOverTime = np.array(self.__optimalChargingSchedule) * np.array(emission_multipliers)
        self.__optimalChargingEmissionsOverTime = moer.get_emissions(np.array(self.__optimalChargingSchedule) * np.array(emission_multipliers))
        self.__optimalChargingEmission = moer.get_total_emission(np.array(self.__optimalChargingSchedule) * np.array(emission_multipliers))
        
    @staticmethod
    def __sanitize_emission_multiplier(emission_multiplier_fn, totalCharge):
        return lambda sc,ec: emission_multiplier_fn(sc,min(ec,totalCharge)) if (sc < totalCharge) else 1.0
    @staticmethod
    def __avg_to_interval(emission_multiplier_fn, sc, ec): 
        # TODO: There is probably a better way to implement this; any thoughts @yhlim?
        return [emission_multiplier_fn(x,x+1) for x in range(sc,ec)]
    @staticmethod
    def __check_constraint(t_start, c_start, dc, constraints): 
        # assuming constraints[t] is the bound on total charge after t+1 intervals
        for t in range(t_start+1, t_start+dc): 
            if (t-1 in constraints) and ((c_start+t-t_start < constraints[t-1][0]) or (c_start+t-t_start > constraints[t-1][0])): 
                return False
        return True

    def __greedy_fit(self, totalCharge:int, totalTime:int, moer:Moer): 
        chargeToDo = totalCharge
        cs, t = [], 0
        while (chargeToDo > 0) and (t < totalTime): 
            chargeToDo -= 1
            cs.append(1)
            t += 1
        self.__optimalChargingSchedule = cs + [0]*(totalTime - t)
        #print(self.__optimalChargingSchedule, totalTime)
        self.__collect_results(moer)

    def __simple_fit(self, totalCharge:int, totalTime:int, moer:Moer):  
        '''
        Assuming  
        - no extra costs 
        - no risk
        Then we can 
        - sort intervals by MOER
        - keep charging until we fill up 
        '''
        print("== Simple fit! ==")
        sorted_times = [x for _, x in sorted(zip(moer.get_emission_interval(0,totalTime,1),range(totalTime)))]
        chargeToDo = totalCharge
        cs, schedule, t = [0] * totalTime, [0] * totalTime, 0
        while (chargeToDo > 0) and (t < totalTime): 
            chargeToDo -= 1
            cs[sorted_times[t]] = 1
            schedule[sorted_times[t]] = 1
            t += 1
        self.__optimalChargingSchedule = cs
        self.__collect_results(moer)
  
    def __dp_fit(self, totalCharge:int, totalTime:int, moer:Moer, emission_multiplier_fn, constraints:dict = {}): 
        # maxUtil[{0..totalCharge}] = emission (with risk penalty)
        # pathHistory[{0,..,T-1},{0..totalCharge}] = charge at last time interval (allows for LinkedList storage of schedule)
        '''
        This is the DP algorithm 
        '''
        print("== Sophisticated fit! ==")
        # This is a matrix with size = number of charge states
        maxUtil = np.full((totalCharge+1,),np.nan)
        maxUtil[0] = 0.
        pathHistory = np.full((totalTime,totalCharge+1),-1,dtype=int)
        for t in range(totalTime): 
            if t in constraints: 
                minCharge, maxCharge = constraints[t]
                minCharge = 0 if minCharge is None else max(0, minCharge)
                maxCharge = totalCharge if maxCharge is None else min(maxCharge, totalCharge)
            else: 
                minCharge, maxCharge = 0, totalCharge
            # print("=== Time step", t, "===")
            newMaxUtil = np.full(maxUtil.shape, np.nan)
            for c in range(minCharge,maxCharge+1): 
                ## Do not charge
                # print("-- charge", c, "| charging off --")
                initVal = True
                if not np.isnan(maxUtil[c]): 
                    newMaxUtil[c] = maxUtil[c]
                    pathHistory[t,c] = c
                    initVal = False
                ## Charge
                # print("-- charge", c, "| charging on --")
                if not np.isnan(maxUtil[c-1]):
                    # moer.get_marginal_util gives lbs/MWh. emission function needs to be how many MWh the interval consumes
                    # which would be power_in_kW * 0.001 * 5/60
                    newUtil = maxUtil[c-1]-moer.get_emission_at(t, emission_multiplier_fn(c-1,c))
                    if initVal or (newUtil > newMaxUtil[c]): 
                        newMaxUtil[c] = newUtil
                        pathHistory[t,c] = c-1
                    initVal = False
            maxUtil = newMaxUtil
            # print(maxUtil)

        solution_found = False
        if not np.isnan(maxUtil[totalCharge]): 
            max_util = maxUtil[totalCharge]
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

    def __variable_contiguous_fit(self, totalCharge:int, totalTime:int, moer:Moer, emission_multiplier_fn, charge_per_interval:list = [], constraints:dict = {}): 
        # maxUtil[t,{0..totalCharge},{0..totalInterval}]=emission (with risk penalty)
        # path[t,{0..totalCharge},{0..totalInterval},:]=[charge,interval]
        '''
        This is the DP algorithm with further constraint on # intervals  
        '''
        print("== Variable contiguous fit! ==")
        totalInterval = len(charge_per_interval)
        maxUtil = np.full((totalTime+1,totalCharge+1,totalInterval+1), np.nan)
        maxUtil[0,0,0] = 0.
        pathHistory = np.full((totalTime,totalCharge+1,totalInterval+1,2), 0, dtype=int)
        for t in range(1,totalTime+1): 
            if t-1 in constraints: 
                minCharge, maxCharge = constraints[t-1]
                minCharge = 0 if minCharge is None else max(0, minCharge)
                maxCharge = totalCharge if maxCharge is None else min(maxCharge, totalCharge)
                constraints[t-1] = (minCharge, maxCharge)
            else: 
                minCharge, maxCharge = 0, totalCharge
            for k in range(0,totalInterval+1): 
                for c in range(minCharge,maxCharge+1): 
                    # print(t,c,k)
                    ## not charging
                    initVal = True
                    if not np.isnan(maxUtil[t-1,c,k]):  
                        # print("Exist non-charging solution")
                        maxUtil[t,c,k] = maxUtil[t-1,c,k]
                        pathHistory[t-1,c,k,:] = [0,0]
                        initVal = False
                    ## charging 
                    if k > 0: 
                        for dc in range(charge_per_interval[k-1][0],min(charge_per_interval[k-1][1],t,c)+1):
                            if not np.isnan(maxUtil[t-dc,c-dc,k-1]) and OptCharger.__check_constraint(t-dc,c-dc,dc,constraints): 
                                # print(f"Exist solution charging {dc}")
                                marginalcost = moer.get_emission_interval(t-dc,t,OptCharger.__avg_to_interval(emission_multiplier_fn,c-dc,c))
                                newUtil = maxUtil[t-dc,c-dc,k-1] - marginalcost
                                if initVal or (newUtil > maxUtil[t,c,k]): 
                                    # print(f"updating... marginalcost={marginalcost}")
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
            # print(t_curr, schedule, dc, di)
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
    
    def fit(self, totalCharge:int, totalTime:int, moer:Moer, charge_per_interval = None, constraints:dict = {}, asap:bool = False, emission_multiplier_fn = None, optimization_method:str = 'auto'): 
        assert len(moer) >= totalTime
        assert optimization_method in ['greedy','simple','sophisticated','auto']
        EMISSION_FN_TOL = 1e-9 # in kw
        if emission_multiplier_fn is None:
            print("Warning: OptCharger did not get an emission_multiplier_fn. Assuming that device uses constant 1kW of power")
            emission_multiplier_fn = lambda sc,ec:1.0
            ef_is_constant = True
        else: 
            ef_is_constant = np.std([emission_multiplier_fn(sc,sc+1) for sc in list(range(totalCharge))]) < EMISSION_FN_TOL
        # Store emission_multiplier_fn for evaluation 
        self.emission_multiplier_fn = emission_multiplier_fn

        if (totalCharge > totalTime): 
            raise Exception(f"Impossible to charge {totalCharge} within {totalTime} intervals.")
        if asap: 
            self.__greedy_fit(totalCharge, totalTime, moer)
        elif (not constraints and not charge_per_interval and ef_is_constant and optimization_method=='auto') or (optimization_method=='simple'):
            if not ef_is_constant:
                print("Warning: Emissions function is non-constant. Using the simple algorithm is suboptimal.") 
            self.__simple_fit(totalCharge, totalTime, moer)
        elif not charge_per_interval: 
            self.__dp_fit(totalCharge, totalTime, moer, OptCharger.__sanitize_emission_multiplier(emission_multiplier_fn, totalCharge), constraints)
        else: 
            self.__variable_contiguous_fit(totalCharge, totalTime, moer, OptCharger.__sanitize_emission_multiplier(emission_multiplier_fn, totalCharge), charge_per_interval, constraints)
    
    def get_energy_usage_over_time(self) -> list:
        """
        Returns:
            list: The energy due to charging at each interval in MWh
        """
        return self.__optimalChargingEnergyOverTime
    
    def get_charging_emissions_over_time(self) -> list:
        """
        Returns:
            list: The emissions due to charging at each interval in lbs. 
        """
        return self.__optimalChargingEmissionsOverTime
    
    def get_charging_emission(self) -> float: 
        """
        Returns:
            float: The summed emissions due to charging in lbs.
                  This excludes penalty terms due to risk aversion
        """
        return self.__optimalChargingEmission
    
    def get_total_emission(self) -> float: 
        """
        Returns:
            float: The summed emissions due to charging and penalty terms in lbs.
        """
        return self.__optimalChargingEmission
    
    def get_schedule(self) -> list: 
        """
        Returns:
            list: The charging schedule as a list, in minutes to charge for each interval.
        """
        return self.__optimalChargingSchedule

    def summary(self): 
        print("-- Model Summary --")
        print("Expected charging emissions: %.2f lbs" % self.__optimalChargingEmission)
        print("Optimal charging schedule:", self.__optimalChargingSchedule)
        print('='*15)


