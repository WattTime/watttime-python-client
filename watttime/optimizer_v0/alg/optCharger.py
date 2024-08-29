# optCharger.py
import numpy as np
from .moer import Moer

TOL = 1e-4
class OptCharger: 
    def __init__(
        self, 
        fixedChargeRate:int = None, 
        minChargeRate:int = None, 
        maxChargeRate:int = None,
        emissionOverhead:bool = True, 
        startEmissionOverhead:float = 0.,
        keepEmissionOverhead:float = 0.,
        stopEmissionOverhead:float = 0.,
    ): 
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
            self.emissionOverhead = (startEmissionOverhead > TOL) or (keepEmissionOverhead > TOL) or (stopEmissionOverhead > TOL) 
        else: 
            self.emissionOverhead = False
            self.startEmissionOverhead = 0.0
            self.keepEmissionOverhead = 0.0
            self.stopEmissionOverhead = 0.0
        self.__optimalChargingEmission = None
        self.__optimalTotalEmission = None
        self.__optimalChargingSchedule = None
    
    def __collect_results(self, moer:Moer, emission_multipliers=1.): 
        self.__optimalChargingEmission = moer.get_total_emission(np.array(self.__optimalChargingSchedule) * emission_multipliers)
        y = np.hstack((0,self.__optimalOnOffSchedule,0))
        yDiff = y[1:] - y[:-1]
        self.__optimalTotalEmission = (
            self.__optimalChargingEmission + 
            np.sum(y) * self.keepEmissionOverhead +
            np.sum(yDiff==1) * self.startEmissionOverhead + 
            np.sum(yDiff==-1) * self.stopEmissionOverhead
        )
    @staticmethod
    def __sanitize_emission_multiplier(emission_multiplier_fn, totalCharge):
        return lambda sc,ec: emission_multiplier_fn(sc,min(ec,totalCharge)) if (sc < totalCharge) else 1.0

    def __greedy_fit(self, totalCharge:int, totalTime:int, moer:Moer): 
        chargeToDo = totalCharge
        cs, t = [], 0
        while (chargeToDo > 0) and (t < totalTime): 
            c = max(min(self.maxChargeRate, chargeToDo), self.minChargeRate)
            chargeToDo -= c
            cs.append(c)
            t += 1
        self.__optimalChargingSchedule = cs + [0]*(totalTime - t)
        self.__optimalOnOffSchedule = [1]*t + [0]*(totalTime - t)
        #print(self.__optimalChargingSchedule, totalTime)
        self.__collect_results(moer)

    def __simple_fit(self, totalCharge:int, totalTime:int, moer:Moer):  
        print("Simple fit!")
        '''
        Assuming  
        - no extra costs 
        - no risk
        Then we can 
        - sort intervals by MOER
        - keep charging until we fill up 
        '''
        sorted_times = [x for _, x in sorted(zip(moer.get_emission_interval(0,totalTime),range(totalTime)))]
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
  
    def __diagonal_fit(self, totalCharge:int, totalTime:int, moer:Moer, emission_multiplier_fn, ra:float = 0., constraints:dict = {}): 
        # maxUtil[{0..totalCharge},{0,1}] = emission (with risk penalty)
        # path[t,{0..totalCharge},{0,1},:] = [charge,{0,1]}]
        '''
        This is the DP algorithm 
        '''
        # This is a matrix with size = number of charge states x number of actions {not charging = 0, charging = 1}
        maxUtil = np.full((totalCharge+1,2), np.nan)
        maxUtil[0,0] = 0.
        pathHistory = np.full((totalTime,totalCharge+1,2,2), 0, dtype=int)
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
                ## update (c,0)
                # print("-- charge", c, "| charging off --")
                initVal = True
                if not np.isnan(maxUtil[c,0]): 
                    newMaxUtil[c,0] = maxUtil[c,0]
                    pathHistory[t,c,0,:] = [c,0]
                    initVal = False
                if not np.isnan(maxUtil[c,1]): 
                    newUtil = maxUtil[c,1] - self.stopEmissionOverhead
                    if initVal or (newUtil > newMaxUtil[c,0]): 
                        newMaxUtil[c,0] = newUtil
                        pathHistory[t,c,0,:] = [c,1]

                ## update (c,1)
                # print("-- charge", c, "| charging on --")
                initVal = True
                for ct in range(self.minChargeRate,min(c,self.maxChargeRate)+1):
                    if not np.isnan(maxUtil[c-ct,0]): 
                        newUtil = maxUtil[c-ct,0] - moer.get_marginal_util(ct,t,ra=ra) * emission_multiplier_fn(c-ct,c) - self.startEmissionOverhead - self.keepEmissionOverhead 
                        if initVal or (newUtil > newMaxUtil[c,1]): 
                            newMaxUtil[c,1] = newUtil
                            pathHistory[t,c,1,:] = [c-ct,0]
                        initVal = False
                    if not np.isnan(maxUtil[c-ct,1]): 
                        newUtil = maxUtil[c-ct,1] - moer.get_marginal_util(ct,t,ra=ra) * emission_multiplier_fn(c-ct,c) - self.keepEmissionOverhead
                        if initVal or (newUtil > newMaxUtil[c,1]): 
                            newMaxUtil[c,1] = newUtil
                            pathHistory[t,c,1,:] = [c-ct,1]
                        initVal = False
            maxUtil = newMaxUtil
            # print(maxUtil)

        solution_found = False
        if not np.isnan(maxUtil[totalCharge,0]): 
            max_util = maxUtil[totalCharge,0]
            m_final = 0
            solution_found = True
        if not np.isnan(maxUtil[totalCharge,1]): 
            newUtil = maxUtil[totalCharge,1] - self.stopEmissionOverhead
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
        optimalPath = np.array(schedule)[::-1,:]
        self.__optimalChargingSchedule = list(np.diff(optimalPath[:,0]))
        emission_multipliers = [emission_multiplier_fn(x,y) for x,y in zip(
            optimalPath[:-1,0], optimalPath[1:,0] 
        )]
        self.__optimalOnOffSchedule = optimalPath[:,1]
        self.__collect_results(moer, emission_multipliers)

    def __contiguous_fit(self, totalCharge:int, totalTime:int, moer:Moer, emission_multiplier_fn, totalIntervals:int = 1, ra:float = 0., constraints:dict = {}): 
        # maxUtil[{0..totalCharge},{0,1},{0..chargeIntervalCount}] = emission (with risk penalty)
        # path[t,{0..totalCharge},{0,1},{0..chargeIntervalCount},:] = [charge,{0,1},{0..chargeIntervalCount}]
        '''
        This is the DP algorithm with further constraint on # intervals  
        '''
        print("== Contiguous fit! ==")
        # This is a matrix with size = number of charge states x number of actions {not charging = 0, charging = 1}
        maxUtil = np.full((totalCharge+1,2,totalIntervals+1), np.nan)
        maxUtil[0,0,0] = 0.
        pathHistory = np.full((totalTime,totalCharge+1,2,totalIntervals+1,3), 0, dtype=int)
        for t in range(totalTime): 
            if t in constraints: 
                minCharge, maxCharge = constraints[t]
                minCharge = 0 if minCharge is None else max(0, minCharge)
                maxCharge = totalCharge if maxCharge is None else min(maxCharge, totalCharge)
            else: 
                minCharge, maxCharge = 0, totalCharge
            newMaxUtil = np.full(maxUtil.shape, np.nan)
            for k in range(0,totalIntervals+1): 
                for c in range(minCharge,maxCharge+1): 
                    ## update (c,0,k)
                    initVal = True
                    if not np.isnan(maxUtil[c,0,k]):  
                        newMaxUtil[c,0,k] = maxUtil[c,0,k]
                        pathHistory[t,c,0,k,:] = [c,0,k]
                        initVal = False
                    if not np.isnan(maxUtil[c,1,k]): 
                        newUtil = maxUtil[c,1,k] - self.stopEmissionOverhead
                        if initVal or (newUtil > newMaxUtil[c,0,k]): 
                            newMaxUtil[c,0,k] = newUtil
                            pathHistory[t,c,0,k,:] = [c,1,k]
                    ## update (c,1,k)
                    if k == 0: 
                        # If charging is on, we must have k > 0
                        continue
                    initVal = True
                    for ct in range(self.minChargeRate,min(c,self.maxChargeRate)+1):
                        if not np.isnan(maxUtil[c-ct,0,k-1]): 
                            newUtil = maxUtil[c-ct,0,k-1] - moer.get_marginal_util(ct,t,ra=ra) * emission_multiplier_fn(c-ct,c) - self.startEmissionOverhead - self.keepEmissionOverhead 
                            if initVal or (newUtil > newMaxUtil[c,1,k]): 
                                newMaxUtil[c,1,k] = newUtil
                                pathHistory[t,c,1,k,:] = [c-ct,0,k-1]
                            initVal = False
                        if not np.isnan(maxUtil[c-ct,1,k]): 
                            newUtil = maxUtil[c-ct,1,k] - moer.get_marginal_util(ct,t,ra=ra) * emission_multiplier_fn(c-ct,c) - self.keepEmissionOverhead
                            if initVal or (newUtil > newMaxUtil[c,1,k]): 
                                newMaxUtil[c,1,k] = newUtil
                                pathHistory[t,c,1,k,:] = [c-ct,1,k]
                            initVal = False
            maxUtil = newMaxUtil
            # print(maxUtil)

        solution_found = False
        for k in range(0,totalIntervals+1): 
            if not np.isnan(maxUtil[totalCharge,0,k]): 
                newUtil = maxUtil[totalCharge,0,k]
                if not solution_found or (newUtil > max_util): 
                    max_util = newUtil
                    m_final = (0,k)
                solution_found = True
            if not np.isnan(maxUtil[totalCharge,1,k]): 
                newUtil = maxUtil[totalCharge,1,k] - self.stopEmissionOverhead
                if not solution_found or (newUtil > max_util): 
                    max_util = newUtil
                    m_final = (1,k)
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
            curr_state = pathHistory[t_curr, curr_state[0], curr_state[1], curr_state[2], :]
            schedule.append(curr_state)
            t_curr -= 1
        optimalPath = np.array(schedule)[::-1,:]
        self.__optimalChargingSchedule = list(np.diff(optimalPath[:,0]))
        print(optimalPath[:,0])
        emission_multipliers = [emission_multiplier_fn(x,y) for x,y in zip(
            optimalPath[:-1,0], optimalPath[1:,0] 
        )]
        self.__optimalOnOffSchedule = optimalPath[:,1]
        self.__collect_results(moer, emission_multipliers)
    
    def fit(self, totalCharge:int, totalTime:int, moer:Moer, totalIntervals:int = 0, constraints:dict = {}, ra:float = 0., asap:bool = False, emission_multiplier_fn = None): 
        assert len(moer) >= totalTime
        if (totalCharge > totalTime * self.maxChargeRate): 
            raise Exception(f"Impossible to charge {totalCharge} within {totalTime} intervals.")
        if asap: 
            self.__greedy_fit(totalCharge, totalTime, moer)
        elif not self.emissionOverhead and ra<TOL and not constraints and emission_multiplier_fn is None and totalIntervals <= 0:
            self.__simple_fit(totalCharge, totalTime, moer)
        elif moer.is_diagonal(): 
            if emission_multiplier_fn is None: 
                emission_multiplier_fn =  lambda sc,ec: 1.
            if totalIntervals <= 0: 
                self.__diagonal_fit(totalCharge, totalTime, moer, OptCharger.__sanitize_emission_multiplier(emission_multiplier_fn, totalCharge), ra, constraints)
            else: 
                self.__contiguous_fit(totalCharge, totalTime, moer, OptCharger.__sanitize_emission_multiplier(emission_multiplier_fn, totalCharge), totalIntervals, ra, constraints)
        else: 
            raise Exception("Not implemented!")
    
    def get_charging_emission(self) -> float: 
        return self.__optimalChargingEmission
    
    def get_total_emission(self) -> float: 
        return self.__optimalTotalEmission
    
    def get_schedule(self) -> list: 
        return self.__optimalChargingSchedule

    def summary(self): 
        print("-- Model Summary --")
        print("Expected charging emissions %.2f" % self.__optimalChargingEmission)
        print("Expected total emissions %.2f" % self.__optimalTotalEmission)
        print("Optimal charging schedule:", self.__optimalChargingSchedule)
        print('='*15)