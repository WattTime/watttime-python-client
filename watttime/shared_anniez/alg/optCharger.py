# optCharger.py
import numpy as np
from .moer import Moer

def initNP(shape,val=np.nan): 
    a = np.empty(shape)
    a.fill(val)
    return a
TOL = 1e-4
class OptCharger: 
    def __init__(
        self, 
        # contiguousWindow = False, 
        fixedChargeRate:int = None, 
        minChargeRate:int = None, 
        maxChargeRate:int = None,
        emissionOverhead:bool = True, 
        startEmissionOverhead:float = 0.,
        keepEmissionOverhead:float = 0.,
        stopEmissionOverhead:float = 0.,
    ): 
        # self.contiguousWindow = contiguousWindow
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
        self.__optimalChargingEmission = None
        self.__optimalTotalEmission = None
        self.__optimalChargingSchedule = None
    
    def __collect_results(self, moer:Moer): 
        self.__optimalChargingEmission = moer.get_total_emission(self.__optimalChargingSchedule)
        y = np.hstack((0,self.__optimalOnOffSchedule,0))
        yDiff = y[1:] - y[:-1]
        self.__optimalTotalEmission = (
            self.__optimalChargingEmission + 
            np.sum(y) * self.keepEmissionOverhead +
            np.sum(yDiff==1) * self.startEmissionOverhead + 
            np.sum(yDiff==-1) * self.stopEmissionOverhead
        )

    def __greedy_fit(self, totalCharge:int, totalTime:int, moer:Moer): 
        print("Greedy fit!")  
        tc = totalCharge
        cs, t = [], 0
        while (tc > 0) and (t < totalTime): 
            c = max(min(self.maxChargeRate, tc), self.minChargeRate)
            tc -= c
            cs.append(c)
            t += 1
        self.__optimalChargingSchedule = cs + [0]*(totalTime - t)
        self.__optimalOnOffSchedule = [1]*t + [0]*(totalTime - t)
        print(self.__optimalChargingSchedule, totalTime)
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
        print("Simplified fit!")
        sorted_times = [x for _, x in sorted(zip(moer.get_emission_interval(0,totalTime),range(totalTime)))]
        tc = totalCharge
        cs, schedule, t = [0] * totalTime, [0] * totalTime, 0
        while (tc > 0) and (t < totalTime): 
            c = max(min(self.maxChargeRate, tc), self.minChargeRate)
            tc -= c
            cs[sorted_times[t]] = c
            schedule[sorted_times[t]] = 1
            t += 1
        self.__optimalChargingSchedule = cs
        self.__optimalOnOffSchedule = schedule
        self.__collect_results(moer)
  
    def __diagonal_fit(self, totalCharge:int, totalTime:int, moer:Moer, ra:float = 0., constraints:dict = {}): 
        # maxUtil[{0..totalCharge},{0,1}] = emission (with risk penalty)
        # path[t,{0..totalCharge},{0,1},:] = [charge,{0,1]}]
        '''
        This is the most complex version of the algorithm 
        '''
        print("Full fit!")
        maxUtil = initNP((totalCharge+1,2))
        maxUtil[0,0] = 0.
        pathHistory = initNP((totalTime,totalCharge+1,2,2),0).astype(int)
        for t in range(totalTime): 
            if t in constraints: 
                minCharge, maxCharge = constraints[t]
                minCharge = 0 if minCharge is None else max(0, minCharge)
                maxCharge = totalCharge if maxCharge is None else min(maxCharge, totalCharge)
            else: 
                minCharge, maxCharge = 0, totalCharge
            # print("=== Time step", t, "===")
            newMaxUtil = initNP(maxUtil.shape)
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
                    if (not np.isnan(maxUtil[c-ct,0])): 
                        newUtil = maxUtil[c-ct,0] - moer.get_marginal_util(ct,t) - self.startEmissionOverhead - self.keepEmissionOverhead 
                        if initVal or (newUtil > newMaxUtil[c,1]): 
                            newMaxUtil[c,1] = newUtil
                            pathHistory[t,c,1,:] = [c-ct,0]
                        initVal = False
                    if (not np.isnan(maxUtil[c-ct,1])): 
                        newUtil = maxUtil[c-ct,1] - moer.get_marginal_util(ct,t) - self.keepEmissionOverhead
                        if initVal or (newUtil > newMaxUtil[c,1]): 
                            newMaxUtil[c,1] = newUtil
                            pathHistory[t,c,1,:] = [c-ct,1]
                        initVal = False
            maxUtil = newMaxUtil
            # print(maxUtil)

        initVal = True
        if not np.isnan(maxUtil[c,0]): 
            max_util = maxUtil[c,0]
            m_final = 0
            initVal = False
        if not np.isnan(maxUtil[c,1]): 
            newUtil = maxUtil[c,1] - self.stopEmissionOverhead
            if initVal or (newUtil > max_util): 
                max_util = newUtil
                m_final = 1
            initVal = False
        if initVal: 
            raise Exception("Solution not found!")
        curr_state, t_curr = [c, m_final], totalTime-1
        schedule = []        
        schedule.append(curr_state)
        while (t_curr >= 0): 
            curr_state = pathHistory[t_curr, curr_state[0], curr_state[1], :]
            schedule.append(curr_state)
            t_curr -= 1
        optimalPath = np.array(schedule)[::-1,:]
        self.__optimalChargingSchedule = list(np.diff(optimalPath[:,0]))
        self.__optimalOnOffSchedule = optimalPath[:,1]
        self.__collect_results(moer)
    
    def fit(self, totalCharge:int, totalTime:int, moer:Moer, constraints:dict = {}, ra:float = 0., asap:bool = False): 
        assert(len(moer) >= totalTime)
        if (totalCharge > totalTime * self.maxChargeRate): 
            raise Exception(f"Impossible to charge {totalCharge} within {totalTime} intervals.")
        if asap: 
            self.__greedy_fit(totalCharge, totalTime, moer)
        elif not self.emissionOverhead and ra<TOL and not constraints:
            self.__simple_fit(totalCharge, totalTime, moer)
        elif moer.is_diagonal(): 
            self.__diagonal_fit(totalCharge, totalTime, moer, ra, constraints)
        else: 
            raise Exception("Not implemented!")
    
    def get_charging_cost(self) -> float: 
        return self.__optimalChargingEmission
    
    def get_total_cost(self) -> float: 
        return self.__optimalTotalEmission
    
    def get_schedule(self) -> list: 
        return self.__optimalChargingSchedule

    def summary(self): 
        print("=== Model Summary ===")
        print("Expected charging emissions %.2f" % self.__optimalChargingEmission)
        print("Expected total emissions %.2f" % self.__optimalTotalEmission)
        print("Optimal charging schedule:", self.__optimalChargingSchedule)


