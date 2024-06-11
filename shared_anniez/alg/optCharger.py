# optCharger.py
import numpy as np
from .moer import Moer

def initNP(shape,val=np.nan): 
    a = np.empty(shape)
    a.fill(val)
    return a

class OptCharger: 
    def __init__(
        self, 
        totalCharge, 
        moer, 
        # contiguousWindow = False, 
        fixedChargeRate = None, 
        minChargeRate = None, 
        maxChargeRate = None,
        startChargeCost = 0,
        keepChargeCost = 0,
        stopChargeCost = 0,
        constraints = {}
    ): 
        self.totalCharge = totalCharge
        self.moer = moer
        # self.contiguousWindow = contiguousWindow
        if fixedChargeRate is not None: 
            self.minChargeRate = fixedChargeRate
            self.maxChargeRate = fixedChargeRate
        else: 
            self.minChargeRate = minChargeRate
            self.maxChargeRate = maxChargeRate
        self.startChargeCost = startChargeCost
        self.keepChargeCost = keepChargeCost
        self.stopChargeCost = stopChargeCost
        self.constraints = constraints
        self.__optimalCost = None
        self.__optimalChargingSchedule = None
    
    def diagonalFit(self): 
        # maxUtil[{0..totalCharge},{0,1}] = cost (with risk penalty)
        # path[t,{0..totalCharge},{0,1},:] = [charge,{0,1]}]
        maxUtil = initNP((self.totalCharge+1,2))
        maxUtil[0,0] = 0.
        pathHistory = initNP((len(self.moer),self.totalCharge+1,2,2),0).astype(int)
        for t in range(1,len(self.moer)+1): 
            if t in self.constraints: 
                minCharge, maxCharge = self.constraints[t]
                minCharge = 0 if minCharge is None else max(0, minCharge)
                maxCharge = self.totalCharge if maxCharge is None else min(maxCharge, self.totalCharge)
            else: 
                minCharge, maxCharge = 0, self.totalCharge
            # print("=== Time step", t, "===")
            newMaxUtil = initNP(maxUtil.shape)
            for c in range(minCharge,maxCharge+1): 
                ## update (c,0)
                # print("-- charge", c, "| charging off --")
                initVal = True
                if not np.isnan(maxUtil[c,0]): 
                    newMaxUtil[c,0] = maxUtil[c,0]
                    pathHistory[t-1,c,0,:] = [c,0]
                    initVal = False
                if not np.isnan(maxUtil[c,1]): 
                    newUtil = maxUtil[c,1] + self.stopChargeCost 
                    if initVal or (newUtil > newMaxUtil[c,0]): 
                        newMaxUtil[c,0] = newUtil
                        pathHistory[t-1,c,0,:] = [c,1]

                ## update (c,1)
                # print("-- charge", c, "| charging on --")
                initVal = True
                for ct in range(self.minChargeRate,min(c,self.maxChargeRate)+1):
                    marg_util,_ = self.moer.getMarginalUtil(ct, t-1)
                    if (not np.isnan(maxUtil[c-ct,0])): 
                        newUtil = maxUtil[c-ct,0] - self.startChargeCost - self.keepChargeCost - marg_util
                        if initVal or (newUtil > newMaxUtil[c,1]): 
                            newMaxUtil[c,1] = newUtil
                            pathHistory[t-1,c,1,:] = [c-ct,0]
                        initVal = False
                    if (not np.isnan(maxUtil[c-ct,1])): 
                        newUtil = maxUtil[c-ct,1] - self.keepChargeCost - marg_util
                        if initVal or (newUtil > newMaxUtil[c,1]): 
                            newMaxUtil[c,1] = newUtil
                            pathHistory[t-1,c,1,:] = [c-ct,1]
                        initVal = False
            maxUtil = newMaxUtil
            # print(maxUtil)

        initVal = True
        if not np.isnan(maxUtil[c,0]): 
            max_util = maxUtil[c,0]
            m_final = 0
            initVal = False
        if not np.isnan(maxUtil[c,1]): 
            newUtil = maxUtil[c,1] - self.stopChargeCost 
            if initVal or (newUtil > max_util): 
                max_util = newUtil
                m_final = 1
            initVal = False
        if initVal: 
            raise Exception("Solution not found!")
        curr_state, t_curr = [c, m_final], len(self.moer)
        schedule = []        
        schedule.append(curr_state)
        while (t_curr > 0): 
            curr_state = pathHistory[t_curr-1, curr_state[0], curr_state[1], :]
            schedule.append(curr_state)
            t_curr -= 1
        self.__optimalUtil = max_util
        self.__optimalPath = np.array(schedule)[::-1,:]
        self.__optimalChargingSchedule = np.diff(self.__optimalPath[:,0])
        self.__optimalCost = self.moer.getTotalCost(self.__optimalChargingSchedule)
    
    def fit(self): 
        if self.moer.isDiagonal(): 
            self.diagonalFit()
        else: 
            raise Exception("Not implemented!")
            pass
    
    def getCost(self): 
        return self.__optimalCost
    
    def getSchedule(self): 
        return self.__optimalChargingSchedule

    def summary(self): 
        print("Optimal utility %.2f" % -self.__optimalUtil)
        print("Expected emissions %.2f" % self.__optimalCost)
        print("Optimal charging schedule:", self.__optimalChargingSchedule)


