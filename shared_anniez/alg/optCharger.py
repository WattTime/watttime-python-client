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
        constraints = []
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
    
    def fit(self, get_schedule = True): 
        # minUtil[{0..totalCharge},{0,1}] = cost (with risk penalty)
        # path[t,{0..totalCharge},{0,1},:] = [charge,{0,1]}]
        minUtil = initNP((self.totalCharge+1,2))
        minUtil[0,0] = 0.
        pathHistory = initNP((self.moer.length(),self.totalCharge+1,2,2),0).astype(int)
        for t in range(1,self.moer.length()+1): 
            # print("=== Time step", t, "===")
            newMinUtil = initNP(minUtil.shape)
            for c in range(0,self.totalCharge+1): 
                ## update (c,0)
                # print("-- charge", c, "| charging off --")
                initVal = True
                if not np.isnan(minUtil[c,0]): 
                    newMinUtil[c,0] = minUtil[c,0]
                    pathHistory[t-1,c,0,:] = [c,0]
                    initVal = False
                if not np.isnan(minUtil[c,1]): 
                    newUtil = minUtil[c,1] + self.stopChargeCost 
                    if initVal or (newUtil < newMinUtil[c,0]): 
                        newMinUtil[c,0] = newUtil
                        pathHistory[t-1,c,0,:] = [c,1]

                ## update (c,1)
                # print("-- charge", c, "| charging on --")
                initVal = True
                for ct in range(self.minChargeRate,min(c,self.maxChargeRate)+1):
                    marg_util = self.moer.getMarginalUtil(ct, t-1)
                    if (not np.isnan(minUtil[c-ct,0])): 
                        newUtil = minUtil[c-ct,0] + self.startChargeCost + self.keepChargeCost + marg_util
                        if initVal or (newUtil < newMinUtil[c,1]): 
                            newMinUtil[c,1] = newUtil
                            pathHistory[t-1,c,1,:] = [c-ct,0]
                        initVal = False
                    if (not np.isnan(minUtil[c-ct,1])): 
                        newUtil = minUtil[c-ct,1] + self.keepChargeCost + marg_util
                        if initVal or (newUtil < newMinUtil[c,1]): 
                            newMinUtil[c,1] = newUtil
                            pathHistory[t-1,c,1,:] = [c-ct,1]
                        initVal = False
            minUtil = newMinUtil
            
        initVal = True
        if not np.isnan(minUtil[c,0]): 
            self.minUtil = minUtil[c,0]
            m_final = 0
            initVal = False
        if not np.isnan(minUtil[c,1]): 
            newUtil = minUtil[c,1] + self.stopChargeCost 
            if initVal or (newUtil < self.minUtil): 
                self.minUtil = newUtil
                m_final = 1
        if get_schedule: 
            curr_state, t_curr = [c, m_final], self.moer.length()
            schedule = []        
            schedule.append(curr_state)
            while (t_curr > 0): 
                curr_state = pathHistory[t_curr-1, curr_state[0], curr_state[1], :]
                schedule.append(curr_state)
                t_curr -= 1
        self.optimalPath = np.array(schedule)[::-1,:]
        self.optimalChargingSchedule = np.diff(self.optimalPath[:,0])
        self.optimalCost = self.moer.getTotalCost(self.optimalChargingSchedule)

    def summary(self): 
        print("Minimum carbon utility %.2f" % self.minUtil)
        print("Expected carbon cost %.2f" % self.optimalCost)
        print("Optimal charging schedule:", self.optimalChargingSchedule)


