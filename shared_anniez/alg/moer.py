# moer.py

import numpy as np 
class Moer: 
    def __init__(self, mu, isDiagonal=False, Sigma=None, sig2=0., ac1=None, ra=None): 
        self.mu = np.array(mu).flatten()
        self.ra = ra if ra is not None else 0.
        self.__diagonal = isDiagonal
        self.__T = self.mu.shape[0]
        if isinstance(sig2, float): 
            sig2 = np.array([sig2] * self.__T)
        else: 
            sig2 = np.array(sig2).flatten()
        if (not self.__diagonal): 
            if Sigma is not None: 
                assert(Sigma.shape == (self.__T, self.__T))
                self.Sigma = Sigma
            elif ac1 is not None: 
                from scipy.linalg import toeplitz
                x = ac1**np.arange(0,self.__T)
                self.Sigma = toeplitz(x, x) * np.diag(sig2)
        else: 
            self.Sigma = sig2

    def setRA(self, ra):
        self.ra = ra

    def __len__(self):
        return self.__T
    
    def isDiagonal(self):
        return self.__diagonal
    
    def getMarginalCost(self, xi, i): 
        return self.mu[i] * xi

    def getUnitUtil(self, xi, i, ra=None): 
        if ra is None:
            ra = self.ra
        return self.getMarginalCost(xi, i) + ra * self.Sigma[i] * xi**2
    
    def getMarginalUtil(self, xi, i, x_old=None, ra=None):
        if ra is None:
            ra = self.ra
        if self.__diagonal: 
            return self.getUnitUtil(xi, i, ra), x_old
        else: 
            return self.getMarginalCost(xi, i) + ra * (2 * xi * self.Sigma[i,:i]@x_old[:i] + self.Sigma[i,i] * xi**2), np.hstack((x_old, xi))
        
    def getTotalCost(self, x): 
        x = np.array(x).flatten()
        return np.dot(self.mu, x)
    
    def getTotalUtil(self, x, ra=None): 
        if ra is None:
            ra = self.ra
        x = np.array(x).flatten()
        if self.__diagonal: 
            return self.getTotalCost(x) + self.ra * np.multiply(self.Sigma, x**2).sum()
        else: 
            return self.getTotalCost(x) + self.ra * x.reshape((1,-1))@self.Sigma@x
