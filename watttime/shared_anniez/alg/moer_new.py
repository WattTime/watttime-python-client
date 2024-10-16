# moer.py

import numpy as np 
class Moer: 
    def __init__(self, mu): 
        self.__mu = np.array(mu).flatten()
        self.__T = self.__mu.shape[0]

    def __len__(self):
        return self.__T
    
    def is_diagonal(self):
        return True
    
    def get_emission_at(self, i, usage): 
        return self.__mu[i] * usage
    
    def get_emission_interval(self, start, end, usage): 
        return np.dot(self.__mu[start:end], usage)
        
    def get_emissions(self, usage):
        usage = np.array(usage).flatten()
        return self.__mu[:usage.shape[0]] * usage

    def get_total_emission(self, usage): 
        return self.get_emissions(usage).sum()
