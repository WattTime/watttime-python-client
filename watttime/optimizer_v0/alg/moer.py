# moer.py

import numpy as np


class Moer:
    def __init__(self, mu, isDiagonal=True, Sigma=None, sig2=0.0, ac1=0.0):
        self.__mu = np.array(mu).flatten()
        self.__T = self.__mu.shape[0]
        self.__diagonal = isDiagonal
        if isinstance(sig2, float):
            sig2 = np.array([sig2] * self.__T)
        else:
            sig2 = np.array(sig2).flatten()
        if not self.__diagonal:
            if Sigma is not None:
                assert Sigma.shape == (self.__T, self.__T)
                self.__Sigma = Sigma
            else:
                from scipy.linalg import toeplitz

                x = ac1 ** np.arange(0, self.__T)
                self.__Sigma = toeplitz(x, x) * np.diag(sig2)
        else:
            self.__Sigma = sig2

    def __len__(self):
        return self.__T

    def is_diagonal(self):
        return self.__diagonal

    def get_emission_at(self, i, xi=1):
        return self.__mu[i] * xi

    def get_emission_interval(self, start, end, xi=1):
        return self.__mu[start:end] * xi

    def get_diagonal_util(self, xi, i, ra=0.0):
        return self.get_emission_at(i, xi) + ra * self.__Sigma[i] * xi**2

    def get_marginal_util(self, xi, i, x_old=None, ra=0.0):
        if self.__diagonal:
            return self.get_diagonal_util(xi, i, ra)
        else:
            return self.get_emission_at(xi, i) + ra * (
                2 * xi * self.__Sigma[i, :i] @ x_old[:i] + self.__Sigma[i, i] * xi**2
            ), np.hstack((x_old, xi))

    def get_total_emission(self, x):
        x = np.array(x).flatten()
        return np.dot(self.__mu[: x.shape[0]], x)

    def get_total_util(self, x, ra=0.0):
        x = np.array(x).flatten()
        if self.__diagonal:
            return (
                self.get_total_emission(x)
                + ra * np.multiply(self.__Sigma[: x.shape[0]], x**2).sum()
            )
        else:
            return (
                self.get_total_emission(x)
                + ra * x.reshape((1, -1)) @ self.__Sigma[: x.shape[0], : x.shape[0]] @ x
            )
