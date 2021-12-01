import numpy as np


from utils import *


class LinearRegression:

    def __init__(self,
                 fit_intercept=True,
                 dimension=None,
                 random_init=False):
        if random_init:
            self.beta = np.random.randn(dimension)
        else:
            self.beta = np.zeros(dimension)
        self.intercept = 0
        self.fit_intercept = fit_intercept

    def fit_matrix_inversion(self, X, Y):
        """Fits parameters beta using matrix inversion.

        Parameters
        ----------
        X: ndarray
            (f, n) where f is number of features and n is number of samples
        Y: ndarray
            (n, ) where n is number of samples
        """

        self.beta = np.linalg.pinv(X.T @ X) @ X.T @ Y
        if self.fit_intercept:
            self.intercept = np.mean(Y, axis=0) - np.mean(X, axis=0) @ self.beta

    def predict(self, X):
        """Returns the prediction for a given feature vector X.
        """
        return X @ self.beta + self.intercept

    def gradient(self, X, Y):
        """Computes gradient of the MSE w.r.t. the parameters beta.

        Parameters
        ----------
        X: ndarray
            (f, n) where f is number of features and n is number of samples
        Y: ndarray
            (n, ) where n is number of samples
        """
        return X.T @ (X @ self.beta - Y)



