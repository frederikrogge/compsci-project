import numpy as np


from utils import *


class BaseModel:

    def __init__(self,
                 fit_intercept=True,
                 dimension=None,
                 random_init=False):
        """Initializes basic model that can either be trained using a gradient-based method or by
        using matrix inversion.

        Parameters
        ----------
        fit_intercept: bool
            Whether to explicitly fit an intercept.
        dimension: int
            Dimension of the parameter vector. Should match dimension of feature vector.
        random_init: bool
            Whether to randomly initialize parameters, otherwise will be set to zero.
        """
        if random_init:
            self.beta = np.random.randn(dimension)
        else:
            self.beta = np.zeros(dimension)
        self.intercept = 0
        self.fit_intercept = fit_intercept

    def predict(self, X):
        """Returns the prediction for a given feature vector X.
        """
        return X @ self.beta + self.intercept


class LinearRegression(BaseModel):

    def __init__(self,
                 fit_intercept=True,
                 dimension=None,
                 random_init=False,
                 reduction='sum'):
        super().__init__(fit_intercept, dimension, random_init)
        self.reduction = reduction

    def fit(self, X, Y):
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

    def gradient(self, X, Y):
        """Computes gradient of the MSE w.r.t. the parameters beta.

        Parameters
        ----------
        X: ndarray
            (f, n) where f is number of features and n is number of samples
        Y: ndarray
            (n, ) where n is number of samples
        """
        gradient = X.T @ (X @ self.beta - Y)
        if self.reduction == 'mean':
            return gradient / X.shape[0]
        elif self.reduction == 'sum':
            return gradient
        raise ValueError('Unknown reduction method {}'.format(self.reduction))


class RidgeRegression(BaseModel):

    def __init__(self,
                 fit_intercept=True,
                 dimension=None,
                 random_init=False,
                 reduction='sum',
                 lbd=1):
        super().__init__(fit_intercept, dimension, random_init)
        self.reduction = reduction
        self.lbd = lbd

    def fit(self, X, Y):
        """Fits parameters beta using matrix inversion.

        Parameters
        ----------
        X: ndarray
            (f, n) where f is number of features and n is number of samples
        Y: ndarray
            (n, ) where n is number of samples

        Todo implement!

        """
        raise NotImplemented

    def gradient(self, X, Y):
        """Computes gradient of the MSE w.r.t. the parameters beta.

        Parameters
        ----------
        X: ndarray
            (f, n) where f is number of features and n is number of samples
        Y: ndarray
            (n, ) where n is number of samples
        """
        gradient = X.T @ (X @ self.beta - Y) + self.lbd * np.square(self.beta)
        if self.reduction == 'mean':
            return gradient / X.shape[0]
        elif self.reduction == 'sum':
            return gradient
        raise ValueError('Unknown reduction method {}'.format(self.reduction))


class LogisticRegression(BaseModel):

    def __init__(self,
                 fit_intercept=True,
                 dimension=None,
                 random_init=False,
                 reduction='sum'):
        super().__init__(fit_intercept, dimension, random_init)
        self.reduction = reduction

    def gradient(self, X, Y):
        """Computes gradient of the MSE w.r.t. the parameters beta.

        Parameters
        ----------
        X: ndarray
            (f, n) where f is number of features and n is number of samples
        Y: ndarray
            (n, ) where n is number of samples
        """
        gradient = X.T @ (logistic_func(X @ self.beta) - Y)
        if self.reduction == 'mean':
            return gradient / X.shape[0]
        elif self.reduction == 'sum':
            return gradient
        raise ValueError('Unknown reduction method {}'.format(self.reduction))


