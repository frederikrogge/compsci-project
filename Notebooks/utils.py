"""
Let's put all the methods and functionality that might be useful across tasks here.
"""
import numpy as np


def FrankeFunction(x, y):
    """
    Returns value of Franke function at (x, y).
    """    
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def R2(y, y_hat):
    return 1 - np.sum((y - y_hat)**2) / np.sum((y - np.mean(y))**2)


def MSE(y, y_hat):
    return np.mean(np.square(y - y_hat))


def z_score(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))