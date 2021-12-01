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


def min_max_scale(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def OLS_linearReg(x_train, y_train):
  # Determine optimal parameters
  beta = np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ y_train
  return beta


def generate_batches(X, Y, batch_size, equal_size=False):
    """Generates batches.
    """
    # Create all batch indices
    batch_numbers = [i for i in range(X.shape[0])]
    np.random.shuffle(batch_numbers)
    # Determine number of batches
    if equal_size:
        n = np.floor(X.shape[0] / batch_size).astype(int)
    else:
        n = np.ceil(X.shape[0] / batch_size).astype(int)
    # Create batches
    x_batches = []
    y_batches = []
    for i in range(n):
        idx = batch_numbers[i*batch_size:max((i+1)*batch_size, X.shape[0])]
        x_batches.append(X[idx, :])
        y_batches.append(Y[idx])
    return x_batches, y_batches