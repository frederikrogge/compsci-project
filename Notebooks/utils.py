"""
Let's put all the methods and functionality that might be useful across tasks here.
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os

from definitions import SAVE_PATH


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
        idx = batch_numbers[i*batch_size:min((i+1)*batch_size, X.shape[0])]
        x_batches.append(X[idx, :])
        y_batches.append(Y[idx])
    return x_batches, y_batches


def logistic_func(x, epsilon=1e-15):
    p = np.clip(1 / (1 + np.exp(-x)), epsilon, 1.-epsilon)
    return p


def plot_metric(y, x=None, show=False, title='', name=None, x_label='x', y_label='y', fig=None, save=False, ax=None,
                x_limit=None, y_limit=None, nx_ticks=None, legend=None, alpha=1.):
    """Plots a metric and/or adds it to an axis object
    """
    if ax is None:
        fig, ax = plt.subplots()

    y = np.array(y)

    if x is None:
        x = np.arange(y.shape[-1])

    # When we have multiple lines
    if len(y.shape) == 2:
        for row in y:
            ax.plot(x, row, alpha=alpha)
    else:
        ax.plot(x, y, alpha=alpha)

    if nx_ticks is not None:
        ax.get_xaxis().set_ticks(x, nx_ticks)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if legend is not None:
        ax.legend(legend, frameon=False)

    if x_limit is not None:
        ax.set_xlim(x_limit)
    if y_limit is not None:
        ax.set_ylim(y_limit)

    plt.tight_layout()

    if save and fig is not None:
        if name is None:
            name = title
        fig.savefig(os.path.join(SAVE_PATH, name), dpi=300, transparent=True, bbox_inches='tight')

    if show:
        plt.show()

    return ax


def save_fig(fig, name):
    fig.savefig(os.path.join(SAVE_PATH, name), dpi=300, transparent=True, bbox_inches='tight')


def plot_3d(x, y, z, show=False, title='', name=None, x_label='x', y_label='y', zlabel='z', fig=None, ax=None,
            save=False, figsize=(10, 10), x_limit=None, y_limit=None, z_limit=None):

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors
    if fig is not None:
        fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    if x_limit is not None:
        ax.set_xlim(x_limit)
    if y_limit is not None:
        ax.set_ylim(y_limit)
    if z_limit is not None:
        ax.set_zlim(z_limit)

    if save and fig is not None:
        if name is None:
            name = title
        fig.savefig(os.path.join(SAVE_PATH, name), dpi=300, transparent=True, bbox_inches='tight')

    if show:
        plt.show()


class ExpLearningScheduler(object):
    """
    Simple learning rate scheduler.
    """

    def __init__(self, lr0=0.01, k=1):
        self.lr0 = lr0
        self.k = k
        self.t = 0

    def update(self, *args):
        lr = self.lr0 * np.exp(-self.k * self.t)
        self.t += 1
        return lr

