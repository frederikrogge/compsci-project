import numpy as np


class Optimizer:
    """Base optimizer class.
    """

    def __init__(self):
        pass

    def reset(self):
        pass

    def update(self, parameters, gradient):
        raise NotImplementedError


class Adam(Optimizer):

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999):
        super(Adam, self).__init__()
        self.m_dw, self.v_dw = 0, 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0

    def reset(self):
        self.t = 0
        self.m_dw, self.v_dw = 0, 0

    def update(self, parameters, gradient):
        """Updates the parameters of a model given the gradient at one time step.

        Parameters
        ----------
        parameters: ndarray
            Model parameters to be updated.
        gradient: ndarray
            The gradient of the parameters.

        """

        # Determine moving average of first-order and second-order momentum
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * gradient
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (gradient ** 2)

        # Perform bias correction
        m_dw_corr = self.m_dw / (1 - self.beta1 ** self.t)
        v_dw_corr = self.v_dw / (1 - self.beta2 ** self.t)

        # Update parameters
        parameters = parameters - self.lr * (m_dw_corr / (np.sqrt(v_dw_corr) + 1e-8))

        # Increment iteration
        self.t += 1
        return parameters


class SGD(Optimizer):

    def __init__(self, lr=1e-3):
        super(SGD, self).__init__()
        self.lr = lr

    def update(self, parameters, gradient):
        parameters = parameters - self.lr * gradient
        return parameters
