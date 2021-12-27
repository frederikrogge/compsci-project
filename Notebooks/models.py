import numpy as np
from sklearn.model_selection import KFold
from utils import *


class BaseModel:

    def __init__(self,
                 fit_intercept=False,
                 dimension=None,
                 random_init=False,
                 reduction='mean'):
        """Initializes basic model that can either be trained using a gradient-based method or by
        using matrix inversion.

        Parameters
        ----------
        fit_intercept: bool
            Whether to explicitly fit an intercept. False if the intercept is included in the design matrix.
        dimension: int
            Dimension of the parameter vector. Should match dimension of feature vector.
        random_init: bool
            Whether to randomly initialize parameters, otherwise will be set to zero.
        reduction: str
            Which reduction method to use (either sum or mean).
        
        """
        if random_init:
            np.random.seed(10)
            self.beta = np.random.randn(dimension)
        else:
            self.beta = np.zeros(dimension)
        self.intercept = 0
        self.fit_intercept = fit_intercept
        self.reduction = reduction
    
    def predict(self, X):
        """Returns the prediction for a given feature vector X.
        """
        return X @ self.beta + self.intercept

    def reduce(self, value, n):
        """Reduces a loss or gradient term.
        """
        if self.reduction == 'mean':
            return value / n
        elif self.reduction == 'sum':
            return value
        else:
            raise ValueError('Unknown reduction method {}'.format(self.reduction))

    def gradient(self, X, y):
        raise NotImplementedError

    def set_intercept(self, X, Y):
        """Calculates the intercept.
        """
        X_off, Y_off = np.mean(X, axis=0), np.mean(Y, axis=0)
        self.intercept = Y_off - X_off @ self.beta

    def MSE_loss(self, X, Y):
        """Computes MSE w.r.t. the parameters beta.

        Parameters
        ----------
        X: ndarray
            (f, n) where f is number of features and n is number of samples
        Y: ndarray
            (n, ) where n is number of samples
        
        """
        loss = (X @ self.beta + self.intercept - Y)**2
        return self.reduce(loss, X.shape[0])

    def simple_SGD(self, X, Y, batch_size=5, lr_eta=0.1, n_epochs=1000):
        if X.shape[0] != len(Y):
            raise TypeError("The number of datapoints must match between X and y")
        else:
            n = len(Y)  #number of datapoints
            m = int(n/batch_size) #number of minibatches
            batch_indices= {}

            ## batching without replacement   
            if m == 1:
                batch_indices[0] = np.arange(len(Y))
            else:
                kf= KFold(n_splits=m , shuffle = True)
                for k,(_, index) in enumerate(kf.split(X)):
                    batch_indices[k] = index
                    
            for epoch in range(1,n_epochs+1):    # looping over the epochs
                for i in range(m):               # looping over the bacthes
                    chosen_batch = np.random.randint(m)

                    # Pick minibatch corresponding to 'chosen_batch'
                    X_train_batch = X[batch_indices[chosen_batch]]
                    y_train_batch = Y[batch_indices[chosen_batch]]
                    
                    # Compute the gradient using the data in minibatch Bk
                    gradient_value = self.gradient(X_train_batch, y_train_batch.reshape(-1,))

                    # Compute new suggestion for beta
                    self.beta = self.beta - lr_eta*(gradient_value)

    def train(self, X, Y, optimizer, batch_size=5, epochs=1000, lr_scheduler=None):
        """Simple gradient-descent-based train method that uses a given optimizer to update
        the parameters of the models.

        Parameters
        ----------
        X: ndarray
            The training data.
        Y: ndarray
            The labels.
        optimizer:
            An optimizer instance than implements the optimizer base class.
        batch_size: int
            Batch size to be used for training.
        epochs: int
            Number of epochs you want the model to be trained.
        lr_scheduler:
            An optional learning rate scheduler that updates the learning rate of the optimizer.

        """
        for _ in range(epochs):

            # Generate batches
            x_batches, y_batches = generate_batches(X, Y, batch_size)

            # Iterate through batches
            for x, y in zip(x_batches, y_batches):

                # Determine the gradient for this batch
                gradient = self.gradient(x, y)

                # Update parameters using optimizer
                self.beta = optimizer.update(self.beta, gradient)

            # Update learning rate if scheduler is provided
            if lr_scheduler is not None:
                optimizer.lr = lr_scheduler.update(optimizer.lr)


class LinearRegression(BaseModel):

    def __init__(self,
                 fit_intercept=False,
                 dimension=None,
                 random_init=False,
                 reduction='mean',
                 **kwargs):
        super().__init__(fit_intercept, dimension, random_init, reduction)

    def fit(self, X, Y):
        """Fits parameters beta using matrix inversion. If 'fit_intercept' is True, the design matrix X is to be
        expected to not include an intercept column. Instead, the intercept will be determined manually by centering
        the data first.

        Parameters
        ----------
        X: ndarray
            (f, n) where f is number of features and n is number of samples. Can be either centered or uncentered.
            Note that if data is centered and 'fit_intercept' is True this class expects inference to be centered as
            well.
        Y: ndarray
            (n, ) where n is number of samples
        """
        # Center data
        Y_unc = Y.copy()
        X_unc = X.copy()
        if self.fit_intercept:
            X_off, Y_off = np.mean(X, axis=0), np.mean(Y, axis=0)
            X, Y = X - X_off, Y - Y_off

        # Fit
        self.beta = np.linalg.pinv(X.T @ X) @ X.T @ Y

        # Calculate intercept
        if self.fit_intercept:
            self.set_intercept(X_unc, Y_unc)

    def gradient(self, X, Y):
        """Computes gradient of the MSE w.r.t. the parameters beta.

        Parameters
        ----------
        X: ndarray
            (f, n) where f is number of features and n is number of samples
        Y: ndarray
            (n, ) where n is number of samples
        """
        grad = X.T @ (X @ self.beta + self.intercept - Y)
        return self.reduce(grad, X.shape[0])


class RidgeRegression(BaseModel):

    def __init__(self,
                 fit_intercept=True,
                 dimension=None,
                 random_init=False,
                 reduction='sum',
                 lbd=1,
                 **kwargs):
        super().__init__(fit_intercept, dimension, random_init, reduction)
        self.lbd = lbd

    def fit(self, X, Y):
        """Fits parameters beta using matrix inversion.

        Parameters
        ----------
        X: ndarray
            (f, n) where f is number of features and n is number of samples
        Y: ndarray
            (n, ) where n is number of samples

        """
        # Center data
        Y_unc = Y.copy()
        X_unc = X.copy()
        if self.fit_intercept:
            X_off, Y_off = np.mean(X, axis=0), np.mean(Y, axis=0)
            X, Y = X - X_off, Y - Y_off

        # Fit
        self.beta = np.linalg.pinv(X.T @ X + self.lbd*np.identity(X.shape[1])) @ X.T @ Y
        
        # Calculate intercept
        if self.fit_intercept:
            self.set_intercept(X_unc, Y_unc)

    def gradient(self, X, Y):
        """Computes gradient of the MSE w.r.t. the parameters beta.

        Parameters
        ----------
        X: ndarray
            (f, n) where f is number of features and n is number of samples
        Y: ndarray
            (n, ) where n is number of samples
        """
        gradient = X.T @ (X @ self.beta+ self.intercept - Y) 
        return self.reduce(gradient, X.shape[0]) + 2*self.lbd * self.beta


class LogisticRegression(BaseModel):

    def __init__(self,
                 fit_intercept=True,
                 dimension=None,
                 random_init=False,
                 reduction='sum',
                 epsilon=1e-15,
                 l2_reg=True,
                 lbd=1,
                 **kwargs):
        super().__init__(fit_intercept, dimension, random_init, reduction)
        self.epsilon = epsilon
        self.l2_reg = l2_reg
        self.lbd = lbd

    def predict(self, X):
        """Returns the prediction for a given feature vector X.
        """
        return logistic_func(X @ self.beta + self.intercept)

    def NLL_loss(self, X, Y ):
        """Computes Negative Log Likelihood w.r.t. the parameters beta.

        Parameters
        ----------
        X: ndarray
            (f, n) where f is number of features and n is number of samples
        Y: ndarray
            (n, ) where n is number of samples
        
        """
        p = logistic_func(X @ self.beta + self.intercept, self.epsilon)
        loss = np.log(p) * Y + np.log(1 - p) * (1 - Y)
        return -self.reduce(np.sum(loss), loss.shape[0])

    def gradient(self, X, Y):
        """Computes gradient of the NLL w.r.t. the parameters beta.

        Parameters
        ----------
        X: ndarray
            (f, n) where f is number of features and n is number of samples
        Y: ndarray
            (n, ) where n is number of samples
        """
        grad = X.T @ (logistic_func(X @ self.beta + self.intercept, self.epsilon) - Y)
        if self.l2_reg == False:
            return self.reduce(grad, X.shape[0])
        else:
            return self.reduce(grad, X.shape[0]) + 2*self.lbd * self.beta

    




