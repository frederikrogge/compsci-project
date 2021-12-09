import numpy as np
from sklearn.model_selection import KFold
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
        

    def MSE_loss(self, X, Y ):
        """Computes MSE w.r.t. the parameters beta.

        Parameters
        ----------
        X: ndarray
            (f, n) where f is number of features and n is number of samples
        Y: ndarray
            (n, ) where n is number of samples
        
        """
        loss = (X @ self.beta+ self.intercept - Y)**2
        if self.reduction == 'mean':
            return np.mean(loss)
        elif self.reduction == 'sum':
            return np.sum(loss)
        raise ValueError('Unknown reduction method {}'.format(self.reduction))


    def simple_SGD(self, X, Y, batch_size = 5 ,lr_eta = 0.1, n_epochs = 1000):
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
                kf= KFold(n_splits = m)
                for k,(_, index) in enumerate(kf.split(X)):
                    batch_indices[k] = index
                    
            for epoch in range(1,n_epochs+1):    # looping over the epochs
                for i in range(m):               # looping over the bacthes
                    chosen_batch = np.random.randint(m)

                    #Pick minibatch corresponding to 'chosen_batch'
                    X_train_batch = X[batch_indices[chosen_batch ]]
                    y_train_batch = Y[batch_indices[chosen_batch ]]  
                    
                    #Compute the gradient using the data in minibatch Bk
                    gradient_value = self.gradient(X_train_batch, y_train_batch.reshape(-1,) )

                    #print(gradient_value)

                    #Compute new suggestion for beta
                    self.beta = self.beta - lr_eta*(gradient_value)

    

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
        grad = X.T @ (X @ self.beta+ self.intercept - Y)
        if self.reduction == 'mean':
            return grad / X.shape[0]
        elif self.reduction == 'sum':
            return grad
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
        gradient = X.T @ (X @ self.beta - Y) 
        if self.reduction == 'mean':
            return (gradient / X.shape[0]) + 2*self.lbd * self.beta
        elif self.reduction == 'sum':
            return gradient + 2*self.lbd *self.beta
        raise ValueError('Unknown reduction method {}'.format(self.reduction))


class LogisticRegression(BaseModel):

    def __init__(self,
                 fit_intercept=True,
                 dimension=None,
                 random_init=False,
                 reduction='sum',
                 epsilon = 1e-15,
                 l2_reg = True,
                 lbd=1):
        super().__init__(fit_intercept, dimension, random_init)
        self.reduction = reduction
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
        p = logistic_func(X @ self.beta+ self.intercept , self.epsilon) 
        loss = np.log(p)* Y + np.log(1 - p) * (1 - Y)
        if self.reduction == 'mean':
            return -np.mean(loss)
        elif self.reduction == 'sum':
            return -np.sum(loss)
        raise ValueError('Unknown reduction method {}'.format(self.reduction))
        

    def gradient(self, X, Y):
        """Computes gradient of the NLL w.r.t. the parameters beta.

        Parameters
        ----------
        X: ndarray
            (f, n) where f is number of features and n is number of samples
        Y: ndarray
            (n, ) where n is number of samples
        """
        grad = X.T @ (logistic_func(X @ self.beta+ self.intercept, self.epsilon) - Y)
        if self.l2_reg == False:
            if self.reduction == 'mean':
                return grad / X.shape[0]
            elif self.reduction == 'sum':
                return grad
            raise ValueError('Unknown reduction method {}'.format(self.reduction))
        else :
            if self.reduction == 'mean':
                return (grad / X.shape[0]) + 2*self.lbd * self.beta
            elif self.reduction == 'sum':
                return grad + 2*self.lbd * self.beta
            raise ValueError('Unknown reduction method {}'.format(self.reduction))

    




