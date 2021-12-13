import numpy as np
from models import *
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class Resampling:
    """Base Resampling class.
    """

    def __init__(self):
        pass


class Bootstrap(Resampling):

    def __init__(self):
        super(Bootstrap, self).__init__()
        

    def resampling(self, x_train, x_test, y_train, y_test,solver = LinearRegression, include_intercept = False, n_bootstraps= 1000 ):
        # The following (m x n_bootstraps) matrix holds the column vectors y_pred
        # for each bootstrap iteration.
        y_hat_test = np.empty((y_test.shape[0], n_bootstraps))   ## test set
        y_hat_train = np.empty((y_train.shape[0], n_bootstraps))   ## train set

        for i in range(n_bootstraps):
            x_, y_ = resample(x_train , y_train)
            
            # Evaluate the new model on the same test data each time.
            # create the model 
            model = LinearRegression(fit_intercept= include_intercept , dimension=x_train.shape[1])
            
            #fit the model on the current sample
            model.fit(x_, y_)
            
            # predict with the fit model
            y_hat_train[:, i] = model.predict(x_train).ravel()
            y_hat_test[:, i] = model.predict(x_test).ravel()
            
            error_train, bias_train ,variance_train = self.error_bias_variance(y_train, y_hat_train)
            error_test, bias_test ,variance_test = self.error_bias_variance(y_test, y_hat_test)
        
        return error_test, bias_test ,variance_test, error_train, bias_train, variance_train
    
   
    def error_bias_variance(self, y, y_hat):
        """
        Returns error, bias^2 and variance 
        
        """
        error =  np.mean( np.mean((y - y_hat)**2, axis=1, keepdims=True) )
        bias = np.mean( (y - np.mean(y_hat, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(y_hat, axis=1, keepdims=True) )
        
        return error, bias, variance
    
class KFoldCV(Resampling):

    def __init__(self):
        super(KFoldCV, self).__init__()
        
        
    def kfold_resample(self, features, labels, degree, k = 5,  include_intercept = False , scaling = True,  solver =LinearRegression):
        
        kfold = KFold(n_splits = k)  # Splitting into folds
        train_MSE = np.zeros(k)
        test_MSE = np.zeros(k)
        
        # iterate over the folds, leaving one fold out for testing each time
        for nk,(train_index, test_index) in enumerate(kfold.split(features)):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # fit polynomials on test and train
            poly = PolynomialFeatures(degree, include_bias=not include_intercept)
            x_train = poly.fit_transform(X_train)
            x_test = poly.fit_transform(X_test)   

            if scaling:
                # Scale data
                scaler = StandardScaler(with_std=True)
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)

            # create the model 
            model = solver(fit_intercept= include_intercept, dimension=x_train.shape[1])

            # fit model
            model.fit(x_train, y_train)

            # predict model
            y_hat_train = model.predict(x_train)
            y_hat_test = model.predict(x_test)

            train_MSE[nk] = MSE(y_hat_train  , y_train)
            test_MSE[nk] = MSE(y_hat_test  , y_test)
           
        return test_MSE, train_MSE
 


