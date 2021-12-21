import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd

from models import *
from utils import *
from optimizers import *

# Create data
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x, y)

# Compute z
z = FrankeFunction(x, y)

# Create features as pairs of (x, y)
features = np.stack([x.ravel(), y.ravel()], axis=1)
outputs = z.ravel()

# Split dataset into train and test set
x_train, x_test, y_train, y_test = train_test_split(features, outputs, test_size=0.2, random_state=20)

# Get polynomial features
degree = 5
poly = PolynomialFeatures(degree, include_bias=True)
x_train = poly.fit_transform(x_train)
x_test = poly.transform(x_test)

# Instantiate model and optimizer
model = LinearRegression(dimension=x_train.shape[1], random_init=True, reduction='sum')
optimizer = SGD(lr=0.01)

# Train model
model.train(x_train, y_train, optimizer, 100, 2000)