# Readymade dataset for training a model, contains data about California housing prices and features.
from sklearn.datasets import fetch_california_housing

# Provides different types of high performance models for regression and classification tasks.
from sklearn.ensemble import RandomForestRegressor

# 
# Split arrays or matrices into random train and test subsets.
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pandas as pd

import joblib