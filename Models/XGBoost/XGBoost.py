import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import torch


## get data
diamonds = sns.load_dataset('diamonds')
diamonds.head()


## EDA
diamonds.info()
diamonds.describe()

## prepare data
X = diamonds.drop('price', axis = 1)
y = diamonds[['price']]
X.dtypes
X_trian, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

## XGBoost dataset
dtrain = xgb.DMatrix(X_trian, label = y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label = y_test, enable_categorical=True)

## set params and fit model
device = "gpu_hist" if torch.cuda.is_available() else "hist"
params = {"objective": "reg:squarederror", "tree_method": device}
model0 = xgb.train(params, dtrain, 
                   num_boost_round=1000, 
                   evals = [(dtrain, "train"), (dtest, 'validation')], # input both train and test data
                   early_stopping_rounds=20,
                   verbose_eval=10 # Every ten rounds
                   )

## XGBoost with cross validation to find best round
results = xgb.cv(
    params = params,
    dtrain = dtrain,
    num_boost_round = 1000,
    early_stopping_rounds = 20,
    nfold = 5,
    verbose_eval = 10,
)

best_round = len(results)
print(f"Best round: {best_round}")
