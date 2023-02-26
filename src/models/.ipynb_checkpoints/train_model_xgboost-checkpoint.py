import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb 
import lightgbm as lgbm
from HROCH import PHCRegressor
import optuna
from optuna.samplers import TPESampler
import datetime
import pickle
import json

# Suppressing warnings
import warnings
warnings.simplefilter("ignore")

# Setting the working directory
base_path = "/home/sagemaker-user/"
import os
os.chdir(base_path)

# Loading the data
train = pd.read_csv("data/processed/train/train.csv")

# Setting the target and feature variables
target = "quality"
features = [col for col in train.columns if col != target]
n_classes = len(train[target].unique())

def objective_xgb(trial):    
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.01, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0001, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0001, 1.0),
        'objective' : "multi:softmax",
        'num_class': n_classes
    }

        
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    fold_scores = []
    for i, (train_idx, val_idx) in enumerate(cv.split(train[features], train[target])):
        X_train, y_train = train.loc[train_idx, features],train.loc[train_idx, target]
        X_val, y_val = train.loc[val_idx, features],train.loc[val_idx, target]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train,
                  y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=50,
                  verbose=500)

        pred_val = model.predict(X_val)

        score = cohen_kappa_score(y_val, pred_val)
        fold_scores.append(score)
    return np.mean(fold_scores)

study = optuna.create_study(direction='maximize', sampler = TPESampler())
study.optimize(func=objective_xgb, n_trials=1)

model = xgb.XGBClassifier(**study.best_params)
model.fit(train.loc[:, features],
                 train.loc[:, target])

date_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_path = base_path + "models/xgboost/" + date_time_str
os.mkdir(file_path)

with open(file_path + '/model.bin', 'wb') as f:
    pickle.dump(model, f)
    
with open(file_path + "/params.json", 'w') as f:
    json.dump(study.best_params, f)