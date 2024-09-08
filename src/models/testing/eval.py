import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, ParameterGrid, KFold, train_test_split
from xgboost import XGBRegressor
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm

def evaluate_model(model, X_train, y_train, X_test, y_test):
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    # Preprocess X_train and X_test
    X_train_prep = preprocess_data(X_train)
    X_test_prep = preprocess_data(X_test)

    # Calculate total number of fits
    n_candidates = np.prod([len(v) for v in param_grid.values()])
    n_splits = 4
    total_fits = n_candidates * n_splits

    print(f"Total number of fits: {total_fits}")

    best_score = float('-inf')
    best_params = None

    # Fit the model with a manual progress bar
    with tqdm(total=total_fits, desc="Grid Search Progress") as pbar:
        for params in ParameterGrid(param_grid):
            cv_scores = []
            for train, val in KFold(n_splits=4, shuffle=False).split(X_train_prep):
                estimator = xgb.XGBRegressor(
                    eval_metric='rmse',
                    early_stopping_rounds=10,
                    verbosity=0,
                    **params
                )
                estimator.fit(
                    X_train_prep.iloc[train], y_train.iloc[train],
                    eval_set=[(X_train_prep.iloc[val], y_train.iloc[val])],
                    verbose=False
                )
                y_pred = estimator.predict(X_train_prep.iloc[val])
                score = -mean_squared_error(y_train.iloc[val], y_pred)
                cv_scores.append(score)
                pbar.update(1)
            
            mean_score = np.mean(cv_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

    print(f"Best parameters: {best_params}")

    # Split the training data to create a validation set
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train_prep, y_train, test_size=0.2, random_state=42
    )

    # Train the best model
    best_model = xgb.XGBRegressor(
        eval_metric='rmse',
        early_stopping_rounds=10,
        verbosity=0,
        **best_params
    )
    best_model.fit(
        X_train_final, y_train_final,
        eval_set=[(X_val_final, y_val_final)],
        verbose=False
    )

    y_pred = best_model.predict(X_test_prep)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"RMSE: {rmse}")

    return best_params