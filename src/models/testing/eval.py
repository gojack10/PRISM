import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(X):
    X_prep = X.copy()
    
    # Handle 'ticker' column
    if 'ticker' in X_prep.columns:
        le = LabelEncoder()
        X_prep['ticker'] = le.fit_transform(X_prep['ticker'])
    
    # Ensure all columns are numeric
    for col in X_prep.columns:
        if X_prep[col].dtype == 'object':
            try:
                X_prep[col] = pd.to_numeric(X_prep[col])
            except ValueError:
                print(f"Warning: Could not convert column '{col}' to numeric. This column will be dropped.")
                X_prep = X_prep.drop(columns=[col])
    
    return X_prep

def evaluate_model(model, X_train, y_train, X_test, y_test):
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
    }

    # Preprocess X_train and X_test
    X_train_prep = preprocess_data(X_train)
    X_test_prep = preprocess_data(X_test)

    grid_search = GridSearchCV(estimator=xgb.XGBRegressor(), 
                               param_grid=param_grid, 
                               cv=5, 
                               n_jobs=-1, 
                               verbose=2)

    grid_search.fit(X_train_prep, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_prep)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Best parameters: {best_params}")
    print(f"RMSE: {rmse}")

    return best_params