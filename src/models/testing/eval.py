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
    le = LabelEncoder()
    X_prep['ticker'] = le.fit_transform(X_prep['ticker'])
    if X_prep['date'].dtype == 'object':
        X_prep['date'] = pd.to_datetime(X_prep['date']).astype(int) // 10**9
    return X_prep

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Preprocess test data
    X_test_prep = preprocess_data(X_test)
    X_train_prep = preprocess_data(X_train)

    # Predict and calculate RMSE
    dtest = xgb.DMatrix(X_test_prep)
    y_pred = model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test['close'], y_pred))
    print(f"Test RMSE: {rmse:.2f}")

    # Grid Search for hyperparameter tuning
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [4, 6, 8],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }

    xgb_model = XGBRegressor(objective="reg:squarederror", seed=42)
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring="neg_mean_squared_error", verbose=1)
    
    # Use preprocessed data for grid search
    grid_search.fit(X_train_prep, y_train['close'])

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best RMSE found: {(-grid_search.best_score_) ** 0.5:.2f}")

    # Plot feature importance
    xgb.plot_importance(model)
    plt.show()

    return grid_search.best_params_