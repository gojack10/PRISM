import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

def init_model(params=None):
    default_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.01,
        "max_depth": 6,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    if params:
        default_params.update(params)
    return default_params

def train_model(X, y, params=None):
    if params is None:
        params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
        }
    
    X_prep = X.copy()
    
    # Handle 'ticker' column if present
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

    print("Data types after preprocessing:")
    print(X_prep.dtypes)

    model = xgb.XGBRegressor(**params, early_stopping_rounds=15, eval_metric='rmse')
    
    # Split data for early stopping
    split_point = int(0.8 * len(X_prep))
    X_train, X_val = X_prep[:split_point], X_prep[split_point:]
    y_train, y_val = y[:split_point], y[split_point:]
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Get feature importances
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': X_prep.columns, 'importance': importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)

    return model, feature_importance, params

def plot_feature_importance(model, X):
    # Get feature importances
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    return feature_importance

def evaluate_model_performance(model, X_test, y_test):
    # prepare data for xgboost
    dtest = xgb.DMatrix(X_test.drop(['ticker', 'date'], axis=1))
    y_pred = model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test['close'], y_pred))
    print(f"\nTest RMSE: {rmse}")

    # calculate rmse for each ticker
    ticker_rmse = {}
    for ticker in X_test['ticker'].unique():
        ticker_mask = X_test['ticker'] == ticker
        ticker_y_test = y_test[ticker_mask]['close']
        ticker_y_pred = y_pred[ticker_mask]
        ticker_rmse[ticker] = np.sqrt(mean_squared_error(ticker_y_test, ticker_y_pred))
    
    print("\nRMSE by ticker:")
    for ticker, rmse in ticker_rmse.items():
        print(f"{ticker}: {rmse}")

    return ticker_rmse
