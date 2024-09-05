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
    print("\nInside train_model function:")
    print("X columns:", X.columns)
    print("y columns:", y.columns if isinstance(y, pd.DataFrame) else "y is a Series")
    
    # Create a copy of X to avoid modifying the original dataframe
    X_prep = X.copy()
    
    # Convert 'ticker' to categorical codes
    le = LabelEncoder()
    X_prep['ticker'] = le.fit_transform(X_prep['ticker'])
    
    # Ensure 'date' is in a numeric format (if it's not already)
    if X_prep['date'].dtype == 'object':
        X_prep['date'] = pd.to_datetime(X_prep['date']).astype(int) // 10**9  # convert to Unix timestamp
    
    print("\nPrepared X data types:")
    print(X_prep.dtypes)
    
    # Prepare data for xgboost
    dtrain = xgb.DMatrix(X_prep, label=y['close'])

    model_params = init_model(params)
    
    # Train the model
    model = xgb.train(model_params, dtrain, num_boost_round=800, verbose_eval=True)

    return model, X_prep, y

def plot_feature_importance(model, X):
    importance = model.get_score(importance_type='weight')
    importance_df = pd.DataFrame({'feature': list(importance.keys()), 'importance': list(importance.values())})
    importance_df = importance_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['feature'], importance_df['importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

    print("\nFeature Importance:")
    print(importance_df)

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
