import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

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
    # reset index to use ticker as a feature
    X_reset = X.reset_index(level='ticker')
    y_reset = y.reset_index(level='ticker')

    # split data, ensuring that we don't mix dates between train and test
    X_train, X_test, y_train, y_test = train_test_split(X_reset, y_reset, test_size=0.2, random_state=42)

    # prepare data for xgboost
    dtrain = xgb.DMatrix(X_train.drop(['ticker', 'date'], axis=1), label=y_train['close'])
    dtest = xgb.DMatrix(X_test.drop(['ticker', 'date'], axis=1), label=y_test['close'])

    model_params = init_model(params)
    watchlist = [(dtrain, "train"), (dtest, "test")]

    model = xgb.train(model_params, dtrain, num_boost_round=800, evals=watchlist, early_stopping_rounds=50, verbose_eval=True)

    return model, X_test, y_test

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
