import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

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
        onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ticker_encoded = onehot.fit_transform(X_prep[['ticker']])
        ticker_columns = pd.DataFrame(
            ticker_encoded, 
            columns=[f'ticker_{ticker}' for ticker in onehot.categories_[0]],
            index=X_prep.index
        )
        X_prep = X_prep.drop('ticker', axis=1).join(ticker_columns)
    
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
    print(f"Number of features after preprocessing: {X_prep.shape[1]}")

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

def plot_feature_importance(importances, feature_names):
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [feature_names[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,10))

    # Create plot title
    plt.title("Feature Importance")

    # Add bars
    plt.bar(range(len(importances)), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(len(importances)), names, rotation=90)

    # Show plot
    plt.tight_layout()
    plt.show()

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

# Add these new functions at the end of the file

def train_ticker_model(X, y, params=None):
    if params is None:
        params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
        }
    
    model = xgb.XGBRegressor(**params, early_stopping_rounds=15, eval_metric='rmse')
    
    # Split data for early stopping
    split_point = int(0.8 * len(X))
    X_train, X_val = X[:split_point], X[split_point:]
    y_train, y_val = y[:split_point], y[split_point:]
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return model

class ConsolidatedModel:
    def __init__(self, ticker_models):
        self.ticker_models = ticker_models
    
    def predict(self, X):
        predictions = []
        for idx, row in X.iterrows():
            ticker = self._get_ticker(row)
            if ticker in self.ticker_models:
                model = self.ticker_models[ticker]
                # Remove ticker columns before prediction
                features = row.drop([col for col in row.index if col.startswith('ticker_')])
                pred = model.predict(features.to_frame().T)[0]
            else:
                # Handle unseen tickers (e.g., use an average prediction)
                pred = np.mean([model.predict(row.drop([col for col in row.index if col.startswith('ticker_')]).to_frame().T)[0] for model in self.ticker_models.values()])
            predictions.append(pred)
        return np.array(predictions)

    def _get_ticker(self, row):
        ticker_columns = [col for col in row.index if col.startswith('ticker_')]
        if ticker_columns:
            return ticker_columns[row[ticker_columns].argmax()].split('_')[1]
        raise ValueError("Unable to determine ticker from input data")

def train_consolidated_model(X, y, ticker_models):
    return ConsolidatedModel(ticker_models)
