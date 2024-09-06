import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def preprocess_data(X):
    X_prep = X.copy()
    
    # Handle 'ticker' column with one-hot encoding
    if 'ticker' in X_prep.columns:
        # Create a one-hot encoder
        onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Fit and transform the 'ticker' column
        ticker_encoded = onehot.fit_transform(X_prep[['ticker']])
        
        # Create DataFrame with encoded ticker columns
        ticker_columns = pd.DataFrame(
            ticker_encoded, 
            columns=[f'ticker_{ticker}' for ticker in onehot.categories_[0]],
            index=X_prep.index
        )
        
        # Drop original 'ticker' column and join encoded columns
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
    
    return X_prep

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

    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(
            eval_metric='rmse',
            early_stopping_rounds=10
        ),
        param_grid=param_grid,
        cv=4,  # Increased number of cross-validation folds
        n_jobs=-1,
        verbose=2,
        scoring='neg_mean_squared_error'
    )

    grid_search.fit(
        X_train_prep, 
        y_train, 
        eval_set=[(X_test_prep, y_test)],
        verbose=False
    )

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_prep)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Best parameters: {best_params}")
    print(f"RMSE: {rmse}")

    return best_params