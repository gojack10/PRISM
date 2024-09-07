import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, ParameterGrid, KFold, train_test_split
from xgboost import XGBRegressor
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm  # Import tqdm for progress bar

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