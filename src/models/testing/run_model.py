from data_loader import load_data
from model import train_model, plot_feature_importance
from eval import evaluate_model, preprocess_data
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

def run_model():
    # Load data
    data, _ = load_data()
    
    if data is None:
        print("Failed to load data. Exiting.")
        return None

    print("\nAvailable features:")
    print(data.columns.tolist())

    # Prepare features and target
    y = data['close']
    X = data.drop(columns=['close'])

    # Sort data by date
    X = X.sort_values('date')
    y = y.loc[X.index]

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # Initialize lists to store performance metrics
    rmse_scores = []
    mae_scores = []
    r2_scores = []

    # Perform time series cross-validation
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Preprocess data
        X_train_prep = preprocess_data(X_train)
        X_test_prep = preprocess_data(X_test)

        # Train initial model
        initial_model, _, _ = train_model(X_train_prep, y_train)

        # Evaluate and get best parameters
        best_params = evaluate_model(initial_model, X_train_prep, y_train, X_test_prep, y_test)

        # Train model with best parameters
        model, _, _ = train_model(X_train_prep, y_train, params=best_params)

        # Make predictions
        y_pred = model.predict(X_test_prep)

        # Calculate performance metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)

    # Print average performance metrics
    print("\nAverage Performance Metrics:")
    print(f"RMSE: {np.mean(rmse_scores):.4f} (+/- {np.std(rmse_scores):.4f})")
    print(f"MAE: {np.mean(mae_scores):.4f} (+/- {np.std(mae_scores):.4f})")
    print(f"R2: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")

    # Train final model on all data
    X_prep = preprocess_data(X)
    final_model, final_feature_importance, _ = train_model(X_prep, y)

    # Plot feature importance
    plot_feature_importance(final_model, X_prep)

    # Print feature importance
    print("\nFeature Importance:")
    print(final_feature_importance)

    return final_model

if __name__ == "__main__":
    best_model = run_model()
    if best_model is not None:
        print("Model training completed successfully.")
    else:
        print("Model training failed.")
