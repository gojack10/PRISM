from data_loader import load_data
from model import train_model, plot_feature_importance
from eval import evaluate_model
from sklearn.model_selection import train_test_split
import pandas as pd

def run_model():
    # Load data
    data, _ = load_data()
    
    if data is None:
        print("Failed to load data. Exiting.")
        return None

    # Prepare features and target
    y = data['close']
    X = data.drop(columns=['close'])

    # Reset index to use ticker as a feature
    X_reset = X.reset_index()
    y_reset = y.reset_index()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_reset, y_reset, test_size=0.2, random_state=42)

    # Train initial model
    initial_model, _, _ = train_model(X_train, y_train)

    # Plot feature importance
    plot_feature_importance(initial_model, X_train)

    # Evaluate model and get best parameters
    best_params = evaluate_model(initial_model, X_train, y_train, X_test, y_test)

    # Retrain the model with the best parameters
    best_model, _, _ = train_model(X_reset, y_reset, params=best_params)

    return best_model

if __name__ == "__main__":
    best_model = run_model()
    if best_model is not None:
        print("Model training completed successfully.")
    else:
        print("Model training failed.")
