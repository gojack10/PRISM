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

    print("\nDataFrame structure after loading:")
    print(data.info())
    print("\nIndex names:", data.index.names)
    print("\nFirst few rows of data:")
    print(data.head())

    # Prepare features and target
    y = data['close']
    X = data.drop(columns=['close'])

    print("\nX DataFrame structure:")
    print(X.info())
    print("\nX index names:", X.index.names)
    print("\nFirst few rows of X:")
    print(X.head())

    # ensure index names are set correctly
    if X.index.names == [None, None]:
        X.index.names = ['ticker', 'date']
        y.index.names = ['ticker', 'date']

    print("\nAfter setting index names:")
    print("X index names:", X.index.names)
    print("y index names:", y.index.names)

    # Reset index to use ticker as a feature
    X_reset = X.reset_index()
    y_reset = y.reset_index()

    print("\nAfter resetting index:")
    print("X_reset columns:", X_reset.columns)
    print("y_reset columns:", y_reset.columns)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_reset, y_reset, test_size=0.2, random_state=42)

    print("\nTrain-test split shapes:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

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
